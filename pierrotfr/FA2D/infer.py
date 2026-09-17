"""FA2D 추론 진입점 — 체크포인트를 읽어 2D 랜드마크를 낸다.

    load_checkpoint     가중치 -> 추론용 모델 (구조·크롭 규약은 ckpt 에서 복원)
    predict             크롭 배치 -> 좌표 (0~1). flip-TTA 선택
    FA2D (클래스)        임의의 사진 한 장 -> 원본 좌표 랜드마크 (검출 · 2 패스 크롭)

⚠ **설정은 사람이 다시 적지 않는다.** 백본 · 입력 크기 · 디코더 깊이 · 분기(refine /
  hm_offset …) · 크롭 규약(face/head · pad)은 전부 체크포인트의 `config` 에 있다.
  학습 저장소의 1,257줄짜리 args 파일을 옮겨 오면 두 곳이 반드시 어긋난다.

⚠ 가중치는 **EMA 를 우선** 읽는다 — 학습 저장소가 모델 선택과 평가에 쓴 쪽이 EMA 다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import torch

from .data import FLIP_MAPPING, SCHEME_N, crop_box, to_tensor
from .geometric import GeoParams, apply_geometric, to_original
from .models import build_model

# 체크포인트 config 키 -> PointQueryNet 인자. 이 표가 "모델을 재현하는 데 필요한 전부"다.
_ARCH_KEYS = {"backbone": "backbone", "image_size": "image_size", "d_model": "d_model",
              "dec_layers": "dec_layers", "dec_heads": "n_heads", "dec_ffn": "ffn_dim",
              "temporal": "temporal", "use_state": "use_state",
              "refine": "refine", "refine_stem": "refine_stem", "refine_k": "refine_k",
              "heatmap": "heatmap", "hm_sigma": "hm_sigma", "hm_offset": "hm_offset",
              "boundary": "boundary"}


@dataclass
class ModelSpec:
    """체크포인트가 스스로 밝히는 자기 설정."""
    model_name : str = "pointquery"
    arch       : dict = field(default_factory=dict)
    image_size : int = 448
    crop_mode  : str = "face"      # 학습 크롭 규약 — 사람이 고르는 값이 아니다
    crop_pad   : float = 0.05
    name       : str = ""
    preset     : str = ""
    epoch      : int | None = None
    params     : int = 0

    def describe(self) -> str:
        head = self.name or "?"
        if self.preset:
            head += f" (preset={self.preset})"
        branches = [k for k in ("refine", "hm_offset", "heatmap", "boundary", "temporal",
                                "use_state") if self.arch.get(k)]
        return (f"{head}\n  {self.arch.get('backbone')} · {self.params / 1e6:.2f}M · "
                f"입력 {self.image_size} · {self.crop_mode} 크롭 pad {self.crop_pad}"
                + (f" · 분기 {'+'.join(branches)}" if branches else "")
                + (f" @ep{self.epoch}" if self.epoch is not None else ""))


def load_checkpoint(ckpt_fp: str, device="cuda", verbose: bool = True):
    """학습 저장소의 `runs/fa2d/<런>/best.pth` -> (모델, ModelSpec)."""
    if not os.path.isfile(ckpt_fp):
        raise SystemExit(f"[FA2D] 체크포인트가 없습니다: {ckpt_fp}")
    ck = torch.load(ckpt_fp, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    if cfg.get("student_face_crop") and cfg.get("crop_mode", "head") != "face":
        # 머리 크롭 교사 옆에서 얼굴 크롭을 따로 렌더링하던 과거 학생 방식. 크롭 경로가 둘이라
        # 이 저장소의 단일 크롭 평가로는 학습 조건을 재현할 수 없다.
        raise SystemExit(f"[FA2D] student_face_crop 방식의 과거 체크포인트는 지원하지 않습니다: "
                         f"{ckpt_fp}\n  crop_mode='face' 로 학습한 체크포인트를 쓰세요.")
    arch = {dst: cfg[src] for src, dst in _ARCH_KEYS.items() if src in cfg}
    model = build_model(cfg.get("model_name", "pointquery"), **arch)

    state = ck.get("ema") if ck.get("ema") is not None else ck.get("model")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        # 조용히 넘어가면 초기값 그대로인 층으로 추론한다 — 구조가 어긋났다는 뜻이다.
        raise SystemExit(f"[FA2D] 체크포인트와 구조가 맞지 않습니다: missing {sorted(missing)[:6]} · "
                         f"unexpected {sorted(unexpected)[:6]}\n  {ckpt_fp}")
    spec = ModelSpec(
        model_name=cfg.get("model_name", "pointquery"), arch=arch,
        image_size=int(cfg.get("image_size", 448)),
        crop_mode=cfg.get("crop_mode", "head"), crop_pad=float(cfg.get("crop_pad", 0.15)),
        name=os.path.basename(os.path.dirname(os.path.abspath(ckpt_fp))),
        preset=cfg.get("preset", ""), epoch=ck.get("epoch"),
        params=sum(p.numel() for p in model.parameters()))
    if verbose:
        print(f"체크포인트 {ckpt_fp}  [{'ema' if ck.get('ema') is not None else 'model'}]\n"
              f"  {spec.describe()}")
    return model.to(device).eval(), spec


# ------------------------------------------------------------------ #
_FLIP_PERM: dict = {}


def flip_perm(scheme: str, device) -> torch.Tensor:
    key = (scheme, str(device))
    if key not in _FLIP_PERM:
        idx = torch.arange(SCHEME_N[scheme])
        for a, b in FLIP_MAPPING[scheme]:
            idx[a], idx[b] = b, a
        _FLIP_PERM[key] = idx.to(device)
    return _FLIP_PERM[key]


@torch.no_grad()
def predict(model, images: torch.Tensor, scheme: str = "wflw98",
            tta: bool = False) -> torch.Tensor:
    """(B,3,S,S) 정규화 크롭 -> (B,N,2) 크롭 정규화 좌표 (0~1).

    `tta=True` — 좌우반전 예측을 **랜드마크 공간에서** 평균한다(추론 2배).
    ⚠ 좌표는 OpenCV 픽셀 중심 x / W 이므로 x' = (W−1)/W − x 로 복원한다. 1−x 를 쓰면
      반전 예측이 +1px, 평균이 +0.5px 밀린다. 복원 후 좌우 번호를 교환한다.
    """
    pts = model(images, scheme)["points"].float()
    if not tta:
        return pts
    q = model(torch.flip(images, dims=[3]), scheme)["points"].float().clone()
    q[..., 0] = (images.shape[-1] - 1.0) / images.shape[-1] - q[..., 0]
    q = q[:, flip_perm(scheme, q.device)]
    return (pts + q) / 2


# ------------------------------------------------------------------ #
class FA2D:
    """임의의 사진 한 장 -> 원본 좌표 랜드마크.

        fa = FA2D(ckpt="runs/fa2d/<런>/best.pth")
        for lmk in fa(img_bgr):          # [(N,2) 원본 픽셀 좌표, …]  큰 얼굴 순
            ...

    크롭을 어떻게 잡나 — 모델은 **GT 랜드마크로 자른 크롭**으로 학습했다. 사진에는 GT 가
    없으므로 두 번 돈다:

        ① 검출 박스를 정사각으로 넓혀 1차 크롭 → 예측
        ② **예측 랜드마크로** 학습과 같은 박스(face/head + pad)를 다시 잡아 → 재예측

    ②부터는 학습 크롭 규약과 같다. 영상에서는 이전 프레임 랜드마크로 ②만 돌면 된다.
    ⚠ 이 경로의 오차에는 검출기 품질이 섞인다 — 벤치마크 수치(GT 크롭)와 나란히 놓지 말 것.
    """

    def __init__(self, ckpt: str = "", model=None, spec: ModelSpec | None = None,
                 device: str = "cuda", scheme: str = "wflw98", tta: bool = False,
                 passes: int = 2, verbose: bool = True):
        if model is None:
            model, spec = load_checkpoint(ckpt, device, verbose)
        self.model, self.spec = model, spec or ModelSpec()
        self.device, self.scheme, self.tta = device, scheme, tta
        self.passes = max(int(passes), 1)
        self._det = None

    def detect(self, img: np.ndarray, max_faces: int = 1) -> list:
        """BGR -> 검출 박스 [x1,y1,x2,y2] (큰 순)."""
        if self._det is None:
            from ..data.detect import FaceDetector
            self._det = FaceDetector()
        boxes = sorted(self._det(img), key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
        return boxes[:max_faces]

    @torch.no_grad()
    def from_box(self, img: np.ndarray, box) -> np.ndarray:
        """박스 하나로 크롭 → 예측 → 원본 좌표 (N,2)."""
        S = self.spec.image_size
        o = apply_geometric(img, None, None, box, GeoParams(out_size=S, pad=self.spec.crop_pad))
        x = to_tensor(o["image"])[None].to(self.device)
        p = predict(self.model, x, self.scheme, self.tta)[0].cpu().numpy() * S
        return to_original(p, o["transform"])

    def refine(self, img: np.ndarray, lmk: np.ndarray) -> np.ndarray:
        """이전 랜드마크로 학습과 같은 박스를 잡아 재예측한다 (② 단계 · 영상 추적)."""
        return self.from_box(img, crop_box(lmk, self.spec.crop_mode))

    def __call__(self, img: np.ndarray, max_faces: int = 1) -> list:
        out = []
        for b in self.detect(img, max_faces):
            # 검출 박스는 눈썹~턱 근처를 잡는다. 정사각으로 넓혀 ①차 크롭을 만든다.
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            h = max(b[2] - b[0], b[3] - b[1]) / 2
            lmk = self.from_box(img, [cx - h, cy - h, cx + h, cy + h])
            for _ in range(self.passes - 1):
                lmk = self.refine(img, lmk)
            out.append(lmk)
        return out
