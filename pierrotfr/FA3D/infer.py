"""추론 진입점 — 체크포인트를 읽어 얼굴을 낸다.

이 저장소의 핵심 파일이다. 세 층으로 나뉜다.

    load_checkpoint / load_deployed     가중치 -> 추론용 모델 (설정은 ckpt 에서 복원)
    predict_* / to_landmarks            모델 -> 62-d -> 68 랜드마크 / 38,365 정점
    FA3D (클래스)                        이미지 한 장 -> 얼굴 (검출·크롭까지 묶은 것)

⚠ **설정은 사람이 다시 적지 않는다.** 백본 종류·입력 크기·테두리 전처리는 전부
  체크포인트 안의 `config` 에 있다. 학습 저장소의 626줄짜리 args 파일을 여기로
  옮겨 오면 두 곳이 반드시 어긋난다 — 실제로 `aug_border` 하나를 빠뜨려 같은
  가중치를 3.688 이 아니라 3.948 로 잘못 잰 적이 있다.

⚠ 다만 **경로는 체크포인트를 믿지 않는다.** 안에 박힌 `test_dir` 등은 학습 서버의
  절대경로다. 데이터 위치는 이 저장소의 `configs/paths.py` 가 정한다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader

from .bfm import BFM, ParamNorm
from .crop import crop_img, parse_roi_box_from_bbox, parse_roi_box_from_landmark
from .data import CropTestDataset, preprocess
from .models import build_model


@dataclass
class ModelSpec:
    """체크포인트가 스스로 밝히는 자기 설정."""
    model_name  : str = "mobilenet_v1"
    num_params  : int = 62
    model_extra : dict = field(default_factory=dict)
    image_size  : int = 120
    # 학습·평가 공통 전처리. 사람이 고르는 값이 아니다 — data.zero_border 참조.
    border      : int = 0
    # 사람이 읽는 꼬리표
    name        : str = ""          # 런 디렉토리 이름
    preset      : str = ""
    epoch       : int | None = None
    best        : float | None = None

    def describe(self) -> str:
        head = f"{self.name or '?'}"
        if self.preset:
            head += f" (preset={self.preset})"
        tail = f"{self.model_name} · 입력 {self.image_size} · 테두리 {self.border}px"
        if self.best is not None:
            tail += f" · 학습 당시 best NME {self.best:.4f}"
        if self.epoch is not None:
            tail += f" @ep{self.epoch}"
        return f"{head}\n  {tail}"


# ------------------------------------------------------------------ #
# ① 가중치 -> 모델
# ------------------------------------------------------------------ #
def load_checkpoint(ckpt_fp: str, device="cuda", verbose: bool = True):
    """학습 저장소의 `runs/fa3d/<런>/best.pth` -> (모델, ModelSpec).

    학습 전용 텐서는 **싣지 않고 버린다**:

        fc_lm.*        68점 회귀 헤드 (논문 2.3절 lrr) — 추론 경로에 없다
        synergy.*      SynergyNet 순환 (MLP_for/MLP_rev) — 추론 경로에 없다

    저자 배포본도 `fc_lm` 을 체크포인트에 담아 둔 채 로더가 버린다. 여기서는
    한 걸음 더 가서 **모듈 자체를 만들지 않는다**(`use_lrr=False`) — 그래야 파라미터
    수와 지연시간이 실제 배포되는 모델의 값이 된다.
    """
    if not os.path.isfile(ckpt_fp):
        raise SystemExit(f"[FA3D] 체크포인트가 없습니다: {ckpt_fp}")
    ck = torch.load(ckpt_fp, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})

    spec = ModelSpec(
        model_name=cfg.get("model_name", "mobilenet_v1"),
        num_params=int(cfg.get("num_params", 62)),
        model_extra=dict(cfg.get("model_extra", {}) or {}),
        image_size=int(cfg.get("image_size", 120)),
        border=int(cfg.get("aug_border", 0)),
        name=os.path.basename(os.path.dirname(os.path.abspath(ckpt_fp))),
        preset=cfg.get("preset", ""),
        epoch=ck.get("epoch"),
        best=ck.get("best"),
    )

    model = build_model(spec.model_name, num_params=spec.num_params, use_lrr=False,
                        **spec.model_extra)
    sd = ck.get("model", ck.get("state_dict", ck))
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    dropped = [k for k in sd if k.startswith(("fc_lm", "synergy"))]
    sd = {k: v for k, v in sd.items() if not k.startswith(("fc_lm", "synergy"))}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        # 여기서 조용히 넘어가면 **초기값 그대로인 층**으로 추론하게 된다.
        raise SystemExit(f"[FA3D] 체크포인트에 없는 가중치가 있습니다 ({len(missing)}개): "
                         f"{sorted(missing)[:8]}\n  {ckpt_fp}")
    if verbose:
        print(f"체크포인트 {ckpt_fp}\n  {spec.describe()}")
        if dropped:
            print(f"  학습 전용 텐서 {len(dropped)}개 제외 "
                  f"({sorted({k.split('.')[0] for k in dropped})})")
        if unexpected:
            print(f"  ⚠ 쓰이지 않은 텐서 {len(unexpected)}개: {sorted(unexpected)[:5]}")
    return model.to(device).eval(), spec


def load_deployed(which: str = "mb1", device="cuda", weight_root: str | None = None,
                  verbose: bool = True):
    """3DDFA_V2 저자 **배포 가중치** -> (모델, ModelSpec). 대조군이다.

    배포본은 DataParallel 로 저장돼 `module.` 접두사가 붙어 있고 `fc_lm` 도 들어
    있다 (공개된 모델 정의에는 없는데도 — 저자가 lrr 분기를 빼고 배포했다는 증거다).

    ⚠ 배포 가중치는 **테두리 전처리를 쓰지 않는다** (border=0). 우리 모델과 나란히
      잴 때 각자 학습 때 쓴 값을 따로 걸어야 한다 — 한쪽 값을 양쪽에 걸면 비교가
      깨진다.
    """
    from configs.paths import V2_ROOT, WEIGHT_ROOT
    name = {"mb1": "mb1_120x120.pth", "mb05": "mb05_120x120.pth"}.get(which)
    if name is None:
        raise SystemExit(f"[FA3D] which={which!r} — 'mb1' | 'mb05'")
    roots = [weight_root or WEIGHT_ROOT]
    if V2_ROOT:
        roots.append(os.path.join(V2_ROOT, "weights"))
    for r in roots:
        fp = os.path.join(r, name)
        if os.path.isfile(fp):
            break
    else:
        raise SystemExit(
            f"[FA3D] 배포 가중치를 찾지 못했습니다: {name}\n"
            + "".join(f"    {r}\n" for r in roots)
            + "  3DDFA_V2 오피셜 저장소의 weights/ 에서 가져오거나,\n"
              "  paths.local.env 의 PIERROTFR_WEIGHT_ROOT / PIERROTFR_3DDFA_V2_ROOT 를 맞추세요.")

    arch = "mobilenet_v1" if which == "mb1" else "mobilenet_v1_x05"
    model = build_model(arch, num_params=62, use_lrr=False)
    sd = torch.load(fp, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if not k.startswith("fc_lm")}
    missing, _ = model.load_state_dict(sd, strict=False)
    if missing:
        raise SystemExit(f"[FA3D] 배포 가중치 로드 실패 — missing {sorted(missing)[:8]}")
    spec = ModelSpec(model_name=arch, border=0,
                     name=f"3DDFA_V2 배포 {which}", preset="deployed")
    if verbose:
        print(f"배포 가중치 {fp}\n  {spec.describe()}")
    return model.to(device).eval(), spec


# ------------------------------------------------------------------ #
# ② 모델 -> 62-d -> 얼굴
# ------------------------------------------------------------------ #
@torch.no_grad()
def predict_params(model, loader, device="cuda") -> np.ndarray:
    """배치 이터러블 -> [N, 62] (정규화 공간)."""
    was_training = model.training
    model.eval()
    out = [model.predict(x.to(device)).float().cpu() for x in loader]
    if was_training:
        model.train()
    return torch.cat(out).numpy()


def predict_filelist(model, root: str, filelist: str, device="cuda",
                     border: int = 0, batch_size: int = 256, num_workers: int = 8,
                     limit: int = 0, flip: bool = False) -> np.ndarray:
    """사전 크롭된 목록 -> [N, 62].

    ⚠ `border` 는 그 모델이 **학습 때 쓴 값**이어야 한다 (`spec.border`).
      호출자가 실수로 빠뜨리기 쉬운 자리라 기본값을 0 으로 두지 않고 싶었지만,
      배포 가중치가 실제로 0 이라 그럴 수 없다 — 대신 `spec.border` 를 넘기라는
      규약을 모든 호출부에서 지킨다.
    """
    ds = CropTestDataset(root, filelist, border=border, flip=flip)
    if limit:
        ds.lines = ds.lines[:limit]
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True)
    return predict_params(model, dl, device)


def to_landmarks(bfm: BFM, param_norm: ParamNorm, params: np.ndarray,
                 device="cuda", size: int = 120, chunk: int = 4096) -> list:
    """정규화 62-d [N,62] -> 크롭 좌표계 68 랜드마크 [N][2,68].

    청크로 나누는 이유: 38,365 정점 basis 와의 matmul 이라 N 이 크면 GPU 메모리가
    먼저 터진다. 68점만 쓰는 sparse 경로라도 배치가 2만이면 무겁다.
    """
    out = []
    p = torch.as_tensor(params, device=device)
    for i in range(0, p.shape[0], chunk):
        l = bfm.landmarks3d(param_norm.denorm(p[i:i + chunk]), size=size)   # [b,3,68]
        out.append(l[:, :2].cpu().numpy())
    return list(np.concatenate(out, axis=0))


# ------------------------------------------------------------------ #
# ③ 이미지 한 장 -> 얼굴 (검출 · 크롭 · 디코딩을 한 덩어리로)
# ------------------------------------------------------------------ #
class FA3D:
    """단일 이미지/프레임 추론기.

        fa = FA3D(ckpt="runs/fa3d/<런>/best.pth")
        for roi in fa.detect(img_bgr):
            param = fa.param(img_bgr, roi)          # [62] 정규화 공간
            lmk, verts = fa.decode(param, roi)      # [3,68] · [3,38365] 원본 좌표

    왜 세 단계로 쪼개 두나 — **영상에서는 파라미터가 단위**이기 때문이다. 평활도
    추적도 62-d 위에서 한 번만 걸어야 랜드마크·메쉬·새 시점이 서로 어긋나지 않는다.
    한 번에 랜드마크까지 내는 API 로 묶으면 그걸 못 한다.
    """

    def __init__(self, ckpt: str = "", model=None, spec: ModelSpec | None = None,
                 device: str = "cuda", cfg=None, verbose: bool = True):
        from configs.eval_fa3d import config, require_bfm
        self.cfg = require_bfm(cfg or config())
        self.device = device
        if model is None:
            model, spec = load_checkpoint(ckpt, device, verbose=verbose)
        self.model, self.spec = model, (spec or ModelSpec())
        self.bfm = BFM(self.cfg.bfm_fp).to(device)
        self.param_norm = ParamNorm(self.cfg.dst_stats).to(device)
        self._det = None
        self._tri = None

    # -- 삼각형 인덱스는 메쉬를 그릴 때만 필요하다 (23MB) — 늦게 읽는다 ----
    @property
    def tri(self) -> np.ndarray:
        if self._tri is None:
            from .bfm import load_tri
            self._tri = load_tri(self.cfg.tri_fp, self.bfm.n_vertex)
        return self._tri

    # -- 검출 -----------------------------------------------------------
    def detect(self, img: np.ndarray, max_faces: int = 1) -> list:
        """BGR 이미지 -> roi_box 리스트 (큰 얼굴 순). 검출기는 늦게 만든다."""
        if self._det is None:
            from .detect import FaceDetector
            self._det = FaceDetector()
        boxes = self._det(img)
        boxes = sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))[:max_faces]
        return [parse_roi_box_from_bbox(b) for b in boxes]

    @staticmethod
    def track(lmk: np.ndarray) -> list:
        """이번 프레임 랜드마크 -> 다음 프레임 roi_box (3DDFA_V2 의 영상 방식)."""
        return parse_roi_box_from_landmark(lmk)

    # -- 추론 -----------------------------------------------------------
    @torch.no_grad()
    def param(self, img: np.ndarray, roi) -> np.ndarray | None:
        """BGR 원본 + roi_box -> 62-d [62] (정규화 공간). 크롭이 비면 None."""
        patch = crop_img(img, roi)
        if patch is None or patch.size == 0:
            return None
        x = preprocess(patch, self.spec.image_size, self.spec.border)
        return self.model.predict(x.to(self.device))[0].float().cpu().numpy()

    def decode(self, param: np.ndarray, roi, dense: bool = True):
        """62-d -> (랜드마크 [3,68], 정점 [3,N] 또는 None) — **원본 이미지 좌표**.

        ⚠ 랜드마크와 메쉬는 **같은 파라미터**에서 나와야 한다. 예전엔 랜드마크만
          평활하고 메쉬는 원본을 써서 둘이 어긋났다.
        """
        size = self.spec.image_size
        p = self.param_norm.denorm(torch.as_tensor(param[None], device=self.device))
        L = self.bfm.landmarks3d(p, size=size)[0].detach().cpu().numpy()   # [3,68]
        V = None
        if dense:
            V = self.bfm.reconstruct(p)[0].detach().cpu().numpy()          # [3,38365]
            V[1] = size - V[1]              # landmarks3d 와 같은 y 규약으로 맞춘다
        sx, sy, ex, ey = roi
        for A in (L, V):
            if A is None:
                continue
            A[0] = A[0] * (ex - sx) / size + sx
            A[1] = A[1] * (ey - sy) / size + sy
        return L, V
