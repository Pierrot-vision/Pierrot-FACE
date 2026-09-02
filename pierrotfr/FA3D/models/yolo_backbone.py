"""Pierrot_3D_Lab 의 YOLO 계열 백본을 FA3D 회귀기로 재사용한다.

동기: 3DDFA_V2 의 MobileNet-V1 은 2017년 구조이고 **스크래치**로 시작한다. 옆 랩에
이미 obj365 로 검출 사전학습을 끝낸 백본이 있고, 백본만 떼면 **더 작다**.

    MobileNet-V1 (3DDFA_V2 기본)   3.27M   스크래치
    YOLO26-n 백본                  1.37M   obj365 (오피셜 전이)
    PierrotXv2-n-accuracy 백본     1.27M   obj365 ep150 (자체 학습)

⚠ 검출 사전학습이 이 태스크에 유리하다는 보장은 없다. 검출은 "어디에 무엇이
  있는가", 여기는 "이 얼굴의 3DMM 계수는 얼마인가"로 요구하는 표현이 다르다.
  다만 저수준 특징(에지·텍스처)은 공유되고, 무엇보다 **스크래치 대비 공짜 초기값**이라
  실측해 볼 값이 있다. 그래서 프리셋으로 갈라 둔다.

⚠ 백본은 stride 32 라 입력이 32의 배수여야 한다. 크롭은 120x120 이므로 네트워크
  입력만 128 로 리사이즈한다 — **GT 좌표계는 120 그대로**다. 이미지를 늘려도
  3DMM 파라미터는 그대로이므로 라벨을 건드리면 안 된다.

넥/헤드는 버린다. 검출용 PAN 넥은 다중 해상도 위치 정보를 살리는 구조인데,
여기 출력은 전역 62-d 회귀라 풀링 뒤에는 의미가 없다.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# 백본 **정의**를 옆 랩에서 가져온다 (가중치가 아니다 — 그건 우리 체크포인트에 있다).
# 서버마다 경로가 다르므로 환경변수로 뺀다. 이 백본으로 학습한 체크포인트를 쓸
# 때만 필요하고, mobilenet 계열만 돌린다면 이 경로가 없어도 된다.
DEFAULT_LAB_ROOT = os.environ.get("PIERROTFR_3D_LAB_ROOT",
                                  "/ai_data_new_ssd/DEV/3d/Pierrot_3D_Lab")


def _import_pose_models(lab_root: str):
    """Pierrot_3D_Lab 을 import 경로에 올린다 (코드 복제 대신 참조).

    복제하면 옆 랩이 백본을 고칠 때마다 조용히 갈라진다. 대신 결합이 생기므로
    경로는 설정으로 빼고, 없을 때 무엇을 해야 하는지 명확히 알린다.
    """
    if not os.path.isdir(lab_root):
        raise SystemExit(
            f"[FA3D] Pierrot_3D_Lab 을 찾을 수 없습니다: {lab_root}\n"
            f"  이 체크포인트는 옆 랩의 YOLO 백본으로 학습됐고, 백본 **정의**가\n"
            f"  그쪽에 있습니다 (가중치는 체크포인트 안에 다 들어 있습니다).\n"
            f"    export PIERROTFR_3D_LAB_ROOT=/path/to/Pierrot_3D_Lab\n"
            f"  mobilenet 계열 체크포인트만 쓴다면 이 경로는 필요 없습니다.")
    if lab_root not in sys.path:
        sys.path.insert(0, lab_root)
    try:
        from pierrot3d.Pose.models import ARCHS, resolve
        from pierrot3d.Pose.models.backbone import Backbone
    except ImportError as e:                                  # noqa: BLE001
        raise SystemExit(f"[FA3D] Pierrot_3D_Lab import 실패: {e}") from e
    return ARCHS, resolve, Backbone


class YoloBackboneRegressor(nn.Module):
    """(YOLO 백본 P3/P4/P5) -> 전역 풀링 concat -> fc_param / fc_lm.

    세 스케일을 모두 쓰는 이유: 62-d 중 T(12) 는 얼굴 전체의 위치·크기라 저해상도
    P5 가 적합하고, 표정 계수(10)는 눈·입 같은 국소 변형이라 P3 의 세밀함이 필요하다.
    P5 만 쓰면 표정 쪽이 뭉개진다.
    """

    def __init__(self, arch: str = "pierrotxv2-n-accuracy", scale: str = "n",
                 num_params: int = 62, num_landmarks: int = 68,
                 use_lrr: bool = False, input_size: int = 128,
                 lab_root: str = DEFAULT_LAB_ROOT, det_weights: str | None = None,
                 dropout: float = 0.0):
        super().__init__()
        ARCHS, resolve, Backbone = _import_pose_models(lab_root)
        if arch not in ARCHS:
            raise SystemExit(f"[FA3D] arch={arch!r} 없음 — 등록된 값: {sorted(ARCHS)}")

        self.arch, self.scale, self.input_size = arch, scale, input_size
        self.use_lrr = use_lrr
        self.backbone = Backbone(resolve(ARCHS[arch], scale))
        feat = sum(self.backbone.out_ch)                      # P3 + P4 + P5 채널
        self.feat_dim = feat          # 학습 저장소의 synergy 순환이 물어보던 값

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc_param = nn.Linear(feat, num_params)
        self.fc_lm = nn.Linear(feat, num_landmarks * 2) if use_lrr else None
        # 회귀 헤드는 작게 시작해야 초기 손실이 폭주하지 않는다. 특히 VDC 는 정점
        # 오차가 파라미터에 제곱으로 들어가 초기값이 크면 첫 스텝에 발산한다.
        for fc in (self.fc_param, self.fc_lm):
            if fc is not None:
                nn.init.normal_(fc.weight, std=0.01)
                nn.init.zeros_(fc.bias)

        if det_weights:
            self.load_detection_weights(det_weights)

    # ---------------------------------------------------------------- #
    def load_detection_weights(self, path: str, verbose: bool = True) -> None:
        """검출 체크포인트에서 `backbone.*` 만 떼어 싣는다 (넥·헤드는 버린다)."""
        if not os.path.isfile(path):
            raise SystemExit(f"[FA3D] 검출 사전학습 가중치가 없습니다: {path}")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck.get("model", ck.get("state_dict", ck))
        if hasattr(sd, "state_dict"):
            sd = sd.state_dict()
        bb = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
        if not bb:
            raise SystemExit(f"[FA3D] {path} 에 backbone.* 텐서가 없습니다")
        missing, unexpected = self.backbone.load_state_dict(bb, strict=False)
        if verbose:
            print(f"[FA3D] 검출 사전학습 로드 {os.path.basename(path)} — "
                  f"{len(bb)} 텐서 · missing {len(missing)} · unexpected {len(unexpected)}")
            if missing or unexpected:
                print(f"        missing {sorted(missing)[:5]} / "
                      f"unexpected {sorted(unexpected)[:5]}")

    # ---------------------------------------------------------------- #
    def features(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_size:
            # 120 -> 128. GT 파라미터는 120 크롭 기준 그대로 유지된다.
            x = F.interpolate(x, size=(self.input_size, self.input_size),
                              mode="bilinear", align_corners=False)
        feats = self.backbone(x)                              # (P3, P4, P5)
        pooled = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in feats]
        return self.dropout(torch.cat(pooled, dim=1))

    def forward(self, x: torch.Tensor) -> dict:
        f = self.features(x)
        out = {"param": self.fc_param(f), "feat": f}
        if self.fc_lm is not None:
            out["landmark"] = self.fc_lm(f).view(f.shape[0], -1, 2)
        return out

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_param(self.features(x))

    def inference_modules(self):
        return [self.backbone, self.fc_param]


def _factory(arch: str):
    def make(pretrained_det: bool = False, det_weights: str | None = None, **kw):
        """⚠ `pretrained_det` 기본값이 학습 저장소와 **반대(False)** 다.

        검출 사전학습 가중치는 학습의 **초기값**이었다. 추론에서는 우리 체크포인트가
        그 위를 통째로 덮으므로 읽어 봐야 버려지고, 없는 파일을 찾다 죽기만 한다.
        학습 때 쓴 model_extra 가 그대로 넘어와도 안전하도록 인자는 남겨 둔다.
        """
        return YoloBackboneRegressor(arch=arch,
                                     det_weights=det_weights if pretrained_det else None,
                                     **kw)
    return make


yolo26_n = _factory("v26")
pierrotxv2_n = _factory("pierrotxv2-n-accuracy")
