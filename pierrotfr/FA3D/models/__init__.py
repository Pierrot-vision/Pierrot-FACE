"""FA3D 백본 레지스트리 — 추론 경로만.

계약은 하나다.

    predict(x) -> [B, 62]      # 추론 경로. lrr 헤드도, synergy 순환도 타지 않는다
    forward(x) -> {"param": [B, 62], "feat": [B, F], "landmark": …}

학습 저장소와의 차이:

  · `use_lrr` 기본값이 **False** 다. 68점 회귀 헤드(fc_lm)는 논문 2.3절의
    landmark-regression **정규화**라 학습에만 있고 추론에서는 버려진다 — 저자
    배포본도 체크포인트에 담아 두고 로더가 버린다. 여기서는 아예 만들지 않는다.
  · `use_synergy` 인자가 없다. SynergyNet 순환(MLP_for/MLP_rev)도 학습 전용이라
    `predict()` 를 타지 않는다. 체크포인트에 남아 있는 `synergy.*` 텐서는
    `infer.load_checkpoint()` 가 걸러 낸다.

즉 **추론 비용은 두 축 모두 0** 이다. 그래서 이 저장소의 모델은 학습 때보다 작다.
"""
from __future__ import annotations

from .mobilenet_v1 import MobileNetV1, mobilenet_v1, mobilenet_v1_x05
from .mobilenet_v2 import (MobileNetV2, mobilenet_v2, mobilenet_v2_imagenet,
                           mobilenet_v2_x05)
from .yolo_backbone import YoloBackboneRegressor, pierrotxv2_n, yolo26_n

MODELS = {
    # ── 논문 원본 (2017년 구조 · 스크래치) ──
    "mobilenet_v1": mobilenet_v1,          # 3.27M · 183.5M MACs — 논문 기본 백본
    "mobilenet_v1_x05": mobilenet_v1_x05,  # 0.85M ·  49.5M MACs — 경량 변종
    # ── SynergyNet(3DV 2021) 백본. V1 보다 작고 논문 수치도 낫다 (3.41 vs 3.51) ──
    "mobilenet_v2": mobilenet_v2,          # 2.30M — 헤드 3분할(ori/shape/exp)
    "mobilenet_v2_x05": mobilenet_v2_x05,  # 경량 변종
    "mobilenet_v2_imagenet": mobilenet_v2_imagenet,  # ImageNet 초기값 (논문 밖 축)
    # ── 옆 랩(Pierrot_3D_Lab)의 검출 사전학습 백본. 더 작고 초기값이 있다 ──
    "yolo26_n": yolo26_n,                  # 백본 1.37M — obj365 오피셜 전이
    "pierrotxv2_n": pierrotxv2_n,          # 백본 1.27M — obj365 ep150 자체 학습
}

# ImageNet 가중치를 받아 오는 변종은 추론에서 의미가 없다 — 우리 체크포인트가 덮는다.
# 그런데 생성자가 torchvision 다운로드를 타므로 이름만 같은 스크래치 구조로 바꾼다.
_IMAGENET_ALIAS = {"mobilenet_v2_imagenet": "mobilenet_v2"}


def build_model(name: str, use_lrr: bool = False, **kwargs):
    """백본을 만든다. 추론 경로만이라 bfm/param_norm 을 받지 않는다.

    학습 저장소의 `build_model(name, use_synergy=..., bfm=..., param_norm=...)` 과
    시그니처가 다르다 — 그쪽 인자는 전부 학습 전용 순환 블록을 붙이기 위한 것이었다.
    """
    if name not in MODELS:
        raise ValueError(f"model={name!r} — 가능: {sorted(MODELS)}")
    if name in _IMAGENET_ALIAS:
        name = _IMAGENET_ALIAS[name]
        kwargs.pop("imagenet", None)
    return MODELS[name](use_lrr=use_lrr, **kwargs)


__all__ = ["MODELS", "build_model", "MobileNetV1",
           "mobilenet_v1", "mobilenet_v1_x05",
           "MobileNetV2", "mobilenet_v2", "mobilenet_v2_x05", "mobilenet_v2_imagenet",
           "YoloBackboneRegressor", "yolo26_n", "pierrotxv2_n"]
