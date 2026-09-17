"""FA2D 모델 레지스트리 — 추론 경로만.

    PointQueryNet   백본 → 점 쿼리 트랜스포머 디코더 → 좌표 · 가시성 · 자세
                    (+ 선택: refine · heatmap · hm_offset · boundary · state)

선택 분기는 **구조의 일부**라 전부 남긴다. 어느 것이 켜졌는지는 체크포인트가 정한다
(`full_v12` = refine, `stu_final_v2` = hm_offset). 뺀 것은 그 분기들의 손실뿐이다.
"""
from __future__ import annotations

from .backbones import BACKBONES, build_backbone, recommended_norm
from .decoder import SCHEMES, PointQueryNet

MODELS = {"pointquery": PointQueryNet}


def build_model(name: str, **kwargs):
    if name not in MODELS:
        raise ValueError(f"model={name!r} — 가능: {sorted(MODELS)}")
    return MODELS[name](**kwargs)


__all__ = ["MODELS", "build_model", "BACKBONES", "build_backbone",
           "recommended_norm", "SCHEMES", "PointQueryNet"]
