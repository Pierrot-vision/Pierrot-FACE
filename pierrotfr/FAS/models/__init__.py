"""FAS 모델 레지스트리 (추론 경로만).

    predict(batch) -> dict
        score   [B]  spoof 확률
        margin  [B]  spoof log-odds (평가 지표는 이것으로 잰다)
        cue     [B, 1, h, w] | None
"""
from __future__ import annotations

from .backbones import BACKBONES, build_backbone
from .instructflip import InstructFLIP
from .minifasnet import MiniFASNet

MODELS = {
    "instructflip": InstructFLIP,     # 176M (LLM 제외) · VLM 계열 (ACM MM 2025)
    "minifasnet": MiniFASNet,         # 0.43M · MobileFaceNet 계열 경량 대조군
}


def build_model(name: str, **kwargs):
    if name not in MODELS:
        raise ValueError(f"model={name!r} — 가능: {sorted(MODELS)}")
    return MODELS[name](**kwargs)


__all__ = ["MODELS", "build_model", "BACKBONES", "build_backbone", "InstructFLIP", "MiniFASNet"]
