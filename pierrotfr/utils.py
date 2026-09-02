"""태스크 공통 소품.

학습 저장소의 같은 파일에는 logger 와 cosine LR 스케줄이 있었다 — 둘 다 학습
전용이라 여기 없다. 남은 것은 모델 크기를 보고할 때 쓰는 두 함수뿐이다.
"""
from __future__ import annotations


def count_params(module) -> tuple[int, int]:
    """(전체, 학습 가능) 파라미터 수."""
    total = sum(p.numel() for p in module.parameters())
    train = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, train


def human(n: int) -> str:
    for unit in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:.0f}{unit}"
        n /= 1000.0
    return f"{n:.1f}T"
