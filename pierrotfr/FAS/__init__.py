"""Face Anti-Spoofing (얼굴 위조 방지) — 추론 · 평가.

    infer.py         체크포인트 로드 (config 로 구조 복원 · LLM 제거) + 배치 추론
    data.py          CelebA-Spoof CSV / 폴더형 벤치마크 + 평가 변환
    instructions.py  Q-Former 가 읽는 고정 지시문
    metrics.py       CelebA-Spoof 공식(APCER/BPCER/ACER · R@FPR) + DG 계열(HTER/AUC/EER)
    models/          InstructFLIP (백본 → Q-Former×2 → 융합 → 분류기 + cue)
"""
from .infer import FASModel, FASSpec, load_checkpoint, score_loader

__all__ = ["FASModel", "FASSpec", "load_checkpoint", "score_loader"]
