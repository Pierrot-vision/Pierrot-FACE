"""FA2D 지표 — 추론 경로만.

    head_nme   크롭 변 길이 기준 평균 L2 (%)  — 규약 무관, 항상 정의된다
    io_nme     inter-ocular 기준 (%)          — 눈 바깥꼬리 거리로 정규화 (WFLW 표준)
    fr10       io_nme > 10% 인 샘플 비율 (%)
    auc10      NME 누적분포의 0~10% 구간 면적 (%)

⚠ 정규화 기준이 다른 NME 를 같은 표에 놓으면 안 된다.

학습 저장소의 같은 파일에서 지터 지표(`clip_metrics`)를 뺐다 — 합성 클립(학습용 궤적
샘플러)이 있어야 잴 수 있다.
"""
from __future__ import annotations

import numpy as np

# 규약별 inter-ocular 정규화에 쓰는 눈 바깥꼬리 인덱스.
OUTER_EYE = {"ibug68": (36, 45), "wflw98": (60, 72), "cofw29": (8, 9), "lapa106": (66, 79)}


def nme(pred: np.ndarray, gt: np.ndarray, scheme: str, side: float,
        norm: str = "head") -> np.ndarray:
    """(B,N,2) 픽셀 좌표 → 샘플별 NME(%)."""
    err = np.linalg.norm(pred - gt, axis=-1).mean(axis=-1)
    if norm == "head":
        d = np.full(len(err), float(side))
    elif norm == "interocular":
        a, b = OUTER_EYE[scheme]
        d = np.linalg.norm(gt[:, a] - gt[:, b], axis=-1)
    else:
        raise ValueError(f"norm={norm!r} — head | interocular")
    return err / np.maximum(d, 1e-9) * 100.0


def failure_rate(nmes: np.ndarray, thr: float = 10.0) -> float:
    return float((np.asarray(nmes) > thr).mean() * 100.0)


def auc(nmes: np.ndarray, thr: float = 10.0) -> float:
    """NME 누적분포의 0~thr 구간 면적(%). 높을수록 좋다."""
    x = np.clip(np.sort(np.asarray(nmes)), 0, thr)
    cdf = np.arange(1, len(x) + 1) / len(x)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(cdf, x) / thr * 100.0)
