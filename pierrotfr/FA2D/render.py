"""2D 랜드마크 그리기 — 점보다 **선**이 자세·표정을 보여 준다.

윤곽선 순서는 경계 헤드와 같은 `BOUNDARY_GROUPS` 를 쓴다 — 모델이 학습한 곡선 정의와
그림의 곡선이 같아야 "어느 곡선 위에서 미끄러졌는지"가 그림에서 읽힌다.
"""
from __future__ import annotations

import cv2
import numpy as np

from .models.boundary import BOUNDARY_GROUPS

COLORS = [(96, 226, 26), (60, 170, 255), (255, 170, 60), (200, 90, 255), (60, 240, 255)]  # BGR


def draw_landmarks(img: np.ndarray, pts: np.ndarray, scheme: str = "wflw98",
                   color=(96, 226, 26), thick: int = 1) -> np.ndarray:
    """(N,2) 원본 좌표. 4배로 키워 그린 뒤 줄여 안티에일리어싱한다."""
    S = 4
    ov = cv2.resize(img, None, fx=S, fy=S, interpolation=cv2.INTER_LINEAR)
    P = np.round(np.asarray(pts, np.float64) * S).astype(np.int32)
    groups = BOUNDARY_GROUPS.get(scheme)
    for w, col in ((thick * S + 3, (18, 18, 18)), (thick * S, color)):   # 그림자 → 본선
        if groups:
            for g in groups:
                cv2.polylines(ov, [P[g].reshape(-1, 1, 2)], False, col, w, cv2.LINE_AA)
    r = max(thick * S, 3)
    for p in P:
        cv2.circle(ov, tuple(int(v) for v in p), r, (255, 255, 255), -1, cv2.LINE_AA)
    return cv2.resize(ov, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
