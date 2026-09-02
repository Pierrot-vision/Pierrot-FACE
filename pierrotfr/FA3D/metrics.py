"""NME — AFLW2000-3D / AFLW 프로토콜.

⚠ 3DDFA 계열이 보고하는 값은 **전체 평균이 아니라 yaw 3구간 평균의 평균**이다.
  AFLW2000-3D 는 정면이 1312/2000 이라 단순 평균을 내면 대각(large-pose) 오차가
  묻힌다. 같은 예측이 단순평균 3.255%, 구간평균 3.683% 로 갈린다 — 다른 논문 수치와
  비교할 때 이 차이를 확인하지 않으면 0.4%p 를 그냥 잃거나 얻는다.

정규화 상수는 GT 랜드마크 bbox 의 √(w·h) 다 (눈 사이 거리가 아니다 — 대각에서
두 눈이 겹쳐 발산한다).

⚠⚠ **이 지표는 사실상 자세를 잰다.** 형상(identity)을 GT 로 완벽히 바꿔도 1.3%
  개선뿐이고, 형상을 아예 안 내고 평균 얼굴을 쓰면 오히려 낫다. 3D 형상 정확도는
  `scripts/FA3D/shape_accuracy.py` 로 따로 재야 한다.
"""
from __future__ import annotations

from math import sqrt

import numpy as np

from .crop import to_original

# AFLW 21점은 68점의 부분집합이 아니다. 눈/입은 여러 점의 평균으로 만든다.
# (3DDFA v1 benchmark_aflw.py 의 ind_68to21, 1-based -> 0-based)
IND_68TO21 = [[g - 1 for g in grp] for grp in
              [[18], [20], [22], [23], [25], [27], [37],
               [37, 38, 39, 40, 41, 42], [40], [43],
               [43, 44, 45, 46, 47, 48], [46], [3], [32], [31], [36], [15],
               [49], [61, 62, 63, 64, 65, 66, 67, 68], [55], [9]]]

# 68점 그룹 — 어느 부위에서 지는지 갈라야 처방이 나온다 (error_analysis.py)
GROUPS = {"윤곽(턱선)": range(0, 17), "눈썹": range(17, 27), "코": range(27, 36),
          "눈": range(36, 48), "입": range(48, 68)}


def nme_aflw2000(pred_lmk, roi_boxes, gt68, size: int = 120) -> np.ndarray:
    """pred_lmk: [N][2,68] (크롭 좌표), gt68: [N,2or3,68] (원본 좌표)"""
    out = []
    for i in range(len(roi_boxes)):
        fit = to_original(pred_lmk[i], roi_boxes[i], size)
        gt = gt68[i][:2]
        llen = sqrt((gt[0].max() - gt[0].min()) * (gt[1].max() - gt[1].min()))
        out.append(np.sqrt(((fit - gt) ** 2).sum(0)).mean() / llen)
    return np.array(out, dtype=np.float32)


def nme_aflw(pred_lmk, roi_boxes, gt68, gt21, size: int = 120) -> np.ndarray:
    """AFLW 21점 프로토콜. bbox 는 68점 GT 로 잡고, 미주석 점(-1)은 제외한다."""
    out = []
    for i in range(len(roi_boxes)):
        fit = to_original(pred_lmk[i], roi_boxes[i], size)
        g68, g21 = gt68[i], gt21[i]
        est = np.stack([fit[:, idx].mean(1) for idx in IND_68TO21], axis=1)  # [2,21]
        llen = sqrt((g68[0].max() - g68[0].min()) * (g68[1].max() - g68[1].min()))
        valid = (g21[0] != -1) & (g21[1] != -1)
        out.append(np.sqrt(((est[:, valid] - g21[:, valid]) ** 2).sum(0)).mean() / llen)
    return np.array(out, dtype=np.float32)


def fold_68to21(lmk: np.ndarray) -> np.ndarray:
    """68점 예측 [2 이상, 68] -> AFLW 21점 [2, 21]. 시각화에서 GT 와 나란히 놓을 때."""
    return np.stack([lmk[:2][:, idx].mean(1) for idx in IND_68TO21], axis=1)


def summarize(nme: np.ndarray, yaws: np.ndarray) -> dict:
    """yaw 구간별 + 논문 규약(구간평균의 평균) + 전체평균."""
    a = np.abs(yaws)
    masks = [a <= 30, (a > 30) & (a <= 60), a > 60]
    means = [float(nme[m].mean() * 100) for m in masks]
    return {
        "NME_0_30": means[0], "NME_30_60": means[1], "NME_60_90": means[2],
        "NME": float(np.mean(means)),          # ★ 논문 규약 — 모델 선택도 이 값이었다
        "NME_all": float(nme.mean() * 100),    # 단순 평균 (다른 논문과 비교할 때만)
        "std": float(np.std(means)),
    }


def format_row(name: str, m: dict) -> str:
    return (f"{name:<16s} "
            f"[0,30] {m['NME_0_30']:.3f} · [30,60] {m['NME_30_60']:.3f} · "
            f"[60,90] {m['NME_60_90']:.3f} | NME {m['NME']:.3f} "
            f"(전체평균 {m['NME_all']:.3f})")
