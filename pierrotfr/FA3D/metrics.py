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

# 68점 좌우 대응 — 이미지를 좌우반전하면 왼눈/오른눈 같은 쌍이 서로 바뀐다.
# ibug 68 규약: 턱 0~16 역순 · 눈썹 17~21 <-> 22~26 · 코 27~30 고정, 31~35 역순 ·
# 눈 36~41 <-> 42~47 (각각 순환) · 입 48~59, 60~67 각각 좌우 대응.
FLIP68 = np.array([16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                   26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
                   27, 28, 29, 30, 35, 34, 33, 32, 31,
                   45, 44, 43, 42, 47, 46, 39, 38, 37, 36, 41, 40,
                   54, 53, 52, 51, 50, 49, 48, 59, 58, 57, 56, 55,
                   64, 63, 62, 61, 60, 67, 66, 65], dtype=np.int64)

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


# ------------------------------------------------------------------ #
# 좌우반전 TTA — 재학습 없이 얻는 개선. 대가는 추론 2배.
# ------------------------------------------------------------------ #
def flip_back(lmk, size: int = 120) -> np.ndarray:
    """좌우반전 입력으로 얻은 랜드마크를 원래 좌표계로. lmk [N, 2 이상, 68].

    x 를 뒤집고 좌우 쌍을 교환한다. **둘 다** 해야 한다 — x 만 뒤집으면 왼눈 자리에
    오른눈 예측이 들어간다.
    """
    out = np.asarray(lmk).copy()
    out[:, 0] = (size - 1.0) - out[:, 0]
    return out[:, :, FLIP68]


def flip_tta(lmk, lmk_flipped, size: int = 120) -> np.ndarray:
    """원본 예측과 좌우반전 예측의 평균.

    ⚠ **파라미터 공간이 아니라 랜드마크 공간에서 평균한다.** 62-d 의 앞 12개는 회전
      행렬이라 선형 평균이 회전을 보존하지 않는다. 랜드마크는 좌표라 평균이 정의된다.
      이 제약 때문에 밀집 메쉬에는 TTA 를 걸 수 없다 — 메쉬는 파라미터에서 나온다.

    실측(AFLW 21,080장): 5.085 -> 4.974. 학습 축을 통틀어 가장 큰 개선이다.
    """
    return (np.asarray(lmk) + flip_back(lmk_flipped, size)) / 2.0
