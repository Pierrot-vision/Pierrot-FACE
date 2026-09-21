"""FAS 지표.

두 계열을 **모두** 낸다. 비교 대상이 다르기 때문이다.

  A. 도메인 일반화(DG) 계열 — FLIP / CFPL / InstructFLIP 이 쓰는 축
       HTER, AUC, TPR@FPR=1%
     ⚠ HTER 임계값을 테스트셋 EER 에서 뽑는 게 이 분야 관행이다. 낙관 편향이
       있지만 선행연구가 모두 그렇게 하므로 비교를 위해 같이 낸다.
       `threshold` 를 주면 (val 에서 구한 값 등) 그쪽을 쓴다 — 이쪽이 정직하다.

  B. CelebA-Spoof 공식 intra-dataset 벤치마크 — AENet / BASN 과 비교하는 축
       APCER, BPCER, ACER, Recall@FPR (1% / 0.5% / 0.1%)
     고정 임계값(확률 0.5 = margin 0) 기준이다.

라벨 규약: 1 = spoof(attack), 0 = live(bona fide).
score 는 **spoof margin(log-odds)** 이다 — 확률은 포화해 순위 지표를 망친다(compute 참조).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

DG_KEYS = ("HTER", "AUC", "TPR@FPR1")
CELEBA_KEYS = ("APCER", "BPCER", "ACER", "R@F1", "R@F0.5", "R@F0.1")
OP_KEYS = ("APCER_op", "BPCER_op", "ACER_op")


def aggregate_by_video(scores, labels, video_ids):
    """video 단위 평균. 프레임 기반 벤치마크(MCIO/WCS)의 표준 집계다.

    CelebA-Spoof 는 이미지 단위라 video_id 가 전부 고유해 사실상 항등이 된다.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    video_ids = np.asarray(video_ids)

    uniq = np.unique(video_ids)
    s = np.array([scores[video_ids == v].mean() for v in uniq])
    y = np.array([labels[video_ids == v].mean().round().astype(int) for v in uniq])
    return s, y


def eer_threshold(scores, labels) -> tuple[float, float]:
    """(EER, 그 임계값). FAR 와 FRR 이 가장 가까운 ROC 지점.

    점수는 실수 margin(log-odds)이다 — [0,1] 격자로 훑지 않고 ROC 의 실제 임계값
    후보에서 고른다. 격자 방식은 확률이 1.0 으로 포화한 동점 무리를 쪼개지 못한다.
    """
    fpr, tpr, thr = roc_curve(labels, scores)
    frr = 1.0 - tpr
    i = int(np.argmin(np.abs(fpr - frr)))
    return float((fpr[i] + frr[i]) / 2), float(thr[i])


def _far_frr(scores, labels, thr) -> tuple[float, float]:
    pred = scores >= thr
    n_live = int((labels == 0).sum())
    n_spoof = int((labels == 1).sum())
    # FAR: live 를 spoof 로 (bona fide 오탐)
    far = float((pred & (labels == 0)).sum()) / n_live if n_live else 1.0
    # FRR: spoof 를 live 로 (공격 누락)
    frr = float((~pred & (labels == 1)).sum()) / n_spoof if n_spoof else 1.0
    return far, frr


def hter(scores, labels, thr) -> float:
    far, frr = _far_frr(np.asarray(scores, dtype=float), np.asarray(labels), thr)
    return (far + frr) / 2


def recall_at_fpr(scores, labels, targets=(0.01, 0.005, 0.001)) -> dict:
    """목표 FPR 이하에서의 최대 recall(TPR). CelebA-Spoof 논문 Table 5 형식."""
    fpr, tpr, _ = roc_curve(labels, scores)
    out = {}
    for t in targets:
        sel = tpr[fpr <= t]
        out[t] = float(sel[-1]) if len(sel) else 0.0
    return out


def threshold_at_bpcer(scores, labels, target_bpcer: float) -> float:
    """live 중 target_bpcer(%) 만 spoof 로 판정되는 임계값 (score >= thr 이면 spoof)."""
    live = np.asarray(scores, dtype=float)[np.asarray(labels) == 0]
    return float(np.quantile(live, 1.0 - target_bpcer / 100.0))


def compute(scores, labels, video_ids=None, threshold: float | None = None,
            fixed_threshold: float = 0.0, op_bpcer: float = 1.0,
            op_threshold: float | None = None) -> dict:
    """전체 지표. threshold=None 이면 테스트셋 EER 임계값을 쓴다(관행).

    ⚠ scores 는 **spoof margin(log-odds)** 이다 — 확률이 아니다. 로짓 차이가 ~17 을
      넘으면 softmax 확률이 float32 로 정확히 1.0 이 되어 대량 동점이 생기고, 그 동점
      무리에 live 가 1% 넘게 섞이면 TPR@FPR=1% 를 정의할 수 없어 0 이 나온다(실측:
      InstructFLIP 1 epoch 에 val 의 42% 가 1.0). margin 은 확률과 순서가 같고
      포화하지 않는다. fixed_threshold=0.0 은 확률 0.5 와 같다.

    val 에서 구한 임계값을 넘기면 누수 없는 수치가 된다 — 논문 대비 비교가
    아니라 실제 성능을 알고 싶을 때는 그쪽을 쓸 것.

    운영점(op) 지표 — APCER_op / BPCER_op / ACER_op:
      공식 벤치마크는 임계값 0.5 고정이지만, 이 모델은 live 에 극단적으로 확신해
      (margin 중앙값 ≈ -83) 0.5 가 live 분포에서 한참 떨어져 있다. 그 결과 live 는
      거의 안 막고(BPCER 0.05) 경계에 걸친 spoof 를 대량 통과시킨다(APCER 9.15).
      배포라면 "BPCER 를 op_bpcer% 로 묶는" 임계값을 **val 에서** 정해 쓴다.
      - op_threshold=None : 이 데이터 자체에서 정한다 (val 에서 호출할 때)
      - op_threshold=값   : 그 값을 쓴다 (test 에 val 의 op_threshold 를 넘길 때 — 누수 없음)
      ⚠ test 에 None 으로 부르면 test 로 임계값을 고르는 셈이라 낙관 편향이 생긴다.
    """
    if video_ids is not None:
        scores, labels = aggregate_by_video(scores, labels, video_ids)
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)

    if len(np.unique(labels)) < 2:
        return {k: float("nan") for k in (*DG_KEYS, *CELEBA_KEYS, *OP_KEYS)} | {
            "threshold": float("nan"), "op_threshold": float("nan"), "n": len(labels)}

    eer, eer_thr = eer_threshold(scores, labels)
    thr = eer_thr if threshold is None else float(threshold)

    rec = recall_at_fpr(scores, labels)
    # 고정 임계값(margin 0 = 확률 0.5) — CelebA-Spoof 공식 축.
    # APCER = 공격(spoof)을 live 로 놓친 비율   = 여기 정의의 FRR
    # BPCER = 진짜(live)를 spoof 로 막은 비율   = 여기 정의의 FAR
    far05, frr05 = _far_frr(scores, labels, fixed_threshold)
    apcer, bpcer = frr05, far05

    op_thr = (threshold_at_bpcer(scores, labels, op_bpcer)
              if op_threshold is None else float(op_threshold))
    far_op, frr_op = _far_frr(scores, labels, op_thr)

    return {
        # DG 계열
        "HTER": hter(scores, labels, thr) * 100,
        "AUC": roc_auc_score(labels, scores) * 100,
        "TPR@FPR1": rec[0.01] * 100,
        "EER": eer * 100,
        # CelebA-Spoof 공식 계열 (고정 임계값 0.5)
        "APCER": apcer * 100,
        "BPCER": bpcer * 100,
        "ACER": (apcer + bpcer) / 2 * 100,
        "R@F1": rec[0.01] * 100,
        "R@F0.5": rec[0.005] * 100,
        "R@F0.1": rec[0.001] * 100,
        # 운영점 (val 에서 BPCER op_bpcer% 로 정한 임계값)
        "APCER_op": frr_op * 100,
        "BPCER_op": far_op * 100,
        "ACER_op": (frr_op + far_op) / 2 * 100,
        "op_threshold": op_thr,
        "op_bpcer_target": float(op_bpcer),
        "threshold": thr,
        "n": int(len(labels)),
    }


def format_row(name: str, m: dict) -> str:
    return (f"{name:16s} HTER {m['HTER']:6.2f} | AUC {m['AUC']:6.2f} | "
            f"T@F1 {m['TPR@FPR1']:6.2f} | ACER {m['ACER']:6.2f} "
            f"(APCER {m['APCER']:5.2f} / BPCER {m['BPCER']:5.2f}) | "
            f"운영점 ACER {m['ACER_op']:5.2f} (APCER {m['APCER_op']:5.2f} / "
            f"BPCER {m['BPCER_op']:5.2f}) | n={m['n']}")
