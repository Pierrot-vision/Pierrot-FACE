#!/usr/bin/env bash
# FA3D 평가 — AFLW2000-3D / AFLW NME (논문 규약 = yaw 3구간 평균의 평균).
#
# 저자 배포 가중치를 **같은 코드로** 함께 재는 것이 이 스크립트의 요점이다.
# 논문 표의 숫자와 우리 숫자를 직접 비교하면 크롭 규약·NME 규약·GT 판본이 전부
# 달라 어긋난다 — 같은 하네스로 잰 값끼리만 비교가 성립한다.
#
# 인자를 받지 않는다. 대상은 아래 블록에서 고친다.
# 체크포인트에 config 가 들어 있어 백본 / 입력크기 / 테두리 전처리는 자동 복원된다.
#
# 사용:  bash scripts/FA3D/eval.sh
set -e
cd "$(dirname "$0")/../.."

# ---- 여기만 바꾼다 ------------------------------------------------ #
CKPTS=(runs/fa3d/meta_paper_aug_synergy_meta_mobilenet_v1/best.pth)
DEPLOYED="mb1"      # 함께 잴 배포 가중치 ("" = 안 잼, "mb1 mb05" 도 가능)
AFLW=1              # 1 = AFLW(21,080장)도 평가 — 실사진 일반화를 보는 축이다
GT=ori              # ori | reannotated | both
# 좌우반전 TTA 행을 추가로 낼지. ⚠ 기본 지표를 덮지 않는다 — 표의 기준은 언제나
# TTA 없는 값이다. 대가는 추론 2배.
TTA=1
# ------------------------------------------------------------------ #

ARGS=(--ckpt "${CKPTS[@]}" --gt "$GT")
[ -n "$DEPLOYED" ] && ARGS+=(--deployed $DEPLOYED)
[ "$AFLW" = "1" ] && ARGS+=(--aflw)
[ "$TTA" = "1" ] && ARGS+=(--tta)

python eval/FA3D/evaluate.py "${ARGS[@]}"
