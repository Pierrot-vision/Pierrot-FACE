#!/usr/bin/env bash
# FA2D 평가 — README 평가 표의 세 축 (① WFLW · ② LaPa 공통 38점 · ③ HRFFA 난이도 프로토콜).
#
# 인자를 받지 않는다. 대상은 아래 블록에서 고친다.
# 체크포인트에 config 가 들어 있어 백본 / 분기 / 크롭 규약은 자동 복원된다.
#
# ⚠ ③ 은 오래 걸린다 (교사 기준 수십 분). 빠른 확인은 AXES="wflw lapa".
#
# 사용:  bash scripts/FA2D/eval.sh
set -e
cd "$(dirname "$0")/../.."

# ---- 여기만 바꾼다 ------------------------------------------------ #
CKPTS=(runs/fa2d/full_v12_pointquery_dinov3-vitl16/best.pth
       runs/fa2d/stu_final_v2_pointquery_vit-t16/best.pth)
AXES="wflw lapa difficulty"   # 잴 축
TTA=0                          # 1 = 좌우반전 TTA 행을 추가로 (기본 지표는 안 덮는다)
# ------------------------------------------------------------------ #

ARGS=(--ckpt "${CKPTS[@]}" --only $AXES)
[ "$TTA" = "1" ] && ARGS+=(--tta)
python eval/FA2D/evaluate.py "${ARGS[@]}"
