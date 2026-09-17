#!/usr/bin/env bash
# FA2D 추론 — 정지 이미지(격자 합본) + 동영상(H.264).
#
# 크롭은 검출 → 예측 → 예측 랜드마크로 다시 자르기(2 패스)다. 검출기를 타므로
# 평가 표(GT 크롭)와 나란히 놓지 말 것.
#
# 사용:  bash scripts/FA2D/infer.sh          (영상까지: VIDEO=clip.mp4 bash …)
set -e
cd "$(dirname "$0")/../.."

# ---- 여기만 바꾼다 ------------------------------------------------ #
CKPT=runs/fa2d/stu_final_v2_pointquery_vit-t16/best.pth
SOURCE=data/samples          # 이미지 | 디렉토리 | 목록.txt
GRID=3                       # 격자 합본 열 수 (0 = 안 만듦)
VIDEO="${VIDEO:-}"           # 비워 두면 영상 단계를 건너뛴다
# ------------------------------------------------------------------ #

PY=${PY:-python}

echo "===== ① 정지 이미지 추론 ====="
N_IMG=0
if [ -d "$SOURCE" ]; then
    N_IMG=$(find "$SOURCE" -maxdepth 1 -type f \
            \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' \) | wc -l)
elif [ -e "$SOURCE" ]; then
    N_IMG=1
fi
if [ "$N_IMG" -gt 0 ]; then
    $PY eval/FA2D/infer.py --ckpt "$CKPT" --source "$SOURCE" --grid "$GRID"
else
    echo "  $SOURCE 에 사진이 없습니다 — 건너뜁니다."
fi

echo
echo "===== ② 동영상 추론 ====="
if [ -n "$VIDEO" ]; then
    $PY eval/FA2D/infer.py --ckpt "$CKPT" --source "$VIDEO" --max-faces 4 --drop-empty
else
    echo "  VIDEO 미지정 — 건너뜁니다.  VIDEO=<경로> bash scripts/FA2D/infer.sh"
fi
