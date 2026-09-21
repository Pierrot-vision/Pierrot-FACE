#!/usr/bin/env bash
# FAS 평가 — README 평가 표 (CelebA-Spoof test 67,170장 · 임계값 0.5 행 + val 기준 행).
#
# 인자를 받지 않는다. 대상은 아래 블록에서 고친다.
# 체크포인트에 config 가 들어 있어 백본 / 정규화 / 입력 크기 / 가공물(ca · ca_wide)이 자동 복원된다.
#
# 사용:  bash scripts/FAS/eval.sh
set -e
cd "$(dirname "$0")/../.."

# ---- 여기만 바꾼다 ------------------------------------------------ #
CKPTS=(runs/fas/clip_instructflip_clip-vit-b16/best.pth
       runs/fas/wide_q_aug_instructflip_clip-vit-b16/best.pth
       runs/fas/minifasnet_minifasnet/best.pth)
# ------------------------------------------------------------------ #

for c in "${CKPTS[@]}"; do
    python eval/FAS/evaluate.py --ckpt "$c" --save
done
