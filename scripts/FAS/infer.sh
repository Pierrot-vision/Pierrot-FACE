#!/usr/bin/env bash
# FAS 추론 — 사진 폴더를 MTCNN 5점 정렬 후 LIVE / SPOOF 판정 + cue 오버레이.
#
# 사용:  bash scripts/FAS/infer.sh
set -e
cd "$(dirname "$0")/../.."

# ---- 여기만 바꾼다 ------------------------------------------------ #
CKPT=runs/fas/clip_instructflip_clip-vit-b16/best.pth
SOURCE=data/samples          # 디렉토리
MARGIN=-33.3                 # val 에서 BPCER 1% 로 정한 운영점 (0 = 공식 규칙, 확률 0.5)
# ------------------------------------------------------------------ #

python eval/FAS/infer.py --ckpt "$CKPT" --dir "$SOURCE" --align --margin "$MARGIN" --save_cue
