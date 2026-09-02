#!/usr/bin/env bash
# FA3D 추론 — 정지 이미지(격자 합본) + 동영상(H.264) + 예측 격자 + 3D 재렌더 + 속도.
#
# 학습 저장소의 그림 갱신 절차에서 **학습 곡선·실험 표 단계를 뺀** 것이다 —
# 둘 다 train.log 와 runs/*/history.json 을 파싱해야 그릴 수 있고, 이 저장소는
# 체크포인트만 읽는다.
#
# ⚠ 속도는 학습이 도는 GPU 에서 재면 경합 때문에 절대값을 믿을 수 없다.
#
# 사용:  bash scripts/FA3D/infer.sh          (영상까지: VIDEO=clip.mp4 bash …)
set -e
cd "$(dirname "$0")/../.."

# ---- 여기만 바꾼다 ------------------------------------------------ #
CKPT=runs/fa3d/meta_paper_aug_synergy_meta_mobilenet_v1/best.pth
SOURCE=data/samples          # 이미지 | 디렉토리 | 목록.txt
GRID=3                       # 격자 합본 열 수 (0 = 안 만듦)
# 동영상 — 비워 두면 영상 단계를 건너뛴다.
# ⚠ 공개 저장소에 올릴 것이라면 직접 촬영본이나 CC 라이선스 클립을 쓸 것.
VIDEO="${VIDEO:-}"
# 평가셋 그림 (0 = 건너뜀). 평가 데이터가 있어야 한다.
EVALSET_FIGS="${EVALSET_FIGS:-1}"
# ------------------------------------------------------------------ #

PY=${PY:-python}

echo "===== ① 정지 이미지 추론 ====="
# ⚠ 갓 clone 한 저장소의 data/samples/ 에는 README 밖에 없다 — 그때 `set -e` 로
#   스크립트 전체가 죽으면 나머지 단계를 못 본다. 사진이 실제로 있는지 먼저 센다.
N_IMG=0
if [ -d "$SOURCE" ]; then
    N_IMG=$(find "$SOURCE" -maxdepth 1 -type f \
            \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' \) \
            | wc -l)
elif [ -e "$SOURCE" ]; then
    N_IMG=1
fi
if [ "$N_IMG" -gt 0 ]; then
    $PY eval/FA3D/infer.py --ckpt "$CKPT" --source "$SOURCE" --grid "$GRID"
else
    echo "  $SOURCE 에 사진이 없습니다 — 건너뜁니다."
    echo "  얼굴이 든 아무 사진이나 넣으면 됩니다 (data/samples/README.md 참고)."
fi

echo
echo "===== ② 동영상 추론 ====="
if [ -n "$VIDEO" ]; then
    $PY eval/FA3D/infer.py --ckpt "$CKPT" --source "$VIDEO" --corner3d 0.24 \
        --max-faces 4 --drop-empty
else
    echo "  VIDEO 미지정 — 건너뜁니다.  VIDEO=<경로> bash scripts/FA3D/infer.sh"
fi

if [ "$EVALSET_FIGS" = "1" ]; then
    echo
    echo "===== ③ 평가셋 예측 격자 (2D) ====="
    $PY scripts/FA3D/pred_grid.py --ckpt "$CKPT" --mode spread
    $PY scripts/FA3D/pred_grid.py --ckpt "$CKPT" --mode worst

    echo
    echo "===== ④ 3D 밀집 복원 (새 시점 재렌더) ====="
    $PY scripts/FA3D/pred_3d.py --ckpt "$CKPT"
fi

echo
echo "===== ⑤ 백본 추론 속도 ====="
# ⚠ 논문의 판매 포인트가 CPU 실시간이라 CPU 배치1 지연시간이 핵심 지표다.
$PY scripts/FA3D/bench_backbone.py
