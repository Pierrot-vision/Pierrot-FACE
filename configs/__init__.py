"""추론·평가 설정 — 학습 config 는 여기에 없다.

    configs/paths.py       데이터/가중치/산출물 루트 (paths.local.env · 환경변수)
    configs/eval_fa3d.py   FA3D 평가셋·BFM 경로 — 체크포인트가 모르는 것만

모델 설정(백본 · 입력크기 · lrr · 테두리 전처리)은 **체크포인트가 통째로 들고 있어**
`pierrotfr/FA3D/infer.py:load_checkpoint()` 가 복원한다. 여기서 다시 적지 않는다 —
두 곳에 적으면 반드시 어긋난다.
"""
