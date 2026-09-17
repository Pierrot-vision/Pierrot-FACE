"""2D Face Alignment (2D 얼굴 정렬) — **추론 경로만**.

    infer.py      체크포인트 로드 · 크롭 배치 추론(flip-TTA) · FA2D 클래스(사진 → 랜드마크)
    benchmark.py  ① WFLW test · ② LaPa 공통 38점 · ③ HRFFA 난이도 프로토콜
    data.py       원본 리더(WFLW · LaPa · 300W Challenge) · 크롭 규약 · 좌우 교환 표
    geometric.py  사영변환 크롭 렌더링 (roll · 카메라 pitch/yaw 를 GT 와 함께)
    metrics.py    io-NME · head-NME · FR10 · AUC10
    render.py     랜드마크 · 윤곽선 그리기
    models/       점 쿼리 디코더 + 백본(DINOv3 ViT-L · ViT-T · 경량 CNN)

학습 저장소(Pierrot_FR_Lab)에서 **빠진 것**: engine.py(학습 루프 · 증류) · losses.py ·
trajectory.py(합성 클립) · depthwarp.py(극단 pitch 합성) · checkpoint.py(초기화) ·
config.py(학습 설정 검증) · 증강 · 배치 샘플러 · 지터 지표.
전부 학습에만 존재하고 추론 경로를 타지 않는다.

기반: HRFFA(PINTO0309, MIT) 의 점 쿼리 디코더 설계 + Peppa 의 stride 4 히트맵·오프셋.
"""
from .data import FLIP_MAPPING, SCHEME_N, EvalDataset, crop_box, face_bbox, to_tensor
from .infer import FA2D, ModelSpec, load_checkpoint, predict
from .models import MODELS, SCHEMES, build_model

__all__ = ["FA2D", "ModelSpec", "load_checkpoint", "predict",
           "EvalDataset", "crop_box", "face_bbox", "to_tensor", "FLIP_MAPPING", "SCHEME_N",
           "MODELS", "SCHEMES", "build_model"]
