"""3D Dense Face Alignment (3D 밀집 얼굴 정렬) — **추론 경로만**.

    bfm.py       3DMM 디코더 — 62-d 를 38,365 정점 / 68 랜드마크로 편다
    crop.py      얼굴 크롭 규약 (3DDFA_V2 이식) — 추론 정확도의 절반이 여기서 갈린다
    data.py      전처리 ((x−127.5)/128 · 테두리 0) + 사전 크롭 평가 데이터셋
    detect.py    얼굴 검출 (FaceBoxes → MTCNN → Haar 폴백) — 데모 전용
    infer.py     체크포인트 로드 · 배치 추론 · FA3D 클래스 (이미지 한 장 → 얼굴)
    render.py    점 스플랫 렌더러 — 새 시점 재렌더 · 메쉬 오버레이 · 68점 그리기
    metrics.py   AFLW2000-3D / AFLW NME (yaw 구간 규약)
    benchmark.py 평가셋 파이프라인 (예측 → 랜드마크 → NME)
    models/      백본 + build_model 레지스트리 (lrr · synergy 헤드 없이)

학습 저장소(Pierrot_FR_Lab)에서 **빠진 것**: losses.py(VDC·fWPDC·lrr) ·
engine.py(meta-joint) · svs.py(short-video-synthesis) · 학습 데이터셋과 증강 ·
models/synergy.py. 전부 학습에만 존재하고 `predict()` 를 타지 않는다.

구현: Towards Fast, Accurate and Stable 3D Dense Face Alignment (ECCV 2020).
저자 공개 저장소(3DDFA_V2)도 추론 전용이고, 학습부는 논문에서 직접 구현했다 —
그 기록은 LAB/FA3D/ 에 있다.
"""
from .bfm import BFM, ParamNorm, load_tri, parse_param
from .crop import (crop_img, parse_roi_box_from_bbox, parse_roi_box_from_landmark,
                   to_original)
from .data import CropTestDataset, preprocess, to_tensor, zero_border
from .infer import (FA3D, ModelSpec, load_checkpoint, load_deployed,
                    predict_filelist, predict_params, to_landmarks)
from .models import MODELS, build_model

__all__ = [
    "BFM", "ParamNorm", "parse_param", "load_tri",
    "crop_img", "parse_roi_box_from_bbox", "parse_roi_box_from_landmark", "to_original",
    "CropTestDataset", "preprocess", "to_tensor", "zero_border",
    "FA3D", "ModelSpec", "load_checkpoint", "load_deployed",
    "predict_params", "predict_filelist", "to_landmarks",
    "build_model", "MODELS",
]
