"""서버 고유 경로의 단일 소스 — 데이터셋 / 가중치 / 산출물 루트.

    FA3D_DATA_ROOT   3DDFA 데이터 루트 (test.data/ test.configs/ bfm/ train.configs/)
    FA2D_DATA_ROOT   2DDFA 데이터 루트 (raw/WFLW · raw/LaPa)
    FAS_DATA_ROOT    Spoof 데이터 루트 (processed/ca/{val,test}.csv + 정렬 crop)
    WEIGHT_ROOT      오피셜 배포 가중치 루트 (대조용 — 3DDFA_V2 의 mb1/mb05)
    CKPT_ROOT        학습 산출물 루트 — 여기서 best.pth 를 읽는다 (학습 저장소의 runs/)
    OUTPUT_ROOT      추론 시각화 출력 루트 (저장소 안 outputs/)

경로는 서버마다 다르므로 코드에 박지 않는다.
우선순위: ① 환경변수  ② 저장소 루트의 paths.local.env  ③ 상대경로 기본값

②를 한 번 만들어 두면 매번 export 하지 않아도 된다. 이 파일은 서버 고유값이라
저장소에 커밋하지 않는다 (paths.local.env.example 을 복사해 쓸 것).

⚠ 학습 저장소(Pierrot_FR_Lab)의 `configs/paths.py` 와 **같은 환경변수 이름**을 쓴다.
  두 저장소가 같은 서버에 있으면 paths.local.env 를 그대로 복사해 쓸 수 있고,
  체크포인트가 가리키던 데이터와 평가 데이터가 어긋날 일이 없다.
"""
from __future__ import annotations

import os

# configs/ 안에 있으므로 저장소 루트는 한 단계 위다
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATHS_FILE = os.path.join(_REPO, "paths.local.env")


def _read_paths_file() -> dict:
    if not os.path.exists(_PATHS_FILE):
        return {}
    found = {}
    with open(_PATHS_FILE, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            value = value.strip().strip('"').strip("'")
            if value:
                found[key.strip()] = os.path.expanduser(value)
    return found


_PATHS = _read_paths_file()
_SOURCES: dict[str, str] = {}


def root(name: str, fallback: str) -> str:
    """환경변수 > paths.local.env > fallback 순으로 루트를 고른다."""
    if os.environ.get(name):
        _SOURCES[name] = "env"
        return os.environ[name]
    if _PATHS.get(name):
        _SOURCES[name] = "paths.local.env"
        return _PATHS[name]
    _SOURCES[name] = "default"
    return fallback


REPO_ROOT = _REPO
CKPT_ROOT = root("PIERROTFR_CKPT_ROOT", os.path.join(_REPO, "runs"))
WEIGHT_ROOT = root("PIERROTFR_WEIGHT_ROOT", os.path.join(_REPO, "checkpoints"))

# ---- FA3D 데이터 루트 ----
# 학습셋과 평가셋이 한 루트 아래 있다. 추론만 한다면 test.data / test.configs / bfm
# 셋이면 충분하고, train.configs 는 형상 정확도(shape_accuracy.py)처럼 62-d GT 가
# 필요한 분석에서만 쓴다 — 없으면 그 스크립트만 안내를 내고 멈춘다.
FA3D_DATA_ROOT = root("PIERROTFR_FA3D_DATA_ROOT", os.path.join(_REPO, "data", "3DDFA"))
FA3D_TEST_DIR = os.path.join(FA3D_DATA_ROOT, "test.data")        # AFLW2000-3D / AFLW 크롭
FA3D_TEST_CFG = os.path.join(FA3D_DATA_ROOT, "test.configs")     # GT 랜드마크 / pose / roi
FA3D_BFM_DIR = os.path.join(FA3D_DATA_ROOT, "bfm")               # BFM basis + 정규화 통계
FA3D_TRAIN_DIR = os.path.join(FA3D_DATA_ROOT, "train_aug_120x120")   # 300W-LP 크롭
FA3D_TRAIN_CFG = os.path.join(FA3D_DATA_ROOT, "train.configs")       # 62-d GT + 리스트

# ---- FA2D 데이터 루트 ----
# 평가만 한다면 WFLW(①·③) 와 LaPa(②) 두 개면 된다.
FA2D_DATA_ROOT = root("PIERROTFR_FA2D_DATA_ROOT", os.path.join(_REPO, "data", "2DDFA"))
FA2D_WFLW_DIR = os.path.join(FA2D_DATA_ROOT, "raw", "WFLW")      # 원본 이미지 + 98점 주석
FA2D_LAPA_DIR = os.path.join(FA2D_DATA_ROOT, "raw", "LaPa")      # LaPa 106점
FA2D_300W_DIR = os.path.join(FA2D_DATA_ROOT, "raw", "300W")      # 300W Challenge (평가 전용)
# DINOv3 ViT-L/16 config — 교사 백본의 **구조 정의**만 읽는다(가중치는 체크포인트에 있다).
# ⚠ DINOv3 는 재배포 금지 라이선스라 이 저장소에 넣지 않는다.
FA2D_DINOV3_DIR = root("PIERROTFR_FA2D_DINOV3_DIR",
                       os.path.join(FA2D_DATA_ROOT, "ckpts", "dinov3-vitl16"))

# ---- FAS 데이터 루트 ----
# CelebA-Spoof 를 MTCNN 5점으로 정렬한 224² crop 과 라벨 CSV. 평가에는 val.csv(임계값) ·
# test.csv 두 개면 된다.
FAS_DATA_ROOT = root("PIERROTFR_FAS_DATA_ROOT", os.path.join(_REPO, "data", "Spoof"))
FAS_CA_DIR = os.path.join(FAS_DATA_ROOT, "processed", "ca")

# 3DDFA_V2 오피셜 저장소 — **선택**이다.
#   · 배포 가중치(mb1/mb05) 대조: WEIGHT_ROOT 에 .pth 만 두면 이 경로 없이도 된다
#   · FaceBoxes 검출기: 이 경로가 있으면 데모가 오피셜과 **같은 검출기**를 쓴다
#     (없으면 pierrotfr/data/detect.py 가 MTCNN → Haar 순으로 폴백한다)
V2_ROOT = root("PIERROTFR_3DDFA_V2_ROOT", "")

# 추론 시각화 산출물 — 학습 산출물(runs/)과 나란한 트리를 쓴다.
#     outputs/fa3d/<런 이름>/    demo.mp4 · grid.jpg · pred_3d.jpg …
#     outputs/fa2d/<런 이름>/
#     outputs/fas/<런 이름>/
OUTPUT_ROOT = os.path.join(_REPO, "outputs")


def work_dir(task: str, tag: str) -> str:
    """학습 저장소와 **같은 규칙** — runs/<태스크>/<런 이름>/best.pth 를 읽는다."""
    return os.path.join(CKPT_ROOT, task, tag)


def output_dir(tag: str, task: str = "fa3d") -> str:
    return os.path.join(OUTPUT_ROOT, task, tag)


def describe() -> str:
    """어떤 경로가 어디서 왔는지 — 로그 첫 줄에 남겨두면 나중에 추적이 된다."""
    return " | ".join(
        f"{k}={v} ({_SOURCES.get(k, '?')})"
        for k, v in (("PIERROTFR_FA3D_DATA_ROOT", FA3D_DATA_ROOT),
                     ("PIERROTFR_FA2D_DATA_ROOT", FA2D_DATA_ROOT),
                     ("PIERROTFR_FAS_DATA_ROOT", FAS_DATA_ROOT),
                     ("PIERROTFR_CKPT_ROOT", CKPT_ROOT),
                     ("PIERROTFR_WEIGHT_ROOT", WEIGHT_ROOT))
    )


def env_hint() -> str:
    """경로 오류 메시지 뒤에 붙일 안내문 (전부 해결돼 있으면 빈 문자열)."""
    unset = [n for n in ("PIERROTFR_FA3D_DATA_ROOT", "PIERROTFR_FA2D_DATA_ROOT",
                         "PIERROTFR_FAS_DATA_ROOT", "PIERROTFR_CKPT_ROOT")
             if _SOURCES.get(n) == "default"]
    if not unset:
        return ""
    return "\n".join([
        f"\n  ⚠ {' / '.join(unset)} 미설정 — 기본 상대경로를 쓰고 있습니다.",
        "    저장소 루트에 paths.local.env 를 만들면 매번 export 하지 않아도 됩니다:",
        "      cp paths.local.env.example paths.local.env   # 그리고 값을 채운다",
        "    (환경변수를 export 하면 그쪽이 항상 우선합니다)",
    ])
