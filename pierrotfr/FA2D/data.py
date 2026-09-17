"""FA2D 평가 데이터 — 원본 배포 형식을 직접 읽고, 학습과 **같은 규약**으로 자른다.

학습 저장소의 `data.py`(848줄)에서 평가·추론에 쓰이는 것만 옮겼다:

    FLIP_MAPPING            규약별 좌우 교환 표 — flip-TTA 가 쓴다
    read_wflw / read_lapa   WFLW 98점 · LaPa 106점 원본 리더
    read_300w_challenge     300W Challenge (평가 전용)
    face_bbox · head_bbox   크롭 박스 규약
    EvalDataset             증강 없는 단일 프레임 크롭

뺀 것: 300W-LP · COFW · 300W 표준 리더(학습 소스), 광도·가림·모션블러 증강, 합성 클립,
scheme 배치 샘플러, 어려운 표본 가중.

⚠ 크롭은 **GT 랜드마크로 잡는다** — 벤치마크 규약이 그렇다. 임의 사진에서는 GT 가 없으므로
  `infer.py` 가 검출 → 예측 → 예측 랜드마크로 다시 자르기(2 패스)로 같은 규약에 맞춘다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .geometric import GeoParams, apply_geometric

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# 랜드마크 bbox → 머리 전체 bbox 외삽 계수 (좌우, 위, 아래). HRFFA 의 임시값이다.
HEAD_EXPAND = (0.30, 0.90, 0.15)

# 규약별 좌우 반전 매핑 — 자기 대응 점은 적지 않는다.
FLIP_MAPPING = {
    "ibug68": [[0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11], [6, 10], [7, 9],
               [17, 26], [18, 25], [19, 24], [20, 23], [21, 22],
               [31, 35], [32, 34],
               [36, 45], [37, 44], [38, 43], [39, 42], [40, 47], [41, 46],
               [48, 54], [49, 53], [50, 52], [55, 59], [56, 58],
               [60, 64], [61, 63], [65, 67]],
    "wflw98": ([[i, 32 - i] for i in range(16)] +
               [[33, 46], [34, 45], [35, 44], [36, 43], [37, 42], [38, 50], [39, 49],
                [40, 48], [41, 47]] +
               [[60, 72], [61, 71], [62, 70], [63, 69], [64, 68], [65, 75], [66, 74],
                [67, 73]] + [[96, 97]] +
               [[55, 59], [56, 58]] +
               [[76, 82], [77, 81], [78, 80], [87, 83], [86, 84],
                [88, 92], [89, 91], [95, 93]]),
    # LaPa 106점 — 공식 문서에 flip 표가 없어 **데이터에서 유도하고 검증했다**.
    #   유도: 정면 샘플에서 대칭축을 스칼라 최적화 → Hungarian 최적 할당 →
    #         상호 합의(i→j 이고 j→i)만 채택. involution 성립, 합의율 평균 0.99.
    #   검증: 보류 split(val 500장)에서 미러+치환 잔차가 매핑 없을 때의 1/3.4,
    #         정면 10% 구간 잔차 0.031(얼굴 폭 대비) — 나머지는 실제 비대칭과 yaw.
    #   자기대응 [16, 51~54, 60, 87, 93, 98, 102] 가 전부 중앙선 점(윤곽 중앙·콧대·
    #   코끝·입술 중앙)이라는 것이 매핑이 맞다는 강한 방증이다.
    # ⚠ 이 표가 틀리면 좌우가 조용히 뒤바뀐 채 학습된다 — 손실은 정상적으로 내려간다.
    "lapa106": [[0, 32],
                 [1, 31],
                 [2, 30],
                 [3, 29],
                 [4, 28],
                 [5, 27],
                 [6, 26],
                 [7, 25],
                 [8, 24],
                 [9, 23],
                 [10, 22],
                 [11, 21],
                 [12, 20],
                 [13, 19],
                 [14, 18],
                 [15, 17],
                 [33, 46],
                 [34, 45],
                 [35, 44],
                 [36, 43],
                 [37, 42],
                 [38, 50],
                 [39, 49],
                 [40, 48],
                 [41, 47],
                 [55, 65],
                 [56, 64],
                 [57, 63],
                 [58, 62],
                 [59, 61],
                 [66, 79],
                 [67, 78],
                 [68, 77],
                 [69, 76],
                 [70, 75],
                 [71, 82],
                 [72, 81],
                 [73, 80],
                 [74, 83],
                 [84, 90],
                 [85, 89],
                 [86, 88],
                 [91, 95],
                 [92, 94],
                 [96, 100],
                 [97, 99],
                 [101, 103],
                 [104, 105]],
}

SCHEME_N = {"ibug68": 68, "wflw98": 98, "cofw29": 29, "lapa106": 106}


def head_bbox_from_landmarks(points: np.ndarray, expand=HEAD_EXPAND):
    """랜드마크에서 머리 전체 bbox 를 외삽한다. 이미지 밖으로 나가도 자르지 않는다."""
    ex, et, eb = expand
    x1, y1 = float(points[:, 0].min()), float(points[:, 1].min())
    x2, y2 = float(points[:, 0].max()), float(points[:, 1].max())
    w, h = x2 - x1, y2 - y1
    return [x1 - ex * w, y1 - et * h, x2 + ex * w, y2 + eb * h]


def face_bbox(points: np.ndarray):
    """랜드마크를 감싸는 **정사각** 얼굴 박스(원본 좌표). pad 는 GeoParams 가 더한다."""
    x1, y1 = float(points[:, 0].min()), float(points[:, 1].min())
    x2, y2 = float(points[:, 0].max()), float(points[:, 1].max())
    cx, cy, side = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) / 2
    return [cx - side, cy - side, cx + side, cy + side]


def crop_box(points: np.ndarray, crop_mode: str):
    """체크포인트의 `crop_mode` 에 맞는 박스. 학습과 다르면 모델이 못 본 크기의 얼굴이 들어온다."""
    if crop_mode == "face":
        return face_bbox(points)
    if crop_mode == "head":
        return head_bbox_from_landmarks(points)
    raise ValueError(f"crop_mode={crop_mode!r} — face | head")


def to_tensor(bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 → RGB → ImageNet 정규화 CHW. ⚠ 채널 순서를 바꾸지 않으면 조용히 틀린다."""
    x = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    m, s = np.array(IMAGENET_MEAN, np.float32), np.array(IMAGENET_STD, np.float32)
    return torch.from_numpy(((x - m) / s).transpose(2, 0, 1).copy())


@dataclass
class Record:
    image_path: str
    points: np.ndarray          # (N,2) 원본 해상도 절대 픽셀
    visibility: list            # (N,) 2=보임 1=가림 0=화면밖 -1=불명
    head_bbox: list
    scheme: str


def read_wflw(root: Path, split: str = "train", expand=HEAD_EXPAND):
    """WFLW 원본 배포 형식(196 좌표 + rect 4 + 속성 6 + 경로)을 직접 읽는다."""
    root = Path(root)
    ann = (root / "WFLW_annotations" / "list_98pt_rect_attr_train_test" /
           f"list_98pt_rect_attr_{split}.txt")
    img_root = root / "WFLW_images"
    recs = []
    with open(ann, encoding="utf-8") as f:
        for line in f:
            t = line.split()
            if len(t) < 207:
                continue
            pts = np.asarray(t[:196], dtype=np.float64).reshape(-1, 2)
            recs.append(Record(str(img_root / t[-1]), pts, [-1] * 98,
                               head_bbox_from_landmarks(pts, expand), "wflw98"))
    if not recs:
        raise FileNotFoundError(f"WFLW 레코드를 못 읽었다: {ann}")
    return recs


def _read_pts(path: Path) -> np.ndarray:
    """iBUG `.pts` 포맷 → (N,2).

        version: 1
        n_points: 68
        {
        x1 y1
        ...
        }
    """
    lines = path.read_text().splitlines()
    try:
        i = lines.index("{")
        j = lines.index("}")
    except ValueError:
        raise ValueError(f"pts 포맷이 아니다: {path}")
    pts = [tuple(map(float, l.split())) for l in lines[i + 1:j] if l.strip()]
    return np.asarray(pts, dtype=np.float64)


def read_300w_challenge(root: Path, split: str = "all", expand=HEAD_EXPAND):
    """300W Challenge (Indoor/Outdoor) — **평가 전용 청정 셋**.

    ⚠ 표준 300W 학습셋(AFW/HELEN/LFPW 3,148장)이 아니다. Indoor 300 +
      Outdoor 300 = 600장의 별도 챌린지 셋이고, train/test 구분이 없다.

    학습에 넣지 않는다. 우리가 **한 번도 학습하지 않은 유일한 평가 축**이므로
    여기에 넣으면 그 값어치가 사라진다(HRFFA 가 테스트 split 을 학습에 넣어
    자기 표를 벤치마크가 아니게 만든 것과 같은 실수가 된다).
    """
    root = Path(root)
    base = root / "300w_extracted" / "300W"
    if not base.is_dir():                       # zip 을 다른 깊이로 푼 경우 탐색
        cands = list(root.rglob("01_Indoor"))
        if not cands:
            raise FileNotFoundError(f"300W Challenge 를 찾을 수 없다: {root}")
        base = cands[0].parent
    dirs = {"indoor": ["01_Indoor"], "outdoor": ["02_Outdoor"],
            "all": ["01_Indoor", "02_Outdoor"]}
    if split not in dirs:
        raise ValueError(f"split={split!r} — indoor | outdoor | all")

    recs = []
    for d in dirs[split]:
        for img in sorted((base / d).glob("*.png")):
            pts_fp = img.with_suffix(".pts")
            if not pts_fp.exists():
                continue
            pts = _read_pts(pts_fp)
            if len(pts) != 68:
                continue
            recs.append(Record(str(img), pts, [-1] * 68,
                               head_bbox_from_landmarks(pts, expand), "ibug68"))
    if not recs:
        raise FileNotFoundError(f"300W Challenge 레코드를 못 읽었다: {base}")
    return recs


def read_lapa(root: Path, split: str = "train", expand=HEAD_EXPAND):
    """LaPa 106점. `<split>/images/<id>.jpg` + `<split>/landmarks/<id>.txt`.

    txt 첫 줄이 점 개수(106), 이후 "x y" 가 이어진다. 원본 해상도 절대 픽셀이라
    head crop + 호모그래피 워프에 그대로 쓸 수 있다.

    규모: train 18,168 / val 2,000 / test 2,000.
    WFLW(7,500) 의 2.4 배이고, **다중 규약 단일 모델을 실제로 검증할 첫 소스**다.
    """
    root = Path(root)
    base = root / "LaPa" / split if (root / "LaPa").is_dir() else root / split
    img_dir, lmk_dir = base / "images", base / "landmarks"
    if not lmk_dir.is_dir():
        raise FileNotFoundError(f"LaPa 를 찾을 수 없다: {lmk_dir}")

    recs = []
    for lmk in sorted(lmk_dir.glob("*.txt")):
        tok = lmk.read_text().split()
        n = int(tok[0])
        if n != 106:
            continue
        pts = np.asarray(tok[1:1 + 2 * n], dtype=np.float64).reshape(n, 2)
        img = img_dir / f"{lmk.stem}.jpg"
        if not img.exists():
            img = img_dir / f"{lmk.stem}.png"
            if not img.exists():
                continue
        recs.append(Record(str(img), pts, [-1] * n,
                           head_bbox_from_landmarks(pts, expand), "lapa106"))
    if not recs:
        raise FileNotFoundError(f"LaPa 레코드를 못 읽었다: {base}")
    return recs


SOURCES = {"wflw": read_wflw, "lapa": read_lapa, "300w_challenge": read_300w_challenge}


class EvalDataset(Dataset):
    """레코드 → 증강 없는 크롭 한 장. (image (3,S,S), points (N,2) 0~1, scheme)"""

    def __init__(self, records, out_size: int, pad: float, crop_mode: str = "face"):
        if not records:
            raise ValueError("빈 레코드 목록")
        self.records, self.out_size, self.pad, self.crop_mode = records, out_size, pad, crop_mode

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i: int):
        rec = self.records[i]
        img = cv2.imread(rec.image_path)
        if img is None:
            raise FileNotFoundError(f"이미지를 못 읽었다: {rec.image_path}")
        o = apply_geometric(img, rec.points, rec.visibility,
                            crop_box(rec.points, self.crop_mode),
                            GeoParams(out_size=self.out_size, pad=self.pad),
                            flip_mapping=FLIP_MAPPING.get(rec.scheme))
        return {"image": to_tensor(o["image"]),
                "points": torch.from_numpy(np.asarray(o["points"], np.float32) / self.out_size)}
