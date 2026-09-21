"""FAS 평가 데이터 — CelebA-Spoof CSV 및 폴더형 벤치마크 + 평가 변환.

CSV 스키마 (학습 저장소의 scripts/FAS/prepare_celeba_spoof.py 가 만든다):
    path,spoof_type,illumination,environment,is_fake,quality
    path 는 CSV 위치 기준 상대경로이며 **정렬된 crop** 을 가리킨다.

라벨 규약: is_fake 1 = spoof, 0 = live.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

# 정규화 상수는 백본에 맞춘다 — 체크포인트 config 의 norm 이 고른다
NORM = {
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "clip": ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    "none": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
}


class CenterFrac:
    """중앙에서 변 길이의 frac 만큼 자른다 (넓은 crop 에서 좁은 배율을 만든다)."""

    def __init__(self, frac: float = 1.0):
        self.frac = frac

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.frac >= 1.0:
            return img
        w, h = img.size
        cw, ch = int(round(w * self.frac)), int(round(h * self.frac))
        l, t = (w - cw) // 2, (h - ch) // 2
        return img.crop((l, t, l + cw, t + ch))


def eval_transform(size: int = 224, norm: str = "clip", center_frac: float = 1.0):
    """학습 저장소의 평가 변환과 같다 — (CenterFrac) → Resize → ToTensor → Normalize."""
    if norm not in NORM:
        raise ValueError(f"norm={norm!r} — 가능: {sorted(NORM)}")
    mean, std = NORM[norm]
    pre = [CenterFrac(center_frac)] if center_frac < 1.0 else []
    return T.Compose(pre + [T.Resize((size, size)), T.ToTensor(), T.Normalize(mean, std)])


def imread_rgb(path) -> Image.Image:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


class CelebASpoofDataset(Dataset):
    def __init__(self, csv_file: str, transform, name: str | None = None):
        csv_file = Path(csv_file)
        if not csv_file.exists():
            raise FileNotFoundError(f"라벨 CSV 가 없습니다: {csv_file}")
        df = pd.read_csv(csv_file, index_col=False)
        missing = {"path", "spoof_type", "is_fake"} - set(df.columns)
        if missing:
            raise ValueError(f"{csv_file}: 필수 컬럼 누락 {sorted(missing)}")
        self.df = df.reset_index(drop=True)
        self.root = csv_file.parent
        self.name = name or csv_file.stem
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict:
        row = self.df.iloc[i]
        return {"image": self.transform(imread_rgb(self.root / row["path"])),
                "label": int(row["is_fake"]), "spoof_type": int(row["spoof_type"]),
                "video_id": i, "path": str(row["path"])}


class FolderFASDataset(Dataset):
    """`<root>/real/*.jpg`, `<root>/fake/*.jpg`. 파일명이 `<video>_frame<N>.jpg` 면 video 단위 집계."""

    def __init__(self, root: str, transform, name: str | None = None):
        root = Path(root)
        files = sorted(p for p in root.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if not files:
            raise FileNotFoundError(f"이미지가 없습니다: {root}")
        self.files = files
        self.labels = [int(p.parent.name == "fake") for p in files]
        stems = [p.stem.split("_frame")[0] for p in files]
        uniq = {s: i for i, s in enumerate(sorted(set(stems)))}
        self.video_ids = [uniq[s] for s in stems]
        self.name = name or root.name
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> dict:
        return {"image": self.transform(imread_rgb(self.files[i])), "label": self.labels[i],
                "spoof_type": -1, "video_id": self.video_ids[i], "path": str(self.files[i])}


def collate(samples: list[dict]) -> dict:
    return {"image": torch.stack([s["image"] for s in samples]),
            "label": torch.tensor([s["label"] for s in samples]),
            "spoof_type": torch.tensor([s["spoof_type"] for s in samples]),
            "video_id": torch.tensor([s["video_id"] for s in samples]),
            "path": [s["path"] for s in samples]}
