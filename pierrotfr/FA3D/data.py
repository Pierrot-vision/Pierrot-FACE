"""추론 전처리 — 크롭된 120x120 을 모델 입력 텐서로.

학습 저장소의 `data.py` 는 300W-LP 데이터셋(463줄, 증강·svs·identity 그룹)이었다.
여기 남는 것은 **모델이 실제로 보는 것**뿐이다:

    (x − 127.5) / 128  ·  테두리 0 처리  ·  BGR 채널 순서 그대로

⚠ 채널은 **BGR** 이다 (`cv2.imread` 순서). 3DDFA 계열 전체가 그렇게 학습됐다.
  RGB 로 넘기면 오류 없이 조금씩 틀린 얼굴이 나온다.

⚠ 정규화는 ImageNet 통계가 아니다. (x−127.5)/128 이라 검정(0)이 −0.996 이 된다 —
  테두리·가림을 0 으로 채우는 게 '평균값'이 아니라 '물리적 검정'인 이유다.
"""
from __future__ import annotations

import os.path as osp

import cv2
import numpy as np
import torch
import torch.utils.data as data

# 3DDFA 계열 공통 전처리: (x − 127.5) / 128. ImageNet 통계가 아니다.
IMG_MEAN, IMG_STD = 127.5, 128.0


def zero_border(img: np.ndarray, border: int) -> np.ndarray:
    """이미지 테두리 `border` px 를 0(검정)으로. **학습·평가 공통 전처리다.**

    ⚠⚠ 이건 증강이 아니다. SynergyNet 은 학습과 평가 **양쪽에** 똑같이 건다
      (`main_train.py:204` · `benchmark.py:116` 의 `CenterCrop(5, mode='test')`).
      한쪽에만 걸면 모델이 못 본 분포가 들어온다 — 실측으로 학습 손실은 오히려
      낮은데 AFLW2000-3D NME 가 3.79 → **10~13** 으로 무너졌다.

    그래서 이 값은 사람이 고르지 않는다. `load_checkpoint()` 가 체크포인트의
    `aug_border` 를 읽어 전처리에 그대로 건다.
    """
    if border <= 0:
        return img
    out = np.zeros_like(img)
    out[border:-border, border:-border] = img[border:-border, border:-border]
    return out


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """BGR uint8 HWC [0,255] -> float CHW 정규화 텐서. 배치 차원은 없다."""
    x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float()
    return x.sub_(IMG_MEAN).div_(IMG_STD)


def preprocess(img: np.ndarray, size: int = 120, border: int = 0) -> torch.Tensor:
    """임의 크기의 **얼굴 크롭** -> [1, 3, size, size] 모델 입력.

    크롭 자체는 `crop.py` 가 만든다 (roi_box 규약이 오피셜과 같아야 한다).
    """
    if img.shape[0] != size or img.shape[1] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    return to_tensor(zero_border(img, border))[None]


class CropTestDataset(data.Dataset):
    """평가용 — 사전 크롭된 120x120 이미지만 낸다 (GT 는 metrics 가 따로 읽는다).

    AFLW2000-3D / AFLW 평가셋은 3DDFA v1 이 배포한 **크롭 완료본**이다. 얼굴 검출을
    타지 않으므로 검출기 성능이 지표에 섞이지 않는다 — 저장된 roi_box 가 3DDFA_V2 의
    `parse_roi_box_from_landmark` 와 0.58px(0.2%) 차이라 크롭 규약도 일치한다.
    """

    def __init__(self, root: str, filelist: str, name: str = "test",
                 border: int = 0):
        for fp in (root, filelist):
            if not osp.exists(fp):
                raise SystemExit(f"[FA3D] 평가 데이터가 없습니다: {fp}")
        self.root, self.name = root, name
        # ⚠ 그 모델이 **학습 때 쓴 값**이어야 한다. 다르면 못 본 분포가 들어온다.
        self.border = int(border)
        self.lines = open(filelist, encoding="utf-8").read().strip().split("\n")

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, i: int):
        img = cv2.imread(osp.join(self.root, self.lines[i]), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"[FA3D] 이미지를 읽지 못했습니다: {self.lines[i]}")
        return to_tensor(zero_border(img, self.border))
