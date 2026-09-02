"""얼굴 크롭 규약 — 3DDFA_V2 `utils/functions.py` 이식.

**추론 정확도의 절반은 여기서 갈린다.** 모델은 120x120 크롭 하나만 보고, 그 크롭이
학습 때와 다른 규약으로 잡히면 같은 가중치가 다른 답을 낸다. 그래서 오피셜 저장소를
import 하지 않고 **여기로 옮겨 왔다** — 데모 하나 돌리자고 외부 저장소를 clone 하게
만들 이유가 없고, 옮겨 온 이상 규약이 눈에 보이는 곳에 있어야 한다.

세 함수뿐이다:

    crop_img                     roi_box 로 자른다 (프레임 밖은 검정으로 패딩)
    parse_roi_box_from_bbox      검출 박스 -> roi_box (아래로 0.14 이동 · 1.58 배)
    parse_roi_box_from_landmark  이전 프레임 랜드마크 -> 다음 roi_box (영상 추적)

⚠ `parse_roi_box_from_bbox` 의 상수 0.14 / 1.58 은 마법 숫자가 아니라 **학습 크롭
  분포**다. 검출 박스는 보통 턱을 자르고 이마를 남기므로 중심을 아래로 내리고
  (0.14) 넓게 잡는다(1.58). 바꾸면 모델이 못 본 크기의 얼굴이 들어온다.

⚠ 영상에서는 **검출 박스가 아니라 이전 프레임 랜드마크로** 다음 크롭을 잡는 것이
  3DDFA_V2 의 방식이다. 매 프레임 검출하면 박스가 튀고, 그 자체가 떨림이 된다.
"""
from __future__ import annotations

from math import sqrt

import numpy as np


def crop_img(img: np.ndarray, roi_box) -> np.ndarray:
    """roi_box(sx, sy, ex, ey)로 자른다. 프레임 밖으로 나간 부분은 0(검정)이다.

    잘라 낸 뒤 리사이즈하지 않는다 — 크기 맞춤은 `data.preprocess()` 가 한다.
    """
    h, w = img.shape[:2]
    sx, sy, ex, ey = [int(round(v)) for v in roi_box]
    dh, dw = ey - sy, ex - sx
    if dh <= 0 or dw <= 0:
        return np.zeros((0, 0, 3), dtype=img.dtype)
    res = (np.zeros((dh, dw, 3), dtype=img.dtype) if img.ndim == 3
           else np.zeros((dh, dw), dtype=img.dtype))

    if sx < 0:
        sx, dsx = 0, -sx
    else:
        dsx = 0
    if ex > w:
        ex, dex = w, dw - (ex - w)
    else:
        dex = dw
    if sy < 0:
        sy, dsy = 0, -sy
    else:
        dsy = 0
    if ey > h:
        ey, dey = h, dh - (ey - h)
    else:
        dey = dh

    if dey > dsy and dex > dsx:
        res[dsy:dey, dsx:dex] = img[sy:ey, sx:ex]
    return res


def parse_roi_box_from_bbox(bbox) -> list:
    """검출 박스(left, top, right, bottom) -> 정사각 roi_box."""
    left, top, right, bottom = bbox[:4]
    old_size = (right - left + bottom - top) / 2
    center_x = right - (right - left) / 2.0
    # 아래로 0.14 — 검출기는 이마를 남기고 턱을 자르는 쪽으로 치우쳐 있다
    center_y = bottom - (bottom - top) / 2.0 + old_size * 0.14
    size = int(old_size * 1.58)
    return [center_x - size / 2, center_y - size / 2,
            center_x - size / 2 + size, center_y - size / 2 + size]


def parse_roi_box_from_landmark(pts) -> list:
    """랜드마크 [2 이상, K] -> 다음 프레임의 정사각 roi_box.

    긴 변 기준 정사각으로 감싼 뒤 대각선 길이를 한 변으로 쓴다 — 얼굴이 돌아가도
    크롭 안에 들어오게 하는 여유다.
    """
    pts = np.asarray(pts)
    bbox = [pts[0].min(), pts[1].min(), pts[0].max(), pts[1].max()]
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    bbox = [center[0] - radius, center[1] - radius,
            center[0] + radius, center[1] + radius]

    llength = sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
    cx, cy = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
    return [cx - llength / 2, cy - llength / 2,
            cx - llength / 2 + llength, cy - llength / 2 + llength]


def to_original(pts: np.ndarray, roi_box, size: int = 120) -> np.ndarray:
    """크롭 좌표(size x size) -> 원본 이미지 좌표. pts [2 이상, K] (in-place 아님).

    ⚠ 3열(z)이 있으면 **건드리지 않는다.** 깊이는 크롭 스케일과 무관한 양이 아니지만,
      3DDFA 계열의 z 는 x·y 와 같은 단위로 쓰이지 않아 오피셜도 변환하지 않는다.
    """
    sx, sy, ex, ey = roi_box
    out = np.array(pts, dtype=np.float64, copy=True)
    out[0] = out[0] * (ex - sx) / size + sx
    out[1] = out[1] * (ey - sy) / size + sy
    return out
