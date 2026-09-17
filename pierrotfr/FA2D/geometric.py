"""크롭 렌더링 — 사영변환 하나로 자르고 돌리고 기울인다. 추론 경로만.

HRFFA(PINTO0309, MIT)의 `dataset/augment/geometric.py` 에서 Lab 이 이식한 것 중
**평가에 쓰이는 부분만** 옮겼다. 학습 증강 샘플러(`GeoPolicy`)는 뺐다.

평가가 이걸 쓰는 곳 둘:
  · ① WFLW · 데모 — 회전 없이 얼굴/머리 박스를 정사각으로 자른다
  · ③ 난이도 프로토콜 — roll 360° · 카메라 pitch/yaw 를 **GT 와 함께 엄밀히** 만든다

모든 변환을 3×3 사영변환 T 하나로 합성해 warp 를 1회만 돌린다. 랜드마크는 T 로 옮긴다.

  - Roll 회전(화면 내 θ): 이미지 2D 회전 = 카메라 z 축 회전. 점·자세 모두 엄밀.
  - 카메라 회전 워프(φ, ψ): 병진 없는 순수 카메라 회전은 장면 형상과 **무관하게**
      H = K @ R_cam @ K⁻¹ 로 표현된다. ⚠ K 는 f = focal_ratio × out_size 인 핀홀 가정이다.
  - 수평 반전: 점은 scheme 의 flip_mapping 으로 교환한다.

가시성: 변환 후 출력 크롭 밖으로 나간 점은 0(화면 밖)으로 갱신한다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


def rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


@dataclass
class GeoParams:
    """크롭 하나의 기하 파라미터(값이 이미 정해진 상태)."""
    out_size: int = 256
    pad: float = 0.15            # bbox 외곽 마진(변 길이 비)
    roll_deg: float = 0.0        # 화면 내 회전
    cam_pitch_deg: float = 0.0   # 카메라 부앙(+ 가 올려다보는 방향)
    cam_yaw_deg: float = 0.0
    scale: float = 1.0
    tx: float = 0.0              # 출력 크기 비의 평행이동
    ty: float = 0.0
    hflip: bool = False
    focal_ratio: float = 1.2


def crop_affine(box, p: GeoParams) -> np.ndarray:
    """bbox(+pad)를 out_size 정사각으로 보내는 상사변환(3×3)."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + 2 * p.pad)
    s = p.out_size / side * p.scale
    th = math.radians(p.roll_deg)
    cos_t, sin_t = math.cos(th), math.sin(th)
    half = p.out_size / 2
    T = np.array([[s * cos_t, -s * sin_t, 0.0],
                  [s * sin_t, s * cos_t, 0.0],
                  [0.0, 0.0, 1.0]])
    T[0, 2] = half + p.tx * p.out_size - (T[0, 0] * cx + T[0, 1] * cy)
    T[1, 2] = half + p.ty * p.out_size - (T[1, 0] * cx + T[1, 1] * cy)
    return T


def camera_homography(p: GeoParams):
    """순수 카메라 회전의 호모그래피 H(출력 크롭 좌표계)와 R_cam."""
    phi, psi = math.radians(p.cam_pitch_deg), math.radians(p.cam_yaw_deg)
    R_cam = ry(psi) @ rx(phi)
    f = p.focal_ratio * p.out_size
    c = p.out_size / 2
    K = np.array([[f, 0, c], [0, f, c], [0, 0, 1.0]])
    return K @ R_cam @ np.linalg.inv(K), R_cam


def flip_matrix(out_size: int) -> np.ndarray:
    return np.array([[-1.0, 0.0, out_size - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def apply_geometric(image, points, visibility, box, p: GeoParams,
                    flip_mapping=None) -> dict:
    """변환을 적용하고 이미지·점·가시성·변환행렬을 한 번에 돌려준다.

    `points` 가 None 이면 이미지만 자른다(데모 — GT 가 없다).
    """
    T = crop_affine(box, p)
    if abs(p.cam_pitch_deg) > 1e-9 or abs(p.cam_yaw_deg) > 1e-9:
        H, _ = camera_homography(p)
        T = H @ T
        # 카메라 회전은 f·tanφ 급의 평행이동 성분을 갖는다 — 박스 중심을 출력 중심으로 되돌린다
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        m = T @ np.array([cx, cy, 1.0])
        mx, my = m[0] / m[2], m[1] / m[2]
        half = p.out_size / 2
        T = np.array([[1.0, 0.0, half + p.tx * p.out_size - mx],
                      [0.0, 1.0, half + p.ty * p.out_size - my],
                      [0.0, 0.0, 1.0]]) @ T
    if p.hflip:
        T = flip_matrix(p.out_size) @ T

    out = cv2.warpPerspective(image, T, (p.out_size, p.out_size),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if points is None:
        return {"image": out, "points": None, "visibility": None, "transform": T}

    pts_h = np.concatenate([points, np.ones((len(points), 1))], axis=1) @ T.T
    pts = pts_h[:, :2] / pts_h[:, 2:3]
    vis = list(visibility)
    if p.hflip and flip_mapping:
        pts = pts.copy()
        for a, b in flip_mapping:
            pts[[a, b]] = pts[[b, a]]
            vis[a], vis[b] = vis[b], vis[a]
    vis = [0 if (x < 0 or y < 0 or x >= p.out_size or y >= p.out_size) else v
           for (x, y), v in zip(pts, vis)]
    return {"image": out, "points": pts, "visibility": vis, "transform": T}


def to_original(pts_crop: np.ndarray, T: np.ndarray) -> np.ndarray:
    """크롭 픽셀 좌표 (N,2) → 원본 이미지 좌표. 호모그래피의 역변환."""
    h = np.concatenate([pts_crop, np.ones((len(pts_crop), 1))], axis=1) @ np.linalg.inv(T).T
    return h[:, :2] / h[:, 2:3]
