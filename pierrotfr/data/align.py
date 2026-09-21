"""얼굴 정렬 — MTCNN 5점 랜드마크 기반 유사변환.

**모든 얼굴 태스크가 이 모듈을 쓴다.** 태스크마다 정렬 규약이 다르면 실험 간
비교가 성립하지 않으므로 태스크 밖(pierrotfr/data/)에 둔다.

규약은 InstructFLIP 원저자가 쓴 mxnet MTCNN(insightface 계열)의
extract_image_chips 와 동일하다 — 평균 얼굴 좌표, padding 비율, to_center
위치까지 그대로다. 원본과 수치적으로 일치함은 학습 저장소의 tests/FAS/test_align.py 가 검증한다.

원본은 mxnet 에 의존하지만 mxnet 은 유지보수가 끝나 최신 CUDA 에서 설치가
어렵다. 검출기만 facenet-pytorch 의 MTCNN(동일 P/R/O-net 계열)으로 갈아끼우고
정렬 수학은 그대로 옮겼다.

정렬 파이프라인:
    이미지 → MTCNN 5점(양 눈·코·입 양끝) → 유사변환 추정 → warpAffine → 224²
    검출 실패 시 → 데이터셋 동봉 바운딩박스로 crop (폴백)
"""
from __future__ import annotations

import math

import cv2
import numpy as np

DESIRED_SIZE = 224
PADDING = 0.37          # 원본 crop_face 의 ratio (enable_wilder=False)

# 원본 mtcnn_detector.py 의 평균 얼굴 좌표.
# 순서: 왼눈, 오른눈, 코, 왼입꼬리, 오른입꼬리 (facenet-pytorch 와 동일)
MEAN_FACE_X = [0.224152, 0.75610125, 0.490127, 0.254149, 0.726104]
MEAN_FACE_Y = [0.2119465, 0.2119465, 0.628106, 0.780233, 0.780233]


def target_points(desired_size: int = DESIRED_SIZE,
                  padding: float = PADDING) -> np.ndarray:
    """정렬 목표 좌표 (5,2). padding 이 클수록 얼굴이 작게(주변이 넓게) 잡힌다."""
    return np.array(
        [
            [
                (padding + MEAN_FACE_X[i]) / (2 * padding + 1) * desired_size,
                (padding + MEAN_FACE_Y[i]) / (2 * padding + 1) * desired_size,
            ]
            for i in range(5)
        ],
        dtype=np.float64,
    )


def similarity_transform(from_pts: np.ndarray, to_pts: np.ndarray) -> np.ndarray:
    """두 점집합 사이 유사변환 행렬 (2,2). 원본 find_tfrom_between_shapes 이식.

    Umeyama 와 같은 계열이지만 원본의 분기(det<0 일 때 s 부호 뒤집기)를 그대로
    따른다 — 미세하게 다른 구현을 쓰면 정렬이 어긋나 사전학습 통계가 흔들린다.
    """
    from_pts = np.asarray(from_pts, dtype=np.float64)
    to_pts = np.asarray(to_pts, dtype=np.float64)

    mean_from = from_pts.mean(axis=0)
    mean_to = to_pts.mean(axis=0)

    sigma_from = 0.0
    cov = np.zeros((2, 2), dtype=np.float64)
    for i in range(from_pts.shape[0]):
        d = from_pts[i] - mean_from
        sigma_from += float(d @ d)
        cov += np.outer(to_pts[i] - mean_to, from_pts[i] - mean_from)

    n = from_pts.shape[0]
    sigma_from /= n
    cov /= n

    s = np.eye(2)
    u, d, vt = np.linalg.svd(cov)
    if np.linalg.det(cov) < 0:
        if d[1] < d[0]:
            s[1, 1] = -1
        else:
            s[0, 0] = -1
    r = u @ s @ vt

    c = 1.0
    if sigma_from != 0:
        c = 1.0 / sigma_from * np.trace(np.diag(d) @ s)
    return c * r


def align_by_landmarks(img: np.ndarray, pts5: np.ndarray,
                       desired_size: int = DESIRED_SIZE,
                       padding: float = PADDING) -> np.ndarray:
    """5점 랜드마크로 정렬된 정사각 crop 을 만든다. pts5: (5,2) = [(x,y)]*5."""
    to_pts = target_points(desired_size, padding)
    from_pts = np.asarray(pts5, dtype=np.float64)

    tran_m = similarity_transform(from_pts, to_pts)
    probe = tran_m @ np.array([1.0, 0.0])
    scale = float(np.linalg.norm(probe))
    angle = 180.0 / math.pi * math.atan2(probe[1], probe[0])

    # 회전 중심은 양 눈의 중점. 목표 위치는 (0.5W, 0.4H) — 눈이 약간 위로 간다.
    from_center = ((from_pts[0][0] + from_pts[1][0]) / 2.0,
                   (from_pts[0][1] + from_pts[1][1]) / 2.0)
    to_center = (desired_size * 0.5, desired_size * 0.4)

    rot = cv2.getRotationMatrix2D(from_center, -angle, scale)
    rot[0][2] += to_center[0] - from_center[0]
    rot[1][2] += to_center[1] - from_center[1]
    return cv2.warpAffine(img, rot, (desired_size, desired_size))


def crop_by_box(img: np.ndarray, box_xywh, box_space: int = 224,
                desired_size: int = DESIRED_SIZE) -> np.ndarray | None:
    """바운딩박스 폴백 crop.

    CelebA-Spoof 의 *_BB.txt 는 224 정규화 공간 좌표라 원본 해상도로 환산이 필요하다.
    box_space=None 이면 이미 픽셀 좌표로 본다.
    """
    H, W = img.shape[:2]
    x, y, w, h = [float(v) for v in box_xywh]
    if box_space:
        x = W * x / box_space; y = H * y / box_space
        w = W * w / box_space; h = H * h / box_space
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(W, int(x + w)), min(H, int(y + h))
    if x1 <= x0 or y1 <= y0:
        return None
    return cv2.resize(img[y0:y1, x0:x1], (desired_size, desired_size))


class FaceAligner:
    """MTCNN 검출 + 랜드마크 정렬. 배치 처리는 detect_batch 를 쓴다.

    원본 설정과 맞춘 부분:
      - thresholds=[0.5, 0.5, 0.5]  (원본 MtcnnDetector threshold)
      - 세로 1280 초과 시 축소 후 검출 (원본 crop_face)
      - select_largest=True         (가장 큰 얼굴 하나)
    """

    def __init__(self, device: str = "cuda", desired_size: int = DESIRED_SIZE,
                 padding: float = PADDING, max_height: int = 1280):
        import torch
        from facenet_pytorch import MTCNN

        self.desired_size = desired_size
        self.padding = padding
        self.max_height = max_height
        self.mtcnn = MTCNN(keep_all=False, select_largest=True, post_process=False,
                           thresholds=[0.5, 0.5, 0.5], device=torch.device(device))

    def _shrink(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        if img.shape[0] <= self.max_height:
            return img, 1.0
        s = self.max_height / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s))), s

    def landmarks(self, img_bgr: np.ndarray) -> np.ndarray | None:
        """(5,2) 랜드마크 또는 None. 좌표는 **축소된** 이미지 기준이다."""
        work, _ = self._shrink(img_bgr)
        rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
        try:
            _, _, pts = self.mtcnn.detect(rgb, landmarks=True)
        except Exception:
            return None
        if pts is None or len(pts) == 0:
            return None
        return np.asarray(pts[0], dtype=np.float64)

    def __call__(self, img_bgr: np.ndarray, fallback_box=None) -> tuple[np.ndarray | None, str]:
        """(정렬된 224² BGR, 상태) 를 반환. 상태: 'mtcnn' | 'box' | 'fail'."""
        work, _ = self._shrink(img_bgr)
        pts = self.landmarks(img_bgr)
        if pts is not None:
            return align_by_landmarks(work, pts, self.desired_size, self.padding), "mtcnn"
        if fallback_box is not None:
            chip = crop_by_box(img_bgr, fallback_box, desired_size=self.desired_size)
            if chip is not None:
                return chip, "box"
        return None, "fail"
