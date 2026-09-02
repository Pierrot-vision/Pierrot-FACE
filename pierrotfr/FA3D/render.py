"""3D 렌더러 — 예측한 얼굴을 눈으로 본다.

68점을 이미지에 찍는 것만으로는 **z 를 볼 수 없다.** 3DDFA_V2 가 2D 정렬과 다른
점이 정확히 그 z 인데도 그렇다. 그래서 같은 예측을 새 시점에서 다시 그린다 —
한 장의 2D 입력에서 나온 형상이 옆에서 봐도 얼굴이면 그 z 가 의미 있는 것이다.

⚠ 오피셜 Sim3DR 은 Cython 빌드가 필요해 쓰지 않는다. 대신 **점 스플랫 + 화가
  알고리즘**(먼 것부터 그리기)이다 — z-buffer 없이도 가려짐이 맞고, 정점이 38,365개라
  면을 안 채워도 빈틈이 보이지 않는다.

⚠ 학습 저장소에서는 이 코드가 `scripts/FA3D/pred_3d.py` 안에 있었고 다른 스크립트가
  그 스크립트를 import 했다. 여기서는 라이브러리로 내린다 — 정지 격자·영상 데모·
  identity 비교가 **같은 렌더러**를 써야 그림끼리 비교가 된다.
"""
from __future__ import annotations

import cv2
import numpy as np

# 좌상단 앞쪽에서 오는 빛 (Lambertian)
LIGHT = np.array([0.35, -0.55, -0.75], np.float32)

# 편차 컬러맵의 고정 스케일 — 사람·모델 간 색을 비교하려면 반드시 공통이어야 한다.
# 300W-LP GT 의 법선 방향 변위 표준편차(≈6,000)의 3σ 근처로 잡는다.
DEV_SCALE = 18000.0

# 68점 표준 연결 — 윤곽선으로 그리면 점만 찍는 것보다 자세·표정이 눈에 들어온다
CHAINS = [(range(0, 17), False), (range(17, 22), False), (range(22, 27), False),
          (range(27, 31), False), (range(31, 36), False),
          (range(36, 42), True), (range(42, 48), True),
          (range(48, 60), True), (range(60, 68), True)]

# 사람별 색 — 여러 명이 나올 때 누가 누구인지 구분되어야 한다 (BGR).
TRACK_COLORS = [(96, 226, 26),      # 초록
                (60, 170, 255),     # 주황
                (255, 170, 60),     # 하늘
                (200, 90, 255),     # 분홍
                (60, 240, 255)]     # 노랑


# ------------------------------------------------------------------ #
# 기하
# ------------------------------------------------------------------ #
def vertex_normals(v: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """면 법선을 정점으로 모아 평균 — Lambertian 음영에 쓴다. v [3,N], tri [3,T]."""
    a, b, c = v[:, tri[0]], v[:, tri[1]], v[:, tri[2]]
    fn = np.cross((b - a).T, (c - a).T).T                       # [3,T]
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n.T, tri[k], fn.T)
    n = n / (np.linalg.norm(n, axis=0, keepdims=True) + 1e-9)
    # ⚠ 삼각형 감김 방향을 신뢰하지 않는다. BFM 의 tri 가 어느 쪽으로 감겼는지에 따라
    #   법선이 통째로 안쪽을 향하고, 그러면 얼굴 전체가 새까맣게 렌더된다(실제로 그랬다).
    #   z 가 클수록 카메라에 가까우므로(코끝 z 131,937 > 귀 17,761), 앞면 법선의
    #   평균 z 는 양수여야 한다. 음수면 통째로 뒤집는다.
    if float(n[2].mean()) < 0:
        n = -n
    return n


def splat(v: np.ndarray, shade: np.ndarray, res: int, radius: int = 1,
          bg=(1.0, 1.0, 1.0), front: np.ndarray | None = None) -> np.ndarray:
    """화가 알고리즘 점 스플랫. v [3,N] (x,y 는 픽셀, z 는 클수록 가까움).

    `front` 는 뒷면 제거(backface culling) 마스크다. 점 스플랫은 면을 안 채우므로
    앞면 사이의 틈으로 뒤통수 정점이 새어 나와 얼룩이 생긴다 — 카메라를 등진 정점
    (법선 z < 0)을 아예 빼면 깨끗해진다.
    """
    img = np.tile(np.asarray(bg, np.float32), (res, res, 1))
    x = np.round(v[0]).astype(np.int32); y = np.round(v[1]).astype(np.int32)
    ok = (x >= radius) & (x < res - radius) & (y >= radius) & (y < res - radius)
    if front is not None:
        ok &= front
    x, y, z, s = x[ok], y[ok], v[2][ok], shade[:, ok]
    o = np.argsort(z)                                            # 먼 것부터 → 가까운 게 덮는다
    x, y, s = x[o], y[o], s[:, o]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            img[y + dy, x + dx] = s.T
    return np.clip(img, 0, 1)


# ------------------------------------------------------------------ #
# 새 시점 렌더 (matplotlib 그림용 — RGB float [0,1])
# ------------------------------------------------------------------ #
def render(shape: np.ndarray, tri: np.ndarray, yaw_deg: float, res: int = 260,
           tint=(0.98, 0.86, 0.76), dev: np.ndarray | None = None,
           exaggerate: float = 1.0) -> np.ndarray:
    """3D 형상을 주어진 yaw 로 돌려 렌더한다. shape [3,N] (모델 좌표).

    `dev` 를 주면 그 편차를 색으로 칠하고(빨강 = 바깥, 파랑 = 안쪽) `exaggerate` 배로
    부풀린다. **`dev` 는 identity(shape 40-d) 편차만** 넘겨야 한다 — 표정(exp 10-d)까지
    섞어 과장하면 벌어진 입이 세 배로 벌어져 얼굴이 기형처럼 보인다.

    ⚠ 왜 과장하나 — BFM 의 shape 40-d 는 평균 얼굴에 얹는 **작은 섭동**이라, 있는
      그대로 그리면 GT 조차 사람마다 같아 보인다. identity 를 눈으로 비교하려면
      그 섭동만 떼어 키워야 한다.
    """
    import matplotlib.pyplot as plt
    if dev is not None:
        shape = shape + (exaggerate - 1.0) * dev
    v = shape - shape.mean(1, keepdims=True)
    t = np.radians(yaw_deg)
    Ry = np.array([[np.cos(t), 0, np.sin(t)],
                   [0, 1, 0],
                   [-np.sin(t), 0, np.cos(t)]], dtype=np.float32)
    v = Ry @ v
    n = vertex_normals(v, tri)
    lam = np.clip(-(n.T @ LIGHT), 0, 1)                          # [N]
    if dev is None:
        shade = (np.asarray(tint, np.float32)[:, None] * (0.30 + 0.70 * lam)[None, :])
    else:
        # 법선 방향 부호 있는 변위 -> 발산 컬러맵. 스케일은 DEV_SCALE 로 통일해야
        # 사람끼리·모델끼리 색이 비교된다.
        sd = np.einsum("ij,ij->j", Ry @ dev, n) / DEV_SCALE
        col = plt.get_cmap("coolwarm")(np.clip(sd * 0.5 + 0.5, 0, 1))[:, :3].T
        shade = (col * (0.45 + 0.55 * lam)[None, :]).astype(np.float32)
    s = 0.80 * res / (np.ptp(v[1]) + 1e-9)                       # 세로 기준 스케일
    p = np.stack([v[0] * s + res / 2, -v[1] * s + res / 2, v[2] * s])
    return splat(p, shade, res, radius=1, front=n[2] > 0)


def overlay(img: np.ndarray, v: np.ndarray, tri: np.ndarray,
            alpha: float = 0.62) -> np.ndarray:
    """RGB float [0,1] 반환 — 원본 위에 예측된 자세 그대로의 메쉬. v [3,N] 픽셀 좌표."""
    h, w = img.shape[:2]
    n = vertex_normals(v, tri)
    lam = np.clip(-(n.T @ LIGHT), 0, 1)
    shade = (np.array([0.35, 0.95, 0.75], np.float32)[:, None] * (0.35 + 0.65 * lam)[None, :])
    canvas = np.zeros((h, w, 3), np.float32); mask = np.zeros((h, w), bool)
    x = np.round(v[0]).astype(np.int32); y = np.round(v[1]).astype(np.int32)
    ok = (x >= 0) & (x < w) & (y >= 0) & (y < h) & (n[2] > 0)
    x, y, z, s = x[ok], y[ok], v[2][ok], shade[:, ok]
    o = np.argsort(z); x, y, s = x[o], y[o], s[:, o]
    canvas[y, x] = s.T; mask[y, x] = True
    out = img.astype(np.float32) / 255.0
    out[mask] = (1 - alpha) * out[mask] + alpha * canvas[mask]
    return np.clip(out, 0, 1)


# ------------------------------------------------------------------ #
# 프레임 위에 그리기 (영상용 — uint8 BGR)
# ------------------------------------------------------------------ #
def draw_landmarks(img: np.ndarray, L: np.ndarray, color, alpha: float = 0.85,
                   thick: int = 1) -> np.ndarray:
    """68점을 윤곽선 중심으로. 점은 작게만 찍는다. L [2 이상, 68] 이미지 좌표.

    ⚠ 점마다 흰 테두리를 두르면 선을 덮어 지저분해진다 — 선이 자세·표정을 보여 주는
      주역이고 점은 보조다. 어두운 그림자 선을 한 겹 깔아 밝은 배경에서도 보이게 한다.
    """
    # 4배로 키워 그리고 줄인다 — 안티에일리어싱만으로는 얇은 선이 계단처럼 보인다
    S = 4
    ov = cv2.resize(img, None, fx=S, fy=S, interpolation=cv2.INTER_LINEAR)
    P = np.round(L[:2].T * S).astype(np.int32)
    for w, col in ((thick * S + 3, (18, 18, 18)), (thick * S, color)):   # 그림자 -> 본선
        for idx, closed in CHAINS:
            cv2.polylines(ov, [P[list(idx)].reshape(-1, 1, 2)], closed, col, w, cv2.LINE_AA)
    # 눈·입은 점이 촘촘해 찍으면 뭉개진다 — 윤곽이 성긴 곳(턱·눈썹·코)에만 정점을 둔다
    for i in list(range(0, 36, 2)):
        cv2.circle(ov, tuple(P[i]), thick * S, (255, 255, 255), -1, cv2.LINE_AA)
    ov = cv2.resize(ov, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
    return cv2.addWeighted(ov, alpha, img, 1 - alpha, 0)


def draw_mesh(img: np.ndarray, verts: np.ndarray, tri: np.ndarray,
              alpha: float = 0.62, tint=(96, 226, 26)) -> np.ndarray:
    """조밀 메쉬를 프레임에 얹는다. verts [3,N] 이미지 좌표 (BGR 입출력)."""
    n = vertex_normals(verts, tri)
    lam = np.clip(-(n.T @ LIGHT), 0, 1)
    shade = (np.array(tint, np.float32)[:, None] / 255.0 * (0.35 + 0.65 * lam)[None, :])
    h, w = img.shape[:2]
    canvas = np.zeros((h, w, 3), np.float32); mask = np.zeros((h, w), bool)
    x = np.round(verts[0]).astype(np.int32); y = np.round(verts[1]).astype(np.int32)
    ok = (x >= 0) & (x < w) & (y >= 0) & (y < h) & (n[2] > 0)      # 뒷면 제거
    x, y, z, sh = x[ok], y[ok], verts[2][ok], shade[:, ok]
    o = np.argsort(z)                        # 먼 것부터 — 가까운 것이 덮는다
    x, y, sh = x[o], y[o], sh[:, o]
    canvas[y, x] = sh.T; mask[y, x] = True
    out = img.astype(np.float32) / 255.0
    out[mask] = (1 - alpha) * out[mask] + alpha * canvas[mask]
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def draw_depth(v: np.ndarray, tri: np.ndarray, res: int = 120) -> np.ndarray:
    """깊이 z 를 컬러맵으로 — 예측 자세 그대로. v [3,N] 크롭 좌표. RGB float 반환."""
    import matplotlib.pyplot as plt
    z = v[2]
    zc = (z - z.min()) / (np.ptp(z) + 1e-9)
    cm = plt.get_cmap("turbo")(zc)[:, :3].T.astype(np.float32)
    n = vertex_normals(v, tri)
    return splat(np.stack([v[0], v[1], z]), cm, res, radius=0,
                 bg=(1, 1, 1), front=n[2] > 0)


def label(img: np.ndarray, text: str, sub: str, color) -> np.ndarray:
    """상단 라벨 — 반투명 띠 위에 모델 이름. 어느 가중치로 낸 그림인지 남긴다."""
    h = 26
    band = img[:h].copy()
    img[:h] = cv2.addWeighted(band, 0.25, np.zeros_like(band), 0.75, 0)
    cv2.rectangle(img, (0, 0), (5, h), color, -1)
    cv2.putText(img, text, (11, 17), cv2.FONT_HERSHEY_DUPLEX, 0.46,
                (255, 255, 255), 1, cv2.LINE_AA)
    if sub:
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.46, 1)
        cv2.putText(img, sub, (17 + tw, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (190, 190, 190), 1, cv2.LINE_AA)
    return img
