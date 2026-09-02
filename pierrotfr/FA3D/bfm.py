"""3DMM(BFM) 디코더 — 62-d 파라미터를 3D 정점/랜드마크로 편다.

모델이 내는 것은 62개의 숫자뿐이다. 얼굴이 되는 것은 여기서다.

파라미터 규약 (62-d):
    p[0:12]   T = f[R; t3d]  (3x4, row-major)   — 상사변환. Euler 각 대신 행렬을
                                                  직접 회귀해 gimbal lock 을 피한다
    p[12:52]  alpha_shp (40)                    — BFM shape PCA 계수
    p[52:62]  alpha_exp (10)                    — 표정 PCA 계수

⚠ 정점 배열은 xyz 가 **교차**된 [x0,y0,z0,x1,...] 평탄형이다. 3xN 으로 펼 때
  `view(N,3).T` 를 써야 한다(`view(3,N)` 이 아니다 — 조용히 틀린 얼굴이 나온다).

⚠ 학습 저장소(Pierrot_FR_Lab)의 같은 파일에서 **손실 계산용 상수(A_norm)와
  numpy 사본(BFMNumpy)을 뺀 것**이다. 둘 다 fWPDC / short-video-synthesis 전용이라
  추론 경로에 없다. 나머지 수식과 좌표 규약은 한 글자도 다르지 않다 — 다르면
  같은 가중치가 다른 얼굴을 낸다.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import torch
import torch.nn as nn

TRANS_DIM, SHAPE_DIM, EXP_DIM = 12, 40, 10


def parse_param(param: torch.Tensor):
    """[B,62] -> (T [B,3,4], R [B,3,3], offset [B,3,1], alpha [B,50,1])"""
    b = param.shape[0]
    T = param[:, :TRANS_DIM].view(b, 3, 4)
    R = T[:, :, :3]
    offset = T[:, :, 3].view(b, 3, 1)
    alpha = param[:, TRANS_DIM:].view(b, SHAPE_DIM + EXP_DIM, 1)
    return T, R, offset, alpha


class BFM(nn.Module):
    """bfm_noneck_v3.pkl 로더 + 재구성.

    3DDFA v1 이 배포한 basis(53,215 정점)와 V2 의 bfm_noneck_v3(38,365 정점)는
    목 정점만 제거·재색인한 것이고 **62-d 파라미터 공간은 동일하다**
    (동일 파라미터 디코딩 시 68 랜드마크 차이 0.002px). 그래서 저자 배포 가중치와
    우리 체크포인트를 **같은 기저로** 디코딩해 나란히 비교할 수 있다.
    """

    def __init__(self, bfm_fp: str, shape_dim: int = SHAPE_DIM, exp_dim: int = EXP_DIM):
        super().__init__()
        if not os.path.isfile(bfm_fp):
            raise SystemExit(f"[FA3D] BFM 파일이 없습니다: {bfm_fp}")
        with open(bfm_fp, "rb") as fh:
            bfm = pickle.load(fh)

        u = bfm["u"].astype(np.float32)                       # [3N, 1]
        w_shp = bfm["w_shp"].astype(np.float32)[:, :shape_dim]
        w_exp = bfm["w_exp"].astype(np.float32)[:, :exp_dim]
        A = np.concatenate((w_shp, w_exp), axis=1)            # [3N, 50]
        kp = bfm["keypoints"].astype(np.int64)                # [204] = 68 x 3

        self.n_vertex = u.shape[0] // 3
        # 학습 대상이 아니므로 buffer — state_dict 에 담기면 체크포인트가 26MB 커진다
        self.register_buffer("u", torch.from_numpy(u).view(-1), persistent=False)
        self.register_buffer("A", torch.from_numpy(A), persistent=False)
        self.register_buffer("u_kp", torch.from_numpy(u[kp]).view(-1), persistent=False)
        self.register_buffer("A_kp", torch.from_numpy(A[kp]), persistent=False)

    # ---------------------------------------------------------------- #
    def shape(self, alpha: torch.Tensor, sparse: bool = False) -> torch.Tensor:
        """투영 전 3D 형상 S = S̄ + Aα  ->  [B, 3, N]

        `alpha` 의 뒤 10 차원(표정)을 0 으로 지우면 **중립 얼굴**이 나온다 —
        identity 를 눈으로 비교할 때 쓴다 (scripts/FA3D/pred_3d.py --compare).
        """
        u, A = (self.u_kp, self.A_kp) if sparse else (self.u, self.A)
        v = u.unsqueeze(0) + (A @ alpha).squeeze(-1)          # [B, 3N]
        return v.view(v.shape[0], -1, 3).permute(0, 2, 1)     # [B, 3, N]

    def reconstruct(self, param: torch.Tensor, sparse: bool = False) -> torch.Tensor:
        """62-d(비정규화) -> 카메라 좌표계 정점 [B, 3, N]"""
        _, R, offset, alpha = parse_param(param)
        return R @ self.shape(alpha, sparse=sparse) + offset

    def landmarks3d(self, param: torch.Tensor, size: int = 120) -> torch.Tensor:
        """68 랜드마크 [B, 3, 68] — 크롭 이미지 좌표계 (깊이 z 유지).

        3DDFA_V2 `utils/tddfa_util.similar_transform` 과 동일한 규약
        (x-1, y 는 위아래 뒤집기). 평가 코드와 어긋나면 NME 가 통째로 틀어진다.
        """
        pts = self.reconstruct(param, sparse=True)            # [B, 3, 68]
        x = pts[:, 0, :] - 1
        y = size - pts[:, 1, :]
        return torch.stack((x, y, pts[:, 2, :]), dim=1)       # [B, 3, 68]

    def landmarks(self, param: torch.Tensor, size: int = 120) -> torch.Tensor:
        """68 랜드마크의 2D 투영 [B, 68, 2] — NME 평가용."""
        return self.landmarks3d(param, size)[:, :2, :].permute(0, 2, 1)


class ParamNorm(nn.Module):
    """62-d 파라미터의 Z-score 정규화 통계. 정규화/역정규화 양방향.

    모델은 정규화 공간에서 회귀한다 — BFM 에 넣기 전에 반드시 `denorm()` 을 통과해야
    한다. 빠뜨리면 오류 없이 **뭉개진 얼굴**이 나온다.
    """

    def __init__(self, mean_std_fp: str):
        super().__init__()
        with open(mean_std_fp, "rb") as fh:
            r = pickle.load(fh)
        # 3DDFA_V2 는 {'mean','std'}, v1 param_whitening.pkl 은 {'param_mean','param_std'}
        mean = r.get("mean", r.get("param_mean"))
        std = r.get("std", r.get("param_std"))
        if mean is None or std is None:
            raise SystemExit(f"[FA3D] mean/std 키를 찾을 수 없습니다: {mean_std_fp}")
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32),
                             persistent=False)
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32),
                             persistent=False)

    def denorm(self, p: torch.Tensor) -> torch.Tensor:
        return p * self.std + self.mean

    def norm(self, p: torch.Tensor) -> torch.Tensor:
        return (p - self.mean) / self.std


def load_tri(tri_fp: str, n_vertex: int) -> np.ndarray:
    """삼각형 인덱스 [3, T] — 메쉬 렌더에 쓴다.

    ⚠ `bfm_noneck_v3.pkl` 안의 'tri' 를 쓰면 안 된다. 인덱스 최대가 46,851 로 그
      파일의 정점 수 38,365 를 넘는다 — 목을 제거하기 **전** 기저(53,215)의 삼각형이
      그대로 남아 있는 것이다. 오피셜도 별도 `configs/tri.pkl` 을 쓴다. 여기서
      범위를 확인하고 넘긴다.
    """
    if not os.path.isfile(tri_fp):
        raise SystemExit(f"[FA3D] tri 파일이 없습니다: {tri_fp}\n"
                         f"  3DDFA_V2 오피셜 저장소의 configs/tri.pkl 을 놓으세요.")
    with open(tri_fp, "rb") as fh:
        tri = np.asarray(pickle.load(fh)).astype(np.int64)
    if tri.shape[0] != 3:
        tri = tri.T
    if tri.max() >= n_vertex:
        raise SystemExit(f"[FA3D] tri 인덱스 최대 {tri.max()} 가 정점 수 {n_vertex} 를 "
                         f"넘습니다: {tri_fp}")
    return tri
