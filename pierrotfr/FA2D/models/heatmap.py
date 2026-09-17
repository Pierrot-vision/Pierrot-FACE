"""쿼리 기반 히트맵 + soft-argmax (D-ViT 방식) — 추론 경로만.

쿼리 토큰과 memory 토큰의 내적으로 점마다 히트맵을 만든다(추가 파라미터가 거의 없다).
  heat[b,n,m] = <q[b,n], k[b,m]> / sqrt(d)
softmax 후 격자 좌표의 기댓값이 soft-argmax 좌표다.

⚠ 크롭 밖 점: 히트맵은 크롭 안에서만 정의되므로 soft-argmax 로 크롭 밖을 낼 수 없다.
  디코더가 크롭 밖 점에는 회귀값을 그대로 쓴다.

학습 저장소의 같은 파일에서 `gaussian_targets` · `AWingLoss`(감독용)를 뺐다.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class QueryHeatmap(nn.Module):
    """디코더 토큰 × memory 토큰 → 히트맵 → soft-argmax 좌표."""

    def __init__(self, d_model: int, sigma_cells: float = 1.0, temp: float = 1.0):
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.scale = 1.0 / math.sqrt(d_model)
        self.sigma_cells = float(sigma_cells)
        self.temp = float(temp)

    def forward(self, dec, mem, h, w):
        """dec (B,N,d) · mem (B,h*w,d) → (logits (B,N,h,w), points (B,N,2))"""
        logits = torch.einsum("bnd,bmd->bnm", self.q(dec), self.k(mem)) * self.scale
        logits = (logits / self.temp).reshape(dec.shape[0], dec.shape[1], h, w)
        p = F.softmax(logits.flatten(2), dim=-1).reshape_as(logits)
        dev = p.device
        ys = ((torch.arange(h, device=dev, dtype=torch.float32) + 0.5) / h).view(1, 1, h, 1)
        xs = ((torch.arange(w, device=dev, dtype=torch.float32) + 0.5) / w).view(1, 1, 1, w)
        x = (p * xs).sum((2, 3))
        y = (p * ys).sum((2, 3))
        return logits, torch.stack([x, y], dim=-1)
