"""stride 4 히트맵 + x/y 오프셋 출력 (Peppa 학생 구조의 이식) — 추론 경로만.

좌표를 stride 16 특징에서 바로 회귀하는 대신 고해상도(stride 4) 격자에서 만든다.
규약이 4 종(29·68·98·106)이라 출력 채널을 점마다 만들지 않고 **점 쿼리와 stride 4
특징의 내적**으로 맵을 만든다.

좌표 디코딩은 기댓값이다: p = Σ softmax(logit) · (셀 중심 + 오프셋).

학습 저장소의 같은 파일에서 `hm_offset_losses`(감독용)를 뺐다.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .refine import HighResStem


class HeatmapOffsetHead(nn.Module):
    """(B,N,d) 쿼리 × stride 4 특징 → 히트맵·오프셋·좌표."""

    def __init__(self, d_model: int, patch_ch: int, stem_ch: int = 64, temp: float = 1.0):
        super().__init__()
        self.stem = HighResStem(stem_ch)
        self.fuse = nn.Sequential(
            nn.Conv2d(stem_ch + patch_ch, d_model, 3, padding=1),
            nn.BatchNorm2d(d_model), nn.GELU(),
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.BatchNorm2d(d_model), nn.GELU())
        self.q_hm = nn.Linear(d_model, d_model)
        self.q_ox = nn.Linear(d_model, d_model)
        self.q_oy = nn.Linear(d_model, d_model)
        self.temp = float(temp)
        for lin in (self.q_ox, self.q_oy):        # 오프셋은 0 에서 시작한다
            nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)

    def forward(self, images: torch.Tensor, patch: torch.Tensor, dec: torch.Tensor) -> dict:
        b, _, H, W = images.shape
        f4 = self.stem(images)                                     # (B, stem, H/4, W/4)
        up = F.interpolate(patch.float(), size=f4.shape[-2:], mode="bilinear",
                           align_corners=False).to(f4.dtype)
        feat = self.fuse(torch.cat([f4, up], dim=1))               # (B, d, H/4, W/4)
        h4, w4 = feat.shape[-2:]
        flat = feat.flatten(2)                                     # (B, d, h4*w4)
        logit = torch.einsum("bnd,bdm->bnm", self.q_hm(dec), flat) / self.temp
        prob = logit.softmax(-1)
        off_x = torch.einsum("bnd,bdm->bnm", self.q_ox(dec), flat)
        off_y = torch.einsum("bnd,bdm->bnm", self.q_oy(dec), flat)

        # 셀 중심(정규화 좌표) — align_corners=False 격자와 같은 규약
        ys, xs = torch.meshgrid(torch.arange(h4, device=feat.device, dtype=torch.float32),
                                torch.arange(w4, device=feat.device, dtype=torch.float32),
                                indexing="ij")
        cx = ((xs + 0.5) / w4).flatten()[None, None]               # (1,1,m)
        cy = ((ys + 0.5) / h4).flatten()[None, None]
        # 오프셋 단위도 셀 크기 — 한 셀 이상 못 움직이게 tanh 로 묶는다
        px = (prob * (cx + torch.tanh(off_x) / w4)).sum(-1)
        py = (prob * (cy + torch.tanh(off_y) / h4)).sum(-1)
        return {"points": torch.stack([px, py], dim=-1),
                "hm4_logits": logit.reshape(b, -1, h4, w4),
                "off_x": off_x.reshape(b, -1, h4, w4),
                "off_y": off_y.reshape(b, -1, h4, w4)}
