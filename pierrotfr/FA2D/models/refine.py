"""국소 좌표 보정 경로 — 고해상도 특징에서 각 점 주변만 읽어 Δ를 낸다.

왜 필요한가 (Phase 10)
---------------------
Peppa 는 ASPP + **skip 연결이 있는 U-Net** 으로 stride 4 특징을 만들고 거기에
히트맵·오프셋 감독을 준다. 우리는 백본 패치 격자(stride 14/16)에 cross-attention 한 뒤
MLP 로 좌표를 회귀할 뿐, **고해상도 특징도 직접적인 공간 감독도 없다.**
이것이 정밀도 병목일 **가능성**이 있다(확정 아님 — Phase 10 참조).

설계 원칙
--------
1. **기존 경로를 유지한다.** 초기 좌표·가시성·자세는 그대로 나온다. 여기에 Δ 만 더한다.
   마지막 층을 0 초기화하므로 **추가 직후 기존 모델과 출력이 정확히 같다.**
2. **전면 cross-attention 을 하지 않는다.** 448 에서 stride 4 는 112x112 = 토큰 12.25 배다.
   초기 좌표 주변 KxK 만 `grid_sample` 로 읽는다.
3. **B/C 를 가른다.** `use_stem=False` 면 DINO 특징만 보간(B: 조밀하게 푸는 것만으로 되나),
   `True` 면 입력에서 얕은 CNN 으로 얻은 stride 4 특징을 결합한다(C: 실제 고해상도 세부가
   필요한가). B 에는 **새 관측 정보가 없다** — 이 구분이 실험의 핵심이다.
4. 보정 헤드는 **규약 간 공유**한다. 점 구분은 기존 규약별 쿼리 토큰이 한다.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class HighResStem(nn.Module):
    """입력 → stride 4 얕은 CNN 특징. **새 관측 정보의 유일한 출처.**"""

    def __init__(self, out_ch: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            # 2x2 pooling 두 번: 셀 중심은 원본 픽셀 4*i+1.5.
            # align_corners=False로 보간한 특징과 같은 격자를 사용한다.
            nn.AvgPool2d(2),
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.AvgPool2d(2),
            nn.Conv2d(32, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.GELU(),
        )
        self.out_ch = out_ch

    def forward(self, x):
        return self.body(x)                      # (B, out_ch, H/4, W/4)


class LocalRefine(nn.Module):
    """초기 좌표 주변 KxK 를 읽어 Δ(정규화 좌표)를 낸다.

    반환은 **Δ 뿐**이다 — 더하는 것은 호출부 책임이라 기존 경로가 그대로 보인다.
    """

    def __init__(self, d_model: int, patch_ch: int, stem_ch: int = 64,
                 k: int = 7, use_stem: bool = True, hidden: int = 256):
        super().__init__()
        if k < 3 or k % 2 != 1:
            raise ValueError("k must be an odd integer >= 3")
        self.k = int(k)
        self.use_stem = bool(use_stem)
        self.stem = HighResStem(stem_ch) if use_stem else None
        # L14의 1024채널을 먼저 축소한다. 112x112로 확대한 뒤 투영하면
        # 48장 배치에서 큰 중간 텐서를 불필요하게 유지한다.
        self.patch_proj = nn.Conv2d(patch_ch, hidden // 2, 1)
        fuse_in = hidden // 2 + (stem_ch if use_stem else 0)
        self.lat = nn.Conv2d(fuse_in, hidden // 2, 1)
        c = (hidden // 2) * self.k * self.k
        self.mlp = nn.Sequential(
            nn.Linear(c + d_model, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 2))
        nn.init.zeros_(self.mlp[-1].weight)                # 추가 직후 항등
        nn.init.zeros_(self.mlp[-1].bias)

    def sample_local(self, features, points, image_hw):
        """픽셀 좌표 / (W,H)를 셀 중심에 맞춰 샘플링한다. 좌표 계산은 FP32."""
        b, n, _ = points.shape
        hi_h, hi_w = features.shape[-2:]
        image_h, image_w = image_hw
        points = points.float()
        k = self.k
        off = torch.arange(k, device=points.device, dtype=points.dtype) - (k - 1) / 2
        dy = (off / hi_h * 2.0).view(1, 1, k, 1)
        dx = (off / hi_w * 2.0).view(1, 1, 1, k)
        # GT는 OpenCV 픽셀 중심 x를 W로 나눈 값이다. 반 픽셀 보정이 필요하다.
        cx = (points[..., 0] * 2 - 1 + 1.0 / image_w).view(b, n, 1, 1)
        cy = (points[..., 1] * 2 - 1 + 1.0 / image_h).view(b, n, 1, 1)
        gx = (cx + dx).expand(b, n, k, k)
        gy = (cy + dy).expand(b, n, k, k)
        grid = torch.stack([gx, gy], dim=-1).reshape(b, n * k, k, 2)
        # ⚠ 크롭 밖 초기 좌표는 zeros padding 으로 읽히고, 그 점은 호출부에서 Δ 를 끈다.
        # BF16으로 격자를 만들면 448 입력에서 subpixel 위치가 손실된다.
        # CPU grid_sample의 BF16 미지원도 피한다. 특징 경로의 gradient는 유지된다.
        with torch.autocast(device_type=features.device.type, enabled=False):
            s = F.grid_sample(features.float(), grid, mode="bilinear",
                              padding_mode="zeros", align_corners=False)
        ch = s.shape[1]
        return s.reshape(b, ch, n, k, k).permute(0, 2, 1, 3, 4).reshape(b, n, ch * k * k)

    def forward(self, images, patch, dec, points):
        """images (B,3,H,W), patch (B,C,h,w), dec (B,N,d), points (B,N,2)."""
        image_h, image_w = images.shape[-2:]
        if image_h % 4 or image_w % 4:
            raise ValueError("local refinement requires image dimensions divisible by 4")
        f = F.interpolate(self.patch_proj(patch), size=(image_h // 4, image_w // 4),
                          mode="bilinear", align_corners=False)
        if self.stem is not None:
            f = torch.cat([f, self.stem(images)], dim=1)
        f = self.lat(f)
        s = self.sample_local(f, points, (image_h, image_w))
        return self.mlp(torch.cat([s.to(dec.dtype), dec], dim=-1))
