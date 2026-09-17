"""경량 CNN 백본 — torchvision ImageNet 사전학습, **전량 사용**.

왜 필요한가
-----------
FAS 레지스트리(CLIP/DINOv2)는 범용 대형 ViT 뿐이라 최소가 DINOv2-S/14(22M)다.
HRFFA 자신의 PP-HGNetV2-B0 / ViT-T 가중치는 릴리스에 없어(asset 50 개 확인) 이식해도
스크래치가 된다. 그래서 같은 급의 torchvision CNN 을 쓴다 — ImageNet 가중치가 함께 온다.

⚠ 첫 판(2026-09-06)의 결함과 수정
--------------------------------
처음엔 "stride 16 에서 끊는다"며 stride-32 단계를 **잘라 버렸다.** 그 결과 사전학습
파라미터의 7.5~23.8% 만 남았고(mobilenetv3s 는 2.54M 중 0.19M), 남은 쪽이 하필
엣지·텍스처 수준의 얕은 스템이었다. ImageNet 사전학습의 의미 표현은 전부 버린 셈이라
"사전학습 CNN"이 아니라 거의 스크래치 얕은 망이 됐다.
측정: ep15 에서 io-NME 9.3~9.6 (같은 일정의 DINOv2 아암 5.0)  → `_ablation_trunc_*`.

수정: **전체 망을 쓰되 격자는 stride 16 을 유지한다.** stride-32 특징을 2배 업샘플해
stride-16 특징과 1x1 lateral 로 융합한다(FPN-lite). 가중치 100% 보존 + 격자 16x16 유지.

계약: 점 쿼리 디코더가 요구하는 (patch (B,C,h,w), cls (B,C)) 를 낸다.

⚠ `pretrained` 기본값이 학습 저장소와 **반대(False)** 다. ImageNet 가중치는 학습의
  초기값이었고, 추론에서는 우리 체크포인트가 통째로 덮는다 — 받아 봐야 버려진다.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models as tvm

# 이름 → (생성자, stride16 경계 인덱스)  — 경계 앞이 s16, 뒤가 s32
CNN_BACKBONES = {
    "mobilenetv3s": (tvm.mobilenet_v3_small, 9),
    "mobilenetv3l": (tvm.mobilenet_v3_large, 13),
    "effnetb0":     (tvm.efficientnet_b0,     6),
    "shufflenet":   (tvm.shufflenet_v2_x1_0, -1),
    "resnet18":     (tvm.resnet18,           -1),
}

FUSE_DIM = 256          # 융합 후 채널. 디코더 d_model 과 무관하게 고정한다.


class CNNBackbone(nn.Module):
    """ImageNet 사전학습 CNN 전량 + stride16/32 FPN-lite 융합."""

    def __init__(self, name: str, image_size: int = 224, pretrained: bool = False,
                 fuse: bool = True):
        super().__init__()
        if name not in CNN_BACKBONES:
            raise ValueError(f"cnn backbone={name!r} — 가능: {sorted(CNN_BACKBONES)}")
        ctor, idx = CNN_BACKBONES[name]
        net = ctor(weights="DEFAULT" if pretrained else None)

        if name in ("mobilenetv3s", "mobilenetv3l", "effnetb0"):
            feats = list(net.features)
            self.s16 = nn.Sequential(*feats[:idx])
            self.s32 = nn.Sequential(*feats[idx:])
        elif name == "shufflenet":
            self.s16 = nn.Sequential(net.conv1, net.maxpool, net.stage2, net.stage3)
            self.s32 = nn.Sequential(net.stage4, net.conv5)
        else:                                   # resnet18
            self.s16 = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                     net.layer1, net.layer2, net.layer3)
            self.s32 = nn.Sequential(net.layer4)

        self.fuse = fuse
        c16, c32 = self._probe(image_size)
        if fuse:
            self.lat16 = nn.Conv2d(c16, FUSE_DIM, 1)
            self.lat32 = nn.Conv2d(c32, FUSE_DIM, 1)
            self.smooth = nn.Conv2d(FUSE_DIM, FUSE_DIM, 3, padding=1)
            self.embed_dim = FUSE_DIM
        else:                                   # 대조군: 잘라 쓰던 첫 판 재현
            self.embed_dim = c16

    @torch.no_grad()
    def _probe(self, size: int):
        f16 = self.s16(torch.zeros(1, 3, size, size))
        f32 = self.s32(f16)
        return int(f16.shape[1]), int(f32.shape[1])

    def forward(self, x: torch.Tensor):
        f16 = self.s16(x)
        if not self.fuse:
            return f16, f16.mean(dim=(2, 3))
        f32 = self.s32(f16)
        up = F.interpolate(self.lat32(f32), size=f16.shape[-2:],
                           mode="bilinear", align_corners=False)
        f = self.smooth(self.lat16(f16) + up)   # (B,FUSE_DIM,h,w) — 격자는 stride16
        return f, f.mean(dim=(2, 3))            # cls = GAP
