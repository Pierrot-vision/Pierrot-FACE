"""MobileNet-V1 백본 + 이중 헤드 (3DMM 파라미터 · 68 랜드마크).

3DDFA_V2 가 배포한 `weights/mb1_120x120.pth` 에는 헤드가 **둘** 들어 있다.

    module.fc_param.weight  [62, 1024]    3DMM 파라미터 회귀 — 추론 경로
    module.fc_lm.weight     [136, 1024]   68점 랜드마크 회귀 — 학습 전용(lrr)

그런데 배포된 모델 정의(`models/mobilenet_v1.py`)에는 `fc` 하나뿐이고, 로더가
`fc_param -> fc` 로 이름만 바꿔 실은 뒤 `fc_lm` 은 **버린다**. 즉 논문 2.3절의
landmark-regression 분기가 공개 코드에서 빠져 있다. 여기서 되살린다.

파라미터 이름은 배포 체크포인트와 **일부러 동일하게** 뒀다 — 사전학습 가중치를
초기값으로 얹거나 결과를 3DDFA_V2 추론 코드에 그대로 넣을 수 있다.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn



class DepthWiseBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int = 1):
        super().__init__()
        inplanes, planes = int(inplanes), int(planes)
        self.conv_dw = nn.Conv2d(inplanes, inplanes, 3, stride, 1,
                                 groups=inplanes, bias=False)
        self.bn_dw = nn.BatchNorm2d(inplanes)
        self.conv_sep = nn.Conv2d(inplanes, planes, 1, 1, 0, bias=False)
        self.bn_sep = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn_dw(self.conv_dw(x)))
        return self.relu(self.bn_sep(self.conv_sep(x)))


class MobileNetV1(nn.Module):
    """120x120 입력 기준 3.27M 파라미터 / 183.5M MACs (widen_factor=1.0)."""

    def __init__(self, num_params: int = 62, num_landmarks: int = 68,
                 widen_factor: float = 1.0, use_lrr: bool = True,
                 input_channel: int = 3):
        super().__init__()
        w = widen_factor
        self.use_lrr = use_lrr

        self.conv1 = nn.Conv2d(input_channel, int(32 * w), 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(32 * w))
        self.relu = nn.ReLU(inplace=True)

        blk = DepthWiseBlock
        self.dw2_1 = blk(32 * w, 64 * w)
        self.dw2_2 = blk(64 * w, 128 * w, stride=2)
        self.dw3_1 = blk(128 * w, 128 * w)
        self.dw3_2 = blk(128 * w, 256 * w, stride=2)
        self.dw4_1 = blk(256 * w, 256 * w)
        self.dw4_2 = blk(256 * w, 512 * w, stride=2)
        self.dw5_1 = blk(512 * w, 512 * w)
        self.dw5_2 = blk(512 * w, 512 * w)
        self.dw5_3 = blk(512 * w, 512 * w)
        self.dw5_4 = blk(512 * w, 512 * w)
        self.dw5_5 = blk(512 * w, 512 * w)
        self.dw5_6 = blk(512 * w, 1024 * w, stride=2)
        self.dw6 = blk(1024 * w, 1024 * w)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        feat = int(1024 * w)
        self.feat_dim = feat          # 학습 저장소의 synergy 순환이 물어보던 값
        self.fc_param = nn.Linear(feat, num_params)
        # 논문 2.3절: 글로벌 풀링 **뒤에** 붙는 별도 태스크 헤드. 추론에선 안 쓴다.
        self.fc_lm = nn.Linear(feat, num_landmarks * 2) if use_lrr else None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # ---------------------------------------------------------------- #
    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        for name in ("dw2_1", "dw2_2", "dw3_1", "dw3_2", "dw4_1", "dw4_2",
                     "dw5_1", "dw5_2", "dw5_3", "dw5_4", "dw5_5", "dw5_6", "dw6"):
            x = getattr(self, name)(x)
        return self.avgpool(x).flatten(1)

    def forward(self, x: torch.Tensor) -> dict:
        f = self.features(x)
        out = {"param": self.fc_param(f), "feat": f}
        if self.fc_lm is not None:
            out["landmark"] = self.fc_lm(f).view(f.shape[0], -1, 2)
        return out

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """추론 경로 — 파라미터만. lrr 분기는 타지 않는다."""
        return self.fc_param(self.features(x))

    def inference_modules(self):
        """추론에 실제로 쓰이는 모듈들 (파라미터 수 보고용)."""
        skip = {"fc_lm"}
        return [m for n, m in self.named_children() if n not in skip]


def mobilenet_v1(**kw):
    return MobileNetV1(widen_factor=1.0, **kw)


def mobilenet_v1_x05(**kw):
    return MobileNetV1(widen_factor=0.5, **kw)
