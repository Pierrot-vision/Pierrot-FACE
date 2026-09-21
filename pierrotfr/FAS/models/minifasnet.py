"""MiniFASNet — 경량 FAS 베이스라인 (0.43M).

출처: MiniVision Silent-Face-Anti-Spoofing → yakhyo/face-anti-spoofing (ONNX 배포판은
uniface.spoofing.MiniFASNet). 구조 자체는 **MobileFaceNet 계열**이다.

InstructFLIP(176M) 대비 **400배 작다.** "그 규모가 정말 값을 하는가"를 재는 대조군으로
같은 데이터·같은 지표에서 돌린다.

구조:
    입력 [B,3,80,80]
      stem       Conv3×3 s2 + BN + PReLU        → [B,32,40,40]
      stem_dw    DWConv3×3                       → [B,32,40,40]
      transition1  InvRes s2                     → [B,64,20,20]
      stage2       InvRes ×4                     → [B,64,20,20]
      transition2  InvRes s2                     → [B,128,10,10]
      stage3       InvRes ×6                     → [B,128,10,10]
      transition3  InvRes s2                     → [B,128,5,5]
      stage4       InvRes ×2                     → [B,128,5,5]
      final_expand Conv1×1                       → [B,512,5,5]
      final_dw     DWConv5×5 (활성 없음)          → [B,512,1,1]   ★ GDConv
      Flatten → Linear(512→128) → BN → Dropout → Linear(128→C)

핵심 두 가지:

1. **GDConv (Global Depthwise Conv)** — GAP 대신 5×5 depthwise 로 5×5 를 1×1 로 접는다.
   GAP 는 모든 위치를 동일 가중치로 평균내지만 GDConv 는 위치별 가중치를 학습한다.
   얼굴처럼 공간 배치가 고정된 입력에서 유리하다. MobileFaceNet 의 시그니처.

2. **3-class 출력** — 이진이 아니다. 원 규약은 {0: 2D fake, 1: real, 2: 3D fake} 로,
   인쇄/재생과 마스크를 나눠 배운다. 물리적으로 다른 아티팩트라 분리가 합리적이다.
   num_classes=2 로 두면 이진으로도 쓸 수 있다.

⚠ 원본은 얼굴 bbox 를 **2.7~4.0배 넓게** 잘라 배경 맥락을 함께 본다(사진 테두리, 화면 베젤,
  손에 든 모습, 마스크 경계). 이 저장소의 기본 전처리는 랜드마크 정렬 타이트 crop 이라
  그 맥락이 없다. 공정하게 비교하려면 prepare 단계에서 align_padding 을 키운 넓은 crop 을
  따로 만들어야 한다 — LAB/FAS/MiniFASNet.md 참조.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# (expand_channels, out_channels) 목록. 원본 yakhyo/face-anti-spoofing 과 동일.
CHANNEL_CONFIGS = {
    "v1se": {
        "stem": 32,
        "transition1": {"expand": 103, "out": 64},
        "stage2": [(13, 64), (26, 64), (13, 64), (52, 64)],
        "transition2": {"expand": 231, "out": 128},
        "stage3": [(154, 128), (52, 128), (26, 128), (52, 128), (26, 128), (26, 128)],
        "transition3": {"expand": 308, "out": 128},
        "stage4": [(26, 128), (26, 128)],
        "final": 512,
    },
    "v2": {
        "stem": 32,
        "transition1": {"expand": 103, "out": 64},
        "stage2": [(13, 64), (13, 64), (13, 64), (13, 64)],
        "transition2": {"expand": 231, "out": 128},
        "stage3": [(231, 128), (52, 128), (26, 128), (77, 128), (26, 128), (26, 128)],
        "transition3": {"expand": 308, "out": 128},
        "stage4": [(26, 128), (26, 128)],
        "final": 512,
    },
}

# 원본 배포 설정 — 변종마다 crop scale 과 dropout 이 다르다
VARIANT_DEFAULTS = {
    "v1se": {"use_se": True, "dropout": 0.75, "crop_scale": 4.0},
    "v2": {"use_se": False, "dropout": 0.2, "crop_scale": 2.7},
}


class ConvBNPReLU(nn.Module):
    """Conv2d + BatchNorm2d + PReLU. bias 는 BN 이 흡수하므로 두지 않는다."""

    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, padding=0,
                 groups=1, activation=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding,
                              groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.prelu = nn.PReLU(out_ch) if activation else nn.Identity()

    def forward(self, x):
        return self.prelu(self.bn(self.conv(x)))


class SqueezeExcite(nn.Module):
    """채널 어텐션. V1SE 에서만 쓰이며 각 stage 의 **마지막 블록**에만 붙는다."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, reduced, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(reduced)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        s = self.pool(x)
        s = self.relu(self.bn1(self.fc1(s)))
        s = torch.sigmoid(self.bn2(self.fc2(s)))
        return x * s


class InvertedResidual(nn.Module):
    """expand(1×1) → depthwise(3×3) → project(1×1, 활성 없음) + 선택적 SE·residual."""

    def __init__(self, in_ch, expand_ch, out_ch, stride=1, use_se=False):
        super().__init__()
        self.use_residual = (in_ch == out_ch) and (stride == 1)
        self.conv = ConvBNPReLU(in_ch, expand_ch, 1)
        self.conv_dw = ConvBNPReLU(expand_ch, expand_ch, 3, stride, 1, groups=expand_ch)
        self.project = ConvBNPReLU(expand_ch, out_ch, 1, activation=False)
        self.se = SqueezeExcite(out_ch) if use_se else nn.Identity()

    def forward(self, x):
        out = self.se(self.project(self.conv_dw(self.conv(x))))
        return out + x if self.use_residual else out


class ResidualStack(nn.Module):
    """InvertedResidual 스택. SE 는 **마지막 블록에만** 적용한다 (원본 규약)."""

    def __init__(self, in_ch, block_configs, use_se=False):
        super().__init__()
        layers, cur = [], in_ch
        n = len(block_configs)
        for i, (exp_ch, out_ch) in enumerate(block_configs):
            layers.append(InvertedResidual(cur, exp_ch, out_ch, 1,
                                           use_se=use_se and i == n - 1))
            cur = out_ch
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MiniFASNetBackbone(nn.Module):
    """80×80 입력을 전제로 한 본체. 다른 해상도를 쓰면 GDConv 커널이 안 맞는다."""

    def __init__(self, config: dict, num_classes: int = 3,
                 dropout: float = 0.2, use_se: bool = False,
                 input_size: int = 80, embed_dim: int = 128):
        super().__init__()
        stem_ch = config["stem"]
        t1, t2, t3 = config["transition1"], config["transition2"], config["transition3"]
        final_ch = config["final"]

        self.stem = ConvBNPReLU(3, stem_ch, 3, 2, 1)
        self.stem_dw = ConvBNPReLU(stem_ch, stem_ch, 3, 1, 1, groups=stem_ch)

        self.transition1 = InvertedResidual(stem_ch, t1["expand"], t1["out"], stride=2)
        self.stage2 = ResidualStack(t1["out"], config["stage2"], use_se)
        self.transition2 = InvertedResidual(config["stage2"][-1][1], t2["expand"],
                                            t2["out"], stride=2)
        self.stage3 = ResidualStack(t2["out"], config["stage3"], use_se)
        self.transition3 = InvertedResidual(config["stage3"][-1][1], t3["expand"],
                                            t3["out"], stride=2)
        self.stage4 = ResidualStack(t3["out"], config["stage4"], use_se)

        self.final_expand = ConvBNPReLU(config["stage4"][-1][1], final_ch, 1)
        # ★ GDConv — stride 2 가 네 번이므로 공간 크기는 input_size/16 이 된다.
        #   80 → 5. 커널을 그 크기로 잡아 1×1 로 접는다.
        grid = input_size // 16
        if grid < 1:
            raise ValueError(f"input_size={input_size} 가 너무 작습니다 (16 이상 필요)")
        self.final_dw = ConvBNPReLU(final_ch, final_ch, grid, groups=final_ch,
                                    activation=False)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(final_ch, embed_dim, bias=False)
        self.bn = nn.BatchNorm1d(embed_dim)
        self.drop = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(embed_dim, num_classes, bias=False)

    def features(self, x):
        x = self.stem_dw(self.stem(x))
        x = self.stage2(self.transition1(x))
        x = self.stage3(self.transition2(x))
        x = self.stage4(self.transition3(x))
        x = self.final_dw(self.final_expand(x))
        return self.bn(self.linear(self.flatten(x)))

    def forward(self, x):
        return self.classifier(self.drop(self.features(x)))


class MiniFASNet(nn.Module):
    """Lab 계약에 맞춘 어댑터.

    forward(batch)  -> {"logits", "cue": None, "label_key", ...}
    predict(batch)  -> {"score": spoof 확률}

    num_classes=3 이면 배치의 `label3` 를 타깃으로 쓴다 (0=2D fake, 1=real, 2=3D fake).
    이 라벨은 데이터셋이 spoof_type 에서 만든다.
    """

    def __init__(self, variant: str = "v2", num_classes: int = 3,
                 image_size: int = 80, dropout: float | None = None,
                 embed_dim: int = 128, **_ignored):
        super().__init__()
        if variant not in CHANNEL_CONFIGS:
            raise ValueError(f"variant={variant!r} — 가능: {sorted(CHANNEL_CONFIGS)}")
        if num_classes not in (2, 3):
            raise ValueError(f"num_classes={num_classes} — 2 또는 3")

        d = VARIANT_DEFAULTS[variant]
        self.variant = variant
        self.num_classes = num_classes
        self.crop_scale = d["crop_scale"]        # 참고용 — 전처리에서 쓴다
        self.net = MiniFASNetBackbone(
            CHANNEL_CONFIGS[variant], num_classes=num_classes,
            dropout=d["dropout"] if dropout is None else dropout,
            use_se=d["use_se"], input_size=image_size, embed_dim=embed_dim)

        # 3-class 일 때 real 인덱스. 원 규약은 {0:2D fake, 1:real, 2:3D fake}.
        self.real_index = 1 if num_classes == 3 else 0

    # ------------------------------------------------------------------ #
    def forward(self, batch: dict) -> dict:
        logits = self.net(batch["image"])
        zero = logits.new_zeros(())
        return {
            "logits": logits,
            "cue": None,
            # 엔진에게 어떤 라벨로 CE 를 걸지 알려준다
            "label_key": "label3" if self.num_classes == 3 else "label",
            "content_lm_loss": zero,
            "style_lm_loss": zero,
        }

    @torch.no_grad()
    def predict(self, batch: dict) -> dict:
        logits = self.net(batch["image"])
        lg = logits.float()
        p = F.softmax(lg, dim=-1)
        # spoof 확률 = 1 - P(real). 3-class 면 2D/3D fake 를 합치는 것과 같다.
        score = 1.0 - p[:, self.real_index]
        # margin = log P(spoof) - log P(real) = logsumexp(fake 로짓) - real 로짓.
        # 확률은 포화하므로 평가(순위 지표)는 margin 으로 한다.
        fake = [i for i in range(lg.shape[1]) if i != self.real_index]
        margin = torch.logsumexp(lg[:, fake], dim=1) - lg[:, self.real_index]
        return {"score": score, "margin": margin, "logits": logits, "cue": None}

    def inference_modules(self) -> list[nn.Module]:
        return [self.net]


def minifasnet_v1se(**kw):
    return MiniFASNet(variant="v1se", **kw)


def minifasnet_v2(**kw):
    return MiniFASNet(variant="v2", **kw)
