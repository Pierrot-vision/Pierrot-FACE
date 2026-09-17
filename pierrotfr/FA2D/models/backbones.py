"""FA2D 백본 어댑터 — 추론 경로만.

디코더가 요구하는 계약은 하나다: `(patch (B,C,h,w), cls (B,C))`.

    dinov3-vitl16   DINOv3 ViT-L/16 — 교사 (`full_v12`)
    vit-t16         ViT-T/16        — 학생 (`stu_final_v2`)
    mobilenetv3s/l · effnetb0 · shufflenet · resnet18 — 경량 CNN (FPN-lite 융합)

학습 저장소와의 차이 두 가지:

  · **사전학습 가중치를 읽지 않는다.** 학습 저장소는 `from_pretrained` 로 백본 초기값을
    받았다. 추론에서는 우리 체크포인트가 그 위를 통째로 덮으므로, 구조만 `from_config`
    로 만든다 — DINOv3 ViT-L 은 1.2GB 를 읽고 버리는 일이 사라진다.
  · **FAS 레지스트리(CLIP · DINOv2)를 가져오지 않는다.** 학습 저장소는 이 어댑터를
    `pierrotfr/FAS` 에서 빌려 왔는데, 공개한 두 모델은 쓰지 않는다. 카테고리 간 import 를
    끊으려고 뺐다. DINOv2 계열 체크포인트를 넣으면 무엇이 없는지 알리고 멈춘다.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .cnn_backbones import CNN_BACKBONES, CNNBackbone

IMAGENET_NORM = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

# ViT-T/16 구조 정의 (HF). 가중치는 쓰지 않는다 — config 만 읽는다.
VIT_T16 = "WinKawaks/vit-tiny-patch16-224"


def _dinov3_dir() -> str:
    """DINOv3 config 가 있는 로컬 디렉토리.

    ⚠ DINOv3 는 **재배포 금지 라이선스**라 이 저장소에 넣을 수 없다. 공식 저장소는 gated 라
    각자 받아 두어야 한다 — 경로는 다른 데이터 경로와 같은 규칙으로 푼다.
    """
    from configs.paths import FA2D_DINOV3_DIR
    return FA2D_DINOV3_DIR


class DINOv3Backbone(nn.Module):
    """DINOv3 ViT-L/16 → (patch (B,C,h,w), cls (B,C)).

    출력 토큰은 [CLS] + register N개 + 패치다. register 를 빼야 패치 격자가 정사각이 된다.
    """

    def __init__(self, image_size: int = 448, path: str = ""):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        path = path or _dinov3_dir()
        try:
            cfg = AutoConfig.from_pretrained(path)
        except OSError as e:
            raise SystemExit(
                f"[FA2D] DINOv3 config 를 찾지 못했습니다: {path}\n"
                f"  DINOv3 는 재배포 금지라 이 저장소에 없습니다. config.json 이 든 디렉토리를\n"
                f"  paths.local.env 의 PIERROTFR_FA2D_DINOV3_DIR 로 지정하세요.") from e
        self.inner = AutoModel.from_config(cfg)
        self.n_reg = int(getattr(cfg, "num_register_tokens", 0))
        self.patch = int(getattr(cfg, "patch_size", 16))
        self.embed_dim = int(cfg.hidden_size)
        if image_size % self.patch:
            raise ValueError(f"image_size={image_size} 가 patch {self.patch} 의 배수가 아니다")

    def forward(self, x: torch.Tensor):
        h = self.inner(pixel_values=x).last_hidden_state
        cls = h[:, 0]
        patch = h[:, 1 + self.n_reg:]
        b, n, c = patch.shape
        g = int(math.isqrt(int(n)))
        if g * g != int(n):
            raise ValueError(f"패치 수 {n} 이 정사각이 아니다")
        return patch.transpose(1, 2).reshape(b, c, g, g), cls


class ViTTinyBackbone(nn.Module):
    """ViT-T/16 → (patch (B,C,h,w), cls (B,C)).

    학습 저장소는 이 가중치의 mean=std=0.5 정규화를 ImageNet 상수로 바꾸려고 차이를
    **patch embedding conv 에 접어 넣었다.** 그 접힌 값은 체크포인트에 이미 들어 있으므로
    여기서 다시 접지 않는다 — 두 번 접으면 입력 분포가 통째로 어긋난다.
    """

    def __init__(self, image_size: int = 256, path: str = VIT_T16):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        cfg = AutoConfig.from_pretrained(path)
        self.inner = AutoModel.from_config(cfg)
        self.embed_dim = int(cfg.hidden_size)
        self.patch = int(cfg.patch_size)
        if image_size % self.patch:
            raise ValueError(f"image_size={image_size} 가 patch {self.patch} 의 배수가 아니다")

    def forward(self, x: torch.Tensor):
        h = self.inner(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state
        cls, patch = h[:, 0], h[:, 1:]
        b, n, c = patch.shape
        g = int(math.isqrt(int(n)))
        if g * g != int(n):
            raise ValueError(f"패치 수 {n} 이 정사각이 아니다")
        return patch.transpose(1, 2).reshape(b, c, g, g), cls


BACKBONES = {k: f"torchvision/{k}" for k in CNN_BACKBONES}
BACKBONES["dinov3-vitl16"] = "local DINOv3"
BACKBONES["vit-t16"] = VIT_T16


def recommended_norm(name: str):
    """입력 정규화 — 공개한 백본은 전부 ImageNet 통계다(vit-t16 은 conv 에 접혀 있다)."""
    return IMAGENET_NORM


def backbone_stride(name: str) -> int:
    return 16


def build_backbone(name: str, image_size: int = 224) -> nn.Module:
    if name == "dinov3-vitl16":
        return DINOv3Backbone(image_size)
    if name == "vit-t16":
        return ViTTinyBackbone(image_size)
    if name in CNN_BACKBONES:
        return CNNBackbone(name, image_size)
    raise SystemExit(
        f"[FA2D] backbone={name!r} 은 이 추론 저장소에 없습니다 — 가능: {sorted(BACKBONES)}\n"
        f"  DINOv2 · CLIP 계열은 학습 저장소가 FAS 레지스트리에서 빌려 쓰던 것이라 옮기지\n"
        f"  않았습니다. 공개한 교사·학생은 dinov3-vitl16 · vit-t16 입니다.")
