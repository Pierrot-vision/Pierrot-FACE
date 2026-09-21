"""시각 백본 — 체크포인트가 밝힌 이름으로 **구조만** 만든다.

학습 저장소는 `from_pretrained` 로 사전학습 가중치를 받아 fine-tune 했다. 추론에서는
그 가중치가 전부 체크포인트에 들어 있으므로 HF 에서는 **config 만** 읽는다
(`from_config`) — 수백 MB 사전학습 가중치를 받을 필요가 없다.

모든 백본은 같은 계약을 지킨다:

    forward(x) -> (tokens, layer_feats)
        tokens      [B, 1+N, D]   cls + 패치 토큰 (content 특징)
        layer_feats list of [B, L_i, D]  각 레이어 hidden (style 통계용)
"""
from __future__ import annotations

import torch
import torch.nn as nn

# 이름 -> (HF 모델 id, 권장 정규화)
BACKBONES = {
    "clip-vit-b16":     ("openai/clip-vit-base-patch16", "clip"),
    "clip-vit-l14":     ("openai/clip-vit-large-patch14", "clip"),
    "dinov2-s14":       ("facebook/dinov2-small", "imagenet"),
    "dinov2-b14":       ("facebook/dinov2-base", "imagenet"),
    "dinov2-b14-reg":   ("facebook/dinov2-with-registers-base", "imagenet"),
    "dinov2-l14-reg":   ("facebook/dinov2-with-registers-large", "imagenet"),
}


class CLIPVisionBackbone(nn.Module):
    def __init__(self, model_id: str, image_size: int = 224):
        super().__init__()
        from transformers import CLIPVisionConfig, CLIPVisionModel

        cfg = CLIPVisionConfig.from_pretrained(model_id)
        self.model = CLIPVisionModel(cfg)
        self.dim = cfg.hidden_size
        self.patch = cfg.patch_size
        self.n_patches = (image_size // cfg.patch_size) ** 2
        self.n_layers = cfg.num_hidden_layers

    def forward(self, x: torch.Tensor):
        out = self.model(pixel_values=x, output_hidden_states=True)
        # hidden_states[0] 은 임베딩 직후라 트랜스포머 블록 출력이 아니다 — 제외한다
        layer_feats = list(out.hidden_states[1:])
        # 원저자 코드(OpenAI clip `ln_post`)처럼 post_layernorm 을 **전체 토큰**에 건다
        tokens = self.model.vision_model.post_layernorm(out.last_hidden_state)
        return tokens, layer_feats


class DINOv2Backbone(nn.Module):
    """registers 변종은 출력이 [cls, reg*4, patch*N] 이라 레지스터 토큰을 잘라낸다."""

    def __init__(self, model_id: str, image_size: int = 224):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        cfg = AutoConfig.from_pretrained(model_id)
        self.model = AutoModel.from_config(cfg)
        self.dim = cfg.hidden_size
        self.patch = cfg.patch_size
        self.n_patches = (image_size // cfg.patch_size) ** 2
        self.n_layers = cfg.num_hidden_layers
        self.n_registers = int(getattr(cfg, "num_register_tokens", 0) or 0)

    def _strip_registers(self, t: torch.Tensor) -> torch.Tensor:
        if self.n_registers == 0:
            return t
        return torch.cat([t[:, :1], t[:, 1 + self.n_registers:]], dim=1)

    def forward(self, x: torch.Tensor):
        out = self.model(pixel_values=x, output_hidden_states=True)
        tokens = self._strip_registers(out.last_hidden_state)
        layer_feats = [self._strip_registers(h) for h in out.hidden_states[1:]]
        return tokens, layer_feats


def build_backbone(name: str, image_size: int = 224) -> nn.Module:
    if name not in BACKBONES:
        raise ValueError(f"backbone={name!r} — 가능: {sorted(BACKBONES)}")
    model_id, _ = BACKBONES[name]
    bb = (CLIPVisionBackbone if name.startswith("clip") else DINOv2Backbone)(model_id, image_size)
    side = int(round(bb.n_patches ** 0.5))
    if side * side != bb.n_patches:
        raise ValueError(f"backbone={name}: 패치 수 {bb.n_patches} 가 정사각이 아닙니다 "
                         f"(image_size={image_size}, patch={bb.patch})")
    return bb
