"""InstructFLIP — 추론 경로만.

원논문: InstructFLIP (ACM MM 2025, arXiv:2507.12060)

    학습: 백본 -> Q-Former×2 -> (FLAN-T5 로 instruction tuning) + 융합 -> 분류
    추론: 백본 -> Q-Former×2 -> 융합 -> 분류        ← LLM 을 아예 타지 않는다

LLM 은 표현을 형성하는 **감독 신호**일 뿐이라 여기에는 없다. 체크포인트의
`llm.` · `llm_proj.` 키는 로드할 때 버린다 (pierrotfr/FAS/infer.py).

content : "어떤 위조 유형인가"  — spoof 의미 자체
style   : "조명/환경/카메라는"  — spoof 와 무관한 nuisance
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import build_backbone
from .connectors import build_connector
from .heads import CueGenerator, FusionModule, NormedLinear


class InstructFLIP(nn.Module):
    def __init__(
        self,
        backbone: str = "clip-vit-b16",
        image_size: int = 224,
        query_module: str = "qformer",   # 'qformer' | 'attn_pool' | 'connector'
        n_query: int = 32,
        qformer_dim: int = 768,
        qformer_depth: int = 2,
        qformer_heads: int = 16,
        fusion_depth: int = 1,
        fusion_heads: int = 16,
        use_cue: bool = True,
        vocab: int = 30522,
        dropout: float = 0.1,
        **_,                             # 학습 전용 인자(llm_name · use_llm · cue_noise)는 무시
    ):
        super().__init__()
        self.backbone = build_backbone(backbone, image_size)
        D = self.backbone.dim

        ckw = dict(dim=qformer_dim, context_dim=D, n_query=n_query,
                   depth=qformer_depth, heads=qformer_heads,
                   vocab=vocab, dropout=dropout)
        self.content_qformer = build_connector(query_module, **ckw)
        self.style_qformer = build_connector(query_module, **ckw)

        # 융합: 이미지 토큰이 스트림, content+style 쿼리가 context
        self.fusion = FusionModule(D, qformer_dim, fusion_heads, fusion_depth, dropout)
        self.classifier = NormedLinear(D, 2)
        self.cue_generator = CueGenerator(D) if use_cue else None

    def encode(self, images: torch.Tensor):
        """이미지 -> (content 토큰 [B, 1+N, D], style 통계 [B, 2L, D] = 레이어별 mean·std)."""
        tokens, layer_feats = self.backbone(images)
        stats = []
        for feat in layer_feats:
            stats.append(feat.mean(dim=1))
            stats.append(feat.std(dim=1))
        return tokens, torch.stack(stats, dim=1)

    @torch.no_grad()
    def predict(self, batch: dict) -> dict:
        """spoof 확률 · margin(log-odds) · cue 맵.

        batch: image [B,3,H,W] + content_ids/mask · style_ids/mask (고정 지시문 토큰)
        """
        content_tok, style_stat = self.encode(batch["image"])
        cq = self.content_qformer(content_tok, batch.get("content_ids"), batch.get("content_mask"))
        sq = self.style_qformer(style_stat, batch.get("style_ids"), batch.get("style_mask"))

        # ⚠ style **특징**(style_stat)은 융합에 넣지 않는다 — style **쿼리**만 들어간다
        fused = self.fusion(content_tok, torch.cat([cq, sq], dim=1))    # [B, 1+N, D]
        logits = self.classifier(fused[:, 0])
        cue = self.cue_generator(fused[:, 1:]) if self.cue_generator is not None else None
        lg = logits.float()
        # margin = spoof log-odds. 평가는 이걸로 한다 — 확률은 포화해 동점이 생긴다.
        return {"score": F.softmax(lg, dim=-1)[:, 1], "margin": lg[:, 1] - lg[:, 0],
                "logits": logits, "cue": cue}
