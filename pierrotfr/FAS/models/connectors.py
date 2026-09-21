"""시각 특징 → 쿼리 표현 변환기 (connector).

백본이 뽑은 특징을 고정 길이 표현으로 바꾸는 부분을 **실험 축**으로 만든다.
원논문은 Q-Former 로 고정돼 있고 그 대안을 재본 적이 없다.

세 방식은 "무엇이 content/style 분리를 만드는가"에서 갈린다:

    qformer     지시문 + 학습형 쿼리. **질문**이 무엇을 볼지 게이팅한다.
                → BLIP-2 계열. InstructFLIP 원본.
    attn_pool   학습형 쿼리만, 지시문 미입력. 게이팅 없이 쿼리가 고정 관점을 갖는다.
                → qformer 와의 차이 = **지시문 게이팅의 기여**
    connector   MLP 투영만. 쿼리도 지시문도 없다.
                → LLaVA / MiniCPM-V(FaceCoT) 계열의 그 connector.
                → attn_pool 과의 차이 = **학습형 쿼리의 기여**

⚠ connector 를 쓰면 content/style 분리가 **지시문이 아니라 입력**에서만 온다
  (패치 토큰 vs 레이어별 통계). 분리 자체는 남지만 근거가 달라지므로,
  이 조건에서 성능이 유지되면 "지시문 게이팅은 불필요했다"는 뜻이 된다.

공통 계약:
    forward(feats [B, M, D_in], text_ids=None, text_mask=None) -> [B, K, D_out]
    .n_out   출력 토큰 수 K
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .qformer import FeedForward, MultiHeadAttention, QFormer


class QFormerConnector(nn.Module):
    """지시문 조건부 쿼리 트랜스포머 (원본 방식)."""

    def __init__(self, dim: int, context_dim: int, n_query: int = 32,
                 depth: int = 2, heads: int = 16, vocab: int = 30522,
                 dropout: float = 0.0, **_):
        super().__init__()
        self.qformer = QFormer(dim, context_dim, n_query, depth, heads,
                               vocab=vocab, dropout=dropout)
        self.n_out = n_query

    def forward(self, feats, text_ids=None, text_mask=None):
        return self.qformer(feats, text_ids, text_mask)


class AttnPoolConnector(nn.Module):
    """학습형 쿼리 + cross-attention. **지시문을 받지 않는다.**

    qformer 와 파라미터 규모를 비슷하게 유지하되 텍스트 경로만 제거해,
    "지시문 게이팅이 없으면 얼마나 나빠지나"를 재는 대조군이다.
    """

    def __init__(self, dim: int, context_dim: int, n_query: int = 32,
                 depth: int = 2, heads: int = 16, dropout: float = 0.0, **_):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.zeros(1, n_query, dim))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

        self.blocks = nn.ModuleList()
        for _i in range(depth):
            self.blocks.append(nn.ModuleDict({
                "ln_q": nn.LayerNorm(dim),
                "self_attn": MultiHeadAttention(dim, heads, dropout),
                "ln_x": nn.LayerNorm(dim),
                "cross": MultiHeadAttention(dim, heads, dropout, context_dim=context_dim),
                "ln_f": nn.LayerNorm(dim),
                "ffn": FeedForward(dim, dropout=dropout),
            }))
        self.ln_out = nn.LayerNorm(dim)
        self.n_out = n_query

    def forward(self, feats, text_ids=None, text_mask=None):
        # text_* 는 계약 유지를 위해 받되 **의도적으로 무시**한다
        x = self.query_tokens.expand(feats.shape[0], -1, -1)
        for b in self.blocks:
            x = x + b["self_attn"](b["ln_q"](x))
            x = x + b["cross"](b["ln_x"](x), context=feats)
            x = x + b["ffn"](b["ln_f"](x))
        return self.ln_out(x)


class MLPConnector(nn.Module):
    """LLaVA / MiniCPM-V 계열 connector — 투영만 한다.

    쿼리도 지시문도 없다. 시각 토큰을 그대로 LLM 토큰 공간으로 옮기는 방식이며
    FaceCoT 가 쓰는 MiniCPM-V 의 connector 가 이 계열이다.

    Args:
        pool  'adaptive' = 출력 토큰을 n_query 로 맞춘다 (Q-Former 와 토큰 수를
                           통제해 비교하려면 이쪽)
              'none'     = 입력 토큰 수를 그대로 유지 (LLaVA 원형에 가깝다.
                           단 content 197 vs style 24 로 길이가 갈린다)
        depth 1 = 단일 Linear, 2 이상 = GELU 를 낀 MLP (LLaVA-1.5 는 2층)
    """

    def __init__(self, dim: int, context_dim: int, n_query: int = 32,
                 depth: int = 2, pool: str = "adaptive", dropout: float = 0.0, **_):
        super().__init__()
        layers: list[nn.Module] = []
        d_in = context_dim
        for i in range(max(depth, 1)):
            last = (i == max(depth, 1) - 1)
            layers.append(nn.Linear(d_in, dim))
            if not last:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
            d_in = dim
        self.proj = nn.Sequential(*layers)
        self.ln_out = nn.LayerNorm(dim)

        if pool not in ("adaptive", "none"):
            raise ValueError(f"pool={pool!r} — 'adaptive' | 'none'")
        self.pool = pool
        self.n_query = n_query
        self.n_out = n_query if pool == "adaptive" else -1   # -1 = 입력에 따라 가변

    def forward(self, feats, text_ids=None, text_mask=None):
        x = self.proj(feats)                       # [B, M, dim]
        if self.pool == "adaptive" and x.shape[1] != self.n_query:
            # [B, M, D] -> [B, D, M] -> pool -> [B, n_query, D]
            x = torch.nn.functional.adaptive_avg_pool1d(
                x.transpose(1, 2), self.n_query).transpose(1, 2)
        return self.ln_out(x)


CONNECTORS = {
    "qformer": QFormerConnector,
    "attn_pool": AttnPoolConnector,
    "connector": MLPConnector,
}


def build_connector(name: str, **kwargs) -> nn.Module:
    if name not in CONNECTORS:
        raise ValueError(f"query_module={name!r} — 가능: {sorted(CONNECTORS)}")
    return CONNECTORS[name](**kwargs)


def uses_instruction(name: str) -> bool:
    """이 방식이 지시문을 실제로 읽는가 — 로그·검증에 쓴다."""
    return name == "qformer"
