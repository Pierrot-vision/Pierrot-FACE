"""Q-Former — BLIP-2 계열 쿼리 트랜스포머의 자립 구현.

원본 InstructFLIP 은 `lavis.models.blip2_models.Qformer` 를 쓴다. LAVIS 는
저자 fork(`kunkunlin1221/LAVIS`)에 묶여 있고 낡은 pin 을 끌고 와 설치가 취약하다.
여기서는 같은 구조를 순수 PyTorch 로 옮겨 그 의존을 제거했다.

구조 (BLIP-2 / InstructBLIP 와 동일한 계약):
    입력  learnable query [B, K, D]  +  instruction 토큰 [B, T, D]
    각 블록:
        self-attn   : [query ; text] 전체에 대해 (query 가 지시문을 읽는다)
        cross-attn  : query 만 이미지 특징을 본다 (cross_attn_freq 마다)
        ffn
    출력  query 부분만 [B, K, D]

⚠ cross-attention 을 query 에만 거는 게 핵심이다. 텍스트 토큰까지 이미지를 보면
  지시문이 이미지에 오염되어 "무엇을 물었는가"가 흐려진다. BLIP-2 원 설계도 같다.

⚠ 텍스트 임베딩은 사전학습 BERT 를 쓰지 않는다. 이 태스크의 지시문은 고정된
  객관식 몇 종뿐이라 대규모 어휘 임베딩(23M 파라미터)이 낭비다. 대신 작은
  전용 임베딩을 두고 토크나이저만 BERT 것을 빌린다.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """self / cross 겸용. context=None 이면 self-attention."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0,
                 context_dim: int | None = None):
        super().__init__()
        assert dim % heads == 0, f"dim({dim}) 이 heads({heads}) 로 나뉘지 않습니다"
        self.h = heads
        self.dh = dim // heads
        context_dim = context_dim or dim

        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(context_dim, dim, bias=True)
        self.v = nn.Linear(context_dim, dim, bias=True)
        self.o = nn.Linear(dim, dim, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        ctx = x if context is None else context
        B, N, _ = x.shape
        M = ctx.shape[1]

        q = self.q(x).view(B, N, self.h, self.dh).transpose(1, 2)
        k = self.k(ctx).view(B, M, self.h, self.dh).transpose(1, 2)
        v = self.v(ctx).view(B, M, self.h, self.dh).transpose(1, 2)

        attn_mask = None
        if mask is not None:
            # mask: [B, M] (1=유효) -> [B, 1, 1, M] bool
            attn_mask = mask[:, None, None, :].bool()

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                             dropout_p=self.drop.p if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, N, -1)
        return self.drop(self.o(out))


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class QFormerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, context_dim: int,
                 use_cross: bool, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, heads, dropout)

        self.use_cross = use_cross
        if use_cross:
            self.ln_cross = nn.LayerNorm(dim)
            self.cross = MultiHeadAttention(dim, heads, dropout, context_dim=context_dim)

        self.ln2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, dropout=dropout)

    def forward(self, x, image_feats, n_query, text_mask=None):
        # self-attn: query + text 전체
        x = x + self.attn(self.ln1(x), mask=text_mask)

        # cross-attn: query 부분만 이미지를 본다
        if self.use_cross:
            q = x[:, :n_query]
            q = q + self.cross(self.ln_cross(q), context=image_feats)
            x = torch.cat([q, x[:, n_query:]], dim=1)

        return x + self.ffn(self.ln2(x))


class QFormer(nn.Module):
    """지시문 조건부 쿼리 트랜스포머.

    Args:
        dim         쿼리/텍스트 임베딩 차원
        context_dim 이미지 특징 차원 (백본 dim)
        n_query     학습형 쿼리 개수 (원본 32)
        depth       블록 수 (원본 2)
        heads       어텐션 헤드 (원본 16)
        cross_freq  몇 블록마다 cross-attention 을 둘지 (원본 2)
        vocab       텍스트 임베딩 어휘 크기
    """

    def __init__(self, dim: int = 768, context_dim: int = 768, n_query: int = 32,
                 depth: int = 2, heads: int = 16, cross_freq: int = 2,
                 vocab: int = 30522, max_text: int = 128, dropout: float = 0.0):
        super().__init__()
        self.n_query = n_query
        self.dim = dim

        self.query_tokens = nn.Parameter(torch.zeros(1, n_query, dim))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

        self.text_embed = nn.Embedding(vocab, dim)
        self.text_pos = nn.Parameter(torch.zeros(1, max_text, dim))
        nn.init.trunc_normal_(self.text_pos, std=0.02)
        self.max_text = max_text

        # 이미지 특징 차원이 다르면 cross-attn 안에서 투영된다 (context_dim 인자)
        self.blocks = nn.ModuleList([
            QFormerBlock(dim, heads, context_dim,
                         use_cross=(i % cross_freq == 0), dropout=dropout)
            for i in range(depth)
        ])
        self.ln_out = nn.LayerNorm(dim)

    def forward(self, image_feats: torch.Tensor,
                text_ids: torch.Tensor | None = None,
                text_mask: torch.Tensor | None = None) -> torch.Tensor:
        """image_feats [B, M, context_dim] -> query 출력 [B, n_query, dim]."""
        B = image_feats.shape[0]
        x = self.query_tokens.expand(B, -1, -1)

        full_mask = None
        if text_ids is not None:
            T = text_ids.shape[1]
            if T > self.max_text:
                text_ids, T = text_ids[:, :self.max_text], self.max_text
                if text_mask is not None:
                    text_mask = text_mask[:, :self.max_text]
            t = self.text_embed(text_ids) + self.text_pos[:, :T]
            x = torch.cat([x, t], dim=1)
            if text_mask is not None:
                q_mask = torch.ones(B, self.n_query, device=x.device, dtype=text_mask.dtype)
                full_mask = torch.cat([q_mask, text_mask], dim=1)

        for blk in self.blocks:
            x = blk(x, image_feats, self.n_query, text_mask=full_mask)

        return self.ln_out(x[:, :self.n_query])
