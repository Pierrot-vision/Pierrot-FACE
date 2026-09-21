"""융합 모듈과 예측 헤드.

원본 논문 식 (5) 는 `Q̂ = Q + ψ(Q, f̃_c)` — 쿼리가 스트림이고 이미지가 context 다.
그런데 **원본 코드는 반대다**: 이미지 토큰이 스트림이고 쿼리가 cross-attention 의
key/value 다 (`FusionModule.forward(content, queries)`).

코드가 맞다. 논문대로면 출력이 [B, 2K, D] = [B, 64, D] 라 cue map 을 정사각으로
펼 수 없다. 코드처럼 [B, 1+N, D] 여야 N=196 -> 14×14 가 성립한다.
여기서는 코드 쪽(= 실제로 동작하는 쪽)을 따른다.

출력 분할:
    [B, 1+N, D] -> t_cls  [B, 1, D]   -> 이진 분류기
                -> t_cue  [B, N, D]   -> cue generator (√N × √N 맵)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .qformer import FeedForward, MultiHeadAttention


class FusionBlock(nn.Module):
    """이미지 토큰을 스트림으로, content+style 쿼리를 context 로 융합."""

    def __init__(self, dim: int, context_dim: int, heads: int = 16, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.self_attn = MultiHeadAttention(dim, heads, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.cross_attn = MultiHeadAttention(dim, heads, dropout, context_dim=context_dim)
        self.ln3 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, dropout=dropout)

    def forward(self, tokens: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.self_attn(self.ln1(tokens))
        tokens = tokens + self.cross_attn(self.ln2(tokens), context=queries)
        # ⚠ 원본은 FFN 에 residual 이 없다(`out = self.ffn(content)`). 표준 트랜스포머
        #   블록과 다르며 깊게 쌓으면 신호가 죽는다. 여기서는 residual 을 넣는다.
        return tokens + self.ffn(self.ln3(tokens))


class FusionModule(nn.Module):
    def __init__(self, dim: int, context_dim: int, heads: int = 16,
                 depth: int = 1, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            FusionBlock(dim, context_dim, heads, dropout) for _ in range(depth)
        ])

    def forward(self, tokens, queries):
        for blk in self.blocks:
            tokens = blk(tokens, queries)
        return tokens


class NormedLinear(nn.Module):
    """열 정규화 선형 분류기 (SSDG 계열 규약).

    가중치를 매 forward 마다 열 방향 L2 정규화한다. 클래스 벡터의 노름 차이가
    결정 경계를 좌우하지 않게 해, 도메인 간 스케일 변화에 덜 민감해진다.
    """

    def __init__(self, in_dim: int, n_classes: int = 2, normalize: bool = True):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)
        self.normalize = normalize
        nn.init.trunc_normal_(self.fc.weight, std=0.02)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        w = self.fc.weight
        if self.normalize:
            w = w / w.norm(dim=0, keepdim=True).clamp(min=1e-8)
        return torch.nn.functional.linear(x, w, self.fc.bias)


class CueGenerator(nn.Module):
    """위조 흔적 맵 생성기.

    ⚠ 이건 정식 세그멘테이션이 아니다. CelebA-Spoof 에 위조 영역 마스크가 없어
      타깃을 **이진 라벨의 공간적 broadcast** (fake=1 전면, live=0 전면)로 만든다.
      공간적으로 균일한 출력을 강요받은 네트워크가 판별에 유리한 국소 증거로
      수렴하는 부수 효과를 노린 보조 감독이다. 해석용으로만 쓸 것 —
      정량 평가 대상이 아니다.
    """

    def __init__(self, dim: int, hidden: int = 32, out_ch: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_ch, 3, 1, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens [B, N, D] (N 은 정사각) -> cue map [B, out_ch, √N, √N]."""
        B, N, D = tokens.shape
        side = int(round(N ** 0.5))
        if side * side != N:
            raise ValueError(f"cue 토큰 수 {N} 이 정사각이 아닙니다")
        grid = tokens.transpose(1, 2).reshape(B, D, side, side)
        return self.net(grid)
