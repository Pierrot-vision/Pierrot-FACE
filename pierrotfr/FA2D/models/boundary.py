"""경계 감독 헤드 (Look at Boundary) — 추론 경로만.

memory 토큰에서 경계 히트맵 K 장을 내고 **다시 memory 에 더해** 점 쿼리가 그것을
보게 한다("look at" 부분). 규약별 경계 정의가 있는 것만(wflw98 · ibug68).

학습 저장소의 같은 파일에서 `boundary_targets` · `boundary_weights`(감독용)를 뺐다.
`BOUNDARY_GROUPS` 는 남긴다 — 경계 헤드의 채널 수를 정하고, 데모가 윤곽선을 그릴 때
같은 점 순서를 쓴다.
"""
from __future__ import annotations

import torch
from torch import nn

# 각 경계선을 이루는 점 순서. 닫힌 곡선은 끝에 시작점을 다시 넣는다.
BOUNDARY_GROUPS = {
    "wflw98": [
        list(range(0, 33)),                       # 얼굴 윤곽
        list(range(33, 38)), [37, 38, 39, 40, 41, 33],     # 왼눈썹 상/하
        list(range(42, 47)), [46, 47, 48, 49, 50, 42],     # 오른눈썹 상/하
        list(range(51, 55)),                      # 콧대
        list(range(55, 60)),                      # 코 아래
        list(range(60, 68)) + [60],               # 왼눈(닫힘)
        list(range(68, 76)) + [68],               # 오른눈(닫힘)
        list(range(76, 83)), [82, 83, 84, 85, 86, 87, 76],  # 바깥 입술 상/하
        list(range(88, 93)), [92, 93, 94, 95, 88],          # 안쪽 입술 상/하
    ],
    "ibug68": [
        list(range(0, 17)),                       # 얼굴 윤곽
        list(range(17, 22)), list(range(22, 27)),  # 눈썹
        list(range(27, 31)), list(range(31, 36)),  # 콧대 · 코 아래
        list(range(36, 42)) + [36], list(range(42, 48)) + [42],   # 눈
        list(range(48, 55)), [54, 55, 56, 57, 58, 59, 48],        # 바깥 입술
        list(range(60, 65)), [64, 65, 66, 67, 60],                # 안쪽 입술
    ],
}
N_BOUNDARY = {k: len(v) for k, v in BOUNDARY_GROUPS.items()}


class BoundaryHead(nn.Module):
    """memory → 경계 히트맵 K 장, 그리고 그것을 memory 에 되먹인다."""

    def __init__(self, d_model: int, n_boundary: int):
        super().__init__()
        self.to_map = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                    nn.Linear(d_model, n_boundary))
        self.fuse = nn.Linear(n_boundary, d_model)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)

    def forward(self, mem, h, w):
        """mem (B,hw,d) → (logits (B,K,h,w), 보강된 mem)"""
        logits = self.to_map(mem)                              # (B,hw,K)
        mem = mem + self.fuse(torch.sigmoid(logits))           # "look at boundary"
        b, hw, k = logits.shape
        return logits.transpose(1, 2).reshape(b, k, h, w), mem
