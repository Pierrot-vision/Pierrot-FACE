"""점 쿼리 트랜스포머 디코더 — FA2D 본체.

HRFFA 의 설계를 따른다(MIT). 학습 저장소의 같은 파일에서 **구조는 한 글자도 바꾸지
않았다** — 다르면 같은 가중치가 다른 좌표를 낸다. 옮기며 뺀 것은 손실 import 뿐이다.

요지:

  - **scheme 별 학습형 쿼리**: 디코더·헤드는 공유하고 쿼리 임베딩만 규약별로 둔다.
    한 모델이 68/98/29 를 모두 내고, 점 수 확장이 쿼리 추가만으로 끝난다.
  - **좌표는 sigmoid 없는 선형 출력**: 크롭 밖으로 나간 점(가시성 0)도 좌표를
    가져야 하므로 [0,1] 로 누르면 안 된다. 정규화 기준은 크롭 변 길이.
  - 점마다 **가시성 3-class**(화면밖/가림/보임).
  - 자세는 6D 회전 + Roll biternion 보조 헤드.

Lab 확장:
  - `temporal`: 이전 프레임 디코더 토큰을 쿼리에 주입한다. `prev_proj` 는
    **zero-init 이고 bias 가 없다** — 얹은 직후 출력이 동일하고, ONNX 에서 첫
    프레임에 0 을 넣는 것이 `prev_tokens=None` 과 정확히 등가가 된다.
  - `use_state`: 눈감김·입벌림 헤드. 전역 풀링이 아니라 **해당 부위 토큰만** 모아
    읽는다(눈감김은 국소 성질이라 전역 풀링하면 묻힌다).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..facestate import STATE_POINTS, state_token_index
from .backbones import build_backbone
from .boundary import BOUNDARY_GROUPS, N_BOUNDARY, BoundaryHead
from .heatmap import QueryHeatmap
from .refine import LocalRefine

SCHEMES = {"ibug68": 68, "wflw98": 98, "cofw29": 29, "lapa106": 106}


def rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """6D 회전 표현 (B,6) → 회전행렬 (B,3,3) (Gram-Schmidt). 짐벌락을 피한다.

    학습 저장소에서는 `losses.py` 에 있었다 — 추론 경로가 쓰는 유일한 함수라 여기로 옮긴다.
    """
    a1, a2 = x[:, :3], x[:, 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    return torch.stack([b1, b2, torch.cross(b1, b2, dim=-1)], dim=-1)


def sincos_pos_embed_2d(d_model: int, h: int, w: int, device) -> torch.Tensor:
    assert d_model % 4 == 0
    quarter = d_model // 4
    omega = 1.0 / (10000 ** (torch.arange(quarter, device=device) / quarter))
    ys, xs = torch.meshgrid(torch.arange(h, device=device),
                            torch.arange(w, device=device), indexing="ij")
    out = []
    for pos in (xs, ys):
        ang = pos.reshape(-1, 1).float() * omega[None]
        out += [torch.sin(ang), torch.cos(ang)]
    return torch.cat(out, dim=1)


class PointQueryNet(nn.Module):
    def __init__(self, backbone: str = "dinov2-b14-reg", image_size: int = 224,
                 d_model: int = 256, dec_layers: int = 4, n_heads: int = 8,
                 ffn_dim: int = 1024, schemes=None,
                 temporal: bool = False, use_state: bool = False,
                 refine: bool = False, refine_stem: bool = True, refine_k: int = 7,
                 heatmap: bool = False, hm_sigma: float = 1.0,
                 hm_offset: bool = False, hm4_sigma: float = 1.5,
                 boundary: bool = False,
                 **_ignored):
        super().__init__()
        self.schemes = dict(schemes or SCHEMES)
        self.backbone = build_backbone(backbone, image_size)
        c = self.backbone.embed_dim

        self.input_proj = nn.Linear(c, d_model)
        self.cls_proj = nn.Linear(c, d_model)
        self.queries = nn.ParameterDict({
            name: nn.Parameter(torch.randn(n, d_model) * 0.02)
            for name, n in self.schemes.items()})
        layer = nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward=ffn_dim,
                                           dropout=0.0, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, dec_layers, norm=nn.LayerNorm(d_model))
        self.coord_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                        nn.Linear(d_model, 2))
        self.vis_head = nn.Linear(d_model, 3)
        self.pose_head = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(),
                                       nn.Linear(d_model, 8))
        nn.init.zeros_(self.pose_head[-1].bias)
        with torch.no_grad():                       # 6D 회전 초기값을 단위행렬 근방으로
            self.pose_head[-1].bias[:6] = torch.tensor([1., 0., 0., 0., 1., 0.])

        self.temporal = bool(temporal)
        self.prev_proj = None
        if self.temporal:
            # bias 를 두지 않는다 — 0 입력이 prev_tokens=None 과 등가여야 한다
            self.prev_proj = nn.Linear(d_model, d_model, bias=False)
            nn.init.zeros_(self.prev_proj.weight)

        # 국소 보정 — 고해상도 특징에서 초기 좌표 주변만 읽어 Δ 를 더한다(Phase 10 아암 B/C).
        #   0 초기화라 추가 직후 출력이 기존과 같다. refine_stem=False 면 DINO 특징만
        #   보간해 쓴다(B: 새 관측 정보 없음), True 면 입력에서 얻은 stride 4 특징을
        #   결합한다(C: 실제 고해상도 세부).
        # 쿼리 기반 히트맵 + soft-argmax (Phase 11 아암 D) — 공간 감독을 추가한다.
        #   D-ViT 는 32x32 히트맵으로 3.75 를 낸다. 우리 특징 격자와 같은 해상도이므로
        #   격자가 아니라 **감독 방식**이 다르다는 것이 이 아암의 가설이다.
        self.heatmap = QueryHeatmap(d_model, sigma_cells=hm_sigma) if heatmap else None

        # 경계 감독(Look at Boundary) — 윤곽 오차의 66% 가 접선(미끄러짐)이라
        #   국소 특징으로는 못 고친다. 경계선을 직접 예측·감독하고 memory 에 되먹인다.
        #   ⚠ 규약별 경계 정의가 있는 것만(wflw98·ibug68). 헤드는 규약 간 공유하되
        #     출력 채널은 최대 개수로 두고 규약별 앞쪽 K 개만 쓴다.
        self.boundary = None
        if boundary:
            self.boundary = BoundaryHead(d_model, max(N_BOUNDARY.values()))

        # stride 4 히트맵 + x/y 오프셋 (Peppa 학생 구조) — 좌표를 stride 16 에서 바로
        #   회귀하는 대신 고해상도 격자에서 만든다.
        self.hm_offset = None
        if hm_offset:
            from .hm_offset import HeatmapOffsetHead
            self.hm_offset = HeatmapOffsetHead(d_model, c)

        self.refine = None
        if refine:
            self.refine = LocalRefine(d_model, c, k=refine_k, use_stem=refine_stem)

        self.use_state = bool(use_state)
        self.state_head = None
        if self.use_state:
            self.state_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(),
                                            nn.Linear(d_model // 2, 1))
        self._pos_cache = {}

    def _pos(self, h, w, d, device):
        key = (h, w, d, str(device))
        if key not in self._pos_cache:
            self._pos_cache[key] = sincos_pos_embed_2d(d, h, w, device)
        return self._pos_cache[key]

    def forward(self, images: torch.Tensor, scheme: str, prev_tokens=None) -> dict:
        patch, cls = self.backbone(images)
        b, c, h, w = patch.shape
        mem = self.input_proj(patch.reshape(b, c, h * w).transpose(1, 2))
        mem = mem + self._pos(h, w, mem.shape[-1], mem.device)[None]

        bnd_logits = None
        if self.boundary is not None and scheme in BOUNDARY_GROUPS:
            k = len(BOUNDARY_GROUPS[scheme])
            bnd_logits, mem = self.boundary(mem, h, w)
            bnd_logits = bnd_logits[:, :k]

        q = self.queries[scheme][None].expand(b, -1, -1)
        if self.prev_proj is not None and prev_tokens is not None:
            q = q + self.prev_proj(prev_tokens)
        dec = self.decoder(q, mem)

        pose = self.pose_head(torch.cat([self.cls_proj(cls), mem.mean(dim=1)], dim=-1))
        coarse = self.coord_head(dec)
        points = coarse
        if self.refine is not None:
            delta = self.refine(images, patch, dec, coarse.detach())
            # ⚠ 크롭 밖 초기 좌표는 보정하지 않는다 — 주변 특징이 padding(0) 이라
            #   Δ 가 의미를 갖지 못한다. 그 점은 기존 회귀값을 그대로 쓴다.
            inside = ((coarse >= 0.0) & (coarse < 1.0)).all(dim=-1, keepdim=True)
            # 작은 Δ가 BF16 좌표 덧셈에서 사라지지 않게 최종 합은 FP32로 한다.
            points = coarse.float() + torch.where(inside, delta.float(), 0.0)
        hm_logits = hm_pts = None
        if self.heatmap is not None:
            hm_logits, hm_pts = self.heatmap(dec, mem, h, w)
            # ⚠ 히트맵은 크롭 안에서만 정의된다(격자 밖에 질량을 둘 수 없다).
            #   크롭 밖 점은 기존 회귀값을 그대로 쓴다.
            inside_hm = ((coarse > 0.0) & (coarse < 1.0)).all(dim=-1, keepdim=True)
            points = torch.where(inside_hm, hm_pts.to(points.dtype), points)
        hm4 = None
        if self.hm_offset is not None:
            hm4 = self.hm_offset(images, patch, dec)
            inside4 = ((coarse > 0.0) & (coarse < 1.0)).all(dim=-1, keepdim=True)
            points = torch.where(inside4, hm4["points"].to(points.dtype), points)
        out = {"points": points,
               "vis_logits": self.vis_head(dec),
               "rot": rot6d_to_matrix(pose[:, :6]),
               "roll_bit": nn.functional.normalize(pose[:, 6:], dim=-1),
               "dec_tokens": dec, "memory": mem}
        if bnd_logits is not None:
            out["bnd_logits"] = bnd_logits
        if self.heatmap is not None:
            out["hm_logits"] = hm_logits
            out["hm_points"] = hm_pts
        if hm4 is not None:
            out |= {k: v for k, v in hm4.items() if k != "points"}
            out["hm4_points"] = hm4["points"]
        if self.refine is not None or hm4 is not None:
            out["coarse_points"] = coarse
        # 상태 유도 인덱스가 정의된 규약에서만 상태 헤드를 태운다(lapa106 등은 미정의)
        if self.state_head is not None and scheme in STATE_POINTS:
            groups = state_token_index(scheme)
            pooled = torch.stack([dec[:, idx].mean(dim=1) for idx in groups], dim=1)
            out["state_logits"] = self.state_head(pooled).squeeze(-1)
        return out

    def inference_modules(self):
        """배포 시 실제로 필요한 모듈(자세 헤드는 학습 안 하면 제외된다)."""
        mods = [self.backbone, self.input_proj, self.decoder,
                self.coord_head, self.vis_head]
        if self.refine is not None:
            mods.append(self.refine)
        if self.heatmap is not None:
            mods.append(self.heatmap)
        if self.hm_offset is not None:
            mods.append(self.hm_offset)
        if self.boundary is not None:
            mods.append(self.boundary)
        if self.state_head is not None:
            mods.append(self.state_head)
        return mods
