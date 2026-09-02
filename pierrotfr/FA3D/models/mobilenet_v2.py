"""MobileNet-V2 백본 + 이중 헤드 — SynergyNet(3DV 2021) 이 쓰는 백본.

3DDFA_V2 의 MobileNet-V1 보다 **작으면서 더 정확하다**:

    MobileNet-V1 (3DDFA_V2)   3.27M   177M MACs   AFLW2000-3D 3.51
    MobileNet-V2 (SynergyNet) 2.30M   ~92M MACs   AFLW2000-3D 3.41

두 논문의 학습 데이터·입력 크기·회귀 대상(62-d)이 같으므로 백본만 갈아끼워도
비교가 성립한다. 다만 SynergyNet 의 3.41 은 백본 덕분만이 아니라 순환 구조
(MLP_for/MLP_rev) 기여가 섞여 있다 — **백본 단독 기여를 보려면 여기서 재야 한다.**

⚠ 헤드 분리 (`split_head=True`)
    SynergyNet 은 62-d 를 한 FC 로 뽑지 않고 **ori(12) / shape(40) / exp(10)** 세
    갈래로 나눈다. 세 그룹이 물리적으로 다른 양이고(상사변환 / 신원 / 표정) 스케일도
    다르므로(§2.5) 분리하면 각 헤드가 자기 스케일에 맞춰 학습된다.
    기본값을 True 로 둬 SynergyNet 과 같은 조건에서 출발한다.
    `split_head=False` 면 3DDFA_V2 처럼 62-d 단일 FC 다.

⚠ dropout 0.2 도 SynergyNet 설정이다. 3DDFA_V2 의 V1 백본에는 dropout 이 없다.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _features(width_mult: float, imagenet: bool):
    """torchvision 의 MobileNetV2 특징 추출부만 떼어 온다.

    150줄짜리 InvertedResidual 을 복제하면 torchvision 이 고쳐질 때 조용히 갈라진다.
    ImageNet 사전학습을 쓸 수 있다는 것도 이유다 — 3DDFA_V2 의 V1 은 스크래치다.
    """
    from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
    if imagenet:
        if abs(width_mult - 1.0) > 1e-6:
            raise SystemExit("[FA3D] ImageNet 가중치는 width_mult=1.0 에만 있습니다")
        net = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    else:
        net = mobilenet_v2(weights=None, width_mult=width_mult)
    return net.features, net.last_channel


class MobileNetV2(nn.Module):
    """120×120 입력 → 62-d 파라미터 (+ 학습 전용 68점 랜드마크)."""

    def __init__(self, num_params: int = 62, num_landmarks: int = 68,
                 width_mult: float = 1.0, use_lrr: bool = True,
                 split_head: bool = True, dropout: float = 0.2,
                 imagenet: bool = False):
        super().__init__()
        if num_params != 62 and split_head:
            raise SystemExit("[FA3D] split_head 는 62-d(12+40+10) 에서만 의미가 있습니다")

        self.features, feat = _features(width_mult, imagenet)
        self.feat_dim = feat          # 학습 저장소의 synergy 순환이 물어보던 값
        self.use_lrr = use_lrr
        self.split_head = split_head
        drop = lambda: nn.Dropout(dropout) if dropout > 0 else nn.Identity()  # noqa: E731

        if split_head:
            # SynergyNet 방식 — 상사변환 / 신원 / 표정을 각자 뽑는다
            self.head_ori = nn.Sequential(drop(), nn.Linear(feat, 12))
            self.head_shape = nn.Sequential(drop(), nn.Linear(feat, 40))
            self.head_exp = nn.Sequential(drop(), nn.Linear(feat, 10))
        else:
            self.fc_param = nn.Sequential(drop(), nn.Linear(feat, num_params))

        self.fc_lm = nn.Linear(feat, num_landmarks * 2) if use_lrr else None

        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 회귀 헤드는 작게 시작해야 초기 VDC 가 폭주하지 않는다 (LAB §3.2)
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    # ---------------------------------------------------------------- #
    def features_vec(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)

    def _param(self, f: torch.Tensor) -> torch.Tensor:
        if self.split_head:
            return torch.cat((self.head_ori(f), self.head_shape(f), self.head_exp(f)), 1)
        return self.fc_param(f)

    def forward(self, x: torch.Tensor) -> dict:
        f = self.features_vec(x)
        out = {"param": self._param(f), "feat": f}
        if self.fc_lm is not None:
            out["landmark"] = self.fc_lm(f).view(f.shape[0], -1, 2)
        return out

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """추론 경로 — 파라미터만. lrr 헤드는 타지 않는다."""
        return self._param(self.features_vec(x))

    def inference_modules(self):
        heads = ([self.head_ori, self.head_shape, self.head_exp]
                 if self.split_head else [self.fc_param])
        return [self.features] + heads


def mobilenet_v2(**kw):
    return MobileNetV2(width_mult=1.0, **kw)


def mobilenet_v2_x05(**kw):
    return MobileNetV2(width_mult=0.5, **kw)


def mobilenet_v2_imagenet(**kw):
    """ImageNet 사전학습에서 출발. 3DDFA_V2 논문은 스크래치라 이건 논문 밖 축이다.

    ⚠ ImageNet 은 224×224 로 학습됐고 여기 입력은 120×120 이다. 저수준 특징은
      옮겨 가지만 이득은 실측 대상이지 전제가 아니다 — scratch 대조군을 함께 돌릴 것.
    """
    kw.setdefault("imagenet", True)
    return MobileNetV2(width_mult=1.0, **kw)
