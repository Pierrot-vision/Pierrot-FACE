"""FAS 체크포인트 로드와 배치 추론.

체크포인트의 config 가 백본 · 쿼리 모듈 · 정규화 · 입력 크기를 밝히므로 다시 지정할
필요가 없다. 학습 때 얹혔던 FLAN-T5(`llm.` · `llm_proj.`)는 버린다.

Q-Former 는 고정 지시문 두 개(content · style)를 읽는다. 문장이 고정이라 BERT
토크나이저로 **한 번만** 토크나이즈해 배치 크기만큼 복제한다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch

from .data import eval_transform
from .instructions import CONTENT_KEYS, STYLE_KEYS, question
from .models import build_model

TOKENIZER = "bert-base-uncased"     # Q-Former 지시문 토크나이저 (어휘만 빌린다)


@dataclass
class FASSpec:
    name: str
    model_name: str
    backbone: str
    image_size: int
    norm: str
    center_frac: float
    data_dir: str          # 학습·평가에 쓴 가공물 디렉토리 이름 (processed/<data_dir>/)
    params: int
    epoch: int | str
    amp: str               # 학습·평가 정밀도 (체크포인트 config)

    def transform(self):
        return eval_transform(self.image_size, self.norm, self.center_frac)


class FASModel:
    """모델 + 고정 지시문 토큰. `(images) -> dict(score, margin, cue)`.

    ⚠ 학습 저장소는 bf16 autocast 로 학습·평가했다. 이 모델은 정밀도에 민감해 fp32 로
      돌리면 같은 가중치가 test ACER 4.6 → 7.4 로 달라진다(실측). 그래서 체크포인트의
      amp 설정을 그대로 따른다 (GPU 에서만).
    """

    def __init__(self, model, instr: dict, device, amp_dtype=None):
        self.model, self.instr, self.device = model, instr, device
        self.amp_dtype = amp_dtype if torch.device(device).type == "cuda" else None

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> dict:
        B = images.shape[0]
        batch = {"image": images.to(self.device)}
        for k, v in self.instr.items():
            batch[k] = v.expand(B, -1)
        with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
            return self.model.predict(batch)


def load_checkpoint(path: str, device="cuda") -> tuple[FASModel, FASSpec]:
    from transformers import AutoTokenizer

    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    extra = dict(cfg.get("model_extra", {}))

    # 지시문을 읽는 모델(InstructFLIP)만 토크나이저가 필요하다
    tok = AutoTokenizer.from_pretrained(TOKENIZER) if cfg["model_name"] == "instructflip" else None
    model = build_model(cfg["model_name"], backbone=cfg["backbone"], image_size=cfg["image_size"],
                        vocab=tok.vocab_size if tok else 30522, **extra)

    # EMA 가중치가 있으면 그쪽을 쓴다 — 학습 때 평가에 쓴 것과 같아야 한다
    key = "ema" if "ema" in ck else "model"
    sd = {k: v for k, v in ck[key].items() if not k.startswith(("llm.", "llm_proj."))}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"[FAS] state_dict 불일치 — missing {missing[:3]} · "
                           f"unexpected {unexpected[:3]}")
    model = model.to(device).eval()

    instr = {}
    for kind in (() if tok is None else ("content", "style")):
        q = question(CONTENT_KEYS[0] if kind == "content" else STYLE_KEYS[0])
        enc = tok([q], return_tensors="pt")
        instr[f"{kind}_ids"] = enc["input_ids"].to(device)
        instr[f"{kind}_mask"] = enc["attention_mask"].to(device)

    spec = FASSpec(name=os.path.basename(os.path.dirname(os.path.abspath(path))),
                   model_name=cfg["model_name"], backbone=cfg["backbone"],
                   image_size=cfg["image_size"], norm=cfg["norm"],
                   center_frac=float(cfg.get("center_frac") or 1.0),
                   data_dir=os.path.basename(os.path.dirname(cfg.get("test_csv") or "ca/test.csv")),
                   params=sum(p.numel() for p in model.parameters()),
                   epoch=ck.get("epoch", "?"),
                   amp=cfg.get("amp_dtype", "fp32") if cfg.get("amp") else "fp32")
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.get("amp_dtype")) if cfg.get("amp") else None
    return FASModel(model, instr, device, amp), spec


@torch.no_grad()
def score_loader(fas: FASModel, loader) -> dict:
    """로더 전체의 margin · 라벨 · video_id · spoof_type · 경로."""
    M, Y, V, S, P = [], [], [], [], []
    for b in loader:
        M.append(fas(b["image"])["margin"].float().cpu().numpy())
        Y.append(b["label"].numpy())
        V.append(b["video_id"].numpy())
        S.append(b["spoof_type"].numpy())
        P += b["path"]
    return {"margin": np.concatenate(M), "label": np.concatenate(Y),
            "video_id": np.concatenate(V), "spoof_type": np.concatenate(S), "path": P}
