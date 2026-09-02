# coding: utf-8
"""NME 가 재지 **않는** 것을 잰다 — 3D 형상(identity) 예측 정확도.

    python scripts/FA3D/shape_accuracy.py --ckpt runs/fa3d/<런>/best.pth

왜 필요한가
-----------
AFLW2000-3D NME 는 68 랜드마크의 2D 거리다. 실측으로 **그 값은 거의 자세만 잰다** —
형상을 GT 로 완벽히 바꿔도 2.185% → 2.157% (1.3% 개선)뿐이고, 형상을 아예 안 내고
평균 얼굴을 써도 2.132% 로 오히려 낫다. 그래서 "NME 3.688 로 배포를 재현했다" 는
**자세를 재현했다**는 뜻이지 3D 얼굴을 맞춘다는 뜻이 아니다.

두 가지를 따로 낸다.
  1. identity 정확도 — GT 의 shape 편차를 방향·크기로 얼마나 맞추나.
     기준선은 **평균 얼굴만 내기**(상대오차 1.000). 못 넘으면 형상을 안 내느니만 못하다.
  2. NME 분해 — 자세와 형상 중 무엇이 NME 를 움직이나.

주의
----
- 표정(exp 10-d)은 0 으로 지운다. identity 를 재는 것이라 섞으면 값이 부풀려진다.
- 300W-LP val 을 쓴다 — AFLW2000-3D 에는 68점 GT 만 있고 62-d 파라미터가 없다.
- **얼굴당 1장만** 쓴다. 179장을 다 넣으면 표본이 사람이 아니라 자세가 된다.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.eval_fa3d import config, require_bfm, require_trainval    # noqa: E402
from pierrotfr.FA3D import (BFM, ParamNorm, load_checkpoint,           # noqa: E402
                            load_deployed, predict_filelist)
from pierrotfr.FA3D.bfm import SHAPE_DIM, TRANS_DIM, parse_param       # noqa: E402
from pred_3d import face_key                                           # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=24000, help="val 앞에서 읽을 장수")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    cfg = require_trainval(require_bfm(config()))
    dev = a.device
    bfm, pn = BFM(cfg.bfm_fp).to(dev), ParamNorm(cfg.dst_stats).to(dev)
    ours, spec_o = load_checkpoint(a.ckpt, dev)
    v2, spec_v = load_deployed("mb1", dev)

    root, lf = cfg.train_root, cfg.val_list
    lst = open(lf, encoding="utf-8").read().strip().split("\n")
    P = np.asarray(pickle.load(open(cfg.val_param, "rb")), np.float32)[:a.n]
    w = pickle.load(open(cfg.src_stats, "rb")); d = pickle.load(open(cfg.dst_stats, "rb"))
    # GT 는 v1 통계로 whitening 돼 있다 — v1 -> raw -> V2 로 갈아 끼워야 모델 출력과
    # 같은 공간에 놓인다.
    tgt = (((P * w["param_std"] + w["param_mean"]) - d["mean"]) / d["std"]).astype(np.float32)
    po = predict_filelist(ours, root, lf, dev, spec_o.border,
                          cfg.batch_size, cfg.num_workers, limit=a.n)
    pv = predict_filelist(v2, root, lf, dev, spec_v.border,
                          cfg.batch_size, cfg.num_workers, limit=a.n)

    first: dict = {}
    for i, fn in enumerate(lst[:a.n]):
        first.setdefault(face_key(fn), i)
    idx = np.array(sorted(first.values()))
    mean_face = bfm.shape(torch.zeros(1, 50, 1, device=dev))

    def id_dev(param):
        """표정을 지운 identity 형상의, 평균 얼굴로부터의 편차 [n, 3V]."""
        p = pn.denorm(torch.as_tensor(param[idx], device=dev))
        _, _, _, al = parse_param(p)
        al = al.clone(); al[:, SHAPE_DIM:] = 0
        return (bfm.shape(al) - mean_face).reshape(len(idx), -1)

    G = id_dev(tgt)
    print(f"■ identity(3D 형상) 예측 정확도 — 300W-LP val, 서로 다른 얼굴 {len(idx):,}명")
    print(f"  {'':<24}{'방향 cos':>9}{'크기비':>8}{'상대오차':>9}{'설명 분산':>10}")
    for name, X in (("평균 얼굴만 내기 (기준선)", torch.zeros_like(G)),
                    (spec_v.name[:23], id_dev(pv)),
                    (spec_o.name[:23], id_dev(po))):
        cos = float(torch.nn.functional.cosine_similarity(X, G, dim=1).mean())
        amp = float((X.norm(dim=1) / G.norm(dim=1)).mean())
        rel = float(((X - G).norm(dim=1) / G.norm(dim=1)).mean())
        print(f"  {name:<24}{cos:+9.3f}{amp:8.3f}{rel:9.3f}{cos ** 2 * 100:9.0f}%")
    print("  → 상대오차 1.000 = '그냥 평균 얼굴을 내는 것'. "
          "못 넘으면 형상을 안 내느니만 못하다.\n")

    g = pn.denorm(torch.as_tensor(tgt, device=dev))
    o = pn.denorm(torch.as_tensor(po, device=dev))

    def lmk(p):
        return bfm.landmarks3d(p, size=cfg.image_size)[:, :2, :]

    def nme(x, y):
        e = (x - y).norm(dim=1)
        px, py = y[:, 0], y[:, 1]
        sz = torch.sqrt((px.max(1).values - px.min(1).values)
                        * (py.max(1).values - py.min(1).values))
        return float((e.mean(1) / sz).mean() * 100)

    def mix(pose, shape):
        return torch.cat([pose[:, :TRANS_DIM], shape[:, TRANS_DIM:]], 1)

    G68 = lmk(g)
    zero = g.clone(); zero[:, TRANS_DIM:] = 0
    print(f"■ NME 는 무엇을 재는가 — 300W-LP val {len(tgt):,}장, GT 기준")
    for name, p in (("전부 예측", o),
                    ("자세만 GT (형상은 우리)", mix(g, o)),
                    ("형상만 GT (자세는 우리)", mix(o, g)),
                    ("형상을 평균 얼굴로 (자세는 GT)", mix(g, zero))):
        print(f"  {name:<30}{nme(lmk(p), G68):7.3f}%")
    print("  → 형상을 GT 로 바꿔도 거의 안 움직인다. **NME 는 사실상 자세 지표다.**")


if __name__ == "__main__":
    main()
