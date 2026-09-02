# coding: utf-8
"""예측 결과 격자 — 오차 분포를 눈으로 본다.

    python scripts/FA3D/pred_grid.py --ckpt runs/fa3d/<런>/best.pth
    python scripts/FA3D/pred_grid.py --ckpt … --mode worst          # 최악 50장
    python scripts/FA3D/pred_grid.py --ckpt … --set aflw            # 사람 주석 21점
    python scripts/FA3D/pred_grid.py --ckpt … --vs <다른 런>/best.pth

평균 NME 하나로는 "어떻게 틀리는지" 를 못 본다. 백분위 균등 표집으로 전체 스펙트럼을,
`--mode worst` 로 실패 유형을 확인한다.

⚠ `--set aflw` 의 GT 는 **사람 주석 21점**이다. 68점 예측을 21점으로 접어 비교하고,
  미주석 점(-1)은 제외한다 — 가려진 점이라 그 자체가 가림의 대리 지표다.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.eval_fa3d import config, require_bfm, require_testset    # noqa: E402
from figstyle import JPG, save                                        # noqa: E402
from pierrotfr.FA3D import BFM, ParamNorm, load_checkpoint, load_deployed  # noqa: E402
from pierrotfr.FA3D import benchmark as bench                         # noqa: E402
from pierrotfr.FA3D.metrics import fold_68to21                        # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", default="spread", choices=["spread", "worst"])
    ap.add_argument("--set", default="aflw2000", choices=["aflw2000", "aflw"])
    ap.add_argument("--vs", default="",
                    help="비교 대상 ckpt. 주면 '배포' 자리에 이 모델이 들어간다")
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    cfg = require_bfm(config())
    require_testset(cfg, a.set)
    dev = a.device
    bfm, pn = BFM(cfg.bfm_fp).to(dev), ParamNorm(cfg.dst_stats).to(dev)

    ours, spec_o = load_checkpoint(a.ckpt, dev)
    other, spec_v = (load_checkpoint(a.vs, dev) if a.vs else load_deployed("mb1", dev))

    root, _ = bench.paths(cfg, a.set)
    lst = open(bench.paths(cfg, a.set)[1], encoding="utf-8").read().split()
    gt = bench.ground_truth(cfg, a.set)
    lo, lv = (bench.to_landmarks(bfm, pn, bench.predict(m, s, cfg, a.set, dev),
                                 dev, cfg.image_size)
              for m, s in ((ours, spec_o), (other, spec_v)))
    no, nv = bench.nme(cfg, a.set, lo, gt), bench.nme(cfg, a.set, lv, gt)
    roi, yaw = gt["roi"], gt["yaw"]
    # AFLW 는 21점 GT 를 그린다 — 68점 GT 는 bbox 정규화에만 쓰인다
    gpts = gt["pts21"] if a.set == "aflw" else gt["pts68"]
    setname = ("AFLW (사람 주석 21점)" if a.set == "aflw" else "AFLW2000-3D (평가셋)")

    order = np.argsort(no)
    sel = (order[-50:][::-1] if a.mode == "worst"
           else order[np.linspace(0, len(order) - 1, 50).astype(int)])

    # 10행 × 5열 — 열을 줄이면 한 장이 커져 랜드마크가 보인다
    size = cfg.image_size
    fig, axes = plt.subplots(10, 5, figsize=(11.5, 23.5))
    for ax, i in zip(axes.ravel(), sel):
        im = cv2.cvtColor(cv2.imread(os.path.join(root, lst[i])), cv2.COLOR_BGR2RGB)
        sx, sy, ex, ey = roi[i]
        g = gpts[i][:2].astype(np.float64).copy()
        g[0] = (g[0] - sx) / (ex - sx) * size
        g[1] = (g[1] - sy) / (ey - sy) * size
        ax.imshow(im)
        if a.set == "aflw":
            keep = (gpts[i][0] != -1) & (gpts[i][1] != -1)   # 미주석(가려진) 점 제외
            e21 = fold_68to21(lo[i])
            ax.scatter(g[0][keep], g[1][keep], s=13, c="#ff2d20", linewidths=0)
            ax.scatter(e21[0][keep], e21[1][keep], s=13, c="#00e05a", linewidths=0)
            ax.scatter(lo[i][0], lo[i][1], s=1.6, c="#00e05a", alpha=0.45, linewidths=0)
        else:
            ax.scatter(g[0], g[1], s=5.5, c="#ff2d20", linewidths=0)          # GT
            ax.scatter(lo[i][0], lo[i][1], s=5.5, c="#00e05a", linewidths=0)  # 예측
        pct = int(np.where(order == i)[0][0] / len(order) * 100)
        win = no[i] <= nv[i]
        ax.set_title(f"p{pct:02d}  {no[i]*100:.2f}%   yaw {yaw[i]:+.0f}°\n"
                     f"대조 {nv[i]*100:.2f}%",
                     fontsize=8.4, pad=4, color="#1b7f3b" if win else "#c62828")
        ax.set_xlim(0, size); ax.set_ylim(size, 0); ax.axis("off")

    ttl = ("최악 50장" if a.mode == "worst" else "오차 백분위 p00(최선)~p99(최악) 균등 표집")
    legend = ("빨강 = GT 21점(주석된 것만),  큰 초록 = 접은 예측,  작은 초록 = 68점 전체"
              if a.set == "aflw" else "빨강 = GT 68점,  초록 = 우리 예측")
    fig.suptitle(f"{setname} 예측 — {spec_o.name}  ·  {ttl}\n{legend}"
                 f"  ·  제목색: 초록 = {spec_v.name[:24]} 보다 우리가 나음", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.972]); fig.subplots_adjust(hspace=0.30, wspace=0.03)
    save(fig, a.out or f"outputs/fa3d/{spec_o.name}/{a.set}_grid_{a.mode}.jpg", **JPG)
    print(f"  표집 50장 중 대조군보다 나음 {int((no[sel] <= nv[sel]).sum())}장")
    print(f"  전체 {len(no):,}장 — 우리 {no.mean()*100:.3f} · 대조 {nv.mean()*100:.3f} · "
          f"우리가 나은 샘플 {int((no <= nv).sum())}장 ({(no <= nv).mean()*100:.0f}%)")


if __name__ == "__main__":
    main()
