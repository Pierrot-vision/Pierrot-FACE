# coding: utf-8
"""두 런의 **차이가 큰 표본**을 나란히 본다 — 무엇이 좋아지고 무엇이 나빠졌나.

    python scripts/FA3D/diff_grid.py --a runs/fa3d/<A>/best.pth --b runs/fa3d/<B>/best.pth

`pred_grid.py --mode worst` 는 두 모델이 **똑같이** 실패하는 크롭·자세 파탄만 보여
준다 (최악 50장에서 배포 337% vs 우리 362%). 축 하나의 효과를 보려면 **그 축이 바꾼
표본**을 봐야 한다 — a 가 b 보다 크게 나빠진 것과 크게 좋아진 것을 각각 뽑는다.

기본 평가셋은 AFLW 다. GT 가 사람 주석 21점이라 **주석된 점 수 자체가 가림의 대리
지표**여서, 무엇이 달라졌는지 읽기에 가장 좋다.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.eval_fa3d import config, require_bfm, require_testset    # noqa: E402
from figstyle import JPG, plt, save                                   # noqa: E402
from pierrotfr.FA3D import BFM, ParamNorm, load_checkpoint            # noqa: E402
from pierrotfr.FA3D import benchmark as bench                         # noqa: E402
from pierrotfr.FA3D.metrics import fold_68to21                        # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="검증 대상 (새 축)")
    ap.add_argument("--b", required=True, help="대조군")
    ap.add_argument("--set", default="aflw", choices=["aflw", "aflw2000"])
    ap.add_argument("--n", type=int, default=15, help="각 방향으로 뽑을 장수")
    ap.add_argument("--min-nme", type=float, default=4.0,
                    help="이 값 미만인 표본은 제외 — 둘 다 잘 맞는 건 볼 게 없다")
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    cfg = require_bfm(config())
    require_testset(cfg, a.set)
    dev = a.device
    bfm, pn = BFM(cfg.bfm_fp).to(dev), ParamNorm(cfg.dst_stats).to(dev)

    root, lf = bench.paths(cfg, a.set)
    lst = open(lf, encoding="utf-8").read().split()
    gt = bench.ground_truth(cfg, a.set)
    roi, yaw = gt["roi"], gt["yaw"]
    is_aflw = a.set == "aflw"
    gpts = gt["pts21"] if is_aflw else gt["pts68"]
    val = ((gpts[:, 0] != -1) & (gpts[:, 1] != -1)) if is_aflw else None

    L, E, names = {}, {}, {}
    for tag, ck in (("a", a.a), ("b", a.b)):
        m, spec = load_checkpoint(ck, dev)
        L[tag] = bench.to_landmarks(bfm, pn, bench.predict(m, spec, cfg, a.set, dev),
                                    dev, cfg.image_size)
        E[tag] = bench.nme(cfg, a.set, L[tag], gt) * 100
        names[tag] = spec.name

    d = E["a"] - E["b"]                       # + 면 a 가 나쁨
    idx = np.where(np.maximum(E["a"], E["b"]) >= a.min_nme)[0]
    if len(idx) < a.n:
        raise SystemExit(f"[FA3D] NME ≥ {a.min_nme}% 인 표본이 {len(idx)}장뿐입니다 "
                         f"— --min-nme 를 낮추세요")
    worse = idx[np.argsort(-d[idx])][:a.n]    # a 가 크게 나빠진 것
    better = idx[np.argsort(d[idx])][:a.n]    # a 가 크게 좋아진 것

    cols = 5
    rows_half = (a.n + cols - 1) // cols
    size = cfg.image_size
    fig, axes = plt.subplots(2 * rows_half, cols,
                             figsize=(2.3 * cols, 2.55 * 2 * rows_half))
    for blk, (sel, col) in enumerate(((worse, "#c62828"), (better, "#1b7f3b"))):
        for k, i in enumerate(sel):
            ax = axes[blk * rows_half + k // cols, k % cols]
            im = cv2.cvtColor(cv2.imread(os.path.join(root, lst[i])), cv2.COLOR_BGR2RGB)
            sx, sy, ex, ey = roi[i]
            g = gpts[i][:2].astype(np.float64).copy()
            g[0] = (g[0] - sx) / (ex - sx) * size
            g[1] = (g[1] - sy) / (ey - sy) * size
            keep = val[i] if is_aflw else np.ones(g.shape[1], bool)
            ax.imshow(im)
            ax.scatter(g[0][keep], g[1][keep], s=13, c="#ff2d20", linewidths=0)
            for tag, cc in (("b", "#3f7fff"), ("a", "#00e05a")):
                pts = fold_68to21(L[tag][i]) if is_aflw else L[tag][i][:2]
                ax.scatter(pts[0][keep], pts[1][keep], s=11, c=cc, linewidths=0)
            sub = (f"주석 {int(val[i].sum())}/21 · " if is_aflw else "")
            ax.set_title(f"{E['a'][i]:.1f}% vs {E['b'][i]:.1f}%  ({d[i]:+.1f})\n"
                         f"{sub}yaw {yaw[i]:+.0f}°", fontsize=7.6, color=col, pad=3)
            ax.set_xlim(0, size); ax.set_ylim(size, 0); ax.axis("off")
        for k in range(len(sel), rows_half * cols):
            axes[blk * rows_half + k // cols, k % cols].axis("off")

    fig.suptitle(f"축 하나가 바꾼 표본 — a: {names['a']}\n대조군 b: {names['b']}\n"
                 f"빨강 = GT · 초록 = a · 파랑 = b   "
                 f"(NME ≥ {a.min_nme}% 인 것만 · 위 = a 가 나빠진 것)",
                 fontsize=10.5)
    fig.tight_layout()
    save(fig, a.out or f"outputs/fa3d/{names['a']}/diff_grid.jpg", **JPG)
    if is_aflw:
        print(f"  a 가 나빠진 표본 상위 {a.n}: 주석 {val[worse].sum(1).mean():.1f}/21 · "
              f"|yaw| {np.abs(yaw[worse]).mean():.0f}°")
        print(f"  a 가 좋아진 표본 상위 {a.n}: 주석 {val[better].sum(1).mean():.1f}/21 · "
              f"|yaw| {np.abs(yaw[better]).mean():.0f}°")


if __name__ == "__main__":
    main()
