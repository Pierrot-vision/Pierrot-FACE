# coding: utf-8
"""샘플 단위 오차 분석 — 평균 NME 하나로는 "어디서 지는지" 를 알 수 없다.

    python scripts/FA3D/error_analysis.py --ckpt runs/fa3d/<런>/best.pth
    python scripts/FA3D/error_analysis.py --ckpt <런A>/best.pth --vs <런B>/best.pth

같은 입력에 대해 두 모델을 세워 놓고 분포·꼬리·랜드마크 그룹·자세를 갈라 본다.
`--vs` 를 안 주면 3DDFA_V2 배포 mb1 이 대조군이다.

⚠ 테두리 전처리는 **각자 학습 때 쓴 값**을 건다 (배포 가중치는 0). 한쪽 값을 양쪽에
  걸면 비교가 깨진다 — 실측으로 `meta_paper_aug`(border=5)를 3.688 이 아니라
  3.948 로 잘못 쟀다.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from configs.eval_fa3d import available, config, require_bfm, require_testset  # noqa: E402
from pierrotfr.FA3D import BFM, ParamNorm, load_checkpoint, load_deployed      # noqa: E402
from pierrotfr.FA3D import benchmark as bench                                  # noqa: E402
from pierrotfr.FA3D.crop import to_original                                    # noqa: E402
from pierrotfr.FA3D.metrics import GROUPS                                      # noqa: E402


def stat(v: np.ndarray) -> dict:
    return dict(mean=v.mean() * 100, p50=np.percentile(v, 50) * 100,
                p90=np.percentile(v, 90) * 100, p99=np.percentile(v, 99) * 100,
                fail=int((v > 0.10).sum()))


def group_errors(lmk: list, roi: np.ndarray, gt: np.ndarray, size: int) -> dict:
    """68점 그룹별 평균 NME — 어느 부위에서 지는지 갈라야 처방이 나온다."""
    out = {}
    for name, idx in GROUPS.items():
        idx = list(idx)
        acc = 0.0
        for i in range(len(roi)):
            g = gt[i][:2]
            ll = np.sqrt((g[0].max() - g[0].min()) * (g[1].max() - g[1].min()))
            f = to_original(lmk[i], roi[i], size)
            acc += np.sqrt(((f[:, idx] - g[:, idx]) ** 2).sum(0)).mean() / ll
        out[name] = acc / len(roi) * 100
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vs", default="", help="대조군 체크포인트 (없으면 배포 mb1)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = require_bfm(config())
    require_testset(cfg, "aflw2000")
    dev = args.device
    bfm, pn = BFM(cfg.bfm_fp).to(dev), ParamNorm(cfg.dst_stats).to(dev)

    ours, spec_o = load_checkpoint(args.ckpt, dev)
    if args.vs:
        other, spec_v = load_checkpoint(args.vs, dev)
    else:
        other, spec_v = load_deployed("mb1", dev)
    print(f"테두리 전처리: {spec_o.name} {spec_o.border}px · "
          f"{spec_v.name} {spec_v.border}px (각자 학습 때 쓴 값)\n")

    have = available(cfg)
    for which in ("aflw2000", "aflw"):
        if not have[which]:
            continue
        name = bench.SETS[which]["label"]
        gt = bench.ground_truth(cfg, which)
        lo, lv = (bench.to_landmarks(bfm, pn, bench.predict(m, s, cfg, which, dev),
                                     dev, cfg.image_size)
                  for m, s in ((ours, spec_o), (other, spec_v)))
        no, nv = bench.nme(cfg, which, lo, gt), bench.nme(cfg, which, lv, gt)

        print("=" * 78); print(f"■ {name}  ({len(no):,}장)"); print("=" * 78)
        so, sv = stat(no), stat(nv)
        w = max(len(spec_o.name), len(spec_v.name), 10) + 2
        print(f"{'':<{w}s}{'평균':>8s}{'중앙값':>9s}{'p90':>8s}{'p99':>8s}{'실패(>10%)':>11s}")
        for tag, s in ((spec_o.name, so), (spec_v.name, sv)):
            print(f"{tag:<{w}s}{s['mean']:>8.3f}{s['p50']:>9.3f}"
                  f"{s['p90']:>8.3f}{s['p99']:>8.3f}{s['fail']:>11d}")
        print(f"{'차이':<{w}s}{so['mean']-sv['mean']:>+8.3f}{so['p50']-sv['p50']:>+9.3f}"
              f"{so['p90']-sv['p90']:>+8.3f}{so['p99']-sv['p99']:>+8.3f}"
              f"{so['fail']-sv['fail']:>+11d}")

        # 우리가 지는 샘플이 전체의 몇 % 이고 격차의 몇 % 를 만드나
        d = no - nv
        worse = d > 0
        print(f"\n  지는 샘플 {worse.sum():,}장 ({worse.mean()*100:.1f}%) · "
              f"이기는 샘플 {(~worse).sum():,}장")
        if abs(d.sum()) > 1e-12:
            top = np.sort(d)[-max(len(d) // 100, 1):]
            print(f"  평균 격차 {d.mean()*100:+.3f}%p 중 "
                  f"상위 1%({len(top)}장)가 {top.sum()/d.sum()*100:.0f}% 를 만든다")

        a = np.abs(gt["yaw"] if gt["yaw"].ndim == 1 else gt["yaw"][:, 0])
        print(f"\n  {'yaw':<10s}{'n':>7s}{'우리':>8s}{'대조':>8s}{'차이':>8s}")
        for lo_, hi_ in ((0, 30), (30, 60), (60, 91)):
            m = (a >= lo_) & (a < hi_)
            if m.sum():
                print(f"  [{lo_:2d},{min(hi_, 90):2d}]{'':<4s}{m.sum():>7d}"
                      f"{no[m].mean()*100:>8.3f}{nv[m].mean()*100:>8.3f}"
                      f"{(no[m]-nv[m]).mean()*100:>+8.3f}")

        # 68점 그룹별 (AFLW 은 GT 가 21점이라 그룹이 안 나뉜다)
        if which == "aflw2000":
            go = group_errors(lo, gt["roi"], gt["pts68"], cfg.image_size)
            gv = group_errors(lv, gt["roi"], gt["pts68"], cfg.image_size)
            print(f"\n  {'부위':<12s}{'우리':>8s}{'대조':>8s}{'차이':>8s}")
            for k in GROUPS:
                print(f"  {k:<12s}{go[k]:>8.3f}{gv[k]:>8.3f}{go[k]-gv[k]:>+8.3f}")
        print()


if __name__ == "__main__":
    main()
