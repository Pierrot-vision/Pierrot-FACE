"""FA2D 체크포인트를 README 평가 표의 세 축으로 잰다.

    python eval/FA2D/evaluate.py --ckpt runs/fa2d/<런>/best.pth
    python eval/FA2D/evaluate.py --ckpt <교사>/best.pth <학생>/best.pth --only wflw lapa
    python eval/FA2D/evaluate.py --ckpt … --tta

    ① WFLW test 2,500        io-NME · FR10
    ② LaPa test 2,000        공통 38점 io-NME · FR10
    ③ HRFFA 난이도 프로토콜   base · roll 최악 · 자세 교란 평균 · 스타일 평균 · 스타일 최악

체크포인트가 백본 · 분기 · 크롭 규약을 스스로 밝히므로 다시 지정할 필요가 없다.

⚠ ③ 은 오래 걸린다(2,500장 × 섭동 9종 + 300장 × 자세 15종). 빠른 확인은 `--only wflw lapa`.
⚠ `--tta` 는 기본 지표를 덮지 않고 **행을 따로** 낸다 — README 표의 수치는 TTA 없는 값이다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.paths import FA2D_LAPA_DIR, FA2D_WFLW_DIR, describe, env_hint, output_dir  # noqa: E402
from pierrotfr.FA2D import benchmark as bench                                        # noqa: E402
from pierrotfr.FA2D.infer import load_checkpoint, predict                            # noqa: E402

AXES = ("wflw", "lapa", "difficulty")


def _need(path: str, what: str) -> None:
    if not os.path.isdir(path):
        raise SystemExit(f"[FA2D] {what} 데이터가 없습니다: {path}\n  (현재 {describe()}){env_hint()}")


def evaluate_one(model, spec, axes, tta: bool, device: str, bs: int) -> dict:
    def fn(x):
        return predict(model, x.to(device), "wflw98", tta).cpu().numpy()

    S, pad, crop = spec.image_size, spec.crop_pad, spec.crop_mode
    res = {}
    if "wflw" in axes:
        t = time.time()
        res["wflw"] = bench.wflw(fn, FA2D_WFLW_DIR, S, pad, crop, bs=bs)
        print(f"  ① WFLW  io-NME {res['wflw']['io_nme']:.3f} · FR10 {res['wflw']['fr10']:.2f}% "
              f"(n={res['wflw']['n']}, {time.time() - t:.0f}s)", flush=True)
    if "lapa" in axes:
        t = time.time()
        res["lapa"] = bench.lapa_common38(fn, FA2D_LAPA_DIR, S, pad, crop, bs=bs)
        print(f"  ② LaPa  공통38 io-NME {res['lapa']['io_nme']:.3f} · FR10 {res['lapa']['fr10']:.2f}% "
              f"(n={res['lapa']['n']}, {time.time() - t:.0f}s)", flush=True)
    if "difficulty" in axes:
        t = time.time()
        res["difficulty"] = bench.difficulty(fn, FA2D_WFLW_DIR, S, pad, crop, bs=bs, verbose=False)
        s = res["difficulty"]["summary"]
        print(f"  ③ 난이도  base {s['base']:.3f} · roll 최악 {s['roll_worst']:.3f} · "
              f"자세 교란 평균 {s['pose_mean']:.3f} · 스타일 평균 {s['style_mean']:.3f} · "
              f"스타일 최악 {s['style_worst']:.3f} ({time.time() - t:.0f}s)", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--only", nargs="*", choices=AXES, default=list(AXES),
                    help="잴 축 (기본 전부)")
    ap.add_argument("--tta", action="store_true",
                    help="좌우반전 TTA 행을 **추가로** 낸다 (기본 지표는 안 덮는다 · 추론 2배)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save", action="store_true", help="outputs/fa2d/<런>/eval_results.json")
    a = ap.parse_args()

    if "wflw" in a.only or "difficulty" in a.only:
        _need(FA2D_WFLW_DIR, "WFLW")
    if "lapa" in a.only:
        _need(FA2D_LAPA_DIR, "LaPa")
    print(f"[eval] {describe()}")

    rows = []
    for fp in a.ckpt:
        model, spec = load_checkpoint(fp, a.device)
        res = {"base": evaluate_one(model, spec, a.only, False, a.device, a.batch_size)}
        if a.tta:
            print("  --- flip-TTA ---")
            res["tta"] = evaluate_one(model, spec, a.only, True, a.device, a.batch_size)
        rows.append((spec, res))
        if a.save:
            out = os.path.join(output_dir(spec.name, "fa2d"), "eval_results.json")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(res, fh, indent=2, ensure_ascii=False)
            print(f"  결과 -> {out}")
        del model
        torch.cuda.empty_cache()

    # README 표와 같은 칸 순서로 한 줄씩 — 그대로 옮겨 적을 값
    print("=" * 100)
    head = f"{'런':<36s}{'Params':>8s}"
    if "wflw" in a.only:
        head += f"{'①WFLW':>8s}{'FR10':>7s}"
    if "lapa" in a.only:
        head += f"{'②LaPa38':>9s}{'FR10':>7s}"
    if "difficulty" in a.only:
        head += f"{'③base':>7s}{'roll최악':>9s}{'자세평균':>9s}{'스타일평균':>10s}{'스타일최악':>10s}"
    print(head)
    for spec, res in rows:
        for tag in ("base", "tta"):
            if tag not in res:
                continue
            r = res[tag]
            line = f"{(spec.name + (' +TTA' if tag == 'tta' else ''))[:35]:<36s}{spec.params / 1e6:>7.2f}M"
            if "wflw" in r:
                line += f"{r['wflw']['io_nme']:>8.3f}{r['wflw']['fr10']:>6.2f}%"
            if "lapa" in r:
                line += f"{r['lapa']['io_nme']:>9.3f}{r['lapa']['fr10']:>6.2f}%"
            if "difficulty" in r:
                s = r["difficulty"]["summary"]
                line += (f"{s['base']:>7.3f}{s['roll_worst']:>9.3f}{s['pose_mean']:>9.3f}"
                         f"{s['style_mean']:>10.3f}{s['style_worst']:>10.3f}")
            print(line)


if __name__ == "__main__":
    main()
