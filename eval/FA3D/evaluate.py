"""FA3D 체크포인트를 AFLW2000-3D / AFLW 로 평가한다.

    python eval/FA3D/evaluate.py --ckpt runs/fa3d/<런>/best.pth
    python eval/FA3D/evaluate.py --ckpt <런A>/best.pth <런B>/best.pth --aflw
    python eval/FA3D/evaluate.py --ckpt <런>/best.pth --deployed mb1 --aflw

체크포인트가 백본·입력크기·테두리 전처리를 스스로 밝히므로 다시 지정할 필요가 없다.
`--deployed` 를 주면 저자 배포 가중치를 **같은 코드로** 재서 나란히 찍는다.

⚠ NME 는 논문 규약 = **yaw 3구간 평균의 평균**이다. 단순 평균(전체평균 칸)과
  0.4%p 넘게 다르다 — 다른 논문 수치와 비교할 때 반드시 왼쪽 값을 쓸 것.

⚠ 두 모델을 나란히 잴 때 테두리 전처리는 **각자 학습 때 쓴 값**을 건다. 한쪽 값을
  양쪽에 걸면 비교가 깨진다 (실측: 값 하나를 빠뜨려 3.688 을 3.948 로 잘못 쟀다).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.eval_fa3d import available, config, require_bfm, require_testset  # noqa: E402
from configs.paths import describe, output_dir                                # noqa: E402
from pierrotfr.FA3D import BFM, ParamNorm, load_checkpoint, load_deployed     # noqa: E402
from pierrotfr.FA3D import benchmark as bench                                 # noqa: E402
from pierrotfr.FA3D.metrics import format_row                                 # noqa: E402

# 같은 평가 코드로 잰 기준선 — 새 수치를 어디에 놓고 읽어야 하는지 알려 준다.
BASELINE = ("참고 — 3DDFA_V2 배포 mb1: AFLW2000-3D 3.683 · AFLW 4.600 / "
            "논문 M+R 3.59 · 4.50 / M+R+S 3.51 · 4.43")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="*", default=[],
                    help="평가할 체크포인트 (여러 개면 같은 데이터로 나란히)")
    ap.add_argument("--deployed", nargs="*", default=[], choices=["mb1", "mb05"],
                    help="저자 배포 가중치도 같은 코드로 잰다 (대조군)")
    ap.add_argument("--gt", default="ori", choices=["ori", "reannotated", "both"],
                    help="AFLW2000-3D 의 68점 GT — 원본 / 재주석본")
    ap.add_argument("--aflw", action="store_true", help="AFLW(21,080장)도 평가")
    ap.add_argument("--batch-size", type=int, default=0, help="0 = configs 기본값")
    ap.add_argument("--tta", action="store_true",
                    help="좌우반전 TTA 행을 **추가로** 낸다 (기본 지표는 안 덮는다). 추론 2배")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save", action="store_true",
                    help="outputs/fa3d/<런>/eval_results.json 을 남긴다")
    args = ap.parse_args()

    if not args.ckpt and not args.deployed:
        raise SystemExit("[eval] --ckpt 또는 --deployed 중 하나는 있어야 합니다")

    cfg = require_bfm(config())
    if args.batch_size:
        cfg.batch_size = args.batch_size
    require_testset(cfg, "aflw2000")
    have = available(cfg)
    if args.aflw and not have["aflw"]:
        print(f"[eval] 주의: AFLW 평가셋이 없어 건너뜁니다 ({cfg.test_dir}/AFLW_GT_crop)")
    print(f"[eval] {describe()}")

    bfm = BFM(cfg.bfm_fp).to(args.device)
    param_norm = ParamNorm(cfg.dst_stats).to(args.device)

    models = [load_checkpoint(fp, args.device) for fp in args.ckpt]
    models += [load_deployed(w, args.device) for w in args.deployed]

    gt_opts = ["ori", "reannotated"] if args.gt == "both" else [args.gt]
    rows = []
    for model, spec in models:
        res = {}
        for o in gt_opts:
            key = "AFLW2000-3D" + ("(re)" if o == "reannotated" else "")
            res[key], _ = bench.evaluate(model, spec, bfm, param_norm, cfg,
                                         "aflw2000", o, args.device)
            if args.tta:
                res[key + "+TTA"], _ = bench.evaluate(
                    model, spec, bfm, param_norm, cfg, "aflw2000", o,
                    args.device, tta=True)
        if args.aflw and have["aflw"]:
            res["AFLW"], _ = bench.evaluate(model, spec, bfm, param_norm, cfg,
                                            "aflw", device=args.device)
            if args.tta:
                res["AFLW+TTA"], _ = bench.evaluate(
                    model, spec, bfm, param_norm, cfg, "aflw",
                    device=args.device, tta=True)
        rows.append((spec.name, res))
        if args.save:
            # ⚠ 학습 저장소(runs/)에 쓰지 않는다 — 여기는 그걸 **읽기만** 하는
            #   저장소이고, 같은 파일 이름으로 학습 쪽 결과를 덮을 위험이 있다.
            out = os.path.join(output_dir(spec.name), "eval_results.json")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(res, fh, indent=2, ensure_ascii=False)
            print(f"  결과 -> {out}")

    print("=" * 88)
    for name, res in rows:
        print(name)
        for set_name, m in res.items():
            line = "  " + format_row(set_name, m)
            base = res.get(set_name[:-4]) if set_name.endswith("+TTA") else None
            if base:
                line += f"   {m['NME'] - base['NME']:+.3f}"
            print(line)
    print("=" * 88)

    # 여러 모델이면 한 줄 요약을 덧붙인다 — 표로 옮겨 적을 값
    if len(rows) > 1:
        sets = list(rows[0][1])
        print(f"{'런':<44s}" + "".join(f"{s:>16s}" for s in sets))
        for name, res in rows:
            print(f"{name[:43]:<44s}"
                  + "".join(f"{res[s]['NME']:>16.3f}" for s in sets if s in res))
    print(BASELINE)


if __name__ == "__main__":
    main()
