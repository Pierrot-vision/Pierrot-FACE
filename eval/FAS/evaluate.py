"""FAS 체크포인트를 README 평가 표의 지표로 잰다 (CelebA-Spoof intra-dataset).

    python eval/FAS/evaluate.py --ckpt runs/fas/<런>/best.pth      # test.csv · 임계값은 val.csv
    python eval/FAS/evaluate.py --ckpt … --csv <다른 CSV> --val_csv <임계값용 CSV>
    python eval/FAS/evaluate.py --ckpt … --folder /path/to/bench --name MSU

    임계값 0.5 : CelebA-Spoof 공식 규칙 (margin 0 = 확률 0.5) — APCER · BPCER · ACER
    val 기준   : val 에서 BPCER 1% 가 되는 margin 을 test 에 그대로 쓴다 (운영점)
    EER · AUC · R@FPR : 임계값과 무관한 순위 지표

체크포인트가 백본 · 정규화 · 입력 크기를 스스로 밝히므로 다시 지정할 필요가 없다.
⚠ test 67,170장 + val 49,459장 — A100 한 장에 약 10분.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.paths import FAS_DATA_ROOT, describe, env_hint, output_dir     # noqa: E402
from pierrotfr.FAS import load_checkpoint, score_loader                  # noqa: E402
from pierrotfr.FAS import metrics as M                                    # noqa: E402
from pierrotfr.FAS.data import CelebASpoofDataset, FolderFASDataset, collate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", default=None,
                    help="CelebA-Spoof 형식 CSV (기본: 체크포인트가 학습한 processed/<ca|ca_wide>/test.csv)")
    ap.add_argument("--val_csv", default=None,
                    help="임계값을 여기서 뽑는다 (기본: 같은 디렉토리의 val.csv). "
                         "'none' 이면 test 자신에서 (낙관 편향)")
    ap.add_argument("--folder", default=None, help="폴더형 벤치마크 (<root>/{real,fake}) — --csv 대신")
    ap.add_argument("--name", default=None)
    ap.add_argument("--op_bpcer", type=float, default=1.0,
                    help="운영점: val 에서 BPCER 가 이 값(%%)이 되는 임계값을 test 에 쓴다")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save", action="store_true", help="outputs/fas/<런>/eval_results.json")
    a = ap.parse_args()

    print(f"[eval] {describe()}")
    fas, spec = load_checkpoint(a.ckpt, a.device)

    # 체크포인트가 학습한 가공물(ca = 정렬 타이트 crop · ca_wide = 박스 2.7배)로 잰다
    data = os.path.join(FAS_DATA_ROOT, "processed", spec.data_dir)
    a.csv = a.csv or os.path.join(data, "test.csv")
    a.val_csv = None if a.val_csv == "none" else (a.val_csv or os.path.join(data, "val.csv"))
    for p in ([a.val_csv] if a.val_csv else []) + ([] if a.folder else [a.csv]):
        if not os.path.exists(p):
            raise SystemExit(f"[FAS] CSV 가 없습니다: {p}\n  (현재 {describe()}){env_hint()}")
    tf = spec.transform()
    print(f"[FAS] {spec.name} · {spec.model_name}/{spec.backbone} · {spec.params / 1e6:.1f}M "
          f"· {spec.image_size}² · norm={spec.norm} · data={spec.data_dir} · {spec.amp} · epoch {spec.epoch}")

    def run(ds):
        t = time.time()
        dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                        num_workers=a.workers, collate_fn=collate)
        r = score_loader(fas, dl)
        print(f"  {ds.name}: {len(ds)}장 ({time.time() - t:.0f}s)", flush=True)
        return r

    res = {}
    thr = op_thr = None
    if a.val_csv:
        v = run(CelebASpoofDataset(a.val_csv, tf, name="val"))
        vm = M.compute(v["margin"], v["label"], v["video_id"], op_bpcer=a.op_bpcer)
        thr, op_thr = vm["threshold"], vm["op_threshold"]
        res["val"] = vm

    ds = (FolderFASDataset(a.folder, tf, name=a.name) if a.folder
          else CelebASpoofDataset(a.csv, tf, name=a.name))
    t = run(ds)
    m = M.compute(t["margin"], t["label"], t["video_id"], threshold=thr,
                  op_bpcer=a.op_bpcer, op_threshold=op_thr)
    res[ds.name] = m

    # README 표와 같은 칸 순서 — 그대로 옮겨 적을 값
    print("=" * 96)
    print(f"{'':28s}{'ACER':>7s}{'APCER':>7s}{'BPCER':>7s}{'EER':>7s}{'AUC':>7s}"
          f"{'R@1%':>7s}{'R@0.5%':>8s}{'R@0.1%':>8s}")
    tail = f"{m['EER']:7.2f}{m['AUC']:7.2f}{m['R@F1']:7.2f}{m['R@F0.5']:8.2f}{m['R@F0.1']:8.2f}"
    print(f"{'임계값 0.5 (margin 0)':28s}{m['ACER']:7.2f}{m['APCER']:7.2f}{m['BPCER']:7.2f}" + tail)
    src = "val" if a.val_csv else "test 자신 ⚠"
    label = f"{src} 기준 (margin {m['op_threshold']:.1f})"
    print(f"{label:28s}{m['ACER_op']:7.2f}{m['APCER_op']:7.2f}{m['BPCER_op']:7.2f}" + tail)
    print(f"  n={m['n']} · HTER {m['HTER']:.2f} (임계값 {m['threshold']:.2f})")

    if a.save:
        out = os.path.join(output_dir(spec.name, "fas"), "eval_results.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2, ensure_ascii=False)
        print(f"  결과 -> {out}")


if __name__ == "__main__":
    main()
