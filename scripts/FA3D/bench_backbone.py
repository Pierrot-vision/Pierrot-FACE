"""백본 지연시간·연산량 실측.

논문의 판매 포인트가 CPU 실시간(단일 코어 50fps)이라 **CPU 배치1 지연시간**이
핵심 지표다. 파라미터 수는 지연시간의 대리 지표가 못 된다 — depthwise conv 나
attention 은 연산량 대비 메모리 대역폭에 묶여 FLOPs 가 적어도 느릴 수 있다.

    python scripts/FA3D/bench_backbone.py
    python scripts/FA3D/bench_backbone.py --threads 1 --repeat 100
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from pierrotfr.FA3D import build_model                       # noqa: E402
from pierrotfr.utils import count_params, human              # noqa: E402


# ------------------------------------------------------------------ #
# MACs 카운터 — conv/linear 만 센다 (논문 표기와 같은 관례).
# 외부 의존(thop/ptflops)을 끌어오지 않으려고 직접 훅을 건다.
# ------------------------------------------------------------------ #
def count_macs(model: nn.Module, x: torch.Tensor) -> int:
    total = [0]

    def conv_hook(m, inp, out):
        # 출력 원소 하나당 (in_ch/groups × k_h × k_w) 곱셈
        total[0] += out.numel() * (m.in_channels // m.groups) * \
            m.kernel_size[0] * m.kernel_size[1]

    def lin_hook(m, inp, out):
        total[0] += out.numel() * m.in_features

    handles = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(lin_hook))
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    return total[0]


def timeit(fn, repeat: int, warmup: int, cuda: bool = False) -> float:
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeat * 1000.0        # ms


def main() -> None:
    ap = argparse.ArgumentParser()
    # ⚠ yolo26_n / pierrotxv2_n 은 백본 **정의**가 옆 랩(Pierrot_3D_Lab)에 있어
    #   PIERROTFR_3D_LAB_ROOT 가 잡혀 있어야 만들어진다. 기본 목록에서는 뺀다 —
    #   재고 싶으면 --models 로 이름을 주면 된다.
    ap.add_argument("--models", nargs="*",
                    default=["mobilenet_v1", "mobilenet_v1_x05",
                             "mobilenet_v2", "mobilenet_v2_x05"])
    ap.add_argument("--threads", type=int, default=0,
                    help="CPU 스레드 수. 0 = 1과 4 둘 다 (논문 표와 같은 조건)")
    ap.add_argument("--repeat", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--gpu-batch", type=int, default=128)
    ap.add_argument("--size", type=int, default=120, help="입력 한 변 (크롭 규격)")
    args = ap.parse_args()

    thread_opts = [1, 4] if args.threads == 0 else [args.threads]
    rows = []

    for name in args.models:
        # 추론 경로만 잰다 — lrr 헤드도 synergy 순환도 배포에서 빠진다.
        # 이 저장소의 build_model 은 애초에 둘을 만들지 않으므로 여기 값이 곧 배포 값이다.
        model = build_model(name, num_params=62, use_lrr=False).eval()
        x120 = torch.randn(1, 3, args.size, args.size)

        total, _ = count_params(model)
        macs = count_macs(model, x120)

        row = {"name": name, "params": total, "macs": macs}

        with torch.no_grad():
            for t in thread_opts:
                torch.set_num_threads(t)
                row[f"cpu_t{t}"] = timeit(lambda: model.predict(x120),
                                          args.repeat, args.warmup)
            torch.set_num_threads(os.cpu_count())

            if torch.cuda.is_available():
                g = model.cuda()
                xg = x120.cuda()
                row["gpu_b1"] = timeit(lambda: g.predict(xg), args.repeat, args.warmup, True)
                xb = torch.randn(args.gpu_batch, 3, args.size, args.size,
                                 device="cuda")
                per = timeit(lambda: g.predict(xb), max(args.repeat // 5, 5),
                             args.warmup, True)
                row["gpu_batch"] = per
                row["gpu_thru"] = args.gpu_batch / (per / 1000.0)
                model.cpu()
        rows.append(row)

    # ---- 출력 ----
    base = rows[0]
    hdr = (f"{'백본':<18s}{'Params':>9s}{'MACs':>10s}"
           + "".join(f"{'CPU t' + str(t):>10s}" for t in thread_opts)
           + f"{'GPU b1':>9s}{'GPU b' + str(args.gpu_batch):>11s}{'img/s':>10s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        line = (f"{r['name']:<18s}{human(r['params']):>9s}{human(r['macs']):>10s}"
                + "".join(f"{r['cpu_t' + str(t)]:>9.2f}m" for t in thread_opts))
        if "gpu_b1" in r:
            line += f"{r['gpu_b1']:>8.2f}m{r['gpu_batch']:>10.2f}m{r['gpu_thru']:>10.0f}"
        print(line)

    print("\n" + "-" * len(hdr))
    print(f"{'(기준 대비)':<18s}", end="")
    print(f"{'':>9s}{'':>10s}", end="")
    for t in thread_opts:
        print(f"{'':>10s}", end="")
    print()
    for r in rows[1:]:
        rel = [f"params ×{r['params'] / base['params']:.2f}",
               f"MACs ×{r['macs'] / base['macs']:.2f}"]
        rel += [f"CPU t{t} ×{r['cpu_t' + str(t)] / base['cpu_t' + str(t)]:.2f}"
                for t in thread_opts]
        if "gpu_b1" in r:
            rel.append(f"GPU b1 ×{r['gpu_b1'] / base['gpu_b1']:.2f}")
        print(f"  {r['name']:<16s} vs {base['name']}: " + " · ".join(rel))
    print("\n※ ×1 미만 = 더 작다/빠르다.  논문 보고치(CPU): MobileNet 6.2ms · "
          "MobileNet×0.75 4.2ms · MobileNet-V3×0.5 3.4ms")


if __name__ == "__main__":
    main()
