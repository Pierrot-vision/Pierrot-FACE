<p align="center">
  <img src="docs/peirrot-face-banner.jpg" width="100%" alt="PIERROT FACE banner"/>
</p>

<h1 align="center">🎭 PIERROT FACE — FA3D</h1>

<p align="center">
  <b>3D dense face alignment (a 3DDFA_V2 reimplementation) — inference distribution</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9-blue.svg" alt="python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.2-ee4c2c.svg" alt="pytorch"/>
  <img src="https://img.shields.io/badge/FA3D-3DDFA__V2-success.svg" alt="fa3d"/>
  <img src="https://img.shields.io/badge/inference--only-✓-brightgreen.svg" alt="inference-only"/>
  <img src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-orange.svg" alt="license"/>
  <img src="https://img.shields.io/badge/commercial%20use-⛔%20prohibited-red.svg" alt="no-commercial"/>
</p>

<p align="center">
  <a href="README.md">한국어</a> | <b>English</b>
</p>

---

## 💡 Introduction

**PIERROT FACE** is a **one-person Face Vision research and development project**. The goal
is to **separate the work into large categories** — anti-spoofing / 3D alignment /
recognition — rather than lumping them into a single pipeline, so that the algorithm at each
stage can be reproduced, swapped and compared independently.

This repository ships **only the inference path of FA3D (3D Dense Face Alignment)**.
Losses, optimization, augmentation and the training datasets have been stripped out of the
training repository; what remains is **exactly what is needed to turn one checkpoint into a
face**.

> **Origin of the name** — Pierrot is originally a pantomime clown character who **mimics and
> imitates others**. It resonates with [Pierrot Universe](https://github.com/Pierrot-vision)'s
> first philosophy (MimiC) — following and combining the good parts of existing research —
> which is why we use this name.

1. **Reproduce first, then combine (MimiC)** — the default direction is to **reproduce a
   verified implementation exactly, and only then combine**. We check parity against the
   official code and tabulate every difference. **When a paper ships no training code, we
   implement it directly from the equations** and record the reasoning (3DDFA_V2 is that case).
2. **Categories and algorithms stay separate** — no category reaches into another's code, and
   algorithms can be swapped inside a category.
3. **Records first** — analysis, design rationale and how to run things all live in [LAB](LAB/).
   **When a comparison is not apples-to-apples, we say so.**
4. **A playground for personal curiosity** — above all, this is where the author's curiosity
   gets worked out.

* The training code and trained weights are not publicly released at this time.

## 📰 News

- 2026-09-02 — 🚀 **FA3D inference code released** — image/video inference · AFLW2000-3D and
  AFLW evaluation · novel-view 3D rendering · shape accuracy. It **reproduces the training
  repository's numbers to three decimal places**
- 2026-09-01 — 🎯 **We beat the released weights** — AFLW2000-3D **3.622** vs 3.683
  (**−0.061**), obtained by combining augmentation · the synergy cycle · occlusion ·
  cross-pose identity consistency 👉 [Phase 5](LAB/FA3D/Exp/Phase_5_증강과_svs.md)
- 2026-08-27 — 🧊 **FA3D (the 3DDFA_V2 training half) reimplemented** — fWPDC, meta-joint
  optimization and landmark-regression regularization, none of which exist in the author's
  repository, written from the paper; the released weights' 3.683 reproduced
  👉 [Phase 1](LAB/FA3D/Exp/Phase_1_구현과_기준선.md)

## 🧊 3D Dense Face Alignment (FA3D)

**3DDFA_V2 (ECCV 2020)** publishes inference code and released weights only — **there is no
training code.** So we reimplemented it directly from the paper's equations. This repository
is **the code that runs that result**.

- 📘 **Algorithm** (equations · derivations · parameter semantics · evaluation protocol) — [LAB/FA3D/3DDFA_V2.md](LAB/FA3D/3DDFA_V2.md)
- 📗 **Implementation and experiment log** (bugs · measurements · analysis) — [LAB/FA3D/FA3D.md](LAB/FA3D/FA3D.md)
- 📙 **Phase-by-phase record** — [LAB/FA3D/Exp/](LAB/FA3D/Exp/) (Phase 1–10)

> ⚠ The LAB documents are written in Korean — they are the original research notes,
> copied here unchanged rather than translated, so the record stays exactly as it was made.

### What it actually predicts (video demo)

![FA3D video demo](docs/FA3D/demo.gif)

The top-left stack holds a **3D mesh crop per detected face**, one colour per person — the
colour is pinned to the tracking ID, so it stays the same as frames go by.
The full clip (95 s, H.264) is [`docs/FA3D/demo_full.mp4`](docs/FA3D/demo_full.mp4); the
version that covers the whole face with the dense mesh is
[`docs/FA3D/demo_3d.mp4`](docs/FA3D/demo_3d.mp4).

```bash
python eval/FA3D/infer.py --ckpt runs/fa3d/<run>/best.pth --source <video> \
    --corner3d 0.24 --max-faces 4 --drop-empty
```

| | |
|---|---|
| Detection | every frame — **if there is no face, nothing is drawn** |
| Tracking | the landmarks set the next crop; if the box jumps, re-detect immediately |
| Smoothing | **once, on the 62-d parameters** (so landmarks and mesh come from one value) |

> ⚠ **Smoothing is off by default.** §2.4 of the paper argues against temporal filtering —
> it *"reduces precision and introduces frame delay"* — which is why the authors reached for
> short-video-synthesis during **training** instead of post-processing. When you do turn it
> on (`--lmk-smooth`), the frame is labelled `smoothing`, and **it is never used for evaluation.**

### Evaluation

NME follows the paper's convention: **the mean of three yaw-bin means**, not a plain average.
Every number below was **measured with this repository's `eval/FA3D/evaluate.py`** — our
checkpoint and the author's released weights have to go through the same code, the same crop
convention and the same ground truth for the comparison to mean anything.

| Model | AFLW2000-3D ↓ | AFLW (21 pts) ↓ |
|---|---|---|
| **Ours** | **3.622** | **5.125** |
| — | — | — |
| 🎯 3DDFA_V2 released mb1 | *3.683* | *4.600* |
| 3DDFA_V2 released mb05 | *3.798* | *4.738* |
| Paper, M+R | *3.590* | *4.500* |
| Paper, M+R+S | *3.510* | *4.430* |

> ⚠ **The weights the authors released do not reproduce the paper's numbers.** Measured with the same evaluation code, mb1 comes out at 3.683 / 4.600 — **consistently 0.09–0.10 worse** than the paper's M+R (3.590 / 4.500) on both benchmarks. So the reproduction target of this project is **the released weights**, not the paper's table.

> ⚠⚠ **This NME is effectively a pose metric.** It barely measures how well the 3D face
> shape is recovered — replacing the predicted shape with ground truth improves it by only
> 1.3%, and emitting no shape at all (the mean face) is *better* still. **So the 3.622 above
> means "we reproduced the pose"**, not "we get the 3D face right." Shape accuracy is measured
> separately with `python scripts/FA3D/shape_accuracy.py`
> 👉 [Phase 6](LAB/FA3D/Exp/Phase_6_지표가_형상을_안_잰다.md)

> For the full experiment log, see [LAB/FA3D/FA3D.md](LAB/FA3D/FA3D.md).

### Predictions — AFLW2000-3D

![50 AFLW2000-3D predictions](docs/FA3D/aflw2000_grid_spread.jpg)

Fifty images sampled evenly across the error percentiles, p00 (best) to p99 (worst).
🟢 green = our prediction · 🔴 red = ground-truth 68 points; **a green title means we beat the
released weights on that image**. The [worst 50](docs/FA3D/aflw2000_grid_worst.jpg) are also here.

### 3D reconstruction — the same prediction, redrawn from a new viewpoint

![dense 3D reconstruction](docs/FA3D/pred_3d.jpg)

Plotting 68 points on the image **cannot show z** — even though z is exactly what separates
3DDFA_V2 from 2D alignment. So columns 3–5 **re-render the same prediction from other
angles**. In the last row (|yaw| 80°) only one cheek is visible in the input, yet a frontal
face comes back.

> Columns 3–5 **exaggerate the identity deviation from the mean face ×3** and colour the
> displacement along the normal (🔴 outward · 🔵 inward). BFM's 40-d shape basis is a **small
> perturbation** on top of the mean face, so drawn as-is **even the ground truth looks the
> same from person to person.**

### ⚡ Inference speed

`python scripts/FA3D/bench_backbone.py` — RTX 6000 Ada, batch 1, **inference path only**
(neither the lrr head nor the synergy cycle exists in this repository's models).

| Backbone | Params | MACs | CPU 1 thread | GPU b1 | GPU b128 throughput |
|---|---:|---:|---:|---:|---:|
| `mobilenet_v1` (paper default) | 3.3M | 177M | 6.12 ms | 1.78 ms | 23,162 img/s |
| `mobilenet_v1_x05` | 850K | 46M | **2.83 ms** | 1.81 ms | 45,253 img/s |
| `mobilenet_v2` (SynergyNet) | 2.3M | 93M | 6.01 ms | 3.44 ms | 18,312 img/s |
| `mobilenet_v2_x05` | 767K | 30M | 3.85 ms | 3.79 ms | 25,864 img/s |

> The paper reports 6.2 ms on CPU for MobileNet — the same place.
> ⚠ **Parameter count is not a proxy for latency.** MobileNetV2 has 30% fewer parameters and
> half the MACs of V1, yet it is the same on CPU and twice as slow on GPU — depthwise and
> inverted-residual blocks are bound by memory bandwidth, not arithmetic.

## 🚀 Install

```bash
git clone https://github.com/Pierrot-vision/Pierrot-FACE
cd Pierrot-FACE

conda create -n fr python=3.9 -y
conda activate fr
pip install -r requirements.txt

# Register machine-specific paths once, and you never have to export them again
cp paths.local.env.example paths.local.env
#   The key one is PIERROTFR_CKPT_ROOT — point it at the training runs/ directory
```

**The minimum you need is three BFM files.** Take them from the official 3DDFA_V2
repository's `configs/` and drop them into `$PIERROTFR_FA3D_DATA_ROOT/bfm/`:

```
bfm_noneck_v3.pkl                  38,365-vertex 3DMM basis
param_mean_std_62d_120x120.pkl     62-d Z-score statistics (the model's output space)
tri.pkl                            76,073 triangles — for mesh rendering
```

To compute metrics you also need the evaluation sets released with 3DDFA v1
(`test.data/` · `test.configs/`); to compute shape accuracy you additionally need the
preprocessed 300W-LP (`train.configs/`). Every entry point tells you what is present and
what is missing.

> Verified on: RTX 6000 Ada / torch 2.x / CUDA 12.
> With ffmpeg present, video is written as H.264 (otherwise use `--gif`).
> The face detector falls back **FaceBoxes → MTCNN → Haar cascade**, so the demo runs with
> nothing extra installed — at the cost of catching frontal faces only.

## ⚡ Quick start

**The `.sh` scripts take no arguments.** What you would change lives in the
`---- 여기만 바꾼다 ----` ("change only here") block at the top of each file. Call the Python
entry points directly and everything is a CLI flag.

```bash
# ── Inference ─────────────────────────────────────────────────────
# Accepts an image, a directory, a list.txt, or a video. Always prints speed.
python eval/FA3D/infer.py --ckpt runs/fa3d/<run>/best.pth --source data/samples --grid 3
python eval/FA3D/infer.py --ckpt … --source clip.mp4 --mesh --corner3d 0.24 --drop-empty
python eval/FA3D/infer.py --ckpt … --source clip.mp4 --gif --fps 12

# ── Evaluation ────────────────────────────────────────────────────
# Measure the released weights with the same code — the only valid comparison
python eval/FA3D/evaluate.py --ckpt runs/fa3d/<run>/best.pth --deployed mb1 --aflw

# ── Analysis ──────────────────────────────────────────────────────
python scripts/FA3D/pred_grid.py      --ckpt … --mode spread  # prediction grid (2D)
python scripts/FA3D/pred_grid.py      --ckpt … --mode worst   # failure modes
python scripts/FA3D/pred_3d.py        --ckpt …                # dense 3D, re-rendered
python scripts/FA3D/pred_3d.py        --ckpt … --compare      # identity: GT vs ours vs released
python scripts/FA3D/shape_accuracy.py --ckpt …                # shape (identity) accuracy
python scripts/FA3D/error_analysis.py --ckpt …                # per-sample error breakdown
python scripts/FA3D/diff_grid.py      --a … --b …             # where two runs diverge
python scripts/FA3D/bench_backbone.py                         # backbone latency · MACs

# ── Bundles ───────────────────────────────────────────────────────
bash scripts/FA3D/eval.sh                     # one evaluation pass
VIDEO=clip.mp4 bash scripts/FA3D/infer.sh     # inference · figures · speed, in one go
```

**The checkpoint carries its own configuration** — backbone, input size and border
preprocessing need not be specified, and cannot be. The border-zeroing step (`aug_border`)
in particular is **preprocessing shared by training and evaluation**, so a human must not
pick it: feed a different value than training used and the model sees a distribution it has
never seen. We once mis-measured the same weights as 3.948 instead of 3.688 that way.

## 📦 Layout

```
Pierrot_FR_Infer/
├── configs/
│   ├── paths.py            # 🗺️ dataset / weight / output roots (paths.local.env · env vars)
│   └── eval_fa3d.py        #    eval-set and BFM paths — only what the checkpoint cannot know
├── paths.local.env         # machine-specific paths — not committed (copy the .example)
├── pierrotfr/FA3D/         # 📦 inference path only
│   ├── infer.py            #   checkpoint loading · batch inference · FA3D class ★entry point
│   ├── bfm.py              #   3DMM decoder — 62-d → 38,365 vertices / 68 landmarks
│   ├── crop.py             #   face crop convention (ported from 3DDFA_V2) — half the accuracy
│   ├── data.py             #   preprocessing ((x−127.5)/128 · border) + eval dataset
│   ├── detect.py           #   face detection (FaceBoxes → MTCNN → Haar fallback) · demo only
│   ├── render.py           #   point-splat renderer — novel view · mesh overlay · 68 points
│   ├── metrics.py          #   AFLW2000-3D / AFLW NME (yaw-bin convention)
│   ├── benchmark.py        #   eval-set pipeline (predict → landmarks → NME)
│   └── models/             #   backbones — neither the lrr head nor the synergy cycle is built
├── eval/FA3D/
│   ├── infer.py            # 🔍 photos/video → 68 points / 3D mesh (+H.264/GIF, timing)
│   └── evaluate.py         #    AFLW2000-3D / AFLW NME (several ckpts + released weights)
├── scripts/FA3D/           # 🧪 figures · analysis · speed (bundled by eval.sh · infer.sh)
├── LAB/                    # 📚 read-only copy of the training repo's experiment record
├── docs/                   # banner · result visualizations · demos
├── data/samples/           # demo photos — not committed
├── runs/                   # (symlink) the training repo's checkpoints — read only
└── outputs/fa3d/<run>/     # inference visualizations — regenerable, not committed
```

### What the training repository has that this one does not

| Removed | Why |
|---|---|
| `train_fa3d.py` · `engine.py` | meta-joint optimization (Algorithm 1) — the training loop |
| `losses.py` | VDC · fWPDC · lrr · parameter-space loss |
| `svs.py` | short-video-synthesis — a training augmentation |
| `models/synergy.py` | the SynergyNet cycle (MLP_for/MLP_rev) — never on the `predict()` path |
| training datasets · `Augment` | 300W-LP loader · ColorJitter · occlusion · identity groups |
| `configs/args_fa3d.py` (626 lines) | hyperparameters — **the checkpoint carries them** |
| the `fc_lm` head | §2.3 lrr regularization — even the authors ship it and throw it away |

The lrr head and the synergy cycle **cost nothing at inference** (they are not on the
`predict()` path). This repository goes one step further and **never builds the modules at
all**, so the numbers in the speed table above are the numbers of the model that actually ships.

## 🤗 Reference

- PIERROT FACE is built by reproducing and combining the good parts of existing research.
- I am always grateful to that work.

📚 Papers —
[3DDFA_V2 (ECCV 2020)](https://guojianzhu.com/assets/pdfs/3162.pdf) ·
[SynergyNet (3DV 2021)](https://arxiv.org/abs/2110.09772)

🛠 Official code — [cleardusk/3DDFA_V2](https://github.com/cleardusk/3DDFA_V2) (source for the BFM, the crop convention and the released weights) · [cleardusk/3DDFA](https://github.com/cleardusk/3DDFA) (v1 — evaluation sets, tensor handling) · [choyingw/SynergyNet](https://github.com/choyingw/SynergyNet)

## 📄 License

This project (code and documentation) is licensed under **CC BY-NC-SA 4.0** — attribution,
non-commercial, share-alike. See [LICENSE](LICENSE) for details.
(Third-party datasets, models and libraries follow their own licenses. In particular the
ported model code, the BFM basis and the released weights follow their original repositories.)

❌ **No commercial use** — paid services, products, APIs, commercial model development and so
on (including model weights and inference outputs). Please contact the maintainer for
commercial licensing.

## 📮 Contact

- Please reach out by [email](mailto:peternara@naver.com) or through a
  [GitHub Issue](https://github.com/Pierrot-vision/Pierrot-FACE/issues). I will answer as
  carefully as I can, for anything I am able to answer.
- Note that questions already answered on GitHub (README, code, documentation) may not get a
  reply — thank you for understanding.
