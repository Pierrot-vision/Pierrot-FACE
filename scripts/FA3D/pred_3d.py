# coding: utf-8
"""3D 밀집 복원 결과를 **3차원으로** 본다 — 정점 38,365개 / 삼각형 76,073개.

    python scripts/FA3D/pred_3d.py --ckpt runs/fa3d/<런>/best.pth
    python scripts/FA3D/pred_3d.py --ckpt … --n 6
    python scripts/FA3D/pred_3d.py --ckpt … --compare      # GT vs 우리 vs 배포

`pred_grid.py` 는 68점을 이미지 위에 찍을 뿐이라 **z 를 못 본다.** 3DDFA_V2 가 2D
정렬과 다른 점이 정확히 그 z 인데, 2D 격자로는 "제대로 된 3D 를 냈는지" 를 판정할 수
없다. 그래서 같은 예측을 새 시점에서 다시 그린다 — 한 장의 2D 입력에서 나온 형상이
옆에서 봐도 얼굴이면 그 z 가 의미 있는 것이다.

⚠ 정면(중립) 열은 **과장해서** 그린다. 있는 그대로 그리면 사람마다 똑같아 보여
  "모델이 한 얼굴만 낸다"고 읽히는데, 그건 절반만 맞다 — BFM shape 40-d 는 평균
  얼굴에 얹는 작은 섭동이라 **GT 도 그렇게 보인다**(`--compare` 가 그 판정이다).
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.eval_fa3d import (config, require_bfm, require_testset,   # noqa: E402
                               require_trainval)
from figstyle import JPG, plt, save                                    # noqa: E402
from pierrotfr.FA3D import (BFM, ParamNorm, load_checkpoint,           # noqa: E402
                            load_deployed, load_tri, predict_filelist,
                            to_landmarks)
from pierrotfr.FA3D import benchmark as bench                          # noqa: E402
from pierrotfr.FA3D.bfm import SHAPE_DIM, parse_param                  # noqa: E402
from pierrotfr.FA3D.render import draw_depth, overlay, render          # noqa: E402

EX = 3.0                # identity 편차 과장 배율 — 모든 열에 같은 값을 쓴다


def face_key(fn: str) -> str:
    """v1 접두와 증강 인덱스를 떼어 300W-LP 의 '얼굴' 키를 만든다.

        HELEN_HELEN_3051471764_1_8_2.jpg  ->  HELEN_3051471764_1

    300W-LP 687,854장의 실제 다양성은 얼굴 **3,837개**뿐이다 — 얼굴당 179장으로
    부풀린 것이라, 사람 단위 통계를 낼 때는 반드시 이걸로 접어야 한다.
    """
    import re
    n = re.sub(r"\.jpg$", "", fn)
    n = re.sub(r"^(AFW|HELEN|IBUG|LFPW)(Flip)?_", "", n)
    return re.sub(r"_\d+_\d+$", "", n)


def identity_shape(bfm, pn, param, idx, dev):
    """표정(exp 10-d)을 지운 identity 형상 [n,3,V].

    ⚠ alpha 는 50-d = shape 40 + **exp 10** 이다. 그대로 그리면 '중립' 이라 써 놓고
      표정을 그린다 — GT 얼굴들의 입이 벌어져 나온 게 그 때문이었다.
    """
    p = pn.denorm(torch.as_tensor(np.asarray(param)[idx], device=dev))
    _, _, _, al = parse_param(p)
    al = al.clone(); al[:, SHAPE_DIM:] = 0
    return bfm.shape(al)


# ------------------------------------------------------------------ #
def compare(a, cfg, dev, bfm, pn, ours, spec_o) -> None:
    """'정면 복원이 다 같은 얼굴 아니냐' 를 판정한다.

    같은 사람들에 대해 **GT · 우리 · 배포** 의 중립(정면) 형상을 같은 렌더러로 나란히
    그린다. GT 도 비슷해 보이면 그건 BFM shape 40-d 기저의 표현력 한계이고,
    GT 만 다르면 우리 모델이 평균 얼굴로 붕괴한 것이다 — 그림 하나로 갈린다.

    AFLW2000-3D 에는 68점 GT 만 있고 62-d 파라미터가 없어 300W-LP val 을 쓴다.
    """
    require_trainval(cfg)
    v2, spec_v = load_deployed("mb1", dev)
    root, lf = cfg.train_root, cfg.val_list
    lst = open(lf, encoding="utf-8").read().strip().split("\n")
    n_read = a.n_read

    P = np.asarray(pickle.load(open(cfg.val_param, "rb")), np.float32)[:n_read]
    w = pickle.load(open(cfg.src_stats, "rb")); d = pickle.load(open(cfg.dst_stats, "rb"))
    tgt = (((P * w["param_std"] + w["param_mean"]) - d["mean"]) / d["std"]).astype(np.float32)
    po = predict_filelist(ours, root, lf, dev, spec_o.border, cfg.batch_size,
                          cfg.num_workers, limit=n_read)
    pv = predict_filelist(v2, root, lf, dev, spec_v.border, cfg.batch_size,
                          cfg.num_workers, limit=n_read)

    # ⚠ **정면 표본만** 쓴다. 300W-LP 는 같은 얼굴을 179 자세로 부풀린 셋이라 아무거나
    #   고르면 입력이 옆얼굴로 나오는데, 이 그림은 '정면 복원'을 입력 얼굴과 대조해
    #   읽는 것이라 입력도 정면이어야 비교가 된다.
    raw = tgt * d["std"] + d["mean"]
    R = raw[:, :12].reshape(-1, 3, 4)[:, :, :3]
    f = np.linalg.norm(R, axis=(1, 2)) / np.sqrt(3)
    Rn = R / f[:, None, None]
    # ⚠ yaw 만 보면 안 된다 — roll 이 큰(고개를 90° 기울인) 표본이 그대로 뽑힌다.
    yaw = np.degrees(np.arcsin(np.clip(-Rn[:, 2, 0], -1, 1)))
    pitch = np.degrees(np.arctan2(Rn[:, 2, 1], Rn[:, 2, 2]))
    roll = np.degrees(np.arctan2(Rn[:, 1, 0], Rn[:, 0, 0]))
    frontal = ((np.abs(yaw) <= a.max_yaw) & (np.abs(pitch) <= a.max_yaw)
               & (np.abs(roll) <= a.max_yaw))

    keys = [face_key(x) for x in lst[:n_read]]
    gtn = np.linalg.norm(raw[:, 12:52], axis=1)
    best: dict = {}
    for i in np.nonzero(frontal)[0]:                 # 얼굴 하나당 정면 표본 하나만
        best.setdefault(keys[i], int(i))
    cand = np.array(sorted(best.values()))
    if len(cand) < a.n:
        raise SystemExit(f"[FA3D] yaw·pitch·roll 이 모두 {a.max_yaw}° 이내인 서로 다른 "
                         f"얼굴이 {len(cand)}명뿐입니다 — --max-yaw 를 올리세요")

    # ⚠ ‖shape‖ **최대**로 고르면 안 된다 — 3DMM 피팅이 터진 이상치가 뽑혀 GT 열이
    #   얼굴이 아닌 덩어리로 나온다(실제로 그랬다). 분포의 중상위(p50~p92)에서
    #   고르게 뽑아 '개성 있지만 정상인' 얼굴을 쓴다.
    o = cand[np.argsort(gtn[cand])]
    lo, hi = int(len(o) * 0.50), int(len(o) * 0.92)
    sel = [int(o[j]) for j in np.linspace(lo, hi - 1, a.n).astype(int)]

    tri = load_tri(cfg.tri_fp, bfm.n_vertex)
    cols_data = (("GT (300W-LP 라벨)", tgt), (f"{spec_o.name[:22]} 예측", po),
                 ("배포 mb1 예측", pv))
    N = {k: identity_shape(bfm, pn, v, sel, dev).cpu().numpy() for k, v in cols_data}

    # ⚠ 다양성 통계는 **그림에 쓴 n 명이 아니라** 서로 다른 얼굴 전체로 낸다.
    #   n 명(그것도 ‖shape‖ 로 고른)에서 낸 표준편차는 표본이 작고 편향돼 있다.
    first: dict = {}
    for i, k in enumerate(keys):
        first.setdefault(k, i)
    big = np.array(sorted(first.values()))
    stat = {}
    for key, param in cols_data:
        S = identity_shape(bfm, pn, param, big, dev)
        S = S - S.mean(2, keepdim=True)
        stat[key] = float(S.std(0).mean()) / float(S.abs().mean()) * 100

    mean_shape = bfm.shape(torch.zeros(1, 50, 1, device=dev))[0].cpu().numpy()
    cols = ["입력"] + list(N)
    fig, axes = plt.subplots(len(sel), len(cols),
                             figsize=(2.15 * len(cols), 2.3 * len(sel)))
    for r, i in enumerate(sel):
        axes[r, 0].imshow(cv2.cvtColor(cv2.imread(os.path.join(root, lst[i])),
                                       cv2.COLOR_BGR2RGB))
        for k, key in enumerate(N):
            axes[r, 1 + k].imshow(render(N[key][r], tri, 0.0,
                                         dev=N[key][r] - mean_shape, exaggerate=EX))
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            for ax, t in zip(axes[r], cols):
                ax.set_title(t, fontsize=10, pad=5)

    g = stat["GT (300W-LP 라벨)"]
    txt = "  ·  ".join(f"{k} {v:.2f}% ({v / g * 100:.0f}%)" for k, v in stat.items())
    fig.suptitle(f"identity(정면·중립) 복원 비교 — 300W-LP val, 서로 다른 사람 {len(sel)}명 "
                 f"(입력은 yaw·pitch·roll 모두 {a.max_yaw:.0f}° 이내인 정면만)\n"
                 f"평균 얼굴로부터의 편차를 ×{EX:.0f} 과장 · 색 = 법선 방향 변위"
                 " (빨강 바깥 / 파랑 안쪽)\n"
                 f"얼굴 {len(big):,}명 기준 정점 표준편차:  {txt}", fontsize=10.5)
    fig.tight_layout()
    save(fig, a.out or f"outputs/fa3d/{spec_o.name}/pred_3d_identity.jpg", **JPG)


# ------------------------------------------------------------------ #
def spectrum(a, cfg, dev, bfm, pn, ours, spec_o) -> None:
    """yaw 스펙트럼을 고르게 — 3D 가 제대로 나왔는지는 옆얼굴에서 갈린다."""
    require_testset(cfg, "aflw2000")
    root, lf = bench.paths(cfg, "aflw2000")
    lst = open(lf, encoding="utf-8").read().split()
    gt = bench.ground_truth(cfg, "aflw2000")
    param = bench.predict(ours, spec_o, cfg, "aflw2000", dev)
    nme = bench.nme(cfg, "aflw2000", to_landmarks(bfm, pn, param, dev, cfg.image_size), gt)

    # ⚠ AFLW2000-3D.pose.npy 는 이미 **degree** 다 (radian 아님). np.degrees 를 또
    #   걸면 68° 가 3,904° 가 되어 표본 선택과 제목이 통째로 틀어진다.
    yaw = gt["yaw"]
    ya = np.abs(yaw if yaw.ndim == 1 else yaw[:, 0])
    sel = [int(np.argmin(np.abs(ya - t))) for t in np.linspace(3, 80, a.n)]

    tri = load_tri(cfg.tri_fp, bfm.n_vertex)
    size = cfg.image_size
    p = pn.denorm(torch.as_tensor(param[sel], device=dev))
    dense = bfm.reconstruct(p).detach().cpu().numpy()             # [n,3,38365]
    _, _, _, alpha = parse_param(p)
    neutral = bfm.shape(alpha).detach().cpu().numpy()             # 투영 전 형상
    al_id = alpha.clone(); al_id[:, SHAPE_DIM:] = 0               # 표정 뺀 identity
    ident = bfm.shape(al_id).detach().cpu().numpy()
    mean_shape = bfm.shape(torch.zeros(1, 50, 1, device=dev))[0].cpu().numpy()

    cols = ["입력 + 68점", "밀집 메쉬 (예측 자세)", f"정면 (identity ×{EX:.0f})",
            f"yaw -55° (×{EX:.0f})", f"yaw +55° (×{EX:.0f})", "깊이 z"]
    fig, axes = plt.subplots(len(sel), len(cols),
                             figsize=(2.05 * len(cols), 2.25 * len(sel)))
    for r, i in enumerate(sel):
        img = cv2.cvtColor(cv2.imread(os.path.join(root, lst[i])), cv2.COLOR_BGR2RGB)
        v = dense[r].copy()
        v[1] = size - v[1]                     # 이미지 y 축(아래로 증가)에 맞춘다
        l68 = to_landmarks(bfm, pn, param[i:i + 1], dev, size)[0]

        axes[r, 0].imshow(img)
        axes[r, 0].scatter(l68[0], l68[1], s=4.5, c="#00e05a", linewidths=0)
        axes[r, 1].imshow(overlay(img, v, tri))
        # 3~5열은 **같은 처리**를 쓴다. 한 열만 과장하면 열끼리 비교가 안 되고,
        # 과장 안 한 열은 사람마다 똑같아 보여 "한 얼굴만 낸다"고 다시 오해된다.
        # 표정은 그대로 두고 **identity 편차만** 과장한다 — 표정까지 3배로 키우면
        # 벌어진 입이 세 배로 벌어져 얼굴이 기형처럼 보인다.
        dv = ident[r] - mean_shape
        for k, ang in enumerate((0.0, -55.0, 55.0)):
            axes[r, 2 + k].imshow(render(neutral[r], tri, ang, dev=dv, exaggerate=EX))
        axes[r, 5].imshow(draw_depth(v, tri, size))

        axes[r, 0].set_ylabel(f"|yaw| {ya[i]:.0f}°\nNME {nme[i]*100:.2f}%", fontsize=8.5)
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            for ax, t in zip(axes[r], cols):
                ax.set_title(t, fontsize=9.5, pad=5)

    fig.suptitle("FA3D 3D 밀집 복원 — 정점 38,365개 · 삼각형 76,073개  "
                 f"({spec_o.name})\n"
                 "3~5열은 같은 예측을 새 시점에서 다시 그린 것 — 한 장의 2D 입력에서 나온 형상이다\n"
                 f"3~5열 모두 평균 얼굴로부터의 편차를 ×{EX:.0f} 과장 · 색 = 법선 방향 변위"
                 " (빨강 바깥 / 파랑 안쪽) — 과장 없이는 GT 도 사람마다 같아 보인다",
                 fontsize=11)
    fig.tight_layout()
    save(fig, a.out or f"outputs/fa3d/{spec_o.name}/pred_3d.jpg", **JPG)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=6, help="행 수 (표본 수)")
    ap.add_argument("--out", default="")
    ap.add_argument("--compare", action="store_true",
                    help="GT vs 우리 vs 배포 의 정면(중립) 형상을 나란히 — identity 붕괴 판정")
    ap.add_argument("--max-yaw", type=float, default=12.0,
                    help="--compare 입력 표본의 yaw·pitch·roll 공통 상한 (정면·수직만)")
    ap.add_argument("--n-read", type=int, default=24000,
                    help="--compare 에서 300W-LP val 앞에서 읽을 장수")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    cfg = require_bfm(config())
    dev = a.device
    bfm, pn = BFM(cfg.bfm_fp).to(dev), ParamNorm(cfg.dst_stats).to(dev)
    ours, spec_o = load_checkpoint(a.ckpt, dev)
    (compare if a.compare else spectrum)(a, cfg, dev, bfm, pn, ours, spec_o)


if __name__ == "__main__":
    main()
