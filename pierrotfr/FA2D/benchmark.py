"""FA2D 벤치마크 — README 평가 표의 세 축을 그대로 잰다.

    ① WFLW test 2,500        io-NME · FR10          (각 모델 자기 크롭 규약)
    ② LaPa test 2,000        공통 38점 io-NME · FR10 (세 연구 모두 학습 안 한 축)
    ③ HRFFA 난이도 프로토콜   base · roll 최악 · 자세 교란 평균 · 스타일 평균 · 스타일 최악

학습 저장소에서는 이 셋이 세 스크립트(`engine.evaluate` · `eval_lapa_common.py` ·
`hrffa_protocol.py`)에 흩어져 있었다. 참조 모델(Peppa · HRFFA)을 **같은 코드로** 재야 해서
모델 무관한 함수로 둔다 — 여기서는 `predict_fn(images) -> 0~1 좌표` 하나만 받는다.

⚠ ② 의 크롭은 ① ③ 과 **렌더러가 다르다**(`copyMakeBorder` + `resize` vs `warpPerspective`).
  학습 저장소가 참조 모델들을 그 렌더러로 재서 표를 만들었으므로 그대로 둔다 — 바꾸면
  우리 값만 서브픽셀 단위로 움직여 참조와의 비교가 흔들린다.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from . import metrics as M
from .data import (IMAGENET_MEAN, IMAGENET_STD, EvalDataset, face_bbox,
                   head_bbox_from_landmarks, read_lapa, read_wflw, to_tensor)
from .geometric import GeoParams, apply_geometric

_MEAN = np.float32(IMAGENET_MEAN)
_STD = np.float32(IMAGENET_STD)


def _batched(predict_fn, x: np.ndarray, bs: int) -> np.ndarray:
    return np.concatenate([predict_fn(torch.from_numpy(x[k:k + bs])) for k in range(0, len(x), bs)])


# ------------------------------------------------------------------ #
# ① WFLW test — 학습 저장소 `engine.evaluate` 와 같은 크롭 경로
# ------------------------------------------------------------------ #
def wflw(predict_fn, wflw_dir: str, size: int, pad: float, crop_mode: str,
         bs: int = 32, workers: int = 8, limit: int = 0) -> dict:
    """전체 test 2,500 장. ⚠ 부분집합(512 장)은 전체보다 0.77 낙관적이었다 — 기본은 전부."""
    recs = read_wflw(wflw_dir, "test")
    if limit:
        recs = recs[:limit]
    dl = DataLoader(EvalDataset(recs, size, pad, crop_mode), batch_size=bs,
                    num_workers=workers, shuffle=False)
    P, G = [], []
    for b in dl:
        P.append(predict_fn(b["image"]))
        G.append(b["points"].numpy())
    pred, gt = np.concatenate(P) * size, np.concatenate(G) * size
    io = M.nme(pred, gt, "wflw98", size, "interocular")
    return {"io_nme": float(io.mean()), "fr10": M.failure_rate(io), "auc10": M.auc(io),
            "head_nme": float(M.nme(pred, gt, "wflw98", size, "head").mean()), "n": len(io)}


# ------------------------------------------------------------------ #
# ② LaPa test — 98점 모델을 106점 GT 와 공통 38점으로 비교
# ------------------------------------------------------------------ #
# 대응은 학습 저장소가 이중 헤드(wflw98/lapa106) 모델로 **실측**해 잔차가 자기 오차 수준인
# 점만 남긴 것이다 — 눈 16 · 동공 2 · 입 20. 윤곽·코·눈썹은 규약이 실제로 달라 뺐다.
LAPA_W_IDX = list(range(60, 76)) + [96, 97] + list(range(76, 96))
LAPA_L_IDX = list(range(66, 74)) + list(range(75, 83)) + [104, 105] + list(range(84, 104))


def _lapa_crop_square(img, X, Y, w, size):
    add = int(max(img.shape[:2]))
    b = cv2.copyMakeBorder(img, add, add, add, add, cv2.BORDER_CONSTANT)
    X, Y, w = int(round(X)) + add, int(round(Y)) + add, int(round(w))
    return cv2.resize(b[Y:Y + w, X:X + w], (size, size)), (X - add, Y - add, w)


def lapa_common38(predict_fn, lapa_dir: str, size: int, pad: float, crop_mode: str,
                  bs: int = 16, limit: int = 0) -> dict:
    recs = read_lapa(lapa_dir, "test")
    if limit:
        recs = recs[:limit]
    out = []
    for k0 in range(0, len(recs), bs):
        X, Gp, IO = [], [], []
        for r in recs[k0:k0 + bs]:
            img = cv2.imread(str(r.image_path))
            p = np.asarray(r.points, np.float64)
            if crop_mode == "face":
                x1, y1 = p.min(0); x2, y2 = p.max(0)
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                h = (1 + 2 * pad) * max(x2 - x1, y2 - y1) / 2
                Xb, Yb, w = cx - h, cy - h, 2 * h
            else:
                x1, y1, x2, y2 = head_bbox_from_landmarks(p)
                s = max(x2 - x1, y2 - y1) * (1 + 2 * pad)
                Xb, Yb, w = (x1 + x2) / 2 - s / 2, (y1 + y2) / 2 - s / 2, s
            c, (X0, Y0, ww) = _lapa_crop_square(img, Xb, Yb, w, size)
            g = (p - [X0, Y0]) * size / ww
            X.append(to_tensor(c)); Gp.append(g[LAPA_L_IDX])
            IO.append(np.linalg.norm(g[66] - g[79]))
        P = predict_fn(torch.stack(X))[:, LAPA_W_IDX] * size
        out.append(np.linalg.norm(P - np.stack(Gp), axis=-1).mean(1) / np.array(IO) * 100)
    v = np.concatenate(out)
    return {"io_nme": float(v.mean()), "fr10": float((v > 10).mean() * 100), "n": len(v)}


# ------------------------------------------------------------------ #
# ③ HRFFA 난이도 프로토콜 — HRFFA evaluate.py 의 어려운 조건을 WFLW test 로 옮긴 것
# ------------------------------------------------------------------ #
#   pose-stress (HRFFA Table 3/4/5) : 300장(linspace) · 15 설정
#   style-shift (HRFFA Table 7)     : 2,500장 · 9 섭동 · 정규화 왕복 후 uint8 섭동
POSE_CONFIGS = ([("base", 0, 0, 0)]
                + [(f"roll{r:+04d}", r, 0, 0) for r in (45, 90, 135, 180, 225, 270, 315)]
                + [(f"cam_p{p:+03d}", 0, p, 0) for p in (-25, -15, 15, 25)]
                + [(f"cam_y{y:+03d}", 0, 0, y) for y in (-15, 15)]
                + [("cam_p+25_y+15", 0, 25, 15), ("cam_p-25_y-15", 0, -25, -15)])


def _style_shifts():
    def gains(b, g, r):
        return lambda img: (img.astype(np.float32) * np.array([b, g, r], np.float32)).clip(0, 255).astype(np.uint8)

    def gamma(g):
        return lambda img: (255.0 * (img.astype(np.float32) / 255.0) ** g).astype(np.uint8)

    def gray(img):
        return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    def jpeg30(img):
        _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 30])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)

    def mblur(length, angle_deg=30.0):
        k = np.zeros((length, length), np.float32); c = length // 2; th = np.deg2rad(angle_deg)
        for t in np.linspace(-c, c, length * 4):
            x, y = int(round(c + t * np.cos(th))), int(round(c + t * np.sin(th)))
            if 0 <= x < length and 0 <= y < length:
                k[y, x] = 1.0
        k /= max(k.sum(), 1.0)
        return lambda img: cv2.filter2D(img, -1, k)

    return [("clean", None), ("mblur9", mblur(9)), ("mblur21", mblur(21)),
            ("warm", gains(0.80, 1.00, 1.20)), ("cool", gains(1.20, 1.00, 0.80)),
            ("gamma0.6", gamma(0.6)), ("gamma1.6", gamma(1.6)), ("gray", gray), ("jpeg30", jpeg30)]


def _norm(bgr):
    return to_tensor(bgr).numpy()


def _perturb(xchw, fn):
    """HRFFA 와 같다 — 정규화 → 역정규화 → uint8 → 섭동 → 정규화. 왕복 양자화까지 같게."""
    x = xchw.transpose(1, 2, 0) * _STD + _MEAN
    bgr = cv2.cvtColor((x.clip(0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return _norm(fn(bgr))


def _io(pred, gt):
    io = np.linalg.norm(gt[:, 60] - gt[:, 72], axis=-1).clip(1e-6)
    return np.linalg.norm(pred - gt, axis=-1).mean(1) / io


def difficulty(predict_fn, wflw_dir: str, size: int, pad: float, crop_mode: str,
               bs: int = 32, n_pose: int = 300, verbose: bool = True) -> dict:
    """얼굴 크롭 모델은 두 단계 모두 얼굴 박스(pad=자기 pad), 머리 크롭 모델은 HRFFA 기본값
    (pose 0.15 · style 0.05)으로 자른다 — 학습 저장소와 같다."""
    box = face_bbox if crop_mode == "face" else head_bbox_from_landmarks
    pad_ps, pad_st = (pad, pad) if crop_mode == "face" else (0.15, 0.05)
    recs = read_wflw(wflw_dir, "test")
    res = {}

    idx = np.linspace(0, len(recs) - 1, n_pose).astype(int)
    imgs = {int(i): cv2.imread(recs[i].image_path) for i in idx}
    for label, r, cp, cy in POSE_CONFIGS:
        X, G = [], []
        for i in idx:
            pts = recs[i].points
            o = apply_geometric(imgs[int(i)], pts, [-1] * 98, box(pts),
                                GeoParams(out_size=size, pad=pad_ps, roll_deg=float(r),
                                          cam_pitch_deg=float(cp), cam_yaw_deg=float(cy)))
            X.append(_norm(o["image"])); G.append(np.asarray(o["points"], np.float32))
        pred = _batched(predict_fn, np.stack(X), bs) * size
        res["ps_" + label] = float(_io(pred, np.stack(G)).mean() * 100)
        if verbose:
            print(f"  pose {label:<14s} {res['ps_' + label]:.3f}", flush=True)

    shifts = _style_shifts()
    acc = {n: [] for n, _ in shifts}
    for k0 in range(0, len(recs), bs):
        X, G = [], []
        for rec in recs[k0:k0 + bs]:
            o = apply_geometric(cv2.imread(rec.image_path), rec.points, [-1] * 98,
                                box(rec.points), GeoParams(out_size=size, pad=pad_st))
            X.append(_norm(o["image"])); G.append(np.asarray(o["points"], np.float32))
        G = np.stack(G)
        for n, fn in shifts:
            Xs = np.stack(X) if fn is None else np.stack([_perturb(x, fn) for x in X])
            acc[n].append(_io(predict_fn(torch.from_numpy(Xs)) * size, G))
    for n in acc:
        res["st_" + n] = float(np.concatenate(acc[n]).mean() * 100)

    roll = [v for k, v in res.items() if k == "ps_base" or k.startswith("ps_roll")]
    ps = [v for k, v in res.items() if k.startswith("ps_") and k != "ps_base"]
    st = [v for k, v in res.items() if k.startswith("st_") and k != "st_clean"]
    res["summary"] = {"base": res["ps_base"], "roll_worst": max(roll),
                      "pose_mean": float(np.mean(ps)),
                      "style_mean": float(np.mean(st)), "style_worst": max(st)}
    return res
