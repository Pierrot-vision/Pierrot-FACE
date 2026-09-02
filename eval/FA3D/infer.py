# coding: utf-8
"""임의의 사진·영상에 FA3D 를 돌린다 — 68 랜드마크 · 3D 밀집 메쉬.

    python eval/FA3D/infer.py --ckpt runs/fa3d/<런>/best.pth --source data/samples
    python eval/FA3D/infer.py --ckpt … --source clip.mp4 --mesh --corner3d 0.24
    python eval/FA3D/infer.py --ckpt … --source clip.mp4 --gif --fps 12

`--source` 는 이미지 · 디렉토리 · 목록.txt · 영상을 모두 받는다. 무엇이 들어왔는지는
확장자로 판단하고, 결과는 `outputs/fa3d/<런 이름>/` 에 쌓인다.

⚠ **평가와 다른 경로다.** 지표(AFLW2000-3D / AFLW)는 사전 크롭된 120x120 을 쓰므로
  검출기를 타지 않는다. 여기는 검출 → 크롭 → 추론이라 **검출기 품질이 결과에
  섞인다** — 데모 그림을 논문 수치와 나란히 놓지 말 것.

⚠ **랜드마크 출력은 평활하지 않는다.** 논문 §2.4 가 명시한다 — 시간 필터링은
  "정밀도를 깎고 프레임 지연을 만든다". 그래서 저자는 후처리 대신 svs 로 학습에서
  안정성을 얻었다. `--lmk-smooth` 는 데모 가독성용이고, 켜면 화면에 표시된다.
  반면 **추적 박스(ROI)는 평활한다** — 그건 출력이 아니라 입력 준비다.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.paths import output_dir                                    # noqa: E402
from pierrotfr.FA3D import FA3D                                         # noqa: E402
from pierrotfr.FA3D.render import (TRACK_COLORS, draw_landmarks,        # noqa: E402
                                   draw_mesh, label)

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VID_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm")


# ------------------------------------------------------------------ #
# 입력 해석 — 무엇이 들어왔는지 한 곳에서 정한다
# ------------------------------------------------------------------ #
def resolve_source(src: str) -> tuple[str, list]:
    if not os.path.exists(src):
        raise SystemExit(f"[FA3D] 입력을 찾을 수 없습니다: {src}")
    if os.path.isdir(src):
        files = sorted(os.path.join(src, f) for f in os.listdir(src)
                       if f.lower().endswith(IMG_EXT))
        if not files:
            raise SystemExit(f"[FA3D] 디렉토리에 이미지가 없습니다: {src}")
        return "images", files
    low = src.lower()
    if low.endswith(".txt"):
        with open(src, encoding="utf-8") as fh:
            files = [l.strip() for l in fh if l.strip()]
        return "images", files
    if low.endswith(VID_EXT):
        return "video", [src]
    if low.endswith(IMG_EXT):
        return "images", [src]
    raise SystemExit(f"[FA3D] 무슨 형식인지 모르겠습니다: {src}\n"
                     f"  이미지{IMG_EXT} · 디렉토리 · 목록.txt · 영상{VID_EXT}")


# ------------------------------------------------------------------ #
# 한 장 그리기 — 정지 이미지와 영상이 같은 함수를 쓴다
# ------------------------------------------------------------------ #
def annotate(fa: FA3D, img: np.ndarray, rois: list, params: list,
             mesh: bool, tids: list | None = None) -> np.ndarray:
    out = img
    for k, (roi, par) in enumerate(zip(rois, params)):
        L, V = fa.decode(par, roi, dense=mesh)
        col = TRACK_COLORS[(tids[k] if tids else k) % len(TRACK_COLORS)]
        out = draw_mesh(out, V, fa.tri, tint=col) if mesh else draw_landmarks(out, L, col)
    return out


def corner_tiles(fa: FA3D, frame: np.ndarray, rois: list, params: list,
                 tids: list, ratio: float) -> np.ndarray:
    """상단 좌측에 얼굴별 **3D 메쉬 크롭**을 세로로 쌓는다.

    본 화면은 68점, 모서리는 3D — 둘을 한 화면에서 본다. ⚠ 본 화면과 **같은
    파라미터**에서 그린다. 따로 추론하면 두 그림이 어긋난다.
    """
    res = int(min(frame.shape[:2]) * ratio)
    for q, (roi, par, tid) in enumerate(zip(rois[:3], params[:3], tids[:3])):
        _, V = fa.decode(par, roi, dense=True)
        sx, sy, ex, ey = roi
        pad = (ex - sx) * 0.12
        x0, y0 = int(sx - pad), int(sy - pad)
        x1, y1 = int(ex + pad), int(ey + pad)
        canvas = np.full((max(y1 - y0, 2), max(x1 - x0, 2), 3), 22, np.uint8)
        V3 = V.copy(); V3[0] -= x0; V3[1] -= y0
        col = TRACK_COLORS[tid % len(TRACK_COLORS)]
        tile = cv2.resize(draw_mesh(canvas, V3, fa.tri, alpha=1.0, tint=col),
                          (res, res), interpolation=cv2.INTER_AREA)
        px, py = 10, 10 + q * (res + 6)
        if py + res > frame.shape[0]:
            break
        frame[py:py + res, px:px + res] = tile
        cv2.rectangle(frame, (px, py), (px + res, py + res), col, 2)
        cv2.putText(frame, f"3D #{tid + 1}", (px + 5, py + res - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (235, 235, 235), 1, cv2.LINE_AA)
    return frame


# ------------------------------------------------------------------ #
# 정지 이미지
# ------------------------------------------------------------------ #
def run_images(fa: FA3D, files: list, a, outdir: str) -> None:
    done, t_det, t_net, n_face = [], 0.0, 0.0, 0
    for fp in files:
        img = cv2.imread(fp, cv2.IMREAD_COLOR)
        if img is None:
            print(f"  ⚠ 읽지 못했습니다: {fp}")
            continue
        t0 = time.perf_counter()
        rois = fa.detect(img, a.max_faces)
        t1 = time.perf_counter()
        params = [p for p in (fa.param(img, r) for r in rois) if p is not None]
        rois = rois[:len(params)]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        t_det += t1 - t0; t_net += t2 - t1; n_face += len(params)

        vis = annotate(fa, img, rois, params, a.mesh) if params else img
        vis = label(vis, "PIERROT FA3D", f"  {fa.spec.name}", TRACK_COLORS[0])
        out = os.path.join(outdir, os.path.basename(fp))
        cv2.imwrite(out, vis)
        done.append(vis)
        print(f"  {os.path.basename(fp):<40s} 얼굴 {len(params)}개 -> {out}")

    if not done:
        raise SystemExit("[FA3D] 처리된 이미지가 없습니다")
    n = len(done)
    print(f"[speed] 검출 {t_det / n * 1000:.1f}ms/장 · "
          f"추론 {t_net / max(n_face, 1) * 1000:.2f}ms/얼굴 "
          f"({n}장 · 얼굴 {n_face}개)")

    if a.grid > 0 and len(done) > 1:
        cols = a.grid
        rows = (len(done) + cols - 1) // cols
        h = max(im.shape[0] for im in done); w = max(im.shape[1] for im in done)
        sheet = np.full((rows * h, cols * w, 3), 24, np.uint8)
        for i, im in enumerate(done):
            r, c = divmod(i, cols)
            sheet[r * h:r * h + im.shape[0], c * w:c * w + im.shape[1]] = im
        gp = os.path.join(outdir, "grid.jpg")
        cv2.imwrite(gp, sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  격자 합본 -> {gp}")


# ------------------------------------------------------------------ #
# 영상 — 검출 게이팅 + 랜드마크 추적
# ------------------------------------------------------------------ #
def _ema(prev, cur, k):
    return cur if (prev is None or k >= 1.0) else [k * c + (1 - k) * p
                                                   for p, c in zip(prev, cur)]


def _drifted(prev, cur, ratio: float) -> bool:
    """중심이 박스 폭의 `ratio` 이상 튀면 추적이 깨진 것으로 본다."""
    if prev is None:
        return False
    pw = max(prev[2] - prev[0], 1e-6)
    dx = ((cur[0] + cur[2]) - (prev[0] + prev[2])) / 2
    dy = ((cur[1] + cur[3]) - (prev[1] + prev[3])) / 2
    return (dx * dx + dy * dy) ** 0.5 > pw * ratio


def _valid(roi, shape) -> bool:
    """추적 박스가 쓸 만한가 — 프레임 밖으로 나가거나 찌그러지면 버린다."""
    if roi is None:
        return False
    sx, sy, ex, ey = roi
    w, h = ex - sx, ey - sy
    if w < 16 or h < 16 or w > shape[1] * 4 or h > shape[0] * 4:
        return False
    return ex > 0 and ey > 0 and sx < shape[1] and sy < shape[0]


def _match(prev: list, cur: list, smooth: float, next_tid: list) -> list:
    """이전 프레임 트랙과 이번 검출을 중심 거리로 잇는다.

    트랙마다 EMA 와 색을 이어 가려면 누가 누구인지 알아야 한다. 얼굴 수가 적어
    단순 최근접으로 충분하다.
    """
    out, used = [], set()
    for c in cur:
        cx, cy = (c[0] + c[2]) / 2, (c[1] + c[3]) / 2
        best, bd = None, 1e18
        for i, pv in enumerate(prev):
            if i in used:
                continue
            px = (pv["roi"][0] + pv["roi"][2]) / 2
            py = (pv["roi"][1] + pv["roi"][3]) / 2
            d = (cx - px) ** 2 + (cy - py) ** 2
            if d < bd and d < ((c[2] - c[0]) * 0.6) ** 2:
                best, bd = i, d
        if best is not None:
            used.add(best)
            out.append({"roi": _ema(prev[best]["roi"], c, smooth),
                        "par": prev[best].get("par"), "tid": prev[best]["tid"]})
        else:
            out.append({"roi": c, "par": None, "tid": next_tid[0]})
            next_tid[0] += 1
    return out


def run_video(fa: FA3D, src: str, a, outdir: str) -> None:
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"[FA3D] 영상을 열지 못했습니다: {src}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_MSEC, a.start * 1000)
    step = max(int(round(src_fps / a.fps)), 1)
    limit = int(a.dur * a.fps) if a.dur > 0 else 10 ** 9

    frames, tracks, next_tid = [], [], [0]
    t_net, n_face, ok = 0.0, 0, True
    while len(frames) < limit:
        for _ in range(step):
            ok, img = cap.read()
            if not ok:
                break
        if not ok:
            break

        # ⚠ **얼굴이 없으면 그리지 않는다.** 추적만 하면 장면이 바뀌어도 박스가 남아
        #   배경에 랜드마크를 그린다 — 실제로 그랬다. 매 프레임 검출로 게이팅한다.
        tracks = _match(tracks, fa.detect(img, a.max_faces), a.smooth, next_tid)
        alive = []
        for t in tracks:
            if not _valid(t["roi"], img.shape):
                continue
            t0 = time.perf_counter()
            par = fa.param(img, t["roi"])
            t_net += time.perf_counter() - t0
            if par is None:
                continue
            n_face += 1
            # ⚠ **평활은 파라미터에 한 번만** 건다 — 랜드마크·메쉬·새 시점이 전부
            #   같은 값에서 나와야 서로 어긋나지 않는다. (기본은 꺼져 있다)
            if a.lmk_smooth > 0 and t["par"] is not None:
                par = a.lmk_smooth * par + (1 - a.lmk_smooth) * t["par"]
            L, _ = fa.decode(par, t["roi"], dense=False)
            cand = fa.track(L)
            alive.append({"roi": t["roi"] if _drifted(t["roi"], cand, a.drift)
                          else _ema(t["roi"], cand, a.smooth),
                          "par": par, "tid": t["tid"], "draw_roi": t["roi"]})
        tracks = alive

        if not alive:
            if not a.drop_empty:
                frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            continue

        rois = [t["draw_roi"] for t in alive]
        pars = [t["par"] for t in alive]
        tids = [t["tid"] for t in alive]
        vis = annotate(fa, img, rois, pars, a.mesh, tids)
        if a.corner3d > 0:
            vis = corner_tiles(fa, vis, rois, pars, tids, a.corner3d)
        sub = f"  {fa.spec.name}" + (f"   ·  smoothing {a.lmk_smooth:.1f}"
                                     if a.lmk_smooth > 0 else "")
        frames.append(cv2.cvtColor(label(vis, "PIERROT FA3D", sub, TRACK_COLORS[0]),
                                   cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise SystemExit("[FA3D] 얼굴을 찾지 못했습니다 — --start 를 바꿔 보세요")
    print(f"[speed] 추론 {t_net / max(n_face, 1) * 1000:.2f}ms/얼굴 "
          f"({len(frames)} 프레임 · 얼굴 {n_face}개)")

    stem = os.path.splitext(os.path.basename(src))[0]
    out = a.out or os.path.join(outdir, f"{stem}{'.gif' if a.gif else '.mp4'}")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    if out.endswith(".gif"):
        import imageio
        imageio.mimsave(out, frames, fps=a.fps, loop=0)
    else:
        # ⚠ cv2.VideoWriter 의 `mp4v` 는 브라우저·다수 플레이어가 못 연다 (실제로
        #   "동영상 로딩 에러"가 났다). 시스템 ffmpeg 에 raw 프레임을 밀어 넣어
        #   **H.264 + yuv420p + faststart** 로 낸다 — 어디서나 재생된다.
        h, w = frames[0].shape[:2]
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
               "-r", str(a.fps), "-i", "-",
               "-c:v", "libx264", "-preset", "medium", "-crf", str(a.crf),
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
        try:
            pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        except FileNotFoundError:
            raise SystemExit("[FA3D] ffmpeg 이 없습니다 — --gif 로 저장하세요") from None
        for f in frames:
            pr.stdin.write(np.ascontiguousarray(f).tobytes())
        pr.stdin.close()
        if pr.wait() != 0:
            raise SystemExit("[FA3D] ffmpeg 인코딩 실패")
    print(f"저장: {out}  ({len(frames)} 프레임 · {os.path.getsize(out) / 1e6:.1f}MB)")


# ------------------------------------------------------------------ #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--source", required=True,
                    help="이미지 | 디렉토리 | 목록.txt | 영상")
    ap.add_argument("--mesh", action="store_true",
                    help="68점 대신 **조밀 3D 메쉬**(정점 38,365)를 얹는다")
    ap.add_argument("--max-faces", type=int, default=4, help="동시에 처리할 얼굴 수")
    ap.add_argument("--outdir", default="", help="기본값 outputs/fa3d/<런 이름>/")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # -- 정지 이미지 --
    ap.add_argument("--grid", type=int, default=0,
                    help="격자 합본의 열 수 (0 = 안 만듦)")
    # -- 영상 --
    ap.add_argument("--start", type=float, default=0.0, help="시작 초")
    ap.add_argument("--dur", type=float, default=0.0, help="길이 초 (0 = 끝까지)")
    ap.add_argument("--fps", type=int, default=12, help="출력 fps")
    ap.add_argument("--crf", type=int, default=20, help="H.264 품질 (낮을수록 고화질)")
    ap.add_argument("--gif", action="store_true", help="mp4 대신 GIF 로 저장")
    ap.add_argument("--out", default="", help="출력 파일 경로를 직접 지정")
    ap.add_argument("--corner3d", type=float, default=0.0,
                    help="상단에 얼굴별 3D 메쉬 크롭을 띄운다 (0 = 끔)")
    ap.add_argument("--drop-empty", action="store_true",
                    help="얼굴이 없는 프레임을 아예 빼 버린다")
    ap.add_argument("--smooth", type=float, default=0.5,
                    help="추적 박스(ROI) EMA 계수 — 1.0 이면 평활 없음. "
                         "⚠ 랜드마크 출력은 이걸로 평활되지 않는다")
    ap.add_argument("--drift", type=float, default=0.35,
                    help="ROI 중심이 폭의 이 배율 이상 튀면 추적이 깨진 것으로 본다")
    ap.add_argument("--lmk-smooth", type=float, default=0.0,
                    help="62-d 파라미터 EMA (0 = 끔). ⚠ 논문 §2.4 는 시간 필터링에 "
                         "반대한다 — 평가에는 절대 쓰지 말 것. 켜면 화면에 표시된다")
    a = ap.parse_args()

    kind, files = resolve_source(a.source)
    fa = FA3D(ckpt=a.ckpt, device=a.device)
    outdir = a.outdir or output_dir(fa.spec.name)
    os.makedirs(outdir, exist_ok=True)
    print(f"입력 {a.source} ({kind} · {len(files)}건) -> {outdir}")

    if kind == "video":
        run_video(fa, files[0], a, outdir)
    else:
        run_images(fa, files, a, outdir)


if __name__ == "__main__":
    main()
