# coding: utf-8
"""임의의 사진·영상에 FA2D 를 돌린다 — 98 랜드마크.

    python eval/FA2D/infer.py --ckpt runs/fa2d/<런>/best.pth --source data/samples --grid 3
    python eval/FA2D/infer.py --ckpt … --source clip.mp4 --max-faces 2
    python eval/FA2D/infer.py --ckpt … --source clip.mp4 --gif --fps 12

크롭은 두 단계다 — ① 검출 박스로 1차 예측 → ② **예측 랜드마크로 학습과 같은 박스**를 다시
잡아 재예측(`--passes`). 영상에서는 이전 프레임 랜드마크로 ②만 돌고, 박스가 튀면 재검출한다.

⚠ **평가와 다른 경로다.** 벤치마크(eval/FA2D/evaluate.py)는 GT 로 크롭을 잡는다. 여기는
  검출기를 타므로 검출 품질이 결과에 섞인다 — 데모 그림을 평가 표와 나란히 놓지 말 것.
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
from pierrotfr.FA2D.infer import FA2D                                   # noqa: E402
from pierrotfr.FA2D.render import COLORS, draw_landmarks                # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VID_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def resolve_source(src: str):
    if not os.path.exists(src):
        raise SystemExit(f"[FA2D] 입력을 찾을 수 없습니다: {src}")
    if os.path.isdir(src):
        files = sorted(os.path.join(src, f) for f in os.listdir(src) if f.lower().endswith(IMG_EXT))
        if not files:
            raise SystemExit(f"[FA2D] 디렉토리에 이미지가 없습니다: {src}")
        return "images", files
    low = src.lower()
    if low.endswith(".txt"):
        return "images", [l.strip() for l in open(src, encoding="utf-8") if l.strip()]
    if low.endswith(VID_EXT):
        return "video", [src]
    if low.endswith(IMG_EXT):
        return "images", [src]
    raise SystemExit(f"[FA2D] 무슨 형식인지 모르겠습니다: {src}")


def run_images(fa: FA2D, files, a, outdir):
    done, t_all, n_face = [], 0.0, 0
    for fp in files:
        img = cv2.imread(fp)
        if img is None:
            print(f"  ⚠ 읽지 못했습니다: {fp}")
            continue
        t0 = time.perf_counter()
        faces = fa(img, a.max_faces)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_all += time.perf_counter() - t0
        n_face += len(faces)
        vis = img
        for k, lmk in enumerate(faces):
            vis = draw_landmarks(vis, lmk, fa.scheme, COLORS[k % len(COLORS)])
        out = os.path.join(outdir, os.path.basename(fp))
        cv2.imwrite(out, vis)
        done.append(vis)
        print(f"  {os.path.basename(fp):<40s} 얼굴 {len(faces)}개 -> {out}")
    if not done:
        raise SystemExit("[FA2D] 처리된 이미지가 없습니다")
    print(f"[speed] 검출+{fa.passes}패스 추론 {t_all / len(done) * 1000:.1f}ms/장 "
          f"({len(done)}장 · 얼굴 {n_face}개)")
    if a.grid > 0 and len(done) > 1:
        h = max(im.shape[0] for im in done); w = max(im.shape[1] for im in done)
        rows = (len(done) + a.grid - 1) // a.grid
        sheet = np.full((rows * h, a.grid * w, 3), 24, np.uint8)
        for i, im in enumerate(done):
            r, c = divmod(i, a.grid)
            sheet[r * h:r * h + im.shape[0], c * w:c * w + im.shape[1]] = im
        gp = os.path.join(outdir, "grid.jpg")
        cv2.imwrite(gp, sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  격자 합본 -> {gp}")


def _center_jump(prev, cur, ratio):
    """랜드마크 중심이 얼굴 폭의 ratio 이상 튀면 추적이 깨진 것으로 본다."""
    w = max(np.ptp(prev[:, 0]), 1e-6)
    return np.linalg.norm(cur.mean(0) - prev.mean(0)) > w * ratio


def run_video(fa: FA2D, src, a, outdir):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"[FA2D] 영상을 열지 못했습니다: {src}")
    step = max(int(round((cap.get(cv2.CAP_PROP_FPS) or 30.0) / a.fps)), 1)
    cap.set(cv2.CAP_PROP_POS_MSEC, a.start * 1000)
    limit = int(a.dur * a.fps) if a.dur > 0 else 10 ** 9
    frames, tracks, n, t_net, n_face, ok = [], [], 0, 0.0, 0, True
    while len(frames) < limit:
        for _ in range(step):
            ok, img = cap.read()
            if not ok:
                break
        if not ok:
            break
        n += 1
        t0 = time.perf_counter()
        # 추적 — 이전 랜드마크로 학습 규약 박스를 잡는다(② 단계만). 주기적으로 재검출한다.
        if not tracks or n % a.redetect == 0:
            tracks = fa(img, a.max_faces)
        else:
            new = []
            for lmk in tracks:
                cur = fa.refine(img, lmk)
                if _center_jump(lmk, cur, a.drift):
                    new = fa(img, a.max_faces)
                    break
                new.append(cur)
            tracks = new
        t_net += time.perf_counter() - t0
        n_face += len(tracks)
        if not tracks and a.drop_empty:
            continue
        vis = img
        for k, lmk in enumerate(tracks):
            vis = draw_landmarks(vis, lmk, fa.scheme, COLORS[k % len(COLORS)])
        frames.append(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise SystemExit("[FA2D] 얼굴을 찾지 못했습니다 — --start 를 바꿔 보세요")
    print(f"[speed] {t_net / max(len(frames), 1) * 1000:.1f}ms/프레임 "
          f"({len(frames)} 프레임 · 얼굴 {n_face}개)")
    stem = os.path.splitext(os.path.basename(src))[0]
    out = a.out or os.path.join(outdir, f"{stem}{'.gif' if a.gif else '.mp4'}")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    if out.endswith(".gif"):
        import imageio
        imageio.mimsave(out, frames, fps=a.fps, loop=0)
    else:
        # ⚠ cv2.VideoWriter 의 mp4v 는 브라우저가 못 연다 — ffmpeg 으로 H.264 + yuv420p
        h, w = frames[0].shape[:2]
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{w}x{h}", "-r", str(a.fps), "-i", "-", "-c:v", "libx264",
               "-crf", str(a.crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
        try:
            pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        except FileNotFoundError:
            raise SystemExit("[FA2D] ffmpeg 이 없습니다 — --gif 로 저장하세요") from None
        for f in frames:
            pr.stdin.write(np.ascontiguousarray(f).tobytes())
        pr.stdin.close()
        if pr.wait() != 0:
            raise SystemExit("[FA2D] ffmpeg 인코딩 실패")
    print(f"저장: {out}  ({len(frames)} 프레임 · {os.path.getsize(out) / 1e6:.1f}MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--source", required=True, help="이미지 | 디렉토리 | 목록.txt | 영상")
    ap.add_argument("--max-faces", type=int, default=4)
    ap.add_argument("--passes", type=int, default=2,
                    help="크롭 반복 횟수. 2 부터 학습과 같은 크롭 규약이 된다")
    ap.add_argument("--tta", action="store_true", help="좌우반전 TTA (추론 2배)")
    ap.add_argument("--outdir", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--grid", type=int, default=0, help="격자 합본 열 수 (0 = 안 만듦)")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=0.0)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--drop-empty", action="store_true")
    ap.add_argument("--redetect", type=int, default=30, help="N 프레임마다 재검출")
    ap.add_argument("--drift", type=float, default=0.35,
                    help="랜드마크 중심이 얼굴 폭의 이 배율 이상 튀면 재검출")
    a = ap.parse_args()

    kind, files = resolve_source(a.source)
    fa = FA2D(ckpt=a.ckpt, device=a.device, tta=a.tta, passes=a.passes)
    outdir = a.outdir or output_dir(fa.spec.name, "fa2d")
    os.makedirs(outdir, exist_ok=True)
    print(f"입력 {a.source} ({kind} · {len(files)}건) -> {outdir}")
    (run_video if kind == "video" else run_images)(fa, files[0] if kind == "video" else files, a, outdir)


if __name__ == "__main__":
    main()
