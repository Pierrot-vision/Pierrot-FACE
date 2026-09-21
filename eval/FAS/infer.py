"""단일 이미지 / 폴더 추론 + cue map 시각화.

    python eval/FAS/infer.py --ckpt runs/fas/<런>/best.pth --image face.jpg --align
    python eval/FAS/infer.py --ckpt … --dir imgs/ --align --margin -33.3 --save_cue

추론 경로는 **LLM 을 타지 않는다** (백본 → Q-Former ×2 → 융합 → 분류기).

입력은 학습 데이터와 같은 **MTCNN 5점 정렬 crop** 이어야 한다. 원본 사진이면 `--align`
을 준다 (없으면 이미 정렬된 crop 으로 보고 그대로 넣는다).

판정 임계값은 margin(spoof log-odds) 단위다.
    --margin 0      CelebA-Spoof 공식 규칙 (확률 0.5)
    --margin <val>  val 에서 BPCER 1% 로 정한 운영점 — eval/FAS/evaluate.py 가 출력한다
                    (clip_instructflip_clip-vit-b16 은 -33.3)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.paths import output_dir          # noqa: E402
from pierrotfr.FAS import load_checkpoint     # noqa: E402


def overlay_cue(bgr: np.ndarray, cue: np.ndarray) -> np.ndarray:
    """cue map 을 원본 위에 히트맵으로 얹는다.

    ⚠ cue map 은 정식 세그멘테이션이 아니다(위조영역 주석 없이 학습된 보조 감독).
      "모델이 어디를 봤나"의 참고일 뿐 위조 영역의 근거로 쓰지 말 것.
    """
    m = cue - cue.min()
    m = m / (m.max() + 1e-8)
    heat = cv2.applyColorMap((m * 255).astype("uint8"), cv2.COLORMAP_JET)
    heat = cv2.resize(heat, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(bgr, 0.6, heat, 0.4, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--align", action="store_true", help="MTCNN 5점 정렬 후 추론 (원본 사진일 때)")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="spoof 판정 임계값 (margin ≥ 이 값이면 SPOOF). 0 = 확률 0.5")
    ap.add_argument("--save_cue", action="store_true",
                    help="cue 오버레이를 outputs/fas/<런>/cue/ 에 저장")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    if not a.image and not a.dir:
        raise SystemExit("--image 또는 --dir 가 필요합니다")

    fas, spec = load_checkpoint(a.ckpt, a.device)
    tf = spec.transform()

    aligner = None
    if a.align:
        from pierrotfr.data.align import FaceAligner
        aligner = FaceAligner(device=a.device)

    files = ([Path(a.image)] if a.image
             else sorted(p for p in Path(a.dir).rglob("*")
                         if p.suffix.lower() in (".jpg", ".jpeg", ".png")))
    dst = Path(output_dir(spec.name, "fas")) / "cue"

    for f in files:
        bgr = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"{f}: 읽기 실패")
            continue
        status = "raw"
        if aligner is not None:
            chip, status = aligner(bgr)
            if chip is None:
                print(f"{f}: 얼굴 검출 실패 — 건너뜀")
                continue
            bgr = chip

        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        out = fas(tf(img).unsqueeze(0))
        margin, prob = float(out["margin"][0]), float(out["score"][0])
        verdict = "SPOOF" if margin >= a.margin else "LIVE"
        print(f"{f.name:40s} {verdict:5s}  margin={margin:8.2f}  p(spoof)={prob:.4f}  ({status})")

        if a.save_cue and out["cue"] is not None:
            dst.mkdir(parents=True, exist_ok=True)
            cue = out["cue"][0, 0].float().cpu().numpy()
            cv2.imwrite(str(dst / f"{f.stem}_cue.jpg"), overlay_cue(bgr, cue))
    if a.save_cue:
        print(f"cue -> {dst}")


if __name__ == "__main__":
    main()
