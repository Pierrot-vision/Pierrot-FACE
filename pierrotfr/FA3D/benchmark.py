"""AFLW2000-3D / AFLW 벤치마크.

평가셋은 3DDFA v1 이 배포한 **사전 크롭된 120x120** 이미지다. 얼굴 검출을 타지
않으므로 검출기 성능이 섞이지 않는다. 저장된 roi_box 가 3DDFA_V2 의
`parse_roi_box_from_landmark` 와 0.58px(0.2%) 차이라 크롭 규약도 일치한다.

⚠ 학습 저장소의 같은 파일과 **인자 규약이 다르다.** 그쪽은 학습 config(cfg) 하나에서
  데이터 경로와 전처리 값을 함께 읽었다. 여기서는 둘의 출처를 갈랐다:

    데이터 경로   configs/eval_fa3d.py  (이 서버의 값)
    전처리 값     ModelSpec             (그 체크포인트가 학습 때 쓴 값)

  체크포인트에 박힌 절대경로는 학습 서버의 것이라 믿을 수 없고, 반대로 테두리
  전처리(border)는 이 서버가 정할 값이 아니라 그 가중치가 정한 값이기 때문이다.
"""
from __future__ import annotations

import numpy as np

from .infer import ModelSpec, predict_filelist, to_landmarks
from .metrics import nme_aflw, nme_aflw2000, summarize

# 평가셋별 파일 규약 — 이름이 셋 다 조금씩 다르다(오피셜 배포본을 그대로 쓴다).
SETS = {
    "aflw2000": {
        "dir": "AFLW2000-3D_crop", "list": "AFLW2000-3D_crop.list",
        "roi": "AFLW2000-3D_crop.roi_box.npy", "yaw": "AFLW2000-3D.pose.npy",
        "label": "AFLW2000-3D",
    },
    "aflw": {
        "dir": "AFLW_GT_crop", "list": "AFLW_GT_crop.list",
        "roi": "AFLW_GT_crop_roi_box.npy", "yaw": "AFLW_GT_crop_yaws.npy",
        "label": "AFLW",
    },
}


def paths(cfg, which: str) -> tuple[str, str]:
    s = SETS[which]
    return f"{cfg.test_dir}/{s['dir']}", f"{cfg.test_dir}/{s['list']}"


def ground_truth(cfg, which: str, gt_option: str = "ori") -> dict:
    """평가에 필요한 GT 를 한 번에 읽는다 — roi · yaw · 랜드마크."""
    s, c = SETS[which], cfg.test_cfg
    out = {"roi": np.load(f"{c}/{s['roi']}"), "yaw": np.load(f"{c}/{s['yaw']}")}
    if which == "aflw2000":
        fp = ("AFLW2000-3D.pts68.npy" if gt_option == "ori"
              else "AFLW2000-3D-Reannotated.pts68.npy")
        out["pts68"] = np.load(f"{c}/{fp}")
    else:
        out["pts68"] = np.load(f"{c}/AFLW_GT_pts68.npy")
        out["pts21"] = np.load(f"{c}/AFLW_GT_pts21.npy")
    return out


def nme(cfg, which: str, lmk: list, gt: dict) -> np.ndarray:
    """샘플별 NME (0~1 스케일). 평균 내기 전 값이라 오차 분석이 여기서 갈린다."""
    if which == "aflw2000":
        return nme_aflw2000(lmk, gt["roi"], gt["pts68"], cfg.image_size)
    return nme_aflw(lmk, gt["roi"], gt["pts68"], gt["pts21"], cfg.image_size)


def predict(model, spec: ModelSpec, cfg, which: str, device="cuda") -> np.ndarray:
    """평가셋 전체 -> [N, 62] (정규화 공간)."""
    root, lst = paths(cfg, which)
    return predict_filelist(model, root, lst, device, border=spec.border,
                            batch_size=cfg.batch_size, num_workers=cfg.num_workers)


def evaluate(model, spec: ModelSpec, bfm, param_norm, cfg,
             which: str = "aflw2000", gt_option: str = "ori",
             device="cuda") -> tuple[dict, np.ndarray]:
    """(요약 지표, 샘플별 NME) — 요약은 논문 규약(yaw 3구간 평균의 평균)."""
    params = predict(model, spec, cfg, which, device)
    lmk = to_landmarks(bfm, param_norm, params, device, cfg.image_size)
    gt = ground_truth(cfg, which, gt_option)
    per = nme(cfg, which, lmk, gt)
    return summarize(per, gt["yaw"]), per
