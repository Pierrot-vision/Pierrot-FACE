"""FA3D 추론·평가 설정 — **체크포인트가 모르는 것만** 담는다.

학습 저장소의 `configs/args_fa3d.py` 는 626 줄짜리 하이퍼파라미터 단일 소스였다.
여기는 그게 필요 없다. 모델 쪽 값은 전부 체크포인트가 들고 있기 때문이다:

    model_name · num_params · use_lrr · model_extra · image_size · aug_border

`pierrotfr/FA3D/infer.py:load_checkpoint()` 가 그걸 복원한다. 그래서 이 파일에
남는 것은 체크포인트가 **알 수 없는 것**뿐이다 — 이 서버의 평가 데이터가 어디
있고, BFM 기저와 정규화 통계가 어디 있는가.

⚠ `aug_border` 만은 예외적으로 체크포인트에서 읽어 **평가 전처리에 그대로 건다.**
  테두리 0 처리는 증강이 아니라 학습·평가 공통 전처리라, 학습 때 쓴 값과 다르면
  모델이 못 본 분포가 들어온다 (실측: 값 하나를 빠뜨려 NME 를 3.688 이 아니라
  3.948 로 잘못 쟀다). 그래서 사람이 고르는 값이 아니라 체크포인트가 정한다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .paths import (FA3D_BFM_DIR, FA3D_TEST_CFG, FA3D_TEST_DIR, FA3D_TRAIN_CFG,
                    FA3D_TRAIN_DIR, describe, env_hint)


@dataclass
class EvalConfig:
    # ---- BFM (모델 출력 62-d 를 정점/랜드마크로 펴는 데 필수) ----------
    # 3DDFA v1 basis(53,215 정점)와 V2 bfm_noneck_v3(38,365)는 목 정점만 제거·재색인한
    # 것이고 62-d 파라미터 공간은 동일하다. 배포 가중치와 우리 체크포인트를 같은
    # 기저로 디코딩해도 되는 근거다.
    bfm_fp    : str = f"{FA3D_BFM_DIR}/bfm_noneck_v3.pkl"
    # 62-d Z-score 통계. 학습이 이 공간으로 재정규화해 두었으므로 **반드시 같은 파일**.
    dst_stats : str = f"{FA3D_BFM_DIR}/param_mean_std_62d_120x120.pkl"
    # ⚠ bfm_noneck_v3.pkl 안의 'tri' 는 쓰면 안 된다 — 인덱스 최대가 46,851 로 이
    #   파일의 정점 수 38,365 를 넘는다 (목 제거 **전** 기저의 삼각형이 남아 있다).
    #   오피셜도 별도 configs/tri.pkl 을 쓴다.
    tri_fp    : str = f"{FA3D_BFM_DIR}/tri.pkl"

    # ---- 평가셋 -------------------------------------------------------
    # 3DDFA v1 이 배포한 **사전 크롭 120x120**. 얼굴 검출을 타지 않으므로 검출기
    # 성능이 지표에 섞이지 않는다.
    test_dir  : str = FA3D_TEST_DIR
    test_cfg  : str = FA3D_TEST_CFG

    # ---- 300W-LP val (형상 정확도 · 학습 도메인 대조 전용) --------------
    # 추론만 할 거면 없어도 된다. AFLW2000-3D 에는 68점 GT 만 있고 62-d 파라미터가
    # 없어서, identity 정확도를 재려면 이쪽이 필요하다.
    train_root : str = FA3D_TRAIN_DIR
    val_list   : str = f"{FA3D_TRAIN_CFG}/train_aug_120x120.list.val"
    val_param  : str = f"{FA3D_TRAIN_CFG}/param_all_norm_val.pkl"
    src_stats  : str = f"{FA3D_TRAIN_CFG}/param_whitening.pkl"

    # ---- 실행 ---------------------------------------------------------
    # GT 좌표계는 120 이다. 백본 입력만 리사이즈될 수 있고(yolo 계열 128),
    # 그건 체크포인트의 model_extra['input_size'] 가 정한다.
    image_size  : int = 120
    batch_size  : int = 256
    num_workers : int = 8


def config() -> EvalConfig:
    return EvalConfig()


# ------------------------------------------------------------------ #
# 무엇이 실제로 있는지 — 없는 것을 미리 말해 준다.
#
# 존재만 보지 않고 '무엇을 못 하게 되는지'까지 알린다. BFM 이 없으면 아무것도
# 못 하지만, 300W-LP val 이 없으면 형상 분석만 못 한다 — 그 둘을 같은 강도로
# 막으면 추론만 하려는 사람이 데이터 640GB 를 받아야 한다고 오해한다.
# ------------------------------------------------------------------ #
def available(cfg: EvalConfig) -> dict:
    return {
        "bfm": os.path.isfile(cfg.bfm_fp) and os.path.isfile(cfg.dst_stats),
        "tri": os.path.isfile(cfg.tri_fp),
        "aflw2000": os.path.isdir(f"{cfg.test_dir}/AFLW2000-3D_crop"),
        "aflw": os.path.isdir(f"{cfg.test_dir}/AFLW_GT_crop"),
        "300wlp_val": os.path.isdir(cfg.train_root) and os.path.isfile(cfg.val_param),
    }


def require_bfm(cfg: EvalConfig) -> EvalConfig:
    """BFM 이 없으면 62-d 를 얼굴로 펼 수 없다 — 추론 자체가 불가능하다."""
    miss = [p for p in (cfg.bfm_fp, cfg.dst_stats) if not os.path.isfile(p)]
    if miss:
        raise SystemExit(
            "[FA3D] BFM 파일이 없습니다:\n"
            + "\n".join(f"    {p}" for p in miss)
            + "\n  3DDFA_V2 오피셜 저장소의 configs/ 에서 가져올 수 있습니다:\n"
              "    bfm_noneck_v3.pkl · param_mean_std_62d_120x120.pkl · tri.pkl\n"
              f"  (현재 {describe()}){env_hint()}")
    return cfg


def require_testset(cfg: EvalConfig, which: str = "aflw2000") -> EvalConfig:
    have = available(cfg)
    if not have.get(which):
        raise SystemExit(
            f"[FA3D] 평가셋이 없습니다: {which} ({cfg.test_dir})\n"
            f"  3DDFA v1 이 배포한 test.data / test.configs 가 필요합니다.\n"
            f"  (현재 {describe()}){env_hint()}")
    return cfg


def require_trainval(cfg: EvalConfig) -> EvalConfig:
    if not available(cfg)["300wlp_val"]:
        raise SystemExit(
            f"[FA3D] 300W-LP val 이 없습니다: {cfg.train_root}\n"
            f"  형상(identity) 정확도는 62-d GT 파라미터가 있어야 잴 수 있고,\n"
            f"  AFLW2000-3D 에는 68점 GT 만 있습니다. 추론·NME 평가에는 필요 없습니다.\n"
            f"  (현재 {describe()}){env_hint()}")
    return cfg
