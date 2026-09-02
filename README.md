<p align="center">
  <img src="docs/peirrot-face-banner.jpg" width="100%" alt="PIERROT FACE banner"/>
</p>

<h1 align="center">🎭 PIERROT FACE — FA3D</h1>

<p align="center">
  <b>3D 밀집 얼굴 정렬(3DDFA_V2 재구현) 추론 배포본</b>
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
  <b>한국어</b> | <a href="README_en.md">English</a>
</p>

---

## 💡 소개

**PIERROT FACE** 는 1인 **얼굴(Face) Vision** 연구·개발 프로젝트입니다.
위조 방지 / 3D 정렬 / 인식 을 하나의 파이프라인에 뭉뚱그리지 않고
**큰 카테고리로 분리**해, 각 단계의 알고리즘을 독립적으로 재현·교체·비교할 수 있게
만드는 것이 목표입니다.

이 저장소는 그중 **FA3D(3D Dense Face Alignment)의 추론 경로만** 떼어 낸 배포본입니다.
학습 저장소에서 손실·최적화·증강·학습 데이터셋을 걷어내고, **체크포인트 하나로 얼굴을
내는 데 필요한 것만** 남겼습니다.

> **이름의 유래** — 피에로(Pierrot)는 원래 무언극에서 **남을 따라 하고 흉내 내는**
> 광대 캐릭터입니다. 기존 연구의 좋은 점을 따라 재현·결합한다는
> [Pierrot Universe](https://github.com/Pierrot-vision) 의 첫 번째 철학(MimiC)과 맞닿아
> 있어 이 이름을 씁니다.

1. **차용과 재현 우선 (MimiC)** — 검증된 구현을 **정확히 재현한 뒤 결합**하는 것이 기본
   방향입니다. 오피셜 코드와 대조해 정합을 확인하고, 다른 점은 표로 남깁니다.
   **학습 코드가 공개되지 않은 논문은 수식에서 직접 구현**하고 그 근거를 문서에 적습니다
   (3DDFA_V2 가 그 경우입니다).
2. **카테고리 · 알고리즘 분리** — 각 카테고리가 서로의 코드에 침범하지 않고, 카테고리
   안에서 알고리즘을 갈아 끼울 수 있습니다.
3. **기록 우선** — 분석 · 설계 근거 · 실행 방법은 전부 [LAB](LAB/) 에 남깁니다.
   **비교가 대등하지 않으면 그 사실도 함께 적습니다.**
4. **개인적 호기심의 실험장** — 무엇보다, 만드는 사람의 호기심을 푸는 공간입니다.

* 현재 **학습 코드와 학습된 가중치는 공개하지 않고 있습니다.**

## 📰 News

- 2026-09-02 — 🚀 **FA3D 추론 코드 공개** — 사진·영상 추론 · AFLW2000-3D/AFLW 평가 ·
  3D 재렌더 · 형상 정확도. 학습 저장소의 수치를 **소수점 세 자리까지 재현**했습니다
- 2026-09-01 — 🎯 **배포 가중치를 넘었습니다** — AFLW2000-3D **3.622** vs 배포 3.683
  (**−0.061**). 증강 · 순환 · 가림 · identity 일관성을 합쳐 얻었습니다
  👉 [Phase 5](LAB/FA3D/Exp/Phase_5_증강과_svs.md)
- 2026-08-27 — 🧊 **FA3D(3DDFA_V2 학습부) 재구현** — 저자 저장소에 없는 fWPDC ·
  meta-joint · landmark-regression 정규화를 논문에서 직접 구현, 배포 가중치 3.683 재현
  👉 [Phase 1](LAB/FA3D/Exp/Phase_1_구현과_기준선.md)

## 🧊 3D Dense Face Alignment (FA3D)

**3DDFA_V2 (ECCV 2020)** 는 추론 코드와 배포 가중치만 공개되어 있고 **학습 코드가 없습니다.**
그래서 논문의 수식에서 직접 재구현했습니다. 이 저장소는 **그 결과를 돌려 보는 코드**입니다.

- 📘 **알고리즘** (수식 · 유도 · 파라미터 의미 · 평가 규약) — [LAB/FA3D/3DDFA_V2.md](LAB/FA3D/3DDFA_V2.md)
- 📗 **구현 · 실험 기록** (버그 · 측정 · 분석) — [LAB/FA3D/FA3D.md](LAB/FA3D/FA3D.md)
- 📙 **실험 단계별 기록** — [LAB/FA3D/Exp/](LAB/FA3D/Exp/) (Phase 1~10)

### 우리 실제 예측 결과 (영상 데모)

![FA3D 영상 데모](docs/FA3D/demo.gif)

### 평가 결과

NME 는 **yaw 3구간 평균** 규약(논문과 동일, 단순 평균이 아님).
아래는 전부 **이 저장소의 `eval/FA3D/evaluate.py` 로 직접 잰 값**입니다 — 우리 체크포인트와
저자 배포 가중치를 같은 코드·같은 크롭·같은 GT 로 재야 비교가 성립합니다.

| 모델 | AFLW2000-3D ↓ | AFLW (21점) ↓ |
|---|---|---|
| **Ours** | **3.622** | **5.125** |
| — | — | — |
| 🎯 3DDFA_V2 배포 mb1 | *3.683* | *4.600* |
| 3DDFA_V2 배포 mb05 | *3.798* | *4.738* |
| 논문 M+R | *3.590* | *4.500* |
| 논문 M+R+S | *3.510* | *4.430* |

> ⚠ **저자가 공개한 가중치는 논문 수치를 재현하지 못합니다** — 같은 평가 코드로 재면 mb1 이 3.683 / 4.600 으로 논문 M+R(3.590 / 4.500)보다 두 벤치마크에서 **일관되게 +0.09~0.10** 나쁩니다. 그래서 이 저장소의 재현 목표는 논문 수치가 아니라 **배포 가중치**입니다.

> 상세 실험 내용은 [LAB/FA3D/FA3D.md](LAB/FA3D/FA3D.md) 참조.

### 예측 결과 — AFLW2000-3D

![AFLW2000-3D 예측 50장](docs/FA3D/aflw2000_grid_spread.jpg)

오차 백분위 p00(최선)~p99(최악)에서 균등하게 뽑은 50장. 🟢 초록 = 우리 예측 ·
🔴 빨강 = GT 68점, **제목이 초록이면 그 장에서 배포 가중치를 이겼다는 뜻**입니다.
[최악 50장](docs/FA3D/aflw2000_grid_worst.jpg) 도 함께 있습니다.

### 3D 복원 결과 — 같은 예측을 새 시점에서 다시 그린 것

![3D 밀집 복원](docs/FA3D/pred_3d.jpg)

## 🚀 Install

```bash
git clone https://github.com/Pierrot-vision/Pierrot-FACE
cd Pierrot-FACE

conda create -n fr python=3.9 -y
conda activate fr
pip install -r requirements.txt

# 서버 고유 경로 등록 — 한 번만 작성해 두면 매번 export 하지 않아도 된다
cp paths.local.env.example paths.local.env
#   PIERROTFR_CKPT_ROOT 를 학습 저장소의 runs/ 로 가리키는 것이 핵심이다
```

**최소 준비물은 BFM 세 파일입니다.** 3DDFA_V2 오피셜 저장소의 `configs/` 에서
가져와 `$PIERROTFR_FA3D_DATA_ROOT/bfm/` 에 두세요:

```
bfm_noneck_v3.pkl                  38,365 정점 3DMM 기저
param_mean_std_62d_120x120.pkl     62-d Z-score 통계 (모델 출력 공간)
tri.pkl                            76,073 삼각형 — 메쉬 렌더용
```

지표를 재려면 3DDFA v1 이 배포한 평가셋(`test.data/` · `test.configs/`)이,
형상 정확도까지 재려면 300W-LP 전처리본(`train.configs/`)이 더 필요합니다.
무엇이 있고 무엇이 없는지는 실행할 때 알려 줍니다.

> 검증 환경: RTX 6000 Ada / torch 2.x / CUDA 12.
> ffmpeg 이 있으면 영상이 H.264 로 저장됩니다 (없으면 `--gif` 를 쓰세요).
> 얼굴 검출기는 **FaceBoxes → MTCNN → Haar cascade** 순으로 폴백하므로,
> 아무것도 설치하지 않아도 데모는 돕니다 (정면만 잡히는 대신).

## ⚡ 실행

**`.sh` 스크립트는 인자를 받지 않습니다.** 바꿀 값은 각 파일 상단의
`---- 여기만 바꾼다 ----` 블록에 있습니다. 파이썬 엔트리를 직접 부르면 전부 CLI 인자입니다.

```bash
# ── 추론 ──────────────────────────────────────────────────────────
# 이미지 · 디렉토리 · 목록.txt · 영상을 모두 받는다. 속도를 항상 함께 출력한다
python eval/FA3D/infer.py --ckpt runs/fa3d/<런>/best.pth --source data/samples --grid 3
python eval/FA3D/infer.py --ckpt … --source clip.mp4 --mesh --corner3d 0.24 --drop-empty
python eval/FA3D/infer.py --ckpt … --source clip.mp4 --gif --fps 12

# ── 평가 ──────────────────────────────────────────────────────────
# 배포 가중치를 같은 코드로 함께 잰다 — 이것이 유일하게 유효한 비교다
python eval/FA3D/evaluate.py --ckpt runs/fa3d/<런>/best.pth --deployed mb1 --aflw

# ── 분석 ──────────────────────────────────────────────────────────
python scripts/FA3D/pred_grid.py      --ckpt … --mode spread  # 예측 격자 (2D)
python scripts/FA3D/pred_grid.py      --ckpt … --mode worst   # 실패 유형
python scripts/FA3D/pred_3d.py        --ckpt …                # 3D 밀집 복원 (새 시점 재렌더)
python scripts/FA3D/pred_3d.py        --ckpt … --compare      # identity: GT vs 우리 vs 배포
python scripts/FA3D/shape_accuracy.py --ckpt …                # 형상(identity) 정확도
python scripts/FA3D/error_analysis.py --ckpt …                # 샘플 단위 오차 분해
python scripts/FA3D/diff_grid.py      --a … --b …             # 두 런이 갈린 표본
python scripts/FA3D/bench_backbone.py                         # 백본 지연시간 · MACs

# ── 묶음 ──────────────────────────────────────────────────────────
bash scripts/FA3D/eval.sh                     # 평가 한 벌
VIDEO=clip.mp4 bash scripts/FA3D/infer.sh     # 추론 · 그림 · 속도 한 벌
```

**체크포인트가 자기 설정을 들고 있습니다** — 백본 · 입력 크기 · 테두리 전처리를
다시 지정할 필요가 없고, 지정할 수도 없습니다. 특히 테두리 0 처리(`aug_border`)는
학습·평가 **공통 전처리**라 사람이 고르면 안 됩니다: 학습 때와 다른 값을 걸면
모델이 못 본 분포가 들어와, 실측으로 같은 가중치를 3.688 이 아니라 3.948 로 잘못
잰 적이 있습니다.

## 📦 구조

```
Pierrot_FR_Infer/
├── configs/
│   ├── paths.py            # 🗺️ 데이터/가중치/산출물 루트 (paths.local.env · 환경변수)
│   └── eval_fa3d.py        #    평가셋·BFM 경로 — 체크포인트가 모르는 것만
├── paths.local.env         # 서버 고유 경로 — 커밋 제외 (.example 을 복사해 쓴다)
├── pierrotfr/FA3D/         # 📦 추론 경로만
│   ├── infer.py            #   체크포인트 로드 · 배치 추론 · FA3D 클래스 ★진입점
│   ├── bfm.py              #   3DMM 디코더 — 62-d → 38,365 정점 / 68 랜드마크
│   ├── crop.py             #   얼굴 크롭 규약 (3DDFA_V2 이식) — 정확도의 절반
│   ├── data.py             #   전처리 ((x−127.5)/128 · 테두리 0) + 평가 데이터셋
│   ├── detect.py           #   얼굴 검출 (FaceBoxes → MTCNN → Haar 폴백) · 데모 전용
│   ├── render.py           #   점 스플랫 렌더러 — 새 시점 · 메쉬 오버레이 · 68점
│   ├── metrics.py          #   AFLW2000-3D / AFLW NME (yaw 구간 규약)
│   ├── benchmark.py        #   평가셋 파이프라인 (예측 → 랜드마크 → NME)
│   └── models/             #   백본 — lrr 헤드도 synergy 순환도 만들지 않는다
├── eval/FA3D/
│   ├── infer.py            # 🔍 사진·영상 → 68점 / 3D 메쉬 (+H.264/GIF, 속도 계측)
│   └── evaluate.py         #    AFLW2000-3D / AFLW NME (여러 ckpt + 배포 가중치)
├── scripts/FA3D/           # 🧪 그림·분석·속도 (eval.sh · infer.sh 로 묶음 실행)
├── LAB/                    # 📚 학습 저장소의 실험 기록 사본 (읽기 전용)
├── docs/                   # 배너 · 결과 시각화 · 데모
├── data/samples/           # 데모용 사진 — 커밋 제외
├── runs/                   # (심볼릭) 학습 저장소의 체크포인트 — 읽기만 한다
└── outputs/fa3d/<런>/      # 추론 시각화 — 재생성 가능, 커밋 제외
```

### 학습 저장소에서 빠진 것

| 빠진 것 | 왜 |
|---|---|
| `train_fa3d.py` · `engine.py` | meta-joint 최적화(Algorithm 1) — 학습 루프 |
| `losses.py` | VDC · fWPDC · lrr · 파라미터 손실 |
| `svs.py` | short-video-synthesis — 학습 증강 |
| `models/synergy.py` | SynergyNet 순환 (MLP_for/MLP_rev) — `predict()` 를 안 탄다 |
| 학습 데이터셋 · `Augment` | 300W-LP 로더 · ColorJitter · 가림 · identity 그룹 |
| `configs/args_fa3d.py` (626줄) | 하이퍼파라미터 — **체크포인트가 들고 있다** |
| `fc_lm` 헤드 | 논문 2.3절 lrr 정규화 — 저자 배포본도 담아 두고 버린다 |

lrr 헤드와 synergy 순환은 **추론 비용이 0** 입니다 (`predict()` 경로에 없습니다).
이 저장소는 한 걸음 더 가서 **모듈 자체를 만들지 않으므로**, 파라미터 수와 지연시간이
곧 실제로 배포되는 모델의 값입니다 (`python scripts/FA3D/bench_backbone.py`).

## 🤗 Reference

- PIERROT FACE 는 기존 연구들의 좋은 점들을 재현·결합하여 만들어집니다.
- 저는 이 연구들에 대해 항상 감사한 마음을 가집니다.

📚 원본 논문 —
[3DDFA_V2 (ECCV 2020)](https://guojianzhu.com/assets/pdfs/3162.pdf) ·
[SynergyNet (3DV 2021)](https://arxiv.org/abs/2110.09772)

🛠 오피셜 코드 — [cleardusk/3DDFA_V2](https://github.com/cleardusk/3DDFA_V2) (BFM · 크롭 규약 · 배포 가중치의 이식 원본) · [cleardusk/3DDFA](https://github.com/cleardusk/3DDFA) (v1 — 평가셋 · 텐서 처리 참조) · [choyingw/SynergyNet](https://github.com/choyingw/SynergyNet)

## 📄 라이센스

이 프로젝트(코드 · 문서)는 **CC BY-NC-SA 4.0** 을 따릅니다 — 출처 표기 · 비영리 ·
동일조건 변경허락. 자세한 내용은 [LICENSE](LICENSE) 참고.
(사용된 서드파티 데이터셋 · 모델 · 라이브러리는 각자의 라이선스를 따릅니다.
특히 이식된 모델 코드와 BFM 기저 · 배포 가중치는 원 저장소를 따릅니다.)

❌ **상업적 사용 금지** — 유료 서비스 · 제품 · API · 상업 모델 개발 등
(모델 가중치 · 추론 결과물 포함). 상업적 이용 문의는 메인테이너에게 연락해 주세요.

## 📮 문의

- [메일](mailto:peternara@naver.com) 또는
  [GitHub Issue](https://github.com/Pierrot-vision/Pierrot-FACE/issues) 를 통해 질문·문의
  부탁드립니다. 대답할 수 있는 내용이라면 성실히 답변드리겠습니다.
- 참고로, 이미 GitHub(README · 코드 · 문서)에 있는 내용을 다시 문의하시면 답을 드리지 못할
  수 있는 점 양해 부탁드립니다.
