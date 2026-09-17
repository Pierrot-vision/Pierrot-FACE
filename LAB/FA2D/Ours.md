# HRFFA+ — 2D 얼굴 정렬 구현 기록

[HRFFA](HRFFA.md) 의 설계를 기반으로, **Peppa_Pig_Face_Engine 이 사후 처리로 하던
일을 모델 안에서** 하게 만든 구현. Lab 규약(`train_fa2d.py`)으로 재구현했다.

공통 사항(랜드마크 규약·데이터셋·NME 정규화 기준)은 [FA2D.md](FA2D.md),
원본 알고리즘 조사는 [HRFFA.md](HRFFA.md) 를 본다.

> ⚠ **아직 본 학습을 돌리지 않았다.** 아래 §5 는 비어 있고, 나머지는 전부
> "코드가 의도대로 동작한다"의 검증이지 **"설계가 옳다"의 검증이 아니다.**
> 판정 기준은 §4.3 에 미리 적어 둔다.

---

## 1. 알고리즘

### 1.1 왜 HRFFA 기반인가

목표가 **극단 자세**이므로 선택지가 사실상 하나다.

| | HRFFA | Peppa_Pig | FAN · uniface · mobilenet-v2 |
|---|---|---|---|
| roll 커버 | **360° 전체** | ±30° 증강 | 없음 |
| pitch 커버 | **±85° 초과** | 정면 위주 | 정면 |
| 학습 코드 | 전체 공개 | 있음 | **없음** |
| 시간축(스무딩·추적) | **0 건** | LK + OneEuro | 없음 |
| 점별 가시성 | **있음** | 없음 | 없음 |

HRFFA 만 극단 자세를 커버하고, Peppa 만 "그다음"(영상에서 쓰기)을 다룬다.
**뼈대는 HRFFA, 실사용 마감은 Peppa** 가 이 구현의 구성이다.

### 1.2 모델 — 점 쿼리 디코더

```
이미지 → 백본 → 패치 (B,C,h,w) + CLS
                  ↓ input_proj + 2D sin-cos pos
                memory (B,hw,d)
                  ↑ cross-attention
  scheme 별 학습형 쿼리 (+ prev_proj(이전 dec_tokens))  → 디코더 ×4
                                                          ├→ coord_head  (B,N,2)
                                                          ├→ vis_head    (B,N,3)
                                                          ├→ pose_head   rot6d + roll
                                                          ├→ state_head  (B,4)
                                                          └→ dec_tokens → 다음 프레임
```

| 설계 | 이유 |
|---|---|
| scheme 별 학습형 쿼리 | 디코더·헤드 공유. 68/98/29 를 한 모델이 내고 확장이 쿼리 추가만으로 끝난다 |
| 좌표는 **sigmoid 없는 선형** | 크롭 밖 점(가시성 0)도 좌표를 가져야 하므로 [0,1] 로 누르면 안 된다 |
| 정규화 기준 = **크롭 변 길이** | 규약 무관하게 항상 정의된다 (head-NME) |
| 백본은 **2 계층** | 대형 ViT(교사)는 FAS 레지스트리 재사용 — HRFFA 의 DINOv3 는 런타임 git clone + 가중치 재배포 금지라 Lab 배포에 부적합. 경량(배포용)은 **torchvision CNN** — HRFFA 의 PP-HGNetV2-B0 가중치가 릴리스에 없어 이식해도 스크래치가 되기 때문 (§1.2.1) |

### 1.2.1 ★ 백본 — 왜 경량 CNN 을 따로 붙였나

**문제.** FAS 레지스트리는 범용 대형 ViT 뿐이라 최소가 `dinov2-s14`(백본 22M)다.
그 결과 우리 최소 모델이 26.7M / 13.2 GFLOPs 로, 참조 구현보다 **10 배 이상** 무거웠다.
디코더·헤드는 4.5M 뿐이므로 원인은 전적으로 백본이다.

**HRFFA 것을 그대로 못 쓰는 이유.** HRFFA 학생은 PP-HGNetV2-B0(CNN) 와 ViT-T/16 을
쓰는데, 그 사전학습 가중치 `PPHGNetV2_B0_stage1.pth` · `vitt_distill.pt` 가
**릴리스에 없다**(asset 50 개 확인). `hgnetv2.py`(227 줄, MIT)는 자족적이라 이식은
쉽지만, 가중치가 없으면 ImageNet 초기화를 못 얻어 이식의 이점이 사라진다.

**선택.** 같은 급의 torchvision CNN 을 쓴다 — 이미 설치돼 있고 ImageNet 가중치가 함께
온다. `stride 32` 단계는 버리고 **stride 16 에서 끊어** 격자 16×16 을 유지한다
(랜드마크는 국소 위치가 중요해 7×7 은 너무 성기다). 디코더가 요구하는
`(patch (B,C,h,w), cls (B,C))` 계약은 CLS 를 GAP 으로 만들어 맞춘다.

| 프리셋 | 백본 | 입력 | 전체 | 백본 | GFLOPs |
|---|---|---:|---:|---:|---:|
| `combined_nano` | mobilenetv3s (d=128·3층) | 256 | **1.09M** | 0.19M | **0.28** |
| `combined_mnv3s` | mobilenetv3s | 256 | 4.71M | 0.19M | 1.00 |
| `combined_effb0` | efficientnet_b0 | 256 | 5.40M | 0.85M | 1.55 |
| `combined_r18` | resnet18 | 256 | 7.41M | 2.78M | 4.60 |
| `combined_small` | dinov2-s14 | 224 | 26.75M | 22.06M | 13.20 |
| `combined` | dinov2-b14-reg | 224 | 91.47M | 86.58M | 48.08 |
| *참조* HRFFA hg0 | HGNetV2-B0 | 256 | 1.6M | — | 1.40 |
| *참조* HRFFA vitt | ViT-T/16 | 256 | 9.0M | — | 4.09 |
| *참조* Peppa Student | 자체 | 256 | 3.25M | — | 1.39 |
| *참조* Peppa Teacher | 자체 | 256 | 11.5M | — | 5.53 |

`combined_nano` 는 백본만 줄이면 디코더(4.5M)가 병목이 되므로 **디코더도 함께** 줄였다
(d_model 256→128 · 4층→3층 · FFN 1024→512). HRFFA hg0(1.6M/1.40G)보다 작다.

> ⚠ **위 표의 NME 는 비교 금지.** HRFFA README §1 이 명시한다 —
> *"all splits of the real-image datasets — including WFLW test, 300W and COFW test —
> are used for training. The 'official' numbers ... must not be compared with published
> benchmark results."* 즉 HRFFA 의 3.36 / 5.32 는 **학습 분포 적합도**이지 일반화 성능이
> 아니다(WFLW test 가 학습셋 안에 있다). 우리 숫자는 WFLW test 를 뺀 **홀드아웃**이다.
> 크기·FLOPs 만 비교 가능하다.

**★ 백본 초기화가 지배적이다 — HRFFA 자신의 표가 증거.** 누출된 평가(=쉬운 조건)에서조차
`vitt-256`(3.36) 과 `hg0-256`(5.32) 이 **58% 벌어진다.** 둘의 차이는 초기화다:
  - `vitt` ← `vitt_distill.pt` — DINOv3 를 **증류해 만든 자체 ViT-T/16** (history/026)
  - `hg0`  ← PP-HGNetV2-B0 — DEIMv2 배포판의 **ImageNet 분류 사전학습**
따라서 ImageNet CNN 초기화로 시작하는 우리 경량 아암이 DINOv2 아암보다 나쁜 것은
버그가 아니라 **예상된 패턴**이다. 경량 모델의 정답 경로는 ImageNet CNN 이 아니라
**우리 교사로부터의 증류**다 (`train_fa2d_distillation.py`).

⚠ 경량 아암은 `backbone_lr_mult 1.0` 이다. ×0.1 은 **대형 사전학습 ViT 를 초기 노이즈에서
보호**하려는 값인데, 용량이 작은 CNN 에 그대로 쓰면 백본이 태스크에 적응하지 못한다.

### 1.3 손실

| 항 | 정의 | 기본 가중 |
|---|---|--:|
| `coord` | smooth-L1(β=0.01) **또는 Wing** | 10.0 |
| `vis` | 3-class CE (화면밖/가림/보임) | 1.0 |
| `roll` | Roll biternion cosine | 0.0 |
| `state` | 4상태 BCE (판정 불가 제외) | 0.0 |
| `temporal` | 시간 일관성 | 0.0 |

점별 가중: 화면 밖(vis==0)은 0.5배.
⚠ 가중은 `(l·w).sum()/w.sum()` 이라 **상대적**이다 — 전 점이 화면 밖이면 정규화로
상쇄된다(보이는 점과 섞였을 때 효과가 난다).

### 1.4 두 알고리즘과의 전면 대조 (09-16 최신)

**우리에게 없는 것도 그대로 적는다.** ✅/❌ 는 구현 여부이지 우열이 아니다.
기준 구성은 현재 교사 최고 아암 **`full_v12`**(얼굴 크롭 448 · refine · 표본 가중)이다.

| 축 | **HRFFA** | **Peppa_Pig** | **우리 (full_v12)** |
|---|---|---|---|
| **입력·전처리** | | | |
| crop 규약 | head + pad **0.05** | **face** bbox × 1.2~1.4 | **face** bbox + ext 0.05 (09-15 머리→얼굴) |
| head bbox 출처 | **DEIMv2 의사라벨** | yolov5-face 검출기 | ⚠ **랜드마크 외삽**(0.30/0.90/0.15) — HRFFA 의 D1 플레이스홀더 |
| 입력 해상도 | 320(교사) / 256 / 96 | 256 / 128 | **448** (교사·학생 모두) |
| 정규화 | imagenet(교사) / center05(학생) | **/255 만** | imagenet |
| **모델** | | | |
| 백본 | **DINOv3** ViT-L/16 · ViT-T/16 · HGNetV2-B0 | MobileNetV3(학생) · **HRNet-W18**(교사) | **DINOv3** ViT-L/16(교사) · **ViT-T/16**(학생) · 경량 CNN 5종 |
| 디코더 | 점 쿼리, **업샘플 없음** | **ASPP + skip U-Net → stride 4** | 점 쿼리 + **국소 보정**(stride 4 CNN, Phase 10 C) |
| 출력 표현 | 좌표 회귀 | **히트맵 + x/y 오프셋** | 좌표 회귀(교사) · **hm4 = stride 4 히트맵 + x/y 오프셋**(학생, 09-14) |
| 손실 | 좌표 + 가시성 | **AWing(히트맵) + Wing(오프셋)** | smooth_l1 + 가시성 + coarse · 학생은 **AWing + 목표가중 L1** 추가 |
| 규약 | 68·98·29 한 모델 | 98 고정 | **68·98·29·106 한 모델** |
| 가시성 | ✅ 3-class (**실제 라벨**) | ❌ | ✅ 3-class (**라벨 없어 미학습**) |
| **증강** | | | |
| 회전 | **360°** | ±30° | **360°** |
| 카메라 pitch/yaw | ±25° / ±15° | 없음 | ±25° / ±15° |
| scale · translate | 0.9~1.02 · 0.03 | 0.7~1.35 · — | 0.9~1.02 · 0.03 |
| 표정 | — | **표본 복제** 눈감음 ×15 · 입벌림 ×2 | 손실 기반 과표집(`hard_mining`) — ⚠ **`full_v12` 부터 실제 동작**(그 전엔 설정만 있고 갱신 호출이 없었다) |
| 극단 pitch | **생성 데이터 7,710장(22%)** | 없음 | ❌ 없음 (깊이 워핑은 ±40° 한계) |
| **학습** | | | |
| 학습셋 | 34,986 (**WFLW test 포함**) | **WFLW train 7,500 만** | WFLW train 7,500 · LaPa train+val 20,168 · COFW **전 분할** · 300W **전 분할** (셋 다 우리 평가셋 아님) |
| 일정 | WSD 200~250ep | 100ep | cosine 100ep(교사) · 250ep(학생, best=145ep) · WSD 시도 → 효과 없음 |
| 파라미터 | 308M / 9.0M / 1.6M | 11.5M / 3.25M | 311M(교사) / **10.54M**(학생) |

### 1.4.0 성능 — **어느 자로 재느냐가 전부다**

각 모델을 **자기 평가 규약**으로 잰 값이다(Phase 14 에서 규약을 맞춰 공개값을 재현했다).
세 벤치마크를 한 표에 둔다 — ① WFLW test 2,500 · ② LaPa test 2,000(누출 없는 축) ·
③ HRFFA 난이도 프로토콜.

| 모델 | 백본 | 파라미터 | 입력 · 크롭 | 얼굴 해상도 | 출력 | 증류 | 학습셋 | WFLW test 학습 | LaPa 학습 | **① WFLW Full** | FR10 | **② LaPa 공통38** | FR10 | **③ base** | roll 최악 | 자세 교란 평균 | 스타일 평균 | 스타일 최악 |
|---|---|--:|---|--:|---|---|---|:--:|:--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| HRFFA 교사 `clean_v3` | DINOv3 ViT-L/16 | 308.2M | 320 · 머리 | 89px | 좌표 | — | 34,986 | ✅ | ❌ | *1.698* | — | **2.547** | 0.70% | *1.861* | *1.901* | *1.855* | *1.830* | *2.404* |
| HRFFA 학생 vitt-256 | ViT-T/16 | 9.0M | 256 · 머리 | — | 좌표 | A | 〃 | ✅ | ❌ | *3.36* | — | 가중치 없음 | — | — | — | — | — | — |
| HRFFA 학생 hg0-256 | HGNetV2-B0 | 1.6M | 256 · 머리 | — | 좌표 | A | 〃 | ✅ | ❌ | *5.32* | — | 가중치 없음 | — | — | — | — | — | — |
| D-ViT (논문) | ViT | 96.4M | 256 · 얼굴 | — | 좌표 | — | WFLW train | ❌ | ❌ | **3.75** | — | 모델 미공개 | — | — | — | — | — | — |
| Peppa Teacher | HRNet-W18 | 11.53M | 256 · 얼굴 | 131px | 히트맵+오프셋 | — | WFLW train 7,500 | ❌ | ❌ | **3.959** | — | 2.164 | 0.65% | 4.440 | **124.851** | **39.097** | 4.941 | 7.271 |
| Peppa Student | MobileNetV3 | 3.25M | 256 · 얼굴 | 131px | 히트맵+오프셋 | B | WFLW train 7,500 | ❌ | ❌ | 4.353 | — | 2.214 | 0.70% | 4.777 | **122.079** | **38.872** | 5.308 | 7.112 |
| **우리 교사 `full_v12`** | DINOv3 ViT-L/16 | 311M | 448 · 얼굴 | 245px | 좌표+coarse | — | 33,357 | ❌ | train+val | **3.916** | **1.48%** | **1.693** | **0.20%** | **3.799** | **3.997** | 4.070 | **3.948** | **4.032** |
| 우리 교사 `full_v11` | DINOv3 ViT-L/16 | 310M | 448 · 머리 | 125px | 좌표+coarse | — | 32,161 | ❌ | train+val | 3.942 | 1.64% | 1.808 | 0.20% | 3.959 | 4.119 | **4.022** | 4.003 | 4.277 |
| **우리 학생 `stu_final_v2`** | ViT-T/16 | 10.54M | 448 · 얼굴 | 245px | hm4+좌표 | **A**(`full_v12` 고정) | 교사와 동일 | ❌ | train+val | **4.227** | 3.00% | **1.766** | **0.15%** | 4.033 | 4.119 | 4.358 | 4.316 | 4.470 |
| 우리 학생 `stu_vitt_face448_hm4` | ViT-T/16 | 10.54M | 448 · 얼굴 | 245px | hm4+좌표 | A(`full_v11` 고정) | WFLW+LaPa | ❌ | train+val | 4.458 | 3.08% | 1.728 | 0.20% | 4.205 | 4.596 | 4.462 | 4.644 | 5.534 |
| 우리 학생 A (초기) | MobileNetV3-L | 8M | 256 · 머리 | 71px | 좌표 | A | WFLW+LaPa | ❌ | train+val | 5.399 | 7.04% | 미측정 | — | — | — | — | — | — |

*기울임* = 그 모델이 학습한 이미지에서 잰 값 — 비교 불가.
증류 **A** = 교사 고정(`distill_mode: hrffa`) · **B** = 교사를 GT 로 함께 미세조정(`distill_mode: peppa`).

**① WFLW test 2,500** — 각 모델 자기 평가 규약(Phase 14 에서 공개값 재현).
**② LaPa test 2,000, 공통 38 점**(눈 16 + 동공 2 + 입 20) — HRFFA 도 Peppa 도 LaPa 를
학습하지 않고 우리는 test 를 뺐다. 규약 다리는 Phase 16 에서 실측해 세웠다(편향 +0.387,
98 점 모델 전부에 동일). ⚠ **우리만 LaPa train+val 을 학습했으므로 도메인 이점이 있다.**
**③ HRFFA 난이도 프로토콜** — pose-stress(roll 360°·카메라 pitch ±25°/yaw ±15°) ·
style-shift(모션블러·색온도·감마·회색조·JPEG), 각 모델 자기 크롭 규약.

| 축 | 공정성 | 결과 |
|---|---|---|
| ① WFLW | HRFFA 는 **비교 불가**(test 학습). Peppa·D-ViT 와만 성립 | 교사 3.916 < Peppa Teacher 3.959 · 학생 4.227 < Peppa Student 4.353 · **D-ViT 3.75 에는 뒤짐** |
| ② LaPa | 셋 다 평가 이미지 미학습 (⚠ 우리만 같은 셋의 train+val 학습) | 우리 1.693~1.808 < Peppa 2.164 / 2.214 < HRFFA 2.547 |
| ③ 난이도 | HRFFA 는 비교 불가. Peppa 와만 성립 | roll 4.119 vs **122.079** — Peppa 는 ±30° 회전 증강뿐이라 무너진다 |

누출 없는 다른 벤치마크(우리 `full_v11`, 두 셋 모두 train 분할만 학습):

| | D-ViT (논문) | HRFFA (⚠ test 학습) | `full_v11` |
|---|--:|--:|--:|
| COFW test (동공 기준) | **4.13** | — | 4.382 |
| 300W Common / Challenging / Full | **2.43 / 4.56 / 2.85** | 0.96 / 1.34 / 1.03 | 2.992 / 5.519 / 3.487 |

> ⚠ **비교 가능성 경고 — 이걸 안 지키면 순위가 통째로 뒤집힌다.**
> - *기울임* 값은 **그 모델이 학습한 이미지**에서 잰 값이다(HRFFA 는 WFLW test · 300W valid ·
>   COFW test 를 전부 학습한다 — `train_all_splits: true`).
> - **상대 모델은 헤드·크롭·색 순서·NME 좌표계를 맞춰야 잰다.** 이걸 틀려 폐기한 값:
>   HRFFA ~~4.840~~(68 점 헤드) · ~~12.489~~(검출 박스) · Peppa ~~4.640~~ · ~~5.004~~(BGR·크롭·정규화).
> - 학습한 이미지에서의 적합도(참고): 우리 `full_v9` 2.849 · Peppa Teacher 1.836 ·
>   Peppa Student 1.754 · 우리 학생 A 3.969.
> - **HRFFA 와는 WFLW 로 순위를 매길 수 없다.** 그들은 평가셋을 학습했고 학습셋도 더
>   많다(300W-LP 0.33 · 의사라벨 자가학습 0.22 를 포함해 34,986). 같은 자로 잰 참조는
>   Peppa(3.959 / 4.353)와 D-ViT(3.75) 뿐이다. HRFFA 와 겨루려면 **양쪽 다 학습하지 않은
>   셋**이 필요하다 — 그들의 소스 목록(300wlp · wflw · selftrain_v2 · selftrain_lookup ·
>   300w · cofw)에 **LaPa 가 없고**, 우리는 LaPa test 를 뺐다(`lapa_splits: (train, val)`).
>   다만 LaPa GT 는 106 점 전용이고 HRFFA 헤드는 68/98/29 뿐이라 공용 규약이 없다.

### 1.4.0.1 능력이 갈리는 지점

| 축 | 우세 | 근거 |
|---|---|---|
| 극단 자세(Pose) | **우리** (6.32 vs 7.00) | `full_v12` · 오염 없음 |
| 가림(Occlusion) | **우리** (4.49 vs 4.85) | 〃 |
| 흐림 | **우리** (4.42 vs 4.54) | 〃 |
| **표정** | **Peppa** (4.00 vs 4.16) | 〃 — 남은 열세 축 |
| **턱선 형상** | **HRFFA** (2.028 vs 2.574) | 자기 크롭끼리 · 21% 우세 |
| 턱선 점 배치 | **우리** (5.450 vs 5.862) | 〃 |
| **누출 없는 축(LaPa)** | **우리** (1.693 vs HRFFA 2.547 · Peppa 2.164) | ⚠ 우리만 LaPa train+val 학습 |
| **회전 강건성** | **우리** (roll 최악 3.997 vs Peppa 124.851) | Peppa 는 ±30° 증강뿐 |

**HRFFA 는 턱선이 어디인지 더 잘 찾고, 우리는 그 선 위 점 배치를 더 잘 맞춘다.**
NME 는 점 배치만 재므로 우리에게 유리한 축만 반영된다. 다만 "전체는 뒤진다"는 옛 서술은
WFLW 누출 수치를 기준으로 한 것이었다 — 09-16 에 **양쪽 다 학습하지 않은 LaPa test** 로
재니 1.693 vs 2.547 로 우리가 앞섰다(⚠ 우리만 LaPa train+val 학습).

### 1.4.1 Peppa 대비 격차 — 관측과 미검증 원인을 구분한다

우리 `teacher_v3`의 WFLW test 2,500장 io-NME는 **4.120**, Peppa
Teacher@256은 **3.95 / 11.53M**이다. Peppa Student@256은 **4.35 / 3.25M**이다.
우리 교사는 약 309M이며 448 입력 FLOPs는 재측정 전까지 미확인이다.
비교표가 336 입력의 386.45를 v3에 붙인 오류를 수정했다. 과거 687도 실측 근거 미확인이다.

**정정:** 기존 “MobileNetV3 + Linear이므로 구조 가설 기각”은 다른 파일을 읽은 오류다.
실제 Peppa는 HRNet-W18 교사/MobileNetV3 학생에 ASPP와 stride 4·8 skip 특징을
결합하고, 히트맵·오프셋을 감독한다. 우리 좌표는 연속값이므로 stride 14가 정밀도 하한은 아니다.
두 해상도의 적합 절편 3.853을 실제 바닥으로 해석하거나, 다른 아암의 회전 개선율을
빼서 Peppa보다 낫다고 주장한 결론은 철회한다.

| 요인 | 관측 | 판단의 한계 |
|---|---|---|
| 회전 범위 | ±30° 아암에서 5.471→4.869, 5.125→4.664 | v3 격차의 기여율로 바로 전이할 수 없음 |
| 얼굴 크기 | v3 눈사이 101.7px, Peppa@256 113.6px | 크롭·증강을 통제한 구조 비교가 아님 |
| 고해상도 국소화 | Peppa는 stride 4 skip 특징과 공간 감독 사용 | 우리 모델의 병목인지는 다음 아암으로 검증 |
| WSD·300W-LP | v6 4.123, v5 4.175, v5y 4.188 | 일정 전체 소진·합성 도메인 부적합을 증명하지 않음 |

300W-LP 아암은 기존 소스의 노출량도 줄였으며, WSD는 FR10이 2.16%→1.88%다.
HRFFA는 test 학습 포함을 명시하므로 공개 WFLW 수치와 일반화 순위를 비교하지 않는다.
쿼리 유지가 회전 강건성을 보장하지 않으며, 히트맵도 회전·다중 규약을 지원할 수 있다.

**다음 검증:** 기존 회귀를 유지하고 국소 보정 경로를 추가하되, 추가 학습 대조군,
DINO 특징만 쓰는 보정, 입력의 stride 4 CNN 특징을 결합한 보정을 비교한다.
NME·FR10과 회전·극단 자세·화면 밖 좌표·추론 비용을 함께 평가한다.
A/B/C 코드 구현과 동작 검증은 완료했다. 실제 NME 효과는 아직 미측정이다.
상세 근거·실행 방법·검증 결과는 [Phase 10](Exp/Phase_10_레버_소진과_예측격자.md)을 따른다.

## 2. 데이터셋

| | 확보 | 경로 |
|---|---|---|
| **WFLW** (98점) | ✅ train 7,500 / test 2,500 | `2DDFA/raw/WFLW` |
| 300W (68점) | 원본 zip 보유, 리더 미구현 | `2DDFA/raw/300W` |
| COFW (29점) | ❌ | — |

**WFLW 이미지는 재구성한 것이다.** 공식 배포처가 Google Drive 단일 링크뿐이라
할당량으로 실패했고, WFLW 가 **WIDER FACE 부분집합**임을 이용해 복원했다.

```
WFLW 주석의 이미지 경로 = WIDER FACE 명명 규약 (`51--Dresses/51_...jpg`)
  → CUHK-CSE/wider_face 에서 WIDER_train(1.4GB) + WIDER_val(363MB)
  → 주석이 참조하는 6,551 장 선별 복사
  → 6,551 / 6,551 전부 확보 ✅ (누락 0)
```

중간 통합 jsonl 을 두지 않고 **원본 배포 형식을 직접 읽는다**(`data.SOURCES`).
변환 단계가 사라지고 원본과 어긋날 여지가 없다.

### 2.1 ⚠ WFLW 에는 가림 라벨이 없다

전 점이 `-1`(불명)이고, 기하 증강이 **크롭 밖 점만** 0 으로 바꾼다. 이 때문에
스모크에서 결함 둘이 드러났다(§3.3). **가시성 헤드를 제대로 감독하려면 COFW 처럼
가림 라벨을 가진 소스가 섞여야 한다.**

---

## 3. 학습 방법

### 3.1 코드 구조

```
train_fa2d.py                   ① GT 지도학습 → 교사
train_fa2d_distillation.py      ② 교사 동결 + 온라인 KD → 학생
configs/args_fa2d.py            PARAMS + PRESETS + FA2D_PRESET
pierrotfr/FA2D/
  geometric.py    177  호모그래피 증강
  trajectory.py    93  궤적 샘플러
  data.py         198  WFLW 리더 · 클립 · SchemeBatchSampler
  models/         191  백본 어댑터 + 점 쿼리 디코더
  losses.py        99  coord(smooth_l1|wing) · vis · roll · temporal
  facestate.py     96  상태 라벨 유도 + 가시성 게이팅
  metrics.py       75  NME/FR/AUC + 지터·지연
  smoother.py      65  OneEuro + 가시성 가중
  benchmark.py     88  지터 평가
  engine.py       192  train_one_epoch / evaluate / run_clip / **distill_step**
  config.py        62  검증
eval/FA2D/evaluate.py           독립 평가 CLI
tests/FA2D/       222  31 케이스
```

### 3.2 프리셋

| 프리셋 | 바꾸는 것 | 묻는 것 |
|---|---|---|
| `base` | — | 정지영상 정렬 기준선 |
| `temporal` | `temporal` · `w_temporal 1.0` · batch 12×4 | **시간 안정성 (§4.3 판정)** |
| `state` | `use_state` · `w_state 0.2` | 상태 헤드가 좌표를 깎지 않는가 |
| `wing` | `coord_loss: wing` | Wing 이 미세 정밀도를 올리는가 |
| `small` | `dinov2-s14` · batch 48×1 | 파라미터 1/4 로 따라오는가 |
| `teacher` | dinov2-l14-reg @336 | 증류의 교사 |
| `student` | dinov2-s14 @224 ← 교사 336 | **경량 배포 후보** |
| `student_temporal` | 학생 + 시간축 | 최종 배포 후보 |
| `smoke` · `smoke_teacher` · `smoke_distill` | 2 epoch × 8 샘플 | 끝까지 도는가 |

### 3.3 실행

```bash
# 단일 모델 (교사만 필요하거나 그것으로 충분할 때)
FA2D_PRESET=base python train_fa2d.py 2>&1 | tee train.log

# 배포 경로 — 2 단계
FA2D_PRESET=teacher python train_fa2d.py                  # ① 교사
FA2D_PRESET=student python train_fa2d_distillation.py     # ② 학생 증류

accelerate launch --num_processes 4 train_fa2d.py         # 4 GPU
python eval/FA2D/evaluate.py --ckpt runs/fa2d/<실험ID>/best.pth
```

산출물은 FA3D/FAS 와 같다 — `best.pth` · `latest.pth` · `history.json` ·
`results.json` · `train.log`.

---

### 3.4 증류 구조

```
crop @teacher_size(336) ─┬→ 교사 (동결 · eval · no_grad)   → t_out
                         └→ resize @224 → 학생             → s_out
                                                              ├ GT 손실 (coord · vis · …)
                                                              └ KD 손실 (coord · vis · tok)
```

| 항목 | 값 | 비고 |
|---|---|---|
| `w_kd_coord` | 0.5 | 교사 예측 좌표로의 smooth-L1 |
| `w_kd_vis` | 0.5 | 가시성 logits KL (T=2, ×T²) — **WFLW 에 없는 라벨을 채우는 채널** |
| `w_kd_tok` | 0.2 | `dec_tokens` 특징. `d_model` 이 다르면 Linear 어댑터 |

**왜 온라인인가** — 증강이 매 스텝 달라지므로 교사 출력을 미리 뽑아 캐시할 수 없다.
그래서 매 스텝 교사를 함께 forward 한다(그만큼 느리다).

**왜 GT 를 함께 쓰나** — 학생은 GT 와 교사 출력을 **둘 다** 본다. 교사가 틀린 곳은
GT 가 잡아 주고, GT 에 없는 정보(가시성 분포·중간 표현)는 교사가 채운다.

**해상도 분리** — 학습 로더가 `teacher_size` 로 렌더링하고 학생 입력은 스텝 안에서
`F.interpolate(antialias=True)` 로 줄인다. 정규화 좌표는 해상도 비독립이라 GT·KD
목표를 그대로 공유한다. val 은 배포 대상이 학생이므로 **학생 해상도로 직접 렌더링**한다.

학생 백본은 `backbone_lr_mult 0.5`(교사는 0.1) — 교사의 사전학습은 보호 대상이지만
학생 백본은 **이 태스크에 맞추는 것이 목적**이다.

---

## 4. 평가 방법

### 4.1 정지영상

| 지표 | 정의 |
|---|---|
| `head_nme` | 크롭 변 길이 기준 (%) — 규약 무관, 항상 정의된다 |
| `io_nme` | inter-ocular 기준 (%) — ibug68/wflw98 만 |
| `fr10` | NME > 10% 비율 (%) |
| `auc10` | NME 누적분포 0~10% 면적 (%) |

⚠ 정규화 기준이 다른 NME 를 같은 표에 놓으면 안 된다. 인용 시 기준과
"학습에 무엇이 들어갔는지"를 함께 적는다.

### 4.2 지터 — 정지영상 벤치마크로는 못 재는 축

합성 클립은 **참 움직임을 알기에** 노이즈와 지연을 분리할 수 있다.

| 지표 | 의미 |
|---|---|
| `jitter_px` | 평균 ‖Δpred − Δgt‖ |
| `lag_ratio` | ⟨Δpred, Δgt⟩ / ‖Δgt‖² — **1.0 완벽 · <1 지연 · >1 과잉** |

동작 확인:

| 상황 | jitter_px | lag_ratio |
|---|--:|--:|
| 완벽 추종 | 0.00 | 1.00 |
| 전혀 안 움직임 | 1.41 | **0.00** |
| 노이즈만 | 1.78 | **0.96** |

### 4.3 ★ 판정 규칙 — 미리 정해 둔다

| 결과 | 결론 |
|---|---|
| `lag_ratio` 유지 + `jitter_px` 하락 + `head_nme` 유지 | **채택** — 사후 필터가 못 하는 것을 했다 |
| `jitter_px` 하락하지만 `lag_ratio` 도 하락 | 기각 — OneEuro 와 다를 바 없다 |
| `head_nme` 악화 | 기각 — 기본 정확도를 깎았다 |

기준선은 `eval/FA2D/evaluate.py` 가 한 번에 낸다 — `raw` / `oneeuro` / `vis_oneeuro`.

---

## 5. 결과

**미실행.** 스모크(2 epoch × 8 샘플)만 통과했다. 아래는 파이프라인이 도는 것을
보인 것이지 성능이 아니다.

```
모델 pointquery · 백본 dinov2-b14-reg | 전체 91M · 학습가능 91M
param groups — g0: 85M lr 2.0e-05 | g1: 1M lr 2.0e-05 | g2: 5M lr 2.0e-04 | g3: 21K lr 2.0e-04
still            | head_nme 22.272 | io_nme 101.299 | fr10 100.000 | auc10 0.000 | n=8
jitter/raw       | jitter_px 6.001 | lag_ratio 0.022
jitter/oneeuro   | jitter_px 4.464 | lag_ratio 0.027
jitter/vis_oneeuro | jitter_px 4.418 | lag_ratio 0.028
```

백본/헤드 LR 10배 분리가 4그룹으로 갈린 것, 지터 3-way 비교가 나오는 것을 확인했다.

### 5.1 측정한 것 — OneEuro 의 한계

합성 신호(노이즈 σ=0.8px, 60 프레임)에서 실측. **참값 대비 RMS(px)**.

| 움직임 | 필터 없음 | OneEuro | 판정 |
|---|--:|--:|---|
| 정지 | 0.798 | **0.309** | 크게 개선 |
| 0.1 px/frame | 0.767 | **0.434** | 개선 |
| 0.5 px/frame | 0.827 | **0.927** | **악화** |

빠른 움직임에서 지연이 노이즈 이득을 삼킨다. 참 움직임을 모르므로 **원리적으로
피할 수 없다.** 이것이 "사후 필터 말고 모델 안으로"의 근거다.

### 5.2 스모크가 잡아낸 결함 둘

**① 가시성 손실 붕괴** — WFLW 는 전 점이 `-1` 이고 증강이 크롭 밖만 0 으로 바꾼다.
마스크에 남는 게 전부 클래스 0 뿐이라 CE 가 **"항상 화면밖이라고 답해라"** 를
학습시킨다. 헤드가 통째로 망가진다.
→ 클래스가 한 종류만 남으면 건너뛰도록 가드.

**② 상태 게이팅 의미론** — "보임(2)만 허용"으로 뒀더니 WFLW 에서 상태 손실이
통째로 0 이 됐다. `-1` 은 "가려졌다"가 아니라 **"모른다"** 다.
→ **가려짐(1)·화면밖(0) 으로 확인된 경우만 제외**. 비용: 불명 중 일부가 실제로
가려져 있을 수 있다.

```
수정 후:  coord 8.7216 · vis 0.0 · state 0.1351 · temporal 0.0347
```
(`vis 0.0` 은 WFLW 에 가림 라벨이 없어서다 — §2.1)

### 5.3 테스트

| | 케이스 |
|---|--:|
| `tests/FA2D/` | **31** |
| Lab 전체 | 141 (FAS/FA3D 110 무영향) |

---

## 6. 한계

1. **본 학습 미실행.** 설계 주장이 검증되지 않았다.
2. **가시성 감독이 사실상 없다.** COFW 확보 전까지 `vis` 손실은 0 이다.
3. **단일 규약.** WFLW(98) 뿐. 다중 규약 단일 모델이라는 설계 이점을 아직 못 쓴다.
4. **합성 움직임 ≠ 실제 움직임.** 진짜 지터·모션블러 상관은 재현되지 않는다.
5. **head bbox 가 랜드마크 외삽**이다. 정식으로는 head 검출기 pseudo-label 이 맞다.
6. **추적 미구현.** 이전 프레임 랜드마크로 head bbox 를 안정화하는 설계는 정했으나
   코드가 없다. ⚠ tight **face** crop 으로 좁히면 안 된다 — 극단 뷰에서 얼굴이
   거의 안 보여 crop 이 틀어진다.

---

## 7. 후속 작업

| 순위 | 할 일 |
|---|---|
| 1 | `base` 본 학습 → 기준선 확보 |
| 2 | `temporal` 학습 → §4.3 판정 |
| 3 | COFW 확보 → 가시성 감독 활성화 |
| 4 | 300W 리더 추가 → 다중 규약 검증 |
| 5 | 추적 기반 head bbox 안정화 |
| 6 | ONNX export (`prev_tokens` 입출력 포함) |
| 7 | WSD 스케줄 — resume 으로 학습 길이를 조절할 수 있게 |

---

## 부록 — 상류 포크 프로토타입

Lab 구현에 앞서 **HRFFA 저장소(`hrffa-plus` 브랜치)에서 같은 확장을 먼저 검증**했다.
그들의 검증된 학습 루프 위에서 설계가 성립하는지 확인하는 것이 목적이었다.

`2d_aligments/High-Angle_Robust_Fast_FaceAlignment` · 커밋 5 · 테스트 61

| 커밋 | 내용 |
|---|---|
| `32ba381` | roll 손실을 rot 과 독립으로 분리 |
| `3273d1e` | 궤적 샘플러 |
| `b2964ca` | 시간축 디코더 + 시간 일관성 손실 |
| `03d594f` | 지터 지표 + OneEuro 기준선 |
| `d487260` | 상태 헤드 + ONNX 계약 확장 |

거기서만 확인된 것 둘 — Lab 구현에도 반영돼 있다.

- **`prev_proj` 는 zero-init 이고 bias 가 없어야 한다.** ONNX 는 선택적 입력이
  어려워 첫 프레임에 0 을 넣게 되는데, bias 가 있으면 `prev_proj(0) = bias ≠ 0`
  이라 학습 때의 `prev_tokens=None` 과 달라진다. bias 를 빼면 **정확히 등가**가
  되어 첫 프레임·추적 유실이 그래프 안에서 해결된다.
- ONNX 계약은 켜졌을 때만 확장된다(`+prev_tokens` / `+dec_tokens` / `+state_logits`).
  onnxruntime 으로 실제 export 해 PyTorch 와 대조했다(3 조합, 오차 < 2e-4).

> 실행 명령은 그쪽 규약(`uv run python -m hrffa.train.train_teacher`)이라 **본
> 구현과 다르다.** 돌릴 것은 `train_fa2d.py` 다.
