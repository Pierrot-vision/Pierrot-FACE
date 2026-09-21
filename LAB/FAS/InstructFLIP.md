# InstructFLIP — 통합 비전-언어 FAS 모델

CelebA-Spoof 를 언어 감독으로 다시 쓰는 VLM 계열 FAS 알고리즘.
**추론 시 LLM 을 타지 않아** 176M 규모로 떨어진다.

> 공통 사항(태스크 정의·지표·데이터셋·코드 구조·설정·실행)은 **[FAS.md](FAS.md)** 를 본다.
> 이 문서는 **알고리즘**만 다룬다. 경량 대조군은 [MiniFASNet.md](MiniFASNet.md).

- 원저장소: <https://github.com/kunkunlin1221/InstructFLIP> · 로컬 사본 `<작업 루트>/InstructFLIP`
- 구현: [`pierrotfr/FAS/models/instructflip.py`](../../pierrotfr/FAS/models/instructflip.py)

---

> **결과 (2026-09-18, CelebA-Spoof 공식 test)** — 공식 ACER **4.93** (AENet 논문 1.63 미달) ·
> val 기준 운영점 ACER **1.52**. 놓친 위조의 75% 가 PC screen. 표와 레시피는
> [FAS.md](FAS.md#현재-최고-레시피) (자동 생성), 분석은 [Phase 1](Exp/Phase_1_첫_학습과_PC_screen.md).
> ⚠ 원논문(외부 7개 데이터셋 평균 HTER 12.68)과는 **평가 데이터가 달라 비교할 수 없다** — 외부 벤치마크는 기관 승인 대기.

## 1. 알고리즘

원논문: **InstructFLIP: Exploring Unified Vision-Language Model for Face Anti-spoofing**
(ACM MM 2025, [arXiv:2507.12060](https://arxiv.org/abs/2507.12060))

### 1.1 핵심 아이디어

지시문을 **직교 두 축**으로 분해해 instruction tuning 한다.

| 축 | 질문 | 담당 |
|---|---|---|
| **content** | "어떤 위조 유형인가?" (11지선다) | spoof 의미 자체 |
| **style** | "조명/환경/카메라 품질은?" | spoof 와 **무관한** nuisance |

그리고 **style 은 학습 신호로만 쓰고 판정 경로에서는 차단**한다.
style 분기가 조명·환경·촬영기기를 따로 책임지므로, content 표현이 그런 nuisance 를 덜 껴안는다.

두 번째 축은 **훈련 효율**이다. 기존 DG 는 leave-one-out 으로 도메인 조합마다 재학습해야 한다
(D1,D2,D3 → D_T). InstructFLIP 은 CelebA-Spoof 를 meta-domain 으로 **한 번만** 학습하고
모든 타깃에 그대로 평가한다.

### 1.2 데이터 흐름

```
image [B,3,224,224]
  │
  ├─ 백본 (CLIP ViT-B/16) ──┬─► content 토큰 [B,197,768]   (cls + 14×14 패치)
  │                         └─► layer_feats 12개
  │                              레이어별 mean/std → style 통계 [B,24,768]
  │
  ├─ content connector(content 토큰, "어떤 위조 유형?") ─► content_q [B,32,768]
  ├─ style   connector(style 통계,  "조명 조건은?")     ─► style_q   [B,32,768]
  │
  ├─【학습만】 llm_proj → soft prompt → frozen FLAN-T5 → 정답 생성 손실
  │
  ├─ fusion(content 토큰, cat[content_q, style_q]) ──► [B,197,768]
  │
  └─ split ─┬─ [:,0]  → classifier    → real/fake   ★판정
            └─ [:,1:] → cue generator → 14×14 맵     (해석용)
```

> ⚠ **style 특징(통계)은 융합에 넣지 않는다.** 도메인 종속 표현이라 일반화 목표와 충돌한다.
> 융합에 들어가는 건 style **쿼리**뿐이며, 그마저 학습 중 nuisance 를 흡수한 표현이다.

### 1.3 LLM 의 역할 — 동결된 채점기

**LLM 은 학습 중에도 파라미터가 갱신되지 않는다** (`requires_grad = 0`, 248M 전체).

```
Q-Former 쿼리 [B,32,768]
   → llm_proj (768 → T5 d_model)         ← 학습됨
   → [soft prompt ; 지시문 토큰] → T5 인코더
   → T5 디코더가 정답 생성:  "(2) Photo"
   → cross-entropy = content_lm_loss
```

핵심은 **"쿼리 32개만 보고 정답을 말할 수 있느냐"** 를 T5 가 채점한다는 점이다. 쿼리에 위조
유형 정보가 없으면 T5 가 못 맞히고 손실이 커진다. 그 손실이 **동결된 T5 를 관통해** 역전파되어
Q-Former 와 백본을 가르친다.

T5 의 언어 지식이 채점 기준이 된다 — `"(2) Photo"` 와 `"(3) Poster"` 가 의미적으로 가깝다는
구조가 손실 지형에 반영된다. 단순 11-way 분류 헤드에는 그 관계가 없다.

**측정한 그래디언트 흐름** (`content_lm_loss` 만 backward):

| 모듈 | grad_norm | grad 받은 텐서 |
|---|---|---|
| backbone | 403.92 | 197개 |
| content connector | 263.61 | 47개 |
| llm_proj | 106.56 | 2개 |
| style connector | 0 | 0개 |
| fusion / classifier | 0 | 0개 |

### 1.4 LLM-free 추론

**추론 경로에서 LLM 을 아예 타지 않는다.** 학습이 끝나면 잘라낸다.

| 모듈 | 파라미터 | 추론 |
|---|---|---|
| 백본 CLIP ViT-B/16 | 86M | **O** |
| content connector (qformer) | 40M | **O** |
| style connector (qformer) | 40M | **O** |
| fusion | 9M | **O** |
| classifier | 2K | **O** |
| cue generator | 222K | O (선택) |
| **llm (FLAN-T5-base)** | **248M** | **X** |
| llm_proj | 591K | **X** |
| **추론 경로 합계** | **176M** | (전체 424M) |

**실측 속도**: batch 32 → 183.5 ms → **이미지당 5.73 ms (174 img/s)**, RTX 6000 Ada 기준.

> ⚠ **"LLM-free"이지 "ViT-only"가 아니다.** connector 2개(80M)가 추론 경로에 그대로 남는다.
> ViT 만 쓰는 구성은 원논문 Table 3 의 베이스라인이며 HTER 21.96 (전체 12.68 대비 9.3점 나쁨).

### 1.5 손실

$$\mathcal{L} = \lambda_1 \mathcal{L}_{content} + \lambda_2 \mathcal{L}_{style} + \lambda_3 \mathcal{L}_{cls} + \lambda_4 \mathcal{L}_{cue}$$

기본값 `0.4 / 0.4 / 0.15 / 0.05`.

> ⚠ 주 태스크인 분류가 **0.15 뿐**이고 언어 감독이 0.8 을 차지한다. 의도된 설계다 —
> 분류기를 직접 세게 학습시키는 대신 언어 감독으로 공유 표현을 만들고 분류기는 거기서
> 얇게 읽어낸다. 함부로 키우면 그 구조가 깨진다.

**cue 손실**: 위조 영역 마스크 주석이 없어 타깃을 **이진 라벨의 공간 broadcast**
(fake = 전면 1, live = 전면 0)로 만든다. SmoothL1(beta=0.01).
공간적으로 균일한 출력을 강요받은 네트워크가 판별에 유리한 국소 증거로 수렴하는 부수 효과를
노린 보조 감독이다. **정식 세그멘테이션이 아니며 정량 평가 대상이 아니다.**

학습 시 **fake 샘플에만** 가우시안 노이즈(σ=0.5)를 융합 출력에 주입한다(cls 토큰 제외).
추론에서는 넣지 않는다.

### 1.6 원논문 성능

7개 벤치마크(MCIO + WCS) 평균, CelebA-Spoof 단독 학습:

| Method | Venue | HTER↓ | AUC↑ | TPR@FPR=1%↑ |
|---|---|---|---|---|
| SSDG-R | CVPR'20 | 28.71 | 77.29 | 18.83 |
| ViT | ECCV'22 | 20.86 | 85.05 | 43.89 |
| SAFAS | CVPR'23 | 33.40 | 70.89 | 8.92 |
| FLIP-MCL | ICCV'23 | 18.82 | 86.94 | 46.48 |
| CFPL | CVPR'24 | 16.15 | 88.65 | 53.98 |
| **InstructFLIP** | MM'25 | **12.68** | **93.68** | **65.23** |

**Table 3 (분기 ablation)**

| CB | SB | Cue | HTER↓ | AUC↑ | TPR@FPR1↑ |
|---|---|---|---|---|---|
| – | – | – | 21.96 | 82.88 | 26.89 |
| ✓ | – | – | 14.25 | 90.23 | 32.32 |
| – | ✓ | – | 16.25 | 88.27 | 41.92 |
| ✓ | ✓ | – | 13.25 | 91.21 | 60.84 |
| ✓ | ✓ | ✓ | **12.68** | **93.68** | **65.23** |

**Table 4 (LLM ablation)**

| Method | HTER↓ | AUC↑ | TPR@FPR1↑ |
|---|---|---|---|
| CFPL | 16.15 | 88.65 | 53.98 |
| InstructFLIP† (LLM → 분류헤드) | 14.39 | 89.97 | 48.33 |
| InstructFLIP | **12.68** | **93.68** | **65.23** |

---

## 2. 원저자 구현과의 차이

| 항목 | 원본 | 여기 | 이유 |
|---|---|---|---|
| Q-Former | LAVIS (저자 fork) | 자립 구현 | fork 의존이 취약. 낡은 pin |
| 백본 | `clip.load("ViT-B/16")` 하드코딩 | 레지스트리 6종 | 실험 축으로 |
| 변환기 | Q-Former 고정 | qformer/attn_pool/connector | 원논문에 ablation 없음 |
| 융합 FFN | residual 없음 | residual 복원 | 표준 트랜스포머 |
| SSL 분기 | 존재하나 가중치 0 | 제거 | 죽은 코드 |
| 정규화 상수 | CLIP 백본에 ImageNet 상수 | 백본별 자동 | std 0.229 vs 0.269 |
| 모델 선택 | 타깃 테스트셋 평균 HTER | val 로 선택 | 누수 제거 |
| 지표 | HTER/AUC/TPR | + APCER/BPCER/ACER/R@FPR | CelebA-Spoof 공식 비교 |
| 학습 프레임워크 | PyTorch Lightning | **accelerate** | 의존 경량화 |
| 정렬 | mxnet MTCNN | facenet-pytorch + 이식 | mxnet 유지보수 종료 |

### 2.1 원본에서 발견한 버그

| 위치 | 문제 | 영향 |
|---|---|---|
| `ca_dataset.py:56` | `self.data = data` 가 mode/resolution 필터를 전부 덮어씀 | live/fake 균형·해상도 층화가 **실제로는 일어나지 않음** |
| `ca_dataset.py:91-112` | live 의 미주석 라벨에 -1 오프셋 → 오답 상수 | style 감독 오염 |
| `rgb_dataset.py:43-46` | `live_only` 가 fake 를 고름 (반전) | `mode: all` 이라 헤드라인 결과에는 무영향 |
| `rgb_dataset.py:55-62` | video_id 가 필터 전 기준이라 정렬 어긋남 | 동일 |
| `_forward_lm_loss` | 배치 전체가 "No GT" 면 빈 텐서로 크래시 | live 만 담긴 배치에서 발생 가능 |

---

## 3. 실험 계획

### 3.1 축

| 축 | 값 | 묻는 것 |
|---|---|---|
| **백본** | clip-vit-b16 / dinov2-b14-reg / dinov2-s14 | dense 표현이 FAS 에 유리한가 |
| **변환기** | qformer / attn_pool / connector | 지시문 게이팅과 학습형 쿼리가 필요한가 |
| **LLM** | on / off | 언어 감독의 기여 |
| **분기** | content+style / content only | 직교 분해의 기여 |
| **cue** | on / off | 보조 감독의 기여 |

### 3.2 변환기 축이 왜 중요한가

**원논문에 Q-Former 자체의 ablation 이 없다.** Table 3 의 베이스라인(`– – –`, 21.96)은
저장소 코드로 확인하면 `ImageEncoder + Classifier(768,2)` 가 전부다 —
**Q-Former·fusion·LLM·지시문 감독을 한꺼번에 뺀 것**이라 기여를 분리할 수 없다.

논문이 가장 가까이 간 지점:

```
12.68   전체
14.39   LLM 제거, Q-Former 유지 (Table 4 †)   ← LLM 기여 ≈ 1.7점
21.96   Q-Former·fusion·LLM 전부 제거          ← 나머지 7.6점이 묶음
```

**Q-Former + fusion 이 7.6점, LLM 이 1.7점**으로 읽힌다. 묶음 쪽이 4배 이상 큰데
그 안의 몫은 아무도 나눠보지 않았다.

여기서는 단계적으로 벗겨낸다:

```
qformer      지시문 + 쿼리
   ↓ 지시문 제거
attn_pool    쿼리만            ← 차이 = 지시문 게이팅 기여
   ↓ 쿼리 제거
connector    MLP 투영만        ← 차이 = 학습형 쿼리 기여
   ↓ 전부 제거
(ViT + linear)                 ← Table 3 베이스라인
```

저장소에 `qformer_layers` ablation config(1/2/4/8층)가 있으나 **논문에 표가 없다.**
층수를 몇으로 뒀는지조차 본문에 없고 코드 기본값 `2` 로만 알 수 있다.

### 3.3 백본 축 가설

DINOv2 공개 지표 (동일 86M):

| | ImageNet linear | ADE20k linear mIoU | NYUd RMSE |
|---|---|---|---|
| CLIP ViT-B/16 | ~80.2 | — | — |
| DINOv2 ViT-B/14 | **84.5** | **47.3** | **0.399** |
| OpenCLIP ViT-G/14 (1.8B) | — | 39.3 | 0.541 |

**86M DINOv2 가 1.8B OpenCLIP 을 dense 과제에서 압도한다.**

FAS 의 판별 근거는 모아레, 인쇄 망점, 스크린 반사 같은 **국소 텍스처 아티팩트**다.
InstructFLIP 구조에서 dense 특징이 직접 쓰이는 곳도 둘이다 — cue generator(14×14 dense
prediction)와 style 분기(레이어별 통계).

> ⚠ **가설이지 결과가 아니다.** 위 수치는 자연 이미지 벤치마크이고, FAS 문헌에서 DINOv2 가
> CLIP 을 이긴다는 정립된 결과는 없다. 다만 `ImageEncoder` 하나만 갈아끼우면 되니 실험 비용이 낮다.

한편 CLIP 을 옹호할 근거도 있다 — optimizer 스텝이 ~1,067회로 짧아, 출발점이 언어 쪽을
향해 있는 게 LM loss 수렴에 유리할 수 있다.

---

## 4. 관련 논문 비교

### InstructFLIP vs FaceCoT

| | **InstructFLIP** | **FaceCoT** |
|---|---|---|
| arXiv | 2507.12060 (2025-07-16) | 2506.01783 (2025-06-02) |
| 지위 | ACM MM 2025 채택 | preprint |
| 기여 | **아키텍처** | **데이터셋** + 학습전략 |
| 언어 활용 | 객관식 instruction | 6단계 서술형 CoT |
| 백본 | CLIP ViT-B/16 (86M) | MiniCPM-V-2.6 (8B) |
| 추론 시 LLM | **제거** | 사용 |
| 추론 규모 | **176M** | 8B |
| 해석 | cue map | CoT 텍스트 |
| 학습 자원 | RTX 4090 ×1 | A100 ×8 |
| 데이터 | CelebA-Spoof 재활용 | 신규 1.08M VQA 구축 |
| 평가 | 7개, 평균 HTER 12.68 / AUC 93.68 | 11개, 평균 HTER 6.30 / AUC 97.77 |

> ⚠ **수치를 나란히 놓으면 안 된다.** 벤치마크 구성이 다르다 — FaceCoT 쪽은 3D 마스크 계열이
> 포함되고 학습에 WFAS 까지 쓴다. FaceCoT 는 InstructFLIP 을 baseline 에 넣지 않았다(제출이
> 한 달 앞섬). 굳이 비교하려면 FaceCoT 의 `Ours-CelebA` 행이 학습 조건이 같다.

두 방향은 **상보적**이다. FaceCoT 의 CoT 어노테이션을 content/style instruction 으로 변환해
넣는 것이 자연스러운 후속 실험이다.
