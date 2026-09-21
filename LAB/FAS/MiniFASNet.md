# MiniFASNet — 경량 FAS 대조군

0.43M 파라미터. InstructFLIP(176M) 의 **약 400배 작다.**
"그 규모가 정말 값을 하는가"를 같은 데이터·같은 지표·같은 엔진으로 재기 위한 대조군.

> 공통 사항(태스크 정의·지표·데이터셋·코드 구조·설정·실행)은 **[FAS.md](FAS.md)** 를 본다.
> 이 문서는 **알고리즘**만 다룬다. 주 알고리즘은 [InstructFLIP.md](InstructFLIP.md).

- 원저장소: <https://github.com/minivision-ai/Silent-Face-Anti-Spoofing>
- 구현: [`pierrotfr/FAS/models/minifasnet.py`](../../pierrotfr/FAS/models/minifasnet.py)

---

> ⚠ **이 저장소의 MiniFASNet 은 원본 레시피가 아니다 (2026-09-18 확인).**
> 원본(MiniVision Silent-Face-Anti-Spoofing)은 ① 분류 + **입력 FFT 스펙트럼 복원 보조감독**(손실 0.5:0.5),
> ② 얼굴 박스 **2.7배/4배** 넓은 crop, ③ **두 모델(V2@2.7 + V1SE@4.0) 예측 합산**,
> ④ SGD lr 0.1 · 배치 1024 · 25 epoch · 자사 비공개 데이터로 학습한다.
> 여기서는 **네트워크 구조만** 옮겼고 ①~③ 이 없다. 1차 런(`minifasnet_minifasnet`, test ACER 19.66)은
> 미수렴까지 겹쳐 MiniFASNet 의 성능으로 읽으면 안 된다 — [Phase 1 ④](Exp/Phase_1_첫_학습과_PC_screen.md).
> 결과 표는 [FAS.md §8](FAS.md#8-실험-결과).

## 1. 개요

출처: MiniVision **Silent-Face-Anti-Spoofing** → [yakhyo/face-anti-spoofing](https://github.com/yakhyo/face-anti-spoofing)
(ONNX 배포판은 `uniface.spoofing.MiniFASNet`). 구조는 **MobileFaceNet 계열**이다.

**왜 넣는가**: InstructFLIP 은 176M, MiniFASNet 은 **0.43M — 400배 차이**다.
"그 규모가 정말 값을 하는가"를 **같은 CelebA-Spoof·같은 엔진·같은 지표**로 잰다.
논문끼리는 데이터·프로토콜이 달라 비교가 안 되지만 여기서는 조건이 통제된다.

## 2. 구조

```
입력 [B,3,80,80]                            ← 224 가 아니라 80. 아주 작다
  stem         Conv3×3 s2 + BN + PReLU   → [B, 32,40,40]
  stem_dw      DWConv3×3                 → [B, 32,40,40]
  transition1  InvRes s2  (expand 103)   → [B, 64,20,20]
  stage2       InvRes ×4                 → [B, 64,20,20]
  transition2  InvRes s2  (expand 231)   → [B,128,10,10]
  stage3       InvRes ×6                 → [B,128,10,10]
  transition3  InvRes s2  (expand 308)   → [B,128, 5, 5]
  stage4       InvRes ×2                 → [B,128, 5, 5]
  final_expand Conv1×1                   → [B,512, 5, 5]
  final_dw     DWConv5×5 (활성 없음)      → [B,512, 1, 1]   ★ GDConv
  Flatten → Linear(512→128) → BN1d → Dropout → Linear(128→3)
출력 [B,3]
```

**InvertedResidual**: `expand(1×1) → depthwise(3×3) → project(1×1, 활성 없음)`.
MobileNetV2 와 같되 활성이 ReLU6 가 아니라 **PReLU** 다(얼굴 도메인 관행).
`in_ch == out_ch and stride == 1` 일 때만 residual 이 붙는다.

**★ GDConv (Global Depthwise Conv)** — 이 구조의 핵심이다.

GAP 대신 **5×5 depthwise conv 로 5×5 격자를 1×1 로 접는다.**
GAP 는 모든 위치를 동일 가중치로 평균내지만 GDConv 는 **위치별 가중치를 학습**한다.
얼굴처럼 공간 배치가 고정된 입력에서 유리하다 — MobileFaceNet 의 시그니처다.

구현에서는 `input_size // 16` 으로 커널을 잡는다(stride 2 가 네 번이므로).
80 → 5×5, 128 → 8×8. 입력 해상도를 바꿔도 자동으로 맞는다.

## 3. 3-class 출력

**이진이 아니다.** 원 규약:

| 인덱스 | 클래스 | 해당 spoof_type |
|---|---|---|
| 0 | **2D fake** | Photo · Poster · A4-paper · PC/Pad/Phone screen (1,2,3,7,8,9) |
| 1 | **real** | Real face (0) |
| 2 | **3D fake** | 2D face/upper-body/region mask · 3D mask (4,5,6,10) |

인쇄·재생과 마스크는 **물리적으로 다른 아티팩트**다(인쇄=망점·평면성, 재생=모아레·베젤
반사, 마스크=재질감·경계선). 하나의 "fake" 로 묶으면 서로 다른 신호를 같은 방향으로
밀어야 한다.

CelebA-Spoof 의 `spoof_type` 에서 3-class 라벨을 자동 생성한다
(`pierrotfr/FAS/instructions.py` 의 `label3()`).

평가는 **이진으로 환산**해 InstructFLIP 과 같은 축에 올린다:

```python
score = 1 - P(real)     # = P(2D fake) + P(3D fake)
```

`num_classes=2` 로 두면 이진으로도 쓸 수 있다.

## 4. 두 변종

| | V1SE | V2 |
|---|---|---|
| 파라미터 | **433,586** | **434,560** |
| SE 모듈 | O (각 stage **마지막 블록만**) | X |
| SE 파라미터 | 19,232 | 0 |
| Dropout | **0.75** | 0.2 |
| 원본 crop scale | **4.0** | 2.7 |
| stage2 확장비 | 13/26/13/52 | 13/13/13/13 |
| stage3 확장비 | 154/52/26/52/26/26 | 231/52/26/77/26/26 |

파라미터 총량이 거의 같다 — V1SE 는 SE 로 19,232 를 더한 대신 stage 확장채널을 줄여 상쇄했다.

**포팅 정확도**: 원본 구현을 직접 실행해 대조했다. 두 값 모두 일치한다
(`tests/FAS/test_minifasnet.py::test_param_count_matches_reference`).

## 5. 블록별 파라미터 (V2)

| 블록 | 파라미터 | 비중 |
|---|---|---|
| stage3 | 120.2K | **27.7%** |
| transition3 | 83.7K | 19.3% |
| final_expand | 67.1K | 15.4% |
| linear (512→128) | 65.5K | 15.1% |
| transition2 | 48.1K | 11.1% |
| 나머지 | ~50K | 11% |

**stage3 + transition3 에 47% 가 몰린다** — 10×10 → 5×5 해상도 구간이다.

## 6. ⚠ crop 맥락 — 공정성 문제

원본은 얼굴 bbox 를 **2.7~4.0배 넓게** 잘라 배경 맥락을 함께 본다:

```python
scale = min((src_h-1)/box_h, (src_w-1)/box_w, self.scale)
# bbox 중심에서 scale 배 확장 → 80×80 리사이즈
```

사진의 테두리, 화면의 베젤, 손에 든 모습, 마스크 경계 — **위조 증거가 얼굴 밖에 있다.**

그런데 이 저장소의 기본 전처리는 **랜드마크 정렬 타이트 crop**(padding 0.37)이라
그 맥락이 없다. **MiniFASNet 에 불리한 조건이다.**

공정하게 비교하려면 `pierrotfr/data/align.py` 의 `padding` 을 키운 넓은 crop 데이터셋을
따로 만들어야 한다. 현재 미구현 — [FAS.md §7 알려진 한계](FAS.md#7-알려진-한계) 참조.

정규화도 다르다. 원본은 `astype(np.float32)` 만 하고 **0~255 원본 스케일**을 그대로 넣어
첫 BN 이 흡수하게 둔다. 여기서는 `norm: 'none'`(ToTensor 의 0~1 까지만)으로 근사했다.

## 7. InstructFLIP 과의 대비

| | **MiniFASNet V2** | **InstructFLIP** |
|---|---|---|
| 파라미터 | **0.43M** | 176M (추론) |
| 입력 | 80×80 | 224×224 |
| crop | 얼굴 **2.7배** (배경 포함) | 랜드마크 정렬 타이트 |
| 출력 | 3-class | 2-class |
| 감독 | 라벨만 | 언어 감독 + cue map |
| 일반화 전략 | 없음 (도메인별 재학습) | meta-domain 단일 학습 |
| 사전학습 | **없음 (스크래치)** | CLIP / DINOv2 |
| 배포 | ONNX, CPU 실시간 | GPU 5.73 ms |

전처리 철학이 정반대다 — MiniFASNet 은 **배경을 넣어** 촬영 맥락을 보고, InstructFLIP 은
**얼굴만 정렬해** 텍스처에 집중한 뒤 style 분기로 촬영 조건을 따로 처리한다.

## 8. 사전학습 가중치

**ImageNet 사전학습은 존재하지 않는다.** 채널 config 가 이 모델 전용이라
(`expand 103/231/308` 같은 비정형 값) 어떤 표준 백본 체크포인트와도 호환되지 않는다.
MobileFaceNet 얼굴인식 가중치도 채널 수가 달라 직접 로드가 안 된다.

선택지는 둘이다:

1. **스크래치 학습** (현재 프리셋) — `lr0: 1e-3`, `backbone_lr_mult: 1.0`.
   0.43M 이라 CelebA-Spoof 규모에서 충분히 학습된다.
2. **Silent-Face FAS 가중치로 초기화** — MiniVision 이 배포한 FAS 학습 완료 가중치
   (`uniface` 의 ONNX 또는 minivision 원본 `.pth`). ImageNet 이 아니라 **FAS 로 학습된**
   것이라 오히려 도메인이 맞다. 다만 그 가중치가 어떤 데이터로 학습됐는지 불명확해
   CelebA-Spoof 평가에 누수 위험이 있다 — 비교 실험에는 스크래치를 권한다.

## 9. 실행

```bash
FAS_PRESET=minifasnet       python train_fas.py    # V2  (0.43M)
FAS_PRESET=minifasnet_v1se  python train_fas.py    # V1SE
FAS_PRESET=clip             python train_fas.py    # InstructFLIP (176M)
```

프리셋 정의:

```python
'minifasnet': {
    'model_name': 'minifasnet',
    'image_size': 80,
    'norm': 'none',                       # 원본은 정규화 없이 BN 이 흡수
    'model_extra': {'variant': 'v2', 'num_classes': 3, 'use_llm': False},
    'w_content': 0.0, 'w_style': 0.0, 'w_cls': 1.0, 'w_cue': 0.0,
    'batch_size': 256, 'grad_accum': 1,   # 0.43M 이라 크게 잡아도 된다
    'lr0': 1.0e-3, 'backbone_lr_mult': 1.0,   # 스크래치라 백본 감쇠 불필요
    'weight_decay': 5.0e-4,
},
```

## 10. 엔진 통합 — 모델별 분기 없이

3-class 모델은 이진 라벨로 CE 를 걸 수 없다. 모델이 **계약을 통해** 자기 요구를 알린다:

```python
# minifasnet.py — 모델이 어떤 라벨을 쓸지 지정
return {"logits": logits, "label_key": "label3" if self.num_classes == 3 else "label", ...}

# engine.py — 엔진은 읽기만 한다
labels = batch[out.get("label_key", "label")]
```

학습 루프에 `if isinstance(model, MiniFASNet)` 같은 분기가 **하나도 없다.**
새 알고리즘은 `models/` 파일 하나 + `MODELS` 등록 + 프리셋 한 블록이면 끝난다.
