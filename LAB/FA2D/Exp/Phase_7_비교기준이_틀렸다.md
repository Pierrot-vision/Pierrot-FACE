# Phase 7 — 비교 기준이 틀렸다: HRFFA 수치는 홀드아웃이 아니다

기간 09-06 · 결과 **참조표 재작성 · 진짜 기준선은 Peppa 3.95 / D-ViT 3.75**

## 문제

"우리 91.5M 이 HRFFA 9.0M(3.36) 보다 나쁘다(4.553)" 를 전제로 원인을 찾고 있었다.
크기가 8 배인데 지면 구조에 결함이 있다는 뜻이므로 그쪽만 파고 있었다.

## 가설

전제부터 확인한다 — 그 3.36 은 우리 4.553 과 같은 종류의 숫자인가.

## 실험

HRFFA README §1 을 읽었다.

> *"all splits of the real-image datasets — **including WFLW test**, 300W and COFW test —
> are used for training. The 'official' numbers therefore measure how well the models fit
> the training distribution and **must not be compared with published benchmark results.**"*

같은 표의 FR10 을 확인했다.

## 결과

| 모델 | WFLW NME | FR10 | 홀드아웃 |
|---|--:|--:|:-:|
| HRFFA vitl-320 | 1.51 | **0.00** | ❌ 학습셋 포함 |
| HRFFA vitt-256 | 3.36 | 0.28 | ❌ |
| HRFFA hg0-256 | 5.32 | 7.00 | ❌ |
| D-ViT (논문) | 3.75 | 1.76 | ✅ |
| Peppa Teacher@256 | **3.95** | — | ✅ |
| Peppa Student@256 | **4.35** | — | ✅ |
| 우리 `combined` | 4.553 | 3.16 | ✅ |

**FR10 = 0.00%** — 2,500 장 중 실패 0 건. 공개 SOTA(D-ViT)도 1.76% 다.
홀드아웃에서 나올 수 없는 값이고, 암기의 지문이다.

HRFFA 가 공개한 유일한 비누출 지표는 **`300wlp_val`**(홀드아웃 241장, head-NME):
vitl-320 0.0033 / vitt-256 0.0071 / vitt-096 0.0100 / hg0-256 0.0158 / hg0-096 0.0185.

## 결론

- **격차는 35%(vs 3.36)가 아니라 4~10%(vs Peppa 3.95 · D-ViT 3.75)였다.**
  잘못된 기준으로 두 차례 원인을 오진했다.
- **Peppa 가 진짜 목표선이다** — WFLW train 학습 / test 평가로 우리와 조건이 같다.
- HRFFA 행은 **크기·FLOPs 만** 비교 가능하다.
- 조건 차이 하나를 표에 명시하기로 했다: 우리·HRFFA 는 **머리 크롭 + 여백**,
  Peppa·D-ViT 는 **얼굴 크롭**. HRFFA 자신도 *"considerable disadvantage against
  paper benchmarks"* 라 적었다.
- 재발 방지: `scripts/FA2D/compare_references.py` 가 표를 생성한다. 손으로 쓰지 않는다.
  홀드아웃 여부·크롭 종류·표본수를 열로 강제한다.
