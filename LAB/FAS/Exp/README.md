# Exp — FAS 실험 기록

한 **국면(Phase)** = md 파일 하나. 국면은 "무엇이 문제인지 알아낸 순간"마다 끊는다.
형식은 [FA3D/Exp/](../../FA3D/Exp/README.md) 와 같다 — 문제 · 가설 · 실험 · 결과 · 결론.
**틀린 가설과 측정 오류도 남긴다.**

## 국면 목록

| # | 국면 | 기간 | 결과 |
|---|---|---|---|
| [1](Phase_1_첫_학습과_PC_screen.md) | 첫 학습과 PC screen | 09-18 | InstructFLIP 공식 ACER **4.93** (AENet 1.63 미달) · val 기준 운영점 **1.52** · 놓친 위조의 75% 가 PC screen · MiniFASNet 은 원본 레시피 아님(판정 불가) |
| [2](Phase_2_넓은crop과_촬영조건_처방.md) | 넓은 crop 과 촬영 조건 처방 | 09-18 | ❌ 후보 채택 안 함 — PC screen 은 오히려 악화(47→54%), 나머지 7개 유형은 크게 개선 · **PC 는 crop·화질 문제가 아니다** · AENet 공개 가중치의 공식 ACER 는 논문 1.63 이 아니라 **2.73** |

## 문서 동기

실험 표와 최고 레시피는 손으로 쓰지 않는다. `python scripts/FAS/sync_tables.py` 가
README 와 [FAS.md](../FAS.md) 에 함께 생성하고, `--check` 로 어긋남을 잡는다.
