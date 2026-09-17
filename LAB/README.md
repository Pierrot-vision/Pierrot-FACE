# LAB — 실험 기록 (읽기 전용 사본)

이 저장소는 **추론 배포본**입니다. 여기 실린 문서는 학습 저장소
(`Pierrot_FR_Lab`)에서 수행한 실험의 기록을 그대로 옮겨 온 것이고,
**이 저장소에서 재현할 수 있는 것은 그 결과의 평가·추론뿐**입니다.

> ⚠ 문서 안의 실행 명령(`python train_fa2d.py`, `train_fa3d.py`, `sync_tables.py`,
> `FA3D_PRESET=…` 등)과 코드 파일 링크 일부는 **학습 저장소의 것**입니다. 이 저장소에는
> 학습 코드가 없습니다. 여기서 되는 것은 [`eval/`](../eval/) 와 [`scripts/`](../scripts/) 입니다.

## 문서 목록

| 문서 | 내용 |
|---|---|
| **[FA2D_vs_FA3D.md](FA2D_vs_FA3D.md)** | **두 태스크의 차이를 한 표로** — 출발점 · 데이터 · 모델 · 지표 · 무엇이 들었나 · 현재 성적 |
| **FA2D** — 2D 얼굴 정렬 | |
| [FA2D/FA2D.md](FA2D/FA2D.md) | 태스크 공통 — 랜드마크 규약 · 데이터셋 · NME 정규화 기준 · 조사한 저장소 |
| [FA2D/HRFFA.md](FA2D/HRFFA.md) | 기반 알고리즘 HRFFA 조사 |
| [FA2D/Ours.md](FA2D/Ours.md) | **구현 · 실험 기록** — HRFFA · Peppa 전면 대조 · 세 벤치마크 통합 표 |
| [FA2D/Exp/](FA2D/Exp/) | 단계별 기록 — Phase 1 조사 → 16 얼굴 크롭 교사와 최종 학생 |
| **FA3D** — 3D 밀집 얼굴 정렬 | |
| [FA3D/3DDFA_V2.md](FA3D/3DDFA_V2.md) | **알고리즘** — 3DMM 파라미터의 의미 · VDC/WPDC/fWPDC 유도 · meta-joint · svs |
| [FA3D/FA3D.md](FA3D/FA3D.md) | **구현 · 실험 기록** — 버그 · 측정 · 분석 |
| [FA3D/Exp/](FA3D/Exp/) | 단계별 기록 — Phase 1 구현·기준선 → 12 잡음 기준선과 flip-TTA |

## 이 저장소에서 되는 것

```bash
python eval/FA2D/evaluate.py --ckpt runs/fa2d/<런>/best.pth   # ① WFLW · ② LaPa · ③ 난이도
python eval/FA3D/evaluate.py --ckpt runs/fa3d/<런>/best.pth --deployed mb1 --aflw
python scripts/FA3D/shape_accuracy.py --ckpt …                 # FA3D — NME 가 재지 않는 형상 정확도
```

## 읽을 산출물

`runs/` 는 학습 저장소를 가리키는 심볼릭 링크이고 커밋하지 않습니다 (`.gitignore`).

| 경로 | 내용 |
|---|---|
| `runs/fa2d/<실험ID>/best.pth` · `runs/fa3d/<실험ID>/best.pth` | 체크포인트 — **설정이 안에 들어 있어** 구조·크롭 규약이 자동 복원됩니다 |
| `outputs/fa2d/<실험ID>/` · `outputs/fa3d/<실험ID>/` | 이 저장소가 만드는 추론 시각화 — 언제든 재생성 |
