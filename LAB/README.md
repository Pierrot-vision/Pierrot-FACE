# LAB — 실험 기록 (읽기 전용 사본)

이 저장소는 **추론 배포본**입니다. 여기 실린 문서는 학습 저장소
(`Pierrot_FR_Lab`)에서 수행한 실험의 기록을 그대로 옮겨 온 것이고,
**이 저장소에서 재현할 수 있는 것은 그 결과의 평가·추론뿐**입니다.

> ⚠ 문서 안의 실행 명령(`python train_fa3d.py`, `scripts/FA3D/sync_tables.py`,
> `FA3D_PRESET=…` 등)은 **학습 저장소의 것**입니다. 이 저장소에는 학습 코드도,
> 실험 표를 생성하는 스크립트도 없습니다. 여기서 되는 것은
> [`eval/FA3D/`](../eval/FA3D/) 와 [`scripts/FA3D/`](../scripts/FA3D/) 입니다.

> ⚠⚠ **FA3D 의 NME 는 사실상 자세 지표입니다.** 3D 형상(identity)은 거의 재지
> 않습니다 — 형상을 GT 로 완벽히 바꿔도 1.3% 개선뿐이고, 형상을 안 내고 평균
> 얼굴을 쓰면 오히려 낫습니다. 형상 정확도는 이 저장소의
> `scripts/FA3D/shape_accuracy.py` 로 따로 잽니다 (그 스크립트는 여기 있습니다).

## 문서 목록

| 문서 | 내용 |
|---|---|
| [FA3D/3DDFA_V2.md](FA3D/3DDFA_V2.md) | **알고리즘** — 3DMM 파라미터의 의미 · VDC/WPDC/fWPDC 유도 · meta-joint · svs · SynergyNet 비교. 추론 코드를 읽기 전에 이것부터 보면 62-d 가 무엇인지 알 수 있습니다 |
| [FA3D/FA3D.md](FA3D/FA3D.md) | **구현 · 실험 기록** — 버그 · 측정 · 분석. 이 저장소가 재현하는 수치의 출처 |
| [FA3D/Exp/](FA3D/Exp/) | **단계별 기록** — Phase 1 구현·기준선 → 10 추론축 소진 |

## 이 저장소에서 되는 것

```bash
bash scripts/FA3D/eval.sh                        # AFLW2000-3D / AFLW NME (배포 가중치와 나란히)
bash scripts/FA3D/infer.sh                       # 사진·영상 추론 + 예측 격자 + 3D 재렌더 + 속도
python scripts/FA3D/shape_accuracy.py --ckpt …   # NME 가 재지 않는 것 (형상 정확도)
python scripts/FA3D/error_analysis.py --ckpt …   # 샘플 단위 오차 분해 (자세·부위·꼬리)
```

## 읽을 산출물

`runs/` 는 학습 저장소를 가리키는 심볼릭 링크이고 커밋하지 않습니다 (`.gitignore`).

| 경로 | 내용 |
|---|---|
| `runs/fa3d/<실험ID>/best.pth` | val 최고 시점 체크포인트 — **설정이 안에 들어 있어** 백본·입력크기·테두리 전처리가 자동 복원됩니다 |
| `runs/fa3d/<실험ID>/results.json` | 학습 저장소가 남긴 최종 벤치마크 결과 (이 저장소로 다시 재면 같은 값이 나와야 합니다) |
| `outputs/fa3d/<실험ID>/` | 이 저장소가 만드는 추론 시각화 — 언제든 재생성 |
