# FA2D — 2D 얼굴 정렬 (2D Face Alignment)

Pierrot_FR Lab 의 **2D 랜드마크 정렬** 태스크 문서. 조사한 공개 저장소들이 **공유하는**
태스크 정의·랜드마크 규약·데이터셋·지표를 모은다.

> ## 📐 알고리즘은 별도 문서로
>
> | 문서 | 알고리즘 | 원저장소 |
> |---|---|---|
> | **[HRFFA.md](HRFFA.md)** | HRFFA — 고각도 강건 whole-head 정렬 + 증류 + ONNX | [PINTO0309/High-Angle_Robust_Fast_FaceAlignment](https://github.com/PINTO0309/High-Angle_Robust_Fast_FaceAlignment) |
> | **[HRFFA_Plus.md](HRFFA_Plus.md)** | HRFFA+ — 우리 확장(시간축·상태 헤드·지터 지표) | 위 저장소의 `hrffa-plus` 브랜치 |
>
> 아래 §5 의 나머지 4개 저장소는 **조사 단계**다 — README·구조만 확인했고 코드는 아직
> 읽지 않았다. 개별 문서는 코드를 읽은 뒤에 만든다.

조사한 저장소 사본: `<작업 루트>/2d_aligments/`

관련 태스크: **3D 밀집 정렬은 [FA3D](../FA3D/FA3D.md)** 를 본다 —
3DMM 파라미터를 회귀해 3D 정점을 복원하는 별개 계열이다.

---

## 1. 태스크

이미지 안 얼굴에서 **미리 정의된 개수의 2D 점**(눈꼬리·코끝·입술 윤곽 등)을 찾는다.

```
이미지 → [검출: 얼굴/머리 어디?] → crop → [정렬: 점이 어디?] → 랜드마크
                                                                    ↓
                          얼굴 인식 정렬 · FAS 전처리 · 표정 · 시선 · 자세
```

Pierrot_FR 안에서의 위치: **다른 태스크의 전처리**다.
[FAS](../FAS/FAS.md) 의 얼굴 정렬 crop, 얼굴 인식의 유사변환 정렬이 여기 출력을 쓴다.
정렬 품질이 나쁘면 하류 태스크가 통째로 흔들린다.

### 1.1 검출 crop 의 두 갈래

정렬 모델은 **무엇을 잘라 넣느냐**로 갈린다. 성능표만 봐서는 안 보이는 축이다.

| | **face crop** | **head crop** |
|---|---|---|
| 대상 | 얼굴 bbox (타이트) | 머리 전체 (+ 배경 여백) |
| 채택 | 대부분의 논문·저장소 | HRFFA |
| 장점 | 벤치마크 관행과 일치, 해상도 효율 | 검출기가 안정적, 극단 자세에서 얼굴 bbox 자체가 무너지지 않음 |
| 단점 | 프로필·고각도에서 얼굴 검출이 먼저 실패 | 유효 해상도 손실, 논문 벤치마크와 직접 비교 불가 |

HRFFA 의 주장: **객체 검출기는 얼굴보다 머리를 훨씬 안정적으로 잡는다.**
pitch 가 커져 얼굴이 거의 안 보이는 프레임에서도 머리는 잡힌다
([HRFFA.md §1](HRFFA.md#1-설계-전제--face-crop-이-아니라-head-crop)).

---

## 2. 랜드마크 규약

같은 "68점"이라도 정의가 다르면 섞을 수 없다. 조사 범위에서 나온 규약:

| 점 수 | 이름 | 출처 | 쓰는 곳 |
|---|---|---|---|
| 5 | 눈 2 · 코 1 · 입꼬리 2 | RetinaFace 계열 | 얼굴 인식 정렬 (uniface) |
| 29 | COFW | COFW | HRFFA (`cofw29`) |
| 68 | iBUG / Multi-PIE | 300W · 300W-LP | 사실상 표준 (face-alignment, Peppa TRAIN, HRFFA `ibug68`) |
| 98 | WFLW | WFLW | 고난도 벤치 (Peppa, PIPNet, HRFFA `wflw98`) |
| 106 | 2d106det | InsightFace | uniface |
| 468 / 478 | Face Mesh | MediaPipe 계열 | uniface (3D 밀집) |

**좌우 반전 매핑(flip mapping)** 이 규약마다 다르다. 증강에서 hflip 을 쓰려면 점 인덱스를
짝지어 교환해야 하고, 이 표가 틀리면 **좌우가 뒤바뀐 채로 학습된다** — 손실은 정상적으로
내려가므로 조용히 망가진다. HRFFA 는 이걸 `dataset/meta/{ibug68,wflw98,cofw29}.json` 에
데이터로 두고 순열이 involution 인지 테스트로 검사한다
(`tests/test_300wlp_flip_index.py`). 규약을 코드에 하드코딩하지 않는 편이 안전하다.

---

## 3. 데이터셋

| 이름 | 규모 | 규약 | 특징 |
|---|---|---|---|
| **300W** | 3,148 train / 689 test | 68 | 표준. test 가 common(554) / challenge(135) 로 나뉨 |
| **WFLW** | 7,500 / 2,500 | 98 | test 가 pose·expression·illumination·make-up·occlusion·blur 6개 부분집합으로 층화 |
| **LaPa** | 18,176 / 2,000 / 2,000 | 106 | 얼굴 파싱 데이터셋의 랜드마크. 우리 학습의 최대 소스 |
| **COFW** | 1,345 / 507 | 29 | 가림(occlusion) 중심. 점마다 가시성 라벨 있음 |
| **300W-LP** | 61,225 (합성 확장) | 68 | 3DMM 으로 대각도 yaw 를 합성. 자세 GT(회전행렬) 보유 |
| **AFLW2000-3D** | 2,000 | 68 | 3D 정렬 평가용 → [FA3D](../FA3D/FA3D.md) 쪽 |

**우리가 학습에 쓰는 것 (09-16 기준, `full_v12` · `stu_final_v2`)**: WFLW train 7,500 ·
LaPa train+val 20,168 · COFW **전 분할** · 300W **전 분할**. 09-15 부터 COFW·300W 는
우리 평가셋이 아니므로 전 분할을 넣는다. **평가 전용으로 남기는 것은 WFLW test 2,500 과
LaPa test 2,000** 이다. 09-12 이전 구성(`full_v11`)은 네 셋 모두 train 분할만 써서 32,161 장이었다.
확보 경로: COFW = CaltechDATA color `.mat` + MMPose 주석, 300W = dlib 배포 묶음 + MMPose 주석
(iBUG 원본은 다운로드 양식이 필요하다). 참조 구현 HRFFA 는 WFLW test 까지 학습에 넣는다 —
대신 **LaPa 는 쓰지 않는다**(소스: 300wlp · wflw · selftrain_v2 · selftrain_lookup · 300w · cofw).

**300W-LP 의 함정**: 합성 확장이라 `|yaw|` 가 커지면 투영된 라벨이 실제 렌더 픽셀에서
어긋난다. HRFFA 는 이걸 실측해 클린 학습에서 `source_yaw_max: {300wlp: 10}` 으로
`|yaw| ≤ 10°` 만 남긴다 (`configs/student_s_base.yaml`). 대각도 데이터를 얻으려고 쓴
데이터셋에서 대각도를 버리는 셈인데, 그만큼 라벨이 못 미덥다는 뜻이다.

---

## 4. 지표 — NME 정규화 기준이 다르면 비교 불가

**NME(Normalized Mean Error)** 가 주 지표다. 평균 L2 오차를 무언가로 나눈다.
**그 "무언가"가 문헌마다 다르고, 값이 통째로 달라진다.**

| 정규화 기준 | 정의 | 쓰는 곳 |
|---|---|---|
| **inter-ocular** | 양 눈 **바깥** 꼬리 거리 | WFLW·300W 표준. HRFFA 표의 기본 |
| **inter-pupil** | 양 눈 **동공** 거리 | 일부 COFW 보고 (D-ViT 논문). inter-ocular 보다 작아 **NME 가 커진다** |
| **bbox 대각/변** | 얼굴 상자 크기 | AFLW 계열 |
| **head-NME** | **crop 변 길이** | HRFFA 의 학습 중 val 지표 |

> ⚠ 정규화 기준이 다른 NME 를 같은 표에 놓으면 안 된다. HRFFA README 가 D-ViT 논문 행을
> "형식 참조용"이라고 못 박은 이유도 이것 + 학습/평가 조건 차이다.

보조 지표:

| 지표 | 의미 |
|---|---|
| **FR@x%** (Failure Rate) | NME 가 x%(보통 10%) 를 넘는 샘플 비율 |
| **AUC@x%** | NME 누적분포곡선의 0~x% 구간 면적. 높을수록 좋음 |

### 4.1 학습셋에 테스트 split 을 넣은 수치는 벤치마크가 아니다

조사 대상 중 HRFFA 는 **WFLW test·300W·COFW 를 전부 학습에 넣는다**
(`train_all_splits: true`). 그 상태의 표는 "벤치마크 점수"가 아니라 **학습 분포에 대한
적합도**다. README 가 명시하고 있고, 유일하게 누수 없는 계기는 300W-LP 홀드아웃
(`300wlp_val`, 241건)뿐이다.

같은 함정이 [FAS 쪽](../FAS/FAS.md#5-평가-지표) 에도 있다 — 임계값을 테스트셋 EER 에서
뽑는 관행. **문헌 수치를 인용할 때는 "무엇으로 나눴는지"와 "무엇이 학습에 들어갔는지"를
같이 적는다.**

---

## 5. 조사한 저장소

`<작업 루트>/2d_aligments/` 에 사본이 있다.

| 저장소 | 출력 | 백본 / 방법 | 배포 | 학습코드 | 상태 |
|---|---|---|---|---|---|
| [HRFFA](HRFFA.md) | 68/98/29 + 점별 가시성 3-class | DINOv3 ViT-L/16 교사 → ViT-T/16 · PP-HGNetV2-B0 학생 (온라인 증류) | ONNX / TFLite / WebGPU | O (전체) | **분석 완료** |
| Peppa_Pig_Face_Landmark | 98 (WFLW) | 자체 경량망, **교사-학생 증류**. 검출은 yolov5-face | PyTorch / ONNX | O (`TRAIN/`) | 조사 |
| face-alignment (1adrianb) | 68, 2D 및 3D | **FAN** (stacked hourglass, heatmap 회귀) | pip 패키지 | X (추론 전용) | 조사 |
| uniface | 5 · 68/98 (PIPNet) · 106 (2d106det) · 468/478 (Face Mesh) | 여러 모델을 감싼 **통합 라이브러리** | ONNX | X | 조사 |
| Face-alignment-mobilenet-v2 | 68 | MobileNetV2 (64×64 입력, 직접 좌표 회귀) | Caffe prototxt | 부분 | 조사 (가중치 없음) |

### 5.1 세 갈래 접근

조사 범위에서 좌표를 뽑는 방식이 셋으로 갈린다. 정확도-속도 트레이드오프의 축이다.

| 방식 | 대표 | 원리 | 성격 |
|---|---|---|---|
| **직접 회귀** | Face-alignment-mobilenet-v2 | FC 로 136(=68×2) 개 값을 바로 출력 | 가장 가볍다. 공간 정보를 FC 에서 잃어 정밀도 한계 |
| **히트맵** | face-alignment (FAN) | 점마다 2D 히트맵 → argmax/soft-argmax | 정확하다. 해상도에 비용이 비례하고 후처리 필요 |
| **쿼리 디코딩** | HRFFA | 점 = 학습형 쿼리, 트랜스포머 디코더가 좌표를 출력 | 점 수 확장이 쿼리 추가만으로 됨. 저해상 격자에서도 서브셀 정밀도 (POPoS 아암) |

### 5.2 증류가 반복해서 나온다

HRFFA 와 Peppa 가 **독립적으로 교사-학생 증류**를 택했다.

| | HRFFA | Peppa |
|---|---|---|
| 교사 | DINOv3 ViT-L/16 @320 (308M) | Teacher@256 (11.53M) |
| 학생 | ViT-T/16 (9.0M) · PP-HGNetV2-B0 (1.6M) | Student@256 (3.25M) |
| 교사 NME | 1.51 (WFLW, 적합도) | 3.95 (WFLW) |
| 학생 NME | 3.36 (vitt-256, ⚠ test 학습) | 4.35 |
| 증류 방식 | **온라인** (교사 동결, 같은 crop 을 교사 320 / 학생 256 으로) | 사전 학습된 교사 |

두 경우 모두 **학생이 교사에 못 미치지만 파라미터를 10배 이상 줄인다**.
CPU 실시간이 목표면 이 구조가 사실상 표준 답이다.

---

## 6. Pierrot_FR 에 들일 때 확인할 것

아직 구현하지 않았다. 착수 시 결정해야 하는 항목만 적어 둔다.

1. **crop 규약** — face crop 인지 head crop 인지. 하류(FAS·인식)의 정렬 규약과 맞춰야 한다.
   [FAS 의 crop 맥락 문제](../FAS/MiniFASNet.md#6--crop-맥락--공정성-문제) 와 같은 종류의 함정이다.
2. **랜드마크 규약** — 68 로 갈지 98 로 갈지. flip mapping 을 데이터(JSON)로 둘 것.
3. **NME 정규화 기준** — 하나로 고정하고 문서에 못 박는다. 여러 기준을 섞어 보고하지 않는다.
4. **누수** — 테스트 split 을 학습에 넣을지. 넣는다면 **누수 없는 계기를 따로 확보**한다
   (HRFFA 의 `300wlp_val` 에 해당하는 것).
5. **배포 형식** — ONNX 로 뺄 거면 트랜스포머 디코더의 배치축 문제를 처음부터 고려한다
   ([HRFFA.md §6](HRFFA.md#6-onnx-export-가-1급-시민)).
