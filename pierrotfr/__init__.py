"""Pierrot_FR — 얼굴 알고리즘 **추론 배포본**.

학습 저장소(Pierrot_FR_Lab)에서 추론 경로만 떼어 낸 것이다. 태스크별로 서브패키지를
나누고, 태스크에 종속되지 않는 코드만 최상위에 둔다.

    pierrotfr/utils.py      파라미터 수 세기 등 소품
    pierrotfr/FA3D/         3D Dense Face Alignment (3DDFA_V2 재구현)

각 태스크 서브패키지의 계약:

    infer.py      체크포인트 -> 모델 -> 예측
    metrics.py    태스크 지표
    models/       백본 + build_model 레지스트리 (학습 전용 헤드 없이)

⚠ 학습 저장소에 있던 optim.py · engine.py · losses.py · 학습 데이터셋은 여기 없다.
  같은 이름의 파일이 두 저장소에 있으면 **추론에 쓰이는 부분만** 옮겨져 있다.
"""

__version__ = "0.1.0"
