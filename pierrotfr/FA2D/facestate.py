"""얼굴 상태(눈감김·입벌림) 헤드가 풀링할 점 인덱스 — 추론 경로만.

상태 헤드는 전역 풀링이 아니라 **해당 부위 토큰만** 모아 읽는다(눈감김은 국소 성질이라
전역 풀링하면 묻힌다). 인덱스는 각 규약의 공식 배포 사양(눈/입 edge 정의)에서 온다.

학습 저장소의 같은 파일에서 라벨 유도(`derive_state_labels`)와 손실(`state_loss`)을 뺐다.
"""
from __future__ import annotations

# 상태별 (가로 폭을 재는 양끝, 세로 열림을 재는 상/하 쌍들)
STATE_POINTS = {
    "ibug68": {"right_eye": ((36, 39), ((37, 41), (38, 40))),
               "left_eye": ((42, 45), ((43, 47), (44, 46))),
               "mouth": ((60, 64), ((61, 67), (62, 66), (63, 65)))},
    "wflw98": {"right_eye": ((60, 64), ((61, 67), (62, 66), (63, 65))),
               "left_eye": ((68, 72), ((69, 75), (70, 74), (71, 73))),
               "mouth": ((88, 92), ((89, 95), (90, 94), (91, 93)))},
    "cofw29": {"right_eye": ((8, 10), ((12, 13),)),
               "left_eye": ((9, 11), ((14, 15),)),
               "mouth": ((22, 23), ((26, 27), (24, 25)))},
}

STATE_NAMES = ("right_eye_closed", "left_eye_closed", "mouth_open", "mouth_open_wide")


def state_token_index(scheme: str):
    """상태별로 풀링할 점 인덱스."""
    spec = STATE_POINTS[scheme]

    def flat(name):
        (a, b), pairs = spec[name]
        return sorted({a, b, *(i for p in pairs for i in p)})

    m = flat("mouth")
    return [flat("right_eye"), flat("left_eye"), m, m]
