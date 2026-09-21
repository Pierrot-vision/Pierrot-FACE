"""지시문 템플릿 — 추론에 쓰는 고정 질문만.

학습에서는 content(위조 유형) · style(조명 / 환경 / 카메라 품질) 질문과 정답을
FLAN-T5 로 감독했다. 추론 경로는 LLM 을 타지 않으므로 **질문 문장**만 필요하다 —
Q-Former 가 이 문장을 읽고 무엇을 볼지 게이팅한다. 학습 저장소의 평가와 같게
content 는 CONTENT_KEYS[0], style 은 STYLE_KEYS[0] 을 쓴다.
"""
from __future__ import annotations

SPOOF_TYPES = [
    "(1) Real face",
    "(2) Photo",
    "(3) Poster",
    "(4) A4-paper",
    "(5) 2D face mask",
    "(6) 2D upper-body mask",
    "(7) 2D region mask",
    "(8) PC screen",
    "(9) Pad screen",
    "(10) Phone screen",
    "(11) 3D mask",
]

ILLUMINATION = ["(1) Normal", "(2) Strong", "(3) Back", "(4) Dark"]
ENVIRONMENT = ["(1) Indoor", "(2) Outdoor"]
CAMERA_QUALITY = ["(1) Low", "(2) Medium", "(3) High"]

_STEM = "Choose the correct option to the following question: "

QUESTIONS = {
    "presentation attack": _STEM + "Which type of presentation attack is in this image?\n"
                           + " ".join(SPOOF_TYPES),
    "illumination": _STEM + "What is the illumination condition in this image?\n"
                    + " ".join(ILLUMINATION),
    "environment": _STEM + "What is the environment in this image?\n"
                   + " ".join(ENVIRONMENT),
    "camera quality": _STEM + "What is the camera quality in this image?\n"
                      + " ".join(CAMERA_QUALITY),
}

CONTENT_KEYS = ("presentation attack",)
STYLE_KEYS = ("illumination", "environment", "camera quality")


def question(key: str) -> str:
    return QUESTIONS[key]
