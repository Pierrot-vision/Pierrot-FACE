"""얼굴 검출 — 크롭을 잡기 위한 것뿐이다.

평가(AFLW2000-3D / AFLW)는 **사전 크롭된 120x120** 을 쓰므로 검출기를 타지 않는다.
검출기가 필요한 곳은 임의의 사진·영상에 돌리는 데모뿐이고, 그래서 여기서 검출기를
갈아 끼울 수 있게 둔다. 지표는 검출기 선택에 영향을 받지 않는다.

우선순위 (앞에서부터 되는 것을 쓴다):

    ① FaceBoxes    3DDFA_V2 오피셜이 쓰는 검출기. PIERROTFR_3DDFA_V2_ROOT 가
                   가리키는 저장소에서 가져온다 — 오피셜 데모와 **같은 크롭**이 된다
    ② MTCNN        facenet-pytorch. pip 로 들어오고 정확도도 충분하다
    ③ Haar cascade OpenCV 동봉. 추가 설치 없이 무조건 되는 최후의 수단 —
                   정면만 잡고 옆얼굴은 자주 놓친다

⚠ **검출기를 바꾸면 크롭이 조금 달라지고 예측도 조금 달라진다.** 논문 수치와
  비교할 값은 언제나 사전 크롭 평가셋(eval/FA3D/evaluate.py)에서 나온 것이다.
  데모 그림을 두 검출기로 만들어 나란히 놓지 말 것.

⚠ FaceBoxes 는 Cython 으로 빌드한 `cpu_nms` 를 요구한다. 외부 저장소에 빌드 산출물을
  들이지 않으려고, 같은 인터페이스의 numpy 구현을 import 전에 주입한다.
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np


def _nms(dets: np.ndarray, thresh: float) -> list:
    """FaceBoxes 가 기대하는 `cpu_nms` 의 numpy 대체 구현."""
    x1, y1, x2, y2, sc = (dets[:, i] for i in range(5))
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = sc.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][ovr <= thresh]
    return keep


# ------------------------------------------------------------------ #
def _try_faceboxes():
    from configs.paths import V2_ROOT
    if not V2_ROOT:
        return None, "PIERROTFR_3DDFA_V2_ROOT 미설정"
    if not os.path.isdir(V2_ROOT):
        return None, f"경로 없음 ({V2_ROOT})"
    if V2_ROOT not in sys.path:
        sys.path.insert(0, V2_ROOT)
    shim = types.ModuleType("FaceBoxes.utils.nms.cpu_nms")
    shim.cpu_nms = _nms
    shim.cpu_soft_nms = lambda dets, *a, **k: _nms(dets, k.get("Nt", 0.3))
    sys.modules.setdefault("FaceBoxes.utils.nms.cpu_nms", shim)
    try:
        from FaceBoxes import FaceBoxes
    except Exception as e:                                    # noqa: BLE001
        return None, f"import 실패 ({e})"
    det = FaceBoxes()
    return (lambda img: [list(map(float, b[:4])) for b in det(img)]), "FaceBoxes (오피셜)"


def _try_mtcnn():
    try:
        import torch
        from facenet_pytorch import MTCNN
    except Exception as e:                                    # noqa: BLE001
        return None, f"facenet-pytorch 없음 ({e})"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # keep_all — 화면에 여러 명이 나오는 장면이 있다
    det = MTCNN(keep_all=True, device=dev, post_process=False)

    def run(img):
        import cv2
        boxes, _ = det.detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return [] if boxes is None else [list(map(float, b[:4])) for b in boxes]

    return run, f"MTCNN (facenet-pytorch, {dev})"


def _try_haar():
    import cv2
    fp = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    if not os.path.isfile(fp):
        return None, "OpenCV haarcascade 파일 없음"
    casc = cv2.CascadeClassifier(fp)

    def run(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found = casc.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
        return [[float(x), float(y), float(x + w), float(y + h)] for x, y, w, h in found]

    return run, "Haar cascade (정면만 — 옆얼굴을 자주 놓칩니다)"


_BACKENDS = {"faceboxes": _try_faceboxes, "mtcnn": _try_mtcnn, "haar": _try_haar}
_ORDER = ("faceboxes", "mtcnn", "haar")


class FaceDetector:
    """BGR 이미지 -> [[left, top, right, bottom], …].

    `backend` 를 주면 그것만 쓰고, 안 되면 왜 안 되는지 말하고 멈춘다. 기본값은
    폴백 순회다 — 데모 하나 돌리려고 외부 저장소를 clone 해야 하는 상황을 피한다.
    """

    def __init__(self, backend: str = "", verbose: bool = True):
        if backend:
            if backend not in _BACKENDS:
                raise SystemExit(f"[FA3D] backend={backend!r} — 가능: {sorted(_BACKENDS)}")
            fn, why = _BACKENDS[backend]()
            if fn is None:
                raise SystemExit(f"[FA3D] 검출기 {backend} 를 쓸 수 없습니다: {why}")
            self.run, self.name = fn, why
        else:
            tried = []
            for key in _ORDER:
                fn, why = _BACKENDS[key]()
                if fn is not None:
                    self.run, self.name = fn, why
                    break
                tried.append(f"{key}: {why}")
            else:
                raise SystemExit("[FA3D] 쓸 수 있는 얼굴 검출기가 없습니다:\n"
                                 + "\n".join(f"    {t}" for t in tried)
                                 + "\n  pip install facenet-pytorch 를 권합니다.")
            if verbose and tried:
                print(f"[FA3D] 검출기 {self.name} "
                      f"(건너뜀 — {' / '.join(tried)})")
            elif verbose:
                print(f"[FA3D] 검출기 {self.name}")

    def __call__(self, img: np.ndarray) -> list:
        return self.run(img)
