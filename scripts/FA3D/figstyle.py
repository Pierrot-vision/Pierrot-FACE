# coding: utf-8
"""문서 그림의 공통 스타일 — 한글 폰트 · 저장 규약.

학습 저장소에서는 스크립트마다 이 여섯 줄이 복사돼 있었다. 폰트 경로가 없는
서버에서 **그림 스크립트가 전부 죽는 것**이 문제라 한 곳으로 모은다 — 여기서는
폰트가 없으면 조용히 기본 폰트로 내려간다 (한글 제목이 네모로 나오지만, 그림은
나온다).

📐 **문서 그림은 전부 JPG 로 저장한다** — 예측 격자처럼 큰 그림은 PNG 대비 약 1/4
   용량이다 (학습 저장소 실측: docs/ 21.2MB → 5.2MB).
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm      # noqa: E402
import matplotlib.pyplot as plt           # noqa: E402

# 서버마다 다르다 — 없으면 넘어간다
_FONTS = [
    "/usr/share/fonts/truetype/nanum/NanumSquareRoundB.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
]

JPG = {"quality": 90, "optimize": True, "progressive": True}

_family = None
for _fp in _FONTS:
    if os.path.isfile(_fp):
        fm.fontManager.addfont(_fp)
        _family = fm.FontProperties(fname=_fp).get_name()
        break

plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight",
                     "axes.unicode_minus": False})
if _family:
    plt.rcParams["font.family"] = _family
else:
    print("[FA3D] 한글 폰트를 찾지 못했습니다 — 제목이 깨져 보일 수 있습니다 "
          "(그림 자체는 정상입니다).")


def save(fig, path: str, **pil_kwargs) -> str:
    """디렉토리를 만들고 저장한 뒤 경로를 찍는다 — 모든 그림 스크립트의 마지막 줄."""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, pil_kwargs=pil_kwargs or None)
    print(f"저장: {path}")
    return path
