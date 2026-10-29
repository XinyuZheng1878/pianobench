"""Piano key geometry in white-key units.

x = 0 is the left edge of A0; white key i spans [i, i+1), so its centre is i + 0.5.
Black-key centres sit on the boundary between their two white neighbours, nudged by
BLACK_NUDGE so that black-key spacing matches the measured ~1.16 (inside a 2/3 group)
and ~1.76-1.80 (between groups) white-key widths.
"""
from __future__ import annotations

import numpy as np

PITCH_MIN, PITCH_MAX = 21, 108
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_WHITE_IN_OCT = {0: 0, 2: 1, 4: 2, 5: 3, 7: 4, 9: 5, 11: 6}
BLACK_NUDGE = {1: -0.08, 3: 0.08, 6: -0.12, 8: 0.0, 10: 0.12}
_A0_WHITE = 1 * 7 + 5  # white keys from C-1 (pitch 0) up to A0


def is_black(p: int) -> bool:
    return (int(p) % 12) in BLACK_NUDGE


def white_index(p: int) -> int:
    """Index of white key p (or of the white key just below black key p); A0 = 0."""
    p = int(p)
    pc = p % 12
    if pc in BLACK_NUDGE:
        pc -= 1
    return (p // 12) * 7 + _WHITE_IN_OCT[pc] - _A0_WHITE


def key_x(p: int) -> float:
    if is_black(p):
        return white_index(p) + 1.0 + BLACK_NUDGE[int(p) % 12]
    return white_index(p) + 0.5


def pitch_name(p: int) -> str:
    p = int(p)
    return f"{NAMES[p % 12]}{p // 12 - 1}"


KEY_X = np.full(128, np.nan)
IS_BLACK = np.zeros(128, dtype=bool)
for _p in range(PITCH_MIN, PITCH_MAX + 1):
    KEY_X[_p] = key_x(_p)
    IS_BLACK[_p] = is_black(_p)
WHITE_PITCHES = np.array([p for p in range(PITCH_MIN, PITCH_MAX + 1) if not is_black(p)])  # 52


def white_pitch_at(x: float) -> int:
    """White key under keyboard coordinate x (clipped to the 88-key range)."""
    return int(WHITE_PITCHES[int(np.clip(np.floor(x), 0, len(WHITE_PITCHES) - 1))])
