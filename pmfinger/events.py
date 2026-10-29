from __future__ import annotations

import numpy as np

from .keys import KEY_X


def norm_x(pitch: np.ndarray, h: int) -> np.ndarray:
    """Key x in a hand-normalised frame where finger index increases with x for both hands."""
    x = KEY_X[np.asarray(pitch)]
    return x if h == 1 else -x


def group_events(onset: np.ndarray, idx, tol: float):
    """Group note indices (sorted by onset) into chord events: onsets within `tol` of the event's first note."""
    events = []
    for n in idx:
        if events and onset[n] - onset[events[-1][0]] <= tol:
            events[-1].append(int(n))
        else:
            events.append([int(n)])
    return events
