"""Clean fingertip tracks: drop glitch frames, interpolate short gaps, smooth valid runs."""
from __future__ import annotations

import warnings

import numpy as np
from scipy.signal import savgol_filter


def runs(mask: np.ndarray):
    """(starts, stops) of True runs in a 1-D bool array; stop is exclusive."""
    m = np.concatenate([[0], np.asarray(mask, dtype=np.int8), [0]])
    d = np.diff(m)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def _nan_median_filter(x: np.ndarray, k: int) -> np.ndarray:
    """Centred running nan-median along axis 0 of a (T, C) array."""
    h = k // 2
    pad = np.full((h,) + x.shape[1:], np.nan)
    win = np.lib.stride_tricks.sliding_window_view(np.concatenate([pad, x, pad], 0), k, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(win, axis=-1)


def clean_tips(tips: np.ndarray, valid: np.ndarray, cfg):
    """Returns (tips, ok): tips with NaN on unusable frames, ok (T, 2) bool."""
    tips = np.array(tips, dtype=np.float64, copy=True)
    T = tips.shape[0]
    ok = np.asarray(valid, dtype=bool) & np.isfinite(tips[..., 0]).all(-1)
    tips[~ok] = np.nan

    # isolated jumps (MANO glitches): any fingertip far from its 5-frame running median
    for h in range(2):
        x = tips[:, h, :, 0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            dev = np.nanmax(np.abs(x - _nan_median_filter(x, 5)), axis=1)
        ok[np.nan_to_num(dev, nan=0.0) > cfg.jump_thresh, h] = False
    tips[~ok] = np.nan

    for h in range(2):
        # interpolate short invalid runs bounded by valid frames; long runs stay invalid
        for a, b in zip(*runs(~ok[:, h])):
            if a == 0 or b == T or b - a > cfg.max_gap_frames:
                continue
            lo, hi = tips[a - 1, h].copy(), tips[b, h].copy()
            for k in range(a, b):
                al = (k - a + 1) / (b - a + 1)
                tips[k, h] = (1 - al) * lo + al * hi
            ok[a:b, h] = True
        # smooth each valid run; only fully finite columns (z may be all NaN)
        if cfg.smooth_window > 1:
            for a, b in zip(*runs(ok[:, h])):
                if b - a < cfg.smooth_window:
                    continue
                seg = tips[a:b, h].reshape(b - a, -1)
                cols = np.isfinite(seg).all(0)
                if cols.any():
                    seg[:, cols] = savgol_filter(seg[:, cols], cfg.smooth_window, cfg.smooth_order, axis=0)
                tips[a:b, h] = seg.reshape(b - a, 5, 3)
    return tips, ok
