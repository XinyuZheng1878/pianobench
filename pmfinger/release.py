"""True key release: follow the assigned finger from the onset until it leaves the key.

MIDI offsets in PM10M come from audio transcription and include pedal sustain, so they only give
an upper bound. The finger is released when it moves sideways > release_lateral white keys or lifts
> release_lift_mm for release_consec consecutive frames, or when it starts its next note.
"""
from __future__ import annotations

import numpy as np


def track_release(tips, ok, onset_f, offset_f, hand, finger, cfg):
    """Returns (release_frame (N,) exclusive end of the hold, -1 if unassigned; uncertain (N,) bool)."""
    T = ok.shape[0]
    N = len(onset_f)
    rel = np.full(N, -1, dtype=int)
    unc = np.zeros(N, dtype=bool)
    nxt = np.full(N, T, dtype=int)
    for h in (0, 1):
        for f in range(5):
            idx = np.flatnonzero((hand == h) & (finger == f))
            idx = idx[np.argsort(onset_f[idx], kind="stable")]
            nxt[idx[:-1]] = onset_f[idx[1:]]

    x_all, z_all = tips[..., 0], tips[..., 2]
    for n in np.flatnonzero((hand >= 0) & (finger >= 0)):
        h, f, c = int(hand[n]), int(finger[n]), int(onset_f[n])
        if c < 0 or c >= T:
            rel[n] = int(np.clip(offset_f[n], 0, T))
            unc[n] = True
            continue
        lim = int(min(offset_f[n], nxt[n] - 1, T - 1))
        if lim <= c:
            rel[n] = c + 1
            continue
        ref = [k for k in range(c, min(c + 2, lim) + 1) if ok[k, h]]
        if not ref:
            rel[n] = lim + 1
            unc[n] = True
            continue
        x0 = float(np.mean(x_all[ref, h, f]))
        zr = z_all[ref, h, f]
        z0 = float(np.mean(zr)) if np.isfinite(zr).all() else np.nan
        run = gap = 0
        found = False
        for k in range(c + 1, lim + 1):
            if not ok[k, h]:
                gap += 1
                run = 0
                if gap > cfg.max_gap_frames:
                    rel[n] = k - gap + 1
                    unc[n] = found = True
                    break
                continue
            gap = 0
            lifted = np.isfinite(z0) and np.isfinite(z_all[k, h, f]) and z_all[k, h, f] - z0 > cfg.release_lift_mm
            if abs(x_all[k, h, f] - x0) > cfg.release_lateral or lifted:
                run += 1
                if run >= cfg.release_consec:
                    rel[n] = k - cfg.release_consec + 1
                    found = True
                    break
            else:
                run = 0
        if not found:
            rel[n] = lim + 1
    return rel, unc


def make_busy(hand, finger, onset_f, rel_f, tol_f):
    """busy(h, frame) -> (5,) bool: finger still holding a key that started before this event."""
    table = {}
    for h in (0, 1):
        for f in range(5):
            m = (hand == h) & (finger == f) & (rel_f >= 0)
            o = np.argsort(onset_f[m], kind="stable")
            table[h, f] = (onset_f[m][o], rel_f[m][o])

    def busy(h, c):
        out = np.zeros(5, dtype=bool)
        for f in range(5):
            on, r = table[h, f]
            i = np.searchsorted(on, c - tol_f, side="left") - 1   # last note of f starting before this event
            out[f] = i >= 0 and r[i] > c + 1
        return out

    return busy
