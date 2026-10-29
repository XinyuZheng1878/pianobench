"""Soft per-finger evidence at each MIDI onset (At-Your-Fingertips style), plus the press cue."""
from __future__ import annotations

import warnings

import numpy as np

from .keys import IS_BLACK, KEY_X


def onset_frames(times, fps: float, av: int) -> np.ndarray:
    return np.rint(np.asarray(times) * fps).astype(int) + int(av)


def _gather(arr: np.ndarray, frames: np.ndarray):
    """arr (T, ...) sampled at integer frames (N, K) -> (N, K, ...), plus in-bounds mask (N, K)."""
    T = arr.shape[0]
    inb = (frames >= 0) & (frames < T)
    return arr[np.clip(frames, 0, T - 1)], inb


def estimate_av_offset(tips, ok, pitch, onset_s, fps, search: int) -> int:
    """Shift (frames) maximising the fraction of onsets with some fingertip within 0.5 key of the key."""
    kx = KEY_X[pitch][:, None, None]
    best, best_rate = 0, -1.0
    for s in sorted(range(-search, search + 1), key=abs):
        f = onset_frames(onset_s, fps, s)[:, None]
        x, inb = _gather(tips[..., 0], f)
        v, _ = _gather(ok, f)
        d = np.abs(x[:, 0] - kx)
        d = np.where((v[:, 0] & inb)[..., None], d, np.inf)
        dmin = np.nan_to_num(d, nan=np.inf).min((1, 2))
        has = np.isfinite(dmin)
        if not has.any():
            continue
        rate = float((dmin[has] < 0.5).mean())
        if rate > best_rate:
            best, best_rate = s, rate
    return best


def _masked_mean(arr, ok, frames):
    a, inb = _gather(arr, frames)
    v, _ = _gather(ok, frames)
    m = (v & inb[..., None])[..., None] & np.isfinite(a)
    cnt = m.sum(1)
    return np.where(cnt > 0, np.where(m, a, 0.0).sum(1) / np.maximum(cnt, 1), np.nan)


def press_bonus(tips, ok, onset_f, fps, cfg):
    """Log bonus for fingers that moved down into the onset more than their hand-mates (needs z)."""
    z = tips[..., 2]
    if cfg.press_beta <= 0 or not np.isfinite(z).any():
        return 0.0
    b0, b1 = max(2, int(round(0.2 * fps))), max(1, int(round(0.07 * fps)))
    before = onset_f[:, None] + np.arange(-b0, -b1 + 1)[None]
    at = onset_f[:, None] + np.arange(-1, max(2, b1) + 1)[None]
    dz = _masked_mean(z, ok, before) - _masked_mean(z, ok, at)  # > 0: finger went down
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dz = dz - np.nanmedian(dz, axis=2, keepdims=True)
    return cfg.press_beta * np.clip(np.nan_to_num(dz) / cfg.press_scale_mm, -1.0, 1.0)


def finger_evidence(tips, ok, pitch, onset_f, fps, bias, cfg, return_wfrac=False):
    """
    Returns
      logE  (N, 2, 5)  log evidence that finger (h, f) pressed note n
      s     (N, 2, 5)  raw weighted Gaussian score in [0, 1]
      avail (N, 2)     enough valid frames for hand h around the onset
      resid (N, 2, 5)  weighted mean (tip_x - key_x) without bias, for bias estimation
    """
    pre = max(1, int(round(cfg.win_pre_s * fps)))
    post = max(0, int(round(cfg.win_post_s * fps)))
    offs = np.arange(-pre, post + 1)
    center = cfg.win_center_s * fps
    w = np.exp(-np.abs(offs - center) / max(cfg.win_tau_s * fps, 1e-6))
    frames = onset_f[:, None] + offs[None]
    x, inb = _gather(tips[..., 0], frames)                      # (N, K, 2, 5)
    v, _ = _gather(ok, frames)                                  # (N, K, 2)
    v = v & inb[..., None]
    raw = x - KEY_X[pitch][:, None, None, None]
    sigma = np.where(IS_BLACK[pitch], cfg.sigma_black, cfg.sigma_white)[:, None, None, None]
    if cfg.kernel == "tri":
        g = np.clip(1.0 - np.abs(raw - bias) / (1.25 * sigma), 0.0, None)
    else:
        g = np.exp(-0.5 * ((raw - bias) / sigma) ** 2)
    wv = w[None, :, None] * v                                   # (N, K, 2)
    wsum = wv.sum(1)
    den = np.maximum(wsum, 1e-9)[..., None]
    s = np.nansum(g * wv[..., None], 1) / den
    resid = np.nansum(np.nan_to_num(raw) * wv[..., None], 1) / den
    avail = wsum >= cfg.min_valid_frac * w.sum()
    s = np.where(avail[..., None], s, 0.0)
    logE = np.log(s + cfg.floor) + press_bonus(tips, ok, onset_f, fps, cfg)
    if return_wfrac:
        return logE, s, avail, resid, wsum / w.sum()
    return logE, s, avail, resid
