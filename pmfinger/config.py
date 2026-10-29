from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    # --- track cleaning ---
    jump_thresh: float = 1.5          # white keys; deviation from 5-frame median that marks a glitch frame
    max_gap_frames: int = 6           # invalid runs up to this length are interpolated
    smooth_window: int = 7            # Savitzky-Golay window (frames); <=1 disables
    smooth_order: int = 2
    # --- audio/video alignment ---
    av_offset_frames: Optional[int] = None   # video_row = round(onset*fps) + av; None -> search per video
    av_search: int = 8
    # --- soft evidence at onset ---
    sigma_white: float = 0.5          # Gaussian width (white keys) around the key centre
    sigma_black: float = 0.4
    win_pre_s: float = 0.13           # evidence window around the onset
    win_post_s: float = 0.07
    win_tau_s: float = 0.05           # exponential decay of frame weights away from the window centre
    win_center_s: float = 0.0         # centre of the frame weighting relative to the onset (>0: after the press)
    kernel: str = "gauss"             # "gauss" or "tri" (linear kernel, half-width = 1.25 * sigma)
    min_valid_frac: float = 0.3       # fraction of window weight that must be on valid frames
    floor: float = 1e-3               # likelihood floor so one bad frame never gives -inf
    press_beta: float = 0.15          # weight of the "finger went down" cue (tie-breaker only)
    press_scale_mm: float = 5.0
    # --- sequence decoding ---
    chord_tol_s: float = 0.04         # onsets closer than this (same hand) form one chord event
    w_seq: float = 1.0                # weight of the learned transition prior
    w_chord: float = 1.0              # weight of the in-chord finger-span prior
    gap_tau_s: float = 1.0            # transition prior fades to uniform over long gaps
    occupancy_penalty: float = 1.0    # nats; for using a finger still holding an earlier key
    occupancy_passes: int = 2
    # --- per-video finger bias (EM) ---
    bias_em_iters: int = 1
    bias_shrink_n: float = 30.0       # pseudo-count pulling the per-video bias to the prior bias
    bias_min_conf: float = 0.9
    # --- release tracking ---
    release_lateral: float = 0.7      # white keys
    release_lift_mm: float = 15.0
    release_consec: int = 3
    # --- output ---
    conf_thresh: float = 0.8          # below -> "low_conf" (no label)
    fake_thresh: float = 0.02         # max evidence below this -> "suspect" (likely transcription ghost)


def apply_overrides(cfg: Config, items) -> Config:
    """Apply ["key=value", ...] overrides, parsing values by the field's current type."""
    for it in items or []:
        k, v = it.split("=", 1)
        k = k.strip()
        if not hasattr(cfg, k):
            raise KeyError(f"unknown config key: {k}")
        cur = getattr(cfg, k)
        if v.strip().lower() == "none":
            val = None
        elif isinstance(cur, bool):
            val = v.strip().lower() in ("1", "true", "yes")
        elif isinstance(cur, int) or cur is None:
            val = int(v)
        elif isinstance(cur, float):
            val = float(v)
        else:
            val = v
        setattr(cfg, k, val)
    return cfg
