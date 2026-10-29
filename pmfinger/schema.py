"""Input contract.

One `.npz` per video, exported by the existing MANO-FK + keyboard-fitting code (PM10M) or the
MediaPipe + homography code (PianoVAM). Fields:

  video_id      str
  fps           float
  tips          (T, 2, 5, 3) float  hand 0 = left, 1 = right; finger 0 = thumb ... 4 = little
                  [..., 0] x along the keyboard in white-key units, 0 = left edge of A0 (see keys.py)
                  [..., 1] y across the keys (optional, NaN ok; unused for now)
                  [..., 2] fingertip height in mm, larger = higher, e.g. depth relative to the wrist
                           from joint angles only (optional, NaN ok)
  valid         (T, 2) bool         per-hand valid flag (PM10M's own flag)
  note_onset    (N,) seconds        MIDI time
  note_offset   (N,) seconds
  note_pitch    (N,) int            MIDI pitch 21..108
  note_velocity (N,) optional
  label_hand    (N,) optional int   0/1, -1 unknown     (PianoVAM GT / auto labels)
  label_finger  (N,) optional int   0..4, -1 unknown
  wrist         (T, 2, 3) optional
  frame_offset  int optional        video frame of row 0 (video frame = frame_offset + row)
  video_path    str optional

Left hand must already be un-mirrored into the same keyboard frame as the right hand, and the
PianoVAM MediaPipe left/right swap must already be fixed.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .keys import PITCH_MAX, PITCH_MIN

PROTECTED_DIR = "/inspire/hdd/global_user/xuguangjun-253208580299/CapWAM"


def check_output_path(path: str) -> None:
    """Refuse to write anywhere under the shared read-only directory."""
    p = os.path.realpath(os.path.abspath(path))
    if p.startswith(os.path.realpath(PROTECTED_DIR)):
        raise SystemExit(f"refusing to write into read-only shared dir: {p}")


@dataclass
class Performance:
    video_id: str
    fps: float
    tips: np.ndarray
    valid: np.ndarray
    onset: np.ndarray
    offset: np.ndarray
    pitch: np.ndarray
    velocity: Optional[np.ndarray] = None
    label_hand: Optional[np.ndarray] = None
    label_finger: Optional[np.ndarray] = None
    wrist: Optional[np.ndarray] = None
    frame_offset: int = 0
    video_path: str = ""

    def __post_init__(self):
        self.fps = float(self.fps)
        self.tips = np.asarray(self.tips, dtype=np.float64)
        self.valid = np.asarray(self.valid, dtype=bool)
        assert self.tips.ndim == 4 and self.tips.shape[1:] == (2, 5, 3), self.tips.shape
        assert self.valid.shape == self.tips.shape[:2], self.valid.shape
        self.onset = np.asarray(self.onset, dtype=np.float64)
        self.offset = np.asarray(self.offset, dtype=np.float64)
        self.pitch = np.asarray(self.pitch).astype(int)
        keep = (self.pitch >= PITCH_MIN) & (self.pitch <= PITCH_MAX)
        if not keep.all():
            warnings.warn(f"{self.video_id}: dropping {int((~keep).sum())} notes outside A0..C8")
        order = np.lexsort((self.pitch[keep], self.onset[keep]))
        idx = np.flatnonzero(keep)[order]
        self.onset, self.offset, self.pitch = self.onset[idx], self.offset[idx], self.pitch[idx]
        for name in ("velocity", "label_hand", "label_finger"):
            v = getattr(self, name)
            if v is not None:
                v = np.asarray(v)[idx]
                setattr(self, name, v.astype(int) if name.startswith("label") else v)

    @property
    def T(self) -> int:
        return self.tips.shape[0]

    @property
    def N(self) -> int:
        return len(self.pitch)


def load_performance(path: str) -> Performance:
    z = np.load(path, allow_pickle=False)

    def get(k):
        return z[k] if k in z.files else None

    return Performance(
        video_id=str(z["video_id"]),
        fps=float(z["fps"]),
        tips=z["tips"],
        valid=z["valid"],
        onset=z["note_onset"],
        offset=z["note_offset"],
        pitch=z["note_pitch"],
        velocity=get("note_velocity"),
        label_hand=get("label_hand"),
        label_finger=get("label_finger"),
        wrist=get("wrist"),
        frame_offset=int(z["frame_offset"]) if "frame_offset" in z.files else 0,
        video_path=str(z["video_path"]) if "video_path" in z.files else "",
    )


def save_performance(perf: Performance, path: str) -> None:
    check_output_path(path)
    d = dict(
        video_id=np.array(perf.video_id), fps=np.array(perf.fps), tips=perf.tips.astype(np.float32),
        valid=perf.valid, note_onset=perf.onset, note_offset=perf.offset, note_pitch=perf.pitch,
        frame_offset=np.array(perf.frame_offset), video_path=np.array(perf.video_path),
    )
    for k, v in (("note_velocity", perf.velocity), ("label_hand", perf.label_hand),
                 ("label_finger", perf.label_finger), ("wrist", perf.wrist)):
        if v is not None:
            d[k] = v
    np.savez_compressed(path, **d)
