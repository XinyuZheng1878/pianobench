"""P0 raw-state adapter. Ported spec rules; no smoothed/future-derived input features."""
from dataclasses import dataclass
from pathlib import Path
import importlib.util
import numpy as np
_spec = importlib.util.spec_from_file_location("_p0_keys", Path(__file__).with_name("keys.py"))
_keys = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_keys)
IS_BLACK, KEY_X = _keys.IS_BLACK, _keys.KEY_X
pitch_name, white_pitch_at = _keys.pitch_name, _keys.white_pitch_at
HAND_ZH = ["左手", "右手"]
FINGER_ZH = ["拇指", "食指", "中指", "无名指", "小指"]
@dataclass
class Perf:
    video_id: str
    fps: float
    T: int
    video_path: str
    pitch: np.ndarray        # (N,)
    on: np.ndarray           # (N,) video frame of key-down
    off: np.ndarray          # (N,) video frame of key-up (exclusive end of the hold)
    hand: np.ndarray         # (N,) -1 unknown
    hand_src: np.ndarray     # (N,) "gold" / "v2" / ""
    finger: np.ndarray       # (N,) finger_at_onset 0..4, -1 unknown (human GT only)
    palm: np.ndarray         # (T, 2) palm-centre x, NaN when the hand is not detected
    tips: np.ndarray         # (T, 2, 5) fingertip x
    ok: np.ndarray           # (T, 2)
    on_s: np.ndarray         # (N,) MIDI onset seconds (grouping uses continuous time, not rounded frames)

    # --------------------------------------------------------------- geometry
    def palm_med(self, h, t, half):
        a, b = max(0, t - half), min(self.T, t + half + 1)
        v = self.palm[a:b, h]
        return float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan

    def palm_past(self, h, t, n):
        """Causal palm estimate at t: median of the n frames up to and including t."""
        v = self.palm[max(0, t - n + 1):t + 1, h]
        return float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan

    def span(self, h, t):
        """Thumb tip to little-finger tip, horizontal distance in white keys (what the question asks)."""
        x = self.tips[t, h]
        return float(abs(x[4] - x[0])) if np.isfinite(x[[0, 4]]).all() else np.nan

    def pressed(self, pitch, f):
        """Is key `pitch` down at video frame f (any note of that key)?"""
        return bool(((self.pitch == pitch) & (self.on <= f) & (self.off > f)).any())

    def sounding_in(self, a, b):
        """Notes held at any time in [a, b) (including ones pressed before a)."""
        return np.flatnonzero((self.on < b) & (self.off > a))

    def visible(self, h, a, b, frac):
        a, b = max(0, a), min(self.T, b)
        return b > a and self.ok[a:b, h].mean() >= frac

    @staticmethod
    def key_name_at(x):
        return pitch_name(white_pitch_at(x))

    # --------------------------------------------------------------- notes
    def notes(self, a, b, h=None):
        m = (self.on >= a) & (self.on < b)
        if h is not None:
            m &= self.hand == h
        return np.flatnonzero(m)

    def hand_unknown_in(self, a, b):
        """Notes pressed or held in [a, b) whose hand is unknown (could belong to either hand)."""
        return np.flatnonzero((self.on < b) & (self.off > a) & (self.hand < 0))

    def groups(self, idx, tol_s):
        """Onset groups: a group starts at the earliest unassigned onset and takes onsets within `tol_s`
        seconds of that first onset (no chaining), measured on the continuous MIDI times."""
        selected = set(int(n) for n in idx)
        hands = {int(self.hand[n]) for n in selected}
        full = sorted(np.flatnonzero(np.isin(self.hand, list(hands))),
                      key=lambda n: (self.on_s[n], self.pitch[n]))
        groups = []
        for n in full:
            if groups and self.on_s[n] - self.on_s[groups[-1][0]] <= tol_s + 1e-9:
                groups[-1].append(int(n))
            else:
                groups.append([int(n)])
        return [[n for n in group if n in selected] for group in groups
                if any(n in selected for n in group)]


def trust_sources(z, r):
    """Hand GT is independent of finger GT; missing auto never corroborates a hand."""
    n = len(r["pitch"])
    gh = np.asarray(z.get("gt_hand", np.full(n, -1)), int)
    gf = np.asarray(z.get("gt_finger", np.full(n, -1)), int)
    ah = np.asarray(z.get("label_hand", np.full(n, -1)), int)
    hp = np.asarray(r["hand_pred"], int)
    gold_hand = np.isin(gh, [0, 1])
    agree = np.isin(hp, [0, 1]) & (hp == ah) & (np.asarray(r["hand_conf"]) >= .9)
    hand = np.where(gold_hand, gh, np.where(agree, hp, -1))
    # A finger number without its annotated hand is not an identifiable (hand,finger).
    gold_finger = gold_hand & (gf >= 0) & (gf < 5)
    finger = np.where(gold_finger, gf, -1)
    source = np.where(gold_hand, "gold", np.where(agree, "v2_auto_agreement", "unknown"))
    return hand, source, finger


def load_raw(perf_path, fing_path):
    """Read frozen upstream export/result, reconstruct 21-point geometry without interpolation."""
    import json
    import cv2
    p = Path(perf_path)
    with np.load(p, allow_pickle=False) as data:
        z = {k: data[k] for k in data.files}
    with np.load(fing_path, allow_pickle=False) as data:
        r = {k: data[k] for k in data.files}
    fps, av = float(z["fps"]), int(r["av_offset"])
    if int(z["frame_offset"]) != 0 or int(r.get("frame_offset", 0)) != 0:
        raise ValueError("nonzero frame_offset")
    if (not np.array_equal(z["note_pitch"], r["pitch"])
            or not np.allclose(z["note_onset"], r["onset"], atol=1e-9, rtol=0)
            or abs(float(r["fps"]) - fps) > 1e-6
            or str(r["video_id"]) != str(z["video_id"])):
        raise ValueError("performance/result timeline mismatch")
    on = np.rint(z["note_onset"] * fps).astype(int) + av
    if not np.array_equal(on, r["onset_frame"]):
        raise ValueError("onset conversion mismatch")
    off_s = z["note_key_offset"]
    if not np.isfinite(off_s).all() or (off_s < z["note_onset"]).any():
        raise ValueError("invalid real note-off")
    off = np.rint(off_s * fps).astype(int) + av  # Never invent a one-frame press.
    hand, source, finger = trust_sources(z, r)
    meta = json.loads(p.with_suffix(".meta.json").read_text(encoding="utf-8"))
    with np.load(p.with_suffix(".skeleton.npz"), allow_pickle=False) as data:
        sk = data["skeleton"]
    cap = cv2.VideoCapture(str(z["video_path"]))
    try:
        if not cap.isOpened():
            raise ValueError("cannot decode source video")
        T = min(len(sk), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        if abs(cap.get(cv2.CAP_PROP_FPS) - fps) > 1e-6:
            raise ValueError("video fps mismatch")
    finally:
        cap.release()
    sk = sk[:T]
    corners = np.float32(meta["keyboard_corners_px"])
    hm = cv2.getPerspectiveTransform(corners, np.float32([[0,0],[52,0],[52,1],[0,1]]))
    if not np.isfinite(hm).all() or abs(np.linalg.det(hm)) < 1e-15:
        raise ValueError("invalid keyboard homography")
    xy = sk[..., :2]
    uv = cv2.perspectiveTransform(np.nan_to_num(xy).reshape(-1,1,2).astype(np.float32), hm).reshape(T,2,21,2)
    ok = np.isfinite(xy).all((2,3))
    # Every queried joint must be on the original image; full-width crop keeps both hands.
    ok &= ((xy[...,0] >= 0) & (xy[...,0] < meta["width"])
           & (xy[...,1] >= 0) & (xy[...,1] < meta["height"])).all(2)
    palm = uv[:, :, [0,5,9,13,17], 0].mean(2)
    tips = uv[:, :, [4,8,12,16,20], 0]
    palm[~ok], tips[~ok] = np.nan, np.nan
    return Perf(str(z["video_id"]), fps, T, str(z["video_path"]), z["note_pitch"], on, off,
                hand, source, finger, palm, tips, ok, z["note_onset"])
