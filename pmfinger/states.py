"""Step 4: derived states for the benchmark (motion states, hand state, music annotations).

Inputs: a Performance (+ PianoVAM extras read straight from the npz) and a fing.npz result.
Everything is indexed by video frame (row of `tips`; video frame = frame_offset + row).

Motion state per (frame, hand, finger), priority HOLD > DOWN > UP > MOVE > AIR:
  DOWN  within T_down before an onset of that finger
  HOLD  onset <= t < release  (PianoVAM: Disklavier key_offset; else PMFinger's tracked release)
  UP    release <= t < release + T_up
  MOVE  otherwise, |d tip_x / dt| > v_move white keys / s
  AIR   otherwise
  UNKNOWN  hand not tracked, or an untrusted note of (possibly) this hand is down / sounding nearby

Trust per note (Benchmark gate, stricter than training):
  hand_trust    manual GT hand, else PMFinger hand with hand_conf >= hand_min and agreeing with the
                PianoVAM auto label when there is one
  finger_trust  manual GT finger (gold), else status ok, conf >= conf_min and agreeing with the auto
                label when there is one (silver)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .events import group_events
from .keys import KEY_X, NAMES, white_pitch_at
from .preprocess import clean_tips, runs

UNKNOWN, DOWN, HOLD, UP, MOVE, AIR = range(6)
STATE_ZH = ["未知", "落下", "按键保持", "抬起", "移动", "空中保持"]
HAND_ZH = ["左手", "右手"]
FINGER_ZH = ["拇指", "食指", "中指", "无名指", "小指"]


@dataclass
class StateConfig:
    t_down_s: float = 0.10
    t_up_s: float = 0.10
    v_move: float = 2.0            # white keys / s
    v_clear_air: float = 1.0       # questions only use AIR whose median speed is below this ...
    v_clear_move: float = 4.0      # ... and MOVE whose median speed is above this (1-4 is visually ambiguous)
    min_seg_s: float = 0.08        # MOVE / AIR runs shorter than this merge into the neighbouring MOVE / AIR
    conf_min: float = 0.9
    hand_min: float = 0.9
    chord_tol_s: float = 0.04
    untrusted_hold_s: float = 1.0  # sounding length assumed for an untrusted note without a known release


def pitch_name(p: int) -> str:
    p = int(p)
    return f"{NAMES[p % 12]}{p // 12 - 1}"


def octave_of_x(x: float) -> int:
    return int(white_pitch_at(x)) // 12 - 1


class Derived:
    """All derived facts for one video."""

    def __init__(self, perf, res, npz=None, cfg: StateConfig | None = None, fcfg: Config | None = None):
        cfg = cfg or StateConfig()
        self.cfg = cfg
        self.perf = perf
        self.vid = perf.video_id
        self.fps = fps = perf.fps
        self.tips, self.ok = clean_tips(perf.tips, perf.valid, fcfg or Config())
        self.T = T = self.ok.shape[0]
        self.av = int(res["av_offset"])
        self.pitch = res["pitch"].astype(int)
        self.N = N = len(self.pitch)
        self.onset_s = res["onset"]
        self.on = res["onset_frame"].astype(int)
        self.conf = res["conf"]
        self.status = res["status"]
        hp, fp = res["hand_pred"].astype(int), res["finger_pred"].astype(int)

        z = npz if npz is not None else {}
        get = (lambda k: z[k] if k in getattr(z, "files", z) else None)
        auto_h, auto_f = get("label_hand"), get("label_finger")
        gt_h, gt_f = get("gt_hand"), get("gt_finger")
        key_off = get("note_key_offset")
        if auto_h is None:
            auto_h = auto_f = np.full(N, -1)
        if gt_h is None:
            gt_h = gt_f = np.full(N, -1)

        has_gt = (gt_h >= 0) & (gt_f >= 0)
        auto_ok_h = (auto_h < 0) | (auto_h == hp)
        auto_ok_f = (auto_h < 0) | ((auto_h == hp) & (auto_f == fp))
        self.hand = np.where(has_gt, gt_h, hp)
        self.finger = np.where(has_gt, gt_f, fp)
        self.hand_trust = has_gt | ((hp >= 0) & (res["hand_conf"] >= cfg.hand_min) & auto_ok_h)
        self.finger_trust = has_gt | ((self.status == 0) & (self.conf >= cfg.conf_min) & auto_ok_f)
        self.label_tier = np.where(has_gt, "gold", np.where(self.finger_trust, "silver", "bronze"))
        self.note_conf = np.where(has_gt, 1.0, self.conf)

        # release frame: true key release when the dataset has it
        if key_off is not None:
            rel = np.rint(key_off * fps).astype(int) + self.av
            self.release_true = True
        else:
            rel = res["release_frame"].astype(int)
            rel = np.where(rel >= 0, rel, np.minimum(res["offset_frame"], self.on + int(fps * cfg.untrusted_hold_s)))
            self.release_true = False
        self.rel = np.maximum(rel, self.on + 1)
        unc = res["release_uncertain"].astype(bool) if not self.release_true else np.zeros(N, bool)
        self.release_trust = ~unc

        self._motion_states()
        self._hand_state()

    # ------------------------------------------------------------------ motion states
    def _motion_states(self):
        c, fps, T = self.cfg, self.fps, self.T
        td, tu = int(round(c.t_down_s * fps)), int(round(c.t_up_s * fps))
        st = np.full((T, 2, 5), AIR, dtype=np.int8)
        x = self.tips[..., 0]
        vx = np.gradient(np.where(np.isfinite(x), x, 0.0), axis=0) * fps
        self.speed = np.abs(vx)
        st[np.abs(vx) > c.v_move] = MOVE
        trusted = self.hand_trust & self.finger_trust & self.release_trust
        for state, lo_fn, hi_fn in ((UP, lambda n: self.rel[n], lambda n: self.rel[n] + tu),
                                    (DOWN, lambda n: self.on[n] - td, lambda n: self.on[n]),
                                    (HOLD, lambda n: self.on[n], lambda n: self.rel[n])):
            for n in np.flatnonzero(trusted):
                a, b = max(0, lo_fn(n)), min(T, hi_fn(n))
                if b > a:
                    seg = st[a:b, self.hand[n], self.finger[n]]
                    if state == UP:
                        seg[:] = UP
                    elif state == DOWN:
                        seg[seg != HOLD] = DOWN
                    else:
                        seg[:] = HOLD
        # merge short MOVE/AIR flicker
        mn = max(1, int(round(c.min_seg_s * fps)))
        for h in range(2):
            for f in range(5):
                s = st[:, h, f]
                for cur, other in ((MOVE, AIR), (AIR, MOVE)):
                    for a, b in zip(*runs(s == cur)):
                        if b - a < mn:
                            left = s[a - 1] if a > 0 else -1
                            right = s[b] if b < T else -1
                            if left == other or right == other:
                                s[a:b] = other
        # unknown: untracked hand, or an untrusted note that may belong to this hand
        st[~self.ok] = UNKNOWN
        bad = ~trusted
        for n in np.flatnonzero(bad):
            hands = [int(self.hand[n])] if (self.hand_trust[n] and self.hand[n] >= 0) else [0, 1]
            a, b = max(0, self.on[n] - td), min(T, self.rel[n] + tu)
            for h in hands:
                st[a:b, h, :] = UNKNOWN
        self.state = st

    # ------------------------------------------------------------------ hand state
    def _hand_state(self):
        x = self.tips[..., 0]
        with np.errstate(invalid="ignore"):
            self.center = np.nanmean(x, axis=2)
            self.lo = np.nanmin(x, axis=2) if np.isfinite(x).any() else x[..., 0]
            self.hi = np.nanmax(x, axis=2) if np.isfinite(x).any() else x[..., 0]
        self.span = self.hi - self.lo
        self.center[~self.ok] = np.nan
        self.span[~self.ok] = np.nan

    def octave_at(self, h, t):
        c = self.center[t, h]
        return None if not np.isfinite(c) else octave_of_x(c)

    # ------------------------------------------------------------------ queries
    def frame_of(self, sec):
        return int(round(sec * self.fps))

    def notes_of(self, h, t0, t1, need_finger=False):
        """Notes of hand h with onset frame in [t0, t1)."""
        m = (self.hand == h) & (self.on >= t0) & (self.on < t1)
        return np.flatnonzero(m)

    def any_untrusted(self, t0, t1, h=None, need_finger=False, need_release=False):
        """True if a note touching [t0, t1) (down, held or pressed) is not trusted enough."""
        touch = (self.on < t1) & (self.rel > t0)
        if h is not None:
            touch &= (self.hand == h) | ~self.hand_trust
        bad = ~self.hand_trust
        if need_finger:
            bad = bad | ~self.finger_trust
        if need_release:
            bad = bad | ~self.release_trust
        return bool((touch & bad).any())

    def edge_quiet(self, t, h, margin_s=0.15):
        """No press or release of hand h within margin of frame t."""
        m = int(round(margin_s * self.fps))
        hm = (self.hand == h)
        near_on = hm & (np.abs(self.on - t) <= m)
        near_rel = hm & (np.abs(self.rel - t) <= m)
        return not (near_on.any() or near_rel.any())

    def tracked(self, t0, t1, hands=(0, 1), min_frac=0.95):
        if t0 < 0 or t1 > self.T or t1 <= t0:
            return False
        return all(self.ok[t0:t1, h].mean() >= min_frac for h in hands)

    def events(self, idx):
        """Group note indices into chord events (sorted by onset), each sorted by pitch."""
        idx = sorted(idx, key=lambda n: (self.onset_s[n], self.pitch[n]))
        evs = group_events(self.onset_s, idx, self.cfg.chord_tol_s)
        return [sorted(e, key=lambda n: self.pitch[n]) for e in evs]

    def motion_clear(self, state, a, b, h, f):
        """True unless frames [a, b) are MOVE / AIR with a median fingertip speed in the ambiguous band."""
        if state not in (MOVE, AIR):
            return True
        v = float(np.median(self.speed[max(0, a):max(a + 1, b), h, f]))
        return v < self.cfg.v_clear_air if state == AIR else v > self.cfg.v_clear_move

    def state_runs(self, t0, t1, h, f):
        """Collapsed state sequence of one finger over [t0, t1): list of (state, start, stop)."""
        s = self.state[t0:t1, h, f]
        out = []
        for k, v in enumerate(s):
            if out and out[-1][0] == v:
                out[-1][2] = t0 + k + 1
            else:
                out.append([int(v), t0 + k, t0 + k + 1])
        return out

    def active_keys(self, t):
        m = (self.on <= t) & (self.rel > t)
        return np.flatnonzero(m)


# ---------------------------------------------------------------------- music theory

CHORDS = {  # interval set from root -> name
    (0, 4, 7): "大三和弦", (0, 3, 7): "小三和弦", (0, 3, 6): "减三和弦", (0, 4, 8): "增三和弦",
    (0, 4, 7, 10): "属七和弦", (0, 4, 7, 11): "大七和弦", (0, 3, 7, 10): "小七和弦",
    (0, 3, 6, 10): "半减七和弦", (0, 3, 6, 9): "减七和弦",
}
INTERVALS = ["纯一度", "小二度", "大二度", "小三度", "大三度", "纯四度", "增四度", "纯五度",
             "小六度", "大六度", "小七度", "大七度", "纯八度"]


def chord_name(pitches):
    """Name of a chord from its pitches, or None. Root by matching rotations; augmented/dim7
    (symmetric) take the lowest note as root."""
    pcs = sorted({int(p) % 12 for p in pitches})
    if len(pcs) < 3:
        return None
    bass = min(pitches) % 12
    cands = []
    for r in pcs:
        iv = tuple(sorted((p - r) % 12 for p in pcs))
        if iv in CHORDS:
            cands.append((r != bass, r, CHORDS[iv]))
    if not cands:
        return None
    _, r, q = sorted(cands)[0]
    return f"{NAMES[r]} {q}"


def interval_name(p1, p2):
    d = abs(int(p2) - int(p1))
    return INTERVALS[d] if d <= 12 else None


def melody_shape(pitches):
    """上行 / 下行 / 先上行后下行 / 先下行后上行, or None for flat or zig-zag lines."""
    d = np.sign(np.diff(pitches))
    d = d[d != 0]
    if len(d) < 2:
        return None
    turns = int((np.diff(d) != 0).sum())
    if turns == 0:
        return "上行" if d[0] > 0 else "下行"
    if turns == 1 and min((d == d[0]).sum(), (d != d[0]).sum()) >= 2:
        return "先上行后下行" if d[0] > 0 else "先下行后上行"
    return None


def crossing_events(der: Derived, idx):
    """Thumb-under (穿指) / finger-over (跨指) between consecutive single notes of one hand.

    Right hand: ascending 3/4 -> thumb is 穿指, descending thumb -> 3/4 is 跨指; left hand mirrored.
    Returns list of (kind, n_prev, n_next)."""
    out = []
    evs = [e for e in der.events(idx)]
    for e0, e1 in zip(evs[:-1], evs[1:]):
        if len(e0) != 1 or len(e1) != 1:
            continue
        a, b = e0[0], e1[0]
        if der.onset_s[b] - der.onset_s[a] > 0.6:
            continue
        h = int(der.hand[a])
        up = der.pitch[b] > der.pitch[a]
        fa, fb = int(der.finger[a]), int(der.finger[b])
        toward_pinky_side = up if h == 1 else not up   # direction in which the finger number grows
        if fa in (2, 3) and fb == 0 and toward_pinky_side:
            out.append(("穿指", a, b))
        elif fa == 0 and fb in (2, 3) and not toward_pinky_side:
            out.append(("跨指", a, b))
    return out


def load_derived(perf_path, fing_path, cfg=None):
    from .pipeline import load_result
    from .schema import load_performance
    perf = load_performance(perf_path)
    z = np.load(perf_path, allow_pickle=False)
    res = load_result(fing_path)
    return Derived(perf, res, z, cfg)
