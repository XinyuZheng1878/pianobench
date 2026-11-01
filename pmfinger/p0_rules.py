"""Spec gates migrated into the upstream fork. Historical generators are untouched."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
_spec = importlib.util.spec_from_file_location("_p0_states", Path(__file__).with_name("p0_states.py"))
_states = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _states
_spec.loader.exec_module(_states)
FINGER_ZH, HAND_ZH = _states.FINGER_ZH, _states.HAND_ZH
IS_BLACK, KEY_X, pitch_name = _states.IS_BLACK, _states.KEY_X, _states.pitch_name
LETTERS = "ABCDEFG"
U_CLIP_S, U_FPS = 5.0, 8          # understanding (T3)
L_CLIP_S = 6.0                    # localisation (T4), 8 fps
P_CTX_S, P_FPS = 2.0, 15          # prediction (T5): frames end at t0 = last delivered frame
CHORD_TOL_S = 0.04                # onset group: onsets within 40 ms after the group's first onset
EDGE_S = 0.15                     # answer-relevant events this close to an interval end -> no question
FINGER_SUBTYPES = {"按键次数·指定手指", "指定手指按指定键", "下一根手指"}   # need human approval before use


class Gen:
    def __init__(self, P, rnd):
        self.P, self.rnd, self.fps = P, rnd, P.fps
        self.edge = int(round(EDGE_S * P.fps))

    def fr(self, s):
        return int(round(s * self.fps))

    # ------------------------------------------------------------------ delivered frames
    def grid(self, c0, c1, fps_out, end_at_last=False):
        """Delivered video frames. For prediction clips the last delivered frame is c1 (= t0, 'now')."""
        n = int(round((c1 - c0) / self.fps * fps_out))
        if end_at_last:
            return [int(round(c1 - (n - 1 - i) * self.fps / fps_out)) for i in range(n)]
        return [int(round(c0 + i * self.fps / fps_out)) for i in range(n)]

    def seen_down(self, n, grid):
        """Delivered frames on which note n's key is down (this note specifically)."""
        P = self.P
        h = P.hand[n]
        return [f for f in grid if P.on[n] <= f < P.off[n] and h >= 0 and P.ok[f, h]]

    def press_visible(self, n, grid):
        """A visible down frame and an earlier delivered up frame identify a new press."""
        P = self.P
        down = self.seen_down(n, grid)
        before = [f for f in grid if f < P.on[n]]
        return bool(down) and bool(before) and not P.pressed(P.pitch[n], before[-1]) and bool(P.ok[before[-1], P.hand[n]])

    @staticmethod
    def smooth_palm(raw, size):
        """Odd-width median, identical to nearest-edge padding; no decoder/SciPy import needed."""
        padded = np.pad(raw, size // 2, mode="edge")
        return np.median(np.lib.stride_tricks.sliding_window_view(padded, size), axis=-1)

    def gap_frame(self, a, b, grid):
        """Is there a delivered frame in [a, b)? (an event at a and another at b look different on it)"""
        return any(a <= f < b for f in grid)

    # ------------------------------------------------------------------ atoms
    def atoms(self, c0, frames, t_end=None, extra=()):
        """Key events and palm positions in the clip; t_end hides everything after it (prediction)."""
        P, out = self.P, []
        a, b = frames[0], (frames[-1] if t_end is None else t_end) + 1
        rel = lambda t: (t - c0) / self.fps
        for n in np.flatnonzero((P.on < b) & (P.off > a)):
            hand = HAND_ZH[P.hand[n]] if P.hand[n] >= 0 else "未知手"
            if a <= P.on[n] < b:
                d = dict(id=f"k{n}", t=rel(P.on[n]), subject=hand, predicate="按下", object=pitch_name(P.pitch[n]),
                         source="MIDI")
                if P.finger[n] >= 0:
                    d.update(finger=FINGER_ZH[P.finger[n]], finger_source="人工金标")
                out.append(d)
            if a <= P.off[n] < b:
                out.append(dict(id=f"r{n}", t=rel(P.off[n]), subject=hand, predicate="松开", object=pitch_name(P.pitch[n]),
                                source="MIDI"))
        for i, t in enumerate(frames):
            if (t_end is not None and t > t_end) or i % 4:
                continue
            for h in (0, 1):
                if P.ok[t, h]:
                    out.append(dict(id=f"p{h}_{i}", t=rel(t), frame=i, subject=HAND_ZH[h], predicate="手掌中心位于",
                                    object=P.key_name_at(P.palm[t, h]), x=round(float(P.palm[t, h]), 2), source="骨架"))
        out += list(extra)
        return sorted(out, key=lambda d: d["t"])

    def geo_atom(self, aid, c0, t, h, predicate, x):
        return dict(id=aid, t=(t - c0) / self.fps, subject=HAND_ZH[h], predicate=predicate,
                    object=self.P.key_name_at(x), x=round(float(x), 2), source="骨架")

    def fact(self, task, sub, level, c0, c1, fps_out, slots, answer_type, answer, options=None, rng=None,
             horizon=None, difficulty=None, end_at_last=False, extra_atoms=()):
        P = self.P
        frames = self.grid(c0, c1, fps_out, end_at_last)
        if options:
            ch = options["choices"]
            expected = 4 if sub == "下一个键（选择）" else len(ch)
            if len(ch) != expected or len(ch) < 2 or len(set(ch)) != len(ch):
                raise ValueError("invalid options: " + str(ch))
            if options["letter"] not in LETTERS[:len(ch)] or answer != options["letter"]:
                raise ValueError("option letter and answer disagree")
        f0 = frames[0]
        fl = [dict(i=i, video_frame=int(f), t=(f - f0) / P.fps) for i, f in enumerate(frames)]
        predict = {"t0": fl[-1]["t"], "horizon": horizon} if horizon is not None else None
        rec = dict(task=task, subtype=sub, level=level, video_id=P.video_id, video_path=P.video_path,
                   clip=dict(frame0=int(f0), frame_last=int(frames[-1]), fps=P.fps, fps_out=fps_out,
                             start_s=round(f0 / P.fps, 3), n_frames=len(frames)),
                   frames=fl, range=rng, predict=predict, slots=slots, answer_type=answer_type, answer=answer,
                   options=options, difficulty=dict(difficulty or {}, frame_interval_s=round(1 / fps_out, 3)),
                   schema_version="0.3.1", split="bench", needs_human=sub in FINGER_SUBTYPES,
                   atoms=self.atoms(f0, frames, frames[-1] if end_at_last else None, extra_atoms))
        return rec

    # ------------------------------------------------------------------ helpers
    def rel(self, t, c0):
        return (t - c0) / self.fps

    def mc(self, correct, pool):
        opts = [correct] + [p for p in dict.fromkeys(pool) if p != correct]
        self.rnd.shuffle(opts)
        return dict(choices=opts, letter=LETTERS[opts.index(correct)])

    def ivl(self, c0, c1, lo, hi):
        L = self.fr(self.rnd.uniform(lo, hi))
        if c1 - c0 <= L + 2:
            return None
        a = self.rnd.randrange(c0 + 1, c1 - L)
        return a, a + L

    def rng_obj(self, a, b, c0):
        return {"type": "timestamp", "start": self.rel(a, c0), "end": self.rel(b, c0)}

    def has_jump(self, h, a, b, max_step=1.0):
        """Skeleton tracking glitch: palm x jumps more than `max_step` white keys between consecutive frames."""
        palm = self.P.palm[max(0, a):b, h, None]
        tips = self.P.tips[max(0, a):b, h]
        x = np.concatenate([palm, tips], axis=1)
        # Missing rows must not hide a tracking jump across the gap.
        if len(x) < 2 or not np.isfinite(x).all():
            return True
        return bool(np.max(np.abs(np.diff(x, axis=0))) > max_step)

    def near_edge(self, times, a, b):
        return any(min(abs(t - a), abs(t - b)) < self.edge for t in times)

    def countable(self, groups, grid):
        """Each counted press must be visible and separable from the previous one on the delivered frames:
        some frame shows it down, its first visible frame comes after the previous group's, and a re-press of
        the same key needs a delivered frame where that key is up in between."""
        P, prev_first = self.P, -1
        for i, g in enumerate(groups):
            if not all(self.press_visible(n, grid) for n in g):
                return False
            vis = sorted({f for n in g for f in self.seen_down(n, grid)})
            if not vis or vis[0] <= prev_first:
                return False
            prev_first = vis[0]
            if i:
                for m in (x for prev in groups[:i] for x in prev):
                    for n in g:
                        if P.pitch[m] == P.pitch[n] and not any(P.off[m] <= f < P.on[n]
                                and not P.pressed(P.pitch[n], f) and P.ok[f, P.hand[n]] for f in grid):
                            return False
        return True

    # ------------------------------------------------------------------ T3 (understanding, 5 s, 8 fps)
    def t3_count_hand(self, c0, c1):
        P, h = self.P, self.rnd.randrange(2)
        iv = self.ivl(c0, c1, 1.5, 4.0)
        if not iv:
            return None
        a, b = iv
        if not P.visible(h, a, b, 0.95) or len(P.hand_unknown_in(a - self.edge, b + self.edge)):
            return None
        if self.near_edge(P.on[P.notes(a - self.edge, b + self.edge, h)], a, b):
            return None
        gs = P.groups(P.notes(a, b, h), CHORD_TOL_S)
        if len(gs) > 12 or not self.countable(gs, self.grid(c0, c1, U_FPS)):
            return None
        ioi = np.diff([P.on_s[g[0]] for g in gs]) if len(gs) > 1 else [9.0]
        return self.fact("T3", "按键次数·整只手", "L2", c0, c1, U_FPS, dict(hand=HAND_ZH[h]), "int", len(gs),
                         rng=self.rng_obj(a, b, c0),
                         difficulty=dict(n_events=len(gs), min_ioi_s=round(float(min(ioi)), 3),
                                         chords=sum(len(g) > 1 for g in gs)))

    def t3_count_finger(self, c0, c1):
        P, h = self.P, self.rnd.randrange(2)
        iv = self.ivl(c0, c1, 1.5, 4.0)
        if not iv:
            return None
        a, b = iv
        idx = P.notes(a - self.edge, b + self.edge, h)
        if not P.visible(h, a, b, 0.95) or len(P.hand_unknown_in(a - self.edge, b + self.edge)):
            return None
        if (P.finger[idx] < 0).any() or self.near_edge(P.on[idx], a, b):
            return None                       # every note that could be the target finger must be human-labelled
        inside = P.notes(a, b, h)
        used = sorted({int(P.finger[n]) for n in inside})
        k = self.rnd.choice(used) if used and self.rnd.random() < 0.8 else self.rnd.randrange(5)
        gs = P.groups([n for n in inside if P.finger[n] == k], CHORD_TOL_S)
        if any(len(g) > 1 for g in gs) or not self.countable(gs, self.grid(c0, c1, U_FPS)):
            return None
        return self.fact("T3", "按键次数·指定手指", "L5", c0, c1, U_FPS, dict(hand=HAND_ZH[h], finger=FINGER_ZH[k]),
                         "int", len(gs), rng=self.rng_obj(a, b, c0),
                         difficulty=dict(n_hand_events=len(P.groups(list(inside), CHORD_TOL_S)), finger=k + 1))

    HOLD_BINS = [(0, .2, "不到 0.2 秒"), (.2, .5, "0.2 秒到不足 0.5 秒"), (.5, 1.0, "0.5 秒到不足 1 秒"),
                 (1.0, 99, "1 秒或更久")]

    def t3_hold(self, c0, c1):
        P = self.P
        snd = P.sounding_in(c0, c1)
        cand = [n for n in P.notes(c0 + self.edge, c1) if P.off[n] < c1 - self.edge
                and (P.pitch[snd] == P.pitch[n]).sum() == 1]
        if not cand:
            return None
        n = self.rnd.choice(cand)
        grid = self.grid(c0, c1, U_FPS)
        down = self.seen_down(n, grid)
        if not down:
            return None
        before = [f for f in grid if f < down[0]]
        after = [f for f in grid if f > down[-1]]
        if not before or not after:
            return None
        lo, hi = (down[-1] - down[0]) / self.fps, (after[0] - before[-1]) / self.fps   # durations the frames allow
        d = (P.off[n] - P.on[n]) / self.fps
        bins = {i for i, (b0, b1, _) in enumerate(self.HOLD_BINS) if b0 <= d < b1}
        if not bins or not all(b0 <= lo and hi < b1 for i, (b0, b1, _) in enumerate(self.HOLD_BINS) if i in bins):
            return None                         # the whole visually possible range must sit in the true bin
        opts = [l for _, _, l in self.HOLD_BINS]
        i = bins.pop()
        return self.fact("T3", "按键保持时长", "L3", c0, c1, U_FPS, dict(key=pitch_name(P.pitch[n])), "choice",
                         LETTERS[i], options=dict(choices=opts, letter=LETTERS[i]), rng={"type": "whole"},
                         difficulty=dict(hold_s=round(d, 3), visual_range_s=[round(lo, 3), round(hi, 3)],
                                         black=bool(IS_BLACK[P.pitch[n]])))

    SPAN_BINS = [(0, 4, "不到 4 个白键宽"), (4, 6, "4 到不足 6 个白键宽"), (6, 8, "6 到不足 8 个白键宽"),
                 (8, 99, "8 个白键宽或更大")]

    def span_bin(self, v):
        return next((i for i, (lo, hi, _) in enumerate(self.SPAN_BINS) if lo <= v < hi), -1)

    def t3_span(self, c0, c1):
        P, h = self.P, self.rnd.randrange(2)
        fr = self.grid(c0, c1, U_FPS)
        i = self.rnd.randrange(2, len(fr) - 2)
        t = fr[i]
        w = self.fr(0.1)
        if not P.visible(h, t - w, t + w + 1, 1.0):
            return None
        sp = [P.span(h, k) for k in range(t - w, t + w + 1)]
        v = P.span(h, t)                       # the asked frame itself
        if self.span_bin(v) < 0 or self.has_jump(h, t - w, t + w + 1) or max(sp) - min(sp) > 0.5 or any(self.span_bin(x) != self.span_bin(v) for x in sp):
            return None
        margin = min(abs(v - lo) for lo, _, _ in self.SPAN_BINS[1:])
        if margin < 0.3:
            return None
        opts = [l for _, _, l in self.SPAN_BINS]
        j = self.span_bin(v)
        geo = [self.geo_atom("g_thumb", fr[0], t, h, "拇指指尖位于", P.tips[t, h, 0]),
               self.geo_atom("g_little", fr[0], t, h, "小指指尖位于", P.tips[t, h, 4])]
        for g in geo:
            g["frame"] = i
        return self.fact("T3", "张开跨度", "L3", c0, c1, U_FPS, dict(hand=HAND_ZH[h], frame=i, time=self.rel(t, fr[0])),
                         "choice", LETTERS[j], options=dict(choices=opts, letter=LETTERS[j]),
                         rng={"type": "frame_point", "frame": i, "t": self.rel(t, fr[0])},
                         difficulty=dict(span=round(v, 2), margin=round(margin, 2)), extra_atoms=geo)

    MOVE_OPTS = ["净位移不足 1 个白键宽", "向左 1 到不足 4 个白键宽", "向右 1 到不足 4 个白键宽", "向左 4 到不足 8 个白键宽",
                 "向右 4 到不足 8 个白键宽", "向左 8 个白键宽或更多", "向右 8 个白键宽或更多"]

    def move_label(self, d):
        ad = abs(d)
        if ad < 1:
            return self.MOVE_OPTS[0]
        side = "左" if d < 0 else "右"
        return next(o for o in self.MOVE_OPTS[1:] if o.startswith(f"向{side}") and
                    ((ad < 4 and "1 到" in o) or (4 <= ad < 8 and "4 到" in o) or (ad >= 8 and "或更多" in o)))

    def t3_move(self, c0, c1):
        P, h = self.P, self.rnd.randrange(2)
        iv = self.ivl(c0, c1, 1.0, 4.0)
        if not iv:
            return None
        grid = self.grid(c0, c1, U_FPS)
        a, b = (min(grid, key=lambda f: abs(f - x)) for x in iv)
        w = self.fr(0.05)
        if a >= b or a - w < grid[0] or b + w > grid[-1]:
            return None
        if not P.visible(h, a, b, 0.9) or not (P.visible(h, a - w, a + w + 1, 1.0) and P.visible(h, b - w, b + w + 1, 1.0)):
            return None
        if self.has_jump(h, a - w, b + w + 1):
            return None
        xa, xb = P.palm_med(h, a, w), P.palm_med(h, b, w)
        d = xb - xa
        if min(abs(abs(d) - e) for e in (1, 4, 8)) < 0.4:
            return None
        lab = self.move_label(d)
        if self.move_label(P.palm[b, h] - P.palm[a, h]) != lab:
            return None  # the delivered endpoint frames must support the median-derived bin
        opts = dict(choices=list(self.MOVE_OPTS), letter=LETTERS[self.MOVE_OPTS.index(lab)])
        geo = [self.geo_atom("g_start", c0, a, h, "区间开始时手掌中心位于", xa),
               self.geo_atom("g_end", c0, b, h, "区间结束时手掌中心位于", xb)]
        return self.fact("T3", "手掌净位移", "L2", c0, c1, U_FPS, dict(hand=HAND_ZH[h]), "choice", opts["letter"],
                         options=opts, rng=self.rng_obj(a, b, c0), extra_atoms=geo,
                         difficulty=dict(dx=round(d, 2), path=round(float(np.nansum(np.abs(np.diff(P.palm[a:b, h])))), 2)))

    def once_in_clip(self, c0, c1):
        """Notes whose key is pressed exactly once in the clip and not held from before it."""
        P = self.P
        snd = P.sounding_in(c0, c1)
        return [n for n in P.notes(c0, c1) if (P.pitch[snd] == P.pitch[n]).sum() == 1]

    def t3_overlap(self, c0, c1):
        """Pressing key B, is key A (pressed earlier) still held? Key state only, no finger."""
        P = self.P
        once = self.once_in_clip(c0, c1)
        pairs = [(m, n) for m in once for n in once
                 if P.on_s[n] - P.on_s[m] > CHORD_TOL_S and P.on[n] - P.on[m] < self.fr(1.5)]
        if not pairs:
            return None
        m, n = self.rnd.choice(pairs)
        held = P.off[m] > P.on[n]
        grid = self.grid(c0, c1, U_FPS)
        if not self.press_visible(m, grid) or not self.press_visible(n, grid):
            return None
        # a delivered frame must show the deciding state: both down together, or A up before B goes down
        if abs(P.off[m] - P.on[n]) / self.fps < 1 / U_FPS:
            return None
        if held and not any(P.on[n] <= f < min(P.off[m], P.off[n]) and P.ok[f, P.hand[m]]
                            and P.ok[f, P.hand[n]] for f in grid):
            return None
        if not held and not self.gap_frame(P.off[m], P.on[n], grid):
            return None
        same = P.hand[m] == P.hand[n] and P.hand[m] >= 0
        sub = "持键重叠" if same else ("双手配合" if P.hand[m] >= 0 and P.hand[n] >= 0 else None)
        if sub is None:
            return None
        opts = dict(choices=["还按着", "已经松开"], letter="A" if held else "B")
        return self.fact("T3", sub, "L4", c0, c1, U_FPS,
                         dict(key_a=pitch_name(P.pitch[m]), hand_a=HAND_ZH[P.hand[m]], key_b=pitch_name(P.pitch[n]),
                              hand_b=HAND_ZH[P.hand[n]]),
                         "choice", opts["letter"], options=opts, rng={"type": "whole"},
                         difficulty=dict(gap_s=round((P.off[m] - P.on[n]) / self.fps, 3)))

    def t3_release_order(self, c0, c1):
        P = self.P
        once = set(self.once_in_clip(c0, c1))
        # Group first; removing repeated keys before grouping would change the 40 ms anchor.
        gs = [[n for n in g if n in once and P.on[n] >= c0 + self.edge]
              for g in P.groups(np.arange(len(P.pitch)), CHORD_TOL_S)]
        gs = [g for g in gs if len(g) >= 2]
        if not gs:
            return None
        m, n = self.rnd.sample(self.rnd.choice(gs), 2)
        if max(P.off[m], P.off[n]) >= c1 - self.edge or P.off[m] == P.off[n]:
            return None
        first, second = (m, n) if P.off[m] < P.off[n] else (n, m)
        grid = self.grid(c0, c1, U_FPS)
        if (not all(self.press_visible(x, grid) for x in (m, n))
                or not any(max(P.on[m], P.on[n]) <= f < min(P.off[m], P.off[n])
                           and P.ok[f, P.hand[m]] and P.ok[f, P.hand[n]] for f in grid)
                or (P.off[second] - P.off[first]) / self.fps < 1 / U_FPS
                or not any(P.off[first] <= f < P.off[second] and P.ok[f, P.hand[m]]
                           and P.ok[f, P.hand[n]] for f in grid)):
            return None                        # no delivered frame shows one key up while the other is still down
        a, b = pitch_name(P.pitch[m]), pitch_name(P.pitch[n])
        letter = "A" if first == m else "B"
        return self.fact("T3", "先松开哪个键", "L4", c0, c1, U_FPS, dict(key_a=a, key_b=b), "choice", letter,
                         options=dict(choices=[f"{a} 先松开", f"{b} 先松开"], letter=letter), rng={"type": "whole"},
                         difficulty=dict(release_gap_s=round((P.off[second] - P.off[first]) / self.fps, 3)))

    # ------------------------------------------------------------------ T4 (localisation, 6 s, 8 fps)
    def t4(self, c0, c1):
        P, out = self.P, []
        edge = self.fr(0.5)
        add = lambda *x: out.append(x)
        for h in (0, 1):
            if not P.visible(h, c0, c1, 0.9) or len(P.hand_unknown_in(c0, c1)):
                continue
            full_groups = P.groups(np.flatnonzero(P.hand == h), CHORD_TOL_S)
            # Do not turn a chord straddling a clip boundary into a partial/single-note group.
            if any(any(c0 <= P.on[n] < c1 for n in g) and not all(c0 <= P.on[n] < c1 for n in g) for g in full_groups):
                continue
            gs = [g for g in full_groups if c0 <= P.on[g[0]] < c1]
            # L1 first local maximum of three consecutive single-note groups
            for i in range(1, len(gs) - 1):
                if all(len(gs[j]) == 1 for j in (i - 1, i, i + 1)):
                    p = [P.pitch[gs[j][0]] for j in (i - 1, i, i + 1)]
                    if p[0] < p[1] > p[2]:
                        n = gs[i][0]
                        add("旋律最高点", "L1", dict(hand=HAND_ZH[h]), "time_point", P.on[n], P.on[n], P.on[n],
                            dict(key=pitch_name(P.pitch[n]), witness=[x for g in gs[:i + 2] for x in g]), ())
                        break
            # L1 first onset group with >= 3 different keys
            big = [g for g in gs if len({P.pitch[n] for n in g}) >= 3]
            if big:
                n = min(big[0], key=lambda x: P.on_s[x])
                add("第一次三音以上同时按下", "L1", dict(hand=HAND_ZH[h]), "time_point", P.on[n], P.on[n], P.on[n], dict(witness=[x for g in gs[:gs.index(big[0]) + 1] for x in g]), ())
            # L2 first press in an octave this hand has not used earlier in the clip; anchor = that note itself
            seen = set()
            for g in gs:
                octs = {int(P.pitch[n]) // 12 - 1 for n in g}
                new = octs - seen
                if seen and new:
                    n = min((x for x in g if int(P.pitch[x]) // 12 - 1 in new), key=lambda x: P.on_s[x])
                    o = int(P.pitch[n]) // 12 - 1
                    if P.on[n] - c0 > edge:
                        add("第一次按进新的八度", "L2", dict(hand=HAND_ZH[h], octave=f"C{o}–B{o}"), "time_point",
                            P.on[n], P.on[n], P.on[n], dict(witness=[x for gg in gs[:gs.index(g) + 1] for x in gg]), ())
                    break
                seen |= octs
            # L2 first big lateral move
            raw = P.palm[c0:c1, h]
            if np.isfinite(raw).mean() > 0.95 and not self.has_jump(h, c0, c1):
                pc = self.smooth_palm(raw, self.fr(0.1) | 1)
                v = np.abs(np.gradient(pc)) * self.fps
                i = 0
                while i < len(v):
                    if v[i] > 3.0:
                        j = i
                        while j < len(v) and v[j] > 1.5:
                            j += 1
                        jj = min(j, len(pc) - 1)
                        if self.fr(0.25) <= (j - i) <= self.fr(1.2) and abs(pc[jj] - pc[i]) >= 7:
                            geo = (self.geo_atom("g_start", c0, c0 + i, h, "移动开始时手掌中心位于", pc[i]),
                                   self.geo_atom("g_end", c0, c0 + jj, h, "移动结束时手掌中心位于", pc[jj]))
                            add("第一次大幅横移", "L2", dict(hand=HAND_ZH[h]), "time_span", [c0 + i, c0 + j], c0 + i, c0 + j,
                                dict(dx=round(float(pc[jj] - pc[i]), 2)), geo)
                            break
                        i = j
                    i += 1
            # L4 unique run of 3 consecutive single notes (no other onset of this hand in between)
            runs = [gs[i:i + 3] for i in range(len(gs) - 2) if all(len(g) == 1 for g in gs[i:i + 3])]
            if runs:
                r = self.rnd.choice(runs)
                keys = [int(P.pitch[g[0]]) for g in r]
                hits = sum(1 for rr in runs if [int(P.pitch[g[0]]) for g in rr] == keys)
                a, b = P.on[r[0][0]], P.on[r[2][0]]
                if hits == 1 and len(set(keys)) > 1 and b - a >= self.fr(0.3):
                    add("三音键序列", "L4", dict(hand=HAND_ZH[h], keys=[pitch_name(p) for p in keys]), "time_span",
                        [a, b], a, b, dict(witness=[gg[0] for gg in r]), ())
            # L5 (finger, key) unique: presses of the same key by this hand with an unknown finger count as rivals
            hand_notes = [n for g in gs for n in g]
            uniq = [n for n in hand_notes if P.finger[n] >= 0 and
                    sum(1 for m in hand_notes if P.pitch[m] == P.pitch[n] and (P.finger[m] == P.finger[n] or P.finger[m] < 0)) == 1]
            if uniq:
                n = self.rnd.choice(uniq)
                add("指定手指按指定键", "L5", dict(hand=HAND_ZH[h], finger=FINGER_ZH[P.finger[n]], key=pitch_name(P.pitch[n])),
                    "time_point", P.on[n], P.on[n], P.on[n], dict(witness=[n]), ())
        # L3 a key held more than 1 s: the only hold of that key touching the clip (key level, any hand)
        snd = P.sounding_in(c0, c1)
        for n in P.notes(c0, c1):
            if P.off[n] - P.on[n] >= self.fps and P.off[n] < c1 and (P.pitch[snd] == P.pitch[n]).sum() == 1:
                add("长按某个键", "L3", dict(key=pitch_name(P.pitch[n])), "time_span", [P.on[n], P.off[n]], P.on[n], P.off[n],
                    dict(hold_s=round((P.off[n] - P.on[n]) / self.fps, 2), witness=[n]), ())
                break
        facts = []
        f0 = self.grid(c0, c1, U_FPS)[0]
        for sub, lvl, slots, at, ans, ta, tb, diff, geo in out:
            if ta - c0 < edge or c1 - tb < edge:
                continue
            witness = diff.pop("witness", [])
            grid = self.grid(c0, c1, U_FPS)
            if witness and not self.countable(P.groups(witness, CHORD_TOL_S), grid):
                continue
            if sub == "长按某个键" and not self.gap_frame(tb, grid[-1] + 1, grid):
                continue
            if sub == "第一次大幅横移":
                ta, tb = [min(grid, key=lambda t: abs(t - x)) for x in (ta, tb)]
                if ta - c0 < edge or c1 - tb < edge or ta >= tb:
                    continue
                xa, xb = P.palm_med(int(slots["hand"] == "右手"), ta, self.fr(.05)), P.palm_med(int(slots["hand"] == "右手"), tb, self.fr(.05))
                if abs(xb - xa) < 7 or not .25 <= (tb-ta)/self.fps <= 1.2:
                    continue
                ans = [ta, tb]
                geo = (self.geo_atom("g_start", c0, ta, int(slots["hand"] == "右手"), "移动开始时手掌中心位于", xa),
                       self.geo_atom("g_end", c0, tb, int(slots["hand"] == "右手"), "移动结束时手掌中心位于", xb))
            ans = [self.rel(x, f0) for x in ans] if at == "time_span" else self.rel(ans, f0)
            diff = dict(diff, answer_pos=round((ta - c0) / (c1 - c0), 2))
            facts.append(self.fact("T4", sub, lvl, c0, c1, U_FPS, slots, at, ans, difficulty=diff, extra_atoms=geo))
        return facts

    # ------------------------------------------------------------------ T5 (prediction; t0 = last frame)
    def t5(self, t0, h):
        P = self.P
        c0 = t0 - self.fr(P_CTX_S)
        if c0 < 0 or t0 + self.fr(0.6) >= P.T or not P.visible(h, c0, t0 + 1, 0.9):
            return []
        if not P.visible(h, t0 - self.fr(0.1), t0 + 1, 1.0):
            return []                          # the hand must be seen at 'now'
        if len(P.hand_unknown_in(c0, t0 + self.fr(0.6))):
            return []
        out, w = [], self.fr(1 / P_FPS)
        hist = P.notes(c0, t0 + 1, h)
        since = (t0 - P.on[hist].max()) / self.fps if len(hist) else 9.0
        base = dict(since_last_onset_s=round(float(since), 3))
        kw = dict(end_at_last=True)
        fut = P.notes(t0 + 1, P.T, h)
        nxt = P.on[fut].min() if len(fut) else 10 ** 9
        for H in (0.2, 0.5):
            tH = t0 + self.fr(H)
            if abs(nxt - tH) >= w and nxt - t0 > self.fr(0.03):
                yes = nxt <= tH
                out.append(self.fact("T5", "会不会发生新的按键", "L2", c0, t0, P_FPS, dict(hand=HAND_ZH[h]), "choice",
                                     "A" if yes else "B", options=dict(choices=["会", "不会"], letter="A" if yes else "B"),
                                     horizon=H, difficulty=dict(base, horizon=H), **kw))
            down_now = np.flatnonzero((P.on <= t0) & (P.off > t0) & (P.hand == h))
            if len(down_now):
                n = int(self.rnd.choice(list(down_now)))
                p = int(P.pitch[n])
                ev = np.concatenate([P.on[P.pitch == p], P.off[P.pitch == p]])
                if np.abs(ev - tH).min() >= w:          # no press / release of this key near the target time
                    still = P.pressed(p, tH)            # key state at t0+H, re-presses included
                    out.append(self.fact("T5", "某个键之后是否处于按下状态", "L4", c0, t0, P_FPS,
                                         dict(hand=HAND_ZH[h], key=pitch_name(p)), "choice", "A" if still else "B",
                                         options=dict(choices=["按下状态", "松开状态"], letter="A" if still else "B"),
                                         horizon=H, difficulty=dict(base, horizon=H, held_so_far_s=round((t0 - P.on[n]) / self.fps, 3),
                                                                     repressed=bool(still and P.off[n] <= tH)), **kw))
        tH = t0 + self.fr(0.5)
        k = self.fr(0.1)
        if P.visible(h, tH - k, tH + k + 1, 1.0) and not self.has_jump(h, t0 - self.fr(0.3), tH + k + 1):
            x_now = P.palm_past(h, t0, self.fr(0.1))
            d = P.palm_med(h, tH, self.fr(0.05)) - x_now
            if abs(abs(d) - 1.0) >= 0.3:
                opts = ["向左", "净位移不足 1 个白键宽", "向右"]
                lab = opts[1] if abs(d) < 1 else ("向右" if d > 0 else "向左")
                v_hist = (x_now - P.palm_past(h, t0 - self.fr(0.2), 3)) / 0.2   # causal velocity
                geo = [self.geo_atom("g_now", self.grid(c0, t0, P_FPS, True)[0], t0, h, "此刻手掌中心位于", x_now)]
                out.append(self.fact("T5", "手掌横移方向", "L2", c0, t0, P_FPS, dict(hand=HAND_ZH[h]), "choice",
                                     LETTERS[opts.index(lab)], options=dict(choices=opts, letter=LETTERS[opts.index(lab)]),
                                     horizon=0.5, extra_atoms=geo, difficulty=dict(base, horizon=0.5, dx=round(d, 2),
                                                                                   const_vel_guess=round(float(v_hist * 0.5), 2)), **kw))
        # next press: group on the full future timeline first, then require it to start inside the horizon
        all_groups = P.groups(np.flatnonzero(P.hand == h), CHORD_TOL_S)
        gs = [g for g in all_groups if any(P.on[x] > t0 for x in g)]
        if gs and all(P.on[x] > t0 for x in gs[0]) and len(gs[0]) == 1 and t0 + self.fr(0.03) < P.on[gs[0][0]] <= tH:
            n = gs[0][0]
            true = int(P.pitch[n])
            palm = P.palm_past(h, t0, self.fr(0.1))
            hist_groups = P.groups(hist, CHORD_TOL_S)
            last = int(P.pitch[hist_groups[-1][0]]) if hist_groups and len(hist_groups[-1]) == 1 else None
            use_last = bool(last is not None and last != true and abs(KEY_X[last] - palm) <= 5 and self.rnd.random() < 0.5)
            reach = [p for p in range(21, 109) if p not in (true, last) and abs(KEY_X[p] - palm) <= 5]
            self.rnd.shuffle(reach)
            pool = ([last] if use_last else []) + reach[:3 - use_last]
            if len(pool) == 3 and abs(KEY_X[true] - palm) <= 5:
                opts = self.mc(pitch_name(true), [pitch_name(p) for p in pool])
                diff = dict(base, horizon=0.5, repeat_of_last=last == true, last_as_distractor=use_last,
                            true_dist_from_palm=round(abs(KEY_X[true] - palm), 2))
                out.append(self.fact("T5", "下一个键（选择）", "L4", c0, t0, P_FPS, dict(hand=HAND_ZH[h]), "choice",
                                     opts["letter"], options=opts, horizon=0.5, difficulty=diff, **kw))
                out.append(self.fact("T5", "下一个键（填空）", "L4", c0, t0, P_FPS, dict(hand=HAND_ZH[h]), "key",
                                     pitch_name(true), horizon=0.5, difficulty=diff, **kw))
            if P.finger[n] >= 0:
                k = int(P.finger[n])
                out.append(self.fact("T5", "下一根手指", "L5", c0, t0, P_FPS, dict(hand=HAND_ZH[h]), "choice", LETTERS[k],
                                     options=dict(choices=list(FINGER_ZH), letter=LETTERS[k]), horizon=0.5,
                                     difficulty=dict(base, horizon=0.5), **kw))
        return out
