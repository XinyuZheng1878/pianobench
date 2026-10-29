"""Turn extracted states (hand position, fingering, true release) into VQA samples.

Every sample refers to video frames ({"video", "frame"}) and carries `answer_struct`, a
machine-checkable version of the answer, plus `state_t0` / `state_t1` hand states where relevant so
the same data can supervise the latent-state (LS) decoder.

Only notes with status == ok are used as facts. A time window is used for questions about what
happened in it only if no uncertain note starts or is plausibly still sounding inside it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .events import group_events
from .keys import pitch_name, white_pitch_at
from .preprocess import clean_tips
from .vqa_templates import FINGER, HAND, PHRASE, T


@dataclass
class VQAConfig:
    per_type: int = 3            # samples per task type per video
    gap_min_s: float = 0.5
    gap_max_s: float = 3.0
    max_seq_events: int = 8
    max_tries: int = 40          # per requested sample
    chord_tol_s: float = 0.04


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


class Ctx:
    def __init__(self, perf, res, fcfg: Config | None = None):
        self.perf = perf
        self.fps = perf.fps
        self.tips, self.ok = clean_tips(perf.tips, perf.valid, fcfg or Config())
        self.T = self.ok.shape[0]
        self.onset_s = res["onset"]
        self.on = res["onset_frame"]
        self.off = res["offset_frame"]
        self.rel = res["release_frame"]
        self.hand = res["hand_pred"]
        self.finger = res["finger_pred"]
        self.pitch = res["pitch"]
        self.conf = res["conf"]
        self.good = res["status"] == 0
        # uncertain notes: assume they may sound for up to 1 s (MIDI offsets include pedal)
        self.bad_end = np.minimum(self.off, self.on + int(round(self.fps)))
        self.video = perf.video_path or perf.video_id

    def ref(self, t):
        return {"video": self.video, "frame": int(self.perf.frame_offset + t)}

    def hand_ok(self, h, t):
        return 0 <= t < self.T and bool(self.ok[t, h])

    def clean(self, t0, t1):
        return not (~self.good & (self.on <= t1) & (self.bad_end >= t0)).any()

    def center(self, h, t):
        return float(np.mean(self.tips[t, h, :, 0]))

    def span(self, h, t):
        x = self.tips[t, h, :, 0]
        return float(x.min()), float(x.max())

    def notes_in(self, h, t0, t1):
        return np.flatnonzero(self.good & (self.hand == h) & (self.on > t0) & (self.on <= t1))

    def active(self, t):
        return np.flatnonzero(self.good & (self.on <= t) & (self.rel > t))

    def state(self, t):
        out = {}
        act = self.active(t)
        for h, name in ((0, "L"), (1, "R")):
            if not self.hand_ok(h, t):
                continue
            lo, hi = self.span(h, t)
            out[name] = dict(
                center_x=round(self.center(h, t), 3), lo_x=round(lo, 3), hi_x=round(hi, 3),
                pressed=[[int(self.finger[n]) + 1, pitch_name(self.pitch[n])] for n in act if self.hand[n] == h],
            )
        return out


# ----------------------------------------------------------------------------- formatting

def _range_text(ctx, h, t, lang):
    lo, hi = ctx.span(h, t)
    P = PHRASE[lang]
    return P["range"].format(lo=pitch_name(white_pitch_at(lo)), hi=pitch_name(white_pitch_at(hi)),
                             mid=pitch_name(white_pitch_at(ctx.center(h, t))))


def _events(ctx, idx, cfg):
    evs = group_events(ctx.onset_s, list(idx), cfg.chord_tol_s)
    return [sorted(ev, key=lambda n: ctx.pitch[n]) for ev in evs]


def _fmt_keys(ctx, evs, lang):
    parts = []
    for ev in evs:
        k = "+".join(pitch_name(ctx.pitch[n]) for n in ev)
        parts.append(k if len(ev) == 1 else f"[{k}]")
    return PHRASE[lang]["arrow"].join(parts)


def _fmt_fingers(ctx, evs, lang):
    parts = []
    for ev in evs:
        f = "+".join(FINGER[lang][ctx.finger[n]] for n in ev)
        parts.append(f if len(ev) == 1 else f"[{f}]")
    return PHRASE[lang]["arrow"].join(parts)


def _fmt_seq(ctx, evs, lang):
    lp, rp = ("（", "）") if lang == "zh" else (" (", ")")
    parts = []
    for ev in evs:
        k = "+".join(pitch_name(ctx.pitch[n]) for n in ev)
        f = "+".join(FINGER[lang][ctx.finger[n]] for n in ev)
        parts.append(f"{k}{lp}{f}{rp}" if len(ev) == 1 else f"[{k}]{lp}{f}{rp}")
    return PHRASE[lang]["arrow"].join(parts)


def _struct_events(ctx, evs):
    return [[[pitch_name(ctx.pitch[n]), int(ctx.finger[n]) + 1] for n in ev] for ev in evs]


def _shift(ctx, h, t0, t1, lang):
    d = ctx.center(h, t1) - ctx.center(h, t0)
    P = PHRASE[lang]
    n = int(round(abs(d)))
    if n == 0:
        return d, P["shift_none"], "none"
    return d, (P["shift_right"] if d > 0 else P["shift_left"]).format(n=n), "right" if d > 0 else "left"


def _qa(task, lang, rng, **kw):
    q, a = T[task][lang][rng.integers(len(T[task][lang]))]
    return _cap(q.format(**kw)), _cap(a.format(**kw))


def _rec(task, lang, images, q, a, struct, **extra):
    r = dict(type=task, lang=lang, images=images, question=q, answer=a, answer_struct=struct)
    r.update(extra)
    return r


def _mc(rng, lang, question, correct, distractors):
    opts = [correct] + [d for d in dict.fromkeys(distractors) if d != correct][:3]
    order = rng.permutation(len(opts))
    opts = [opts[i] for i in order]
    letters = "ABCD"[: len(opts)]
    k = opts.index(correct)
    q = question + "\n" + PHRASE[lang]["options"] + " " + "  ".join(f"{l}. {o}" for l, o in zip(letters, opts))
    return q, PHRASE[lang]["mc_answer"].format(letter=letters[k], text=correct), letters[k], opts


# ----------------------------------------------------------------------------- samplers

def _sample_window(ctx, rng, cfg, hands=(0, 1)):
    gap = int(rng.uniform(cfg.gap_min_s, cfg.gap_max_s) * ctx.fps)
    good = np.flatnonzero(ctx.good)
    if len(good) and rng.random() < 0.7:     # anchor on a note so most windows contain playing
        t0 = int(ctx.on[rng.choice(good)]) - int(rng.integers(1, max(2, gap)))
    else:
        t0 = int(rng.integers(0, max(1, ctx.T - gap - 1)))
    t1 = t0 + gap
    if t0 < 0 or t1 >= ctx.T or not all(ctx.hand_ok(h, t0) and ctx.hand_ok(h, t1) for h in hands):
        return None
    return t0, t1


def _pick_note(ctx, rng):
    good = np.flatnonzero(ctx.good)
    if not len(good):
        return None
    n = int(rng.choice(good))
    t = int(ctx.on[n]) + 1
    if not ctx.hand_ok(ctx.hand[n], t) or t >= ctx.rel[n]:
        t = int(ctx.on[n])
    if not ctx.hand_ok(ctx.hand[n], t):
        return None
    return n, t


# ----------------------------------------------------------------------------- tasks

def g_finger_of_note(ctx, rng, lang, cfg, mc=False):
    p = _pick_note(ctx, rng)
    if p is None:
        return None
    n, t = p
    h, f = int(ctx.hand[n]), int(ctx.finger[n])
    task = "finger_hand_of_note" if (not mc and rng.random() < 0.3) else "finger_of_note"
    kw = dict(hand=HAND[lang][h], key=pitch_name(ctx.pitch[n]), finger=FINGER[lang][f], fnum=f + 1)
    q, a = _qa(task, lang, rng, **kw)
    struct = dict(hand="LR"[h], finger=f + 1, key=kw["key"], midi=int(ctx.pitch[n]))
    if mc:
        dis = [FINGER[lang][i] for i in rng.permutation(5) if i != f]
        q, a, letter, opts = _mc(rng, lang, q, FINGER[lang][f], dis)
        struct["choice"] = letter
        return _rec("finger_of_note_mc", lang, [ctx.ref(t)], q, a, struct, choices=opts, conf=float(ctx.conf[n]))
    return _rec(task, lang, [ctx.ref(t)], q, a, struct, conf=float(ctx.conf[n]))


def g_key_of_finger(ctx, rng, lang, cfg):
    good = np.flatnonzero(ctx.good)
    if not len(good):
        return None
    n = int(rng.choice(good))
    h, f = int(ctx.hand[n]), int(ctx.finger[n])
    hold = int(ctx.rel[n] - ctx.on[n])
    t = int(ctx.on[n]) + int(rng.integers(0, max(1, min(hold, int(0.3 * ctx.fps)))))
    if not ctx.hand_ok(h, t) or not ctx.clean(t - 2, t):
        return None
    q, a = _qa("key_of_finger", lang, rng, hand=HAND[lang][h], finger=FINGER[lang][f], key=pitch_name(ctx.pitch[n]))
    return _rec("key_of_finger", lang, [ctx.ref(t)], q, a,
                dict(hand="LR"[h], finger=f + 1, key=pitch_name(ctx.pitch[n])), conf=float(ctx.conf[n]))


def g_sequence_between(ctx, rng, lang, cfg):
    h = int(rng.integers(2))
    w = _sample_window(ctx, rng, cfg, (h,))
    if w is None or not ctx.clean(*w):
        return None
    t0, t1 = w
    evs = _events(ctx, ctx.notes_in(h, t0, t1), cfg)
    if len(evs) > cfg.max_seq_events:
        return None
    seq = _fmt_seq(ctx, evs, lang) if evs else PHRASE[lang]["seq_empty"].format(hand=HAND[lang][h])
    q, a = _qa("sequence_between", lang, rng, hand=HAND[lang][h], seq=seq)
    return _rec("sequence_between", lang, [ctx.ref(t0), ctx.ref(t1)], q, a,
                dict(hand="LR"[h], events=_struct_events(ctx, evs)),
                state_t0=ctx.state(t0), state_t1=ctx.state(t1))


def g_count_between(ctx, rng, lang, cfg):
    h = int(rng.integers(3))                 # 2 = both hands
    hands = (0, 1) if h == 2 else (h,)
    w = _sample_window(ctx, rng, cfg, hands)
    if w is None or not ctx.clean(*w):
        return None
    t0, t1 = w
    cnt = int(sum(len(ctx.notes_in(x, t0, t1)) for x in hands))
    q, a = _qa("count_between", lang, rng, hand=HAND[lang][h], count=cnt)
    return _rec("count_between", lang, [ctx.ref(t0), ctx.ref(t1)], q, a, dict(hand="LRB"[h], count=cnt))


def g_hand_shift(ctx, rng, lang, cfg, mc=False):
    h = int(rng.integers(2))
    w = _sample_window(ctx, rng, cfg, (h,))
    if w is None:
        return None
    t0, t1 = w
    d, text, direction = _shift(ctx, h, t0, t1, lang)
    q, a = _qa("hand_shift", lang, rng, hand=HAND[lang][h], shift=text)
    struct = dict(hand="LR"[h], dx_white_keys=round(d, 2), direction=direction)
    if mc:
        P = PHRASE[lang]
        n = max(1, int(round(abs(d))))
        dis = [P["shift_none"], P["shift_right"].format(n=n), P["shift_left"].format(n=n),
               P["shift_right"].format(n=n + 3), P["shift_left"].format(n=n + 3)]
        rng.shuffle(dis)
        q, a, letter, opts = _mc(rng, lang, q, text, dis)
        struct["choice"] = letter
        return _rec("hand_shift_mc", lang, [ctx.ref(t0), ctx.ref(t1)], q, a, struct, choices=opts)
    return _rec("hand_shift", lang, [ctx.ref(t0), ctx.ref(t1)], q, a, struct)


def g_pressed_at(ctx, rng, lang, cfg):
    p = _pick_note(ctx, rng)
    if p is None:
        return None
    t = p[1] + int(rng.integers(0, max(1, int(0.2 * ctx.fps))))
    if not (ctx.hand_ok(0, t) and ctx.hand_ok(1, t)) or not ctx.clean(t - int(ctx.fps), t):
        return None
    P = PHRASE[lang]
    act = ctx.active(t)
    parts = []
    for h in (1, 0):
        items = [P["press_item"].format(finger=FINGER[lang][ctx.finger[n]], key=pitch_name(ctx.pitch[n]))
                 for n in sorted(act, key=lambda n: ctx.finger[n]) if ctx.hand[n] == h]
        parts.append(_cap(P["state"].format(hand=HAND[lang][h], items=P["item_sep"].join(items) or P["hand_none"])))
    q, a = _qa("pressed_at", lang, rng, state=P["state_sep"].join(parts))
    return _rec("pressed_at", lang, [ctx.ref(t)], q, a, ctx.state(t))


def g_hand_register(ctx, rng, lang, cfg):
    h = int(rng.integers(2))
    t = int(rng.integers(0, ctx.T))
    if not ctx.hand_ok(h, t):
        return None
    q, a = _qa("hand_register", lang, rng, hand=HAND[lang][h], range=_range_text(ctx, h, t, lang))
    lo, hi = ctx.span(h, t)
    return _rec("hand_register", lang, [ctx.ref(t)], q, a,
                dict(hand="LR"[h], lo_x=round(lo, 3), hi_x=round(hi, 3), center_x=round(ctx.center(h, t), 3)))


def g_predict_fingering(ctx, rng, lang, cfg):
    h = int(rng.integers(2))
    w = _sample_window(ctx, rng, cfg, (h,))
    if w is None or not ctx.clean(*w):
        return None
    t0, t1 = w
    evs = _events(ctx, ctx.notes_in(h, t0, t1), cfg)
    if not evs or len(evs) > cfg.max_seq_events:
        return None
    q, a = _qa("predict_fingering", lang, rng, hand=HAND[lang][h], keys=_fmt_keys(ctx, evs, lang),
               fingers=_fmt_fingers(ctx, evs, lang))
    return _rec("predict_fingering", lang, [ctx.ref(t0)], q, a,
                dict(hand="LR"[h], events=_struct_events(ctx, evs)),
                state_t0=ctx.state(t0), state_t1=ctx.state(t1), target_frame=ctx.ref(t1))


def g_predict_hand_pos(ctx, rng, lang, cfg):
    h = int(rng.integers(2))
    w = _sample_window(ctx, rng, cfg, (h,))
    if w is None or not ctx.clean(*w):
        return None
    t0, t1 = w
    evs = _events(ctx, ctx.notes_in(h, t0, t1), cfg)
    if not evs or len(evs) > cfg.max_seq_events:
        return None
    q, a = _qa("predict_hand_pos", lang, rng, hand=HAND[lang][h], keys=_fmt_keys(ctx, evs, lang),
               range=_range_text(ctx, h, t1, lang))
    lo, hi = ctx.span(h, t1)
    return _rec("predict_hand_pos", lang, [ctx.ref(t0)], q, a,
                dict(hand="LR"[h], lo_x=round(lo, 3), hi_x=round(hi, 3), center_x=round(ctx.center(h, t1), 3)),
                state_t0=ctx.state(t0), state_t1=ctx.state(t1), target_frame=ctx.ref(t1))


GENERATORS = {
    "finger_of_note": g_finger_of_note,
    "finger_of_note_mc": lambda c, r, l, cfg: g_finger_of_note(c, r, l, cfg, mc=True),
    "key_of_finger": g_key_of_finger,
    "sequence_between": g_sequence_between,
    "count_between": g_count_between,
    "hand_shift": g_hand_shift,
    "hand_shift_mc": lambda c, r, l, cfg: g_hand_shift(c, r, l, cfg, mc=True),
    "pressed_at": g_pressed_at,
    "hand_register": g_hand_register,
    "predict_fingering": g_predict_fingering,
    "predict_hand_pos": g_predict_hand_pos,
}


def generate_for_video(perf, res, rng, langs=("zh",), cfg: VQAConfig | None = None, types=None, fcfg=None):
    cfg = cfg or VQAConfig()
    ctx = Ctx(perf, res, fcfg)
    out, seen = [], set()
    for name in types or GENERATORS:
        fn = GENERATORS[name]
        made = tries = 0
        while made < cfg.per_type and tries < cfg.per_type * cfg.max_tries:
            tries += 1
            lang = langs[int(rng.integers(len(langs)))]
            rec = fn(ctx, rng, lang, cfg)
            if rec is None:
                continue
            key = (rec["type"], rec["lang"], tuple(i["frame"] for i in rec["images"]), rec["question"])
            if key in seen:
                continue
            seen.add(key)
            rec = dict(id=f"{perf.video_id}-{rec['type']}-{made:03d}-{rec['lang']}", video_id=perf.video_id, **rec)
            out.append(rec)
            made += 1
    return out
