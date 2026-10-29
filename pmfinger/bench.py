"""Step 5: Benchmark question generators (T1-T5 + control), on top of states.Derived.

Every generator returns a *fact* (one checked statement about one interval / cut / clip) or None.
`expand()` turns a fact into questions: one per interval-spec (whole / timestamp / frames /
boundary_frames) x QA type (单选 / 填空 / 描述). All questions of a fact share `fact_id`, so a human
confirms it once.

Time conventions (match the system prompt):
  clip time t = (video_frame - clip_start_frame) / fps, from 0; "第 a 秒到第 b 秒之间" = [a, b).
  An event belongs to an interval if its onset (key press) falls inside.
  Setting F frame k of the 8 fps sequence is at t = k / 8 (15 fps for prediction clips).
Interval ends are snapped to a 0.25 s grid so timestamps and 8 fps frame numbers coincide.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from .states import (AIR, DOWN, FINGER_ZH, HAND_ZH, HOLD, MOVE, STATE_ZH, UNKNOWN, UP, Derived,
                     chord_name, crossing_events, interval_name, melody_shape, octave_of_x, pitch_name)
from .keys import NAMES

SYSTEM_PROMPT = """你会看到一段从钢琴正上方俯拍的演奏视频，画面中是琴键和演奏者的双手。
约定：
1. 左手、右手指演奏者本人的左右手。
2. 手指编号：1 拇指，2 食指，3 中指，4 无名指，5 小指。
3. 键名用音名加八度编号，C4 是中央 C；黑键写作升号，如 G#3。
4. 时间以视频开头为 0 秒，帧号从 0 开始；“第 a 秒到第 b 秒之间”包含 a、不包含 b。
5. 一次“按键”指手指把一个键按下去一次，按下的时刻落在区间内才算。
6. 手指的运动状态只有五种：
   - 落下：手指正向下按键，即将把键按下（按下前约 0.1 秒内）；
   - 按键保持：手指把键按住不放的这段时间；
   - 抬起：手指刚松开键、离开琴键（松开后约 0.1 秒内）；
   - 移动：没有按键，手指沿键盘左右方向明显地平移，一眼能看出在挪动；
   - 空中保持：没有按键，手指悬在琴键上方，位置基本不变（只有轻微晃动）。
7. 最终答案必须写在花括号里，例如 {B}、{3}、{G#3}、{E4, F#4, G4}；花括号只用于最终答案。"""

F_FPS, P_FPS = 8, 15
GRID = 0.25
LETTERS = "ABCDE"
STATE_OPTS = ["落下", "抬起", "按键保持", "空中保持", "移动"]


@dataclass
class BenchConfig:
    edge_margin_s: float = 0.15
    min_track: float = 0.95
    range_types: tuple = ("whole", "timestamp", "frames", "boundary_frames")
    group_prob: float = 0.25        # chance a fact is emitted in all interval specs x QA types (diagnostics)
    describe_prob: float = 0.2      # chance a 填空 question is worded as 描述


# ============================================================================ helpers

def fmt_t(x):
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    return s + ".0" if "." not in s else s


def snap(x):
    return round(x / GRID) * GRID


class Sampler:
    def __init__(self, der: Derived, rng, cfg: BenchConfig):
        self.d, self.rng, self.cfg = der, rng, cfg

    def anchor_frame(self, h=None, need_finger=False):
        m = self.d.hand_trust.copy()
        if h is not None:
            m &= self.d.hand == h
        if need_finger:
            m &= self.d.finger_trust
        idx = np.flatnonzero(m)
        if not len(idx):
            return None
        return int(self.d.on[self.rng.choice(idx)])

    def interval(self, h, need_finger=False, need_release=False, length=(2.0, 4.0), hands=None,
                 anchor=None, relaxed=False):
        """Sample clip + interval around a trusted note of hand h (or around `anchor`). Returns dict or None.
        relaxed: skip the edge / trust checks (control questions that do not depend on labels)."""
        d, rng, fps = self.d, self.rng, self.d.fps
        t = anchor if anchor is not None else self.anchor_frame(h, need_finger)
        if t is None:
            return None
        Li = snap(rng.uniform(*length))
        p0 = snap(rng.uniform(0, 2.0))
        p1 = snap(rng.uniform(0, 2.0))
        L = min(10.0, p0 + Li + p1)
        ta = t - int(round(rng.uniform(0.2, Li - 0.2) * fps))
        c0 = ta - int(round(p0 * fps))
        c1 = c0 + int(round(L * fps))
        tb = ta + int(round(Li * fps))
        hands = hands if hands is not None else ((0, 1) if h is None else (h,))
        if c0 < 0 or c1 > d.T or not d.tracked(c0, c1, (0, 1), self.cfg.min_track):
            return None
        hs = (0, 1) if h is None else (h,)
        if relaxed:
            return dict(c0=c0, c1=c1, ta=ta, tb=tb, a=p0, b=p0 + Li, L=L)
        if not all(d.edge_quiet(ta, x, self.cfg.edge_margin_s) and d.edge_quiet(tb, x, self.cfg.edge_margin_s) for x in hs):
            return None
        for x in hs:
            if d.any_untrusted(ta, tb, x, need_finger, need_release):
                return None
        return dict(c0=c0, c1=c1, ta=ta, tb=tb, a=p0, b=p0 + Li, L=L)

    def clip(self, length=(6.0, 10.0)):
        d, fps = self.d, self.d.fps
        L = snap(self.rng.uniform(*length))
        t = self.anchor_frame()
        if t is None:
            return None
        c0 = t - int(round(self.rng.uniform(0.5, L - 0.5) * fps))
        c1 = c0 + int(round(L * fps))
        if c0 < 0 or c1 > d.T or not d.tracked(c0, c1, (0, 1), self.cfg.min_track):
            return None
        return dict(c0=c0, c1=c1, L=L)


def options_mc(rng, correct, distractors, fixed=None):
    """Return (options, letter). `fixed` keeps a canonical order (e.g. fingers 1-5)."""
    if fixed is not None:
        opts = list(fixed)
    else:
        ds = [x for x in dict.fromkeys(distractors) if x != correct]
        opts = [correct] + ds[:4]
        opts = [opts[i] for i in rng.permutation(len(opts))]
    return opts, LETTERS[opts.index(correct)]


def fact(task, subtype, level, der, span, body, *, mc=None, fill=None, depends=(), notes=(), extra=None):
    """span: dict with c0,c1 (+ a,b,ta,tb for interval questions)."""
    notes = [int(n) for n in notes]
    confs = [float(der.note_conf[n]) for n in notes] if notes else []
    tiers = sorted({str(der.label_tier[n]) for n in notes}) if notes else []
    f = dict(task=task, subtype=subtype, level=level, video_id=der.vid, span=span, body=body,
             mc=mc, fill=fill, depends=list(depends), k_notes=len(notes),
             min_conf=round(min(confs), 3) if confs else None,
             label_tier=("gold" if tiers == ["gold"] else "silver" if "bronze" not in tiers else "bronze") if tiers else "n/a")
    f["focus_frames"] = sorted({int(der.on[n]) + int(round(0.05 * der.fps)) for n in notes})[:8]
    if extra:
        f.update(extra)
    key = f"{task}|{subtype}|{der.vid}|{span.get('ta', span['c0'])}|{span.get('tb', span['c1'])}|{body}"
    f["fact_id"] = hashlib.md5(key.encode()).hexdigest()[:10]
    return f


def keys_text(ps):
    return "、".join(pitch_name(p) for p in ps)


def finger_opts():
    return [f"{i + 1} {FINGER_ZH[i]}" for i in range(5)]


# ============================================================================ T1 弹了什么

def t1_key_sequence(s: Sampler):
    d, rng = s.d, s.rng
    h = int(rng.integers(2))
    sp = s.interval(h, length=(2.0, 4.0))
    if not sp:
        return None
    evs = d.events(d.notes_of(h, sp["ta"], sp["tb"]))
    if not 2 <= len(evs) <= 6 or any(len(e) > 1 for e in evs):
        return None
    ps = [int(d.pitch[e[0]]) for e in evs]
    seq = [pitch_name(p) for p in ps]
    j = int(rng.integers(len(ps)))
    dis = []
    for k in (2, -2, 12, -12):
        q = list(ps)
        q[j] += k
        if 21 <= q[j] <= 108:
            dis.append(" → ".join(pitch_name(p) for p in q))
    if len(ps) >= 2:
        q = list(ps)
        q[0], q[1] = q[1], q[0]
        dis.append(" → ".join(pitch_name(p) for p in q))
    corr = " → ".join(seq)
    dis = [dis[i] for i in rng.permutation(len(dis)) if dis[i] != corr]
    opts, letter = options_mc(rng, corr, dis[:3])
    if len(opts) < 4:
        return None
    body = f"{HAND_ZH[h]}依次按了哪些键？"
    return fact("T1", "按键序列", "L4", d, sp, body,
                mc=dict(options=opts, answer=letter),
                fill=dict(answer=seq, answer_type="key_sequence",
                          instr="把键名按顺序写在花括号里，用逗号分隔，如 {E4, F#4, G4}。"),
                depends=["hand"], notes=[e[0] for e in evs], extra=dict(hand=h))


def t1_register(s: Sampler):
    d, rng = s.d, s.rng
    h = int(rng.integers(2))
    sp = s.interval(h, length=(2.0, 4.0))
    if not sp:
        return None
    idx = d.notes_of(h, sp["ta"], sp["tb"])
    if len(idx) < 3:
        return None
    octs = d.pitch[idx] // 12 - 1
    vals, cnt = np.unique(octs, return_counts=True)
    k = int(np.argmax(cnt))
    if cnt[k] / len(octs) < 0.75:
        return None
    o = int(vals[k])
    lo = int(np.clip(o - rng.integers(0, 4), 0, 5))
    opts_o = [x for x in range(lo, lo + 4)]
    if o not in opts_o:
        return None
    opts = [f"第 {x} 组（C{x}–B{x}）" for x in opts_o]
    body = f"{HAND_ZH[h]}按的键主要在哪个八度？"
    return fact("T1", "音区", "L2", d, sp, body,
                mc=dict(options=opts, answer=LETTERS[opts_o.index(o)]),
                fill=dict(answer=o, answer_type="int",
                          instr="用八度编号回答（C4 所在的八度编号为 4），把数字写在花括号里。"),
                depends=["hand"], notes=idx, extra=dict(hand=h))


MELODY_OPTS = ["上行", "下行", "先上行后下行", "先下行后上行"]
TEXTURE_OPTS = ["只有单音", "只有和弦或双音", "单音和和弦（双音）都有"]


def t1_melody(s: Sampler):
    d, rng = s.d, s.rng
    h = int(rng.integers(2))
    sp = s.interval(h, length=(2.0, 4.0))
    if not sp:
        return None
    evs = d.events(d.notes_of(h, sp["ta"], sp["tb"]))
    if len(evs) < 4 or any(len(e) > 1 for e in evs):
        return None
    shape = melody_shape([int(d.pitch[e[0]]) for e in evs])
    if shape is None:
        return None
    body = f"{HAND_ZH[h]}弹的旋律走向是怎样的？"
    return fact("T1", "旋律走向", "L1", d, sp, body,
                mc=dict(options=MELODY_OPTS, answer=LETTERS[MELODY_OPTS.index(shape)]),
                depends=["hand"], notes=[e[0] for e in evs], extra=dict(hand=h))


def t1_texture(s: Sampler):
    d, rng = s.d, s.rng
    h = int(rng.integers(2))
    sp = s.interval(h, length=(2.0, 4.0))
    if not sp:
        return None
    evs = d.events(d.notes_of(h, sp["ta"], sp["tb"]))
    if len(evs) < 3:
        return None
    fr = np.mean([len(e) >= 2 for e in evs])
    ans = TEXTURE_OPTS[0] if fr == 0 else TEXTURE_OPTS[1] if fr == 1 else TEXTURE_OPTS[2] if 0.25 <= fr <= 0.75 else None
    if ans is None:
        return None
    body = f"{HAND_ZH[h]}弹的是单音还是和弦？"
    return fact("T1", "织体", "L1", d, sp, body, mc=dict(options=TEXTURE_OPTS, answer=LETTERS[TEXTURE_OPTS.index(ans)]),
                depends=["hand"], notes=[n for e in evs for n in e], extra=dict(hand=h))


def t1_chord(s: Sampler):
    d, rng = s.d, s.rng
    h = int(rng.integers(2))
    sp = s.interval(h, length=(1.5, 3.0))
    if not sp:
        return None
    evs = d.events(d.notes_of(h, sp["ta"], sp["tb"]))
    big = [e for e in evs if len(e) >= 2]
    if len(big) != 1:
        return None
    e = big[0]
    ps = [int(d.pitch[n]) for n in e]
    if len(e) == 2:
        name = interval_name(*ps)
        if name is None or name == "纯一度":
            return None
        from .states import INTERVALS
        i = INTERVALS.index(name)
        dis = [INTERVALS[j] for j in (i - 1, i + 1, i - 2, i + 2) if 1 <= j <= 12]
        opts, letter = options_mc(rng, name, dis[:3])
        body = f"{HAND_ZH[h]}同时按下的两个键构成什么音程？"
        return fact("T1", "音程", "L4", d, sp, body, mc=dict(options=opts, answer=letter),
                    fill=dict(answer=name, answer_type="interval_name",
                              instr="用“小三度”“纯五度”这样的音程名回答，写在花括号里。"),
                    depends=["hand"], notes=e, extra=dict(hand=h))
    name = chord_name(ps)
    if name is None:
        return None
    root, q = name.split(" ")
    ri = NAMES.index(root)
    others = ["大三和弦", "小三和弦", "减三和弦", "增三和弦"] if "七" not in q else ["属七和弦", "大七和弦", "小七和弦", "半减七和弦"]
    dis = [f"{root} {x}" for x in others if x != q][:2] + [f"{NAMES[(ri + k) % 12]} {q}" for k in (7, 5)]
    opts, letter = options_mc(rng, name, dis)
    body = f"{HAND_ZH[h]}按下的和弦是什么？"
    return fact("T1", "和弦", "L4", d, sp, body, mc=dict(options=opts, answer=letter),
                fill=dict(answer=name, answer_type="chord",
                          instr="用“根音 + 和弦性质”回答，如 {C 大三和弦}、{G 属七和弦}。"),
                depends=["hand"], notes=e, extra=dict(hand=h))


# ============================================================================ T2 谁弹的

def _unique_key(d, sp, h=None, need_finger=False):
    idx = np.flatnonzero((d.on >= sp["ta"]) & (d.on < sp["tb"]))
    if h is not None:
        cand = [n for n in idx if d.hand[n] == h]
    else:
        cand = list(idx)
    if need_finger:
        cand = [n for n in cand if d.finger_trust[n]]
    cand = [n for n in cand if (d.pitch[idx] == d.pitch[n]).sum() == 1]
    return cand


def t2_which_hand(s: Sampler):
    d, rng = s.d, s.rng
    sp = s.interval(None, length=(2.0, 4.0))
    if not sp:
        return None
    cand = _unique_key(d, sp)
    if not cand:
        return None
    n = int(rng.choice(cand))
    h = int(d.hand[n])
    body = f"{pitch_name(d.pitch[n])} 是哪只手按的？"
    return fact("T2", "哪只手", "L2", d, sp, body, mc=dict(options=["左手", "右手"], answer=LETTERS[h]),
                depends=["hand"], notes=[n], extra=dict(hand=h, key=pitch_name(d.pitch[n])))


def t2_which_finger(s: Sampler):
    d, rng = s.d, s.rng
    h = int(rng.integers(2))
    sp = s.interval(h, need_finger=True, length=(2.0, 4.0))
    if not sp:
        return None
    cand = _unique_key(d, sp, h, need_finger=True)
    if not cand:
        return None
    n = int(rng.choice(cand))
    f = int(d.finger[n])
    body = f"{pitch_name(d.pitch[n])} 是用{HAND_ZH[h]}哪根手指按的？"
    return fact("T2", "哪根手指", "L5", d, sp, body, mc=dict(options=finger_opts(), answer=LETTERS[f]),
                fill=dict(answer=f + 1, answer_type="int", instr="用手指编号（1–5）回答，写在花括号里。"),
                depends=["hand", "finger"], notes=[n],
                extra=dict(hand=h, finger=f, key=pitch_name(d.pitch[n]), black_key=bool(d.pitch[n] % 12 in (1, 3, 6, 8, 10))))


CROSS_OPTS = ["有穿指（拇指从其他手指下方穿过）", "有跨指（其他手指从拇指上方跨过）", "穿指和跨指都有", "都没有"]


def t2_crossing(s: Sampler):
    d, rng = s.d, s.rng
    h = int(rng.integers(2))
    anchor = None
    if rng.random() < 0.6:   # crossings are rare: anchor half the samples on one
        cr = crossing_events(d, np.flatnonzero((d.hand == h) & d.finger_trust))
        if cr:
            anchor = int(d.on[cr[int(rng.integers(len(cr)))][2]])
    sp = s.interval(h, need_finger=True, length=(2.0, 4.0), anchor=anchor)
    if not sp:
        return None
    idx = d.notes_of(h, sp["ta"], sp["tb"])
    evs = d.events(idx)
    if len(evs) < 4 or sum(len(e) == 1 for e in evs) < 4:
        return None
    kinds = {k for k, _, _ in crossing_events(d, idx)}
    ans = CROSS_OPTS[2] if len(kinds) == 2 else CROSS_OPTS[0] if "穿指" in kinds else CROSS_OPTS[1] if "跨指" in kinds else CROSS_OPTS[3]
    body = f"{HAND_ZH[h]}在弹奏中有没有穿指或跨指？"
    return fact("T2", "穿指跨指", "L5", d, sp, body, mc=dict(options=CROSS_OPTS, answer=LETTERS[CROSS_OPTS.index(ans)]),
                depends=["hand", "finger"], notes=idx, extra=dict(hand=h))


# ============================================================================ T3 怎么弹的

def _num_opts(rng, n, lo=0):
    c = sorted({n, *[n + k for k in (-2, -1, 1, 2) if n + k >= lo]})
    pick = [n] + [x for x in rng.permutation([x for x in c if x != n])][:3]
    pick = sorted(int(x) for x in pick)
    return [str(x) for x in pick], LETTERS[pick.index(n)]


def t3_count_hand(s: Sampler):
    d, rng = s.d, s.rng
    h = int(rng.integers(2))
    sp = s.interval(h, length=(1.5, 3.5))
    if not sp:
        return None
    idx = d.notes_of(h, sp["ta"], sp["tb"])
    n = len(idx)
    if not 1 <= n <= 12:
        return None
    opts, letter = _num_opts(rng, n, 0)
    body = f"{HAND_ZH[h]}一共按了几次键（同时按下的几个键分别计数）？"
    return fact("T3", "按键次数（手）", "L2", d, sp, body, mc=dict(options=opts, answer=letter),
                fill=dict(answer=n, answer_type="int", instr="把次数写在花括号里。"),
                depends=["hand"], notes=idx, extra=dict(hand=h))


def t3_count_finger(s: Sampler):
    d, rng = s.d, s.rng
    h = int(rng.integers(2))
    f = int(rng.integers(5))
    sp = s.interval(h, need_finger=True, length=(1.5, 3.5))
    if not sp:
        return None
    idx = d.notes_of(h, sp["ta"], sp["tb"])
    n = int((d.finger[idx] == f).sum())
    if n > 6 or len(idx) == 0:
        return None
    opts, letter = _num_opts(rng, n, 0)
    body = f"{HAND_ZH[h]}{FINGER_ZH[f]}一共按了几次键？"
    return fact("T3", "按键次数（手指）", "L5", d, sp, body, mc=dict(options=opts, answer=letter),
                fill=dict(answer=n, answer_type="int", instr="把次数写在花括号里。"),
                depends=["hand", "finger"], notes=idx, extra=dict(hand=h, finger=f))


def t3_state_sequence(s: Sampler):
    d, rng = s.d, s.rng
    h = int(rng.integers(2))
    sp = s.interval(h, need_finger=True, need_release=True, length=(1.0, 2.5))
    if not sp:
        return None
    idx = d.notes_of(h, sp["ta"] - int(d.fps), sp["tb"] + int(d.fps))
    fs = sorted({int(d.finger[n]) for n in idx}) or list(range(5))
    f = int(rng.choice(fs))
    runs_ = d.state_runs(sp["ta"], sp["tb"], h, f)
    if any(r[0] == UNKNOWN for r in runs_):
        return None
    mn = int(round(0.05 * d.fps))
    runs_ = [r for r in runs_ if r[2] - r[1] >= mn]
    if not all(d.motion_clear(r[0], r[1], r[2], h, f) for r in runs_):
        return None  # a MOVE / AIR stretch whose speed is near the threshold: not clearly visible
    seq = []
    for r in runs_:
        nm = STATE_ZH[r[0]]
        if not seq or seq[-1] != nm:
            seq.append(nm)
    if not 2 <= len(seq) <= 6 or "落下" not in seq and "抬起" not in seq:
        return None
    first, last = d.state_runs(sp["ta"], sp["tb"], h, f)[0], d.state_runs(sp["ta"], sp["tb"], h, f)[-1]
    if min(first[2] - first[1], last[2] - last[1]) < int(0.1 * d.fps):
        return None
    body = f"{HAND_ZH[h]}{FINGER_ZH[f]}依次经历了哪些运动状态？"
    return fact("T3", "运动状态序列", "L3", d, sp, body,
                fill=dict(answer=seq, answer_type="state_sequence",
                          instr="从“落下、抬起、按键保持、空中保持、移动”中选词，按时间顺序写在花括号里，用逗号分隔，连续相同的状态只写一次。"),
                depends=["hand", "finger", "release"], notes=idx, extra=dict(hand=h, finger=f))


HOLD_BINS = [(0, 0.25, "不到 0.25 秒"), (0.25, 0.5, "0.25 到 0.5 秒"), (0.5, 1.0, "0.5 到 1 秒"), (1.0, 99, "超过 1 秒")]


def t3_hold(s: Sampler):
    d, rng = s.d, s.rng
    h = int(rng.integers(2))
    sp = s.interval(h, need_release=True, length=(2.0, 4.0))
    if not sp:
        return None
    cand = [n for n in _unique_key(d, sp, h) if d.rel[n] < sp["c1"] and d.release_trust[n]]
    if not cand:
        return None
    n = int(rng.choice(cand))
    dur = (d.rel[n] - d.on[n]) / d.fps
    lab = None
    for lo, hi, name in HOLD_BINS:
        if lo <= dur < hi:
            lab = name
            if min(abs(dur - lo), abs(hi - dur)) < 0.15 * max(lo, 0.25):
                return None
    opts = [x[2] for x in HOLD_BINS]
    body = f"{HAND_ZH[h]}按下 {pitch_name(d.pitch[n])} 后按住了多久？"
    return fact("T3", "保持时长", "L3", d, sp, body, mc=dict(options=opts, answer=LETTERS[opts.index(lab)]),
                depends=["hand", "release"], notes=[n], extra=dict(hand=h, hold_s=round(dur, 3)))


SPAN_BINS = [(0, 3, "不到 3 个白键"), (3, 5, "3 到 5 个白键"), (5, 7, "5 到 7 个白键"), (7, 99, "7 个白键以上")]


def t3_span(s: Sampler):
    d, rng = s.d, s.rng
    h = int(rng.integers(2))
    sp = s.interval(h, length=(1.5, 3.0))
    if not sp:
        return None
    v = np.nanmax(d.span[sp["ta"]:sp["tb"], h])
    if not np.isfinite(v):
        return None
    lab = None
    for lo, hi, name in SPAN_BINS:
        if lo <= v < hi:
            lab = name
            if min(abs(v - lo), abs(hi - v)) < 0.4 and lo > 0:
                return None
    opts = [x[2] for x in SPAN_BINS]
    body = f"{HAND_ZH[h]}张得最开的时候，拇指和小指的指尖之间大约隔了多宽？"
    return fact("T3", "张开跨度", "L3", d, sp, body, mc=dict(options=opts, answer=LETTERS[opts.index(lab)]),
                depends=["pose"], extra=dict(hand=h, span_white=round(float(v), 2)))


MOVE_OPTS = ["向左移动超过 5 个白键", "向左移动 1.5 到 5 个白键", "基本没动（不到 1 个白键）",
             "向右移动 1.5 到 5 个白键", "向右移动超过 5 个白键"]


def _move_label(dx):
    a = abs(dx)
    if a < 1.0:
        return MOVE_OPTS[2]
    if 1.5 <= a < 4.5:
        return MOVE_OPTS[1] if dx < 0 else MOVE_OPTS[3]
    if a >= 5.5:
        return MOVE_OPTS[0] if dx < 0 else MOVE_OPTS[4]
    return None


def t3_hand_move(s: Sampler):
    d, rng = s.d, s.rng
    h = int(rng.integers(2))
    sp = s.interval(h, length=(1.0, 3.0))
    if not sp:
        return None
    k = 2
    x0 = np.nanmean(d.center[sp["ta"] - k:sp["ta"] + k + 1, h])
    x1 = np.nanmean(d.center[sp["tb"] - k:sp["tb"] + k + 1, h])
    if not (np.isfinite(x0) and np.isfinite(x1)):
        return None
    lab = _move_label(x1 - x0)
    if lab is None:
        return None
    body = f"从区间开始到结束，{HAND_ZH[h]}整体往哪边移动、移动了多远（左 = 低音方向）？"
    return fact("T3", "手的移动", "L2", d, sp, body, mc=dict(options=MOVE_OPTS, answer=LETTERS[MOVE_OPTS.index(lab)]),
                depends=["pose"], extra=dict(hand=h, dx_white=round(float(x1 - x0), 2)))


# ============================================================================ T4 什么时候弹的

def _t(d, c0, frame):
    return round((frame - c0) / d.fps, 3)


def t4_melody(s: Sampler):
    d, rng = s.d, s.rng
    sp = s.clip()
    if not sp:
        return None
    h = int(rng.integers(2))
    if d.any_untrusted(sp["c0"], sp["c1"], h):
        return None
    evs = d.events(d.notes_of(h, sp["c0"], sp["c1"]))
    k = int(rng.integers(3, 5))
    starts = [i for i in range(len(evs) - k + 1) if all(len(evs[i + j]) == 1 for j in range(k))]
    if not starts:
        return None
    i = int(rng.choice(starts))
    run = [evs[i + j][0] for j in range(k)]
    ps = tuple(int(d.pitch[n]) for n in run)
    seqs = [tuple(int(d.pitch[evs[i2 + j][0]]) for j in range(k)) for i2 in range(len(evs) - k + 1)
            if all(len(evs[i2 + j]) == 1 for j in range(k))]
    if seqs.count(ps) != 1 or len(set(ps)) < 2:
        return None
    t0, t1 = _t(d, sp["c0"], d.on[run[0]]), _t(d, sp["c0"], d.rel[run[-1]])
    if t1 > sp["L"] - 0.1 or t1 - t0 > 3.0:
        return None
    body = f"{HAND_ZH[h]}依次弹 {keys_text(ps)} 的那一段是第几秒到第几秒？"
    return fact("T4", "旋律定位", "L4", d, sp, body,
                fill=dict(answer=[t0, t1], answer_type="time_span",
                          instr="把起止时间（秒）写在花括号里，用逗号分隔，如 {1.5, 2.3}。"),
                depends=["hand"], notes=run, extra=dict(hand=h))


def t4_chord(s: Sampler):
    d, rng = s.d, s.rng
    sp = s.clip()
    if not sp:
        return None
    h = int(rng.integers(2))
    if d.any_untrusted(sp["c0"], sp["c1"], h):
        return None
    evs = d.events(d.notes_of(h, sp["c0"], sp["c1"]))
    big = [e for e in evs if len(e) >= 3]
    if len(big) != 1 or len(evs) < 3:
        return None
    t0 = _t(d, sp["c0"], d.on[big[0][0]])
    if not 0.3 < t0 < sp["L"] - 0.3:
        return None
    body = f"这段视频里{HAND_ZH[h]}只弹了一次三个或更多键同时按下的和弦，是在第几秒？"
    return fact("T4", "和弦定位", "L1", d, sp, body,
                fill=dict(answer=t0, answer_type="time_point", instr="把时间（秒）写在花括号里，如 {3.2}。",
                          frame_ok=True),
                depends=["hand"], notes=big[0], extra=dict(hand=h))


def t4_register(s: Sampler):
    d, rng = s.d, s.rng
    sp = s.clip()
    if not sp:
        return None
    h = int(rng.integers(2))
    c = d.center[sp["c0"]:sp["c1"], h]
    if not np.isfinite(c).all():
        return None
    octs = np.array([octave_of_x(x) for x in c])
    w = int(0.5 * d.fps)
    ch = [k for k in range(w, len(octs) - w) if octs[k] != octs[k - 1]
          and (octs[k - w:k] == octs[k - 1]).all() and (octs[k:k + w] == octs[k]).all()]
    if not ch:
        return None
    k = ch[0]
    o = int(octs[k])
    if (octs[:k] == o).any():
        return None
    body = f"{HAND_ZH[h]}的手掌中心第一次移到第 {o} 组（C{o}–B{o}）上方是在第几秒？"
    return fact("T4", "音区定位", "L2", d, sp, body,
                fill=dict(answer=round(k / d.fps, 3), answer_type="time_point",
                          instr="把时间（秒）写在花括号里，如 {3.2}。", frame_ok=True),
                depends=["pose"], extra=dict(hand=h, octave=o))


def t4_move(s: Sampler):
    d, rng = s.d, s.rng
    sp = s.clip()
    if not sp:
        return None
    h = int(rng.integers(2))
    c = d.center[sp["c0"]:sp["c1"], h]
    if not np.isfinite(c).all():
        return None
    v = np.gradient(c) * d.fps
    fast = np.abs(v) > 3.0
    from .preprocess import runs as _runs
    segs = []
    for a, b in zip(*_runs(fast)):
        dx = c[min(b, len(c) - 1)] - c[a]
        if abs(dx) >= 4.0:
            segs.append((a, b, dx))
    if len(segs) != 1:
        return None
    a, b, dx = segs[0]
    if a < int(0.3 * d.fps) or b > len(c) - int(0.3 * d.fps) or (b - a) / d.fps < 0.1:
        return None
    dirn = "左" if dx < 0 else "右"
    body = f"{HAND_ZH[h]}有一次向{dirn}大幅移动（超过 4 个白键），是第几秒到第几秒？"
    return fact("T4", "动作定位", "L3", d, sp, body,
                fill=dict(answer=[round(a / d.fps, 3), round(b / d.fps, 3)], answer_type="time_span",
                          instr="把起止时间（秒）写在花括号里，用逗号分隔，如 {1.5, 2.3}。"),
                depends=["pose"], extra=dict(hand=h, dx_white=round(float(dx), 2)))


def t4_fingering(s: Sampler):
    d, rng = s.d, s.rng
    sp = s.clip()
    if not sp:
        return None
    h = int(rng.integers(2))
    if d.any_untrusted(sp["c0"], sp["c1"], h, need_finger=True):
        return None
    idx = d.notes_of(h, sp["c0"], sp["c1"])
    cands = []
    for n in idx:
        same = [m for m in idx if d.pitch[m] == d.pitch[n]]
        if len(same) >= 2 and sum(d.finger[m] == d.finger[n] for m in same) == 1:
            cands.append(n)
    if not cands:
        return None
    n = int(rng.choice(cands))
    t0 = _t(d, sp["c0"], d.on[n])
    if not 0.3 < t0 < sp["L"] - 0.3:
        return None
    body = f"{pitch_name(d.pitch[n])} 被{HAND_ZH[h]}按了不止一次，其中用{FINGER_ZH[int(d.finger[n])]}按的那一次是在第几秒？"
    return fact("T4", "指法定位", "L5", d, sp, body,
                fill=dict(answer=t0, answer_type="time_point", instr="把时间（秒）写在花括号里，如 {3.2}。", frame_ok=True),
                depends=["hand", "finger"], notes=[m for m in idx if d.pitch[m] == d.pitch[n]],
                extra=dict(hand=h, finger=int(d.finger[n])))


# ============================================================================ T5 接下来弹什么

def _cut(s: Sampler, h, H, need_finger=False):
    d, rng, fps = s.d, s.rng, s.d.fps
    t = s.anchor_frame(h, need_finger)
    if t is None:
        return None
    t = t + int(round(rng.uniform(-1.0, 1.0) * fps))
    c0 = t - 2 * int(fps)
    tH = t + int(round(H * fps))
    if c0 < 0 or tH + int(0.6 * fps) >= d.T or not d.tracked(c0, tH + 1, (0, 1), s.cfg.min_track):
        return None
    if d.any_untrusted(t - int(0.5 * fps), tH + int(0.1 * fps), h, need_finger, need_release=need_finger):
        return None
    on_h = d.on[d.hand == h]
    if (np.abs(on_h - t) <= int(0.05 * fps)).any():
        return None
    return dict(c0=c0, c1=t, cut=t, tH=tH, L=2.0, H=H)


def t5_state(s: Sampler):
    d, rng = s.d, s.rng
    h = int(rng.integers(2))
    H = float(rng.choice([0.2, 0.5]))
    sp = _cut(s, h, H, need_finger=True)
    if not sp:
        return None
    f = int(rng.integers(5))
    k = int(0.05 * d.fps)
    w = d.state[sp["tH"] - k:sp["tH"] + k + 1, h, f]
    now = int(d.state[sp["cut"] - 1, h, f])
    if (w == UNKNOWN).any() or len(set(w.tolist())) != 1 or now == UNKNOWN:
        return None
    if not d.motion_clear(int(w[0]), sp["tH"] - k, sp["tH"] + k + 1, h, f):
        return None  # MOVE / AIR with a speed near the threshold: not clearly visible
    st = STATE_ZH[int(w[0])]
    body = f"{fmt_t(H)} 秒后，{HAND_ZH[h]}{FINGER_ZH[f]}处于哪种运动状态？"
    return fact("T5", "运动状态", "L3", d, sp, body, mc=dict(options=STATE_OPTS, answer=LETTERS[STATE_OPTS.index(st)]),
                depends=["hand", "finger", "release"],
                notes=d.notes_of(h, sp["cut"] - int(d.fps), sp["tH"] + int(d.fps)),
                extra=dict(hand=h, finger=f, horizon=H, baseline_copy=LETTERS[STATE_OPTS.index(STATE_ZH[now])]))


def t5_will_press(s: Sampler):
    d, rng = s.d, s.rng
    h = int(rng.integers(2))
    H = float(rng.choice([0.2, 0.5]))
    sp = _cut(s, h, H)
    if not sp:
        return None
    on_h = d.on[d.hand == h]
    if (np.abs(on_h - sp["tH"]) <= int(0.06 * d.fps)).any():
        return None
    will = bool(((on_h > sp["cut"]) & (on_h <= sp["tH"])).any())
    past = bool(((on_h > sp["cut"] - (sp["tH"] - sp["cut"])) & (on_h <= sp["cut"])).any())
    opts = ["会", "不会"]
    body = f"接下来 {fmt_t(H)} 秒内，{HAND_ZH[h]}会不会按下新的键？"
    return fact("T5", "会不会按键", "L2", d, sp, body, mc=dict(options=opts, answer="A" if will else "B"),
                depends=["hand"], extra=dict(hand=h, horizon=H, baseline_copy="A" if past else "B"))


DIR_OPTS = ["向左（低音方向）", "基本不动", "向右（高音方向）"]


def t5_move(s: Sampler):
    d, rng = s.d, s.rng
    h = int(rng.integers(2))
    H = 0.5
    sp = _cut(s, h, H)
    if not sp:
        return None
    x0 = np.nanmean(d.center[sp["cut"] - 3:sp["cut"], h])
    x1 = np.nanmean(d.center[sp["tH"] - 2:sp["tH"] + 3, h])
    if not (np.isfinite(x0) and np.isfinite(x1)):
        return None
    dx = x1 - x0
    lab = DIR_OPTS[0] if dx < -1.0 else DIR_OPTS[2] if dx > 1.0 else DIR_OPTS[1] if abs(dx) < 0.5 else None
    if lab is None:
        return None
    body = f"接下来 {fmt_t(H)} 秒内，{HAND_ZH[h]}整体会往哪边移动？"
    return fact("T5", "移动方向", "L2", d, sp, body, mc=dict(options=DIR_OPTS, answer=LETTERS[DIR_OPTS.index(lab)]),
                depends=["pose"], extra=dict(hand=h, horizon=H, dx_white=round(float(dx), 2), baseline_copy="B"))


def _next_note(d, sp, h, need_finger):
    on = d.on
    nxt = np.flatnonzero((d.hand == h) & (on > sp["cut"] + int(0.05 * d.fps)) & (on <= sp["tH"]))
    prev = np.flatnonzero((d.hand == h) & (on < sp["cut"]))
    if not len(nxt) or not len(prev):
        return None
    first = nxt[np.argmin(on[nxt])]
    same = nxt[np.abs(on[nxt] - on[first]) <= int(d.cfg.chord_tol_s * d.fps)]
    if len(same) != 1 or (need_finger and not d.finger_trust[first]):
        return None
    last = prev[np.argmax(on[prev])]
    return int(first), int(last)


def t5_next_key(s: Sampler):
    d, rng = s.d, s.rng
    h = int(rng.integers(2))
    sp = _cut(s, h, 0.5)
    if not sp:
        return None
    r = _next_note(d, sp, h, False)
    if not r:
        return None
    n, last = r
    p, q = int(d.pitch[n]), int(d.pitch[last])
    oct_ = p + 12 if p + 12 <= 108 else p - 12
    semi = p + 2 if rng.random() < 0.5 else p - 2
    opts_p = [p, q, semi, oct_]
    if len(set(opts_p)) < 4:
        return None
    names = [pitch_name(x) for x in opts_p]
    order = rng.permutation(4)
    opts = [names[i] for i in order]
    body = f"接下来 0.5 秒内，{HAND_ZH[h]}按下的下一个键是什么？"
    return fact("T5", "下一个键", "L4", d, sp, body, mc=dict(options=opts, answer=LETTERS[opts.index(names[0])]),
                depends=["hand"], notes=[n, last],
                extra=dict(hand=h, horizon=0.5, baseline_copy=LETTERS[opts.index(names[1])],
                           distractor_roles={LETTERS[opts.index(names[1])]: "刚弹的键",
                                             LETTERS[opts.index(names[2])]: "相差 2 半音",
                                             LETTERS[opts.index(names[3])]: "相差一个八度"}))


def t5_next_finger(s: Sampler):
    d, rng = s.d, s.rng
    h = int(rng.integers(2))
    sp = _cut(s, h, 0.5, need_finger=True)
    if not sp:
        return None
    r = _next_note(d, sp, h, True)
    if not r:
        return None
    n, last = r
    f = int(d.finger[n])
    body = f"接下来 0.5 秒内，{HAND_ZH[h]}按下一个键会用哪根手指？"
    return fact("T5", "下一根手指", "L5", d, sp, body, mc=dict(options=finger_opts(), answer=LETTERS[f]),
                depends=["hand", "finger"], notes=[n, last],
                extra=dict(hand=h, horizon=0.5, baseline_copy=LETTERS[int(d.finger[last])] if d.finger_trust[last] else None))


# ============================================================================ control

def c1_pedal(s: Sampler):
    d, rng = s.d, s.rng
    sp = s.interval(None, length=(2.0, 4.0), relaxed=True)
    if not sp:
        return None
    opts = ["踩了", "没踩", "无法从画面判断"]
    return fact("C1", "踏板", "-", d, sp, "演奏者有没有踩延音踏板？", mc=dict(options=opts, answer="C"))


GENERATORS = {
    ("T1", "按键序列"): t1_key_sequence, ("T1", "音区"): t1_register, ("T1", "旋律走向"): t1_melody,
    ("T1", "织体"): t1_texture, ("T1", "和弦与音程"): t1_chord,
    ("T2", "哪只手"): t2_which_hand, ("T2", "哪根手指"): t2_which_finger, ("T2", "穿指跨指"): t2_crossing,
    ("T3", "按键次数（手）"): t3_count_hand, ("T3", "按键次数（手指）"): t3_count_finger,
    ("T3", "运动状态序列"): t3_state_sequence, ("T3", "保持时长"): t3_hold, ("T3", "张开跨度"): t3_span,
    ("T3", "手的移动"): t3_hand_move,
    ("T4", "旋律定位"): t4_melody, ("T4", "和弦定位"): t4_chord, ("T4", "音区定位"): t4_register,
    ("T4", "动作定位"): t4_move, ("T4", "指法定位"): t4_fingering,
    ("T5", "运动状态"): t5_state, ("T5", "会不会按键"): t5_will_press, ("T5", "移动方向"): t5_move,
    ("T5", "下一个键"): t5_next_key, ("T5", "下一根手指"): t5_next_finger,
    ("C1", "踏板"): c1_pedal,
}


# ============================================================================ fact -> questions

def needs_review(f):
    return bool({"finger", "release"} & set(f["depends"]))


def answer_key(f):
    if f.get("mc"):
        return f["mc"]["answer"]
    a = f["fill"]["answer"]
    if f["fill"]["answer_type"] in ("time_point", "time_span"):
        return "t"
    return str(a)


def boundary_unique(der: Derived, c0, L, a, b):
    """Frames at clip times a and b must not look like any other 8 fps frame >= 0.375 s away."""
    fps = der.fps
    ks = np.arange(0, int(L * F_FPS))
    xs = np.array([der.tips[min(der.T - 1, c0 + int(round(k / F_FPS * fps))), :, :, 0] for k in ks])  # (K, 2, 5)
    for t in (a, b):
        k0 = int(round(t * F_FPS))
        if k0 >= len(ks):
            k0 = len(ks) - 1
        ref = xs[k0]
        far = np.abs(ks - k0) >= 3
        if not far.any():
            continue
        dist = np.nanmax(np.abs(xs[far] - ref[None]).reshape(far.sum(), -1), axis=1)
        if np.nanmin(dist) < 0.5:
            return False
    return True


def range_text(rt, a, b):
    if rt == "whole":
        return "这段视频里"
    if rt == "timestamp":
        return f"第 {fmt_t(a)} 秒到第 {fmt_t(b)} 秒之间"
    if rt == "frames":
        return f"第 {int(round(a * F_FPS))} 帧到第 {int(round(b * F_FPS))} 帧之间"
    return "从图 A 的画面到图 B 的画面之间（图 A、图 B 在视频之后单独给出）"


def make_clip_id(vid, c0, c1):
    return f"pvam__{vid}__{c0:07d}_{c1:07d}"


def expand(der: Derived, f, rng, cfg: BenchConfig, meta=None):
    """Fact -> list of question dicts."""
    task, sp = f["task"], f["span"]
    fps = der.fps
    out = []
    qa_types = []
    if f.get("mc"):
        qa_types.append("单选")
    if f.get("fill"):
        qa_types.append("填空")
    group = rng.random() < cfg.group_prob
    if task in ("T1", "T2", "T3", "C1"):
        rts = list(cfg.range_types)
        if not group:
            rts = [rts[int(rng.integers(len(rts)))]]
            qa_types = [qa_types[int(rng.integers(len(qa_types)))]]
        for rt in rts:
            if rt == "whole":
                c0, c1 = sp["ta"], sp["tb"]
                a, b, L = 0.0, (sp["tb"] - sp["ta"]) / fps, (sp["tb"] - sp["ta"]) / fps
                L = round(L, 3)
                b = L
            else:
                c0, c1, a, b, L = sp["c0"], sp["c1"], sp["a"], sp["b"], sp["L"]
            if rt == "whole" and not 2.0 <= L <= 10.0:
                continue
            if rt == "boundary_frames" and not boundary_unique(der, c0, L, a, b):
                continue
            for qt in qa_types:
                out.append(_question(der, f, qt, rt, c0, c1, a, b, L, rng, cfg, meta))
    elif task == "T4":
        for qt in qa_types:
            out.append(_question(der, f, qt, None, sp["c0"], sp["c1"], None, None, sp["L"], rng, cfg, meta))
            if f["fill"].get("frame_ok") and group:
                out.append(_question(der, f, qt, None, sp["c0"], sp["c1"], None, None, sp["L"], rng, cfg, meta, as_frame=True))
    else:  # T5
        for qt in qa_types:
            out.append(_question(der, f, qt, None, sp["c0"], sp["c1"], None, None, sp["L"], rng, cfg, meta))
    return out


def _question(der, f, qt, rt, c0, c1, a, b, L, rng, cfg, meta, as_frame=False):
    body = f["body"]
    describe = qt == "填空" and rng.random() < cfg.describe_prob and f["task"] != "T4"
    if f["task"] == "T5":
        H = f["horizon"]
        stem = f"视频在切点处结束（切点 = 视频结尾，第 {fmt_t(L)} 秒）。{body}"
        predict = {"cut": L, "horizon": H}
    elif f["task"] == "T4":
        stem = body
        if as_frame:
            stem = stem.replace("是在第几秒", "是在第几帧（8 fps 帧序列）")
        predict = None
    else:
        stem = f"{range_text(rt, a, b)}，{body}"
        predict = None
    if qt == "单选":
        opts = f["mc"]["options"]
        q = stem + "\n" + "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts)) + "\n把选项字母写在花括号里。"
        answer, atype = f["mc"]["answer"], "choice"
    else:
        opts = None
        instr = f["fill"]["instr"]
        if describe:
            instr = "先用几句话分析，再" + instr.replace("把", "把最终答案", 1)
        q = stem + instr
        answer, atype = f["fill"]["answer"], f["fill"]["answer_type"]
        if as_frame:
            answer, atype = int(round(answer * F_FPS)), "frame"
            q = stem + "把帧号写在花括号里，如 {25}。"
    rng_obj = None
    if rt:
        rng_obj = {"type": rt, "start": round(a, 3), "end": round(b, 3)}
        if rt == "frames":
            rng_obj.update(start_frame=int(round(a * F_FPS)), end_frame=int(round(b * F_FPS)))
        if rt == "boundary_frames":
            rng_obj.update(image_a_frame=int(round(a * F_FPS)), image_b_frame=min(int(round(b * F_FPS)), int(L * F_FPS) - 1))
    d = der
    notes_per_s = None
    if c1 > c0:
        notes_per_s = round(float(((d.on >= c0) & (d.on < c1)).sum() / ((c1 - c0) / d.fps)), 2)
    tags = dict(source="PianoVAM", label_tier=f["label_tier"], k_notes=f["k_notes"], min_conf=f["min_conf"],
                notes_per_s=notes_per_s, interval_len=round(b - a, 3) if rt else None, range_type=rt or "none",
                hand=f.get("hand"), finger=f.get("finger"), black_key=f.get("black_key"),
                horizon=f.get("horizon"), qa_variant="描述" if describe else qt, as_frame=as_frame,
                setting_only="F" if (rt == "frames" or as_frame) else None)
    if meta:
        tags.update(skill=meta.get("skill"), split=meta.get("split"), piece=meta.get("piece"))
    return dict(
        task=f["task"], subtype=f["subtype"], level=f["level"], fact_id=f["fact_id"],
        clip_id=make_clip_id(d.vid, c0, c1), clip=dict(video_id=d.vid, start_frame=int(c0), end_frame=int(c1),
                                                       fps=d.fps, length_s=round((c1 - c0) / d.fps, 3)),
        frame_fps=P_FPS if f["task"] == "T5" else F_FPS,
        range=rng_obj, predict=predict, qa_type="描述" if describe else qt,
        question=q, options=opts, answer_type=atype, answer=answer,
        review=dict(needs_review=needs_review(f), depends=f["depends"], focus_frames=f.get("focus_frames", [])),
        baselines=dict(copy=f.get("baseline_copy")) if f["task"] == "T5" else {},
        extra={k: v for k, v in f.items() if k in ("distractor_roles", "dx_white", "hold_s", "span_white", "octave", "key")},
        tags=tags,
    )
