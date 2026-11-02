"""Benchmark-only generators that the P0 registry does not cover: T1 (what was played) and T2 which hand.

They follow the P0 rules exactly: 40 ms onset groups anchored at the earliest onset (no chaining),
trusted hands only (gold or confident v2 agreeing with the auto label), every counted press visible on the
delivered 8 fps frames, no answer-relevant onset within 0.15 s of an interval end. The P0 module itself is
not changed; this file subclasses its generator.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("p0_rules", Path(__file__).with_name("p0_rules.py"))
if "p0_rules" in sys.modules:
    rules = sys.modules["p0_rules"]
else:
    rules = importlib.util.module_from_spec(_spec)
    sys.modules["p0_rules"] = rules
    _spec.loader.exec_module(rules)
from pmfinger.states import CHORDS, INTERVALS, NAMES, melody_shape  # noqa: E402

HAND_ZH, LETTERS, pitch_name = rules.HAND_ZH, rules.LETTERS, rules.pitch_name
CHORD_TOL_S, U_FPS = rules.CHORD_TOL_S, rules.U_FPS
SHAPES = ["上行", "下行", "先上行后下行", "先下行后上行"]
TEXTURES = ["只有单音", "只有双音或和弦", "单音和双音（和弦）都有"]
TRIADS = ["大三和弦", "小三和弦", "减三和弦"]          # augmented triads are symmetric: no unique root, never asked
OCT_LO, OCT_HI = 1, 7                                 # complete octaves on an 88-key piano (C1-B1 ... C7-B7)


def octave_of(p):
    return int(p) // 12 - 1


def octave_label(o):
    return f"C{o}–B{o}八度范围"


def triad_name(pitches):
    """Root-position-independent name of a 3-pitch-class major/minor/diminished triad, else None."""
    pcs = sorted({int(p) % 12 for p in pitches})
    if len(pcs) != 3:
        return None
    for r in pcs:
        iv = tuple(sorted((p - r) % 12 for p in pcs))
        name = CHORDS.get(iv)
        if name in TRIADS:
            return NAMES[r], name
    return None


TEMPLATES = {
    "按键序列": "{rt}，{hand}的单音按键序列依次是什么？每次按下都算，重复按同一键也要重复写出。"
               "把全部键名按时间顺序写在花括号里，用逗号分隔。",
    "音区": "{rt}，{hand}按下的键主要在哪个八度范围？按每个键的按下次数统计，同时按下的多个键各算一次；"
           "所选范围包含至少75%的按下次数。",
    "旋律走向": "{rt}，{hand}的连续单音旋律走向是什么？只比较按下的键的高低，相邻重复键不计为转折。",
    "织体": "{rt}，{hand}按下的键属于哪种织体？以每组最早按下后40毫秒内的键为一组，不链式延长；"
           "按每组的不同键数区分单音与双音（和弦）。",
    "音程": "{rt}，{hand}只有一次同时按下两个或更多不同键，这一次构成什么音程？同时按下以最早按下后40毫秒为限，不链式延长。",
    "和弦": "{rt}，{hand}只有一次同时按下两个或更多不同键，这一次构成什么和弦？同时按下以最早按下后40毫秒为限，不链式延长。",
    "哪只手": "{rt}，{key} 只被按下一次，这一次是哪只手按下的？左右手均指演奏者本人。",
}


class BenchGen(rules.Gen):
    """P0 generator plus T1 and T2-which-hand on 5 s understanding clips at 8 fps."""

    # ------------------------------------------------------------------ shared window for T1
    def _window(self, c0, c1, h, lo=1.5, hi=4.0):
        """An interval [a, b) whose presses by hand h are trusted, visible, countable and away from its ends."""
        P = self.P
        iv = self.ivl(c0, c1, lo, hi)
        if not iv:
            return None
        a, b = iv
        if not P.visible(h, a, b, 0.95) or len(P.hand_unknown_in(a - self.edge, b + self.edge)):
            return None
        if self.near_edge(P.on[P.notes(a - self.edge, b + self.edge, h)], a, b):
            return None
        gs = P.groups(P.notes(a, b, h), CHORD_TOL_S)
        if not gs or not self.countable(gs, self.grid(c0, c1, U_FPS)):
            return None
        return a, b, gs

    def _t1(self, c0, c1, sub, slots, answer_type, answer, options=None, a=None, b=None, difficulty=None):
        return self.fact("T1", sub, "L1", c0, c1, U_FPS, slots, answer_type, answer, options=options,
                         rng=self.rng_obj(a, b, c0), difficulty=difficulty)

    # ------------------------------------------------------------------ T1
    def t1_sequence(self, c0, c1):
        P, h = self.P, self.rnd.randrange(2)
        w = self._window(c0, c1, h)
        if not w:
            return None
        a, b, gs = w
        if not 2 <= len(gs) <= 6 or any(len(g) != 1 for g in gs):
            return None
        keys = [pitch_name(P.pitch[g[0]]) for g in gs]
        return self._t1(c0, c1, "按键序列", dict(hand=HAND_ZH[h]), "key_seq", keys, a=a, b=b,
                        difficulty=dict(n=len(keys)))

    def t1_register(self, c0, c1):
        P, h = self.P, self.rnd.randrange(2)
        w = self._window(c0, c1, h)
        if not w:
            return None
        a, b, gs = w
        ps = [int(P.pitch[n]) for g in gs for n in g]
        if len(ps) < 3:
            return None
        octs = [octave_of(p) for p in ps]
        o = max(set(octs), key=octs.count)
        share = octs.count(o) / len(octs)
        if share < 0.75 or not OCT_LO <= o <= OCT_HI:
            return None
        near = sorted((x for x in range(OCT_LO, OCT_HI + 1) if x != o), key=lambda x: (abs(x - o), x))[:3]
        opts = self.mc(octave_label(o), [octave_label(x) for x in near])
        return self._t1(c0, c1, "音区", dict(hand=HAND_ZH[h]), "choice", opts["letter"], options=opts, a=a, b=b,
                        difficulty=dict(share=round(share, 3), n=len(ps)))

    def t1_shape(self, c0, c1):
        P, h = self.P, self.rnd.randrange(2)
        w = self._window(c0, c1, h)
        if not w:
            return None
        a, b, gs = w
        if len(gs) < 3 or any(len(g) != 1 for g in gs):
            return None
        shape = melody_shape([int(P.pitch[g[0]]) for g in gs])
        if shape not in SHAPES:
            return None
        opts = self.mc(shape, SHAPES)
        return self._t1(c0, c1, "旋律走向", dict(hand=HAND_ZH[h]), "choice", opts["letter"], options=opts, a=a, b=b,
                        difficulty=dict(n=len(gs)))

    def t1_texture(self, c0, c1):
        P, h = self.P, self.rnd.randrange(2)
        w = self._window(c0, c1, h)
        if not w:
            return None
        a, b, gs = w
        if len(gs) < 3:
            return None
        sizes = [len({int(P.pitch[n]) for n in g}) for g in gs]
        tex = TEXTURES[0] if max(sizes) == 1 else TEXTURES[1] if min(sizes) >= 2 else TEXTURES[2]
        opts = self.mc(tex, TEXTURES)
        return self._t1(c0, c1, "织体", dict(hand=HAND_ZH[h]), "choice", opts["letter"], options=opts, a=a, b=b,
                        difficulty=dict(sizes=sizes))

    def t1_harmony(self, c0, c1):
        """Exactly one onset group with >= 2 different keys: an interval (2 keys) or a triad (3 pitch classes)."""
        P, h = self.P, self.rnd.randrange(2)
        w = self._window(c0, c1, h)
        if not w:
            return None
        a, b, gs = w
        multi = [g for g in gs if len({int(P.pitch[n]) for n in g}) >= 2]
        if len(multi) != 1:
            return None
        ps = sorted({int(P.pitch[n]) for n in multi[0]})
        if len(ps) == 2:
            d = ps[1] - ps[0]
            if not 1 <= d <= 12:
                return None
            pool = [INTERVALS[x] for x in sorted(range(1, 13), key=lambda x: (abs(x - d), x)) if x != d][:3]
            opts, kind = self.mc(INTERVALS[d], pool), "音程"
        else:
            t = triad_name(ps)
            if not t:
                return None
            root, q = t
            pool = [f"{root} {x}" for x in TRIADS + ["增三和弦"] if x != q][:3]
            opts, kind = self.mc(f"{root} {q}", pool), "和弦"
        return self._t1(c0, c1, "和弦与音程", dict(hand=HAND_ZH[h], kind=kind), "choice", opts["letter"], options=opts, a=a, b=b,
                        difficulty=dict(keys=[pitch_name(p) for p in ps]))

    # ------------------------------------------------------------------ T2 which hand
    def t2_hand(self, c0, c1):
        P = self.P
        iv = self.ivl(c0, c1, 1.5, 4.0)
        if not iv:
            return None
        a, b = iv
        if len(P.hand_unknown_in(a - self.edge, b + self.edge)):
            return None
        idx = P.notes(a, b)
        held_before = P.sounding_in(a, a + 1)
        cand = [n for n in idx if (P.pitch[idx] == P.pitch[n]).sum() == 1 and P.hand[n] >= 0
                and not (P.pitch[held_before] == P.pitch[n]).any()
                and min(P.on[n] - a, b - P.on[n]) >= self.edge]
        if not cand:
            return None
        n = self.rnd.choice(cand)
        h = int(P.hand[n])
        if not P.visible(h, a, b, 0.95) or not P.visible(1 - h, a, b, 0.95):
            return None
        if not self.press_visible(n, self.grid(c0, c1, U_FPS)):
            return None
        other = P.palm_med(1 - h, P.on[n], self.fr(0.05))
        opts = self.mc(HAND_ZH[h], HAND_ZH)
        return self.fact("T2", "哪只手", "L1", c0, c1, U_FPS, dict(key=pitch_name(P.pitch[n])), "choice", opts["letter"],
                         options=opts, rng=self.rng_obj(a, b, c0),
                         difficulty=dict(other_hand_dist=None if not np.isfinite(other)
                                         else round(abs(other - rules.KEY_X[int(P.pitch[n])]), 2)))


GENERATORS = {"T1/按键序列": "t1_sequence", "T1/音区": "t1_register", "T1/旋律走向": "t1_shape",
              "T1/织体": "t1_texture", "T1/和弦与音程": "t1_harmony", "T2/哪只手": "t2_hand"}


def bench_question(fact):
    s = fact["slots"]
    rt = rules_range_text(fact.get("range"))
    q = TEMPLATES[s.get("kind", fact["subtype"])].format(rt=rt, hand=s.get("hand", ""), key=s.get("key", ""))
    if fact.get("options"):
        q += "选项：" + "；".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(fact["options"]["choices"])) \
             + "。把选项字母写在花括号里。"
    return q


def rules_range_text(r):
    if not r or r["type"] == "whole":
        return "这段视频里"
    return f"从第 {r['start']:.3f} 秒到第 {r['end']:.3f} 秒（不含）之间"
