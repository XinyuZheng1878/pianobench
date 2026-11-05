"""Score fingering for Hanon, The Virtuoso Pianist Part I, Nos. 1-20 (Schirmer 1900, as engraved by the
Mutopia Project: scores/hanon/hanonNN.ily, CC BY-SA 4.0).

Hanon prints the fingering of the first figure of each direction (ascending, then after the double bar
descending) and says the same fingering is kept for every following figure. So:

  * every printed finger is used as is;
  * inside the first figure of a section the hand stays in one position, so an unprinted note on a key
    already played in that figure takes the same finger (No. 1: c1 e2 f3 g4 a5 g f e -> g4 f3 e2);
  * every later figure (8 sixteenths) gets, for its unprinted notes, the finger of the same position in that
    first figure, but only if its melodic shape (diatonic steps between consecutive notes) is the same and
    none of its printed fingers disagrees with the first figure; otherwise only printed fingers are used;
  * the final whole/half note of each hand gets a finger only when printed.

score_notes(n) -> list of dict(t=<position in sixteenths>, pitch=<MIDI>, hand=0 left / 1 right,
                               finger=0..4 (0 = thumb) or -1, printed=bool, section=0 asc / 1 desc)
align(on, pitch, notes) -> per MIDI note index into notes (or -1), via note-sequence alignment.
"""
import re
from pathlib import Path

import numpy as np

SCORE_DIR = Path(__file__).resolve().parents[1] / "scores" / "hanon"
STEP = {"c": 0, "d": 1, "e": 2, "f": 3, "g": 4, "a": 5, "b": 6}
SEMI = [0, 2, 4, 5, 7, 9, 11]
FIG = 8


def _strip(src):
    src = re.sub(r"%[^\n]*", "", src)
    out, i = [], 0
    while i < len(src):                               # drop \markup { ... } (balanced) and "strings"
        if src.startswith("\\markup", i):
            j = src.find("{", i)
            k = re.match(r"\\markup\s*", src[i:])
            if j < 0 or src[i + k.end()] != "{":      # \markup word
                i += k.end()
                m = re.match(r"(\\\w+\s*)*[^\s{}]+", src[i:])
                i += m.end() if m else 0
                continue
            depth, i = 0, j
            while i < len(src):
                depth += {"{": 1, "}": -1}.get(src[i], 0)
                i += 1
                if depth == 0:
                    break
            continue
        if src[i] == '"':
            i = src.index('"', i + 1) + 1
            continue
        out.append(src[i])
        i += 1
    return "".join(out)


def _block(src, name):
    m = re.search(rf"^{name}\s*=\s*(\\relative\s+([a-g](?:is|es)?[',]*)\s*)?{{", src, re.M)
    if not m:
        raise ValueError(f"no {name} block")
    i, depth = m.end() - 1, 0
    for j in range(i, len(src)):
        depth += {"{": 1, "}": -1}.get(src[j], 0)
        if depth == 0:
            return m.group(2), src[i + 1:j]
    raise ValueError("unbalanced braces")


TOKEN = re.compile(r"""
    (?P<bar>\\bar\s*\|\|?)                                  # placeholder (strings removed: see _parse)
  | (?P<cmd>\\(?:time\s+\d+/\d+|key\s+[a-g]\w*\s+\\\w+|clef\s+\w+|[A-Za-z]+))
  | (?P<chord><[^>]*>\s*\d*\.?)
  | (?P<note>(?<![A-Za-z])[a-gr](?:is|es)?(?![A-Za-z])[',]*\s*(?:\d+\.?)?)
  | (?P<fing>[-_^]\s*\d)
  | (?P<other>\S)
""", re.X)


def _parse(path):
    src = Path(path).read_text(encoding="utf-8")
    src = src.replace('\\bar "||"', "\\DOUBLEBAR").replace('\\bar "|."', "")
    src = _strip(src)
    res = {}
    for hand, name in ((1, "RH"), (0, "LH")):
        rel, body = _block(src, name)
        prev = None if rel is None else _abs(rel)[1]
        notes, t, dur, section = [], 0, 16, 0
        for m in TOKEN.finditer(body):
            k, s = m.lastgroup, m.group(0)
            if k == "cmd":
                if s == "\\DOUBLEBAR":
                    section = 1
                continue
            if k in ("bar", "other"):
                continue
            if k == "fing":
                if notes and notes[-1]["finger"] < 0:
                    notes[-1]["finger"], notes[-1]["printed"] = int(s[-1]) - 1, True
                continue
            if k == "chord":                           # Hanon only uses one-note "chords" to place a finger
                inner, d = re.match(r"<([^>]*)>\s*(\d*)", s).groups()
                nm = re.match(r"\s*([a-g](?:is|es)?[',]*)", inner).group(1)
                f = re.search(r"[-_^]\s*(\d)", inner)
                s = nm + d
                fing = int(f.group(1)) - 1 if f else -1
            else:
                fing = -1
            nm, d = re.match(r"([a-gr](?:is|es)?[',]*)\s*(\d*)", s).groups()
            if d:
                dur = int(d)
            if nm[0] == "r":
                t += 16 // dur
                continue
            midi, step = _abs(nm) if rel is None else _rel(nm, prev)
            prev = step
            notes.append(dict(t=t, pitch=midi, step=step, hand=hand, finger=fing, printed=fing >= 0, section=section))
            t += 16 // dur
        res[hand] = notes
    return res


def _abs(nm):
    """Absolute LilyPond pitch (c = C3 = MIDI 48) -> (midi, diatonic step)."""
    st = STEP[nm[0]]
    acc = 1 if "is" in nm else -1 if "es" in nm else 0
    octv = 3 + nm.count("'") - nm.count(",")
    return 12 * (octv + 1) + SEMI[st] + acc, 7 * octv + st


def _rel(nm, prev):
    """\relative pitch: the octave nearest the previous note (within a fourth), then ' and , marks."""
    st = STEP[nm[0]]
    acc = 1 if "is" in nm else -1 if "es" in nm else 0
    d = min((7 * o + st for o in range(-2, 12)), key=lambda x: abs(x - prev))
    d += 7 * (nm.count("'") - nm.count(","))
    octv, s = divmod(d, 7)
    return 12 * (octv + 1) + SEMI[s] + acc, d


def score_notes(n, plays=1):
    """plays=2 unfolds the repeat: the whole exercise is inside one volta, followed by one closing note per
    hand, so a take that observes the repeat (folder name 反复) is the body twice and then the last note."""
    path = SCORE_DIR / f"hanon{int(n):02d}.ily"
    hands = _parse(path)
    out = []
    for h, notes in hands.items():
        _propagate(notes)
        body, last = notes[:-1], notes[-1]
        L = last["t"]
        for k in range(plays):
            out += [dict(x, t=x["t"] + k * L, play=k) for x in body]
        out.append(dict(last, t=last["t"] + (plays - 1) * L, play=plays - 1))
    out.sort(key=lambda x: (x["t"], x["pitch"]))
    return out


def _shape(fig):
    return tuple(b["step"] - a["step"] for a, b in zip(fig, fig[1:]))


def _propagate(notes):
    """Fill unprinted fingers; returns the number of printed fingers that disagree with the template."""
    conflicts = 0
    for sec in (0, 1):
        ns = [x for x in notes if x["section"] == sec]
        figs = [ns[i:i + FIG] for i in range(0, len(ns) - len(ns) % FIG, FIG)]
        if not figs:
            continue
        first = figs[0]
        # the hand stays in one position inside a figure: an unprinted note on a key already played in the
        # figure takes the same finger (e.g. No. 1: c1 e2 f3 g4 a5 g f e -> g4 f3 e2)
        seen = {}
        for x in first:
            if x["finger"] < 0 and x["pitch"] in seen:
                x["finger"] = seen[x["pitch"]]
            if x["finger"] >= 0:
                seen.setdefault(x["pitch"], x["finger"])
        tmpl, shp = [x["finger"] for x in first], _shape(first)   # may keep -1 where nothing is printed
        for fig in figs[1:]:
            if len(fig) != FIG or _shape(fig) != shp:
                continue
            bad = sum(x["printed"] and f >= 0 and x["finger"] != f for x, f in zip(fig, tmpl))
            if bad:                     # the score changes fingering in this figure: keep only what is printed
                conflicts += bad
                continue
            for x, f in zip(fig, tmpl):
                if x["finger"] < 0:
                    x["finger"] = f
    return conflicts


def _groups(on, pitch, tol=0.04):
    order = np.lexsort((pitch, on))
    seq, start = [], None
    for i in order:
        if start is None or on[i] - start > tol:
            seq.append([])
            start = on[i]
        seq[-1].append(int(i))
    return [i for g in seq for i in sorted(g, key=lambda k: pitch[k])]


def align(on, pitch, notes):
    """Global alignment (match = same pitch, gaps free of cost beyond -1) of the MIDI notes, in onset order
    with near-simultaneous notes sorted by pitch, against the score notes in time/pitch order.
    Returns idx[n_midi] -> score index or -1."""
    a = _groups(np.asarray(on, float), np.asarray(pitch, int))
    A = np.asarray(pitch, int)[a]
    B = np.array([x["pitch"] for x in notes])
    n, m = len(A), len(B)
    S = np.zeros((n + 1, m + 1), np.int32)
    S[:, 0] = -np.arange(n + 1)
    S[0, :] = -np.arange(m + 1)
    for i in range(1, n + 1):
        eq = np.where(B == A[i - 1], 2, -2)
        row, up = S[i], S[i - 1]
        diag = up[:-1] + eq
        vert = up[1:] - 1
        best = np.maximum(diag, vert)
        for j in range(1, m + 1):                     # horizontal gaps need a running max
            row[j] = max(best[j - 1], row[j - 1] - 1)
    idx = np.full(len(A), -1)
    i, j = n, m
    while i > 0 and j > 0:
        if S[i, j] == S[i - 1, j - 1] + (2 if A[i - 1] == B[j - 1] else -2):
            if A[i - 1] == B[j - 1]:
                idx[a[i - 1]] = j - 1
            i, j = i - 1, j - 1
        elif S[i, j] == S[i - 1, j] - 1:
            i -= 1
        else:
            j -= 1
    return idx


NUM_RE = re.compile(r"(?:hanon|哈农|hn)\D{0,8}?(\d{1,2})(?!\d)", re.I)
# only an explicit wrong-FINGERING mark skips a take; hand-shape errors (thumb flat, wrist too high, ...,
# "同时多项错误") keep the printed fingering, so those takes are labelled
WRONG_RE = re.compile(r"错误?指法|指法错|wrong[\s_-]*finger|incorrect[\s_-]*finger", re.I)


def identify(folder_name):
    """(number or None, wrong_fingering: bool) from a take folder name."""
    m = NUM_RE.search(folder_name)
    n = int(m.group(1)) if m and 1 <= int(m.group(1)) <= 20 else None
    return n, bool(WRONG_RE.search(folder_name))


def _local_keep(t, o, half=12, tol=0.5, rounds=3):
    """Matched pairs (score time t in sixteenths, onset o in s) that agree with the LOCAL tempo: each pair is
    compared with a line fitted to its neighbours (+-half matched pairs, itself excluded). Human playing drifts
    from a single global tempo by far more than a sixteenth over a minute, so a global line is not usable."""
    keep = np.ones(len(t), bool)
    for _ in range(rounds):
        new = np.zeros(len(t), bool)
        for i in range(len(t)):
            lo, hi = max(0, i - half), min(len(t), i + half + 1)
            nb = [j for j in range(lo, hi) if j != i and keep[j]]
            if len(nb) < 6 or np.ptp(t[nb]) == 0:
                continue
            a, b = np.polyfit(t[nb], o[nb], 1)
            new[i] = a > 0 and abs(o[i] - (a * t[i] + b)) <= tol * a
        keep = new
    return keep


def _warp_match(on, pitch, notes, idx):
    """Re-match every MIDI note through a tempo map built from the reliable sequence matches: map its onset to
    score time, then take the same-pitch score note within half a sixteenth (nearest first, one-to-one). This
    does not depend on the order of near-simultaneous notes, so the two hands drifting apart by tens of ms
    (which swaps them in onset order) loses nothing."""
    k = np.flatnonzero(idx >= 0)
    if len(k) < 20:
        return np.full(len(pitch), -1), 0
    k = k[np.argsort(np.asarray(on)[k])]
    t = np.array([notes[j]["t"] for j in idx[k]], float)
    o = np.asarray(on, float)[k]
    keep = _local_keep(t, o)
    if keep.sum() < 20:
        return np.full(len(pitch), -1), int(keep.sum())
    ao, at = o[keep], np.maximum.accumulate(t[keep])
    x = np.asarray(on, float)
    pred = np.interp(x, ao, at)
    for side in (x < ao[0], x > ao[-1]):     # beyond the first/last anchor: extend the edge tempo
        e = slice(0, 10) if side is (x < ao[0]) else slice(-10, None)
        if side.any() and np.ptp(ao[e]) > 0:
            a, b = np.polyfit(ao[e], at[e], 1)
            pred[side] = a * x[side] + b
    st = np.array([x["t"] for x in notes], float)
    sp = np.array([x["pitch"] for x in notes])
    cand = []
    for n, (pt, p) in enumerate(zip(pred, pitch)):
        if np.isnan(pt):
            continue
        for j in np.flatnonzero((sp == p) & (np.abs(st - pt) <= 0.5)):
            cand.append((abs(st[j] - pt), n, j))
    out, used = np.full(len(pitch), -1), set()
    for _, n, j in sorted(cand):
        if out[n] < 0 and j not in used:
            out[n] = j
            used.add(j)
    return out, int(keep.sum())


def label_take(on, pitch, folder_name, min_match=0.8, number=None):
    """gt_hand, gt_finger arrays (-1 where unknown) and a report dict. Labels only Hanon 1-20 takes whose
    folder name does not mark wrong fingering and in which at least min_match of the MIDI notes are matched
    to the score (matching: sequence alignment -> local-tempo anchors -> tempo-map re-match)."""
    nn = len(pitch)
    gh, gf = np.full(nn, -1), np.full(nn, -1)
    num, wrong = identify(folder_name)
    if number is not None:                   # explicit mapping wins over the folder name (0 = do not label)
        num, wrong = (number or None), False
    rep = dict(hanon=num, wrong_fingering_in_name=wrong, used=False)
    if num is None or wrong:
        rep["reason"] = "not a Hanon 1-20 take" if num is None else "folder name marks wrong fingering"
        return gh, gf, rep
    pitch = np.asarray(pitch, int)
    one = score_notes(num, 1)
    # octave: the shift that best lines up the pitch distributions (take vs print)
    sp = np.array([x["pitch"] for x in one])
    shift = min((0, 12, -12, 24, -24), key=lambda s: abs(np.median(pitch) - np.median(sp + s)))
    best = None
    for plays in (1, 2, 3):                  # without / with the repeat (反复), or a partial extra pass
        notes = [dict(x, pitch=x["pitch"] + shift) for x in score_notes(num, plays)]
        idx = align(on, pitch, notes)
        idx2, n_anchor = _warp_match(on, pitch, notes, idx)
        m = int((idx2 >= 0).sum())
        if best is None or m > best[0]:
            best = (m, plays, idx2, notes, n_anchor, int((idx >= 0).sum()))
    m, plays, idx, notes, n_anchor, n_seq = best
    rep.update(octave_shift=shift, plays=plays, repeat_in_name="反复" in folder_name, n_seq_match=n_seq,
               n_anchor=n_anchor, n_score=len(notes), n_midi=nn, n_matched=m,
               match_midi=round(m / max(1, nn), 3), match_score=round(m / max(1, len(notes)), 3))
    if rep["match_midi"] < min_match:
        rep["reason"] = f"notes match the score below {min_match}"
        return gh, gf, rep
    for k in np.flatnonzero(idx >= 0):
        x = notes[idx[k]]
        gh[k] = x["hand"]
        gf[k] = x["finger"] if x["finger"] >= 0 else -1
    rep.update(used=True, n_gt_finger=int((gf >= 0).sum()), n_gt_hand=int((gh >= 0).sum()))
    return gh, gf, rep
