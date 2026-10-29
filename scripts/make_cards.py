"""Expansion step A: turn every checked fact into a structured "fact card" for the LLM.

  python -B scripts/make_cards.py --bench work/bench --release work/release --perf work/perf --fing work/fing \
      --out work/cards.jsonl

A card holds only facts (no wording to copy): the clip, the interval, who is asked about, every key press
in the window with hand / finger / press / release time, the answer and options, measured values, and
(for subtypes that need it) the clip frames to show the LLM. The LLM later rewrites only the question
stem and writes an explanation; answers and options never pass through it.

Times are clip seconds of the card's canonical clip (the timestamp variant when the fact has one).
"""
import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmfinger.bench import F_FPS  # noqa: E402
from pmfinger.states import FINGER_ZH, HAND_ZH, STATE_ZH, load_derived, pitch_name  # noqa: E402

warnings.simplefilter("ignore", RuntimeWarning)

# subtypes whose question / explanation profit from seeing frames, and how many frames to attach
VISUAL = {
    ("T2", "穿指跨指"): 16, ("T3", "运动状态序列"): 16, ("T3", "张开跨度"): 12, ("T3", "手的移动"): 12,
    ("T4", "动作定位"): 16, ("T4", "音区定位"): 16, ("T5", "运动状态"): 12, ("T5", "移动方向"): 12,
}
RANGE_TASKS = ("T1", "T2", "T3", "C1")
PLACEHOLDER = "【区间】"


def canonical(qs):
    """Pick the question variant whose clip/interval the card describes."""
    for rt in ("timestamp", "whole", "frames", "boundary_frames", None):
        for q in qs:
            if (q["range"] or {}).get("type") == rt or (rt is None and not q["range"]):
                if not q["tags"].get("as_frame"):
                    return q
    return qs[0]


def frames_for(q, root, n_max):
    """Official frames of the canonical clip inside the asked window, evenly thinned to n_max."""
    fl = q["media"]["frames"]
    fps = q["frame_fps"]
    if q["range"]:
        a, b = q["range"]["start"], q["range"]["end"]
    elif q["task"] == "T5":
        a, b = max(0.0, q["predict"]["cut"] - 1.5), q["predict"]["cut"]
    else:
        a, b = 0.0, q["clip"]["length_s"]
    ks = [k for k in range(len(fl)) if a - 1e-6 <= k / fps < b + 1e-6]
    if len(ks) > n_max:
        ks = [ks[int(round(i))] for i in np.linspace(0, len(ks) - 1, n_max)]
    return [{"path": fl[k], "t": round(k / fps, 3)} for k in ks]


def answer_display(q):
    a = q["answer"]
    if q["answer_type"] == "choice":
        return q["options"]["ABCDE".index(a)]
    if isinstance(a, list):
        return ", ".join(str(x) for x in a)
    return str(a)


def fact_display(f):
    if f.get("mc"):
        return f["mc"]["options"]["ABCDE".index(f["mc"]["answer"])]
    v = f["fill"]["answer"]
    return ", ".join(str(x) for x in v) if isinstance(v, list) else str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, help="dir with facts.jsonl")
    ap.add_argument("--release", required=True, help="dir with questions.jsonl (frame paths) and clips/")
    ap.add_argument("--perf", required=True)
    ap.add_argument("--fing", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    facts = {json.loads(l)["fact_id"]: json.loads(l) for l in open(os.path.join(a.bench, "facts.jsonl"), encoding="utf-8")}
    qs_by = {}
    for l in open(os.path.join(a.release, "questions.jsonl"), encoding="utf-8"):
        q = json.loads(l)
        qs_by.setdefault(q["fact_id"], []).append(q)
    ders = {}
    n_vis = 0
    with open(a.out, "w", encoding="utf-8") as fo:
        for fid, f in facts.items():
            if fid not in qs_by:
                continue
            q = canonical(qs_by[fid])
            vid = f["video_id"]
            if vid not in ders:
                ders[vid] = load_derived(os.path.join(a.perf, f"{vid}.npz"), os.path.join(a.fing, f"{vid}.fing.npz"))
            d = ders[vid]
            c0, c1, fps = q["clip"]["start_frame"], q["clip"]["end_frame"], d.fps
            # window whose key presses are listed: the interval; the whole clip for T4; for T5 the 2 s
            # context plus what happens up to the horizon (marked after_cut, never to be described as seen)
            if q["range"]:
                w0, w1 = c0 + int(round(q["range"]["start"] * fps)), c0 + int(round(q["range"]["end"] * fps))
            elif q["task"] == "T5":
                w0, w1 = c0, c1 + int(round((q["predict"]["horizon"] + 0.3) * fps))
            else:
                w0, w1 = c0, c1
            hands = [f["hand"]] if f.get("hand") is not None else [0, 1]
            m = (d.on >= w0) & (d.on < w1) & np.isin(d.hand, hands)
            events = []
            for n in np.flatnonzero(m):
                events.append(dict(key=pitch_name(d.pitch[n]), hand=HAND_ZH[int(d.hand[n])],
                                   finger=f"{int(d.finger[n]) + 1} {FINGER_ZH[int(d.finger[n])]}" if d.finger_trust[n] else None,
                                   press_s=round((d.on[n] - c0) / fps, 2), release_s=round((d.rel[n] - c0) / fps, 2),
                                   **({"after_cut": True} if q["task"] == "T5" and d.on[n] >= c1 else {})))
            measured = {k: f[k] for k in ("dx_white", "span_white", "hold_s", "octave", "horizon") if k in f}
            if f["task"] == "T5" and f["subtype"] == "运动状态" and f.get("baseline_copy"):
                measured["state_at_cut"] = f["mc"]["options"]["ABCDE".index(f["baseline_copy"])]
            if f["subtype"] == "运动状态序列":
                measured["state_sequence"] = f["fill"]["answer"]
            n_frames = VISUAL.get((f["task"], f["subtype"]), 0)
            card = dict(
                fact_id=fid, task=f["task"], subtype=f["subtype"], level=f["level"], video_id=vid,
                clip_id=q["clip_id"], clip_length_s=q["clip"]["length_s"],
                interval=({"start_s": q["range"]["start"], "end_s": q["range"]["end"]} if q["range"] else None),
                predict=q["predict"],
                subject=dict(hand=HAND_ZH[f["hand"]] if f.get("hand") is not None else None,
                             finger=f"{f['finger'] + 1} {FINGER_ZH[f['finger']]}" if f.get("finger") is not None else None,
                             key=f.get("key")),
                template_stem=f["body"],
                needs_placeholder=f["task"] in RANGE_TASKS,
                answer=dict(type=("choice" if f.get("mc") else f["fill"]["answer_type"]),
                            display=fact_display(f),
                            fill_value=f["fill"]["answer"] if f.get("fill") else None,
                            options=f["mc"]["options"] if f.get("mc") else None),
                events=events, measured=measured,
                frames=frames_for(q, a.release, n_frames) if n_frames else [],
                state_names=STATE_ZH[1:],
            )
            n_vis += bool(card["frames"])
            fo.write(json.dumps(card, ensure_ascii=False) + "\n")
    print(f"{sum(1 for _ in open(a.out, encoding='utf-8'))} cards ({n_vis} with frames) -> {a.out}")


if __name__ == "__main__":
    main()
