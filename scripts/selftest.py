"""Synthetic end-to-end check (no real data needed). Run this first on the server:

  python scripts/selftest.py

Simulates a right hand playing C-major scales (thumb-under / finger-over), a left hand playing block
chords with pedal-long MIDI offsets, MANO-like ring/little-finger outward bias, jitter, an invalid
segment, glitch frames and a 2-frame audio/video offset; then checks fingering accuracy, AV-offset
recovery, release error, prior fitting and VQA generation.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmfinger import Config, FingerPrior, Performance, load_performance, run_fingering, save_performance  # noqa: E402
from pmfinger.keys import KEY_X  # noqa: E402
from pmfinger.metrics import count_metrics, summarize  # noqa: E402
from pmfinger.vqa import generate_for_video  # noqa: E402


def synth(seed=0, fps=30.0, av_true=2):
    rng = np.random.default_rng(seed)
    notes = []  # onset, midi_offset, pitch, hand, finger, true_hold
    up, fu = [60, 62, 64, 65, 67, 69, 71, 72], [0, 1, 2, 0, 1, 2, 3, 4]
    down, fd = [71, 69, 67, 65, 64, 62, 60], [3, 2, 1, 0, 2, 1, 0]
    t = 1.0
    for _ in range(6):
        base = 12 * int(rng.integers(0, 2))
        for p, f in zip(up + down, fu + fd):
            notes.append((t, t + 0.8, p + base, 1, f, 0.2))
            t += 0.25
        t += 0.75
    chords = [([36, 40, 43], [4, 2, 0]), ([41, 45, 48], [4, 2, 0]), ([43, 47, 50], [4, 2, 0])]
    tt, i = 1.0, 0
    while tt < t:
        ps, fs = chords[i % 3]
        for p, f in zip(ps, fs):
            notes.append((tt, tt + 1.5, p, 0, f, 0.6))
        tt += 1.0
        i += 1

    T = int((t + 1.5) * fps)
    times = np.arange(T) / fps
    tips = np.zeros((T, 2, 5, 3))
    tips[..., 1] = 0.5
    tips[..., 2] = 20.0
    true_rel = {}
    for h in (0, 1):
        d = 1.0 if h == 1 else -1.0
        hn = sorted([n for n in notes if n[3] == h], key=lambda n: n[0])
        kt, kb = [], []
        for n in hn:
            b = KEY_X[n[2]] - d * n[4]
            if kt and abs(kt[-1] - n[0]) < 1e-6:
                kb[-1] = (kb[-1] + b) / 2
            else:
                kt.append(n[0])
                kb.append(b)
        base_t = np.interp(times, kt, kb)
        for f in range(5):
            tips[:, h, f, 0] = base_t + d * f
        for n in hn:
            a, b = int(round(n[0] * fps)) - 1, int(round((n[0] + n[5]) * fps))
            tips[a:b, h, n[4], 0] = KEY_X[n[2]]
            tips[a:b, h, n[4], 2] = 0.0
            tips[max(0, a - 4):a, h, n[4], 2] = 26.0
            true_rel[(round(n[0], 4), n[2])] = b
        tips[:, h, 3:5, 0] += d * 0.2                       # ring/little finger project outward
    tips[..., 0] += rng.normal(0, 0.12, tips[..., 0].shape)
    tips[..., 2] += rng.normal(0, 1.5, tips[..., 2].shape)
    valid = np.ones((T, 2), dtype=bool)
    valid[200:330, 0] = False                               # long invalid segment
    for k in rng.choice(T, 12, replace=False):              # glitch frames
        tips[k, 1, :, 0] += 3.0

    notes.sort(key=lambda n: (n[0], n[2]))
    on = np.array([n[0] for n in notes]) - av_true / fps    # MIDI clock leads video by av_true frames
    perf = Performance(
        video_id="synthetic", fps=fps, tips=tips, valid=valid,
        onset=on, offset=np.array([n[1] for n in notes]) - av_true / fps,
        pitch=np.array([n[2] for n in notes]),
        label_hand=np.array([n[3] for n in notes]), label_finger=np.array([n[4] for n in notes]),
        video_path="synthetic.mp4",
    )
    rel = np.array([true_rel[(round(n[0], 4), n[2])] for n in notes])
    return perf, rel


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    return ok


def main():
    perf, rel_true = synth()
    cfg = Config()
    all_ok = True

    with tempfile.TemporaryDirectory() as d:          # npz round trip
        p = str(Path(d) / "s.npz")
        save_performance(perf, p)
        perf = load_performance(p)

    res = run_fingering(perf, FingerPrior.default(), cfg)
    s = summarize([count_metrics(res, perf.label_hand, perf.label_finger)])
    print(json.dumps(s, ensure_ascii=False))
    # the simulated press starts 1 frame before the onset and holds for 6, so 1..3 are all reasonable
    all_ok &= check("av offset recovered", abs(res["av_offset"] - 2) <= 1, f"(got {res['av_offset']}, want 2±1)")
    all_ok &= check("coverage", s["coverage"] > 0.7, f"{s['coverage']:.3f}")
    all_ok &= check("hand+finger acc (default prior)", s["hand_finger_acc"] > 0.9, f"{s['hand_finger_acc']:.3f}")
    print("estimated bias:", np.round(res["bias"], 2).tolist())

    good = (res["status"] == 0) & ~res["release_uncertain"]
    err = np.abs(res["release_frame"][good] - rel_true[good])
    midi_err = np.abs(res["offset_frame"][good] - rel_true[good])
    all_ok &= check("release error (frames, median)", np.median(err) <= 3,
                    f"{np.median(err):.1f} vs MIDI-offset error {np.median(midi_err):.1f}")

    prior = FingerPrior.fit([perf])
    res2 = run_fingering(perf, prior, cfg)
    s2 = summarize([count_metrics(res2, perf.label_hand, perf.label_finger)])
    all_ok &= check("hand+finger acc (fitted prior)", s2["hand_finger_acc"] > 0.9, f"{s2['hand_finger_acc']:.3f}")

    rng = np.random.default_rng(0)
    recs = generate_for_video(perf, res, rng, ("zh", "en"))
    types = sorted({r["type"] for r in recs})
    all_ok &= check("vqa generated", len(recs) > 15, f"{len(recs)} samples, types={types}")
    json.dumps(recs, ensure_ascii=False)
    for r in recs[:: max(1, len(recs) // 8)]:
        print("-", r["type"], r["lang"], [i["frame"] for i in r["images"]])
        print("  Q:", r["question"].replace("\n", " | "))
        print("  A:", r["answer"])
    print("ALL PASS" if all_ok else "SOME CHECKS FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
