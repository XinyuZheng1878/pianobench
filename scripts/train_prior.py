"""Learn the fingering transition prior (and the global per-finger tip bias) from labelled notes.

Use PianoVAM auto labels on the non-test pieces; exclude the 11 human-GT pieces with --exclude.

  python scripts/train_prior.py --inputs '/path/pianovam_npz/*.npz' --exclude test_ids.txt \
      --out ~/work/prior.npz --fit-bias
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmfinger import Config, FingerPrior, apply_overrides, load_performance  # noqa: E402
from pmfinger.evidence import estimate_av_offset, finger_evidence, onset_frames  # noqa: E402
from pmfinger.preprocess import clean_tips  # noqa: E402
from pmfinger.schema import check_output_path  # noqa: E402


def read_ids(path):
    if not path:
        return set()
    return {l.strip() for l in open(path, encoding="utf-8") if l.strip()}


def fit_bias(perfs, cfg):
    res = [[[] for _ in range(5)] for _ in range(2)]
    for perf in perfs:
        if perf.label_hand is None:
            continue
        tips, ok = clean_tips(perf.tips, perf.valid, cfg)
        av = cfg.av_offset_frames
        if av is None:
            av = estimate_av_offset(tips, ok, perf.pitch, perf.onset, perf.fps, cfg.av_search)
        on_f = onset_frames(perf.onset, perf.fps, av)
        _, _, avail, resid = finger_evidence(tips, ok, perf.pitch, on_f, perf.fps, np.zeros((2, 5)), cfg)
        for n in np.flatnonzero((perf.label_hand >= 0) & (perf.label_finger >= 0)):
            h, f = perf.label_hand[n], perf.label_finger[n]
            if avail[n, h] and abs(resid[n, h, f]) < 1.0:
                res[h][f].append(resid[n, h, f])
    bias = np.zeros((2, 5))
    for h in range(2):
        for f in range(5):
            if res[h][f]:
                bias[h, f] = np.median(res[h][f])
            print(f"bias hand={'LR'[h]} finger={f + 1}: {bias[h, f]:+.3f}  (n={len(res[h][f])})")
    return bias


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="glob of performance npz files with label_* fields")
    ap.add_argument("--exclude", help="txt with video_ids to leave out (test set)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fit-bias", action="store_true", help="also estimate global tip bias (needs tips)")
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--set", nargs="*", default=[], help="Config overrides key=value")
    args = ap.parse_args()
    check_output_path(args.out)
    cfg = apply_overrides(Config(), args.set)
    excl = read_ids(args.exclude)
    perfs = [p for p in (load_performance(f) for f in sorted(glob.glob(args.inputs))) if p.video_id not in excl]
    print(f"{len(perfs)} performances (excluded {len(excl)} ids)")
    bias = fit_bias(perfs, cfg) if args.fit_bias else None
    prior = FingerPrior.fit(perfs, chord_tol=cfg.chord_tol_s, alpha=args.alpha, bias=bias)
    print(f"transition pairs: {prior.counts[0]}, in-chord pairs: {prior.counts[1]}")
    prior.save(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
