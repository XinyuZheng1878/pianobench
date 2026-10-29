"""Step 4 check: motion-state statistics and how well tracked releases match the true key releases.

  python -B scripts/check_states.py --perf 'work/perf/2024-*[0-9].npz' --fing work/fing --out work/qc_states.json

PianoVAM has the Disklavier key-up time, so the benchmark uses it directly. This script reports how far
PMFinger's tracked release (what PM10M will have to rely on) is from it, and the share of frames per state.
"""
import argparse
import glob
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmfinger.pipeline import load_result  # noqa: E402
from pmfinger.states import STATE_ZH, load_derived  # noqa: E402

warnings.simplefilter("ignore", RuntimeWarning)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perf", required=True)
    ap.add_argument("--fing", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for p in sorted(glob.glob(a.perf)):
        vid = Path(p).stem
        fp = os.path.join(a.fing, f"{vid}.fing.npz")
        d = load_derived(p, fp)
        r = load_result(fp)
        frac = np.bincount(d.state.ravel(), minlength=6) / d.state.size
        row = dict(video_id=vid, hand_trust=round(float(d.hand_trust.mean()), 3),
                   finger_trust=round(float(d.finger_trust.mean()), 3),
                   state_frac={STATE_ZH[i]: round(float(frac[i]), 3) for i in range(6)})
        if d.release_true:
            m = d.finger_trust & (r["release_frame"] >= 0)
            e = (r["release_frame"][m] - d.rel[m]) / d.fps
            row.update(tracked_release_median_abs_ms=round(float(np.median(np.abs(e)) * 1000)),
                       tracked_release_within_100ms=round(float((np.abs(e) < 0.1).mean()), 3),
                       tracked_release_late_200ms=round(float((e > 0.2).mean()), 3))
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
    Path(a.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
