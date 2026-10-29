"""Run fingering + true-release extraction on many videos.

  python scripts/run_fingering.py --inputs '/path/pm10m_npz/*.npz' --prior ~/work/prior.npz \
      --out ~/work/fing --workers 16
"""
import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmfinger import STATUS, Config, FingerPrior, apply_overrides, load_performance, run_fingering, save_result  # noqa: E402
from pmfinger.calibrate import Calibrator  # noqa: E402
from pmfinger.schema import check_output_path  # noqa: E402


def work(job):
    path, prior_path, out_dir, overrides, calib_path = job
    cfg = apply_overrides(Config(), overrides)
    prior = FingerPrior.load(prior_path) if prior_path else FingerPrior.default()
    calib = Calibrator.load(calib_path) if calib_path else None
    perf = load_performance(path)
    res = run_fingering(perf, prior, cfg, calib)
    save_result(res, os.path.join(out_dir, f"{perf.video_id}.fing.npz"))
    st = np.bincount(res["status"], minlength=len(STATUS))
    return dict(video_id=perf.video_id, notes=int(perf.N), av_offset=int(res["av_offset"]),
                valid_frac=[round(float(v), 3) for v in res["valid_frac"]],
                **{s: int(c) for s, c in zip(STATUS, st)},
                coverage=round(float(st[0] / max(perf.N, 1)), 3),
                bias=np.round(res["bias"], 3).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--prior", default="", help="prior.npz from train_prior.py (default: hand-written prior)")
    ap.add_argument("--calib", default="", help="calib.npz from fit_calibration.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    check_output_path(args.out)
    os.makedirs(args.out, exist_ok=True)
    files = sorted(glob.glob(args.inputs))
    jobs = [(f, args.prior, args.out, args.set, args.calib) for f in files]
    summary = os.path.join(args.out, "summary.jsonl")
    with open(summary, "w", encoding="utf-8") as fo, ProcessPoolExecutor(args.workers) as ex:
        for f, r in zip(files, ex.map(work, jobs)):
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(r["video_id"], "notes", r["notes"], "coverage", r["coverage"], "av", r["av_offset"])
    print("summary ->", summary)


if __name__ == "__main__":
    main()
