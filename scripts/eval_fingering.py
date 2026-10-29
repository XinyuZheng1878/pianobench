"""Evaluate on PianoVAM human GT (label_* fields), optionally sweeping config values.

  python scripts/eval_fingering.py --inputs '/path/pianovam_gt_npz/*.npz' --prior ~/work/prior.npz
  python scripts/eval_fingering.py ... --sweep sigma_white=0.4,0.5,0.6 w_seq=0.5,1,2

Never tune on these 11 pieces and then report the same number; tune on a held-out split of the
auto-labelled pieces (pass them as --inputs) and only run the GT pieces once at the end.
"""
import argparse
import glob
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmfinger import Config, FingerPrior, apply_overrides, load_performance, run_fingering  # noqa: E402
from pmfinger.calibrate import Calibrator  # noqa: E402
from pmfinger.metrics import count_metrics, summarize  # noqa: E402


def work(job):
    path, prior_path, overrides, calib_path = job
    cfg = apply_overrides(Config(), overrides)
    prior = FingerPrior.load(prior_path) if prior_path else FingerPrior.default()
    calib = Calibrator.load(calib_path) if calib_path else None
    perf = load_performance(path)
    if perf.label_hand is None or perf.label_finger is None:
        return None
    res = run_fingering(perf, prior, cfg, calib)
    return count_metrics(res, perf.label_hand, perf.label_finger)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--prior", default="")
    ap.add_argument("--calib", default="", help="calib.npz from fit_calibration.py (learned confidence)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--sweep", nargs="*", default=[], help="key=v1,v2,... (cartesian product)")
    args = ap.parse_args()
    files = sorted(glob.glob(args.inputs))
    keys = [s.split("=", 1)[0] for s in args.sweep]
    vals = [s.split("=", 1)[1].split(",") for s in args.sweep]
    with ProcessPoolExecutor(args.workers) as ex:
        for combo in itertools.product(*vals) if vals else [()]:
            ov = list(args.set) + [f"{k}={v}" for k, v in zip(keys, combo)]
            counts = [c for c in ex.map(work, [(f, args.prior, ov, args.calib) for f in files]) if c is not None]
            s = summarize(counts)
            print(json.dumps(dict(overrides=ov, **s), ensure_ascii=False))


if __name__ == "__main__":
    main()
