"""Fit the learned confidence (calibrate.Calibrator) on labelled TRAINING pieces (auto labels).

  python -B scripts/fit_calibration.py --inputs 'split/train/*.npz' --prior prior.npz \
      --set sigma_white=0.4 w_seq=1 --out calib.npz
Evaluate it on the validation split with eval_fingering.py --calib calib.npz.
"""
import argparse
import glob
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmfinger import Config, FingerPrior, apply_overrides, load_performance, run_fingering  # noqa: E402
from pmfinger.calibrate import Calibrator, note_features  # noqa: E402
from pmfinger.schema import check_output_path  # noqa: E402


def work(job):
    path, prior_path, overrides = job
    cfg = apply_overrides(Config(), overrides)
    prior = FingerPrior.load(prior_path) if prior_path else FingerPrior.default()
    perf = load_performance(path)
    if perf.label_hand is None:
        return None
    res = run_fingering(perf, prior, cfg)
    lab = (perf.label_hand >= 0) & (perf.label_finger >= 0) & (res['finger_pred'] >= 0)
    y = (res['hand_pred'] == perf.label_hand) & (res['finger_pred'] == perf.label_finger)
    return note_features(res)[lab], y[lab].astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', required=True)
    ap.add_argument('--prior', default='')
    ap.add_argument('--set', nargs='*', default=[])
    ap.add_argument('--l2', type=float, default=1e-2)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    check_output_path(args.out)
    files = sorted(glob.glob(args.inputs))
    with ProcessPoolExecutor(args.workers) as ex:
        parts = [p for p in ex.map(work, [(f, args.prior, args.set) for f in files]) if p is not None]
    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    cal = Calibrator.fit(X, y, l2=args.l2)
    p = cal.predict(X)
    print(f'fitted on {len(y)} notes from {len(parts)} pieces; base accuracy {y.mean():.4f}; '
          f'log-loss {-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)):.4f}')
    cal.save(args.out)
    print('saved', args.out)


if __name__ == '__main__':
    main()
