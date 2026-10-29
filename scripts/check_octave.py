"""Whole-video octave / key-offset sanity check for exported PM10M performances.

For each video, shift every MIDI pitch by s semitones (s in -24, -12, -1, 0, +1, +12, +24) and
measure how often some fingertip lies on the shifted key at the onset (best AV offset per shift).
The exported octave is right only if s = 0 wins by a clear margin.

  python -B scripts/check_octave.py --inputs 'pm10m20/*.npz' --out octave_check.json
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmfinger import Config, load_performance  # noqa: E402
from pmfinger.evidence import _gather, onset_frames  # noqa: E402
from pmfinger.keys import KEY_X, PITCH_MAX, PITCH_MIN  # noqa: E402
from pmfinger.preprocess import clean_tips  # noqa: E402
from pmfinger.schema import check_output_path  # noqa: E402

SHIFTS = (-24, -12, -1, 0, 1, 12, 24)


def hit_rate(tips, ok, pitch, onset_s, fps, av, tol=0.5):
    f = onset_frames(onset_s, fps, av)[:, None]
    x, inb = _gather(tips[..., 0], f)
    v, _ = _gather(ok, f)
    d = np.abs(x[:, 0] - KEY_X[pitch][:, None, None])
    d = np.where((v[:, 0] & inb)[..., None], d, np.inf)
    dmin = np.nan_to_num(d, nan=np.inf).min((1, 2))
    has = np.isfinite(dmin)
    return float((dmin[has] < tol).mean()) if has.any() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--av-search', type=int, default=12)
    ap.add_argument('--min-margin', type=float, default=0.15)
    args = ap.parse_args()
    check_output_path(args.out)
    cfg = Config()
    rows = []
    for path in sorted(glob.glob(args.inputs)):
        perf = load_performance(path)
        tips, ok = clean_tips(perf.tips, perf.valid, cfg)
        res = {}
        for s in SHIFTS:
            p = perf.pitch + s
            keep = (p >= PITCH_MIN) & (p <= PITCH_MAX)
            best = max((hit_rate(tips, ok, p[keep], perf.onset[keep], perf.fps, a), a)
                       for a in range(-args.av_search, args.av_search + 1))
            res[s] = dict(rate=round(best[0], 4), av=best[1], frac_notes=round(float(keep.mean()), 3))
        rates = {s: r['rate'] for s, r in res.items()}
        win = max(rates, key=rates.get)
        second = max(v for s, v in rates.items() if s != win)
        m_oct = rates[0] - max(rates[s] for s in (-24, -12, 12, 24))
        m_semi = rates[0] - max(rates[-1], rates[1])
        rows.append(dict(video_id=perf.video_id, best_shift=win, best_rate=rates[win], second_rate=second,
                         rate_at_0=rates[0], margin_octave=round(m_oct, 4), margin_semitone=round(m_semi, 4),
                         av_at_0=res[0]['av'], av_at_edge=abs(res[0]['av']) == args.av_search,
                         ok=bool(win == 0 and m_oct >= args.min_margin), shifts=res))
    Path(args.out).write_text(json.dumps(rows, indent=1))
    print('video | best_shift | rate@0 | best_other_octave | margin_octave | margin_semitone | av@0 | OK')
    for r in rows:
        print(f"{r['video_id']} | {r['best_shift']:+d} | {r['rate_at_0']:.3f} | "
              f"{r['rate_at_0'] - r['margin_octave']:.3f} | {r['margin_octave']:.3f} | {r['margin_semitone']:.3f} | "
              f"{r['av_at_0']}{' EDGE' if r['av_at_edge'] else ''} | {'OK' if r['ok'] else 'CHECK'}")
    bad = [r['video_id'] for r in rows if not r['ok']]
    print(f'OCTAVE_CHECK_DONE {len(rows) - len(bad)}/{len(rows)} ok; needs review: {bad}')


if __name__ == '__main__':
    main()
