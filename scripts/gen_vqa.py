"""Generate VQA jsonl from performance npz + fingering results.

  python scripts/gen_vqa.py --perf '/path/pm10m_npz/*.npz' --fing ~/work/fing \
      --out ~/work/vqa/pm10m_vqa.jsonl --lang zh en --per-type 3
"""
import argparse
import collections
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmfinger import load_performance, load_result  # noqa: E402
from pmfinger.schema import check_output_path  # noqa: E402
from pmfinger.vqa import GENERATORS, VQAConfig, generate_for_video  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perf", required=True, help="glob of performance npz")
    ap.add_argument("--fing", required=True, help="dir with <video_id>.fing.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", nargs="+", default=["zh"], choices=["zh", "en"])
    ap.add_argument("--per-type", type=int, default=3)
    ap.add_argument("--types", nargs="*", default=None, choices=list(GENERATORS))
    ap.add_argument("--min-coverage", type=float, default=0.6,
                    help="skip videos where fewer than this fraction of notes got a confident label")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    check_output_path(args.out)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cfg = VQAConfig(per_type=args.per_type)
    stats, skipped = collections.Counter(), []
    with open(args.out, "w", encoding="utf-8") as fo:
        for i, path in enumerate(sorted(glob.glob(args.perf))):
            perf = load_performance(path)
            fpath = os.path.join(args.fing, f"{perf.video_id}.fing.npz")
            if not os.path.exists(fpath):
                skipped.append((perf.video_id, "no fingering result"))
                continue
            res = load_result(fpath)
            cov = float((res["status"] == 0).mean()) if len(res["status"]) else 0.0
            if cov < args.min_coverage:
                skipped.append((perf.video_id, f"coverage {cov:.2f}"))
                continue
            rng = np.random.default_rng([args.seed, i])
            for rec in generate_for_video(perf, res, rng, tuple(args.lang), cfg, args.types):
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats[rec["type"]] += 1
    print("samples per type:", dict(stats), "total", sum(stats.values()))
    print(f"skipped {len(skipped)} videos", skipped[:10])


if __name__ == "__main__":
    main()
