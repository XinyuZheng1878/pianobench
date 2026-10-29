"""Download the parts of PianoVAM_v1 needed for the QA probes.

Only selected recordings are fetched (whole dataset is ~45 GB).
In mainland China set:  export HF_ENDPOINT=https://hf-mirror.com

  python download.py --root data/pianovam                       # test split + GT recordings + blurry
  python download.py --root data/pianovam --splits test --no-skeleton --dry-run
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import HfApi, hf_hub_download

REPO = "PianoVAM/PianoVAM_v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--splits", default="test,special(blurry)",
                    help="comma list of metadata splits to fetch (train,valid,test,ext-train,special(blurry),special(4hands))")
    ap.add_argument("--no-gt", action="store_true", help="skip recordings that have Fingering_GT (used for finger_id)")
    ap.add_argument("--no-skeleton", action="store_true", help="skip Handskeleton (needed only for sync_check)")
    ap.add_argument("--limit", type=int, default=0, help="max recordings per split (0 = all)")
    ap.add_argument("--gt-limit", type=int, default=0, help="max Fingering_GT recordings (0 = all 11)")
    ap.add_argument("--smallest", action="store_true",
                    help="with --limit/--gt-limit: pick the smallest videos (>= --min-mb) instead of the first by date")
    ap.add_argument("--min-mb", type=float, default=30, help="with --smallest: skip very short recordings")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=6, help="files downloaded in parallel")
    a = ap.parse_args()
    os.makedirs(a.root, exist_ok=True)

    api = HfApi()
    info = api.dataset_info(REPO, files_metadata=True)
    sizes = {s.rfilename: (s.lfs.size if s.lfs else s.size) or 0 for s in info.siblings}
    files = set(sizes)

    hf_hub_download(REPO, "metadata.json", repo_type="dataset", local_dir=a.root)
    with open(os.path.join(a.root, "metadata.json"), encoding="utf-8") as f:
        raw = json.load(f)
    meta = list(raw.values()) if isinstance(raw, dict) else raw

    def vsize(rec):
        return sizes.get(f"Video/{rec}.mp4", 0)

    def pick(recs, n):
        recs = [r for r in recs if f"Video/{r}.mp4" in files]
        if a.smallest:
            big_enough = [r for r in recs if vsize(r) >= a.min_mb * 1e6]
            recs = sorted(big_enough or recs, key=vsize)
        return recs[: n or None]

    wanted = []
    splits = [s.strip() for s in a.splits.split(",") if s.strip()]
    for sp in splits:
        wanted += pick(sorted(e["record_time"] for e in meta if e.get("split") == sp), a.limit)
    if not a.no_gt:
        wanted += pick(sorted(p.split("/")[1][:-4] for p in files if p.startswith("Fingering_GT/")), a.gt_limit)
    wanted = list(dict.fromkeys(wanted))  # dedupe, keep order

    subdirs = ["TSV", "Fingering", "Fingering_GT", "Video"] + ([] if a.no_skeleton else ["Handskeleton"])
    ext = {"TSV": "tsv", "Fingering": "tsv", "Fingering_GT": "tsv", "Video": "mp4", "Handskeleton": "json"}
    todo = [f"{d}/{r}.{ext[d]}" for r in wanted for d in subdirs if f"{d}/{r}.{ext[d]}" in files]
    total = sum(sizes[p] for p in todo)
    print(f"{len(wanted)} recordings, {len(todo)} files, {total / 1e9:.2f} GB")
    for p in todo:
        print(f"  {p}  {sizes[p] / 1e6:.1f} MB")
    if a.dry_run:
        return
    left = [p for p in todo if not os.path.exists(os.path.join(a.root, p))]
    print(f"already here: {len(todo) - len(left)} files; downloading {len(left)} files "
          f"({sum(sizes[p] for p in left) / 1e9:.2f} GB) with {a.workers} in parallel", flush=True)
    # big files first so the parallel slots stay busy until the end
    left.sort(key=lambda p: -sizes[p])
    done_b, errs = 0, []
    with ThreadPoolExecutor(max(1, a.workers)) as ex:
        futs = {ex.submit(hf_hub_download, REPO, p, repo_type="dataset", local_dir=a.root): p for p in left}
        for k, fu in enumerate(as_completed(futs), 1):
            p = futs[fu]
            try:
                fu.result()
                done_b += sizes[p]
                print(f"  [{k}/{len(left)}] ok  {p}  (total {done_b / 1e9:.2f} GB)", flush=True)
            except Exception as e:  # keep going; the caller reruns and missing files are fetched again
                errs.append(p)
                print(f"  [{k}/{len(left)}] FAILED {p}: {type(e).__name__}: {str(e)[:200]}", flush=True)
    if errs:
        raise SystemExit(f"{len(errs)} files failed; run again to retry them")
    print("done")


if __name__ == "__main__":
    main()
