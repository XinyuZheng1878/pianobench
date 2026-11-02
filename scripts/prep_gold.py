"""Benchmark step 1: assemble the PianoVAM inputs of the 11 gold-fingering recordings.

Big files (Video, Handskeleton, TSV, metadata.json) are linked from an existing full PianoVAM copy;
only the small label files (Fingering, Fingering_GT) are downloaded. Nothing in --full is modified.

  HF_ENDPOINT=https://hf-mirror.com python3 -B scripts/prep_gold.py \
      --full /inspire/hdd/global_user/xuguangjun-253208580299/piano_zsy/data/PianoVAM --out ../gold_data

Writes <out>/{metadata.json, Video/, Handskeleton/, TSV/, Fingering/, Fingering_GT/} and <out>/gold_ids.txt.
The gold list is every recording that has a Fingering_GT file in the HuggingFace repo (expected: 11).
"""
import argparse
import json
import os
from pathlib import Path

REPO = "PianoVAM/PianoVAM_v1"
LINKED = {"Video": ".mp4", "Handskeleton": ".json", "TSV": ".tsv"}
LABELS = ("Fingering", "Fingering_GT")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", required=True, help="existing full PianoVAM copy (read only)")
    ap.add_argument("--out", required=True, help="new folder for the gold-recording inputs")
    ap.add_argument("--expect", type=int, default=11, help="number of gold recordings expected")
    a = ap.parse_args()
    from huggingface_hub import HfApi, hf_hub_download

    full, out = Path(a.full).resolve(), Path(a.out).resolve()
    if out == full or out.is_relative_to(full):
        raise SystemExit("--out must not be inside --full")
    files = set(HfApi().list_repo_files(REPO, repo_type="dataset"))
    gold = sorted(Path(p).stem for p in files if p.startswith("Fingering_GT/") and p.endswith(".tsv"))
    print(f"gold recordings on the hub: {len(gold)}")
    if len(gold) != a.expect:
        raise SystemExit(f"expected {a.expect} gold recordings, found {len(gold)}: {gold}")

    out.mkdir(parents=True, exist_ok=True)
    # metadata: the hub copy is the reference; the full copy must agree so linked videos match it
    hf_hub_download(REPO, "metadata.json", repo_type="dataset", local_dir=str(out))
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    if (full / "metadata.json").read_bytes() != (out / "metadata.json").read_bytes():
        raise SystemExit("metadata.json in --full differs from the hub copy; check which PianoVAM version it is")
    ids = {e["record_time"] for e in (meta.values() if isinstance(meta, dict) else meta)}
    missing_meta = [g for g in gold if g not in ids]
    if missing_meta:
        raise SystemExit(f"gold recordings missing from metadata.json: {missing_meta}")

    problems = []
    for sub, ext in LINKED.items():
        (out / sub).mkdir(exist_ok=True)
        for g in gold:
            src, dst = full / sub / (g + ext), out / sub / (g + ext)
            if not src.is_file():
                problems.append(str(src))
                continue
            if dst.is_symlink() or dst.exists():
                if dst.resolve() != src.resolve():
                    raise SystemExit(f"{dst} exists and points elsewhere")
                continue
            os.symlink(src, dst)
    if problems:
        raise SystemExit("missing in --full:\n  " + "\n  ".join(problems))

    for sub in LABELS:
        for g in gold:
            p = f"{sub}/{g}.tsv"
            if p not in files:
                raise SystemExit(f"{p} is not on the hub")
            if not (out / p).is_file():
                hf_hub_download(REPO, p, repo_type="dataset", local_dir=str(out))
    (out / "gold_ids.txt").write_text("\n".join(gold) + "\n", encoding="utf-8")

    for g in gold:
        split = next(e.get("split") for e in (meta.values() if isinstance(meta, dict) else meta) if e["record_time"] == g)
        print(f"  {g}  split={split}")
    print(f"done: {len(gold)} recordings -> {out}  (list in {out / 'gold_ids.txt'})")


if __name__ == "__main__":
    main()
