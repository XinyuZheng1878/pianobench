"""List which SHCM takes got score (Hanon) fingering, from the meta.json files export_shcm.py wrote.

  python3 -B scripts/score_fingering_report.py <work>/perf
"""
import json
import sys
from pathlib import Path


def main():
    rows = []
    for m in sorted(Path(sys.argv[1]).glob("shcm-*.meta.json")):
        d = json.loads(m.read_text(encoding="utf-8"))
        s = d.get("score_fingering", {})
        rows.append([d["video_id"], d["piece"], s.get("hanon") or "-", "yes" if s.get("used") else "no",
                     s.get("match_midi", "-"), s.get("octave_shift", "-"), d.get("n_gt", 0), d["n_notes"],
                     s.get("reason", "")])
    print("\t".join(["take", "folder", "hanon", "labelled", "match", "octave_shift", "gt_fingers", "notes", "why_not"]))
    for r in rows:
        print("\t".join(map(str, r)))
    used = [r for r in rows if r[3] == "yes"]
    print(f"# {len(used)}/{len(rows)} takes labelled, {sum(r[6] for r in used)} notes with score fingering",
          file=sys.stderr)


if __name__ == "__main__":
    main()
