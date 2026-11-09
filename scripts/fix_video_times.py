"""Fix the video release's questions whose times reach past the end of the clip (audit_bench.py: time_in_clip,
example_is_answer). Rewrites <video_release>/questions.jsonl and index.html in place; the old file is kept as
questions.before_time_fix.jsonl. The videos are not touched.

  python3 -B scripts/fix_video_times.py --video <video_release_dir> --bench <bench_dir>

Why the times overshoot: the frame release counts a clip as frames / 8 fps, but the video clip stops at the last
delivered frame, so it is a little shorter (up to 0.12 s). Two kinds of question are affected:
  * a window "从第 a 秒到第 b 秒（不含）" whose end b lies after the clip end: b is cut back to the clip end, but only
    when the recorded key presses/releases show nothing happens in the cut piece (the answer stays valid);
    otherwise the question is left alone and reported
  * time-choice questions with an option later than the clip end: the options are drawn again within the clip
    (same generator, same seed) and the answer letter and the explanation conclusion follow
  * a fill-in question whose "例如 {X}" example is the answer: the example is changed to another key
Nothing is deleted. Run audit_bench.py again afterwards.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import audit_bench as AB  # noqa: E402
import export_bench_gold as EB  # noqa: E402
from export_video_qa import write_index  # noqa: E402
from make_explanations import release_conclusion  # noqa: E402

END = re.compile(r"到第\s*([0-9]+(?:\.[0-9]+)?)\s*秒（不含）")
EXAMPLE_KEY = re.compile(r"例如\s*\{([^}]*)\}")
ALT_KEYS = ["G#3", "E4", "B3"]


def load(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def fix_row(r, bench, expl):
    """Returns (changed, notes)."""
    qid, dur = r["qid"], r["duration_s"]
    notes, changed = [], False
    kinds = {k for k, _ in AB.check_one(r, Path("."), None) if k in ("time_in_clip", "example_is_answer")}
    # 1. example equals the answer
    if "example_is_answer" in kinds:
        alt = next(k for k in ALT_KEYS if k != r["answer"])
        r["question"] = EXAMPLE_KEY.sub(lambda m: m.group(0).replace(m.group(1), alt), r["question"], count=1)
        notes.append(f"example -> {{{alt}}}")
        changed = True
    if "time_in_clip" not in kinds:
        return changed, notes
    b = bench.get(qid)
    if b is None:
        return changed, notes + ["NOT FIXED: question not found in the bench folder"]
    # 2. options later than the clip end
    opts = r.get("options") or []
    late = [o for o in opts if any(t > dur + 0.05 for t in AB.times_in(str(o)))]
    if late:
        ans = b["private_target"]["answer"]
        at = "time_span" if isinstance(ans, list) else "time_point"
        head = r["question"].split("选最接近的一项")[0]
        tc = EB.time_choice(head, at, ans, dur, qid, None)
        if tc is None:
            return changed, notes + [f"NOT FIXED: no 4 options fit inside {dur:.2f} s ({late})"]
        stem, choices, letter = tc
        r["question"], r["options"], r["answer"] = stem, choices, letter
        if expl.get(qid):
            r["explanation"] = release_conclusion(expl[qid], dict(answer_type="choice", options=choices, answer=letter))
        notes.append(f"options {opts} -> {choices}, answer {letter}")
        changed = True
    # 3. window end after the clip end
    for m in list(END.finditer(r["question"].split("选项：")[0])):
        x = float(m.group(1))
        if x <= dur + 0.05:
            continue
        seen = [a for a in (b.get("atoms") or []) if a.get("predicate") in ("按下", "松开") and dur <= a["t"] < x]
        if seen:
            notes.append(f"NOT FIXED: window end {x} > clip {dur:.2f} s and {len(seen)} key events fall in between")
            continue
        r["question"] = r["question"].replace(m.group(0), f"到第 {dur:.3f} 秒（不含）")
        r["explanation"] = r.get("explanation", "").replace(f"到第 {x:.2f} 秒（不含）", f"到第 {dur:.2f} 秒（不含）")
        notes.append(f"window end {x} -> {dur:.3f}")
        changed = True
    return changed, notes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="export_video_qa.py output")
    ap.add_argument("--bench", required=True, help="build_bench_gold.py output (questions.jsonl with atoms, explanations.jsonl)")
    a = ap.parse_args()
    vdir, bdir = Path(a.video), Path(a.bench)
    rows = load(vdir / "questions.jsonl")
    bench = {r["qid"]: r for r in load(bdir / "questions.jsonl")}
    expl = {}
    if (bdir / "explanations.jsonl").exists():
        expl = {r["qid"]: r["explanation"] for r in load(bdir / "explanations.jsonl")}
    n_fixed, todo = 0, []
    for r in rows:
        ch, notes = fix_row(r, bench, expl)
        if ch:
            n_fixed += 1
        left = [k for k, _ in AB.check_one(r, vdir, None) if k in ("time_in_clip", "example_is_answer")]
        if notes:
            print(f"{r['id']} {r['qid']} {r['subtype']}: " + " | ".join(notes) + (f"  [still: {left}]" if left else ""))
        if left:
            todo.append(r["qid"])
    backup = vdir / "questions.before_time_fix.jsonl"
    if not backup.exists():
        backup.write_text((vdir / "questions.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    with open(vdir / "questions.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_index(rows, vdir)
    print(f"changed {n_fixed} questions; {len(todo)} still flagged -> {vdir / 'questions.jsonl'} (old: {backup.name})")


if __name__ == "__main__":
    main()
