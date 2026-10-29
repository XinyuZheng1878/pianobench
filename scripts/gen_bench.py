"""Step 5: generate Benchmark questions (T1-T5 + control) from performance + fingering results.

  python -B scripts/gen_bench.py --perf 'work/perf/2024-*[0-9].npz' --fing work/fing --out work/bench \
      --per-task 30 [--videos 2024-09-02_14-10-41 ...] [--group-prob 0.25]

Writes
  questions.jsonl   one question per line (format in the Benchmark tab; hidden fields under clip/review/tags)
  facts.jsonl       one checked fact per line; questions of a fact share fact_id
  review.tsv        facts that need human confirmation (depend on fingering or key release)
  system_prompt.txt
  gen_report.txt    counts per task / subtype / answer, attempts and rejections
"""
import argparse
import collections
import glob
import json
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmfinger.bench import GENERATORS, SYSTEM_PROMPT, BenchConfig, Sampler, answer_key, expand  # noqa: E402
from pmfinger.schema import check_output_path  # noqa: E402
from pmfinger.states import load_derived  # noqa: E402

warnings.simplefilter("ignore", RuntimeWarning)


def overlaps(f, g):
    a = f["span"].get("ta", f["span"]["c0"]), f["span"].get("tb", f["span"]["c1"])
    b = g["span"].get("ta", g["span"]["c0"]), g["span"].get("tb", g["span"]["c1"])
    return f["video_id"] == g["video_id"] and a[0] < b[1] and b[0] < a[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perf", required=True)
    ap.add_argument("--fing", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--per-task", type=int, default=30, help="facts per task (split evenly over subtypes)")
    ap.add_argument("--control-frac", type=float, default=0.05)
    ap.add_argument("--with-control", action="store_true",
                    help="also generate C1 control questions (pedal, answer 'cannot tell'); off by default: "
                         "the benchmark keeps only questions with one unambiguous, data-backed answer")
    ap.add_argument("--group-prob", type=float, default=0.25)
    ap.add_argument("--max-tries", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    check_output_path(a.out)
    os.makedirs(a.out, exist_ok=True)
    cfg = BenchConfig(group_prob=a.group_prob)
    rng = np.random.default_rng(a.seed)

    ders, metas = [], {}
    for p in sorted(glob.glob(a.perf)):
        vid = Path(p).stem
        if a.videos and vid not in a.videos:
            continue
        fp = os.path.join(a.fing, f"{vid}.fing.npz")
        if not os.path.exists(fp):
            continue
        ders.append(load_derived(p, fp))
        mp = p.replace(".npz", ".meta.json")
        metas[vid] = json.load(open(mp, encoding="utf-8")) if os.path.exists(mp) else {}
    print("videos:", [d.vid for d in ders])
    # weight videos by length so long recordings give proportionally more facts
    w = np.array([d.T for d in ders], float)
    w /= w.sum()

    by_task = collections.defaultdict(list)
    for (task, sub) in GENERATORS:
        if task == "C1" and not a.with_control:
            continue
        by_task[task].append(sub)
    targets = {}
    for task, subs in by_task.items():
        tot = max(1, round(a.per_task * 5 * a.control_frac)) if task == "C1" else a.per_task
        for sub in subs:
            targets[task, sub] = math.ceil(tot / len(subs))

    facts, report = [], []
    for (task, sub), n_target in targets.items():
        fn = GENERATORS[task, sub]
        got, tries, ans_cnt = [], 0, collections.Counter()
        while len(got) < n_target and tries < a.max_tries:
            tries += 1
            d = ders[int(rng.choice(len(ders), p=w))]
            f = fn(Sampler(d, rng, cfg))
            if f is None:
                continue
            k = answer_key(f)
            n_cls = len(f["mc"]["options"]) if f.get("mc") else 6
            if task != "C1" and f.get("mc") and ans_cnt[k] >= math.ceil(n_target / n_cls) + 1:
                continue
            if any(overlaps(f, g) for g in got):
                continue
            ans_cnt[k] += 1
            got.append(f)
        facts += got
        report.append(f"{task} {sub}: {len(got)}/{n_target} facts in {tries} tries; answers {dict(ans_cnt)}")
        print(report[-1])

    qs = []
    der_by = {d.vid: d for d in ders}
    for f in facts:
        for q in expand(der_by[f["video_id"]], f, rng, cfg, metas.get(f["video_id"])):
            qs.append(q)
    cnt = collections.Counter()
    order = {t: i for i, t in enumerate(["T1", "T2", "T3", "T4", "T5", "C1"])}
    qs.sort(key=lambda q: (order[q["task"]], q["subtype"], q["fact_id"]))
    for q in qs:
        cnt[q["task"]] += 1
        q["qid"] = f"{q['task']}-{cnt[q['task']]:04d}"
    with open(os.path.join(a.out, "questions.jsonl"), "w", encoding="utf-8") as fo:
        for q in qs:
            fo.write(json.dumps({"qid": q.pop("qid"), **q}, ensure_ascii=False) + "\n")
    with open(os.path.join(a.out, "facts.jsonl"), "w", encoding="utf-8") as fo:
        for f in facts:
            fo.write(json.dumps(f, ensure_ascii=False, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)) + "\n")
    with open(os.path.join(a.out, "review.tsv"), "w", encoding="utf-8") as fo:
        fo.write("fact_id\ttask\tsubtype\tvideo_id\tk_notes\tmin_conf\tlabel_tier\tdepends\tkeep(1/0)\tnote\n")
        rv = [f for f in facts if {"finger", "release"} & set(f["depends"])]
        rv.sort(key=lambda f: (-(f["k_notes"] or 0), f["min_conf"] or 1))
        for f in rv:
            fo.write(f"{f['fact_id']}\t{f['task']}\t{f['subtype']}\t{f['video_id']}\t{f['k_notes']}\t{f['min_conf']}\t"
                     f"{f['label_tier']}\t{','.join(f['depends'])}\t\t\n")
    Path(a.out, "system_prompt.txt").write_text(SYSTEM_PROMPT + "\n", encoding="utf-8")
    by = collections.Counter((q["task"], q["subtype"]) for q in [json.loads(l) for l in open(os.path.join(a.out, "questions.jsonl"), encoding="utf-8")])
    lines = report + ["", f"facts {len(facts)}, questions {len(qs)}, needing review {sum(1 for f in facts if {'finger','release'} & set(f['depends']))} facts"]
    lines += [f"  {t} {s}: {n} questions" for (t, s), n in sorted(by.items())]
    Path(a.out, "gen_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[len(report):]))


if __name__ == "__main__":
    main()
