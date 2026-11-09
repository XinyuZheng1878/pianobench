"""Benchmark step 3: generate and render test questions from the 11 gold-fingering recordings.

Uses the P0 rules (T2 which finger, T3, T4, T5; gold-only fingers, 40 ms onset groups, visibility gates,
T5 'now' = last delivered frame) plus pmfinger/bench_gold.py (T1 and T2 which hand, same gates).
No model API is called. Answers are computed by the program only.

  python3 -B scripts/build_bench_gold.py --work ../gold_work --gold-ids ../gold_data/gold_ids.txt \
      --out ../gold_work/bench_v1 --per-subtype 20

Every window of every recording is scanned (stride 0.5 s); then, per subtype, questions are picked so that the
answers are as even as possible and the recordings take turns, with no two questions of one subtype overlapping
in time in the same recording. Writes into --out (must be new):
  questions.jsonl      one row per question (public question, private answer, cited facts, media)
  results/<qid>.json   the same, one file per question
  media/<qid>/         delivered frames (frame_*.jpg) and 3-frame packs
  report.md            counts per subtype / answer / recording, and why candidates were rejected
  index.html           page to look through the questions with their frames and answers
"""
import argparse
import collections
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
import p0_restore as P0  # noqa: E402  (loads p0_rules / p0_states / p0_templates)
from batch_sft import atomic_json, digest, file_hash, frame_pack  # noqa: E402
from pmfinger import bench_gold as BG  # noqa: E402

PROFILE = "pianobench-gold-v1"
EXTRA = BG.GENERATORS                                    # T1/*, T2/哪只手
FINGER = P0.FINGER


def freeze(P, f, source_hash):
    """Validate with the P0 gates and freeze the public question / private answer."""
    f = P0.validate_fact(P, f)
    key = f["task"] + "/" + f["subtype"]
    public = {k: f[k] for k in ("task", "subtype", "level", "video_id", "clip", "frames", "range", "predict",
                                "slots", "answer_type")}
    if key in EXTRA:
        public["semantic_id"] = PROFILE + "/" + key
        public["question"] = BG.bench_question(f)
    else:
        public["semantic_id"] = P0.PROFILE + "/" + P0.REGISTRY[key]
        public["question"] = (f"这段视频里，{f['slots']['hand']}按下 {f['slots']['key']} 的那一刻用的是哪根手指？"
                              + P0.templates.opts_text(f) if f["task"] == "T2" and f["subtype"] == "哪根手指"
                              else P0.templates.bench_question(f))
    public["split"] = "bench_gold"
    if f.get("options"):
        public["options"] = f["options"]["choices"]
    target = dict(answer=f["answer"], difficulty=f["difficulty"])
    if f["task"] == "T5":
        cut = f["frames"][-1]["video_frame"]
        target.update(cut_native=cut, target_native=cut + round(f["predict"]["horizon"] * P.fps))
    fact_hash = digest([public, dict(private_target=target), f["atoms"], source_hash])
    needs = f["subtype"] in FINGER
    return dict(qid=PROFILE + "-" + fact_hash[:20], fact_hash=fact_hash, source_hash=source_hash, question=public,
                atoms=f["atoms"], private_target=target, needs_human=needs,
                status="needs_human_GT_review" if needs else "candidate")


def answer_class(row):
    q, a = row["question"], row["private_target"]["answer"]
    if q.get("options"):
        return q["options"]["ABCDEFG".index(a)]
    if q["answer_type"] == "int":
        return str(a)
    if q["answer_type"] in ("time_point", "time_span"):
        t = a if isinstance(a, (int, float)) else a[0]
        L = q["frames"][-1]["t"] or 1
        return ["early", "middle", "late"][min(2, int(3 * t / L))]
    if q["answer_type"] == "key_seq":
        return f"len{len(a)}"
    return str(a)


def scan(P, source_hash, rnd, stride_s, passes=1):
    """All eligible candidates of a recording: {key: [(c0, c1, row)]}, plus rejection counts."""
    g = BG.BenchGen(P, rnd)
    cand = collections.defaultdict(list)
    rejected = collections.defaultdict(collections.Counter)
    seen = set()

    def take(key, f, c0, c1):
        try:
            if f is None or f["task"] + "/" + f["subtype"] != key:
                raise P0.GateError("no_eligible_fact_under_spec_gates")
            row = freeze(P, f, source_hash)
            if row["qid"] in seen:
                raise P0.GateError("exact_duplicate")
            seen.add(row["qid"])
            cand[key].append((c0, c1, row))
        except (ValueError, IndexError, StopIteration) as e:
            rejected[key][str(e) or type(e).__name__] += 1

    step = max(1, round(P.fps * stride_s))
    for c0 in range(0, P.T - round(2.6 * P.fps), step):
        c1 = c0 + round(5 * P.fps)
        for _ in range(passes if c1 <= P.T else 0):       # each pass draws the hand / note again
            for sub, method in P0.T3.items():
                take("T3/" + sub, getattr(g, method)(c0, c1), c0, c1)
            take("T2/哪根手指", P0.t2_finger(g, c0, c1), c0, c1)
            for key, method in EXTRA.items():
                take(key, getattr(g, method)(c0, c1), c0, c1)
        c1 = c0 + round(6 * P.fps)
        if c1 <= P.T:
            found = {f["subtype"]: f for f in g.t4(c0, c1)}
            for sub in P0.T4:
                take("T4/" + sub, found.get(sub), c0, c1)
        cut = c0 + round(2 * P.fps)
        for h in (0, 1):
            found = collections.defaultdict(list)
            for f in g.t5(cut, h):
                found[f["subtype"]].append(f)
            for sub in P0.T5:
                for f in found.get(sub, [None]):
                    take("T5/" + sub, f, c0, cut + 1)
    return cand, rejected


def row_span(row):
    fr = row["question"]["frames"]
    return fr[0]["video_frame"], fr[-1]["video_frame"] + 1


def select(pool, n, rnd, existing=()):
    """Pick until the subtype has n questions (existing ones count): the answer that is rarest so far goes
    first, the recording used least so far goes first, and no two questions of one subtype overlap in time
    in the same recording."""
    have = {r["qid"] for r in existing}
    by_cls = collections.defaultdict(lambda: collections.defaultdict(list))
    for vid, c0, c1, row in pool:
        if row["qid"] not in have:
            by_cls[answer_class(row)][vid].append((c0, c1, row))
    for vids in by_cls.values():
        for lst in vids.values():
            rnd.shuffle(lst)
    spans, turn, count = collections.defaultdict(list), collections.Counter(), collections.Counter()
    for r in existing:
        v = r["question"]["video_id"]
        spans[v].append(row_span(r))
        turn[v] += 1
        count[answer_class(r)] += 1
    taken = []
    tie = {c: rnd.random() for c in by_cls}
    while len(existing) + len(taken) < n:
        live = [c for c in by_cls if any(by_cls[c].values())]
        if not live:
            break
        cls = min(live, key=lambda c: (count[c], tie[c]))
        picked = False
        for v in sorted((v for v in by_cls[cls] if by_cls[cls][v]), key=lambda v: (turn[v], v)):
            lst = by_cls[cls][v]
            while lst:
                c0, c1, row = lst.pop()
                if any(max(c0, a) < min(c1, b) for a, b in spans[v]):
                    continue
                taken.append(row)
                spans[v].append((c0, c1))
                turn[v] += 1
                count[cls] += 1
                picked = True
                break
            if picked:
                break
        if not picked:
            by_cls.pop(cls)                       # every remaining candidate of this answer overlaps
    return taken


def report(out, rows, pools, rejected, videos):
    L = ["# Benchmark 出题报告", "",
         f"题目总数：{len(rows)}；其中需人工复核指法的：{sum(r['needs_human'] for r in rows)}", "",
         "| 子题 | 候选 | 入选 | 答案分布 | 录音分布 |", "|---|---:|---:|---|---|"]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["question"]["task"] + "/" + r["question"]["subtype"]].append(r)
    for key in sorted(set(pools) | set(by)):
        sel = by.get(key, [])
        ans = collections.Counter(answer_class(r) for r in sel)
        vids = collections.Counter(r["question"]["video_id"][5:16] for r in sel)
        L.append(f"| {key} | {len(pools.get(key, []))} | {len(sel)} | {dict(ans)} | {dict(vids)} |")
    L += ["", "## 每段录音的入选题数", ""]
    vc = collections.Counter(r["question"]["video_id"] for r in rows)
    L += [f"- {v}: {vc.get(v, 0)}" for v in videos]
    L += ["", "## 候选被拒原因（前 5 项）", ""]
    for key in sorted(rejected):
        top = rejected[key].most_common(5)
        L.append(f"- {key}: " + "；".join(f"{k} ×{n}" for k, n in top))
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, help="folder with perf/ and fing/ from `bench_gold.sh derive`")
    ap.add_argument("--gold-ids", required=True)
    ap.add_argument("--out", required=True, help="new folder")
    ap.add_argument("--per-subtype", type=int, default=20, help="questions per subtype over all recordings")
    ap.add_argument("--stride", type=float, default=0.5, help="window step in seconds")
    ap.add_argument("--passes", type=int, default=1,
                    help="draws per window for the T1-T3 generators (each picks a hand at random; 3 ~ both hands)")
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--no-media", action="store_true", help="skip frame rendering (fast dry run)")
    ap.add_argument("--extend", default="", help="earlier bench folder: keep all its questions unchanged (same qid, "
                    "answer and frames) and only add new ones up to --per-subtype")
    a = ap.parse_args()
    if a.stride <= 0 or a.passes < 1 or a.per_subtype < 1:
        ap.error("stride, passes and per-subtype must be positive")
    work, out = Path(a.work).resolve(), Path(a.out).resolve()
    if out.exists():
        raise SystemExit(f"{out} already exists; choose a new --out")
    videos = Path(a.gold_ids).read_text(encoding="utf-8").split()
    keep = []
    if a.extend:
        prev = Path(a.extend).resolve()
        keep = [json.loads(l) for l in open(prev / "questions.jsonl", encoding="utf-8")]
        print(f"extending {prev}: keeping its {len(keep)} questions", flush=True)
    out.mkdir(parents=True)
    (out / "perf").symlink_to(work / "perf", target_is_directory=True)   # read-only, for the frame renderer
    for r in keep:                                                      # reuse the frames already rendered
        if "media" in r:
            (out / "media").mkdir(exist_ok=True)
            (out / "media" / r["qid"]).symlink_to(prev / "media" / r["qid"], target_is_directory=True)

    pools, rejected, perfs = collections.defaultdict(list), collections.defaultdict(collections.Counter), {}
    for vid in videos:
        files = [work / "perf" / (vid + s) for s in (".npz", ".meta.json", ".skeleton.npz")] + [work / "fing" / (vid + ".fing.npz")]
        source_hash = digest({p.name: file_hash(p) for p in files})
        P = P0.state.load_raw(files[0], files[-1])
        perfs[vid] = P
        cand, rej = scan(P, source_hash, random.Random(digest([a.seed, vid])), a.stride, a.passes)
        for key, lst in cand.items():
            pools[key] += [(vid, c0, c1, row) for c0, c1, row in lst]
        for key, c in rej.items():
            rejected[key].update(c)
        print(f"{vid}: {sum(len(v) for v in cand.values())} candidates", flush=True)

    rnd = random.Random(a.seed)
    kept_by = collections.defaultdict(list)
    for r in keep:
        kept_by[r["question"]["task"] + "/" + r["question"]["subtype"]].append(r)
    new = []
    for key in sorted(set(pools) | set(kept_by)):
        new += select(pools.get(key, []), a.per_subtype, rnd, kept_by.get(key, []))
    for r in new:
        r["batch"] = Path(a.out).name
    rows = keep + new
    rows.sort(key=lambda r: (r["question"]["task"], r["question"]["subtype"], r["question"]["video_id"],
                             r["question"]["clip"]["frame0"]))
    print(f"selected {len(rows)} questions ({len(keep)} kept, {len(new)} new)", flush=True)

    for i, row in enumerate(new):
        q, P = row["question"], perfs[row["question"]["video_id"]]
        if not a.no_media:
            frames = q["frames"]
            compat = dict(task=q["task"], clip_id=row["qid"], frame_fps=q["clip"]["fps_out"],
                          delivery_frames=[f["video_frame"] for f in frames],
                          clip=dict(start_frame=frames[0]["video_frame"], end_frame=frames[-1]["video_frame"] + 1),
                          predict=dict(cut=q["predict"]["t0"]) if q["predict"] else None)
            d = SimpleNamespace(vid=q["video_id"], fps=P.fps, perf=SimpleNamespace(video_path=P.video_path))
            try:
                row["media"] = frame_pack(out, compat, d)
                if [f["video_frame"] for f in frames] != [f["native_frame"] for f in row["media"]["frames"]]:
                    raise P0.GateError("actual_delivery_grid_mismatch")
            except Exception as e:  # noqa: BLE001
                row.update(status="delivery_failed", error=f"{type(e).__name__}: {e}")
        atomic_json(out / "results" / (row["qid"] + ".json"), row)
        if (i + 1) % 50 == 0:
            print(f"  rendered {i + 1}/{len(new)}", flush=True)
    for row in keep:
        atomic_json(out / "results" / (row["qid"] + ".json"), row)
    ok = [r for r in rows if r["status"] != "delivery_failed"]
    with open(out / "questions.jsonl", "w", encoding="utf-8") as fo:
        for r in ok:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    report(out, ok, pools, rejected, videos)
    if not a.no_media:
        P0.render_page(out, ok)
    print(json.dumps(dict(out=str(out), questions=len(ok), kept=len(keep), added=len(new),
                          delivery_failed=len(rows) - len(ok),
                          needs_human=sum(r["needs_human"] for r in ok)), ensure_ascii=False))


if __name__ == "__main__":
    main()
