"""Benchmark step 4: turn a gold bench folder into the release that run_models.py / evaluate.py read.

  python3 -B scripts/export_bench_gold.py --bench ../gold_work/bench_v2 --out ../gold_work/release_v2 \
      [--approvals ../gold_work/review/approvals.jsonl] [--include-unreviewed]

What it does
  * answer-balance rule: within a subtype no single answer may exceed half of the questions; the surplus of
    the over-represented answer is dropped, newest batch first (the bench folder itself is not changed);
  * questions that need human review (finger from gold labels) are exported only when an approval with the
    same qid and fact_hash says "approve" (or with --include-unreviewed, for a dry evaluation run);
  * writes questions.jsonl in the evaluation schema, system_prompt.txt, media/ (linked to the bench) and
    export_report.md (what was kept, dropped and why).
"""
import argparse
import collections
import json
from pathlib import Path

LETTERS = "ABCDEFG"
ANSWER_TYPE = {"key_seq": "key_sequence"}      # bench name -> evaluate.py name; others are identical

SYSTEM_PROMPT = """你会看到一段从钢琴正上方俯拍的演奏视频（按时间顺序抽取的若干帧），画面中是琴键和演奏者的双手。
约定：
1. 左手、右手指演奏者本人的左右手；画面左侧是低音方向。
2. 手指编号：1 拇指，2 食指，3 中指，4 无名指，5 小指。
3. 键名用音名加八度编号，C4 是中央 C；黑键写作升号，如 G#3。
4. 时间以第一张帧为 0 秒，每张帧前都标有它的帧号和时间；“第 a 秒到第 b 秒（不含）之间”包含 a、不包含 b。
5. 一次“按键”指一个键被按下一次；几乎同时按下的键，以其中最早按下的时刻起 40 毫秒内为一组，不链式延长。
6. 预测题里，最后一张帧就是“现在”，问的是现在之后会发生什么。
7. 最终答案必须写在花括号里，例如 {B}、{3}、{G#3}、{E4, F#4, G4}、{1.25}、{0.50, 1.75}；花括号只用于最终答案。"""


def answer_class(q, a):
    if q.get("options"):
        return q["options"][LETTERS.index(a)]
    return json.dumps(a, ensure_ascii=False)


def balance(rows):
    """Drop the surplus of any answer above half of its subtype; drop the newest batch first."""
    by = collections.defaultdict(list)
    for r in rows:
        by[r["question"]["task"] + "/" + r["question"]["subtype"]].append(r)
    kept, dropped = [], collections.Counter()
    for key, lst in by.items():
        cnt = collections.Counter(answer_class(r["question"], r["private_target"]["answer"]) for r in lst)
        top, n_top = cnt.most_common(1)[0] if cnt else (None, 0)
        others = len(lst) - n_top
        if n_top > others and len(cnt) > 1:
            excess = n_top - others
            # newest first: rows added by --extend carry a batch name, the first batch has none
            order = sorted((r for r in lst if answer_class(r["question"], r["private_target"]["answer"]) == top),
                           key=lambda r: r.get("batch", ""), reverse=True)
            drop = {id(r) for r in order[:excess]}
            lst = [r for r in lst if id(r) not in drop]
            dropped[key] += excess
        elif len(cnt) == 1 and len(lst) > 1:
            dropped[key] += len(lst)       # a subtype with only one answer cannot be scored
            lst = []
        kept += lst
    return kept, dropped


def to_eval(r):
    q, t = r["question"], r["private_target"]
    at = ANSWER_TYPE.get(q["answer_type"], q["answer_type"])
    a = t["answer"]
    if at in ("time_point", "time_span"):
        a = round(float(a), 3) if at == "time_point" else [round(float(x), 3) for x in a]
    frames = r["media"]["frames"]
    t0 = frames[0]["t"]
    return dict(
        qid=r["qid"], fact_id=r["qid"], task=q["task"], subtype=q["subtype"], level=q["level"],
        video_id=q["video_id"], question=q["question"], answer_type=at, answer=a,
        options=q.get("options"), frame_fps=q["clip"]["fps_out"],
        media=dict(frames=[f["path"] for f in frames], frame_times=[round(f["t"] - t0, 3) for f in frames]),
        clip=dict(start_frame=frames[0]["native_frame"], end_frame=frames[-1]["native_frame"] + 1,
                  length_s=round(frames[-1]["t"] - t0 + 1.0 / q["clip"]["fps_out"], 3)),
        range=q.get("range"), predict=q.get("predict"),
        qa_type="单选" if at == "choice" else "填空",
        tags=dict(range_type=(q.get("range") or {}).get("type", "whole"), semantic_id=q["semantic_id"],
                  needs_human=r["needs_human"], batch=r.get("batch", "v1")),
        extra={}, baselines={})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--out", required=True, help="new folder")
    ap.add_argument("--approvals", default="", help="jsonl of {qid, fact_hash, decision, reviewer}")
    ap.add_argument("--include-unreviewed", action="store_true",
                    help="also export finger questions nobody has approved yet (dry evaluation only)")
    a = ap.parse_args()
    bench, out = Path(a.bench).resolve(), Path(a.out).resolve()
    if out.exists():
        raise SystemExit(f"{out} already exists; choose a new --out")
    rows = [json.loads(l) for l in open(bench / "questions.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r.get("status") != "delivery_failed" and "media" in r]

    approved, rejected = set(), set()
    if a.approvals:
        for l in open(a.approvals, encoding="utf-8"):
            x = json.loads(l)
            if not str(x.get("reviewer", "")).strip():
                continue
            key = (x["qid"], x["fact_hash"])
            (approved if x.get("decision") == "approve" else rejected).add(key)
    review_drop = collections.Counter()
    keep = []
    for r in rows:
        key = (r["qid"], r["fact_hash"])
        k = r["question"]["task"] + "/" + r["question"]["subtype"]
        if key in rejected:
            review_drop[k] += 1
            continue
        if r["needs_human"] and key not in approved and not a.include_unreviewed:
            review_drop[k] += 1
            continue
        keep.append(r)
    keep, bal_drop = balance(keep)

    out.mkdir(parents=True)
    (out / "media").symlink_to(bench / "media", target_is_directory=True)
    keep.sort(key=lambda r: (r["question"]["task"], r["question"]["subtype"], r["qid"]))
    with open(out / "questions.jsonl", "w", encoding="utf-8") as fo:
        for r in keep:
            fo.write(json.dumps(to_eval(r), ensure_ascii=False) + "\n")
    (out / "system_prompt.txt").write_text(SYSTEM_PROMPT + "\n", encoding="utf-8")

    by = collections.defaultdict(list)
    for r in keep:
        by[r["question"]["task"] + "/" + r["question"]["subtype"]].append(r)
    keys = sorted(set(by) | set(bal_drop) | set(review_drop))
    L = ["# 测评集导出报告", "", f"来源：{bench}", f"题目：{len(keep)}（导出前 {len(rows)}）",
         f"未经人工批准而排除的指法题：{sum(review_drop.values())}" + ("（本次用 --include-unreviewed 全部保留）" if a.include_unreviewed else ""),
         "", "| 子题 | 题数 | 因答案超过一半去掉 | 因复核去掉 | 答案分布 |", "|---|---:|---:|---:|---|"]
    for k in keys:
        cnt = collections.Counter(answer_class(r["question"], r["private_target"]["answer"]) for r in by.get(k, []))
        L.append(f"| {k} | {len(by.get(k, []))} | {bal_drop.get(k, 0)} | {review_drop.get(k, 0)} | {dict(cnt)} |")
    (out / "export_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"exported {len(keep)} questions -> {out}  (balance dropped {sum(bal_drop.values())}, "
          f"review dropped {sum(review_drop.values())})")


if __name__ == "__main__":
    main()
