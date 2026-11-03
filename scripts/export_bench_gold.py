"""Benchmark step 4: turn a gold bench folder into the release that run_models.py / evaluate.py read.

  python3 -B scripts/export_bench_gold.py --bench ../gold_work/bench_v2 --out ../gold_work/release_v2 \
      [--approvals ../gold_work/review/approvals.jsonl] [--include-unreviewed]

What it does
  * answer-balance rule: within a subtype no single answer may exceed half of the questions; the surplus of
    the over-represented answer is dropped, newest batch first (the bench folder itself is not changed);
  * questions that need human review (finger from gold labels) are exported only when an approval with the
    same qid and fact_hash says "approve" (or with --include-unreviewed, for a dry evaluation run);
  * time questions (time point / time span) become 4-option choices whose options are >= 1 s apart
    (--time-as-fill keeps the old free-form seconds); every question starts with its frame count and frame rate;
  * writes questions.jsonl in the evaluation schema, system_prompt.txt, media/ (linked to the bench),
    index.html (browse every question with 5 of its frames) and export_report.md (what was kept, dropped and why).
"""
import argparse
import collections
import json
import random
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
7. 最终答案必须写在花括号里，例如 {B}、{3}、{G#3}、{E4, F#4, G4}；花括号只用于最终答案。"""


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


TIME_TAILS = ("把两个时间（秒）写在花括号里，用逗号分隔。", "把时间（秒）写在花括号里。")
STEP_S = 1.0          # every option of a time question is at least this far (in seconds) from the others


def time_choice(question, at, ans, length, seed):
    """Turn a time-point / time-span question into 4 options at least STEP_S apart (in time, or at one end
    of a span), so that the model only has to locate the moment roughly, not to the frame."""
    rnd = random.Random(seed)
    if at == "time_point":
        t = float(ans)
        cand = [t + k * STEP_S for k in (-3, -2, -1, 1, 2, 3) if 0 <= t + k * STEP_S <= length]
        true = t
        show = lambda x: f"第 {x:.1f} 秒"
    else:
        a, b = float(ans[0]), float(ans[1])
        cand = []
        for da in (-2, -1, 0, 1, 2):
            for db in (-2, -1, 0, 1, 2):
                x, y = a + da * STEP_S, b + db * STEP_S
                if (da, db) != (0, 0) and 0 <= x < y <= length:
                    cand.append((x, y))
        true = (a, b)
        show = lambda x: f"第 {x[0]:.1f} 秒到第 {x[1]:.1f} 秒"
    # the true option's rank among the 4 (earliest ... latest) is drawn uniformly, so "pick the middle one" is no
    # better than chance; ranks that the clip cannot supply enough earlier / later options for are skipped
    lower, higher = [c for c in cand if c < true], [c for c in cand if c > true]
    ranks = [k for k in range(4) if len(lower) >= k and len(higher) >= 3 - k]
    if not ranks:
        return None
    k = rnd.choice(ranks)
    opts = [true] + rnd.sample(lower, k) + rnd.sample(higher, 3 - k)
    labels = [show(o) for o in opts]
    if len(set(labels)) != 4:
        return None
    order = sorted(range(4), key=lambda i: opts[i])     # options listed from earliest to latest
    choices = [labels[i] for i in order]
    letter = LETTERS[order.index(0)]
    stem = question
    for tail in TIME_TAILS:
        stem = stem.replace(tail, "")
    stem = stem.rstrip() + "选最接近的一项。选项：" + "；".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices)) \
        + "。把选项字母写在花括号里。"
    return stem, choices, letter


def to_eval(r, time_as_choice=True):
    q, t = r["question"], r["private_target"]
    at = ANSWER_TYPE.get(q["answer_type"], q["answer_type"])
    a = t["answer"]
    if at in ("time_point", "time_span"):
        a = round(float(a), 3) if at == "time_point" else [round(float(x), 3) for x in a]
    frames = r["media"]["frames"]
    t0 = frames[0]["t"]
    length = round(frames[-1]["t"] - t0 + 1.0 / q["clip"]["fps_out"], 3)
    question, options = q["question"], q.get("options")
    if time_as_choice and at in ("time_point", "time_span"):
        tc = time_choice(question, at, a, len(frames) / q["clip"]["fps_out"], r["qid"])
        if tc is None:
            return None
        question, options, a = tc
        at = "choice"
    fps = q["clip"]["fps_out"]
    question = (f"（本题共 {len(frames)} 帧，覆盖 {len(frames) / fps:g} 秒，每 {1.0 / fps:.3g} 秒一帧，即 {fps:g} 帧/秒。）"
                + question)
    return dict(
        qid=r["qid"], fact_id=r["qid"], task=q["task"], subtype=q["subtype"], level=q["level"],
        video_id=q["video_id"], question=question, answer_type=at, answer=a,
        options=options, frame_fps=q["clip"]["fps_out"],
        media=dict(frames=[f["path"] for f in frames], frame_times=[round(f["t"] - t0, 3) for f in frames]),
        clip=dict(start_frame=frames[0]["native_frame"], end_frame=frames[-1]["native_frame"] + 1, length_s=length),
        range=q.get("range"), predict=q.get("predict"),
        qa_type="单选" if at == "choice" else "填空",
        tags=dict(range_type=(q.get("range") or {}).get("type", "whole"), semantic_id=q["semantic_id"],
                  needs_human=r["needs_human"], batch=r.get("batch", "v1")),
        extra={}, baselines={})


def browse_page(rows, title):
    """A self-contained index.html next to questions.jsonl: every question with its answer and 5 of its frames."""
    import html
    cards = []
    for e in rows:
        fr, ts = e["media"]["frames"], e["media"]["frame_times"]
        pick = sorted({round(i * (len(fr) - 1) / 4) for i in range(5)})
        imgs = "".join(f'<figure><img loading="lazy" src="{html.escape(fr[i])}"><figcaption>第 {i} 帧 · {ts[i]:.3f} 秒'
                       f'</figcaption></figure>' for i in pick)
        ans = e["answer"]
        if e.get("options"):
            ans = f'{ans}. {e["options"][LETTERS.index(ans)]}'
        elif isinstance(ans, list):
            ans = ", ".join(map(str, ans))
        cards.append(f'<article><p class="m">{html.escape(e["task"])} · {html.escape(e["subtype"])} · '
                     f'{len(fr)} 帧，{e["frame_fps"]:g} 帧/秒 · {html.escape(e["qid"])}</p>'
                     f'<p>{html.escape(e["question"])}</p><p class="a">答案：<b>{html.escape(str(ans))}</b></p>'
                     f'<div class="s">{imgs}</div></article>')
    return f"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>
body{{font:15px/1.6 system-ui,sans-serif;margin:0 auto;max-width:1100px;padding:16px;background:#fafafa;color:#222}}
article{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px 16px;margin:12px 0}}
.m{{color:#777;font-size:13px;margin:0}}.a{{color:#155}}.s{{display:flex;gap:6px;overflow-x:auto}}
figure{{margin:0;flex:0 0 200px}}img{{width:200px;border-radius:4px}}figcaption{{font-size:12px;color:#777}}
</style><h1>{html.escape(title)}</h1><p>共 {len(rows)} 题；每题展示 5 帧（模型拿到的是全部帧）。题目文件：questions.jsonl，帧：media/。</p>
{"".join(cards)}"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--out", required=True, help="new folder")
    ap.add_argument("--approvals", nargs="*", default=[],
                    help="one or more jsonl files of {qid, fact_hash, decision, reviewer} (from the review page)")
    ap.add_argument("--time-as-fill", action="store_true",
                    help="keep time questions as free-form seconds (default: 4-option choice, options >= 1 s apart)")
    ap.add_argument("--include-unreviewed", action="store_true",
                    help="also export finger questions nobody has approved yet (dry evaluation only)")
    a = ap.parse_args()
    bench, out = Path(a.bench).resolve(), Path(a.out).resolve()
    if out.exists():
        raise SystemExit(f"{out} already exists; choose a new --out")
    rows = [json.loads(l) for l in open(bench / "questions.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r.get("status") != "delivery_failed" and "media" in r]

    approved, rejected = set(), set()
    for path in a.approvals:
        for l in open(path, encoding="utf-8"):
            if not l.strip():
                continue
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
    n_time_drop = collections.Counter()
    evals = []
    with open(out / "questions.jsonl", "w", encoding="utf-8") as fo:
        for r in keep:
            e = to_eval(r, not a.time_as_fill)
            if e is None:
                n_time_drop[r["question"]["task"] + "/" + r["question"]["subtype"]] += 1
                continue
            evals.append(e)
            fo.write(json.dumps(e, ensure_ascii=False) + "\n")
    (out / "index.html").write_text(browse_page(evals, f"测评集 {out.name}"), encoding="utf-8")
    keep = [r for r in keep if to_eval(r, not a.time_as_fill) is not None]
    (out / "system_prompt.txt").write_text(SYSTEM_PROMPT + "\n", encoding="utf-8")

    by = collections.defaultdict(list)
    for r in keep:
        by[r["question"]["task"] + "/" + r["question"]["subtype"]].append(r)
    keys = sorted(set(by) | set(bal_drop) | set(review_drop))
    L = ["# 测评集导出报告", "", f"来源：{bench}", f"题目：{len(keep)}（导出前 {len(rows)}）",
         f"时间题：{'保持填写秒数' if a.time_as_fill else '改为 4 选 1（选项之间至少相差 1 秒）'}；无法构造选项而去掉 {sum(n_time_drop.values())} 题",
         f"未经人工批准而排除的指法题：{sum(review_drop.values())}" + ("（本次用 --include-unreviewed 全部保留）" if a.include_unreviewed else ""),
         "", "| 子题 | 题数 | 因答案超过一半去掉 | 因复核去掉 | 答案分布 |", "|---|---:|---:|---:|---|"]
    for k in keys:
        cnt = collections.Counter(answer_class(r["question"], r["private_target"]["answer"]) for r in by.get(k, []))
        L.append(f"| {k} | {len(by.get(k, []))} | {bal_drop.get(k, 0)} | {review_drop.get(k, 0)} | {dict(cnt)} |")
    (out / "export_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"浏览页：{out / 'index.html'}")
    print(f"exported {len(keep)} questions -> {out}  (balance dropped {sum(bal_drop.values())}, "
          f"review dropped {sum(review_drop.values())})")


if __name__ == "__main__":
    main()
