"""Write a 解析 (answer explanation) for every question of a bench folder, from the program facts only.

  python3 -B scripts/make_explanations.py --bench <bench_dir> [--release <release_dir>]

No model is called. Each explanation lists the recorded events the answer is computed from (MIDI key
presses/releases, the gold/score finger at a press, skeleton palm positions), restricted to the asked time
range / hand / key, and ends with the program answer:

  【依据】约1.23秒，右手按下E4（按下时使用中指）。…【结论】所以答案是{B}（E4）。

Writes <bench_dir>/explanations.jsonl ({qid, explanation}) and explanations.html; with --release it also
adds an "explanation" field to <release_dir>/questions.jsonl (matched by qid; the original is kept as
questions.no_explanation.jsonl).
"""
import argparse
import html
import json
from pathlib import Path

MAX_LINES = 14
PALM_SUBTYPES = ("张开跨度", "手掌净位移", "手掌横移方向", "第一次大幅横移")
LETTERS = "ABCDEFG"


def sentence(a, with_finger):
    finger = f"（按下时使用{a['finger']}）" if with_finger and a.get("finger") else ""
    if a["predicate"] in ("按下", "松开"):
        return f"约{a['t']:.2f}秒，{a['subject']}{a['predicate']}{a['object']}{finger}。"
    x = f"（横坐标 {a['x']:.2f} 个白键宽）" if "x" in a else ""
    return f"约{a['t']:.2f}秒，{a['subject']}{a['predicate']}{a['object']}{x}。"


def answer_text(q, ans):
    if q.get("options"):
        opts = q["options"] if isinstance(q["options"], list) else q["options"]["choices"]
        i = LETTERS.index(ans) if isinstance(ans, str) and ans in LETTERS else None
        return f"{{{ans}}}（{opts[i]}）" if i is not None and i < len(opts) else f"{{{ans}}}"
    if q["answer_type"] == "time_span" and isinstance(ans, list):
        return f"{{{ans[0]:.2f}, {ans[1]:.2f}}}（第 {ans[0]:.2f} 秒到第 {ans[1]:.2f} 秒）"
    if q["answer_type"] == "time_point":
        return f"{{{float(ans):.2f}}}（第 {float(ans):.2f} 秒）"
    if isinstance(ans, list):
        return "{" + ", ".join(map(str, ans)) + "}"
    return "{" + str(ans) + "}"


def pick(q, atoms):
    slots, rng, sub = q.get("slots") or {}, q.get("range") or {}, q["subtype"]
    lo, hi = rng.get("start", -1e9), rng.get("end", 1e9)
    inside = [a for a in atoms if lo - 1e-6 <= a["t"] < hi + 1e-6]
    hand = slots.get("hand")
    if hand:
        inside = [a for a in inside if a.get("subject") in (hand, "未知手") or a.get("predicate") not in ("按下", "松开", "手掌中心位于")]
    if sub in PALM_SUBTYPES:
        geo = [a for a in inside if a["predicate"] not in ("按下", "松开", "手掌中心位于")]   # the measured positions
        palm = [a for a in inside if a["predicate"] == "手掌中心位于"]
        ev = geo or palm[:: max(1, len(palm) // 6)]
    else:
        ev = [a for a in inside if a["predicate"] == "按下"]
        if sub in ("先松开哪个键", "按键保持时长", "持键重叠", "长按某个键", "某个键之后是否处于按下状态"):
            ev = [a for a in inside if a["predicate"] in ("按下", "松开")]
    key = slots.get("key")
    if key:
        ev = [a for a in ev if a.get("object") == key] or ev
    keys = slots.get("keys")
    if keys:
        ev = [a for a in ev if a.get("object") in keys] or ev
    return sorted(ev, key=lambda a: a["t"]), (lo, hi)


def explain(row):
    q, atoms = row["question"], row.get("atoms", [])
    ans = row["private_target"]["answer"]
    ev, (lo, hi) = pick(q, atoms)
    scope = []
    if lo > -1e8:
        scope.append(f"所问区间为第 {lo:.2f} 秒到第 {hi:.2f} 秒（不含）")
    if (q.get("slots") or {}).get("hand"):
        scope.append(f"只看{q['slots']['hand']}")
    head = "【范围】" + "，".join(scope) + "。" if scope else ""
    if ev:
        fing = "手指" in q["subtype"]
        body = "".join(sentence(a, fing) for a in ev[:MAX_LINES])
        presses = sum(a["predicate"] == "按下" for a in ev)
        if len(ev) > MAX_LINES:
            body += f"……（共 {len(ev)} 条记录，其中按下 {presses} 次）"
        elif q["task"] == "T3" and "次数" in q["subtype"]:
            body += f"以上共按下 {presses} 次（同时按下的不同键分别计数）。"
    else:
        body = "所问范围内没有记录到相关按键。"
    tail = ""
    if q["task"] == "T5":
        pr = q.get("predict") or {}
        tail = (f"【说明】这是预测题：上面只列出“现在”（第 {pr.get('t0', 0):.2f} 秒）及之前的记录；"
                f"答案由现在之后 {pr.get('horizon', 0)} 秒内的实际演奏记录（MIDI 与骨架）确定。")
    if "手指" in q["subtype"] and any(a.get("finger") for a in ev):
        src = "乐谱印刷指法（哈农）" if str(q.get("video_id", "")).startswith("shcm") else "人工标注"
        tail += f"【来源】按键来自 MIDI，指法来自{src}。"
    return head + "【依据】" + body + tail + "【结论】所以答案是" + answer_text(q, ans) + "。"


def release_conclusion(e, d):
    """The release turns time questions into 4-option choices: end the explanation with the release answer."""
    if d.get("answer_type") != "choice" or not d.get("options"):
        return e
    body, sep, concl = e.rpartition("【结论】")
    i = LETTERS.index(d["answer"])
    was = concl.removeprefix("所以答案是").rstrip("。")
    extra = f"；程序值为 {was}" if "秒" in was else ""
    return f"{body}【结论】所以答案是{{{d['answer']}}}（{d['options'][i]}{extra}）。"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--release", default="")
    a = ap.parse_args()
    bench = Path(a.bench)
    rows = [json.loads(x) for x in (bench / "questions.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    out, errors = {}, []
    for r in rows:
        try:
            out[r["qid"]] = explain(r)
        except Exception as e:                      # keep going; report at the end
            errors.append((r.get("qid"), f"{type(e).__name__}: {e}"))
    with open(bench / "explanations.jsonl", "w", encoding="utf-8") as f:
        for qid, e in out.items():
            f.write(json.dumps(dict(qid=qid, explanation=e), ensure_ascii=False) + "\n")
    page = ["<meta charset=utf-8><title>解析</title><style>body{font:14px sans-serif;max-width:900px;margin:auto}"
            "article{border-bottom:1px solid #ccc;padding:8px 0}</style>"]
    for r in rows:
        if r["qid"] in out:
            q = r["question"]
            page.append(f"<article><b>{html.escape(q['task'] + ' ' + q['subtype'])}</b><p>{html.escape(q['question'])}</p>"
                        f"<p>{html.escape(out[r['qid']])}</p><small>{r['qid']}</small></article>")
    (bench / "explanations.html").write_text("\n".join(page), encoding="utf-8")
    print(f"explanations: {len(out)}/{len(rows)} -> {bench / 'explanations.jsonl'}, {bench / 'explanations.html'}")
    for qid, e in errors[:10]:
        print("  FAILED", qid, e)
    if a.release:
        rel = Path(a.release) / "questions.jsonl"
        orig = rel.parent / "questions.no_explanation.jsonl"
        if not orig.exists():                       # first run: keep the original; reruns start from it
            orig.write_text(rel.read_text(encoding="utf-8"), encoding="utf-8")
        lines = orig.read_text(encoding="utf-8").splitlines()
        n, new = 0, []
        for x in lines:
            if not x.strip():
                continue
            d = json.loads(x)
            e = out.get(d.get("qid")) or out.get(d.get("source_qid", ""))
            if e:
                d["explanation"], n = release_conclusion(e, d), n + 1
            new.append(json.dumps(d, ensure_ascii=False))
        rel.write_text("\n".join(new) + "\n", encoding="utf-8")
        print(f"release: explanation added to {n}/{len(new)} questions in {rel}")


if __name__ == "__main__":
    main()
