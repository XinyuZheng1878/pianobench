"""Step 8b: score model outputs on the benchmark.

  python -B scripts/evaluate.py --questions work/release/questions.jsonl \
      --runs work/runs/qwen_F.jsonl work/runs/joyai_F.jsonl work/runs/qwen_blind.jsonl --out work/report

Each run file has one line per question: {"qid": ..., "output": "<raw model output>"} (extra keys are fine;
"model"/"setting" are used as the run label when present, else the file name).
Writes report.md, results.csv (one row per run x question) and summary.json.

Answer extraction: regex \\{([^{}]*)\\} on the output with <think>...</think> removed; the last brace
counts. Missing brace or unparsable content -> score 0 and a format error. Built-in baselines:
random, majority (per subtype), and copy-current for T5.
"""
import argparse
import collections
import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

BRACE = re.compile(r"\{([^{}]*)\}")
NOTE_BASE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
ACC = {"#": 1, "♯": 1, "＃": 1, "b": -1, "♭": -1, "": 0}
CN_NUM = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
STATES = ["落下", "抬起", "按键保持", "空中保持", "移动"]
INTERVALS = ["纯一度", "小二度", "大二度", "小三度", "大三度", "纯四度", "增四度", "纯五度",
             "小六度", "大六度", "小七度", "大七度", "纯八度"]
TASKS = ["T1", "T2", "T3", "T4", "T5"]


# ------------------------------------------------------------------ parsing

def clean_output(s):
    s = s or ""
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S)
    s = s.replace("</response>", "").replace("<response>", "")
    return s


def braces(s):
    return BRACE.findall(clean_output(s))


def norm(s):
    s = s.strip()
    tbl = str.maketrans("０１２３４５６７８９ＡＢＣＤＥａｂｃｄｅ，．", "0123456789ABCDEabcde,.")
    return s.translate(tbl)


def parse_key(s):
    s = norm(s).replace(" ", "")
    m = re.fullmatch(r"([A-Ga-g])([#♯＃b♭]?)(-?\d)", s) or None
    if m:
        l, acc, o = m.groups()
    else:
        m = re.fullmatch(r"([A-Ga-g])(-?\d)([#♯＃b♭]?)", s)
        if not m:
            return None
        l, o, acc = m.groups()
    return (int(o) + 1) * 12 + NOTE_BASE[l.upper()] + ACC[acc]


def parse_int(s):
    s = norm(s).replace(" ", "")
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    if s and all(c in CN_NUM for c in s):
        if s == "十":
            return 10
        if s.startswith("十"):
            return 10 + CN_NUM[s[1]]
        if len(s) == 2 and s[1] == "十":
            return CN_NUM[s[0]] * 10
        if len(s) == 3 and s[1] == "十":
            return CN_NUM[s[0]] * 10 + CN_NUM[s[2]]
        if len(s) == 1:
            return CN_NUM[s]
    return None


def split_list(s):
    return [x for x in re.split(r"[,，、;；→\s]+|->", norm(s)) if x]


def parse_floats(s):
    v = re.findall(r"-?\d+(?:\.\d+)?", norm(s))
    return [float(x) for x in v]


def parse_chord(s):
    s = norm(s).replace(" ", "")
    m = re.fullmatch(r"([A-Ga-g])([#♯＃b♭]?)(.+和弦)", s)
    if not m:
        return None
    return (NOTE_BASE[m.group(1).upper()] + ACC[m.group(2)]) % 12, m.group(3)


def edit_distance(a, b):
    dp = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(b) + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[-1]


def tiou(p, g):
    inter = max(0.0, min(p[1], g[1]) - max(p[0], g[0]))
    union = max(p[1], g[1]) - min(p[0], g[0])
    return inter / union if union > 0 else 0.0


def score(q, output):
    """Returns dict(score, fmt_err, pred, extra...)."""
    bs = braces(output)
    at, ans = q["answer_type"], q["answer"]
    r = dict(score=0.0, fmt_err=False, pred=None)
    if not bs:
        r["fmt_err"] = True
        return r
    raw = bs[-1]
    r["pred"] = raw
    if at == "choice":
        s = norm(raw).replace(" ", "").rstrip(".．。")
        letter = None
        if re.fullmatch(r"[A-Ea-e]", s):
            letter = s.upper()
        elif q.get("options"):
            for i, o in enumerate(q["options"]):
                if s == o.replace(" ", "") or re.sub(r"^[A-Ea-e][.．、:：]", "", s) == o.replace(" ", ""):
                    letter = "ABCDE"[i]
        if letter is None:
            r["fmt_err"] = True
            return r
        r.update(score=float(letter == ans), letter=letter)
    elif at in ("int", "frame"):
        v = parse_int(raw)
        if v is None:
            r["fmt_err"] = True
            return r
        if at == "frame":
            r.update(score=float(abs(v - ans) <= 2), abs_err=abs(v - ans))
        else:
            r.update(score=float(v == ans), abs_err=abs(v - ans), signed_err=v - ans)
    elif at == "key":
        v = parse_key(raw)
        g = parse_key(ans)
        if v is None:
            r["fmt_err"] = True
            return r
        r.update(score=float(v == g), within2=abs(v - g) <= 2, octave_err=(v - g) % 12 == 0 and v != g)
    elif at == "key_sequence":
        items = split_list(raw)
        v = [parse_key(x) for x in items]
        if not v or any(x is None for x in v):
            r["fmt_err"] = True
            return r
        g = [parse_key(x) for x in ans]
        r.update(score=max(0.0, 1 - edit_distance(v, g) / len(g)), exact=v == g)
    elif at == "key_set":
        v = {parse_key(x) for x in split_list(raw)}
        if None in v or not v:
            r["fmt_err"] = True
            return r
        g = {parse_key(x) for x in ans}
        tp = len(v & g)
        f1 = 2 * tp / (len(v) + len(g))
        r.update(score=f1, exact=v == g)
    elif at == "state_sequence":
        items = split_list(raw)
        if not items or any(x not in STATES for x in items):
            r["fmt_err"] = True
            return r
        items = [x for i, x in enumerate(items) if i == 0 or x != items[i - 1]]
        r.update(score=max(0.0, 1 - edit_distance(items, ans) / len(ans)), exact=items == ans)
    elif at == "chord":
        v, g = parse_chord(raw), parse_chord(ans)
        if v is None:
            r["fmt_err"] = True
            return r
        r.update(score=float(v == g), quality_only=v[1] == g[1])
    elif at == "interval_name":
        s = norm(raw).replace(" ", "").replace("三全音", "增四度").replace("减五度", "增四度")
        if s not in INTERVALS:
            r["fmt_err"] = True
            return r
        r.update(score=float(s == ans))
    elif at == "time_point":
        v = parse_floats(raw)
        if len(v) != 1:
            r["fmt_err"] = True
            return r
        r.update(score=float(abs(v[0] - ans) <= 0.25), abs_err=abs(v[0] - ans))
    elif at == "time_span":
        v = parse_floats(raw)
        if len(v) != 2 or v[1] < v[0]:
            r["fmt_err"] = True
            return r
        t = tiou(v, ans)
        r.update(score=float(t >= 0.5), tiou=t)
    else:
        raise ValueError(at)
    return r


# ------------------------------------------------------------------ baselines

def loo(counter, q, fact_ans):
    c = collections.Counter(counter)
    k = ((q["task"], q["subtype"], q["answer_type"]), q["fact_id"])
    if k in fact_ans:
        c[fact_ans[k]] -= 1
        if c[fact_ans[k]] <= 0:
            del c[fact_ans[k]]
    return c


def baseline_outputs(qs, kind, rng):
    out = {}
    if kind == "majority":   # most frequent answer *content* per subtype, leave-one-fact-out
        maj = collections.defaultdict(collections.Counter)
        fact_ans = {}
        for q in qs:
            a = q["options"]["ABCDE".index(q["answer"])] if q["answer_type"] == "choice" else q["answer"]
            key = (q["task"], q["subtype"], q["answer_type"])
            k = (key, q["fact_id"])
            if k not in fact_ans:      # count each fact once
                fact_ans[k] = json.dumps(a, ensure_ascii=False)
                maj[key][fact_ans[k]] += 1
    for q in qs:
        if kind == "random":
            if q["answer_type"] == "choice":
                out[q["qid"]] = "{" + "ABCDE"[int(rng.integers(len(q["options"])))] + "}"
            elif q["answer_type"] in ("time_point", "frame"):
                L = q["clip"]["length_s"]
                v = rng.uniform(0, L)
                out[q["qid"]] = "{" + (f"{v:.2f}" if q["answer_type"] == "time_point" else str(int(v * 8))) + "}"
            elif q["answer_type"] == "time_span":
                L = q["clip"]["length_s"]
                a, b = sorted(rng.uniform(0, L, 2))
                out[q["qid"]] = "{%.2f, %.2f}" % (a, b)
            else:
                out[q["qid"]] = "{}"
        elif kind == "majority" and q["answer_type"] in ("time_point", "frame", "time_span"):
            L = q["clip"]["length_s"]   # no "majority" for times: answer the clip middle / whole clip
            out[q["qid"]] = {"time_point": "{%.2f}" % (L / 2), "frame": "{%d}" % int(L * 4),
                             "time_span": "{0, %.2f}" % L}[q["answer_type"]]
        elif kind == "majority" and q["answer_type"] == "choice":
            cnt = loo(maj[q["task"], q["subtype"], q["answer_type"]], q, fact_ans)
            for txt, _ in cnt.most_common():
                t = json.loads(txt)
                if t in q["options"]:
                    out[q["qid"]] = "{" + "ABCDE"[q["options"].index(t)] + "}"
                    break
            else:
                out[q["qid"]] = "{A}"
        elif kind == "majority":
            cnt = loo(maj[q["task"], q["subtype"], q["answer_type"]], q, fact_ans)
            if not cnt:
                out[q["qid"]] = "{}"
                continue
            a = json.loads(cnt.most_common(1)[0][0])
            if isinstance(a, list):
                a = ", ".join(str(x) for x in a)
            out[q["qid"]] = "{" + str(a) + "}"
        elif kind == "copy":
            c = (q.get("baselines") or {}).get("copy")
            if c:
                out[q["qid"]] = "{" + c + "}"
    return out


# ------------------------------------------------------------------ aggregation

def boot_ci(x, rng, n=1000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan)
    m = rng.integers(0, len(x), (n, len(x)))
    v = x[m].mean(1)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def overall(rows):
    ts = [np.mean([r["score"] for r in rows if r["task"] == t]) for t in TASKS if any(r["task"] == t for r in rows)]
    return float(np.mean(ts)) if ts else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--baselines", nargs="*", default=["random", "majority", "copy"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    qs = [json.loads(l) for l in open(a.questions, encoding="utf-8")]
    qmap = {q["qid"]: q for q in qs}

    runs = collections.OrderedDict()
    for p in a.runs:
        outs, label = {}, None
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            if r.get("error") or r.get("output") is None:
                continue
            outs[r["qid"]] = r["output"]
            label = label or (f"{r.get('model', '')}·{r.get('setting', '')}".strip("·") or None)
        runs[label or Path(p).stem] = outs
    for b in a.baselines:
        runs[f"baseline:{b}"] = baseline_outputs(qs, b, np.random.default_rng(a.seed))

    rows = []
    for name, outs in runs.items():
        for q in qs:
            if name == "baseline:copy" and q["qid"] not in outs:
                continue
            if q["qid"] not in outs and not name.startswith("baseline:"):
                continue   # unanswered (e.g. setting-F-only question in a V run) -> not scored
            r = score(q, outs.get(q["qid"], ""))
            rows.append(dict(run=name, qid=q["qid"], task=q["task"], subtype=q["subtype"], level=q["level"],
                             qa_type=q["qa_type"], range_type=q["tags"]["range_type"], answer_type=q["answer_type"],
                             answer=json.dumps(q["answer"], ensure_ascii=False), pred=r.get("pred"),
                             score=r["score"], fmt_err=r["fmt_err"],
                             extra=json.dumps({k: v for k, v in r.items() if k not in ("score", "fmt_err", "pred")}, ensure_ascii=False, default=float)))
    with open(os.path.join(a.out, "results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    by_run = collections.defaultdict(list)
    for r in rows:
        by_run[r["run"]].append(r)
    summary, L = {}, []
    L.append("# 钢琴手部视频问答 · 评测报告\n")
    L.append(f"题目 {len(qs)} 道（对照题 {sum(q['task'] == 'C1' for q in qs)} 道，不计入总分）。分数为该任务所有题的平均分（0–1），括号内为 95% bootstrap 置信区间。"
             "baseline:majority 按子类型取出现最多的答案内容（留一法：不看本题所属事实）；baseline:copy 只用于 T5，答“照抄当前”。\n")
    L.append("## 1. 主榜（任务分与总分）\n")
    L.append("| 运行 | 题数 | " + " | ".join(TASKS) + " | 总分 | 格式错误率 |")
    L.append("|---" * (len(TASKS) + 4) + "|")
    for name, rs in by_run.items():
        main_rs = [r for r in rs if r["task"] in TASKS]
        cells = []
        for t in TASKS:
            x = [r["score"] for r in main_rs if r["task"] == t]
            if x:
                lo, hi = boot_ci(x, rng)
                cells.append(f"{np.mean(x):.2f} ({lo:.2f}–{hi:.2f})")
            else:
                cells.append("–")
        ov = overall(main_rs)
        fe = np.mean([r["fmt_err"] for r in main_rs]) if main_rs else float("nan")
        L.append(f"| {name} | {len(main_rs)} | " + " | ".join(cells) + f" | **{ov:.2f}** | {fe:.0%} |")
        summary[name] = dict(n=len(main_rs), overall=ov, fmt_err=fe,
                             tasks={t: float(np.mean([r['score'] for r in main_rs if r['task'] == t]))
                                    for t in TASKS if any(r['task'] == t for r in main_rs)})
    L.append("\n## 2. 按难度等级（跨任务）\n")
    lv = ["L1", "L2", "L3", "L4", "L5"]
    L.append("| 运行 | " + " | ".join(lv) + " |")
    L.append("|---" * (len(lv) + 1) + "|")
    for name, rs in by_run.items():
        L.append(f"| {name} | " + " | ".join(
            f"{np.mean([r['score'] for r in rs if r['level'] == l]):.2f}" if any(r['level'] == l for r in rs) else "–" for l in lv) + " |")
    L.append("\n## 3. 按子类型\n")
    subs = sorted({(r["task"], r["subtype"]) for r in rows if r["task"] in TASKS})
    names = list(by_run)
    L.append("| 任务 | 子类型 | 题数 | " + " | ".join(names) + " |")
    L.append("|---" * (len(names) + 3) + "|")
    for t, s in subs:
        n = sum(1 for q in qs if q["task"] == t and q["subtype"] == s)
        L.append(f"| {t} | {s} | {n} | " + " | ".join(
            f"{np.mean([r['score'] for r in by_run[nm] if r['task'] == t and r['subtype'] == s]):.2f}"
            if any(r['task'] == t and r['subtype'] == s for r in by_run[nm]) else "–" for nm in names) + " |")
    L.append("\n## 4. 诊断\n")
    L.append("**区间指定方式**（T1–T3）：\n")
    rts = ["whole", "timestamp", "frames", "boundary_frames"]
    L.append("| 运行 | " + " | ".join(rts) + " |")
    L.append("|---" * (len(rts) + 1) + "|")
    for name, rs in by_run.items():
        L.append(f"| {name} | " + " | ".join(
            f"{np.mean([r['score'] for r in rs if r['range_type'] == x and r['task'] in ('T1', 'T2', 'T3')]):.2f}"
            if any(r['range_type'] == x and r['task'] in ('T1', 'T2', 'T3') for r in rs) else "–" for x in rts) + " |")
    L.append("\n**单选与填空**：\n")
    L.append("| 运行 | 单选 | 填空 | 描述 |")
    L.append("|---|---|---|---|")
    for name, rs in by_run.items():
        L.append(f"| {name} | " + " | ".join(
            f"{np.mean([r['score'] for r in rs if r['qa_type'] == x and r['task'] in TASKS]):.2f}"
            if any(r['qa_type'] == x and r['task'] in TASKS for r in rs) else "–" for x in ("单选", "填空", "描述")) + " |")
    L.append("\n**单选的选项偏好**（各字母被选比例；真值分布见最后一行）：\n")
    L.append("| 运行 | A | B | C | D | E |")
    L.append("|---|---|---|---|---|---|")
    for name, rs in list(by_run.items()) + [("真值", None)]:
        if rs is None:
            c = collections.Counter(q["answer"] for q in qs if q["answer_type"] == "choice" and q["task"] in TASKS)
        else:
            c = collections.Counter(json.loads(r["extra"]).get("letter") for r in rs if r["answer_type"] == "choice" and r["task"] in TASKS)
            c.pop(None, None)
        tot = sum(c.values()) or 1
        L.append(f"| {name} | " + " | ".join(f"{c[x] / tot:.0%}" for x in "ABCDE") + " |")
    L.append("\n**对照题**（踏板，只能答“无法从画面判断”）：\n")
    for name, rs in by_run.items():
        cr = [r for r in rs if r["task"] == "C1"]
        if cr:
            letters = collections.Counter(json.loads(r["extra"]).get("letter") for r in cr)
            fab = (letters["A"] + letters["B"]) / len(cr)
            L.append(f"- {name}：编答案比例 {fab:.0%}（{len(cr)} 题）")
    L.append("\n**T2 手指题里错成相邻手指的比例**：\n")
    for name, rs in by_run.items():
        wrong = [(r, qmap[r["qid"]]) for r in rs if r["subtype"] == "哪根手指" and r["score"] == 0 and not r["fmt_err"]]
        if wrong:
            adj = 0
            for r, q in wrong:
                v = json.loads(r["extra"]).get("letter") if q["answer_type"] == "choice" else r["pred"]
                g = q["answer"]
                try:
                    vi = "ABCDE".index(v) if q["answer_type"] == "choice" else int(v) - 1
                    gi = "ABCDE".index(g) if q["answer_type"] == "choice" else int(g) - 1
                    adj += abs(vi - gi) == 1
                except (ValueError, TypeError):
                    pass
            L.append(f"- {name}：{adj}/{len(wrong)}")
    L.append("\n**T5 下一个键：选中各干扰项的比例**：\n")
    for name, rs in by_run.items():
        c = collections.Counter()
        n = 0
        for r in rs:
            q = qmap[r["qid"]]
            if q["subtype"] != "下一个键" or r["fmt_err"]:
                continue
            n += 1
            roles = q["extra"].get("distractor_roles", {})
            c[roles.get(json.loads(r["extra"]).get("letter"), "正确" if r["score"] else "其他")] += 1
        if n:
            L.append(f"- {name}：" + "，".join(f"{k} {v / n:.0%}" for k, v in c.most_common()))
    L.append("\n**看画面 vs 盲测**：模型分数接近盲测或多数类基线，说明它主要靠题干和先验作答，没有真正用上画面。\n")
    Path(a.out, "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n".join(L[:12 + len(by_run)]))
    print("report ->", os.path.join(a.out, "report.md"))


if __name__ == "__main__":
    main()
