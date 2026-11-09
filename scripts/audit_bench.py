"""Quality audit of a released benchmark (the export_video_qa.py output; a frames release works too).

  python3 -B scripts/audit_bench.py --release <video_release_dir> [--work ../shcm_work/20261024] \
      [--per-subtype 3] [--extra-finger 20] [--out <release_dir>/audit]

Writes <out>/:
  audit_report.md   1. automatic checks on every question (counts + the failing questions)
                    2. shortcut baselines per subtype (random, always-the-commonest-answer, copy-current)
                    3. cross-source agreement per take (score/human finger vs. the skeleton-based guess,
                       hand labels, fingertip-to-key distance at onsets) -- low agreement = look at that take
  failures.jsonl    one line per failed check {qid, check, detail}
  audit.html        human review of a stratified sample: video, question, options, answer, explanation and
                    four buttons (对 / 错 / 有歧义 / 看不清) + a note; marks are kept in the browser and
                    "导出结果" downloads audit_results.csv
  sample.jsonl      the sampled questions

Automatic checks:
  answer_roundtrip   the evaluator scores "{<gold answer>}" as correct (format and parser agree)
  option_line        the option line inside the question equals the options list, options are distinct
  explanation        the explanation's conclusion names the gold answer
  answer_in_stem     the correct option / fill answer is written in the question stem (a giveaway)
  media              the video exists and has end_frame - start_frame frames (or every frame image exists)
  time_in_clip       every time in the answer and options lies inside the clip
  duplicate          the same clip and the same question text appear twice
"""
import argparse
import collections
import html
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import evaluate as EV  # noqa: E402

LETTERS = "ABCDEFG"


def fmt_answer(q):
    a = q["answer"]
    if isinstance(a, list):
        return "{" + ", ".join(f"{x:.2f}" if isinstance(x, float) else str(x) for x in a) + "}"
    return "{" + (f"{a:.2f}" if isinstance(a, float) else str(a)) + "}"


def stem(q):
    return q["question"].split("选项：")[0]


# wording shared by whole subtypes; it contains option words by accident ("左右手" has 右手, "会不会" has 不会)
BOILERPLATE = ("左右手均指演奏者本人", "会不会", "左右手")
EXAMPLE = re.compile(r"例如\s*\{[^}]*\}")


def stem_for_leak(q):
    s = EXAMPLE.sub("", stem(q))
    for b in BOILERPLATE:
        s = s.replace(b, "")
    return s


def times_in(text):
    return [float(x) for x in re.findall(r"第\s*([0-9]+(?:\.[0-9]+)?)\s*秒", text)]


def check_one(q, root, video_frames):
    fails = []
    r = EV.score(q, fmt_answer(q))
    if r["score"] < 1 or r["fmt_err"]:
        fails.append(("answer_roundtrip", f"gold {q['answer']!r} scores {r['score']}"))
    if q["answer_type"] == "choice":
        o = q.get("options") or []
        line = "选项：" + "；".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(o)) + "。"
        if line not in q["question"]:
            fails.append(("option_line", "option line in the text differs from options"))
        if len(set(map(str, o))) != len(o):
            fails.append(("option_line", "duplicate options"))
        if q["answer"] not in LETTERS[:len(o)]:
            fails.append(("option_line", f"answer {q['answer']} outside {len(o)} options"))
        else:
            right = str(o[LETTERS.index(q["answer"])])
            others = [str(x) for x in o if str(x) != right]
            st = stem_for_leak(q)
            if len(right) >= 2 and right in st and not any(x in st for x in others if len(x) >= 2):
                fails.append(("answer_in_stem", f"correct option '{right}' is in the stem"))
    else:
        a = q["answer"]
        pat = rf"(?<![A-G#0-9]){re.escape(a)}(?![#0-9])" if isinstance(a, str) and len(a) >= 2 else None
        if pat and re.search(pat, stem_for_leak(q)):
            fails.append(("answer_in_stem", f"answer '{a}' is in the stem"))
        elif pat and re.search(pat, "".join(EXAMPLE.findall(stem(q)))):
            fails.append(("example_is_answer", f"the example in the stem is the answer '{a}'"))
    ex = q.get("explanation") or ""
    if ex:
        want = "{" + (q["answer"] if q["answer_type"] == "choice" else "")
        concl = ex.rpartition("【结论】")[2]
        if q["answer_type"] == "choice" and want not in concl:
            fails.append(("explanation", f"conclusion '{concl[:40]}' does not name {q['answer']}"))
    dur = q.get("duration_s")
    if dur:
        ts = times_in(stem(q))
        if q["answer_type"] == "choice":
            for c in q.get("options") or []:
                ts += times_in(str(c))
        elif q["answer_type"] == "time_point" and isinstance(q["answer"], (int, float)):   # counts are not times
            ts.append(float(q["answer"]))
        elif q["answer_type"] == "time_span" and isinstance(q["answer"], list):
            ts += [float(x) for x in q["answer"]]
        bad = [t for t in ts if t > dur + 0.05 or t < 0]
        if bad:
            fails.append(("time_in_clip", f"times {bad} outside 0-{dur:.2f} s"))
    if q.get("video"):
        p = root / q["video"]
        if not p.exists() or p.stat().st_size == 0:
            fails.append(("media", "video missing or empty"))
        elif video_frames is not None:
            n = video_frames(p)
            src = q.get("source") or {}
            want_n = src.get("end_frame", 0) - src.get("start_frame", 0)
            if want_n and abs(n - want_n) > 1:
                fails.append(("media", f"video has {n} frames, expected {want_n}"))
    else:
        miss = [f for f in (q.get("media") or {}).get("frames", []) if not (root / f).exists()]
        if miss:
            fails.append(("media", f"{len(miss)} frame images missing"))
    return fails


def baselines(qs):
    rng = np.random.default_rng(0)
    rows = {}
    for kind in ("random", "copy"):
        try:
            outs = EV.baseline_outputs(qs, kind, rng)
        except Exception:  # noqa: BLE001
            continue
        for q in qs:
            if q["qid"] in outs:
                rows.setdefault((q["task"] + "/" + q["subtype"], kind), []).append(EV.score(q, outs[q["qid"]])["score"])
    return rows


def take_agreement(work):
    """Per take: score/human finger vs PMFinger guess, gold hand vs geometric hand, tip-to-key at onset."""
    out = []
    for p in sorted((work / "perf").glob("*.npz")):
        if p.name.endswith(".skeleton.npz"):
            continue
        tid = p.name[:-4]
        with np.load(p, allow_pickle=False) as z:
            d = {k: z[k] for k in ("gt_hand", "gt_finger", "label_hand", "note_pitch") if k in z.files}
        d.setdefault("label_hand", np.full(len(d["note_pitch"]), -1))
        meta = json.loads(p.with_suffix(".meta.json").read_text(encoding="utf-8")) if p.with_suffix(".meta.json").exists() else {}
        f = work / "fing" / f"{tid}.fing.npz"
        row = dict(take=tid, piece=meta.get("piece", ""), notes=len(d["note_pitch"]),
                   tip_to_key=meta.get("median_tip_to_key_whitekeys"))
        gh, gf, lh = d["gt_hand"], d["gt_finger"], d["label_hand"]
        both = (gh >= 0) & (lh >= 0)
        row["gold_vs_geom_hand"] = float((gh[both] == lh[both]).mean()) if both.any() else None
        row["n_gold_finger"] = int((gf >= 0).sum())
        if f.exists():
            with np.load(f, allow_pickle=False) as z:
                hp, fp = z["hand_pred"], z["finger_pred"]
            if len(hp) == len(gf):
                m = (gf >= 0) & (fp >= 0)
                row["gold_vs_auto_finger"] = float((gf[m] == fp[m]).mean()) if m.any() else None
                m2 = (gh >= 0) & (hp >= 0)
                row["gold_vs_auto_hand"] = float((gh[m2] == hp[m2]).mean()) if m2.any() else None
        out.append(row)
    return out


AUDIT_JS = """
<script>
const KEY='bench_audit_'+document.title;
let marks={};try{marks=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
function save(){try{localStorage.setItem(KEY,JSON.stringify(marks))}catch(e){};count()}
function count(){const n=Object.values(marks).filter(m=>m.v).length;document.getElementById('cnt').textContent=n+' / '+document.querySelectorAll('article').length}
document.querySelectorAll('article').forEach(a=>{const id=a.dataset.qid;
 a.querySelectorAll('button[data-v]').forEach(b=>{if((marks[id]||{}).v===b.dataset.v)b.classList.add('on');
  b.onclick=()=>{marks[id]=Object.assign(marks[id]||{},{v:b.dataset.v});a.querySelectorAll('button[data-v]').forEach(x=>x.classList.toggle('on',x===b));save()}});
 const t=a.querySelector('textarea');t.value=(marks[id]||{}).note||'';t.oninput=()=>{marks[id]=Object.assign(marks[id]||{},{note:t.value});save()}});
document.getElementById('dl').onclick=()=>{const rows=[['qid','subtype','verdict','note']];
 document.querySelectorAll('article').forEach(a=>{const m=marks[a.dataset.qid]||{};rows.push([a.dataset.qid,a.dataset.sub,m.v||'',(m.note||'').replace(/[\\n,]/g,' ')])});
 const blob=new Blob(['\\ufeff'+rows.map(r=>r.join(',')).join('\\n')],{type:'text/csv'});const u=URL.createObjectURL(blob);
 const l=document.createElement('a');l.href=u;l.download='audit_results.csv';l.click();URL.revokeObjectURL(u)};
count();
</script>"""


def audit_page(sample, rel_prefix, title):
    css = ("body{font:14px sans-serif;max-width:960px;margin:auto;padding:0 12px}article{border:1px solid #bbb;margin:14px 0;padding:10px}"
           "button{margin:4px;padding:4px 12px}button.on{background:#264;color:#fff}textarea{width:100%;height:36px}"
           "header{position:sticky;top:0;background:#fff;padding:8px 0;border-bottom:1px solid #ccc}")
    cards = []
    for q in sample:
        opts = "".join(f"<li>{LETTERS[i]}. {html.escape(str(o))}</li>" for i, o in enumerate(q.get("options") or []))
        media = (f"<video src='{rel_prefix}{html.escape(q['video'])}' controls preload='none' width='640'></video>" if q.get("video")
                 else "".join(f"<img src='{rel_prefix}{html.escape(f)}' width='200'>" for f in (q.get("media") or {}).get("frames", [])))
        src = q.get("source") or {}
        cards.append(
            f"<article data-qid='{html.escape(q['qid'])}' data-sub='{html.escape(q['task'] + '/' + q['subtype'])}'>"
            f"<h3>{html.escape(q['task'] + ' ' + q['subtype'])}</h3>{media}"
            f"<p>{html.escape(q['question'])}</p><ol style='list-style:none'>{opts}</ol>"
            f"<p><b>标准答案</b> {html.escape(json.dumps(q['answer'], ensure_ascii=False))}</p>"
            f"<details><summary>解析</summary><p>{html.escape(q.get('explanation', ''))}</p></details>"
            f"<small>{html.escape(str(src.get('piece', '')))} · 原视频第 {src.get('start_frame', '?')}–{src.get('end_frame', '?')} 帧 · {html.escape(q['qid'])}</small><br>"
            "<button data-v='对'>对</button><button data-v='错'>错</button><button data-v='有歧义'>有歧义</button>"
            "<button data-v='看不清'>看不清</button><textarea placeholder='备注：错在哪 / 为什么有歧义'></textarea></article>")
    return (f"<!doctype html><meta charset=utf-8><title>{html.escape(title)}</title><style>{css}</style>"
            f"<header><b>{html.escape(title)}</b> · 已标 <span id=cnt></span> · <button id=dl>导出结果</button>"
            " · 判断标准：只看视频，标准答案是否唯一正确</header>" + "\n".join(cards) + AUDIT_JS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True)
    ap.add_argument("--work", default="", help="folder with perf/ and fing/ (cross-source agreement)")
    ap.add_argument("--per-subtype", type=int, default=3)
    ap.add_argument("--extra-finger", type=int, default=20, help="extra sampled questions whose answer is a finger")
    ap.add_argument("--no-video-check", action="store_true", help="skip opening every video (faster)")
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    root = Path(a.release)
    out = Path(a.out) if a.out else root / "audit"
    out.mkdir(parents=True, exist_ok=True)
    qs = [json.loads(x) for x in (root / "questions.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    for q in qs:
        q.setdefault("fact_id", q.get("qid"))

    vf = None
    if not a.no_video_check:
        import cv2

        def vf(p):
            c = cv2.VideoCapture(str(p))
            n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
            c.release()
            return n

    fails = []
    seen = {}
    for i, q in enumerate(qs, 1):
        for c, d in check_one(q, root, vf):
            fails.append(dict(qid=q["qid"], subtype=q["task"] + "/" + q["subtype"], check=c, detail=d))
        src = q.get("source") or {}
        fr = (q.get("media") or {}).get("frames") or [None]
        key = (src.get("video_id"), src.get("start_frame"), src.get("end_frame"), fr[0], fr[-1], stem(q))
        if key in seen:
            fails.append(dict(qid=q["qid"], subtype=q["task"] + "/" + q["subtype"], check="duplicate", detail=f"same as {seen[key]}"))
        seen[key] = q["qid"]
        if i % 200 == 0:
            print(f"  checked {i}/{len(qs)}", flush=True)
    with open(out / "failures.jsonl", "w", encoding="utf-8") as f:
        for x in fails:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    base = baselines(qs)
    subs = sorted({q["task"] + "/" + q["subtype"] for q in qs})
    n_by = collections.Counter(q["task"] + "/" + q["subtype"] for q in qs)

    rnd = random.Random(a.seed)
    by = collections.defaultdict(list)
    for q in qs:
        by[q["task"] + "/" + q["subtype"]].append(q)
    sample, picked = [], set()
    for k in subs:
        lst = sorted(by[k], key=lambda q: q["qid"])
        rnd.shuffle(lst)
        for q in lst[:a.per_subtype]:
            sample.append(q)
            picked.add(q["qid"])
    fing = [q for q in qs if "手指" in q["subtype"] and q["qid"] not in picked]
    rnd.shuffle(fing)
    sample += fing[:a.extra_finger]
    with open(out / "sample.jsonl", "w", encoding="utf-8") as f:
        for q in sample:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    prefix = os.path.relpath(root.resolve(), out.resolve()).replace(os.sep, "/") + "/"
    (out / "audit.html").write_text(audit_page(sample, prefix, f"人工抽查 {root.name}（{len(sample)} 题）"), encoding="utf-8")

    agree = take_agreement(Path(a.work)) if a.work else []

    fc = collections.Counter(x["check"] for x in fails)
    L = [f"# 数据质量检查：{root.name}", "", f"题目 {len(qs)} 道，子题 {len(subs)} 个。", "",
         "## 1. 自动检查", "", "| 检查 | 不通过题数 | 说明 |", "|---|---:|---|"]
    desc = dict(answer_roundtrip="标准答案交给判分程序能否判对", option_line="题干里的选项与选项列表一致、无重复",
                explanation="解析结论与标准答案一致", answer_in_stem="题干里直接写出了正确答案", example_is_answer="题干里的示例答案恰好等于正确答案",
                media="视频存在且帧数正确", time_in_clip="答案和选项里的时间都在片段内", duplicate="同一片段同一题出现两次")
    for c in desc:
        L.append(f"| {c} | {fc.get(c, 0)} | {desc[c]} |")
    if fails:
        L += ["", "不通过的前 30 条（全部见 failures.jsonl）：", ""]
        L += [f"- `{x['qid']}` {x['subtype']} · {x['check']}：{x['detail']}" for x in fails[:30]]
    L += ["", "## 2. 捷径基线（不看视频能得多少分）", "",
          "随机：随机选/随机填；总选同一字母：每题都选这个子题里最常见的正确字母；总答同一内容：每题都答这个子题里最常见的答案内容"
          "（两者都是事后挑最有利的，是上限）；照抄现在：预测题答“和现在一样”。模型分数要明显高于这几列才说明它看懂了视频。", "",
          "| 子题 | 题数 | 随机 | 总选同一字母 | 总答同一内容 | 照抄现在 |", "|---|---:|---:|---:|---:|---:|"]
    for k in subs:
        lst = by[k]
        let = collections.Counter(q["answer"] for q in lst if q["answer_type"] == "choice")
        con = collections.Counter((str(q["options"][LETTERS.index(q["answer"])]) if q["answer_type"] == "choice"
                                   else json.dumps(q["answer"], ensure_ascii=False)) for q in lst)
        cells = [f"{np.mean(base[(k, 'random')]):.0%}" if base.get((k, "random")) else "—",
                 f"{max(let.values()) / len(lst):.0%}" if let else "—",
                 f"{max(con.values()) / len(lst):.0%}",
                 f"{np.mean(base[(k, 'copy_current')]):.0%}" if base.get((k, "copy_current")) else "—"]
        L.append(f"| {k} | {n_by[k]} | " + " | ".join(cells) + " |")
    if agree:
        L += ["", "## 3. 跨来源一致性（每遍录音）", "",
              "金标指法 vs 骨架自动指法、金标手 vs 手掌远近判的手，一致率越高越可信；明显偏低的录音值得人工看。",
              "指尖到键距离是按下时最近指尖离该键的中位距离（白键宽），大于 1 说明键盘标定或音画同步有问题。", "",
              "| 录音 | 曲目 | 音符 | 金标指法音数 | 金标手 vs 几何手 | 金标手 vs 自动手 | 金标指 vs 自动指 | 指尖到键 |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
        pct = lambda v: "—" if v is None else f"{v:.0%}"  # noqa: E731
        for r in agree:
            L.append(f"| {r['take']} | {r['piece']} | {r['notes']} | {r['n_gold_finger']} | {pct(r.get('gold_vs_geom_hand'))}"
                     f" | {pct(r.get('gold_vs_auto_hand'))} | {pct(r.get('gold_vs_auto_finger'))} | {r.get('tip_to_key', '—')} |")
    L += ["", "## 4. 人工抽查", "",
          f"`audit.html` 里有 {len(sample)} 道：每个子题随机 {a.per_subtype} 道，另加 {min(a.extra_finger, len(fing))} 道指法题。"
          "逐题看视频，判断标准答案是否唯一正确，点 对 / 错 / 有歧义 / 看不清，写一句备注，最后点“导出结果”得到 audit_results.csv。",
          "建议两个人各看一遍，有分歧的题再一起看。"]
    (out / "audit_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"checks: {len(fails)} failures {dict(fc)}")
    print(f"-> {out / 'audit_report.md'}, {out / 'failures.jsonl'}, {out / 'audit.html'} ({len(sample)} questions)")


if __name__ == "__main__":
    main()
