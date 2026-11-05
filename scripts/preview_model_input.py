"""Show exactly what the model receives in the evaluation, for a few questions per subtype.

  python3 -B scripts/preview_model_input.py --release <release_dir> [--setting F|V] [--per-subtype 2]

Builds the messages with run_models.build_messages (the same function the evaluation uses) and writes
<release_dir>/model_input.html: the system prompt, then for every sampled question the user message item by
item (text, frames, video) as sent, and below a line what is NOT sent (answer, explanation).
Also writes model_input_sample.json (the raw messages, images as file paths).
"""
import argparse
import collections
import html
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_models import build_messages  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True)
    ap.add_argument("--setting", default="F", choices=["F", "V", "blind"])
    ap.add_argument("--per-subtype", type=int, default=2)
    a = ap.parse_args()
    root = Path(a.release)
    system = (root / "system_prompt.txt").read_text(encoding="utf-8").strip()
    qs = [json.loads(x) for x in (root / "questions.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    by = collections.defaultdict(list)
    for q in qs:
        if len(by[(q["task"], q["subtype"])]) < a.per_subtype:
            by[(q["task"], q["subtype"])].append(q)
    sample = [q for k in sorted(by) for q in by[k]]
    raw, cards = [], []
    for q in sample:
        msgs = build_messages(q, str(root), system, a.setting, 0, True, "file")
        raw.append(dict(qid=q["qid"], messages=msgs, answer=q["answer"]))
        items = []
        for c in msgs[1]["content"]:
            if c["type"] == "text":
                items.append(f"<div class=t>{html.escape(c['text'])}</div>")
            elif c["type"] == "image_url":
                rel = os.path.relpath(c["image_url"]["url"], root)
                items.append(f"<img src='{html.escape(rel)}'>")
            elif c["type"] == "video_url":
                rel = os.path.relpath(c["video_url"]["url"].removeprefix("file://"), root)
                items.append(f"<video src='{html.escape(rel)}' controls width=640></video>")
        ans = json.dumps(q["answer"], ensure_ascii=False)
        cards.append(f"<article><h3>{html.escape(q['task'] + ' ' + q['subtype'])}</h3>"
                     f"<div class=lab>发给模型的 user 消息（按顺序）</div>{''.join(items)}"
                     f"<div class=lab>以下不发给模型</div><p><b>标准答案</b> {html.escape(ans)}</p>"
                     f"<p>{html.escape(q.get('explanation', ''))}</p><small>{q['qid']}</small></article>")
    css = ("body{font:14px sans-serif;max-width:1000px;margin:auto}article{border:1px solid #bbb;margin:16px 0;padding:10px}"
           ".t{background:#f3f3f3;margin:4px 0;padding:4px;white-space:pre-wrap}img{max-width:320px;margin:2px}"
           ".lab{color:#a33;font-weight:bold;margin-top:8px}pre{white-space:pre-wrap;background:#eef}")
    (root / "model_input.html").write_text(
        f"<meta charset=utf-8><title>模型输入</title><style>{css}</style><h2>setting {a.setting} · 每题都先发这段 system prompt</h2>"
        f"<pre>{html.escape(system)}</pre>" + "\n".join(cards), encoding="utf-8")
    (root / "model_input_sample.json").write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(sample)} questions -> {root / 'model_input.html'}, {root / 'model_input_sample.json'}")


if __name__ == "__main__":
    main()
