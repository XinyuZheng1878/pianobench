"""One self-contained web page of sample questions (videos embedded), to open anywhere or share.

  python3 -B scripts/make_sample_page.py --release <video_release_dir> [--per-subtype 2] [--out samples.html]

Takes --per-subtype questions per subtype (seeded random, different answers preferred), groups them by task
(T1-T5) and writes ONE html file: video, question, options with the correct one marked, answer, explanation
(folded) and where the clip comes from. Videos are re-encoded small (480 px wide, CRF 28) when ffmpeg is
available and embedded as data, so the file needs nothing else; about 3-8 MB for 2 per subtype.
"""
import argparse
import base64
import collections
import html
import json
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

LETTERS = "ABCDEFG"
TASKS = {
    "T1": ("理解", "给定一段时间，弹了什么：按键序列、音程和弦、织体、音区、旋律走向"),
    "T2": ("识别", "某一个音是哪只手、哪根手指弹的"),
    "T3": ("动态", "一段时间里的按键次数、保持与松开、双手配合、手型张开与移动"),
    "T4": ("定位", "某件事发生在视频的第几秒"),
    "T5": ("预测", "视频最后一刻就是“现在”，问接下来很短时间内会发生什么"),
}
CSS = """
:root{--bg:#f6f5f2;--card:#fff;--ink:#1d1d1b;--mute:#6b6a66;--line:#e2e0da;--acc:#1f6f5c;--accbg:#e4f1ec;--tag:#efeee9}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#1f1f1d;--ink:#ecebe7;--mute:#a19f99;--line:#34332f;--acc:#7cc7b1;--accbg:#203530;--tag:#2a2926}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 -apple-system,"PingFang SC","Noto Sans SC",sans-serif}
.wrap{max-width:980px;margin:0 auto;padding:28px 16px 60px}h1{font-size:26px;margin:0 0 6px}.sub{color:var(--mute);margin:0 0 18px}
.stats{display:flex;flex-wrap:wrap;gap:10px;margin:0 0 26px}.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px}
.stat b{display:block;font-size:20px}.stat span{color:var(--mute);font-size:13px}
nav{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 28px}nav a{color:var(--acc);text-decoration:none;border:1px solid var(--line);border-radius:999px;padding:4px 12px;background:var(--card)}
h2{font-size:20px;margin:36px 0 4px}.tdesc{color:var(--mute);margin:0 0 14px}
.q{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin:0 0 16px;display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:16px}
@media (max-width:760px){.q{grid-template-columns:1fr}}
video{width:100%;border-radius:8px;background:#000}.tags{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 8px}
.tag{background:var(--tag);border-radius:6px;padding:1px 8px;font-size:12px;color:var(--mute)}.tag.k{color:var(--ink);font-weight:600}
.stem{margin:0 0 10px}ol{list-style:none;padding:0;margin:0 0 10px}li{border:1px solid var(--line);border-radius:8px;padding:5px 10px;margin:4px 0}
li.ok{border-color:var(--acc);background:var(--accbg);font-weight:600}li.ok::after{content:"  ✓ 正确答案";color:var(--acc);font-weight:400;font-size:13px}
.ans{margin:0 0 8px}.ans code{background:var(--accbg);color:var(--acc);border-radius:6px;padding:1px 8px;font-size:15px}
details{color:var(--mute);font-size:14px}summary{cursor:pointer;color:var(--acc)}.src{color:var(--mute);font-size:12px;margin-top:8px}
"""


def small_video(path, tmp):
    if not shutil.which("ffmpeg"):
        return path.read_bytes()
    out = Path(tmp) / (path.stem + ".mp4")
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-vf", "scale=480:-2", "-c:v", "libx264",
                        "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart",
                        str(out)], capture_output=True)
    return out.read_bytes() if r.returncode == 0 else path.read_bytes()


def pick(qs, n, seed):
    by = collections.defaultdict(list)
    for q in qs:
        by[(q["task"], q["subtype"])].append(q)
    rnd = random.Random(seed)
    out = []
    for k in sorted(by):
        lst = sorted(by[k], key=lambda q: q["qid"])
        rnd.shuffle(lst)
        chosen, seen = [], set()
        for q in lst:                                  # different answers first
            a = json.dumps(q["answer"], ensure_ascii=False)
            if a not in seen:
                chosen.append(q)
                seen.add(a)
            if len(chosen) == n:
                break
        chosen += [q for q in lst if q not in chosen][: n - len(chosen)]
        out += chosen
    return out


def card(q, data_uri):
    opts = q.get("options") or []
    lis = "".join(f"<li class='{'ok' if q['answer_type'] == 'choice' and LETTERS[i] == q['answer'] else ''}'>"
                  f"{LETTERS[i]}. {html.escape(str(o))}</li>" for i, o in enumerate(opts))
    stem = q["question"].split("选项：")[0]
    a = q["answer"]
    ans = (f"{{{a}}}" if q["answer_type"] == "choice" else
           "{" + (", ".join(map(str, a)) if isinstance(a, list) else str(a)) + "}")
    src = q.get("source") or {}
    media = (f"<video controls preload='metadata' playsinline src='{data_uri}'></video>" if data_uri
             else "<p class='tdesc'>（视频缺失）</p>")
    return (f"<div class='q'><div>{media}</div><div>"
            f"<div class='tags'><span class='tag k'>{html.escape(q['subtype'])}</span>"
            f"<span class='tag'>{'选择题' if q['answer_type'] == 'choice' else '填空题'}</span>"
            f"<span class='tag'>{q.get('duration_s', 0):.1f} 秒</span></div>"
            f"<p class='stem'>{html.escape(stem)}</p>{'<ol>' + lis + '</ol>' if lis else ''}"
            f"<p class='ans'>标准答案 <code>{html.escape(ans)}</code></p>"
            f"<details><summary>解析</summary><p>{html.escape(q.get('explanation', '') or '—')}</p></details>"
            f"<div class='src'>{html.escape(str(src.get('piece', '')))} · 原视频第 {src.get('start_frame', '?')}–"
            f"{src.get('end_frame', '?')} 帧 · {html.escape(q['qid'])}</div></div></div>")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True)
    ap.add_argument("--per-subtype", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="")
    ap.add_argument("--title", default="钢琴演奏视频问答 benchmark · 样题")
    a = ap.parse_args()
    root = Path(a.release)
    qs = [json.loads(x) for x in (root / "questions.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    sample = pick(qs, a.per_subtype, a.seed)
    n_sub = len({(q["task"], q["subtype"]) for q in qs})
    pieces = sorted({str((q.get("source") or {}).get("piece", "")).split("_TIME")[0] for q in qs} - {""})
    by_task = collections.Counter(q["task"] for q in qs)
    secs = collections.defaultdict(list)
    with tempfile.TemporaryDirectory() as tmp:
        for i, q in enumerate(sample, 1):
            uri = ""
            if q.get("video") and (root / q["video"]).exists():
                uri = "data:video/mp4;base64," + base64.b64encode(small_video(root / q["video"], tmp)).decode()
            secs[q["task"]].append(card(q, uri))
            if i % 10 == 0:
                print(f"  {i}/{len(sample)}", flush=True)
    stats = (f"<div class='stats'><div class='stat'><b>{len(qs)}</b><span>道题</span></div>"
             f"<div class='stat'><b>{n_sub}</b><span>个子题</span></div>"
             f"<div class='stat'><b>{sum(q['answer_type'] == 'choice' for q in qs)}</b><span>道选择题</span></div>"
             f"<div class='stat'><b>{sum(q['answer_type'] != 'choice' for q in qs)}</b><span>道填空题</span></div>"
             f"<div class='stat'><b>{len(sample)}</b><span>道样题（本页）</span></div></div>")
    nav = "<nav>" + "".join(f"<a href='#{t}'>{t} {TASKS.get(t, ('', ''))[0]} · {by_task[t]} 题</a>" for t in sorted(secs)) + "</nav>"
    body = "".join(f"<h2 id='{t}'>{t} {TASKS.get(t, ('', ''))[0]}</h2><p class='tdesc'>{TASKS.get(t, ('', ''))[1]}"
                   f"（全集 {by_task[t]} 题）</p>" + "".join(secs[t]) for t in sorted(secs))
    intro = ("<p class='sub'>俯拍钢琴演奏视频 + 问题 + 唯一答案。答案由 MIDI、手部骨架和乐谱指法程序计算；"
             "模型只看到视频和题目，答案写在花括号里自动判分。曲目：" + html.escape("、".join(pieces)) + "。</p>")
    page = (f"<!doctype html><html lang='zh'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(a.title)}</title><style>{CSS}</style></head><body><div class='wrap'>"
            f"<h1>{html.escape(a.title)}</h1>{intro}{stats}{nav}{body}</div></body></html>")
    out = Path(a.out) if a.out else root / "samples.html"
    out.write_text(page, encoding="utf-8")
    print(f"{len(sample)} sample questions -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
