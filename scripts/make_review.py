"""Benchmark step 6: build an offline review package (web page + small frames) for human checking.

  python3 -B scripts/make_review.py --bench ../gold_work/bench_v2 --out ../gold_work/review_v2 \
      --sample 3 --zip /inspire/hdd/project/project-public/xuguangjun-253208580299/review_v2.zip

Which questions: every question that needs human review (fingers from gold labels), plus --sample questions
of every other subtype. The package opens in any browser without a server: unzip, double-click index.html.
Reviewers mark each question 通过 / 有问题, then click 导出结果; the downloaded approvals_<name>.jsonl goes to
export_bench_gold.py --approvals. Decisions are bound to qid and fact_hash, so they cannot approve a changed fact.
"""
import argparse
import collections
import html
import json
import random
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmfinger.keys import key_x, pitch_name  # noqa: E402

LETTERS = "ABCDEFG"


def pick_frames(n, keep_every, last=True):
    idx = list(range(0, n, keep_every))
    if last and n - 1 not in idx:
        idx.append(n - 1)
    return idx


NAME_TO_PITCH = {pitch_name(p): p for p in range(21, 109)}


def asked_keys(slots):
    ks = [slots.get(k) for k in ("key", "key_a", "key_b")] + list(slots.get("keys") or [])
    return [k for k in dict.fromkeys(ks) if k in NAME_TO_PITCH]


def key_marker(meta, crop, tile_w, key):
    """Pixel position (in the delivered tile) of the middle of `key` along the keyboard's front edge."""
    import cv2
    corners = np.float32(meta["keyboard_corners_px"])
    inv = cv2.getPerspectiveTransform(np.float32([[0, 0], [52, 0], [52, 1], [0, 1]]), corners)
    x = key_x(NAME_TO_PITCH[key])
    top, front = cv2.perspectiveTransform(np.float32([[[x, 0.0]], [[x, 1.0]]]), inv).reshape(2, 2)
    x0, y0, W, _ = crop
    s = tile_w / W
    return [(float((p[0] - x0) * s), float((p[1] - y0) * s + 26)) for p in (top, front)]   # 26 px label strip


def evidence(row):
    """The facts the answer rests on, as short lines (key events and geometry, in clip seconds)."""
    out = []
    for a in row["atoms"]:
        if a.get("source") == "骨架" and a["id"].startswith("p"):
            continue                                    # routine palm positions: too many to read
        s = f"{a['t']:.3f}s  {a['subject']} {a['predicate']} {a['object']}"
        if "finger" in a:
            s += f"（{a['finger']}，{a.get('finger_source', '')}）"
        if "x" in a:
            s += f"（横坐标 {a['x']:.2f} 白键）"
        out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--out", required=True, help="new folder for the review package")
    ap.add_argument("--sample", type=int, default=3, help="questions per subtype that needs no finger review")
    ap.add_argument("--width", type=int, default=800, help="frame width in the package (pixels)")
    ap.add_argument("--every", type=int, default=2, help="keep every n-th delivered frame (the last one always)")
    ap.add_argument("--zip", default="", help="also write the package as this .zip file")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    bench, out = Path(a.bench).resolve(), Path(a.out).resolve()
    if out.exists():
        raise SystemExit(f"{out} already exists; choose a new --out")
    rows = [json.loads(l) for l in open(bench / "questions.jsonl", encoding="utf-8")]
    rows = [r for r in rows if "media" in r and r.get("status") != "delivery_failed"]

    rnd = random.Random(a.seed)
    by = collections.defaultdict(list)
    for r in rows:
        by[r["question"]["task"] + "/" + r["question"]["subtype"]].append(r)
    chosen = []
    for key in sorted(by):
        lst = by[key]
        if any(r["needs_human"] for r in lst):
            chosen += [r for r in lst if r["needs_human"]]
        else:
            chosen += rnd.sample(lst, min(a.sample, len(lst)))

    (out / "img").mkdir(parents=True)
    metas = {}
    items = []
    for r in chosen:
        q, frames = r["question"], r["media"]["frames"]
        t0 = frames[0]["t"]
        vid = q["video_id"]
        if vid not in metas:
            metas[vid] = json.loads((bench / "perf" / f"{vid}.meta.json").read_text(encoding="utf-8"))
        keys = asked_keys(q["slots"])
        imgs = []
        for i in pick_frames(len(frames), a.every):
            src = bench / frames[i]["path"]
            dst = out / "img" / r["qid"][-20:] / f"{i:03d}.jpg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(src) as im:
                im = im.convert("RGB")
                dr = ImageDraw.Draw(im)
                for j, k in enumerate(keys):       # mark the asked key(s): a coloured bar along the key
                    (xa, ya), (xb, yb) = key_marker(metas[vid], r["media"]["crop"], im.width, k)
                    col = [(255, 200, 0), (0, 200, 255), (255, 80, 200)][j % 3]
                    dr.line([(xa, ya), (xb, yb)], fill=col, width=3)
                    dr.text((xa - 10, max(28, ya - 14)), k, fill=col)
                h = round(im.height * a.width / im.width)
                im.resize((a.width, h)).save(dst, quality=72)
            imgs.append(dict(src=str(dst.relative_to(out)), label=f"第 {i} 帧 · {frames[i]['t'] - t0:.3f} 秒"))
        ans = r["private_target"]["answer"]
        shown = ans
        if q.get("options"):
            shown = f"{ans}. {q['options'][LETTERS.index(ans)]}"
        elif isinstance(ans, list):
            shown = ", ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in ans)
        elif isinstance(ans, float):
            shown = f"{ans:.3f}"
        items.append(dict(qid=r["qid"], fact_hash=r["fact_hash"], task=q["task"], subtype=q["subtype"],
                          video=q["video_id"], finger=r["needs_human"], keys=keys, question=q["question"], answer=str(shown),
                          evidence=evidence(r), images=imgs))

    page = PAGE.replace("__DATA__", json.dumps(items, ensure_ascii=False)).replace("__BENCH__", html.escape(bench.name))
    (out / "index.html").write_text(page, encoding="utf-8")
    n_f = sum(i["finger"] for i in items)
    print(f"review package: {len(items)} questions ({n_f} finger, {len(items) - n_f} sampled) -> {out}/index.html")
    if a.zip:
        z = Path(a.zip)
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(out.rglob("*")):
                if p.is_file():
                    zf.write(p, Path(out.name) / p.relative_to(out))
        print(f"zip: {z} ({z.stat().st_size / 1e6:.1f} MB)")


PAGE = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Benchmark 人工复核 · __BENCH__</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1d2330;--quiet:#5e6675;--line:#dfe3ea;--ok:#1f8a4c;--bad:#c23a2b;--acc:#2f5bd3}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--card);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;flex-wrap:wrap;gap:10px;align-items:center}
header h1{font-size:16px;margin:0 8px 0 0}input,select,button{font:inherit;padding:5px 9px;border:1px solid var(--line);border-radius:6px;background:#fff}
button{cursor:pointer}button.primary{background:var(--acc);color:#fff;border-color:var(--acc)}
main{max-width:1100px;margin:0 auto;padding:16px}
article{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:0 0 16px}
article.ok{border-left:6px solid var(--ok)}article.bad{border-left:6px solid var(--bad)}
.meta{color:var(--quiet);font-size:13px}.ans{font-weight:600;color:var(--acc)}
.ev{font:13px/1.5 ui-monospace,Menlo,monospace;background:#f2f4f8;border-radius:6px;padding:8px;max-height:220px;overflow:auto;white-space:pre-wrap}
.frames{display:flex;gap:8px;overflow-x:auto;padding:6px 0}.frames figure{margin:0;flex:0 0 auto}
.frames img{width:380px;border-radius:4px;display:block;cursor:zoom-in}
#zoom{position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;align-items:center;justify-content:center;z-index:9;cursor:zoom-out}
#zoom img{max-width:96vw;max-height:92vh}#zoom div{position:absolute;bottom:12px;color:#fff;font-size:14px}.frames figcaption{font-size:12px;color:var(--quiet)}
.act{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:8px}.act input{flex:1;min-width:200px}
.tag{display:inline-block;font-size:12px;padding:1px 7px;border-radius:9px;background:#fff3d6;color:#8a5a00;margin-left:6px}
</style></head><body>
<header><h1>Benchmark 人工复核</h1>
<label>复核人 <input id="who" placeholder="填你的名字" size="8"></label>
<select id="filter"></select>
<span id="progress" class="meta"></span>
<button id="import">导入已有结果</button><input type="file" id="file" accept=".jsonl" hidden>
<button class="primary" id="export">导出结果</button></header>
<main id="list"></main><div id="zoom"><img><div></div></div>
<script>
const DATA = __DATA__;
const KEY = "bench-review-__BENCH__";
let state = {};
try { state = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { state = {}; }
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} };
const who = document.getElementById("who");
try { who.value = localStorage.getItem(KEY + "-who") || ""; } catch (e) {}
who.oninput = () => { try { localStorage.setItem(KEY + "-who", who.value); } catch (e) {} };
const keys = [...new Set(DATA.map(d => d.task + "/" + d.subtype))];
const sel = document.getElementById("filter");
sel.innerHTML = '<option value="">全部子题</option><option value="__todo">只看未复核</option><option value="__finger">只看指法题</option>' +
  keys.map(k => `<option>${k}</option>`).join("");
function esc(s){return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));}
function render(){
  const f = sel.value, list = document.getElementById("list");
  const shown = DATA.filter(d => !f || (f==="__todo" ? !state[d.qid] : f==="__finger" ? d.finger : d.task+"/"+d.subtype===f));
  list.innerHTML = shown.map(d => {
    const s = state[d.qid] || {};
    return `<article id="${d.qid}" class="${s.decision==="approve"?"ok":s.decision==="reject"?"bad":""}">
      <div class="meta">${esc(d.task)} / ${esc(d.subtype)} · ${esc(d.video)} · ${esc(d.qid.slice(-20))}${d.finger?'<span class="tag">指法题，必须复核</span>':''}</div>
      <p>${esc(d.question)}</p>
      <p>程序给出的答案：<span class="ans">${esc(d.answer)}</span>${d.keys.length?`　<span class="meta">画面里已用彩色竖线标出：${esc(d.keys.join("、"))}</span>`:""}</p>
      <details><summary>程序依据（按键与骨架记录，时间为片段内秒数）</summary><div class="ev">${esc(d.evidence.join("\n"))}</div></details>
      <div class="frames">${d.images.map(i=>`<figure><img loading="lazy" src="${i.src}"><figcaption>${esc(i.label)}</figcaption></figure>`).join("")}</div>
      <div class="act"><button data-q="${d.qid}" data-d="approve">✔ 通过：答案对，画面里看得出来</button>
      <button data-q="${d.qid}" data-d="reject">✘ 有问题</button>
      <input data-n="${d.qid}" placeholder="有问题时写原因（如：看不清手指 / 答案错 / 时间对不上）" value="${esc(s.note||"")}"></div></article>`;
  }).join("") || "<p>没有符合条件的题。</p>";
  const done = DATA.filter(d => state[d.qid]).length;
  document.getElementById("progress").textContent = `已复核 ${done} / ${DATA.length}（指法题 ${DATA.filter(d=>d.finger&&state[d.qid]).length} / ${DATA.filter(d=>d.finger).length}）`;
}
document.getElementById("list").onclick = e => {
  const b = e.target.closest("button[data-q]"); if (!b) return;
  if (!who.value.trim()) { alert("请先在左上角填写复核人名字"); who.focus(); return; }
  const q = b.dataset.q, note = document.querySelector(`input[data-n="${q}"]`).value;
  state[q] = {decision: b.dataset.d, note, reviewer: who.value.trim(), at: new Date().toISOString()};
  save(); const y = window.scrollY; render(); window.scrollTo(0, y);
};
document.getElementById("list").oninput = e => {
  const q = e.target.dataset.n; if (q && state[q]) { state[q].note = e.target.value; save(); }
};
sel.onchange = render;
const zoom = document.getElementById("zoom");
document.getElementById("list").addEventListener("click", e => {
  if (e.target.tagName !== "IMG") return;
  zoom.querySelector("img").src = e.target.src;
  zoom.querySelector("div").textContent = e.target.nextElementSibling ? e.target.nextElementSibling.textContent + "　（← → 切换，点击关闭）" : "";
  zoom.style.display = "flex"; zoom.cur = e.target;
});
zoom.onclick = () => { zoom.style.display = "none"; };
document.addEventListener("keydown", e => {
  if (zoom.style.display !== "flex" || !zoom.cur) return;
  const fig = zoom.cur.parentElement, nxt = e.key === "ArrowRight" ? fig.nextElementSibling : e.key === "ArrowLeft" ? fig.previousElementSibling : null;
  if (e.key === "Escape") zoom.style.display = "none";
  if (nxt) { const im = nxt.querySelector("img"); zoom.cur = im; zoom.querySelector("img").src = im.src;
    zoom.querySelector("div").textContent = nxt.querySelector("figcaption").textContent + "　（← → 切换，点击关闭）"; }
});
document.getElementById("export").onclick = () => {
  const lines = DATA.filter(d => state[d.qid]).map(d => JSON.stringify(Object.assign({qid:d.qid, fact_hash:d.fact_hash}, state[d.qid])));
  if (!lines.length) { alert("还没有复核任何题"); return; }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lines.join("\n") + "\n"], {type:"application/json"}));
  a.download = `approvals_${(who.value.trim()||"reviewer")}.jsonl`; a.click();
};
document.getElementById("import").onclick = () => document.getElementById("file").click();
document.getElementById("file").onchange = async e => {
  const t = await e.target.files[0].text(); let n = 0;
  for (const l of t.split("\n")) { if (!l.trim()) continue; try { const x = JSON.parse(l);
    const d = DATA.find(d => d.qid === x.qid && d.fact_hash === x.fact_hash);
    if (d) { state[x.qid] = {decision:x.decision, note:x.note||"", reviewer:x.reviewer||"", at:x.at||""}; n++; } } catch (err) {} }
  save(); render(); alert(`导入了 ${n} 条`);
};
render();
</script></body></html>
"""

if __name__ == "__main__":
    main()
