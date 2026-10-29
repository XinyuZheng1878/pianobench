"""Step 8a: run a model on the benchmark through any OpenAI-compatible endpoint (vLLM, SGLang, LM Studio,
mlx_vlm.server, hosted APIs).

  # setting F (official frames, main leaderboard)
  python -B scripts/run_models.py --release work/release --base-url http://localhost:8001/v1 \
      --model Qwen/Qwen3-VL-8B-Instruct --tag qwen --setting F --out work/runs/qwen_F.jsonl
  # setting V (native mp4; the server samples frames)    --setting V
  # blind baseline (question only, no video)               --setting blind

Messages: system = system_prompt.txt; user = an intro line, then for setting F every frame preceded by
its label "[第 k 帧 · t 秒]", then (boundary-frame questions) "图 A：" / "图 B：" images, then the question.
Greedy decoding, max 1024 output tokens. Resumable: answered qids are skipped; failed ones retried.
Output lines: {"qid", "model", "setting", "output", "latency_s", "usage", "error"?}
"""
import argparse
import base64
import io
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image


def img_url(path, max_side, as_path):
    if as_path and not max_side:
        return os.path.abspath(path)
    im = Image.open(path).convert("RGB")
    if max_side and max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((int(im.width * s), int(im.height * s)))
    if as_path:
        cache = os.path.join(os.path.dirname(os.path.abspath(path)), f".s{max_side}")
        os.makedirs(cache, exist_ok=True)
        out = os.path.join(cache, os.path.basename(path))
        if not os.path.exists(out):
            im.save(out, quality=90)
        return out
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def build_messages(q, root, system, setting, max_side, as_path, video_as):
    content = []
    if setting == "F":
        fl = q["media"]["frames"]
        fps = q["frame_fps"]
        content.append({"type": "text", "text": f"下面是视频按 {fps} fps 抽取的 {len(fl)} 帧，按时间顺序排列，"
                                                f"第 k 帧对应第 k/{fps} 秒。"})
        for k, f in enumerate(fl):
            t = f"{k / fps:.3f}".rstrip("0").rstrip(".")
            content.append({"type": "text", "text": f"[第 {k} 帧 · {t if '.' in t else t + '.0'} 秒]"})
            content.append({"type": "image_url", "image_url": {"url": img_url(os.path.join(root, f), max_side, as_path)}})
    elif setting == "V":
        p = os.path.abspath(os.path.join(root, q["media"]["video"]))
        if video_as == "file":
            url = "file://" + p
        else:
            url = "data:video/mp4;base64," + base64.b64encode(open(p, "rb").read()).decode()
        content.append({"type": "video_url", "video_url": {"url": url}})
    else:
        content.append({"type": "text", "text": "（本题不提供视频，请根据题目尽量作答。）"})
    if setting != "blind" and q.get("range") and q["range"]["type"] == "boundary_frames":
        for lab, key in (("A", "image_a"), ("B", "image_b")):
            content.append({"type": "text", "text": f"图 {lab}："})
            content.append({"type": "image_url", "image_url": {"url": img_url(os.path.join(root, q["media"][key]), max_side, as_path)}})
    content.append({"type": "text", "text": q["question"]})
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True, help="dir with questions.jsonl, system_prompt.txt, clips/")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--setting", default="F", choices=["F", "V", "blind"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="", help="e.g. T1,T5")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-side", type=int, default=0, help="downscale frames (e.g. 896 on a 16 GB Mac)")
    ap.add_argument("--image-as", default="base64", choices=["base64", "path"])
    ap.add_argument("--video-as", default="file", choices=["file", "base64"])
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--smoke", action="store_true", help="send 2 questions, print raw replies, write nothing")
    a = ap.parse_args()

    root = a.release
    system = open(os.path.join(root, "system_prompt.txt"), encoding="utf-8").read().strip()
    qs = [json.loads(l) for l in open(os.path.join(root, "questions.jsonl"), encoding="utf-8")]
    if a.tasks:
        keep = set(a.tasks.split(","))
        qs = [q for q in qs if q["task"] in keep]
    if a.setting == "V":
        qs = [q for q in qs if q["tags"].get("setting_only") != "F"]
    if a.setting == "blind":   # same text for all interval specs except the boundary-image ones
        qs = [q for q in qs if not (q.get("range") and q["range"]["type"] == "boundary_frames")]
    if a.limit:
        qs = qs[: a.limit]
    url = a.base_url.rstrip("/") + "/chat/completions"
    head = {"Authorization": f"Bearer {a.api_key}"}

    def call(q):
        body = {"model": a.model, "temperature": a.temperature, "max_tokens": a.max_tokens,
                "messages": build_messages(q, root, system, a.setting, a.max_side, a.image_as == "path", a.video_as)}
        r = requests.post(url, json=body, headers=head, timeout=a.timeout)
        if not r.ok:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        j = r.json()
        return j["choices"][0]["message"].get("content") or "", j.get("usage")

    if a.smoke:
        picks = [qs[0]] + [q for q in qs if q["task"] == "T5"][:1]
        for q in picks:
            t = time.time()
            try:
                out, _ = call(q)
                print(f"[smoke] {q['qid']} {time.time() - t:.1f}s  GT={q['answer']}\n  reply: {out[:300]}")
            except Exception as e:  # noqa: BLE001
                print(f"[smoke] {q['qid']} failed: {e}")
                raise SystemExit(1)
        return

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            r = json.loads(l)
            if not r.get("error"):
                done.add(r["qid"])
    todo = [q for q in qs if q["qid"] not in done]
    print(f"{len(qs)} questions, {len(done)} done, {len(todo)} to run ({a.setting})")
    lock = threading.Lock()

    def run(q):
        rec = {"qid": q["qid"], "model": a.tag or a.model, "setting": a.setting}
        t = time.time()
        try:
            rec["output"], rec["usage"] = call(q)
        except Exception as e:  # noqa: BLE001
            rec.update(output=None, error=f"{type(e).__name__}: {str(e)[:300]}")
        rec["latency_s"] = round(time.time() - t, 2)
        with lock, open(a.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    n_err = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(run, q) for q in todo]), 1):
            r = fut.result()
            n_err += bool(r.get("error"))
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  errors {n_err}" + (f"  last: {r['error']}" if r.get("error") else ""))
    if n_err:
        print(f"{n_err} failed; re-run the same command to retry them")


if __name__ == "__main__":
    main()
