"""Benchmark step 5: the final release as  video + question + answer  (one short clip per question).

  python3 -B scripts/export_video_qa.py --bench <bench_dir> --release <release_dir> --out <video_release_dir>

For every question in <release_dir>/questions.jsonl (the export_bench_gold.py release, optionally with
explanations from make_explanations.py):
  videos/<n>.mp4     the clip the question is about: native frames from the first to the last delivered frame
                     (for prediction questions the clip ends at "now"; nothing after it is shown), same crop and
                     orientation as the delivered frames (keyboard + hands; turned 180 degrees when the camera
                     sees A0 on the right), native frame rate, no audio
  questions.jsonl    the release rows, ready for  run_models.py --release <out> --setting V : media.video added,
                     question text rewritten for video input; plus id, video, duration_s, explanation, source
  system_prompt.txt  conventions for video input
  index.html         browse: video player, question, options, answer, explanation
"""
import argparse
import html
import json
import re
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

SYSTEM_PROMPT = """你会看到一段从钢琴正上方俯拍的演奏视频，画面中是琴键和演奏者的双手。
约定：
1. 左手、右手指演奏者本人的左右手；画面左侧是低音方向。
2. 手指编号：1 拇指，2 食指，3 中指，4 无名指，5 小指。
3. 键名用音名加八度编号，C4 是中央 C；黑键写作升号，如 G#3。
4. 时间以视频开头为 0 秒；“第 a 秒到第 b 秒（不含）之间”包含 a、不包含 b。
5. 一次“按键”指一个键被按下一次；几乎同时按下的键，以其中最早按下的时刻起 40 毫秒内为一组，不链式延长。
6. 预测题里，视频的最后一刻就是“现在”，问的是现在之后会发生什么。
7. 最终答案必须写在花括号里，例如 {B}、{3}、{G#3}、{E4, F#4, G4}；花括号只用于最终答案。"""

PREFIX = re.compile(r"^（本题共[^）]*）")


def video_question(text, duration, predict):
    text = PREFIX.sub("", text)
    text = text.replace("最后一张帧", "视频最后一刻").replace("第一张帧", "视频开头").replace("这些帧", "这段视频")
    head = f"（视频时长 {duration:.2f} 秒，时间从视频开头 0 秒算起" + ("；视频最后一刻就是“现在”" if predict else "") + "。）"
    return head + text


def write_clip(src, f0, f1, crop, flip, path):
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    if abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - f0) > 0.1:
        cap.release()
        raise ValueError("video seek mismatch")
    tmp = path.with_suffix(".tmp.avi")
    vw = None
    n = 0
    try:
        for _ in range(f1 - f0 + 1):
            ok, im = cap.read()
            if not ok:
                break
            if crop:
                x0, y0, x1, y1 = crop
                im = im[y0:y1, x0:x1]
            if flip:
                im = cv2.rotate(im, cv2.ROTATE_180)
            if im.shape[0] % 2 or im.shape[1] % 2:                      # H.264 needs even sizes
                im = im[: im.shape[0] // 2 * 2, : im.shape[1] // 2 * 2]
            if vw is None:
                vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"MJPG"), fps, (im.shape[1], im.shape[0]))
            vw.write(im)
            n += 1
    finally:
        cap.release()
        if vw is not None:
            vw.release()
    if n != f1 - f0 + 1:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"video ended early ({n} of {f1 - f0 + 1} frames)")
    if shutil.which("ffmpeg"):                                         # browser-playable H.264
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-c:v", "libx264", "-preset", "veryfast",
                            "-crf", "20", "-pix_fmt", "yuv420p", "-an", str(path)], capture_output=True, text=True)
        if r.returncode == 0:
            tmp.unlink()
            return fps, n
    # no ffmpeg / no libx264: MPEG-4 Part 2 via OpenCV (plays in most players, not all browsers)
    cap = cv2.VideoCapture(str(tmp))
    vw = None
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if vw is None:
            vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (im.shape[1], im.shape[0]))
        vw.write(im)
    cap.release()
    vw.release()
    tmp.unlink()
    return fps, n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True, help="build_bench_gold.py output (has perf/ and results/)")
    ap.add_argument("--release", required=True, help="export_bench_gold.py output")
    ap.add_argument("--out", required=True, help="new folder")
    a = ap.parse_args()
    bench, rel, out = Path(a.bench), Path(a.release), Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} already exists; choose a new --out")
    (out / "videos").mkdir(parents=True)
    src_rows = {}
    for x in (bench / "questions.jsonl").read_text(encoding="utf-8").splitlines():
        if x.strip():
            r = json.loads(x)
            src_rows[r["qid"]] = r
    meta, vpath = {}, {}
    rows, failed = [], []
    lines = [json.loads(x) for x in (rel / "questions.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    for i, e in enumerate(lines, 1):
        qid = e["qid"]
        try:
            r = src_rows[qid]
            q = r["question"]
            vid = q["video_id"]
            if vid not in meta:
                meta[vid] = json.loads((bench / "perf" / f"{vid}.meta.json").read_text(encoding="utf-8"))
                with np.load(bench / "perf" / f"{vid}.npz", allow_pickle=False) as z:
                    vpath[vid] = str(z["video_path"])
            c = np.asarray(meta[vid].get("keyboard_corners_px") or [[0, 0], [1, 0]], float)
            flip = bool(c[0, 0] > c[1, 0])
            fr = r["media"]["frames"]
            f0, f1 = fr[0]["native_frame"], fr[-1]["native_frame"]
            crop = r["media"].get("crop")
            name = f"{i:05d}.mp4"
            fps, n = write_clip(vpath[vid], f0, f1, crop, flip, out / "videos" / name)
            dur = n / fps
            # the evaluation row (run_models.py --setting V reads media.video and question) plus readable fields
            row = dict(e, id=f"{i:05d}", video=f"videos/{name}", duration_s=round(dur, 3), fps=round(fps, 3),
                       question=video_question(e["question"], dur, bool(e.get("predict"))),
                       explanation=e.get("explanation", ""),
                       source=dict(qid=qid, video_id=vid, piece=meta[vid].get("piece"),
                                   start_frame=f0, end_frame=f1 + 1))
            row["media"] = dict(e.get("media") or {}, video=f"videos/{name}")
            rows.append(row)
        except Exception as ex:  # noqa: BLE001
            failed.append((qid, f"{type(ex).__name__}: {ex}"))
        if i % 50 == 0:
            print(f"  {i}/{len(lines)} clips", flush=True)
    with open(out / "questions.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "system_prompt.txt").write_text(SYSTEM_PROMPT + "\n", encoding="utf-8")
    if (rel / "media").exists() and not (out / "media").exists():          # frame paths stay valid (setting F)
        (out / "media").symlink_to((rel / "media").resolve(), target_is_directory=True)
    cards = []
    for r in rows:
        opts = "".join(f"<li>{'ABCDEFG'[k]}. {html.escape(str(o))}</li>" for k, o in enumerate(r["options"] or []))
        cards.append(f"<article><h3>{r['id']} · {html.escape(r['task'] + ' ' + r['subtype'])}</h3>"
                     f"<video src='{r['video']}' controls preload='none' width='640'></video>"
                     f"<p>{html.escape(r['question'])}</p><ol style='list-style:none'>{opts}</ol>"
                     f"<p><b>答案</b> {html.escape(json.dumps(r['answer'], ensure_ascii=False))}</p>"
                     f"<details><summary>解析</summary><p>{html.escape(r['explanation'])}</p></details>"
                     f"<small>{html.escape(str(r['source']['piece']))} · {r['source']['qid']}</small></article>")
    (out / "index.html").write_text(
        "<meta charset=utf-8><title>视频题</title><style>body{font:14px sans-serif;max-width:900px;margin:auto}"
        "article{border-bottom:1px solid #ccc;padding:10px 0}</style>\n" + "\n".join(cards), encoding="utf-8")
    print(f"done: {len(rows)} questions with video -> {out}/questions.jsonl, {out}/videos/, {out}/index.html")
    for qid, err in failed[:20]:
        print("  FAILED", qid, err)
    if failed:
        print(f"  {len(failed)} failed")


if __name__ == "__main__":
    main()
