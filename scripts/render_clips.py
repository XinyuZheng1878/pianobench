"""Step 7: render clips, official frame sequences and per-clip annotations for a question set.

  python -B scripts/render_clips.py --questions work/bench/questions.jsonl --perf work/perf --fing work/fing \
      --out work/release [--ablation-fps 2 4] [--no-annotations] [--review]

For every clip in questions.jsonl:
  clips/<clip_id>/video.mp4         crop = keyboard box + hands above it, long side <= 1280, no audio
  clips/<clip_id>/frames_8fps/      setting F frames for understanding / localisation (frames_15fps/ for prediction)
                                    frame k = clip time k / fps; file 000k.jpg
  clips/<clip_id>/frames_2fps|4fps/ ablation (with --ablation-fps)
  clips/<clip_id>/handskeleton.npz  (n_frames, 2, 21, 3) at the native frame rate, x/y in clip pixels, hand 0 = left
  clips/<clip_id>/keys.json         every key held in the clip: key, hand, press / release time (s, clip time)
  clips/<clip_id>/pose.json         per official frame: hand centre / octave / span and the 5 finger motion states
  clips/<clip_id>/notes.json        notes with hand, finger (fingering), confidence, label tier
  clips/<clip_id>/clip.mid          MIDI of the clip, time from 0 (note-off = true key release)
Also copies questions.jsonl (with frame paths filled in) and system_prompt.txt next to clips/.
--review also writes review/<fact_id>.jpg contact sheets with key·finger labels drawn at the fingertips.
"""
import argparse
import collections
import json
import os
import shutil
import struct
import subprocess
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmfinger.schema import check_output_path  # noqa: E402
from pmfinger.states import FINGER_ZH, HAND_ZH, STATE_ZH, load_derived, octave_of_x, pitch_name  # noqa: E402

warnings.simplefilter("ignore", RuntimeWarning)
TIP_IDX = [4, 8, 12, 16, 20]


def crop_box(meta, W, H):
    c = np.array(meta["keyboard_corners_px"], float)        # LT, RT, RB, LB
    kh = (c[3, 1] - c[0, 1] + c[2, 1] - c[1, 1]) / 2
    x0 = max(0, min(c[0, 0], c[3, 0]) - 0.02 * W)
    x1 = min(W, max(c[1, 0], c[2, 0]) + 0.02 * W)
    y0 = max(0, min(c[0, 1], c[1, 1]) - 0.35 * kh)
    y1 = min(H, max(c[2, 1], c[3, 1]) + 1.1 * kh)
    x0, y0 = int(x0) // 2 * 2, int(y0) // 2 * 2
    w, h = (int(x1) - x0) // 2 * 2, (int(y1) - y0) // 2 * 2
    s = min(1.0, 1280 / max(w, h))
    ow, oh = int(round(w * s / 2)) * 2, int(round(h * s / 2)) * 2
    return x0, y0, w, h, ow, oh, ow / w


def write_midi(path, notes, tempo_us=500000, tpq=480):
    """notes: list of (pitch, velocity, on_s, off_s). Minimal type-0 SMF at 120 bpm."""
    ev = []
    for p, v, a, b in notes:
        ev.append((a, 1, p, max(1, min(127, int(v)))))
        ev.append((max(b, a + 0.01), 0, p, 0))
    ev.sort(key=lambda e: (e[0], e[1]))
    ticks_per_s = tpq * 1e6 / tempo_us

    def vlq(n):
        out = [n & 0x7F]
        n >>= 7
        while n:
            out.append(0x80 | (n & 0x7F))
            n >>= 7
        return bytes(reversed(out))

    data = bytearray(vlq(0) + b"\xff\x51\x03" + tempo_us.to_bytes(3, "big"))
    last = 0
    for t, on, p, v in ev:
        tk = int(round(t * ticks_per_s))
        data += vlq(tk - last) + bytes([0x90 if on else 0x80, p, v])
        last = tk
    data += vlq(0) + b"\xff\x2f\x00"
    with open(path, "wb") as f:
        f.write(b"MThd" + struct.pack(">IHHH", 6, 0, 1, tpq) + b"MTrk" + struct.pack(">I", len(data)) + data)


def render_video(src, dst, c0, n, fps, box):
    x0, y0, w, h, ow, oh, _ = box
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{c0 / fps:.6f}", "-i", src, "-frames:v", str(n), "-an",
           "-vf", f"crop={w}:{h}:{x0}:{y0},scale={ow}:{oh}", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "20", "-pix_fmt", "yuv420p", "-r", f"{fps:g}", dst]
    subprocess.run(cmd, check=True)


def read_frames(path):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        out.append(im)
    cap.release()
    return out


def sample_offsets(n_native, fps, f_fps, cut=False):
    """Native frame offsets of the official frames: frame k at t = k / f_fps (must be < clip end)."""
    ks = []
    k = 0
    while True:
        off = int(round(k / f_fps * fps))
        if off >= n_native:
            break
        ks.append((k, off))
        k += 1
    return ks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--perf", required=True, help="dir with <video_id>.npz, .skeleton.npz, .meta.json")
    ap.add_argument("--fing", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ablation-fps", nargs="*", type=float, default=[])
    ap.add_argument("--no-annotations", action="store_true", help="test split with hidden answers")
    ap.add_argument("--review", action="store_true")
    ap.add_argument("--quality", type=int, default=90)
    a = ap.parse_args()
    check_output_path(a.out)
    qs = [json.loads(l) for l in open(a.questions, encoding="utf-8")]
    clips = {}
    for q in qs:
        c = clips.setdefault(q["clip_id"], dict(q["clip"], fps_sets=set(), facts=set()))
        c["fps_sets"].add(q["frame_fps"])
        c["facts"].add(q["fact_id"])
    by_video = collections.defaultdict(list)
    for cid, c in clips.items():
        by_video[c["video_id"]].append(cid)

    root = Path(a.out)
    (root / "clips").mkdir(parents=True, exist_ok=True)
    frame_paths = {}
    for vid, cids in by_video.items():
        perf_p = os.path.join(a.perf, f"{vid}.npz")
        der = load_derived(perf_p, os.path.join(a.fing, f"{vid}.fing.npz"))
        meta = json.load(open(os.path.join(a.perf, f"{vid}.meta.json"), encoding="utf-8"))
        skel = np.load(os.path.join(a.perf, f"{vid}.skeleton.npz"))["skeleton"]
        z = np.load(perf_p)
        vel = z["note_velocity"]
        src = str(z["video_path"])
        W, H = meta["width"], meta["height"]
        box = crop_box(meta, W, H)
        x0, y0, _, _, ow, oh, s = box
        fps = der.fps
        for cid in sorted(cids):
            c = clips[cid]
            c0, c1 = c["start_frame"], c["end_frame"]
            n = c1 - c0
            d = root / "clips" / cid
            d.mkdir(parents=True, exist_ok=True)
            vpath = d / "video.mp4"
            if not vpath.exists():
                render_video(src, str(vpath), c0 + der.perf.frame_offset, n, fps, box)
            frames = read_frames(str(vpath))
            if len(frames) < n:
                print(f"WARNING {cid}: decoded {len(frames)} < {n} frames")
            sets = sorted(c["fps_sets"] | set(a.ablation_fps))
            for f_fps in sets:
                sub = d / f"frames_{f_fps:g}fps"
                sub.mkdir(exist_ok=True)
                lst = []
                for k, off in sample_offsets(min(n, len(frames)), fps, f_fps):
                    p = sub / f"{k:04d}.jpg"
                    if not p.exists():
                        cv2.imwrite(str(p), frames[off], [cv2.IMWRITE_JPEG_QUALITY, a.quality])
                    lst.append(str(p.relative_to(root)))
                frame_paths[cid, f_fps] = lst
            if a.no_annotations:
                continue
            # ---- annotations
            sk = skel[c0:c1].copy()
            sk[..., 0] = (sk[..., 0] - x0) * s
            sk[..., 1] = (sk[..., 1] - y0) * s
            np.savez_compressed(d / "handskeleton.npz", skeleton=sk.astype(np.float32), fps=fps,
                                note="(frames, hand 0=left/1=right, 21 MediaPipe joints, x px / y px / z rel)")
            m = (der.on < c1) & (der.rel > c0)
            notes = []
            for i in np.flatnonzero(m):
                notes.append(dict(key=pitch_name(der.pitch[i]), midi=int(der.pitch[i]),
                                  press=round((der.on[i] - c0) / fps, 3), release=round((der.rel[i] - c0) / fps, 3),
                                  velocity=int(vel[i]), hand=HAND_ZH[int(der.hand[i])] if der.hand_trust[i] else None,
                                  finger=int(der.finger[i]) + 1 if der.finger_trust[i] else None,
                                  conf=round(float(der.note_conf[i]), 3), tier=str(der.label_tier[i])))
            json.dump(notes, open(d / "notes.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
            json.dump([dict(key=x["key"], hand=x["hand"], press=x["press"], release=x["release"]) for x in notes],
                      open(d / "keys.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
            pose = []
            f_fps = min(c["fps_sets"])
            for k, off in sample_offsets(n, fps, f_fps):
                t = c0 + off
                hands = {}
                for h in (0, 1):
                    cx = der.center[t, h]
                    hands[HAND_ZH[h]] = None if not np.isfinite(cx) else dict(
                        center_white=round(float(cx), 2), octave=octave_of_x(cx), span_white=round(float(der.span[t, h]), 2),
                        finger_states={f"{i + 1}": STATE_ZH[int(der.state[t, h, i])] for i in range(5)})
                pose.append(dict(frame=k, t=round(k / f_fps, 3), hands=hands))
            json.dump(dict(fps=f_fps, frames=pose), open(d / "pose.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
            write_midi(d / "clip.mid", [(x["midi"], x["velocity"], max(0.0, x["press"]), x["release"]) for x in notes])
            if a.review:
                review_sheet(root, cid, c, frames, der, c0, fps, skel, x0, y0, s, qs, src, box)
        print(vid, "clips", len(cids))

    # fill frame paths into the questions, copy prompt
    out_q = root / "questions.jsonl"
    with open(out_q, "w", encoding="utf-8") as fo:
        for q in qs:
            fl = frame_paths.get((q["clip_id"], q["frame_fps"]), [])
            q["media"] = dict(video=f"clips/{q['clip_id']}/video.mp4", frames=fl)
            if q["range"] and q["range"]["type"] == "boundary_frames":
                f8 = frame_paths.get((q["clip_id"], 8), [])
                ia, ib = q["range"]["image_a_frame"], q["range"]["image_b_frame"]
                q["media"]["image_a"] = f8[min(ia, len(f8) - 1)]
                q["media"]["image_b"] = f8[min(ib, len(f8) - 1)]
            if q["task"] == "T5":   # the last official frame must be before the cut
                last_t = (len(fl) - 1) / q["frame_fps"]
                assert last_t < q["predict"]["cut"], (q["qid"], last_t)
            fo.write(json.dumps(q, ensure_ascii=False) + "\n")
    src_prompt = Path(a.questions).parent / "system_prompt.txt"
    if src_prompt.exists():
        shutil.copy(src_prompt, root / "system_prompt.txt")
    print("clips", len(clips), "->", root / "clips", "; questions ->", out_q)


def grab_frame(src, f, fps, box):
    """One cropped frame from the source video (used to show reviewers what happens after a cut)."""
    x0, y0, w, h, ow, oh, _ = box
    out = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{f / fps:.6f}", "-i", src, "-frames:v", "1",
                          "-vf", f"crop={w}:{h}:{x0}:{y0},scale={ow}:{oh}", "-f", "image2pipe", "-vcodec", "png", "-"],
                         capture_output=True, check=True).stdout
    return cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)


def review_sheet(root, cid, c, frames, der, c0, fps, skel, x0, y0, s, qs, src=None, box=None):
    """Contact sheet per fact with 'key·finger' drawn at the assigned fingertip of every held key."""
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 20)
        big = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 26)
    except OSError:
        font = big = ImageFont.load_default()
    rv = root / "review"
    rv.mkdir(exist_ok=True)
    f_fps = min(c["fps_sets"])
    focus = sorted({ff - c0 for q in qs if q["clip_id"] == cid for ff in q["review"].get("focus_frames", [])
                    if ff - c0 >= 0})
    if src is not None:   # prediction facts: the presses are after the cut, outside the clip
        frames = list(frames)
        for off in focus:
            if off >= len(frames):
                while len(frames) < off:
                    frames.append(None)
                frames.append(grab_frame(src, c0 + off + der.perf.frame_offset, fps, box))
    focus = [off for off in focus if off < len(frames) and frames[off] is not None]
    if focus:   # frames just after the presses the fact is about
        offs = [(off / fps * f_fps, off) for off in focus][:8]
    else:
        offs = sample_offsets(len(frames), fps, f_fps)
        step = max(1, len(offs) // 8)
        offs = offs[::step][:8]
    tiles = []
    for k, off in offs:
        t = c0 + off
        im = Image.fromarray(cv2.cvtColor(frames[off], cv2.COLOR_BGR2RGB))
        dr = ImageDraw.Draw(im)
        for i in der.active_keys(t):
            h, f = int(der.hand[i]), int(der.finger[i])
            p = skel[t, h, TIP_IDX[f], :2]
            if not np.isfinite(p).all():
                continue
            px, py = (p[0] - x0) * s, (p[1] - y0) * s
            col = (255, 60, 60) if h == 1 else (60, 160, 255)
            if not der.finger_trust[i]:
                col = (160, 160, 160)
            dr.ellipse([px - 5, py - 5, px + 5, py + 5], outline=col, width=2)
            dr.text((px - 18, py - 30), f"{pitch_name(der.pitch[i])}·{f + 1}", fill=col, font=font)
        dr.text((6, 4), f"{off / fps:.2f}s" + ("（切点之后）" if off >= c["end_frame"] - c0 else ""), fill=(255, 255, 0), font=font)
        tiles.append(im)
    if not tiles:
        return
    cols = 2
    tw, th = tiles[0].size
    for fid in sorted(c["facts"]):
        fq = [q for q in qs if q["fact_id"] == fid]
        if not fq or not fq[0]["review"]["needs_review"]:
            continue
        q = fq[0]
        rows = (len(tiles) + cols - 1) // cols
        sheet = Image.new("RGB", (tw * cols, th * rows + 110), (20, 20, 20))
        for i, tl in enumerate(tiles):
            sheet.paste(tl, ((i % cols) * tw, (i // cols) * th + 110))
        dr = ImageDraw.Draw(sheet)
        txt = q["question"].split("\n")[0][:90]
        ans = q["answer"] if q["answer_type"] != "choice" else f"{q['answer']}. {q['options'][ord(q['answer']) - 65]}"
        dr.text((8, 6), f"{fid} · {q['task']} {q['subtype']} · {cid}", fill=(200, 200, 200), font=font)
        dr.text((8, 34), txt, fill=(255, 255, 255), font=big)
        dr.text((8, 70), f"答案：{ans}    （红 = 右手，蓝 = 左手，灰 = 指法不可信；标注为 键名·手指）", fill=(120, 255, 120), font=font)
        sheet.save(rv / f"{fid}.jpg", quality=85)


if __name__ == "__main__":
    main()
