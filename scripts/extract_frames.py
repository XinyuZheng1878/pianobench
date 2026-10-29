"""Dump the frames referenced by a VQA jsonl to jpg and write a jsonl with image paths.

  python scripts/extract_frames.py --jsonl vqa.jsonl --video-root /path/to/videos \
      --out-dir ~/work/vqa/frames --out-jsonl ~/work/vqa/vqa_with_images.jsonl

`images[i].video` is the npz's video_path (or video_id); relative names are joined with --video-root
and, if there is no extension, --ext is appended.
"""
import argparse
import collections
import json
import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmfinger.schema import check_output_path  # noqa: E402


def resolve(video, root, ext):
    p = video if os.path.isabs(video) or not root else os.path.join(root, video)
    return p if os.path.splitext(p)[1] else p + ext


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--video-root", default="")
    ap.add_argument("--ext", default=".mp4")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--quality", type=int, default=92)
    args = ap.parse_args()
    check_output_path(args.out_dir)
    check_output_path(args.out_jsonl)
    recs = [json.loads(l) for l in open(args.jsonl, encoding="utf-8")]
    need = collections.defaultdict(set)
    for r in recs:
        for im in r["images"]:
            need[im["video"]].add(im["frame"])

    paths, missing = {}, []
    for video, frames in need.items():
        src = resolve(video, args.video_root, args.ext)
        stem = Path(src).stem
        os.makedirs(os.path.join(args.out_dir, stem), exist_ok=True)
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            missing.append(src)
            continue
        cur = 0
        for f in sorted(frames):          # sequential read: frame-accurate, unlike seeking
            while cur < f and cap.grab():
                cur += 1
            if cur < f:
                break
            ok, img = cap.read()
            cur += 1
            if not ok:
                break
            dst = os.path.join(args.out_dir, stem, f"{f:07d}.jpg")
            cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            paths[video, f] = dst
        cap.release()

    kept = 0
    with open(args.out_jsonl, "w", encoding="utf-8") as fo:
        for r in recs:
            if all((im["video"], im["frame"]) in paths for im in r["images"]):
                for im in r["images"]:
                    im["path"] = paths[im["video"], im["frame"]]
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
                kept += 1
    print(f"kept {kept}/{len(recs)} samples; unreadable videos: {len(missing)}", missing[:5])


if __name__ == "__main__":
    main()
