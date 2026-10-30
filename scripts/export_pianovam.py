"""Step 1 (local stand-in for the server exporter): PianoVAM recording -> performance.npz + skeleton.npz.

  python -B scripts/export_pianovam.py --root data/pianovam --out work/perf [--recs 2024-09-02_14-10-41 ...]

Per recording writes
  <rec>.npz           performance.npz (pmfinger/schema.py) plus PianoVAM-only extras:
                        note_key_offset  true key release from the Disklavier (s); note_offset is the
                                         pedal-extended offset, so PMFinger behaves as on PM10M
                        gt_hand, gt_finger  manual Fingering_GT labels (-1 = none), aligned with notes
  <rec>.skeleton.npz  full MediaPipe 21-point skeleton in video pixels, (T, 2, 21, 3), hand 0 = left
  <rec>.meta.json     dataset, split, player, skill, piece, video size, keyboard corners

Geometry: fingertip pixels are mapped through the homography of the four keyboard corners in
metadata.json onto x in [0, 52] white keys (A0 left edge = 0) and y in [0, 1] (front edge = 1).
MediaPipe's "Left"/"Right" are swapped for this top view (verified: hit rate 0.97-0.99 swapped vs
<0.1 unswapped), so "Left" is stored as the player's right hand.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmfinger.schema import check_output_path  # noqa: E402

TIP_IDX = [4, 8, 12, 16, 20]
N_WHITE = 52


def read_tsv(path):
    """Read the dataset TSV with the standard library; no pandas runtime required."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        names = f.readline().lstrip("#").strip().split("\t")
        rows = list(csv.DictReader(f, fieldnames=names, delimiter="\t"))
    df = {k: np.array([row[k] for row in rows], dtype=str) for k in names}
    for c in ("onset", "key_offset", "frame_offset"):
        df[c] = df[c].astype(float)
    for c in ("note", "velocity"):
        df[c] = df[c].astype(float).astype(int)
    return df


def parse_pt(s):
    a, b = str(s).replace("(", "").replace(")", "").split(",")[:2]
    return float(a), float(b)


def video_info(path):
    if shutil.which("ffprobe") is None:
        cap = cv2.VideoCapture(path)
        try:
            if not cap.isOpened():
                raise ValueError(f"cannot open video: {path}")
            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if W <= 0 or H <= 0 or not np.isfinite(fps) or fps <= 0 or count <= 0:
                raise ValueError(f"invalid video metadata: {path}")
            return W, H, fps, count
        finally:
            cap.release()
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,nb_frames", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return int(s["width"]), int(s["height"]), float(num) / float(den), int(s.get("nb_frames", 0) or 0)


def labels(df):
    hand = np.array([{"L": 0, "R": 1}.get(str(x).strip(), -1) for x in df["hand"]])

    def parse_finger(x):
        try:
            value = float(str(x).strip())
            return int(value) - 1 if np.isfinite(value) and value.is_integer() else -1
        except ValueError:
            return -1

    fin = np.array([parse_finger(x) for x in df["finger"]])
    bad_finger = (hand < 0) | (fin < 0) | (fin > 4)
    return hand, np.where(bad_finger, -1, fin)


def export(root, rec, meta, out_dir):
    W, H, fps, nb = video_info(os.path.join(root, "Video", f"{rec}.mp4"))
    corners = np.float32([parse_pt(meta[k]) for k in ("Point_LT", "Point_RT", "Point_RB", "Point_LB")])
    dst = np.float32([[0, 0], [N_WHITE, 0], [N_WHITE, 1], [0, 1]])
    Hm = cv2.getPerspectiveTransform(corners, dst)

    with open(os.path.join(root, "Handskeleton", f"{rec}.json"), encoding="utf-8") as f:
        raw = json.load(f)
    T = max(int(k) for k in raw) + 1
    skel = np.full((T, 2, 21, 3), np.nan, dtype=np.float32)
    for k, fr in raw.items():
        if not isinstance(fr, dict):
            continue
        for mp_name, h in (("Left", 1), ("Right", 0)):   # swapped: MediaPipe "Left" = player's right
            pts = fr.get(mp_name)
            if pts:
                a = np.asarray(pts, dtype=np.float32)[:21]
                skel[int(k), h, :, 0] = a[:, 0] * W
                skel[int(k), h, :, 1] = a[:, 1] * H
                skel[int(k), h, :, 2] = a[:, 2]
    valid = np.isfinite(skel[..., 0]).all(-1)
    tip_px = skel[:, :, TIP_IDX, :2].reshape(-1, 1, 2)
    uv = cv2.perspectiveTransform(np.nan_to_num(tip_px).astype(np.float32), Hm).reshape(T, 2, 5, 2)
    tips = np.full((T, 2, 5, 3), np.nan)
    tips[..., :2] = uv
    tips[~valid] = np.nan
    wrist = np.full((T, 2, 3), np.nan)
    wuv = cv2.perspectiveTransform(np.nan_to_num(skel[:, :, 0, :2]).reshape(-1, 1, 2).astype(np.float32), Hm)
    wrist[..., :2] = wuv.reshape(T, 2, 2)
    wrist[~valid] = np.nan

    df = read_tsv(os.path.join(root, "Fingering", f"{rec}.tsv"))
    gt_path = os.path.join(root, "Fingering_GT", f"{rec}.tsv")
    lh, lf = labels(df)
    gh = np.full(len(df["onset"]), -1)
    gf = np.full(len(df["onset"]), -1)
    if os.path.exists(gt_path):
        g = read_tsv(gt_path)
        gl = labels(g)
        key = {(round(o, 4), int(p)): i for i, (o, p) in enumerate(zip(g["onset"], g["note"]))}
        for i, (o, p) in enumerate(zip(df["onset"], df["note"])):
            j = key.get((round(o, 4), int(p)))
            if j is not None:
                gh[i], gf[i] = gl[0][j], gl[1][j]
    pitch = df["note"]
    onset = df["onset"]
    keep = (pitch >= 21) & (pitch <= 108)
    order = np.flatnonzero(keep)[np.lexsort((pitch[keep], onset[keep]))]   # schema's own sort order

    d = dict(video_id=np.array(rec), fps=np.array(fps), tips=tips.astype(np.float32), valid=valid,
             note_onset=onset[order], note_offset=df["frame_offset"][order],
             note_key_offset=df["key_offset"][order], note_pitch=pitch[order],
             note_velocity=df["velocity"][order], label_hand=lh[order], label_finger=lf[order],
             gt_hand=gh[order], gt_finger=gf[order], wrist=wrist.astype(np.float32),
             frame_offset=np.array(0), video_path=np.array(os.path.abspath(os.path.join(root, "Video", f"{rec}.mp4"))))
    np.savez_compressed(os.path.join(out_dir, f"{rec}.npz"), **d)
    np.savez_compressed(os.path.join(out_dir, f"{rec}.skeleton.npz"), skeleton=skel, video_size=np.array([W, H]))
    m = dict(video_id=rec, dataset="PianoVAM", split=meta.get("split"), player=meta.get("P1_name"),
             skill=meta.get("P1_skill"), composer=meta.get("composer"), piece=meta.get("piece"),
             width=W, height=H, fps=fps, n_frames_video=nb, n_frames_skeleton=T,
             keyboard_corners_px=corners.tolist(), valid_frac=valid.mean(0).round(3).tolist(),
             n_notes=int(len(order)), n_gt=int((gh[order] >= 0).sum()))
    with open(os.path.join(out_dir, f"{rec}.meta.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=1)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--recs", nargs="*", default=None)
    a = ap.parse_args()
    check_output_path(a.out)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.root, "metadata.json"), encoding="utf-8") as f:
        raw = json.load(f)
    meta = {e["record_time"]: e for e in (raw.values() if isinstance(raw, dict) else raw)}
    recs = a.recs or sorted(p.stem for p in Path(a.root, "Handskeleton").glob("*.json")
                            if Path(a.root, "Video", p.stem + ".mp4").exists())
    for rec in recs:
        m = export(a.root, rec, meta[rec], a.out)
        print(rec, m["split"], m["skill"], "notes", m["n_notes"], "gt", m["n_gt"], "valid", m["valid_frac"])


if __name__ == "__main__":
    main()
