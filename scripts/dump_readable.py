"""Write the per-take npz data as plain text files anyone can open (Excel, a text editor, pandas).

  python3 -B scripts/dump_readable.py --work ../shcm_work/20261024 [--out ../shcm_work/20261024/readable]

For every take <id> in <work>/perf (with <work>/fing/<id>.fing.npz when present) writes <out>/<id>/:
  notes.tsv      one row per MIDI note, in the same style as PianoVAM's Fingering_GT:
                   onset  key_offset  note  note_name  velocity  hand  finger  finger_source
                   auto_hand  auto_finger  auto_conf  auto_status
                 hand L/R, finger 1-5 (1 = thumb), empty = unknown; hand/finger = gold (score or human),
                 auto_* = PMFinger's guess from the skeleton (not gold)
  skeleton.csv   one row per frame x hand x joint: frame, time_s, hand (L/R), joint (0-20, MediaPipe order:
                 0 wrist, 4 thumb tip, 8 index tip, 12 middle tip, 16 ring tip, 20 little tip), x_px, y_px, z;
                 frames where a hand was not detected are left out
  fingertips.csv one row per frame x hand: frame, time_s, hand, then thumb_x ... little_x = fingertip position
                 along the keyboard in white-key widths (0 = left edge of A0; C4 = 23.5)
  meta.json      copy of the take's meta (source folder, camera, keyboard corners, score fingering report)
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FINGERS = ["thumb", "index", "middle", "ring", "little"]
STATUS = ["ok", "low_conf", "no_view", "suspect", "overfull"]


def name(p):
    return f"{NAMES[p % 12]}{p // 12 - 1}"


def hl(h):
    return {0: "L", 1: "R"}.get(int(h), "")


def fn(f):
    return str(int(f) + 1) if 0 <= int(f) <= 4 else ""


def dump(perf, fing, out):
    out.mkdir(parents=True, exist_ok=True)
    tid = perf.name[:-4]
    with np.load(perf, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    r = {}
    if fing.exists():
        with np.load(fing, allow_pickle=False) as z:
            r = {k: z[k] for k in z.files}
    meta_p = perf.with_suffix(".meta.json")
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    src = "score" if (meta.get("score_fingering") or {}).get("used") else ("human" if meta.get("dataset") != "SHCM" else "")
    n = len(d["note_pitch"])
    same = r and len(r.get("pitch", [])) == n and np.array_equal(r["pitch"], d["note_pitch"])
    with open(out / "notes.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["onset", "key_offset", "note", "note_name", "velocity", "hand", "finger", "finger_source",
                    "auto_hand", "auto_finger", "auto_conf", "auto_status"])
        for i in range(n):
            gh, gf = int(d["gt_hand"][i]), int(d["gt_finger"][i])
            row = [f"{d['note_onset'][i]:.6f}", f"{d['note_key_offset'][i]:.6f}", int(d["note_pitch"][i]),
                   name(int(d["note_pitch"][i])), int(d["note_velocity"][i]) if "note_velocity" in d else "",
                   hl(gh), fn(gf), src if gf >= 0 else ""]
            if same:
                st = int(r["status"][i]) if "status" in r else -1
                row += [hl(r["hand_pred"][i]), fn(r["finger_pred"][i]), f"{float(r['conf'][i]):.3f}",
                        STATUS[st] if 0 <= st < len(STATUS) else ""]
            else:
                row += ["", "", "", ""]
            w.writerow(row)
    fps = float(d["fps"])
    sk_p = perf.with_name(tid + ".skeleton.npz")
    if sk_p.exists():
        with np.load(sk_p, allow_pickle=False) as z:
            sk = z["skeleton"]
        with open(out / "skeleton.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "time_s", "hand", "joint", "x_px", "y_px", "z"])
            for t in range(sk.shape[0]):
                for h in (0, 1):
                    if not np.isfinite(sk[t, h, :, 0]).all():
                        continue
                    for j in range(21):
                        x, y, zz = sk[t, h, j]
                        w.writerow([t, f"{t / fps:.4f}", hl(h), j, f"{x:.1f}", f"{y:.1f}", f"{zz:.4f}"])
    tips = d["tips"]
    with open(out / "fingertips.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "hand"] + [f"{x}_x" for x in FINGERS])
        for t in range(tips.shape[0]):
            for h in (0, 1):
                xs = tips[t, h, :, 0]
                if np.isfinite(xs).all():
                    w.writerow([t, f"{t / fps:.4f}", hl(h)] + [f"{x:.2f}" for x in xs])
    if meta_p.exists():
        shutil.copy(meta_p, out / "meta.json")
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, help="folder with perf/ and fing/")
    ap.add_argument("--out", default="", help="default <work>/readable")
    ap.add_argument("--only", default="", help="comma-separated take ids (default: all)")
    a = ap.parse_args()
    work = Path(a.work)
    out = Path(a.out) if a.out else work / "readable"
    only = set(filter(None, a.only.split(",")))
    perfs = sorted(p for p in (work / "perf").glob("*.npz") if not p.name.endswith(".skeleton.npz"))
    for p in perfs:
        tid = p.name[:-4]
        if only and tid not in only:
            continue
        n = dump(p, work / "fing" / f"{tid}.fing.npz", out / tid)
        print(f"{tid}: {n} notes -> {out / tid}")


if __name__ == "__main__":
    main()
