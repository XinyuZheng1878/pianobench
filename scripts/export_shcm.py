"""Export the 上音 (SHCM) self-collected recordings into the same files export_pianovam.py writes,
so the rest of the pipeline (run_fingering.py, build_bench_gold.py, ...) runs unchanged.

  python3 -B scripts/export_shcm.py --root <SHCM_Piano>/20261024 --skeleton <skel_dir> --out <work>/perf \
      --corners "539,319 90,335 90,281 539,265" [--cam CAM_5.avi] [--swap auto|yes|no]

Input layout (one folder per take, as delivered on Baidu Netdisk):
  <root>/PLAYER_0x/<piece>_TIME_<len>_BPM_<bpm>[_...]/CAM_1..5.avi, midi_C2MIDI_Pro.mid, mic_*.wav, <id>.csv
  <skel_dir>/<take_id>.json   MediaPipe skeleton of --cam, from scripts/extract_skeleton.py

Per take writes <take_id>.npz / .skeleton.npz / .meta.json exactly like export_pianovam.py, with
  * notes from the piano's own MIDI (no pedal events in these files, so offset = key release);
  * gt_hand / gt_finger all -1 (no human fingering for this data);
  * label_hand from a simple geometric rule (which palm is nearer the key at the onset, with a clear
    margin), independent of PMFinger, so that p0's "v2 confident AND agrees with the auto label" gate
    has a second opinion to agree with.
--corners are the keyboard corners in video pixels in the order A0-back, C8-back, C8-front, A0-front
(the same order and meaning as PianoVAM's Point_LT, Point_RT, Point_RB, Point_LB: x = 0 at A0's left
edge, y = 1 at the player's edge). Camera positions are fixed per recording day, so one set per day.
take_id = shcm-<date>-<player>-<nn>, nn = position of the folder in name order; the original folder
name is kept in meta.json.
"""
import argparse
import json
import os
import struct
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmfinger.keys import key_x  # noqa: E402
from pmfinger.schema import check_output_path  # noqa: E402

TIP_IDX = [4, 8, 12, 16, 20]
PALM_IDX = [0, 5, 9, 13, 17]
N_WHITE = 52


# ---------------------------------------------------------------- MIDI (no third-party dependency)
def _varlen(b, i):
    v = 0
    while True:
        c = b[i]
        i += 1
        v = (v << 7) | (c & 0x7F)
        if c < 0x80:
            return v, i


def read_midi(path):
    """Return (onset_s, offset_s, pitch, velocity) for every note, sorted by onset then pitch."""
    b = Path(path).read_bytes()
    if b[:4] != b"MThd":
        raise ValueError(f"not a MIDI file: {path}")
    _, ntr, div = struct.unpack(">HHH", b[8:14])
    if div & 0x8000:
        raise ValueError("SMPTE time division not supported")
    i, events = 14, []        # (tick, order, kind, a, b)
    for _ in range(ntr):
        assert b[i:i + 4] == b"MTrk", "bad track chunk"
        ln = struct.unpack(">I", b[i + 4:i + 8])[0]
        j, end, tick, status = i + 8, i + 8 + ln, 0, 0
        while j < end:
            d, j = _varlen(b, j)
            tick += d
            c = b[j]
            if c == 0xFF:
                typ, j = b[j + 1], j + 2
                n, j = _varlen(b, j)
                if typ == 0x51:
                    events.append((tick, 0, "tempo", int.from_bytes(b[j:j + 3], "big"), 0))
                j += n
                continue
            if c in (0xF0, 0xF7):
                n, j = _varlen(b, j + 1)
                j += n
                continue
            if c & 0x80:
                status, j = c, j + 1
            kind = status & 0xF0
            if kind in (0x80, 0x90, 0xA0, 0xB0, 0xE0):
                p, v = b[j], b[j + 1]
                j += 2
                if kind == 0x90 and v > 0:
                    events.append((tick, 2, "on", p, v))
                elif kind == 0x80 or (kind == 0x90 and v == 0):
                    events.append((tick, 1, "off", p, 0))
            else:                                    # 0xC0, 0xD0: one data byte
                j += 1
        i = end
    events.sort(key=lambda e: (e[0], e[1]))
    tempo, last_tick, t = 500000, 0, 0.0
    open_, notes = {}, []
    for tick, _, kind, a, v in events:
        t += (tick - last_tick) * tempo / 1e6 / div
        last_tick = tick
        if kind == "tempo":
            tempo = a
        elif kind == "on":
            if a in open_:                           # re-strike before release: close the old one
                s, vel = open_.pop(a)
                notes.append((s, t, a, vel))
            open_[a] = (t, v)
        elif kind == "off" and a in open_:
            s, vel = open_.pop(a)
            notes.append((s, t, a, vel))
    for a, (s, vel) in open_.items():
        notes.append((s, t, a, vel))
    notes.sort(key=lambda x: (x[0], x[2]))
    arr = np.array(notes, dtype=float).reshape(-1, 4)
    return arr[:, 0], arr[:, 1], arr[:, 2].astype(int), arr[:, 3].astype(int)


# ---------------------------------------------------------------- keyboard geometry
def video_info(path):
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise ValueError(f"cannot open video: {path}")
        return (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                float(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        cap.release()


def take_list(root):
    out = []
    for pdir in sorted(Path(root).glob("PLAYER_*")):
        for n, t in enumerate(sorted(p for p in pdir.iterdir() if p.is_dir()), 1):
            out.append((f"shcm-{Path(root).name}-{pdir.name.replace('PLAYER_', 'P')}-{n:02d}", t))
    return out


def export(tid, tdir, skel_path, corners, cam, swap, out_dir):
    vpath = str(tdir / cam)
    W, H, fps, nb = video_info(vpath)
    Hm = cv2.getPerspectiveTransform(np.float32(corners), np.float32([[0, 0], [N_WHITE, 0], [N_WHITE, 1], [0, 1]]))
    raw = json.loads(Path(skel_path).read_text())
    T = max(int(k) for k in raw) + 1
    skel = np.full((T, 2, 21, 3), np.nan, dtype=np.float32)
    names = {"Left": 0, "Right": 1}
    for k, fr in raw.items():
        for mp_name, pts in fr.items():
            a = np.asarray(pts, dtype=np.float32)[:21]
            skel[int(k), names[mp_name], :, 0] = a[:, 0] * W
            skel[int(k), names[mp_name], :, 1] = a[:, 1] * H
            skel[int(k), names[mp_name], :, 2] = a[:, 2]

    def to_kb(px):
        return cv2.perspectiveTransform(np.nan_to_num(px).reshape(-1, 1, 2).astype(np.float32), Hm).reshape(px.shape)

    # which MediaPipe label is the player's left hand: the one nearer A0 when both are seen
    wx = to_kb(skel[:, :, 0, :2])[..., 0]
    both = np.isfinite(skel[:, :, 0, 0]).all(1)
    frac_label0_lower = float((wx[both, 0] < wx[both, 1]).mean()) if both.any() else float("nan")
    if swap == "auto":
        do_swap = frac_label0_lower < 0.5
    else:
        do_swap = swap == "yes"
    if do_swap:
        skel = skel[:, ::-1].copy()

    valid = np.isfinite(skel[..., 0]).all(-1)
    tips = np.full((T, 2, 5, 3), np.nan)
    tips[..., :2] = to_kb(skel[:, :, TIP_IDX, :2])
    tips[~valid] = np.nan
    wrist = np.full((T, 2, 3), np.nan)
    wrist[..., :2] = to_kb(skel[:, :, 0, :2])
    wrist[~valid] = np.nan
    palm_x = to_kb(skel[:, :, PALM_IDX, :2])[..., 0].mean(-1)
    palm_x[~valid] = np.nan

    on, off, pitch, vel = read_midi(tdir / "midi_C2MIDI_Pro.mid")
    keep = (pitch >= 21) & (pitch <= 108)
    on, off, pitch, vel = on[keep], off[keep], pitch[keep], vel[keep]
    # geometric hand label: nearer palm at the onset frame, only with a clear margin (>= 1.5 white keys)
    lh = np.full(len(pitch), -1)
    for i, (o, p) in enumerate(zip(on, pitch)):
        f = int(round(o * fps))
        if 0 <= f < T and np.isfinite(palm_x[f]).all():
            d = np.abs(palm_x[f] - key_x(int(p)))
            if abs(d[0] - d[1]) >= 1.5:
                lh[i] = int(np.argmin(d))
    # corner check: distance (white keys) from the pressed key to the nearest fingertip at each onset;
    # with correct corners and sync the median is well under one white key
    near = []
    for o, p in zip(on, pitch):
        f = int(round(o * fps))
        if 0 <= f < T:
            tx = tips[f, :, :, 0]
            if np.isfinite(tx).any():
                near.append(float(np.nanmin(np.abs(tx - key_x(int(p))))))
    tip_err = float(np.median(near)) if near else float("nan")
    neg = -np.ones(len(pitch), dtype=int)
    d = dict(video_id=np.array(tid), fps=np.array(fps), tips=tips.astype(np.float32), valid=valid,
             note_onset=on, note_offset=off, note_key_offset=off, note_pitch=pitch, note_velocity=vel,
             label_hand=lh, label_finger=neg, gt_hand=neg, gt_finger=neg, wrist=wrist.astype(np.float32),
             frame_offset=np.array(0), video_path=np.array(os.path.abspath(vpath)))
    np.savez_compressed(os.path.join(out_dir, f"{tid}.npz"), **d)
    np.savez_compressed(os.path.join(out_dir, f"{tid}.skeleton.npz"), skeleton=skel, video_size=np.array([W, H]))
    m = dict(video_id=tid, dataset="SHCM", source_folder=str(tdir), camera=cam, player=tdir.parent.name,
             piece=tdir.name, width=W, height=H, fps=fps, n_frames_video=nb, n_frames_skeleton=T,
             keyboard_corners_px=[list(map(float, c)) for c in corners], mediapipe_label_swapped=bool(do_swap),
             frac_label0_nearer_A0=round(frac_label0_lower, 3), valid_frac=valid.mean(0).round(3).tolist(),
             n_notes=int(len(pitch)), n_label_hand=int((lh >= 0).sum()), n_gt=0,
             median_tip_to_key_whitekeys=round(tip_err, 3))
    with open(os.path.join(out_dir, f"{tid}.meta.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=1)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="one recording day, e.g. .../SHCM_Piano/20261024")
    ap.add_argument("--skeleton", required=True, help="folder of <take_id>.json from extract_skeleton.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--corners", required=True, help='"x,y x,y x,y x,y": A0-back C8-back C8-front A0-front')
    ap.add_argument("--cam", default="CAM_5.avi")
    ap.add_argument("--swap", default="auto", choices=["auto", "yes", "no"])
    ap.add_argument("--list", action="store_true", help="only print take_id -> folder and exit")
    ap.add_argument("--ids-out", default="", help="write the exported take ids here (one per line)")
    a = ap.parse_args()
    takes = take_list(a.root)
    if a.list:
        for tid, t in takes:
            print(tid, t)
        return
    check_output_path(a.out)
    os.makedirs(a.out, exist_ok=True)
    corners = [tuple(float(v) for v in c.split(",")) for c in a.corners.split()]
    assert len(corners) == 4, "need four corners"
    done = []
    for tid, t in takes:
        sk = Path(a.skeleton) / f"{tid}.json"
        if not sk.exists() or not (t / "midi_C2MIDI_Pro.mid").exists() or not (t / a.cam).exists():
            print(tid, "SKIP (missing skeleton, MIDI or video)", t.name)
            continue
        m = export(tid, t, sk, corners, a.cam, a.swap, a.out)
        done.append(tid)
        print(tid, m["piece"], "notes", m["n_notes"], "hand-labelled", m["n_label_hand"], "valid", m["valid_frac"],
              "swap", m["mediapipe_label_swapped"], m["frac_label0_nearer_A0"],
              "tip-to-key", m["median_tip_to_key_whitekeys"])
    if a.ids_out:
        Path(a.ids_out).write_text("\n".join(done) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
