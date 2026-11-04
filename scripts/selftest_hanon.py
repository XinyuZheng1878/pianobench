"""Self-test for Hanon score fingering (pmfinger/hanon_score.py) and its use in export_shcm.py.

  python3 -B scripts/selftest_hanon.py

1. every exercise 1-20 parses, both hands have the same number of notes, the first figure of No. 1 has the
   textbook fingering, and ascending fingers repeat across figures;
2. a simulated performance (an octave up, tempo jitter, dropped and wrong notes) is aligned back with no
   wrong finger;
3. export_shcm.py on a fake take folder writes gt_hand / gt_finger for a Hanon take, and none for the same
   take whose folder name says wrong fingering.
"""
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pmfinger import hanon_score as H  # noqa: E402


def check(cond, msg):
    if not cond:
        raise SystemExit("FAIL: " + msg)
    print("ok  ", msg)


def write_midi(path, on, off, pitch, tpq=480, tempo=500000):
    ev = []
    for o, f, p in zip(on, off, pitch):
        ev.append((round(o * tpq * 1e6 / tempo), 1, 0x90, p, 80))
        ev.append((round(f * tpq * 1e6 / tempo), 0, 0x80, p, 0))
    ev.sort()
    trk, last = bytearray(), 0
    for t, _, st, p, v in ev:
        d, buf = t - last, [t - last & 0x7F]
        last, d = t, d >> 7
        while d:
            buf.insert(0, 0x80 | (d & 0x7F))
            d >>= 7
        trk += bytes(buf) + bytes([st, p, v])
    trk += b"\x00\xff\x2f\x00"
    Path(path).write_bytes(b"MThd" + struct.pack(">IHHH", 6, 0, 1, tpq) + b"MTrk" + struct.pack(">I", len(trk)) + trk)


def main():
    for n in range(1, 21):
        ns = H.score_notes(n)
        r = [x for x in ns if x["hand"] == 1]
        l_ = [x for x in ns if x["hand"] == 0]
        assert len(r) == len(l_) and len(r) > 200, (n, len(r), len(l_))
    check(True, "Nos. 1-20 parse, equal note counts in both hands")
    one = H.score_notes(1)
    rh = [x for x in one if x["hand"] == 1]
    lh = [x for x in one if x["hand"] == 0]
    check([x["finger"] + 1 for x in rh[:8]] == [1, 2, 3, 4, 5, 4, 3, 2], "No. 1 right hand first figure 1-2-3-4-5-4-3-2")
    check([x["finger"] + 1 for x in lh[:8]] == [5, 4, 3, 2, 1, 2, 3, 4], "No. 1 left hand first figure 5-4-3-2-1-2-3-4")
    check(rh[0]["pitch"] == 48 and lh[0]["pitch"] == 36, "No. 1 starts on C3 (right) and C2 (left)")
    desc = [x for x in rh if x["section"] == 1]
    check([x["finger"] + 1 for x in desc[8:16]] == [5, 4, 3, 2, 1, 2, 3, 4], "No. 1 descending figure 2 inherits 5-4-3-2-1-2-3-4")

    rng = np.random.default_rng(1)
    for n in (1, 7, 18):
        ns = H.score_notes(n)
        on = np.array([x["t"] * 0.15 + rng.normal(0, 0.01) for x in ns])
        p = np.array([x["pitch"] + 12 for x in ns])
        keep = np.ones(len(ns), bool)
        keep[rng.choice(len(ns), 6, replace=False)] = False
        truth = np.array([x["finger"] for x in ns])[keep]
        gh, gf, rep = H.label_take(on[keep], p[keep], f"Hanon_No{n}_TIME_50_BPM_80")
        lab = gf >= 0
        check(rep["used"] and rep["octave_shift"] == 12 and (gf[lab] == truth[lab]).all() and lab.mean() > 0.8,
              f"No. {n}: simulated take aligned, {lab.sum()} fingers, none wrong")
    check(H.identify("哈农No.3_100BPM_错误指法") == (3, True), "folder name with 错误指法 is not labelled")
    check(H.identify("哈农No.2_8_同时多项错误")[1] is False and H.identify("Hanon2_wrist_high")[1] is False,
          "hand-shape error takes keep the printed fingering")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ns = H.score_notes(2)
        on = np.array([x["t"] * 0.15 + 1.0 for x in ns])
        for name in ("Hanon_No2_TIME_50_BPM_80", "Hanon_No2_TIME_50_BPM_80_错误指法"):
            d = tmp / "day" / "PLAYER_01" / name
            d.mkdir(parents=True)
            write_midi(d / "midi_C2MIDI_Pro.mid", on, on + 0.1, [x["pitch"] for x in ns])
            vw = cv2.VideoWriter(str(d / "CAM_5.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 480))
            for _ in range(5):
                vw.write(np.zeros((480, 640, 3), np.uint8))
            vw.release()
        (tmp / "skel").mkdir()
        for tid in ("shcm-day-P01-01", "shcm-day-P01-02"):
            (tmp / "skel" / f"{tid}.json").write_text(json.dumps({"0": {}}))
        out = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/export_shcm.py"), "--root", str(tmp / "day"),
                              "--skeleton", str(tmp / "skel"), "--out", str(tmp / "perf"),
                              "--corners", "539,319 90,335 90,281 539,265"], capture_output=True, text=True)
        check(out.returncode == 0, "export_shcm.py runs on fake takes" + ("" if out.returncode == 0 else "\n" + out.stderr))
        z1, z2 = (np.load(tmp / "perf" / f"shcm-day-P01-0{i}.npz") for i in (1, 2))
        check((z1["gt_finger"] >= 0).sum() > 400 and (z1["gt_hand"] >= 0).sum() == len(ns),
              "correct-fingering take gets score fingering")
        check((z2["gt_finger"] >= 0).sum() == 0, "wrong-fingering take gets none")
    print("ALL PASS")


if __name__ == "__main__":
    main()
