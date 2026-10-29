"""Fingering metrics against labelled notes (PianoVAM human GT), summed over videos."""
from __future__ import annotations

import numpy as np

THRESHOLDS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
NBINS = 1000
COVERAGES = (0.70, 0.80, 0.88, 0.95)


def count_metrics(res, label_hand, label_finger, thresholds=THRESHOLDS) -> dict:
    lab = (label_hand >= 0) & (label_finger >= 0)
    out = lab & (res["status"] == 0)
    hand_ok = res["hand_pred"] == label_hand
    hf_ok = hand_ok & (res["finger_pred"] == label_finger)
    wrong_f = out & hand_ok & ~hf_ok
    df = res["finger_pred"] - label_finger
    c = dict(
        n_lab=int(lab.sum()), n_out=int(out.sum()),
        hand_ok=int((out & hand_ok).sum()), hf_ok=int((out & hf_ok).sum()),
        hf_ok_all=int((lab & hf_ok).sum()),
        err_finger=int(wrong_f.sum()), err_adjacent=int((wrong_f & (np.abs(df) == 1)).sum()),
        err_toward_thumb=int((wrong_f & (df < 0)).sum()), err_toward_pinky=int((wrong_f & (df > 0)).sum()),
    )
    # confidence histograms, so accuracy at a fixed coverage can be compared across confidence definitions
    cand = lab & (res["finger_pred"] >= 0) & np.isin(res["status"], (0, 1))
    b = np.clip((res["conf"] * NBINS).astype(int), 0, NBINS - 1)
    c["hist_all"] = np.bincount(b[cand], minlength=NBINS).tolist()
    c["hist_ok"] = np.bincount(b[cand & hf_ok], minlength=NBINS).tolist()
    for th in thresholds:
        sel = lab & (res["conf"] >= th) & (res["finger_pred"] >= 0) & np.isin(res["status"], (0, 1))
        c[f"cov@{th}"] = int(sel.sum())
        c[f"acc@{th}"] = int((sel & hf_ok).sum())
    return c


def summarize(counts) -> dict:
    tot = {}
    for c in counts:
        for k, v in c.items():
            if isinstance(v, list):
                tot[k] = (np.asarray(tot[k]) + np.asarray(v)) if k in tot else np.asarray(v)
            else:
                tot[k] = tot.get(k, 0) + v
    n = max(tot.get("n_lab", 0), 1)
    no = max(tot.get("n_out", 0), 1)
    s = dict(
        notes=tot.get("n_lab", 0),
        coverage=tot.get("n_out", 0) / n,
        hand_acc=tot.get("hand_ok", 0) / no,
        hand_finger_acc=tot.get("hf_ok", 0) / no,
        hand_finger_acc_no_abstain=tot.get("hf_ok_all", 0) / n,
        finger_err_adjacent_frac=tot.get("err_adjacent", 0) / max(tot.get("err_finger", 0), 1),
        finger_err_thumb_vs_pinky=f'{tot.get("err_toward_thumb", 0)}:{tot.get("err_toward_pinky", 0)}',
    )
    if "hist_all" in tot:
        # take the most confident notes until the target coverage of ALL labelled notes is reached
        ha, ho = tot["hist_all"][::-1].cumsum(), tot["hist_ok"][::-1].cumsum()
        for cov in COVERAGES:
            i = int(np.searchsorted(ha, cov * n))
            s[f"acc@cov{cov:.2f}"] = float(ho[i] / ha[i]) if i < len(ha) else None
    curve = []
    for k in tot:
        if k.startswith("hist_"):
            continue
        if k.startswith("cov@"):
            th = k[4:]
            curve.append((float(th), tot[k] / n, tot[f"acc@{th}"] / max(tot[k], 1)))
    s["curve(th,coverage,acc)"] = sorted(curve)
    return s
