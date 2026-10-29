"""End-to-end fingering for one video: clean -> align -> evidence -> hands -> HMM -> release -> refine."""
from __future__ import annotations

import numpy as np

from .decode import assign_hands, decode_all
from .evidence import estimate_av_offset, finger_evidence, onset_frames
from .keys import IS_BLACK
from .preprocess import clean_tips
from .release import make_busy, track_release
from .schema import check_output_path

STATUS = ["ok", "low_conf", "no_view", "suspect", "overfull"]
OK, LOW_CONF, NO_VIEW, SUSPECT, OVERFULL = range(5)


def update_bias(resid, hand, finger, conf, prior_bias, cfg):
    """Per-video fingertip offset (tip_x - key_x) per (hand, finger), shrunk toward the prior bias."""
    new = prior_bias.copy()
    for h in (0, 1):
        for f in range(5):
            m = (hand == h) & (finger == f) & (conf >= cfg.bias_min_conf)
            r = resid[m, h, f]
            r = r[np.abs(r - prior_bias[h, f]) < 1.0]
            if len(r):
                n0 = cfg.bias_shrink_n
                new[h, f] = (len(r) * np.median(r) + n0 * prior_bias[h, f]) / (len(r) + n0)
    return new


def _note_context(onset_s, hand, tol):
    """Chord size (all hands) and log inter-onset interval to the previous note of the same hand."""
    order = np.argsort(onset_s, kind="stable")
    on = onset_s[order]
    chord = np.empty(len(on), dtype=float)
    chord[order] = np.searchsorted(on, on + tol, "right") - np.searchsorted(on, on - tol, "left")
    ioi = np.full(len(on), 5.0)
    for h in (0, 1):
        idx = np.flatnonzero(hand == h)
        if len(idx) > 1:
            ioi[idx[1:]] = np.maximum(np.diff(onset_s[idx]), 1e-3)
    return chord, np.log(ioi)


def run_fingering(perf, prior, cfg, calib=None) -> dict:
    fps = perf.fps
    tips, ok = clean_tips(perf.tips, perf.valid, cfg)
    av = cfg.av_offset_frames
    if av is None:
        av = estimate_av_offset(tips, ok, perf.pitch, perf.onset, fps, cfg.av_search)
    on_f = onset_frames(perf.onset, fps, av)
    off_f = onset_frames(perf.offset, fps, av)
    tol_f = max(1, int(round(cfg.chord_tol_s * fps)))

    bias = prior.bias.copy()
    for it in range(cfg.bias_em_iters + 1):
        logE, s, avail, resid, wfrac = finger_evidence(tips, ok, perf.pitch, on_f, fps, bias, cfg, return_wfrac=True)
        hand, hand_conf = assign_hands(logE, avail)
        busy = None
        for _ in range(max(1, cfg.occupancy_passes)):
            finger, finger_conf, overfull = decode_all(logE, hand, perf.onset, on_f, perf.pitch, prior, cfg, busy)
            rel, unc = track_release(tips, ok, on_f, off_f, hand, finger, cfg)
            busy = make_busy(hand, finger, on_f, rel, tol_f)
        if it < cfg.bias_em_iters:
            bias = update_bias(resid, hand, finger, hand_conf * finger_conf, prior.bias, cfg)

    N = perf.N
    hp, fp = np.clip(hand, 0, 1), np.clip(finger, 0, 4)
    s_hand = s[np.arange(N), hp]                                   # (N, 5) evidence of the chosen hand
    srt = np.sort(s_hand, 1)
    chord, log_ioi = _note_context(perf.onset, hand, cfg.chord_tol_s)
    feats = dict(finger_conf=finger_conf, hand_conf=hand_conf, ev_sel=s_hand[np.arange(N), fp],
                 ev_margin=srt[:, -1] - srt[:, -2], chord=chord, log_ioi=log_ioi,
                 black=IS_BLACK[perf.pitch].astype(float), valid_w=wfrac[np.arange(N), hp])
    conf_raw = hand_conf * finger_conf
    if calib is not None:
        from .calibrate import FEATURES
        conf = calib.predict(np.stack([feats[k] for k in FEATURES], 1))
    else:
        conf = conf_raw
    status = np.full(perf.N, OK, dtype=int)
    status[conf < cfg.conf_thresh] = LOW_CONF
    status[s.max((1, 2)) < cfg.fake_thresh] = SUSPECT
    status[overfull] = OVERFULL
    status[(hand < 0) | ((finger < 0) & ~overfull)] = NO_VIEW
    good = status == OK
    return dict(
        video_id=perf.video_id, fps=fps, av_offset=av, bias=bias, frame_offset=perf.frame_offset,
        onset=perf.onset, offset=perf.offset, pitch=perf.pitch,
        onset_frame=on_f, offset_frame=off_f,
        hand_pred=hand, finger_pred=finger, hand_conf=hand_conf, finger_conf=finger_conf, conf=conf,
        conf_raw=conf_raw, **{k: v for k, v in feats.items() if k not in ("finger_conf", "hand_conf")},
        status=status,
        hand=np.where(good, hand, -1), finger=np.where(good, finger, -1),
        release_frame=rel, release_s=np.where(rel >= 0, (rel - av) / fps, np.nan), release_uncertain=unc,
        valid_frac=ok.mean(0),
    )


def save_result(res: dict, path: str) -> None:
    check_output_path(path)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in res.items()})


def load_result(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.files}
    for k in ("video_id",):
        out[k] = str(out[k])
    for k in ("fps",):
        out[k] = float(out[k])
    for k in ("av_offset", "frame_offset"):
        out[k] = int(out[k])
    return out
