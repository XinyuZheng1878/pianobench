"""Hand assignment + per-hand HMM over onset events (states = ordered finger sets for the chord)."""
from __future__ import annotations

import itertools

import numpy as np
from scipy.special import logsumexp

from .events import group_events, norm_x

# states for a k-note event: k distinct fingers, increasing along the hand-normalised x
COMBOS = {k: np.array(list(itertools.combinations(range(5), k)), dtype=int) for k in range(1, 6)}


def assign_hands(logE, avail):
    H = np.where(avail, logsumexp(logE, axis=2), -np.inf)       # (N, 2)
    has = avail.any(1)
    hand = np.where(has, H.argmax(1), -1)
    m = np.where(has, H.max(1), 0.0)[:, None]
    p = np.exp(H - m)
    p = p / np.maximum(p.sum(1, keepdims=True), 1e-300)
    return hand, np.where(has, p.max(1), 0.0)


def _forward_backward(em, tr):
    n = len(em)
    alpha = [em[0]]
    for i in range(1, n):
        alpha.append(em[i] + logsumexp(alpha[-1][:, None] + tr[i], axis=0))
    beta = [np.zeros_like(e) for e in em]
    for i in range(n - 2, -1, -1):
        beta[i] = logsumexp(tr[i + 1] + (em[i + 1] + beta[i + 1])[None, :], axis=1)
    post = []
    for a, b in zip(alpha, beta):
        lp = a + b
        p = np.exp(lp - lp.max())
        post.append(p / p.sum())
    return post


def _viterbi(em, tr):
    delta, back = em[0], [None]
    for i in range(1, len(em)):
        sc = delta[:, None] + tr[i]
        back.append(sc.argmax(0))
        delta = em[i] + sc.max(0)
    path = [int(delta.argmax())]
    for i in range(len(em) - 1, 0, -1):
        path.append(int(back[i][path[-1]]))
    return path[::-1]


def decode_hand(h, idx, logE, onset_s, onset_f, pitch, prior, cfg, busy=None):
    """Returns ({note: finger}, {note: marginal of that finger}, [notes dropped from >5-note events])."""
    xn = norm_x(pitch, h)
    ev_notes, ev_states, em, tr, overfull = [], [], [], [], []
    for ev in group_events(onset_s, idx, cfg.chord_tol_s):
        if len(ev) > 5:
            keep = sorted(ev, key=lambda n: -logE[n, h].max())[:5]
            overfull += [n for n in ev if n not in keep]
            ev = keep
        ev = sorted(ev, key=lambda n: xn[n])
        k = len(ev)
        S = COMBOS[k]
        e = logE[ev, h, :][np.arange(k)[None, :], S].sum(1)
        for j in range(k - 1):
            e = e + cfg.w_chord * prior.chord[S[:, j], S[:, j + 1], prior.bin(xn[ev[j + 1]] - xn[ev[j]])]
        if busy is not None:
            e = e - cfg.occupancy_penalty * busy(h, int(onset_f[ev].min()))[S].sum(1)
        if ev_notes:
            P, Sp = ev_notes[-1], ev_states[-1]
            gap = onset_s[ev].min() - onset_s[P].min()
            pairs = {(0, 0), (len(P) - 1, k - 1)}
            M = np.zeros((len(Sp), len(S)))
            for a, b in pairs:
                M += prior.seq_logp(Sp[:, a][:, None], S[:, b][None, :], xn[ev[b]] - xn[P[a]], gap, cfg.gap_tau_s)
            tr.append(cfg.w_seq * M / len(pairs))
        else:
            tr.append(None)
        ev_notes.append(ev)
        ev_states.append(S)
        em.append(e)

    finger, conf = {}, {}
    if not ev_notes:
        return finger, conf, overfull
    post = _forward_backward(em, tr)
    path = _viterbi(em, tr)
    for ev, S, p, s in zip(ev_notes, ev_states, post, path):
        for j, n in enumerate(ev):
            f = int(S[s, j])
            finger[n] = f
            conf[n] = float(p[S[:, j] == f].sum())
    return finger, conf, overfull


def decode_all(logE, hand, onset_s, onset_f, pitch, prior, cfg, busy=None):
    N = len(hand)
    finger = np.full(N, -1, dtype=int)
    conf = np.zeros(N)
    overfull = np.zeros(N, dtype=bool)
    for h in (0, 1):
        f, c, of = decode_hand(h, np.flatnonzero(hand == h), logE, onset_s, onset_f, pitch, prior, cfg, busy)
        for n, v in f.items():
            finger[n] = v
            conf[n] = c[n]
        overfull[of] = True
    return finger, conf, overfull
