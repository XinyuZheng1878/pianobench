"""Fingering priors: P(next finger | prev finger, displacement) and in-chord span prior.

Displacements are in the hand-normalised frame (events.norm_x), so one table serves both hands.
`default()` is a hand-written parametric prior; `fit()` learns tables from labelled notes
(PianoVAM auto labels on non-test pieces), backing off to the default.
"""
from __future__ import annotations

import numpy as np

from .events import group_events, norm_x

D, B = 10.0, 0.5
NB = int(2 * D / B) + 2                      # +2 overflow bins
CENTRES = -D + (np.arange(NB) - 0.5) * B


def dbin(d):
    return np.clip(np.floor((np.asarray(d, dtype=float) + D) / B).astype(int) + 1, 0, NB - 1)


def _default_seq() -> np.ndarray:
    f = np.arange(5)
    f1, f2, d = f[:, None, None], f[None, :, None], CENTRES[None, None, :]
    mu = (f2 - f1) * 1.0
    sd = 0.8 + 0.5 * np.abs(f2 - f1)
    dens = np.exp(-0.5 * ((d - mu) / sd) ** 2) / sd
    under = (f1 >= 1) & (f1 <= 3) & (f2 == 0)        # thumb passes under, moving outward
    over = (f1 == 0) & (f2 >= 1) & (f2 <= 3)         # finger crosses over the thumb, moving inward
    cross = (np.where(under, np.exp(-0.5 * ((d - 1.5) / 1.0) ** 2), 0.0)
             + np.where(over, np.exp(-0.5 * ((d + 1.5) / 1.0) ** 2), 0.0))
    dens = dens + 0.15 * cross + 1e-4
    return np.log(dens / dens.sum(1, keepdims=True))


def _default_chord() -> np.ndarray:
    f = np.arange(5)
    lo, hi, d = f[:, None, None], f[None, :, None], CENTRES[None, None, :]
    k = hi - lo
    dens = np.where(k > 0, np.exp(-0.5 * ((d - 1.3 * k) / (0.6 + 0.4 * np.abs(k))) ** 2), 0.0) + 1e-6
    return np.log(dens / dens.sum(1, keepdims=True))


class FingerPrior:
    def __init__(self, seq: np.ndarray, chord: np.ndarray, bias: np.ndarray):
        self.seq, self.chord, self.bias = seq, chord, np.asarray(bias, dtype=float)

    bin = staticmethod(dbin)

    def seq_logp(self, f1, f2, d, gap, tau):
        lp = self.seq[f1, f2, dbin(d)]
        lam = np.exp(-max(float(gap), 0.0) / tau)
        return np.log(lam * np.exp(lp) + (1.0 - lam) / 5.0)

    @classmethod
    def default(cls) -> "FingerPrior":
        return cls(_default_seq(), _default_chord(), np.zeros((2, 5)))

    @classmethod
    def load(cls, path: str) -> "FingerPrior":
        z = np.load(path)
        return cls(z["seq"], z["chord"], z["bias"])

    def save(self, path: str) -> None:
        np.savez(path, seq=self.seq, chord=self.chord, bias=self.bias)

    @classmethod
    def fit(cls, perfs, chord_tol=0.04, gap_max=2.0, alpha=2.0, bias=None):
        seq_c = np.zeros((5, 5, NB))
        ch_c = np.zeros((5, 5, NB))
        for perf in perfs:
            lh, lf = perf.label_hand, perf.label_finger
            if lh is None or lf is None:
                continue
            labeled = (lh >= 0) & (lf >= 0)
            unl_on = np.sort(perf.onset[~labeled])
            for h in (0, 1):
                xn = norm_x(perf.pitch, h)
                prev = None
                for ev in group_events(perf.onset, np.flatnonzero(labeled & (lh == h)), chord_tol):
                    ev = sorted(ev, key=lambda n: xn[n])
                    for a, b in zip(ev[:-1], ev[1:]):
                        if lf[b] > lf[a]:
                            ch_c[lf[a], lf[b], dbin(xn[b] - xn[a])] += 1
                    if prev is not None:
                        tp = perf.onset[prev].min()
                        te = perf.onset[ev].min()
                        # skip pairs with an unlabelled note in between: we don't know what happened there
                        broken = (np.searchsorted(unl_on, te - chord_tol)
                                  > np.searchsorted(unl_on, tp + chord_tol, side="right"))
                        if te - tp <= gap_max and not broken:
                            for a, b in {(prev[0], ev[0]), (prev[-1], ev[-1])}:
                                seq_c[lf[a], lf[b], dbin(xn[b] - xn[a])] += 1
                    prev = ev
        seq = seq_c + alpha * np.exp(_default_seq())
        chord = ch_c + alpha * np.exp(_default_chord())
        prior = cls(np.log(seq / seq.sum(1, keepdims=True)), np.log(chord / chord.sum(1, keepdims=True)),
                    np.zeros((2, 5)) if bias is None else bias)
        prior.counts = (int(seq_c.sum()), int(ch_c.sum()))
        return prior
