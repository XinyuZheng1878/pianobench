"""Learned confidence: logistic regression on per-note features -> P(hand and finger correct).

Fit only on training/validation notes with labels (PianoVAM auto labels), never on the sealed
human-GT pieces. Plain numpy (IRLS with L2), so no sklearn dependency on the server.
"""
from __future__ import annotations

import numpy as np

FEATURES = ["finger_conf", "hand_conf", "ev_sel", "ev_margin", "chord", "log_ioi", "black", "valid_w"]


def note_features(res: dict) -> np.ndarray:
    return np.stack([np.asarray(res[k], dtype=float) for k in FEATURES], 1)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class Calibrator:
    def __init__(self, mean, std, w):
        self.mean, self.std, self.w = np.asarray(mean), np.asarray(std), np.asarray(w)

    def _design(self, X):
        Xs = (X - self.mean) / self.std
        return np.concatenate([Xs, np.ones((len(X), 1))], 1)

    def predict(self, X):
        return _sigmoid(self._design(X) @ self.w)

    @classmethod
    def fit(cls, X, y, l2=1e-2, iters=50):
        mean = X.mean(0)
        std = X.std(0) + 1e-6
        c = cls(mean, std, np.zeros(X.shape[1] + 1))
        A = c._design(X)
        reg = l2 * np.eye(A.shape[1])
        reg[-1, -1] = 0.0
        w = c.w
        for _ in range(iters):
            p = _sigmoid(A @ w)
            g = A.T @ (p - y) + reg @ w
            Hm = (A * (p * (1 - p))[:, None]).T @ A + reg
            step = np.linalg.solve(Hm, g)
            w = w - step
            if np.abs(step).max() < 1e-8:
                break
        c.w = w
        return c

    def save(self, path):
        np.savez(path, mean=self.mean, std=self.std, w=self.w, features=np.array(FEATURES))

    @classmethod
    def load(cls, path):
        z = np.load(path)
        assert list(z["features"]) == FEATURES, "calibrator was fitted on a different feature set"
        return cls(z["mean"], z["std"], z["w"])
