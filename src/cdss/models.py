"""Light heads on frozen DINOv2 features, and the temporal input.

Input per keyframe t: [f_t ; mean(f_{t-W+1..t})] from the 1-fps stream of the same video
(past frames only, so it is usable live). The head is trained on the official train
videos only, with the three-rater mean as a soft target.
"""
from __future__ import annotations

import numpy as np
import torch

from .data import Keyframes


def load_features(path) -> tuple[dict, np.ndarray]:
    z = np.load(path)
    index = {(int(v), int(f)): i for i, (v, f) in enumerate(zip(z["vid"], z["frame"]))}
    return index, z["feat"].astype(np.float32)


def temporal_inputs(kf: Keyframes, index: dict, F: np.ndarray, window: int = 5,
                    vid_frames: dict | None = None) -> np.ndarray:
    """[current ; mean of the last `window` 1-fps frames] for every keyframe."""
    if vid_frames is None:
        vid_frames = {}
        for (v, f) in index:
            vid_frames.setdefault(v, []).append(f)
        vid_frames = {v: np.sort(fs) for v, fs in vid_frames.items()}
    X = []
    for v, f in zip(kf.vid, kf.frame):
        fs = vid_frames[int(v)]
        pos = int(np.searchsorted(fs, f))
        past = fs[max(0, pos - window + 1):pos + 1]
        cur = F[index[(int(v), int(f))]]
        X.append(np.concatenate([cur, F[[index[(int(v), int(p))] for p in past]].mean(0)]))
    return np.stack(X)


class Head:
    """K independent sigmoids (criteria) or one softmax over the 2^3 joint configurations."""

    def __init__(self, dim: int, kind: str = "marginal", device: str = "cuda"):
        self.kind, self.device = kind, device
        self.out = 3 if kind == "marginal" else 8
        self.lin = torch.nn.Linear(dim, self.out).to(device)
        self.T = 1.0

    def _x(self, X):
        return torch.from_numpy((X - self.mu) / self.sd).float().to(self.device)

    def _target(self, raters: np.ndarray) -> torch.Tensor:
        if self.kind == "marginal":
            return torch.from_numpy(raters.mean(1).astype(np.float32))
        # joint: empirical distribution of the 3 raters' configurations
        code = raters[:, :, 0] * 4 + raters[:, :, 1] * 2 + raters[:, :, 2]      # (N, 3)
        Y = np.zeros((len(raters), 8), np.float32)
        for r in range(3):
            Y[np.arange(len(raters)), code[:, r]] += 1 / 3
        return torch.from_numpy(Y)

    def _loss(self, logits, Y):
        if self.kind == "marginal":
            return torch.nn.functional.binary_cross_entropy_with_logits(logits, Y)
        return -(Y * torch.log_softmax(logits, 1)).sum(1).mean()

    def fit(self, X, raters, epochs=200, lr=1e-3, wd=1e-3, seed=0):
        torch.manual_seed(seed)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xt, Y = self._x(X), self._target(raters).to(self.device)
        opt = torch.optim.AdamW(self.lin.parameters(), lr=lr, weight_decay=wd)
        for _ in range(epochs):
            perm = torch.randperm(len(Xt), device=self.device)
            for i in range(0, len(Xt), 1024):
                idx = perm[i:i + 1024]
                loss = self._loss(self.lin(Xt[idx]), Y[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    def fit_temperature(self, X, raters):
        with torch.no_grad():
            L = self.lin(self._x(X))
        Y = self._target(raters).to(self.device)
        logT = torch.zeros(1, device=self.device, requires_grad=True)
        opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=200)

        def closure():
            opt.zero_grad(); l = self._loss(L / logT.exp(), Y); l.backward(); return l
        opt.step(closure)
        self.T = float(logT.exp())
        return self.T

    def predict(self, X) -> np.ndarray:
        """marginal: (N, 3) P(criterion met); joint: (N, 8) P(configuration)."""
        with torch.no_grad():
            L = self.lin(self._x(X)) / self.T
            P = torch.sigmoid(L) if self.kind == "marginal" else torch.softmax(L, 1)
        return P.cpu().numpy()
