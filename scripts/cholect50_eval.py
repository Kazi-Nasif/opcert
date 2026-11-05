"""Certified "which structure is being clipped / divided" on CholecT50 (30 evaluation
operations, 200 random 15/15 calibration-evaluation halvings, 3 classifier seeds).

A certificate names the top class of a frame ("clipping the cystic duct") and is wrong
if the annotated target differs. Methods
  Point world               always name the argmax
  CRC (frames pooled)       P(certificate issued and wrong) <= a, frame level
  LTT (frames pooled)       same risk, high probability, delta = 0.1
  Surgery-level        P(any wrong certificate in an operation) <= a (exact)
  ... + smoothing           causal mean of class probabilities over the last 5 event frames
  Surgery-level, k=1   P(more than one wrong certificate in an operation) <= a
Metrics: operations with >= 1 wrong certificate, frame error (wrong certificates / frames),
coverage (frames certified), precision (correct / certified).

    python scripts/cholect50_eval.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cdss import riskcontrol as RC        # noqa: E402

ALPHAS = (0.10, 0.20)


def order_stat(m, alpha):
    m = np.sort(np.asarray(m, float)); k = math.ceil((len(m) + 1) * (1 - alpha))
    return math.inf if len(m) == 0 or k > len(m) else float(m[k - 1])


def smooth(P, vids, frames, w=5):
    out = P.copy()
    for v in np.unique(vids):
        ix = np.flatnonzero(vids == v); ix = ix[np.argsort(frames[ix])]
        for j, i in enumerate(ix):
            out[i] = P[ix[max(0, j - w + 1):j + 1]].mean(0)
    return out


def main():
    rows = []
    for s in (0, 1, 2):
        z = np.load(ROOT / "experiments" / "cholect50" / f"target_dinov2l_s{s}" / "probs_eval.npz")
        vids, frames, y = z["video"], z["frame"], z["y"]
        for name_p, P in (("raw", z["probs"]), ("smooth", smooth(z["probs"], vids, frames))):
            top, conf = P.argmax(1), P.max(1)
            wrong = top != y
            uv = np.unique(vids)
            for rep in range(200):
                rng = np.random.default_rng(rep)
                cm = np.isin(vids, rng.choice(uv, len(uv) // 2, replace=False)); tm = ~cm

                def add(method, alpha, cert):
                    tv = vids[tm]; bad = cert & wrong[tm]
                    rows.append({"seed": s, "rep": rep, "alpha": alpha, "method": method,
                                 "ops_wrong_ge1": float(np.mean([bad[tv == v].any() for v in np.unique(tv)])),
                                 "ops_wrong_gt1": float(np.mean([bad[tv == v].sum() > 1 for v in np.unique(tv)])),
                                 "frame_error": float(bad.mean()), "coverage": float(cert.mean()),
                                 "precision": float((cert & ~wrong[tm]).sum() / max(cert.sum(), 1))})
                if name_p == "raw":
                    add("Point world", None, np.ones(tm.sum(), bool))
                for alpha in ALPHAS:
                    if name_p == "raw":
                        grid = RC.threshold_grid(conf[cm])
                        L = (conf[cm][:, None] > grid[None]) & wrong[cm][:, None]
                        for nm, fn in (("CRC (frames pooled)", RC.crc_index), ("LTT (frames pooled)", RC.ltt_index)):
                            g = fn(L, alpha); lam = np.inf if g is None else grid[g]
                            add(nm, alpha, conf[tm] > lam)
                    for k, nm in ((0, "Surgery-level"), (1, "Surgery-level, k = 1")):
                        m = []
                        for v in np.unique(vids[cm]):
                            w = np.sort(conf[cm & (vids == v) & wrong])[::-1]
                            m.append(w[k] if len(w) > k else -np.inf)
                        lam = order_stat(m, alpha)
                        add(nm + (" + smoothing" if name_p == "smooth" else ""), alpha, conf[tm] > lam)
        print("seed", s, "done", flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "experiments" / "cholect50"; df.to_csv(out / "target_certification.csv", index=False)
    df["alpha"] = df.alpha.fillna(-1)
    g = df.groupby(["alpha", "method"])[["ops_wrong_ge1", "ops_wrong_gt1", "frame_error", "coverage", "precision"]].mean().round(3)
    pd.set_option("display.width", 220); print(g.to_string())


if __name__ == "__main__":
    main()
