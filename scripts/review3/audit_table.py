"""Per-operation audit table (review round 3, R-A 'minimal reproducible release').

Fixed official split (31 calibration -> 40 test operations), DINOv2-L, 3 seeds, alpha = 0.10.
One row per (seed, operation): role, scored keyframes, unsafe keyframes, the operation statistic m_v
(max smoothed P(CVS) over unsafe keyframes, 10 s), the calibrated threshold, whether the operation
errs, how many statements were issued (true / false), and the time-out: score at the clip, reference,
decision of the final rule (confirm alpha_c = 0.10, conditional STOP alpha_s = 0.20, else defer).

    python scripts/review3/audit_table.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "review"))
from cdss.data import keyframes                       # noqa: E402
from common import endoscapes_ops, order_stat         # noqa: E402

A, W, WT = 0.10, 10, 5


def main():
    te = set(keyframes("test").vid.tolist())
    rows = []
    for seed in (0, 1, 2):
        ops = endoscapes_ops("dinov2l_b8", seed, (W, WT))
        cal = [o for o in ops if o["vid"] not in te]
        m = [o["p"][W][~o["cvs"]].max() for o in cal if (~o["cvs"]).any()]
        lam = order_stat(m, A)
        pc = np.array([o["p"][WT][-1] for o in cal]); bc = np.array([bool(o["cvs"][-1]) for o in cal])
        lam_c = order_stat(pc[~bc], A)
        pos = np.sort(pc[bc]); r = math.floor(0.20 * (len(pos) + 1)); mu = -math.inf if r == 0 else float(pos[r - 1])
        for o in ops:
            p, y = o["p"][W], o["cvs"]
            c = p > lam; pt = o["p"][WT][-1]
            dec = "confirm" if pt > lam_c else ("stop" if pt < mu else "defer")
            rows.append({"seed": seed, "vid": o["vid"], "role": "test" if o["vid"] in te else "calibration",
                         "scored_keyframes": len(y), "unsafe_keyframes": int((~y).sum()),
                         "m_v": float(p[~y].max()) if (~y).any() else np.nan, "threshold": lam,
                         "errs": bool((c & ~y).any()), "issued_true": int((c & y).sum()), "issued_false": int((c & ~y).sum()),
                         "timeout_score": float(pt), "timeout_cvs": bool(y[-1]), "timeout_lambda_c": lam_c,
                         "timeout_stop_below": mu, "timeout_decision": dec})
    df = pd.DataFrame(rows)
    out = ROOT / "experiments" / "review3" / "audit_official_split.csv"; df.to_csv(out, index=False)
    t = df[df.role == "test"]
    print(t.groupby("seed").agg(ops=("vid", "size"), errs=("errs", "sum"), issued_true=("issued_true", "sum"),
                                issued_false=("issued_false", "sum")).to_string())
    print("wrote", out)


if __name__ == "__main__":
    main()
