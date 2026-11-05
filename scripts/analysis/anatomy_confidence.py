"""Why the surgery-level anatomy certificate rarely speaks: per-operation maximum confidence of a
WRONG duct/artery label on the calibration operations (official split, seed 0) and the
confidence of correct vs wrong candidate labels on the test operations.

    python scripts/analysis/anatomy_confidence.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
from cdss import anatomy as A                         # noqa: E402
from cdss.data import keyframes                       # noqa: E402
from surgery_level import order_stat_threshold, prepare  # noqa: E402

sv = np.random.default_rng(0).choice(np.unique(keyframes("val").vid), 10, replace=False)
frames, *_ = prepare(0, sv)
tv = set(keyframes("test").vid.tolist())
cal = [f for f in frames if f["vid"] not in tv]; tst = [f for f in frames if f["vid"] in tv]
per = {}
for f in cal:
    w = [f["prob"][t] for t in A.CLIP_TARGETS if f["cand"][t] is not None and not f["correct"][t]]
    per.setdefault(f["vid"], []).append(max(w) if w else -np.inf)
m = np.array([max(v) for v in per.values()])
print("calibration operations", len(m), "with at least one wrong candidate label", int(np.isfinite(m).sum()))
print("per-operation max WRONG confidence (sorted):", np.round(np.sort(m[np.isfinite(m)]), 3))
for a in (0.05, 0.1, 0.2):
    print(f"surgery-level threshold alpha={a}: {order_stat_threshold(m, a):.3f}")
c = np.array([f["prob"][t] for f in tst for t in A.CLIP_TARGETS if f["cand"][t] is not None and f["correct"][t]])
w = np.array([f["prob"][t] for f in tst for t in A.CLIP_TARGETS if f["cand"][t] is not None and not f["correct"][t]])
print("test candidate labels: correct", len(c), "wrong", len(w), f"(wrong share {len(w) / (len(c) + len(w)):.2f})")
for q in (0.5, 0.8, 0.9, 0.936):
    print(f"  confidence > {q}: correct {np.mean(c > q):.2f}   wrong {np.mean(w > q):.2f}")
present = [(f["gt_labels"] == t).any() for f in tst for t in A.CLIP_TARGETS]
miss = [(f["gt_labels"] == t).any() and not (f["y"] == t).any() for f in tst for t in A.CLIP_TARGETS]
print(f"structure annotated but not matched by any detected region (detector miss): {sum(miss) / max(sum(present), 1):.2f}")
