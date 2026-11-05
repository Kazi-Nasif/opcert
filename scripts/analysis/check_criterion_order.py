"""Which Cholec80-CVS column matches which Endoscapes criterion? Compare per-criterion AP
of the Endoscapes-trained model under both orderings (Cholec80 raw columns:
C1 = two_structures, C2 = cystic_plate, C3 = hepatocystic_triangle as written by
prepare_cholec80.py BEFORE the fix; result: model C2 = hepatocystic triangle, model C3 =
cystic plate, so the columns were swapped and prepare_cholec80.py now writes Endoscapes
order). Run it on a label file written by the old version to reproduce the check."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
df = pd.read_csv(ROOT / "data" / "cholec80" / "cholec80_cvs_frames.csv")
for thr in (2, 1):
    Y = (df[["C1", "C2", "C3"]].to_numpy() >= thr)
    for s in (0, 1, 2):
        L = np.load(ROOT / "experiments" / "finetune" / f"dinov2l_b8_s{s}" / "logits_cholec80.npz")["logits"]
        ap = lambda j, k: average_precision_score(Y[:, k], L[:, j])
        print(f"score>={thr} seed {s}: model C2 vs cystic_plate {ap(1,1):.3f} | vs hepatocystic {ap(1,2):.3f}   "
              f"model C3 vs hepatocystic {ap(2,2):.3f} | vs cystic_plate {ap(2,1):.3f}   (prevalence plate {Y[:,1].mean():.3f}, triangle {Y[:,2].mean():.3f})")
