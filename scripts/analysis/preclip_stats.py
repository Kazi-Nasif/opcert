"""How often is CVS achieved in the last W seconds before the first clip?
Endoscapes: the region of interest ends at the first clip/cut, so the window is the tail of
each video's keyframes. Cholec80: the end of the calot-triangle-dissection phase."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cdss.data import keyframes
for split in ("val", "test"):
    kf = keyframes(split); cvs = kf.cvs("majority")
    for W in (30, 60, 120, 1e9):
        last, anyw, frac, n = [], [], [], []
        for v in np.unique(kf.vid):
            m = kf.vid == v; fr = kf.frame[m]; c = cvs[m]
            win = fr >= fr.max() - W * 25
            last.append(c[np.argmax(fr)]); anyw.append(c[win].any()); frac.append(c[win].mean()); n.append(win.sum())
        print(f"Endoscapes {split} W={W:>5}s: videos {len(last)}, CVS at last keyframe {np.mean(last):.2f}, "
              f"any CVS in window {np.mean(anyw):.2f}, CVS frame share {np.mean(frac):.3f}, keyframes/window {np.mean(n):.1f}")
c = pd.read_csv(ROOT / "data" / "cholec80" / "cholec80_cvs_frames.csv")
for W in (30, 60, 120, 1e9):
    g = c.groupby("video")
    last = g.apply(lambda d: d.sort_values("frame").cvs_authors.iloc[-1])
    win = g.apply(lambda d: d[d.frame >= d.frame.max() - W * 25].cvs_authors.mean())
    anyw = g.apply(lambda d: d[d.frame >= d.frame.max() - W * 25].cvs_authors.any())
    print(f"Cholec80 W={W:>5}s: CVS at last frame {last.mean():.2f}, any in window {anyw.mean():.2f}, share {win.mean():.3f}")
