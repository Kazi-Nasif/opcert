"""Standalone assets of the architecture overview (fig_overview_v4.py), for assembling the figure in a
drawing tool: the surgical frames exactly as used there and the order-statistic graph on its own.

  overview_assets/past_op_{1,2,3}_labelled.png   three calibration operations (surgeons' CVS rating, boxes)
  overview_assets/new_op_frame_{1,2,3}.png       three frames of the held-out test operation (no labels)
  overview_assets/anatomy_candidates.png         our Faster R-CNN output on a held-out test frame (cropped)
  overview_assets/order_statistic.{pdf,png,svg}  the 31 sorted real calibration maxima, m_(j), j formula
  overview_assets/order_statistic_large.{pdf,png,svg}  the same graph at 3.6 x 4.2 in, 600 dpi PNG
  overview_assets/order_statistic.csv            the plotted values

    python scripts/figures/export_overview_assets.py
"""
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_overview_v4 import ALPHA, ASSETS, BLU, INK, LINE, MUTE, calibration  # noqa: E402

OUT = ASSETS.parent / "overview_assets"; OUT.mkdir(exist_ok=True)


def main():
    for k in (1, 2, 3):
        shutil.copy(ASSETS / f"past_op_{k}_labelled.png", OUT / f"past_op_{k}_labelled.png")
    strip = Image.open(ASSETS / "new_op_strip.png").convert("RGB"); f5 = strip.width / 5
    for n, i in enumerate((0, 2, 4), 1):
        strip.crop((int(i * f5), 0, int((i + 1) * f5), strip.height)).save(OUT / f"new_op_frame_{n}.png")
    Image.open(ASSETS / "anatomy_detector_output.png").crop((52, 0, 597, 480)).save(OUT / "anatomy_candidates.png")

    m, j, lam, _ = calibration()
    V = len(m)
    pd.DataFrame({"rank": np.arange(1, V + 1), "m_v": m, "is_m_j": np.arange(1, V + 1) == j}).to_csv(
        OUT / "order_statistic.csv", index=False)
    plt.rcParams.update({"savefig.bbox": "tight", "savefig.pad_inches": 0.02, "mathtext.fontset": "cm"})
    for name, size, k, dpi in (("order_statistic", (1.3, 1.5), 1.0, 300), ("order_statistic_large", (3.6, 4.2), 2.6, 600)):
        fig, ax = plt.subplots(figsize=size)
        ax.grid(False)
        ax.bar(np.arange(V), m, width=0.78, color=BLU[1], ec="none", zorder=2)
        ax.bar([j - 1], [m[j - 1]], width=0.78, color=BLU[2], ec="none", zorder=3)
        ax.axhline(lam, c=INK, ls=(0, (2.2, 1.6)), lw=0.8 * k, zorder=4)
        ymax = float(m.max()) * 1.35
        ax.set_xlim(-0.8, V - 0.2); ax.set_ylim(0, ymax); ax.set_xticks([]); ax.set_yticks([])
        for sname, sp in ax.spines.items():
            sp.set_visible(sname in ("left", "bottom")); sp.set_color(LINE); sp.set_linewidth(0.6 * k)
        ax.text(0.3, lam + 0.03 * ymax, r"$m_{(j)}$", fontsize=7 * k, ha="left", va="bottom")
        ax.set_title(f"{V} sorted $m_v$", fontsize=6.5 * k, color=MUTE, loc="center", fontweight="normal")
        ax.set_xlabel(r"$j=\lceil(V{+}1)(1{-}\alpha)\rceil$", fontsize=6.5 * k, color=MUTE)
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"{name}.{ext}", dpi=dpi)
        with plt.rc_context({"svg.fonttype": "none"}):
            fig.savefig(OUT / f"{name}.svg")
        plt.close(fig)
    print(f"V = {V}, j = {j}, m_(j) = {lam:.3f}, alpha = {ALPHA}; files in {OUT}")


if __name__ == "__main__":
    main()
