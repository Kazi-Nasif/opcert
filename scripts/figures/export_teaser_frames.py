"""Four separate frames (no labels) of the teaser operation (fig_timeline.select_operation), chosen among the
36 assessed timestamps so that the dissection window is covered by an instrument or smoke; the time stamps
therefore differ from the frames picked automatically by fig_teaser.py, the operation, thresholds and
decisions are the same.  Crop (8 % at each side) as in the teaser.

  teaser_frames/{1_first_false,2_highest_false,3_cvs_achieved,4_last_assessed}.png   softened for display
  teaser_frames/{...}_raw.png                                                          the original crop
  teaser_frames/SOURCE.txt                                                             index, time, score, labels

The displayed frames are softened uniformly (SOFT below: Gaussian blur, reduced colour saturation and
contrast, slightly brighter); the caption should say so.  The model only ever saw the raw frames.

    python scripts/figures/export_teaser_frames.py
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures" / "teaser_frames"; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_timeline import ALPHA, SMOOTH, select_operation  # noqa: E402

SOFT = dict(blur=1.5, saturation=0.65, contrast=0.90, brightness=1.05)   # display softening (717 px wide crop)
# keyframe index -> (file stem, label); all must satisfy the stated decision (asserted below)
PICKS = [(8, "1_first_false", "first false confirmation"), (12, "2_highest_false", "highest false score"),
         (33, "3_cvs_achieved", "CVS achieved (surgeons)"), (35, "4_last_assessed", "last assessed timestamp")]


def soften(im):
    im = im.filter(ImageFilter.GaussianBlur(SOFT["blur"]))
    im = ImageEnhance.Color(im).enhance(SOFT["saturation"])
    im = ImageEnhance.Contrast(im).enhance(SOFT["contrast"])
    return ImageEnhance.Brightness(im).enhance(SOFT["brightness"])


def main():
    S = select_operation()
    kt, ky, kp, lam_frame, lam_op, files, kfr, v = (S[k] for k in ("kt", "ky", "kp", "lam_frame", "lam_op", "files", "kfr", "v"))
    fx = (kp > lam_frame) & ~ky
    assert fx[8] and fx[12] and ky[33] and ky[35] and 35 == len(kt) - 1 and not (kp[12] > lam_op)
    assert kp[12] >= np.sort(kp[fx])[-2]                          # second-highest false score
    lines = []
    for i, stem, lab in PICKS:
        im = Image.open(files[(v, int(kfr[i]))]).convert("RGB"); W_, H_ = im.size
        im = im.crop((int(0.08 * W_), 0, int(0.92 * W_), H_))
        im.save(OUT / f"{stem}_raw.png"); soften(im).save(OUT / f"{stem}.png")
        lines.append(f"{stem} ({lab}): keyframe index {i} (video {v}, frame {int(kfr[i])}), t = {kt[i]:.2f} min, "
                     f"p = {kp[i]:.3f}, CVS (surgeons) = {int(ky[i])}, frame-level false confirmation = {int(fx[i])}")
    (OUT / "SOURCE.txt").write_text(
        f"Endoscapes test video {v}; alpha = {ALPHA}, {SMOOTH} s smoothing; thresholds frame-level {lam_frame:.3f}, "
        f"per operation {lam_op:.3f}.\nFrames cropped 8 % at each side as in fig_teaser.py; *_raw.png is the original crop, "
        f"the displayed frame is softened with {SOFT}.\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
