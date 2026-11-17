"""Four separate labelled frames of the teaser operation (fig_timeline.select_operation), chosen among the
36 assessed timestamps so that the dissection window is covered by an instrument or smoke; the time stamps
therefore differ from the frames picked automatically by fig_teaser.py, the operation, thresholds and
decisions are the same.  Crop (8 % at each side) as in the teaser.

  teaser_frames/{1_first_false,2_highest_false,3_cvs_achieved,4_last_assessed}.png           frame only
  teaser_frames/{...}_labelled.png                                                            with time + label
  teaser_frames/SOURCE.txt                                                                    index, time, score, labels

    python scripts/figures/export_teaser_frames.py
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures" / "teaser_frames"; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_timeline import ALPHA, SMOOTH, select_operation  # noqa: E402

FONT = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
CRC, GREEN, INK = "#C8440A", "#138A60", "#202B36"
# keyframe index -> (file stem, label, colour); all must satisfy the stated decision (asserted below)
PICKS = [(8, "1_first_false", "first false confirmation", CRC), (12, "2_highest_false", "highest false score", CRC),
         (33, "3_cvs_achieved", "CVS achieved (surgeons)", GREEN), (35, "4_last_assessed", "last assessed timestamp", INK)]


def main():
    S = select_operation()
    kt, ky, kp, lam_frame, lam_op, files, kfr, v = (S[k] for k in ("kt", "ky", "kp", "lam_frame", "lam_op", "files", "kfr", "v"))
    fx = (kp > lam_frame) & ~ky
    assert fx[8] and fx[12] and ky[33] and ky[35] and 35 == len(kt) - 1 and not (kp[12] > lam_op)
    assert kp[12] >= np.sort(kp[fx])[-2]                          # second-highest false score
    font = ImageFont.truetype(FONT, 34); lines = []
    for i, stem, lab, col in PICKS:
        im = Image.open(files[(v, int(kfr[i]))]).convert("RGB"); W_, H_ = im.size
        im = im.crop((int(0.08 * W_), 0, int(0.92 * W_), H_))
        im.save(OUT / f"{stem}.png")
        lab_im = Image.new("RGB", (im.width, im.height + 92), "white"); lab_im.paste(im, (0, 0))
        d = ImageDraw.Draw(lab_im)
        d.text((8, im.height + 6), f"{kt[i]:.1f} min", fill="#555555", font=font)
        d.text((8, im.height + 48), lab, fill=col, font=font)
        lab_im.save(OUT / f"{stem}_labelled.png")
        lines.append(f"{stem}: keyframe index {i} (video {v}, frame {int(kfr[i])}), t = {kt[i]:.2f} min, "
                     f"p = {kp[i]:.3f}, CVS (surgeons) = {int(ky[i])}, frame-level false confirmation = {int(fx[i])}")
    (OUT / "SOURCE.txt").write_text(
        f"Endoscapes test video {v}; alpha = {ALPHA}, {SMOOTH} s smoothing; thresholds frame-level {lam_frame:.3f}, "
        f"per operation {lam_op:.3f}.\nFrames cropped 8 % at each side as in fig_teaser.py.\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
