"""External test set from Cholec80 + Cholec80-CVS.

Frames: the calot-triangle-dissection phase (Cholec80 phase annotations), i.e. the same
region of interest as Endoscapes (dissection before the first clip), sampled at 1 fps and
resized to Endoscapes' 854x480.
Labels: Cholec80-CVS (Rios et al., Sci Data 2023) scores each criterion 0/1/2 on time
segments; frames outside annotated segments score 0 (per the dataset description).
  criterion met  := score == 2                      (fully achieved)
  CVS (strict)   := all three criteria score 2
  CVS (authors)  := total >= 5                       (the dataset's own definition)
Excluded: Cholec80 videos that also appear in Endoscapes (CAMMA overlap list):
66, 67, 68, 70, 71, 72.

    python scripts/prepare_cholec80.py --workers 16
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
C80 = ROOT / "data" / "cholec80"
OUT = C80 / "frames"
OVERLAP = {66, 67, 68, 70, 71, 72}
PHASE = "CalotTriangleDissection"
FPS = 25


def labels_for(video: int, seconds: np.ndarray, cvs: pd.DataFrame) -> np.ndarray:
    """(n, 3) criterion scores in ENDOSCAPES order: C1 two structures, C2 hepatocystic
    triangle, C3 cystic plate. (Cholec80-CVS lists cystic_plate before
    hepatocystic_triangle; the mapping was verified empirically with
    scripts/analysis/check_criterion_order.py.)"""
    S = np.zeros((len(seconds), 3), int)
    for _, r in cvs[cvs.video == video].iterrows():
        a = r.initial_minute * 60 + r.initial_second
        b = r.final_minute * 60 + r.final_second
        m = (seconds >= a) & (seconds <= b)
        S[m] = [r.two_structures, r.hepatocystic_triangle, r.cystic_plate]
    return S


def work(video: int) -> dict:
    import cv2
    ph = pd.read_csv(C80 / "phase_annotations" / f"video{video:02d}-phase.txt", sep="\t")
    frames = ph.loc[ph.Phase == PHASE, "Frame"].to_numpy()
    frames = frames[frames % FPS == 0]                     # 1 fps
    cvs = pd.read_excel(C80 / "cholec80-CVS.xlsx")
    seconds = frames // FPS
    S = labels_for(video, seconds, cvs)
    odir = OUT / f"video{video:02d}"; odir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(C80 / "videos" / f"video{video:02d}.mp4"))
    want, pos, kept = set(frames.tolist()), 0, []
    last = frames.max() if len(frames) else -1
    while pos <= last:
        ok = cap.grab()
        if not ok:
            break
        if pos in want:
            ok, img = cap.retrieve()
            if ok:
                img = cv2.resize(img, (854, 480), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(odir / f"{pos}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                kept.append(pos)
        pos += 1
    keep = np.isin(frames, kept)
    return {"video": video, "frame": frames[keep].tolist(), "scores": S[keep].tolist()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    vids = [v for v in range(1, 81) if v not in OVERLAP]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(work, vids):
            for f, s in zip(r["frame"], r["scores"]):
                rows.append({"video": r["video"], "frame": f, "C1": s[0], "C2": s[1], "C3": s[2]})
            print(f"video {r['video']:02d}: {len(r['frame'])} frames", flush=True)
    df = pd.DataFrame(rows)
    df["cvs_strict"] = (df[["C1", "C2", "C3"]] == 2).all(1)
    df["cvs_authors"] = df[["C1", "C2", "C3"]].sum(1) >= 5
    df.to_csv(C80 / "cholec80_cvs_frames.csv", index=False)
    print(json.dumps({"videos": df.video.nunique(), "frames": len(df),
                      "cvs_strict": float(df.cvs_strict.mean()), "cvs_authors": float(df.cvs_authors.mean()),
                      "criterion_met_rate": (df[["C1", "C2", "C3"]] == 2).mean().round(3).to_dict(),
                      "videos_with_cvs": int(df.groupby("video").cvs_authors.any().sum())}, indent=1))


if __name__ == "__main__":
    main()
