"""Endoscapes2023 (CAMMA, IHU Strasbourg; CC BY-NC-SA 4.0) — Critical View of Safety.

Every CVS keyframe (one per 5 s of the dissection phase) is rated by three surgeons on the
three CVS criteria:
  C1  two structures (cystic duct, cystic artery) and only two enter the gallbladder
  C2  the hepatocystic triangle is cleared of fat and fibrous tissue
  C3  the lower third of the gallbladder is separated from the liver (cystic plate exposed)
The multi-society Safe Cholecystectomy guideline (Brunt et al., Surg Endosc 2020) calls for
CVS before any ductal structure is clipped or divided, so the decision rule is
  clip/divide is permitted only if C1 and C2 and C3.

Labels are kept per rater: the reference standard (majority, unanimous, single rater) is a
choice made at evaluation time, never baked into the data.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "endoscapes"
CRITERIA = ("C1", "C2", "C3")
SPLITS = ("train", "val", "test")


@lru_cache(maxsize=1)
def metadata() -> pd.DataFrame:
    m = pd.read_csv(DATA / "all_metadata.csv")
    split = {}
    for s in SPLITS:
        for x in (DATA / f"{s}_vids.txt").read_text().split():
            split[int(float(x))] = s
    m["split"] = m.vid.map(split)
    m["file"] = m.apply(lambda r: DATA / r.split / f"{r.vid}_{r.frame}.jpg", axis=1)
    return m


@dataclass
class Keyframes:
    """CVS-rated frames of one split, in (video, time) order."""
    vid: np.ndarray            # (N,)
    frame: np.ndarray          # (N,) frame index in the source video (25 fps)
    raters: np.ndarray         # (N, 3 raters, 3 criteria) in {0,1}
    files: list[Path]

    @property
    def soft(self) -> np.ndarray:                       # (N, 3) mean over raters
        return self.raters.mean(1)

    def reference(self, kind: str = "majority") -> np.ndarray:
        """(N, 3) bool criteria under a reference standard."""
        if kind == "majority":
            return self.raters.sum(1) >= 2
        if kind == "unanimous":
            return self.raters.all(1)
        if kind == "any":                               # most lenient: some rater says met
            return self.raters.any(1)
        if kind.startswith("rater"):
            return self.raters[:, int(kind[-1]) - 1].astype(bool)
        raise ValueError(kind)

    def cvs(self, kind: str = "majority") -> np.ndarray:
        return self.reference(kind).all(1)


def keyframes(split: str) -> Keyframes:
    m = metadata()
    k = m[(m.split == split) & m.is_ds_keyframe].sort_values(["vid", "frame"])
    R = np.array([[ast.literal_eval(r[f"cvs_annotator_{i}"]) for i in (1, 2, 3)]
                  for _, r in k.iterrows()], dtype=np.int8)
    return Keyframes(k.vid.to_numpy(), k.frame.to_numpy(), R, list(k.file))


def all_frames(split: str) -> pd.DataFrame:
    """Every 1-fps frame of the region of interest (for temporal context)."""
    m = metadata()
    return m[m.split == split].sort_values(["vid", "frame"]).reset_index(drop=True)
