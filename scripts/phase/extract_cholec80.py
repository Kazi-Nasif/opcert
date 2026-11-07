"""Second task (review R2, P3): surgical phase on Cholec80.  Frozen, generic DINOv2-L features
for every 1-fps frame of all 80 Cholec80 videos (frame index 0, 25, 50, ... at 25 fps, aligned
with the phase annotations).  Nothing here is trained on surgical data.

Stage 1: ffmpeg decodes every 25th frame, resized to 602 x 336 (as the Endoscapes features).
Stage 2: [CLS ; mean patch] of DINOv2-L = 2048-d per frame.
Output: experiments/phase/dinov2l_cholec80.npz with vid, frame (25-fps index), phase (0..6), feat.

    python scripts/phase/extract_cholec80.py
"""
from __future__ import annotations

import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
C80 = ROOT / "data" / "cholec80"
FR = C80 / "frames_1fps"
OUT = ROOT / "experiments" / "phase"
PHASES = ["Preparation", "CalotTriangleDissection", "ClippingCutting", "GallbladderDissection",
          "GallbladderPackaging", "CleaningCoagulation", "GallbladderRetraction"]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def decode(v):
    d = FR / f"video{v:02d}"
    if (d / "done").exists():
        return v
    d.mkdir(parents=True, exist_ok=True)
    subprocess.run([ffmpeg_exe(), "-loglevel", "error", "-y", "-i", str(C80 / "videos" / f"video{v:02d}.mp4"),
                    "-vf", "select=not(mod(n\\,25)),scale=602:336", "-vsync", "vfr", "-q:v", "3",
                    str(d / "%06d.jpg")], check=True)
    (d / "done").touch()
    return v


def labels(v):
    ph = pd.read_csv(C80 / "phase_annotations" / f"video{v:02d}-phase.txt", sep="\t")
    return dict(zip(ph.Frame.to_numpy(), ph.Phase.map({p: i for i, p in enumerate(PHASES)}).to_numpy()))


class Frames(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        x = (np.asarray(Image.open(self.files[i]).convert("RGB"), np.float32) / 255 - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1))


@torch.no_grad()
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(16) as pool:
        for v in pool.imap_unordered(decode, range(1, 81)):
            print("decoded", v, flush=True)
    rows = []
    for v in range(1, 81):
        lab = labels(v)
        files = sorted((FR / f"video{v:02d}").glob("*.jpg"))
        for k, f in enumerate(files):
            fr = 25 * k
            if fr in lab:
                rows.append((v, fr, lab[fr], str(f)))
    df = pd.DataFrame(rows, columns=["vid", "frame", "phase", "file"])
    print(df.groupby("vid").size().describe(), flush=True)
    from transformers import AutoModel
    model = AutoModel.from_pretrained("facebook/dinov2-large", torch_dtype=torch.float16).cuda().eval()
    dl = DataLoader(Frames(list(df.file)), batch_size=96, num_workers=16, pin_memory=True)
    feats = []
    for i, x in enumerate(dl):
        h = model(pixel_values=x.cuda(non_blocking=True).half()).last_hidden_state.float()
        feats.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).cpu().numpy().astype(np.float16))
        if i % 200 == 0:
            print("batch", i, "of", len(dl), flush=True)
    np.savez(OUT / "dinov2l_cholec80.npz", vid=df.vid.to_numpy(), frame=df.frame.to_numpy(),
             phase=df.phase.to_numpy(), feat=np.concatenate(feats))
    print("saved", len(df), flush=True)


if __name__ == "__main__":
    sys.exit(main())
