"""Run the Endoscapes-trained CVS models (frozen, selected checkpoints) on the Cholec80
external frames. Nothing is retrained on Cholec80.

    python scripts/infer_cholec80.py --seeds 0 1 2
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from finetune import build_model, predict     # noqa: E402

C80 = ROOT / "data" / "cholec80"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--arch", choices=["dinov2l", "resnet50", "surgenetxl", "endovit"], default="dinov2l")
    a = ap.parse_args()
    df = pd.read_csv(C80 / "cholec80_cvs_frames.csv")
    files = [C80 / "frames" / f"video{v:02d}" / f"{f}.jpg" for v, f in zip(df.video, df.frame)]
    for s in a.seeds:
        run = ROOT / "experiments" / "finetune" / (f"dinov2l_b8_s{s}" if a.arch == "dinov2l" else f"{a.arch}_s{s}")
        model = build_model(a.arch).cuda()
        model.load_state_dict(torch.load(run / "best.pt", map_location="cuda"))
        L = predict(model, files)
        np.savez(run / "logits_cholec80.npz", video=df.video.to_numpy(), frame=df.frame.to_numpy(), logits=L)
        print(f"seed {s}: {L.shape}", flush=True)
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
