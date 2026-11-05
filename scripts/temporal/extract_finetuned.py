"""Per-frame embeddings of the fine-tuned DINOv2-L CVS model ([CLS ; mean patch] = 2048-d,
the input of its linear head) for every 1-fps frame of the Endoscapes region of interest,
train / val / test.  Input of the two-stage temporal models (as in TeCNO-style pipelines:
frame encoder first, temporal model second).

    python scripts/temporal/extract_finetuned.py --seed 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from cdss.data import all_frames                 # noqa: E402
from finetune import CVSFrames, CVSNet           # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    model = CVSNet(8).cuda().eval()
    model.load_state_dict(torch.load(ROOT / "experiments" / "finetune" / f"dinov2l_b8_s{a.seed}" / "best.pt",
                                     map_location="cuda"))
    out = ROOT / "experiments" / "features"; out.mkdir(exist_ok=True)
    for split in ("train", "val", "test"):
        df = all_frames(split)
        dl = DataLoader(CVSFrames(list(df.file)), batch_size=64, num_workers=16, pin_memory=True)
        F = []
        for x, _ in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = model.backbone(pixel_values=x.cuda(non_blocking=True)).last_hidden_state
            F.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).float().cpu())
        F = torch.cat(F).numpy().astype(np.float16)
        np.savez(out / f"ft_dinov2l_s{a.seed}_{split}.npz", vid=df.vid.to_numpy(), frame=df.frame.to_numpy(), feat=F)
        print(split, F.shape, flush=True)


if __name__ == "__main__":
    main()
