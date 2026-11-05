"""Frozen DINOv2-Large features for every 1-fps frame of the Endoscapes region of interest.

Per frame: [CLS ; mean of patch tokens] = 2048-d, at 336 x 602 (24 x 43 patches).
Nothing is trained. Output: experiments/features/dinov2l_{split}.npz with vid, frame, feat.

    CUDA_VISIBLE_DEVICES=5 python scripts/extract_features.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cdss.data import SPLITS, all_frames          # noqa: E402

OUT = ROOT / "experiments" / "features"
H, W = 336, 602
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class Frames(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        im = Image.open(self.files[i]).convert("RGB").resize((W, H), Image.BICUBIC)
        x = (np.asarray(im, np.float32) / 255 - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1))


@torch.no_grad()
def main():
    from transformers import AutoModel
    model = AutoModel.from_pretrained("facebook/dinov2-large", torch_dtype=torch.float16).cuda().eval()
    OUT.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        out = OUT / f"dinov2l_{split}.npz"
        if out.exists():
            print("skip", split); continue
        df = all_frames(split)
        dl = DataLoader(Frames(list(df.file)), batch_size=64, num_workers=16)
        feats = []
        for x in dl:
            h = model(pixel_values=x.cuda().half()).last_hidden_state.float()
            feats.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).cpu().numpy())
        F = np.concatenate(feats).astype(np.float16)
        np.savez(out, vid=df.vid.to_numpy(), frame=df.frame.to_numpy(), feat=F)
        print(split, F.shape, flush=True)


if __name__ == "__main__":
    main()
