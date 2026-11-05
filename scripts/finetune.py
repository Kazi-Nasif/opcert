"""Fine-tune DINOv2-Large for the three CVS criteria on Endoscapes-CVS201.

Protocol (no leakage into conformal calibration or test):
  train      official train videos (120)            -> gradient updates
  select     10 of the 41 val videos (fixed seed)   -> epoch / checkpoint selection
  calibrate  the other 31 val videos                -> untouched here
  test       official test videos (40)              -> untouched here

Target: three-surgeon mean per criterion (soft BCE). Augmentation keeps anatomy
orientation (no flips). Last N transformer blocks + head are trained.

After training, logits are written for EVERY 1-fps frame of val and test, so temporal
and window-level decisions can be built without re-running the network.

    python scripts/finetune.py --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cdss.data import all_frames, keyframes                      # noqa: E402

H, W = 336, 602
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class CVSFrames(Dataset):
    def __init__(self, files, targets=None, train=False, size=(H, W), mean=MEAN, std=STD):
        H, W = size
        self.mean, self.std = mean, std
        import torchvision.transforms.v2 as T
        self.files, self.targets = files, targets
        self.tf = T.Compose([
            T.RandomResizedCrop((H, W), scale=(0.6, 1.0), ratio=(W / H * 0.85, W / H * 1.15), antialias=True),
            T.RandomApply([T.ColorJitter(0.3, 0.3, 0.2, 0.03)], p=0.8),
            T.RandomApply([T.RandomRotation(10)], p=0.3),
        ]) if train else T.Resize((H, W), antialias=True)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        x = torch.from_numpy(np.asarray(Image.open(self.files[i]).convert("RGB"))).permute(2, 0, 1)
        x = (self.tf(x).float() / 255 - self.mean) / self.std
        y = torch.zeros(3) if self.targets is None else torch.from_numpy(self.targets[i]).float()
        return x, y


class CVSNet(torch.nn.Module):
    def __init__(self, train_blocks: int = 8):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained("facebook/dinov2-large")
        for p in self.backbone.parameters():
            p.requires_grad = False
        for blk in self.backbone.encoder.layer[-train_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in self.backbone.layernorm.parameters():
            p.requires_grad = True
        d = self.backbone.config.hidden_size
        self.head = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(2 * d, 3))

    def forward(self, x):
        h = self.backbone(pixel_values=x).last_hidden_state
        return self.head(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1))


class ResNetCVS(torch.nn.Module):
    """The Endoscapes benchmark baseline: ImageNet ResNet50, fully fine-tuned, 3 outputs."""

    def __init__(self):
        super().__init__()
        import torchvision
        m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(2048, 3))
        self.backbone, self.head = m, m.fc

    def forward(self, x):
        return self.backbone(x)


class TimmViTCVS(torch.nn.Module):
    """Surgical foundation models as drop-in backbones (timm ViTs, last `train_blocks` blocks
    trained, head on [CLS ; mean of patch tokens] as for DINOv2-L).
      surgenetxl  SurgeNetXL DINOv2 ViT-L/14 (+registers), 336 px     (CC BY-NC 4.0)
      endovit     EndoViT MAE ViT-B/16 on Endo700k, 224 px             (Apache-2.0)"""

    SPECS = {
        "surgenetxl": dict(repo="rlpddejong/SurgeNetXL_DINOv1-v3", file="DINOv2_ViTl14_size336_SurgeNetXL.pth",
                           timm="vit_large_patch14_reg4_dinov2", size=(336, 602),
                           mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        "endovit": dict(repo="egeozsoy/EndoViT", file="pytorch_model.bin", timm="vit_base_patch16_224",
                        size=(224, 400), mean=(0.3464, 0.2280, 0.2228), std=(0.2520, 0.2128, 0.2093)),
    }

    def __init__(self, arch: str, train_blocks: int = 8):
        super().__init__()
        import timm
        from huggingface_hub import hf_hub_download
        spec = self.SPECS[arch]
        sd = torch.load(hf_hub_download(spec["repo"], spec["file"]), map_location="cpu", weights_only=False)
        sd = sd.get("model", sd)
        kw = dict(pretrained=False, num_classes=0, dynamic_img_size=True)
        if arch == "surgenetxl":
            kw.update(img_size=336, reg_tokens=sd["register_tokens"].shape[1], init_values=1e-5)
        self.backbone = timm.create_model(spec["timm"], **kw)
        from timm.models.vision_transformer import checkpoint_filter_fn
        sd = {k: v for k, v in sd.items() if not k.startswith(("decoder", "mask_token", "dino_head", "ibot_head"))}
        if arch == "surgenetxl":
            # official DINOv2 layout: pos_embed covers [CLS + patches]; timm's register-token ViT keeps
            # only patch positions (no_embed_class). DINOv2 adds the CLS position to the CLS token, so
            # folding it into cls_token is exact.
            assert self.backbone.no_embed_class
            pe = sd.pop("pos_embed")
            sd["cls_token"] = sd["cls_token"] + pe[:, :1]
            sd["pos_embed"] = pe[:, 1:]
            sd["reg_token"] = sd.pop("register_tokens")
        else:
            sd = checkpoint_filter_fn(sd, self.backbone)
        missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
        bad = [k for k in missing if not k.startswith(("head", "fc_norm"))]
        assert not bad and not unexpected, (bad[:5], unexpected[:5])
        for prm in self.backbone.parameters():
            prm.requires_grad = False
        for blk in self.backbone.blocks[-train_blocks:]:
            for prm in blk.parameters():
                prm.requires_grad = True
        for prm in self.backbone.norm.parameters():
            prm.requires_grad = True
        d = self.backbone.num_features
        self.head = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(2 * d, 3))
        self.size = spec["size"]
        self.mean = torch.tensor(spec["mean"]).view(3, 1, 1); self.std = torch.tensor(spec["std"]).view(3, 1, 1)

    def forward(self, x):
        h = self.backbone.forward_features(x)
        k = self.backbone.num_prefix_tokens
        return self.head(torch.cat([h[:, 0], h[:, k:].mean(1)], 1))


def preprocessing(model):
    """(size, mean, std) the model expects."""
    if isinstance(model, TimmViTCVS):
        return dict(size=model.size, mean=model.mean, std=model.std)
    return dict(size=(H, W), mean=MEAN, std=STD)


def build_model(arch: str, train_blocks: int = 8):
    if arch == "dinov2l":
        return CVSNet(train_blocks)
    if arch == "resnet50":
        return ResNetCVS()
    return TimmViTCVS(arch, train_blocks)


@torch.no_grad()
def predict(model, files, bs=64, workers=12):
    model.eval()
    dl = DataLoader(CVSFrames(files, **preprocessing(model)), batch_size=bs, num_workers=workers)
    out = []
    for x, _ in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(x.cuda(non_blocking=True)).float().cpu())
    return torch.cat(out).numpy()


def mean_ap(logits, Y):
    from sklearn.metrics import average_precision_score
    return [float(average_precision_score(Y[:, j], logits[:, j])) for j in range(3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--train_blocks", type=int, default=8)
    ap.add_argument("--lr_backbone", type=float, default=1e-5)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--arch", choices=["dinov2l", "resnet50", "surgenetxl", "endovit"], default="dinov2l")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    tag = f"dinov2l_b{a.train_blocks}" if a.arch == "dinov2l" else a.arch
    out = ROOT / "experiments" / "finetune" / f"{tag}_s{a.seed}"
    out.mkdir(parents=True, exist_ok=True)

    tr, va = keyframes("train"), keyframes("val")
    sel_vids = np.random.default_rng(0).choice(np.unique(va.vid), 10, replace=False)  # same as feasibility
    sel = np.isin(va.vid, sel_vids)
    sel_files = [f for f, m in zip(va.files, sel) if m]
    sel_Y = va.reference("majority")[sel]

    model = build_model(a.arch, a.train_blocks).cuda()
    head_ids = {id(p) for p in model.head.parameters()}
    params = [{"params": [p for p in model.backbone.parameters() if p.requires_grad and id(p) not in head_ids],
               "lr": a.lr_backbone},
              {"params": model.head.parameters(), "lr": a.lr_head}]
    opt = torch.optim.AdamW(params, weight_decay=0.05)
    dl = DataLoader(CVSFrames(tr.files, tr.soft.astype(np.float32), train=True, **preprocessing(model)), batch_size=a.bs,
                    shuffle=True, num_workers=16, drop_last=True, pin_memory=True, persistent_workers=True)
    steps = a.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr_backbone, a.lr_head], total_steps=steps,
                                                pct_start=0.1, anneal_strategy="cos")
    best, log = -1.0, []
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tot = 0.0
        for x, y in dl:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x).float(), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
        aps = mean_ap(predict(model, sel_files), sel_Y)
        rec = {"epoch": ep, "loss": tot / len(dl), "sel_AP": aps, "sel_mAP": float(np.mean(aps)),
               "sec": time.time() - t0}
        log.append(rec); print(json.dumps(rec), flush=True)
        if rec["sel_mAP"] > best:
            best = rec["sel_mAP"]
            torch.save({k: v for k, v in model.state_dict().items()}, out / "best.pt")
    (out / "train_log.json").write_text(json.dumps({"args": vars(a), "log": log, "best_sel_mAP": best}, indent=1))

    # logits for every 1-fps frame of val and test with the selected checkpoint
    model.load_state_dict(torch.load(out / "best.pt", map_location="cuda"))
    for split in ("val", "test"):
        df = all_frames(split)
        L = predict(model, list(df.file))
        np.savez(out / f"logits_{split}.npz", vid=df.vid.to_numpy(), frame=df.frame.to_numpy(), logits=L)
        print(split, L.shape, flush=True)
    te = keyframes("test")
    z = np.load(out / "logits_test.npz")
    idx = {(int(v), int(f)): i for i, (v, f) in enumerate(zip(z["vid"], z["frame"]))}
    Lk = z["logits"][[idx[(int(v), int(f))] for v, f in zip(te.vid, te.frame)]]
    aps = mean_ap(Lk, te.reference("majority"))
    print("TEST single-frame AP (majority):", [round(100 * x, 1) for x in aps], "mAP", round(100 * np.mean(aps), 1))
    (out / "test_ap.json").write_text(json.dumps({"AP": aps, "mAP": float(np.mean(aps))}))


if __name__ == "__main__":
    main()
