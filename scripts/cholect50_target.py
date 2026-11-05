"""CholecT50: which structure is being clipped or divided?

Frames: every 1-fps frame in which a clipper applies a clip, or scissors cut, per the
CholecT50 triplet labels (Nwoye et al.; CC BY-NC-SA 4.0). Label: the action's target,
grouped as cystic duct / cystic artery / other.
Videos: 20 training videos (the 6 that also appear in Endoscapes + 14 at random, fixed
seed) and a 30-video evaluation pool that the classifier never sees; 4 of the training
videos are held out for epoch selection.
Model: DINOv2-L, last 8 blocks + linear head, 3-way cross-entropy, same recipe as the CVS
model. Writes softmax probabilities for every eval-pool frame (with video, frame, label).

    python scripts/cholect50_target.py --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from finetune import CVSFrames, CVSNet              # noqa: E402

C50 = ROOT / "data" / "cholect50" / "CholecT50"
OVERLAP_ENDOSCAPES = {66, 68, 70, 96, 103, 110}
CLASSES = ["cystic_duct", "cystic_artery", "other"]


def frames():
    rows = []
    for f in sorted((C50 / "labels").glob("VID*.json")):
        d = json.loads(f.read_text()); v = int(d["video"])
        for fr, anns in d["annotations"].items():
            tg = [a[8] for a in anns if (a[1] == 4 and a[7] == 4) or (a[1] == 3 and a[7] == 5)]
            if not tg:
                continue
            y = 0 if 2 in tg else 1 if 3 in tg else 2        # duct > artery > other if several
            rows.append((v, int(fr), y))
    return rows


def split(seed_split: int = 0):
    vids = sorted({r[0] for r in frames()})
    rest = [v for v in vids if v not in OVERLAP_ENDOSCAPES]
    rng = np.random.default_rng(seed_split)
    extra = list(rng.choice(rest, 14, replace=False))
    train = sorted(set(OVERLAP_ENDOSCAPES) | set(int(x) for x in extra))
    pool = sorted(v for v in vids if v not in train)
    sel = list(rng.choice(train, 4, replace=False))
    return train, sel, pool


class Net(CVSNet):
    def __init__(self):
        super().__init__(8)
        d = self.backbone.config.hidden_size
        self.head = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(2 * d, 3))


@torch.no_grad()
def predict(model, files, bs=64):
    model.eval()
    dl = DataLoader(CVSFrames(files), batch_size=bs, num_workers=12)
    out = []
    for x, _ in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.softmax(model(x.cuda()).float(), 1).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    rows = frames(); train, sel, pool = split()
    path = lambda v, f: C50 / "videos" / f"VID{v:02d}" / f"{f:06d}.png"
    tr = [r for r in rows if r[0] in train and r[0] not in sel]
    se = [r for r in rows if r[0] in sel]
    ev = [r for r in rows if r[0] in pool]
    out = ROOT / "experiments" / "cholect50" / f"target_dinov2l_s{a.seed}"; out.mkdir(parents=True, exist_ok=True)
    (out / "split.json").write_text(json.dumps({"train": train, "selection": [int(x) for x in sel], "eval_pool": pool,
                                                "n_frames": {"train": len(tr), "selection": len(se), "eval": len(ev)}}))
    model = Net().cuda()
    params = [{"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": 1e-5},
              {"params": model.head.parameters(), "lr": 1e-3}]
    opt = torch.optim.AdamW(params, weight_decay=0.05)
    Y = np.eye(3, dtype=np.float32)[[r[2] for r in tr]]
    dl = DataLoader(CVSFrames([path(v, f) for v, f, _ in tr], Y, train=True), batch_size=32, shuffle=True,
                    num_workers=16, drop_last=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-5, 1e-3], total_steps=a.epochs * len(dl), pct_start=0.1)
    best, log = -1, []
    for ep in range(a.epochs):
        model.train(); t0 = time.time()
        for x, y in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = torch.nn.functional.cross_entropy(model(x.cuda()).float(), y.cuda().argmax(1))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        P = predict(model, [path(v, f) for v, f, _ in se])
        acc = float((P.argmax(1) == np.array([r[2] for r in se])).mean())
        log.append({"epoch": ep, "sel_acc": acc, "sec": time.time() - t0}); print(json.dumps(log[-1]), flush=True)
        if acc > best:
            best = acc; torch.save(model.state_dict(), out / "best.pt")
    model.load_state_dict(torch.load(out / "best.pt"))
    P = predict(model, [path(v, f) for v, f, _ in ev])
    np.savez(out / "probs_eval.npz", video=np.array([r[0] for r in ev]), frame=np.array([r[1] for r in ev]),
             y=np.array([r[2] for r in ev]), probs=P)
    acc = float((P.argmax(1) == np.array([r[2] for r in ev])).mean())
    (out / "train_log.json").write_text(json.dumps({"log": log, "best_sel_acc": best, "eval_acc": acc}))
    print("EVAL accuracy", round(acc, 4), "frames", len(ev))


if __name__ == "__main__":
    main()
