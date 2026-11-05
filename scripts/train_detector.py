"""Anatomy detector for Endoscapes-BBox201: torchvision Faster R-CNN R50-FPN v2 (COCO init).

Trained on the official train split only, fixed schedule (no selection on val, so the
conformal calibration videos stay untouched). Afterwards, every val/test detection
keyframe is exported as a set of candidate *regions*, each with its full class
distribution (6 structures + background), which is what the world model needs: the
region's identity is uncertain, not just its top-1 label.

    python scripts/train_detector.py --epochs 30
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "endoscapes"
OUT = ROOT / "experiments" / "detector"
CLASSES = ["cystic_plate", "calot_triangle", "cystic_artery", "cystic_duct", "gallbladder", "tool"]  # COCO ids 1..6


def load_split(split):
    d = json.load(open(DATA / split / "annotation_coco.json"))
    anns = {}
    for a in d["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    imgs = [i for i in d["images"] if i.get("is_det_keyframe")]
    return imgs, anns


class DetFrames(Dataset):
    def __init__(self, split, train=False):
        self.split, self.train = split, train
        self.imgs, self.anns = load_split(split)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        info = self.imgs[i]
        im = Image.open(DATA / self.split / info["file_name"]).convert("RGB")
        x = torchvision.transforms.functional.to_tensor(im)
        boxes, labels = [], []
        for a in self.anns.get(info["id"], []):
            bx, by, bw, bh = a["bbox"]
            if bw > 1 and bh > 1:
                boxes.append([bx, by, bx + bw, by + bh]); labels.append(a["category_id"])
        t = {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
             "labels": torch.tensor(labels, dtype=torch.int64)}
        if self.train:
            x = torchvision.transforms.ColorJitter(0.3, 0.3, 0.2, 0.02)(x)
        return x, t, info["id"]


def collate(b):
    return tuple(zip(*b))


def build(num_classes=7):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    m = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, num_classes)
    return m


@torch.no_grad()
def regions(model, x, max_regions=12, nms_iou=0.5, min_fg=0.05):
    """Class-agnostic candidate regions with full softmax over [background, 6 classes]."""
    model.eval()
    images, _ = model.transform([x])
    feats = model.backbone(images.tensors)
    props, _ = model.rpn(images, feats)
    rh = model.roi_heads
    bf = rh.box_head(rh.box_roi_pool(feats, props, images.image_sizes))
    logits, deltas = rh.box_predictor(bf)
    P = torch.softmax(logits, -1)                                     # (n, 7), col 0 = background
    fg = 1 - P[:, 0]
    k = P[:, 1:].argmax(1) + 1
    boxes = rh.box_coder.decode(deltas, props).reshape(-1, P.shape[1], 4)[torch.arange(len(k)), k]
    boxes = torchvision.ops.clip_boxes_to_image(boxes, images.image_sizes[0])
    keep = torchvision.ops.nms(boxes, fg, nms_iou)
    keep = keep[fg[keep] >= min_fg][:max_regions]
    sx = x.shape[2] / images.image_sizes[0][1]; sy = x.shape[1] / images.image_sizes[0][0]
    b = boxes[keep] * torch.tensor([sx, sy, sx, sy], device=boxes.device)
    return b.cpu().numpy(), P[keep].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    out = OUT / f"frcnn_s{a.seed}"; out.mkdir(parents=True, exist_ok=True)
    model = build().cuda()
    dl = DataLoader(DetFrames("train", train=True), batch_size=8, shuffle=True, num_workers=8,
                    collate_fn=collate, drop_last=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.02, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(0.7 * a.epochs * len(dl)), int(0.9 * a.epochs * len(dl))], 0.1)
    warm = 200
    step = 0
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tot = 0
        for xs, ts, _ in dl:
            xs = [x.cuda() for x in xs]; ts = [{k: v.cuda() for k, v in t.items()} for t in ts]
            if step < warm:
                for g in opt.param_groups:
                    g["lr"] = 0.02 * (step + 1) / warm
            loss = sum(model(xs, ts).values())
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 10); opt.step(); sched.step()
            tot += loss.item(); step += 1
        print(json.dumps({"epoch": ep, "loss": tot / len(dl), "sec": round(time.time() - t0, 1)}), flush=True)
    torch.save(model.state_dict(), out / "model.pt")

    # standard detection metric (sanity vs. the Endoscapes benchmark), then region export
    from torchmetrics.detection import MeanAveragePrecision
    for split in ("val", "test"):
        ds = DetFrames(split)
        metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
        export = {}
        model.eval()
        for i in range(len(ds)):
            x, t, iid = ds[i]
            with torch.no_grad():
                pred = model([x.cuda()])[0]
            metric.update([{k: v.cpu() for k, v in pred.items()}], [t])
            B, P = regions(model, x.cuda())
            export[str(iid)] = {"boxes": B.round(1).tolist(), "probs": P.round(5).tolist(),
                                "gt_boxes": t["boxes"].tolist(), "gt_labels": t["labels"].tolist()}
        m = metric.compute()
        res = {"mAP": float(m["map"]), "mAP50": float(m["map_50"]),
               "per_class_AP": {CLASSES[int(c) - 1]: float(v) for c, v in zip(m["classes"], m["map_per_class"])}}
        print(split, json.dumps(res), flush=True)
        (out / f"det_metrics_{split}.json").write_text(json.dumps(res, indent=1))
        (out / f"regions_{split}.json").write_text(json.dumps(export))


if __name__ == "__main__":
    main()
