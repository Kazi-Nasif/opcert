"""Causal (online) temporal CVS models on top of per-frame embeddings.

Every prediction at second t uses frames <= t only, so the model can run live in the OR and
its outputs plug into the same certification code as the frame models.

  lstm         2-layer unidirectional LSTM (hidden 256)
  tcn          two-stage causal dilated TCN (TeCNO-style; 8 layers per stage, kernel 3,
               dilations 1..128, 64 channels; stage 2 refines stage-1 probabilities)
  transformer  4-layer causal transformer (d = 256, 4 heads), attention limited to the
               last 120 s, sinusoidal time encoding
Inputs: 'frozen' = generic DINOv2-L features (nothing trained on CVS), or 'ft' = embeddings
of the fine-tuned DINOv2-L CVS model of the same seed (two-stage training, as is standard).
Loss: BCE on the three criteria at rated keyframes (three-surgeon mean as soft target).
Training videos only; epoch chosen by keyframe mAP on the 10 selection videos.
Certification-aware training (--cert > 0): adds a loss that targets the quantity the surgery-level
certificate thresholds, the per-operation MAXIMUM of log P(CVS) over unsafe keyframes m_v (smooth
max, temperature tau).  For every training operation v in the batch and every CVS keyframe x+ in
the batch, softplus((m_v - log P(CVS | x+) + margin) / tau) is penalised: CVS frames should rank
above the most convincing unsafe frame of EVERY operation, not merely be classified on average.
Writes experiments/finetune/temporal_{model}_{feat}[_cert]_s{seed}/logits_{val,test}.npz for every
1-fps frame (same format as the frame models) and test_ap.json.

    python scripts/temporal/train_temporal.py --model tcn --feat ft --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cdss.data import keyframes                  # noqa: E402

FEAT = ROOT / "experiments" / "features"


def load_split(split, feat, seed):
    f = FEAT / (f"dinov2l_{split}.npz" if feat == "frozen" else f"ft_dinov2l_s{seed}_{split}.npz")
    z = np.load(f); vid, frame, X = z["vid"], z["frame"], z["feat"].astype(np.float32)
    kf = keyframes(split)
    soft = {(int(v), int(fr)): s for v, fr, s in zip(kf.vid, kf.frame, kf.soft)}
    maj = {(int(v), int(fr)): m for v, fr, m in zip(kf.vid, kf.frame, kf.reference("majority"))}
    seqs = []
    for v in np.unique(vid):
        ix = np.flatnonzero(vid == v); ix = ix[np.argsort(frame[ix])]
        Y = np.full((len(ix), 3), np.nan, np.float32); M = np.full((len(ix), 3), np.nan, np.float32)
        for j, i in enumerate(ix):
            k = (int(v), int(frame[i]))
            if k in soft:
                Y[j] = soft[k]; M[j] = maj[k]
        seqs.append({"vid": int(v), "frame": frame[ix], "X": X[ix], "Y": Y, "M": M})
    return seqs


class Causal1d(nn.Module):
    def __init__(self, c, d):
        super().__init__()
        self.pad = 2 * d
        self.conv = nn.Conv1d(c, c, 3, dilation=d)
        self.out = nn.Conv1d(c, c, 1)

    def forward(self, x):
        return x + self.out(F.relu(self.conv(F.pad(x, (self.pad, 0)))))


class Stage(nn.Module):
    def __init__(self, cin, c=64, layers=8):
        super().__init__()
        self.inp = nn.Conv1d(cin, c, 1)
        self.layers = nn.ModuleList([Causal1d(c, 2 ** i) for i in range(layers)])
        self.head = nn.Conv1d(c, 3, 1)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        h = self.inp(x)
        for l in self.layers:
            h = self.drop(l(h))
        return self.head(h)


class TCN(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.s1, self.s2 = Stage(d_in), Stage(3)

    def forward(self, x):                       # x: (B, T, D)
        o1 = self.s1(x.transpose(1, 2))
        o2 = self.s2(torch.sigmoid(o1))
        return [o1.transpose(1, 2), o2.transpose(1, 2)]


class LSTM(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.proj = nn.Sequential(nn.Dropout(0.3), nn.Linear(d_in, 256), nn.ReLU())
        self.rnn = nn.LSTM(256, 256, 2, batch_first=True, dropout=0.2)
        self.head = nn.Linear(256, 3)

    def forward(self, x):
        return [self.head(self.rnn(self.proj(x))[0])]


class Transformer(nn.Module):
    def __init__(self, d_in, d=256, window=120):
        super().__init__()
        self.window = window
        self.proj = nn.Sequential(nn.Dropout(0.3), nn.Linear(d_in, d))
        layer = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.2, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 4)
        self.head = nn.Linear(d, 3); self.d = d

    def forward(self, x):
        T = x.shape[1]
        pos = torch.arange(T, device=x.device, dtype=torch.float32)[:, None]
        div = torch.exp(torch.arange(0, self.d, 2, device=x.device) * (-math.log(10000.0) / self.d))
        pe = torch.zeros(T, self.d, device=x.device); pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        i = torch.arange(T, device=x.device)
        mask = (i[None, :] > i[:, None]) | (i[:, None] - i[None, :] >= self.window)   # causal, last 120 s
        return [self.head(self.enc(self.proj(x) + pe, mask=mask))]


def build(name, d_in):
    return {"lstm": LSTM, "tcn": TCN, "transformer": Transformer}[name](d_in)


def batches(seqs, mu, sd, bs, shuffle, rng):
    order = rng.permutation(len(seqs)) if shuffle else np.arange(len(seqs))
    for b in range(0, len(order), bs):
        group = [seqs[i] for i in order[b:b + bs]]
        T = max(len(s["X"]) for s in group)
        X = np.zeros((len(group), T, group[0]["X"].shape[1]), np.float32)
        Y = np.full((len(group), T, 3), np.nan, np.float32)
        for j, s in enumerate(group):
            X[j, :len(s["X"])] = (s["X"] - mu) / sd; Y[j, :len(s["X"])] = s["Y"]
        yield group, torch.from_numpy(X).cuda(), torch.from_numpy(Y).cuda()


@torch.no_grad()
def predict(model, seqs, mu, sd):
    model.eval(); out = []
    for s in seqs:                                 # one video at a time: no padding, fully causal
        x = torch.from_numpy((s["X"] - mu) / sd)[None].cuda()
        out.append(model(x)[-1][0].float().cpu().numpy())
    return out


def key_map(seqs, logits):
    L = np.concatenate([l[~np.isnan(s["M"][:, 0])] for s, l in zip(seqs, logits)])
    M = np.concatenate([s["M"][~np.isnan(s["M"][:, 0])] for s in seqs])
    return [float(average_precision_score(M[:, c], L[:, c])) for c in range(3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lstm", "tcn", "transformer"], required=True)
    ap.add_argument("--feat", choices=["frozen", "ft"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--cert", type=float, default=0.0)
    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--bs", type=int, default=4)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); rng = np.random.default_rng(a.seed)
    tr = load_split("train", a.feat, a.seed)
    va = load_split("val", a.feat, a.seed); te = load_split("test", a.feat, a.seed)
    sel_vids = set(np.random.default_rng(0).choice(np.unique(keyframes("val").vid), 10, replace=False).tolist())
    sel = [s for s in va if s["vid"] in sel_vids]
    Xall = np.concatenate([s["X"] for s in tr]); mu, sd = Xall.mean(0), Xall.std(0) + 1e-6
    model = build(a.model, Xall.shape[1]).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3 if a.model != "transformer" else 3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    tag = f"temporal_{a.model}_{a.feat}" + ("_cert" if a.cert > 0 else "")
    out = ROOT / "experiments" / "finetune" / f"{tag}_s{a.seed}"; out.mkdir(parents=True, exist_ok=True)
    best, log = -1.0, []
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tot = 0.0
        for group, X, Y in batches(tr, mu, sd, a.bs, True, rng):
            m = ~torch.isnan(Y)
            outs = model(X)
            loss = sum(F.binary_cross_entropy_with_logits(o[m], Y[m]) for o in outs)
            if a.cert > 0:
                s = F.logsigmoid(outs[-1]).sum(-1)                         # log P(CVS) per frame, (B, T)
                Mj = torch.full(Y.shape[:2], float("nan"), device=Y.device)
                for j, g in enumerate(group):                              # majority reference per keyframe
                    Mj[j, :len(g["M"])] = torch.from_numpy(np.where(np.isnan(g["M"][:, 0]), np.nan,
                                                                    np.all(g["M"] > 0.5, 1).astype(float)))
                pos = s[Mj == 1]
                mv = [a.tau * torch.logsumexp(s[j][Mj[j] == 0] / a.tau, 0) for j in range(len(group))
                      if (Mj[j] == 0).any()]
                if len(pos) and mv:
                    mv = torch.stack(mv)
                    loss = loss + a.cert * F.softplus((mv[:, None] - pos[None, :] + a.margin) / a.tau).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item()
        sched.step()
        aps = key_map(sel, predict(model, sel, mu, sd))
        log.append({"epoch": ep, "loss": tot, "sel_mAP": float(np.mean(aps)), "sec": time.time() - t0})
        if log[-1]["sel_mAP"] > best:
            best = log[-1]["sel_mAP"]; torch.save(model.state_dict(), out / "best.pt")
    model.load_state_dict(torch.load(out / "best.pt"))
    for split, seqs in (("val", va), ("test", te)):
        L = predict(model, seqs, mu, sd)
        np.savez(out / f"logits_{split}.npz", vid=np.concatenate([np.full(len(s["X"]), s["vid"]) for s in seqs]),
                 frame=np.concatenate([s["frame"] for s in seqs]), logits=np.concatenate(L).astype(np.float32))
    aps = key_map(te, predict(model, te, mu, sd))
    (out / "test_ap.json").write_text(json.dumps({"AP": aps, "mAP": float(np.mean(aps))}))
    (out / "train_log.json").write_text(json.dumps({"args": vars(a), "log": log, "best_sel_mAP": best}))
    print(tag, a.seed, "sel", round(best, 4), "TEST mAP", round(float(np.mean(aps)), 4), flush=True)


if __name__ == "__main__":
    main()
