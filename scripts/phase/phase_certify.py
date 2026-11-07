"""Second task (review R2, P3): certified surgical-phase statements on Cholec80.

Perception: a causal two-stage TCN (TeCNO-style; every output at second t uses frames <= t only)
on frozen, generic DINOv2-L features (scripts/phase/extract_cholec80.py), 7 phases, cross-entropy.
Videos 1-32 train, 33-40 select the epoch and fit the temperature, 41-80 form the
calibration/evaluation pool (200 video-level halvings, 20/20), 3 seeds.  Every 1-fps frame is scored.

Statements, each issued iff its score exceeds a calibrated threshold:
  (A) "clipping has begun"  score P(phase >= ClippingCutting); wrong on every frame before the first
      ClippingCutting frame.  Per-operation statistic: the maximum score before that onset.
  (B) "the phase is k"      the displayed phase label (argmax), score = its probability; wrong when the
      label is wrong.  Per-operation statistic: the most confident wrong label.
Methods: point world (A: score > 0.5; B: always display argmax), pooled frame-level CRC and
Learn-then-Test (delta 0.1) at alpha, and the surgery-level certificate (ours).
Metrics: % operations with >= 1 wrong statement, frame error rate, sensitivity / coverage, delay from
the true clipping onset to the first (correct) confirmation, accuracy of displayed labels.

    python scripts/phase/phase_certify.py --train      # trains 3 seeds, writes per-frame outputs
    python scripts/phase/phase_certify.py              # certification only
"""
from __future__ import annotations

import argparse
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "review"))
from cdss import riskcontrol as RC          # noqa: E402
from common import order_stat               # noqa: E402

OUT = ROOT / "experiments" / "phase"
TRAIN, SELECT, POOL = range(1, 33), range(33, 41), range(41, 81)
CLIP = 2
ALPHAS = (0.05, 0.10, 0.20)
DELTA = 0.1
NC = 7


class Causal1d(nn.Module):
    def __init__(self, c, d):
        super().__init__()
        self.pad = 2 * d; self.conv = nn.Conv1d(c, c, 3, dilation=d); self.out = nn.Conv1d(c, c, 1)

    def forward(self, x):
        return x + self.out(F.relu(self.conv(F.pad(x, (self.pad, 0)))))


class Stage(nn.Module):
    def __init__(self, cin, c=64, layers=10):
        super().__init__()
        self.inp = nn.Conv1d(cin, c, 1)
        self.layers = nn.ModuleList([Causal1d(c, 2 ** i) for i in range(layers)])
        self.head = nn.Conv1d(c, NC, 1); self.drop = nn.Dropout(0.3)

    def forward(self, x):
        h = self.inp(x)
        for l in self.layers:
            h = self.drop(l(h))
        return self.head(h)


class TCN(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.s1, self.s2 = Stage(d_in), Stage(NC)

    def forward(self, x):                      # (1, T, D) -> list of (1, T, NC)
        o1 = self.s1(x.transpose(1, 2)); o2 = self.s2(F.softmax(o1, 1))
        return [o1.transpose(1, 2), o2.transpose(1, 2)]


def load():
    z = np.load(OUT / "dinov2l_cholec80.npz")
    seqs = {}
    for v in np.unique(z["vid"]):
        m = np.flatnonzero(z["vid"] == v); m = m[np.argsort(z["frame"][m])]
        seqs[int(v)] = {"X": z["feat"][m].astype(np.float32), "y": z["phase"][m].astype(int), "frame": z["frame"][m]}
    return seqs


def train(seed, seqs, epochs=50):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    Xtr = np.concatenate([seqs[v]["X"] for v in TRAIN]); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    model = TCN(Xtr.shape[1]).cuda()
    opt = torch.optim.AdamW(model.parameters(), 5e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    tens = lambda v: torch.from_numpy((seqs[v]["X"] - mu) / sd)[None].cuda()
    best, best_acc = None, -1
    for ep in range(epochs):
        model.train()
        for v in rng.permutation(list(TRAIN)):
            y = torch.from_numpy(seqs[v]["y"]).cuda()
            loss = sum(F.cross_entropy(o[0], y) for o in model(tens(v)))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            acc = np.mean(np.concatenate([model(tens(v))[-1][0].argmax(1).cpu().numpy() == seqs[v]["y"] for v in SELECT]))
        if acc > best_acc:
            best_acc, best = acc, {k: t.detach().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best); model.eval()
    with torch.no_grad():
        logits = {v: model(tens(v))[-1][0].float().cpu().numpy() for v in seqs}
    Ls = torch.tensor(np.concatenate([logits[v] for v in SELECT])); Ys = torch.tensor(np.concatenate([seqs[v]["y"] for v in SELECT]))
    logT = torch.zeros(1, requires_grad=True); o = torch.optim.LBFGS([logT], lr=0.1, max_iter=200)

    def closure():
        o.zero_grad(); l = F.cross_entropy(Ls / logT.exp(), Ys); l.backward(); return l
    o.step(closure)
    T = float(logT.exp())
    np.savez(OUT / f"phase_tcn_s{seed}.npz", T=T, sel_acc=best_acc,
             **{f"v{v}": logits[v] for v in seqs})
    print(f"seed {seed}: selection accuracy {best_acc:.3f}, T {T:.2f}", flush=True)


def ops_for(seed, seqs, w):
    z = np.load(OUT / f"phase_tcn_s{seed}.npz"); T = float(z["T"])
    ops = []
    for v in POOL:
        L = z[f"v{v}"] / T
        L = np.stack([L[max(0, i - w + 1):i + 1].mean(0) for i in range(len(L))]) if w > 1 else L
        P = np.exp(L - L.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
        y = seqs[v]["y"]
        on = np.flatnonzero(y == CLIP)
        onset = on[0] if len(on) else len(y)
        lab = P.argmax(1)
        ops.append({"vid": v, "pA": P[:, CLIP:].sum(1), "bA": np.arange(len(y)) >= onset, "onset": onset,
                    "sB": P.max(1), "okB": lab == y})
    return ops


def eval_A(tst, lam):
    err = [((o["pA"] > lam) & ~o["bA"]).any() for o in tst]
    fr = sum(((o["pA"] > lam) & ~o["bA"]).sum() for o in tst) / sum((~o["bA"]).sum() for o in tst)
    tp = sum(((o["pA"] > lam) & o["bA"]).sum() for o in tst); npos = sum(o["bA"].sum() for o in tst)
    delay, conf = [], 0
    for o in tst:
        hit = np.flatnonzero((o["pA"] > lam) & o["bA"])
        if len(hit):
            conf += 1; delay.append(hit[0] - o["onset"])
    return {"ops_err": float(np.mean(err)), "frame_err": fr, "sens": tp / max(npos, 1),
            "ops_confirmed": conf / len(tst), "delay_med_s": float(np.median(delay)) if delay else np.nan}


def eval_B(tst, lam):
    iss = [o["sB"] > lam for o in tst]
    wrong = [i & ~o["okB"] for i, o in zip(iss, tst)]
    n = sum(len(i) for i in iss); ni = sum(i.sum() for i in iss); nw = sum(w.sum() for w in wrong)
    return {"ops_err": float(np.mean([w.any() for w in wrong])), "frame_err": nw / n, "coverage": ni / n,
            "issued_acc": 1 - nw / ni if ni else np.nan}


def job(args):
    seed, w, ops, rep, ci, ti = args
    cal, tst = [ops[i] for i in ci], [ops[i] for i in ti]
    rows = []
    base = {"seed": seed, "w": w, "rep": rep}
    rows.append({**base, "task": "A", "alpha": np.nan, "method": "point world", **eval_A(tst, 0.5)})
    rows.append({**base, "task": "B", "alpha": np.nan, "method": "point world", **eval_B(tst, -np.inf)})
    negA = np.concatenate([o["pA"][~o["bA"]] for o in cal])
    pA = np.concatenate([o["pA"] for o in cal]); yA = np.concatenate([o["bA"] for o in cal])
    gA = RC.threshold_grid(pA); LA = ((pA[:, None] > gA[None]) & ~yA[:, None])[~yA].astype(float)
    sB = np.concatenate([o["sB"] for o in cal]); okB = np.concatenate([o["okB"] for o in cal])
    gB = RC.threshold_grid(sB); LB = ((sB[:, None] > gB[None]) & ~okB[:, None]).astype(float)
    mA = [o["pA"][~o["bA"]].max() for o in cal if (~o["bA"]).any()]
    mB = [o["sB"][~o["okB"]].max() for o in cal if (~o["okB"]).any()]
    for a in ALPHAS:
        g = RC.ltt_index(LA, a, DELTA)
        for name, lam in (("CRC frame-level", order_stat(negA, a)), ("LTT frame-level", math.inf if g is None else gA[g]),
                          ("surgery-level (ours)", order_stat(mA, a))):
            rows.append({**base, "task": "A", "alpha": a, "method": name, **eval_A(tst, lam)})
        gc = RC.crc_index(LB, a); gl = RC.ltt_index(LB, a, DELTA)
        for name, lam in (("CRC frame-level", math.inf if gc is None else gB[gc]), ("LTT frame-level", math.inf if gl is None else gB[gl]),
                          ("surgery-level (ours)", order_stat(mB, a))):
            rows.append({**base, "task": "B", "alpha": a, "method": name, **eval_B(tst, lam)})
    return rows


def halves(n, reps=200):
    for rep in range(reps):
        ci = np.sort(np.random.default_rng(rep).choice(n, n // 2, replace=False))
        yield rep, ci, np.setdiff1d(np.arange(n), ci)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--train", action="store_true")
    a = ap.parse_args()
    seqs = load()
    if a.train:
        for seed in (0, 1, 2):
            train(seed, seqs)
    perc = []
    for seed in (0, 1, 2):
        z = np.load(OUT / f"phase_tcn_s{seed}.npz")
        acc = np.mean(np.concatenate([z[f"v{v}"].argmax(1) == seqs[v]["y"] for v in POOL]))
        vacc = np.mean([np.mean(z[f"v{v}"].argmax(1) == seqs[v]["y"]) for v in POOL])
        perc.append({"seed": seed, "frame_acc": acc, "video_acc": vacc})
    print(pd.DataFrame(perc).describe().round(4).to_string(), flush=True)
    rows = []
    for seed in (0, 1, 2):
        for w in (1, 10):
            ops = ops_for(seed, seqs, w)
            with Pool(20) as pool:
                for r in pool.imap_unordered(job, [(seed, w, ops, rep, ci, ti) for rep, ci, ti in halves(len(ops))], chunksize=4):
                    rows += r
    df = pd.DataFrame(rows); df.to_csv(OUT / "phase_certify.csv", index=False); pd.DataFrame(perc).to_csv(OUT / "phase_perception.csv", index=False)
    df["alpha"] = df.alpha.fillna(-1)
    pd.set_option("display.width", 250)
    for t, cols in (("A", ["ops_err", "frame_err", "sens", "ops_confirmed", "delay_med_s"]), ("B", ["ops_err", "frame_err", "coverage", "issued_acc"])):
        print(df[df.task == t].groupby(["w", "alpha", "method"])[cols].mean().round(4).to_string())


if __name__ == "__main__":
    main()
