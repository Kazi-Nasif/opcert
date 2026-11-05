"""Review: sensitivity of the anatomy certificates to the detection pipeline.

Task: overlay "cystic duct" / "cystic artery" on Endoscapes-BBox201 keyframes; an overlay is
wrong if its region is not matched (IoU) to that structure.  Variants, one factor at a time
(base: all exported regions, IoU 0.5, one candidate per structure = argmax_r P_r(t)):
  min_fg      keep regions with foreground probability >= 0.3 / 0.5
  iou         truth matching at IoU 0.3 / 0.7
  assign      'all'  overlay every region whose P_r(t) exceeds the threshold
              'map'  candidate = region labelled t by the K-constrained point world
  drop        delete 10 / 20 / 30 % of regions at random -- at test only (deployment
              degradation) or everywhere (a weaker detector, recalibrated)
  noise       mix class probabilities with Dirichlet(1) noise, eps = 0.1 / 0.3, test only / both
  jitter      shift/scale predicted boxes by N(0, s) of their size, s = 0.1 / 0.2 (truth re-matched)
  K           point world with / without "at most one region per structure"
Methods: point world, frame-level conformal (frames pooled; CRC for the loss "any wrong
overlay in the frame"), surgery-level (ours).  200 video splits x 3 detector seeds.

    python scripts/review/anatomy_robustness.py
"""
from __future__ import annotations

import json
import sys
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ALPHAS, ROOT, order_stat, save, sel_vids, summary  # noqa: E402
from cdss import anatomy as A                                        # noqa: E402
from evaluate_anatomy import fit_T                                   # noqa: E402

TARGETS = A.CLIP_TARGETS


def raw_frames(seed):
    det = ROOT / "experiments" / "detector" / f"frcnn_s{seed}"
    out = []
    for split in ("val", "test"):
        reg = json.loads((det / f"regions_{split}.json").read_text())
        names = {str(i["id"]): i["file_name"] for i in
                 json.loads((ROOT / "data" / "endoscapes" / split / "annotation_coco.json").read_text())["images"]}
        for iid, d in reg.items():
            vid, fr = (int(x) for x in Path(names[iid]).stem.split("_"))
            out.append({"vid": vid, "frame": fr, "B": np.array(d["boxes"]).reshape(-1, 4),
                        "P": np.array(d["probs"]).reshape(-1, 7), "G": np.array(d["gt_boxes"]).reshape(-1, 4),
                        "GL": np.array(d["gt_labels"], int), "key": f"{vid}_{fr}"})
    return out


def variant(frames, T, name, level, apply_cal, seed):
    """Returns per-frame dicts with tempered probs Pt, truth y, K-flag for the point world,
    and whether the degradation applies (cal frames only degraded if apply_cal)."""
    out = []
    for f in frames:
        rng = np.random.default_rng(zlib.crc32(f"{f['key']}|{name}|{level}|{seed}".encode()))
        B, P = f["B"].copy(), f["P"].copy(); thr = 0.5
        if name == "min_fg":
            keep = (1 - P[:, 0]) >= level; B, P = B[keep], P[keep]
        if name == "iou":
            thr = level
        if name == "jitter" and len(B):
            wh = np.stack([B[:, 2] - B[:, 0], B[:, 3] - B[:, 1]], 1)
            c = (B[:, :2] + B[:, 2:]) / 2 + rng.normal(0, level, (len(B), 2)) * wh
            wh = wh * np.exp(rng.normal(0, level, (len(B), 2)))
            B = np.concatenate([c - wh / 2, c + wh / 2], 1)
        g = {"vid": f["vid"], "B": B, "P": P, "y": A.region_truth(B, f["G"], f["GL"], thr), "thr": thr}
        # degradations that may apply to test only
        Bd, Pd = B, P
        if name == "drop" and len(B):
            keep = rng.random(len(B)) >= level; Bd, Pd = B[keep], P[keep]
        if name == "noise" and len(P):
            Pd = (1 - level) * P + level * rng.dirichlet(np.ones(P.shape[1]), len(P))
        g["Bd"], g["Pd"] = Bd, Pd
        g["yd"] = A.region_truth(Bd, f["G"], f["GL"], thr) if name == "drop" else g["y"]
        out.append(g)
    def finalize(g, use_degraded):
        B, P, y = (g["Bd"], g["Pd"], g["yd"]) if use_degraded else (g["B"], g["P"], g["y"])
        return {"vid": g["vid"], "Pt": A.temper(P, T) if len(P) else P, "y": y}
    return [(finalize(g, apply_cal), finalize(g, True)) for g in out]


def candidates(f, assign):
    Pt, y = f["Pt"], f["y"]
    cands = {t: [] for t in TARGETS}
    if len(Pt) == 0:
        return cands
    if assign == "map":
        yy = A.map_world(Pt, True)
    for t in TARGETS:
        if assign == "argmax":
            r = int(np.argmax(Pt[:, t])); cands[t] = [(Pt[r, t], y[r] == t)]
        elif assign == "all":
            cands[t] = [(Pt[r, t], y[r] == t) for r in range(len(Pt)) if Pt[r, t] > 1e-3]
        else:
            rs = np.flatnonzero(yy == t)
            cands[t] = [(Pt[r, t], y[r] == t) for r in rs]
    return cands


def prep(frames, assign, use_K):
    for f in frames:
        f["c"] = candidates(f, assign)
        wrong = [c for t in TARGETS for c, ok in f["c"][t] if not ok]
        f["mw"] = max(wrong) if wrong else -np.inf
        f["certifiable"] = sum(int((f["y"] == t).any()) for t in TARGETS)
        # projected frame-level certificate of Prop. 1 (the frame-level method of main Table 2)
        f["pcand"], f["pprob"] = A.projected_bits(f["Pt"], None)
        f["pcorrect"] = A.projected_truth(f["y"], f["pcand"], None)
        f["pscore"] = A.projected_score(f["pprob"], f["pcorrect"])
        if len(f["Pt"]):
            yy = A.map_world(f["Pt"], use_K)
            f["map_ok"] = [f["y"][r] == t for t in TARGETS for r in range(len(yy)) if yy[r] == t]
        else:
            f["map_ok"] = []
    return frames


def score(frames, lam):
    by_v, n_over, n_ok, cert = {}, 0, 0, 0
    for f in frames:
        bad = False
        for t in TARGETS:
            for c, ok in f["c"][t]:
                if c > lam:
                    n_over += 1; n_ok += ok; bad |= not ok
        cert += f["certifiable"]
        by_v[f["vid"]] = by_v.get(f["vid"], False) or bad
    return {"ops_err": float(np.mean(list(by_v.values()))), "precision": n_ok / max(n_over, 1),
            "recall": n_ok / max(cert, 1)}


def score_projected(frames, q):
    by_v, n_over, n_ok, cert = {}, 0, 0, 0
    for f in frames:
        ov = A.projected_certify(f["pcand"], f["pprob"], q)
        bad = False
        for t, r in ov.items():
            if r is None:
                continue
            ok = bool(f["pcorrect"][t]); n_over += 1; n_ok += ok; bad |= not ok
        cert += f["certifiable"]
        by_v[f["vid"]] = by_v.get(f["vid"], False) or bad
    return {"ops_err": float(np.mean(list(by_v.values()))), "precision": n_ok / max(n_over, 1),
            "recall": n_ok / max(cert, 1)}


def score_map(frames):
    by_v, n_over, n_ok, cert = {}, 0, 0, 0
    for f in frames:
        n_over += len(f["map_ok"]); n_ok += sum(f["map_ok"]); cert += f["certifiable"]
        by_v[f["vid"]] = by_v.get(f["vid"], False) or (not all(f["map_ok"]))
    return {"ops_err": float(np.mean(list(by_v.values()))), "precision": n_ok / max(n_over, 1),
            "recall": n_ok / max(cert, 1)}


VARIANTS = ([("base", 0, True, "argmax", True)]
            + [("min_fg", l, True, "argmax", True) for l in (0.3, 0.5)]
            + [("iou", l, True, "argmax", True) for l in (0.3, 0.7)]
            + [("assign", 0, True, a, True) for a in ("all", "map")]
            + [("K", 0, True, "argmax", False)]
            + [("drop", l, c, "argmax", True) for l in (0.1, 0.2, 0.3) for c in (False, True)]
            + [("noise", l, c, "argmax", True) for l in (0.1, 0.3) for c in (False, True)]
            + [("jitter", l, True, "argmax", True) for l in (0.1, 0.2)])


def run_variant(args):
    seed, frames, T, (name, level, apply_cal, assign, use_K) = args
    pairs = variant(frames, T, name, level, apply_cal, seed)
    cal_all = prep([p[0] for p in pairs], assign, use_K); tst_all = prep([p[1] for p in pairs], assign, use_K)
    vids = np.array([f["vid"] for f in cal_all]); uv = np.unique(vids)
    rows = []
    base = {"seed": seed, "variant": name, "level": level, "degrade_cal": apply_cal, "assign": assign, "K": use_K}
    for rep in range(200):
        cv = set(np.random.default_rng(rep).choice(uv, len(uv) // 2, replace=False).tolist())
        cal = [f for f in cal_all if f["vid"] in cv]; tst = [f for f in tst_all if f["vid"] not in cv]
        rows.append({**base, "rep": rep, "alpha": None, "method": "Point world", **score_map(tst)})
        for alpha in ALPHAS:
            lam_f = order_stat([f["mw"] for f in cal], alpha)
            m = {}
            for f in cal:
                m[f["vid"]] = max(m.get(f["vid"], -np.inf), f["mw"])
            lam_s = order_stat(list(m.values()), alpha)
            rows.append({**base, "rep": rep, "alpha": alpha, "method": "Frame-level (threshold)", **score(tst, lam_f)})
            q = order_stat([f["pscore"] for f in cal], alpha)
            rows.append({**base, "rep": rep, "alpha": alpha, "method": "Frame-level (projected, Tab. 2)",
                         **score_projected(tst, q)})
            rows.append({**base, "rep": rep, "alpha": alpha, "method": "Surgery-level (ours)", **score(tst, lam_s)})
    return rows


def main():
    sv = set(sel_vids().tolist())
    jobs = []
    for seed in (0, 1, 2):
        fr = raw_frames(seed)
        sel = [f for f in fr if f["vid"] in sv]
        Ls = torch.tensor(np.concatenate([np.log(np.clip(f["P"], A.EPS, 1)) for f in sel if len(f["P"])]))
        ys = torch.tensor(np.concatenate([A.region_truth(f["B"], f["G"], f["GL"]) for f in sel if len(f["P"])]))
        T = fit_T(lambda T: torch.nn.functional.cross_entropy(Ls / T, ys))
        pool_frames = [f for f in fr if f["vid"] not in sv]
        jobs += [(seed, pool_frames, T, v) for v in VARIANTS]
    rows = []
    with Pool(20) as pool:
        for r in pool.imap_unordered(run_variant, jobs):
            rows += r
            print(r[0]["seed"], r[0]["variant"], r[0]["level"], r[0]["degrade_cal"], flush=True)
    df = save(rows, "anatomy_robustness")
    summary(df, ["variant", "level", "degrade_cal", "assign", "K", "method", "alpha"], ["ops_err", "precision", "recall"])


if __name__ == "__main__":
    main()
