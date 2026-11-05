"""Surgery-level certification for structure identification and clip permission.

Overlay rule (all methods): on each anatomy keyframe, show target t (cystic duct / artery)
on its candidate region r_t = argmax_r P_r(t) iff its confidence exceeds a threshold
(for clip: identity confidence AND P(CVS) each exceed their thresholds).

Methods
  MAP (point world)       overlay the K-constrained MAP labelling (clip: and CVS MAP)
  Frame-level ours        projected certificate, frame-level guarantee, frames pooled
  LTT (frames pooled)     frame-level risk control, frames pooled
  Surgery-level      threshold = ceil((V+1)(1-a))-th order statistic of
                          m_v = max over the calibration video's frames/targets of the
                          confidence of a WRONG overlay  ->  P(any wrong overlay in a new
                          operation) <= a  (exact, videos exchangeable)
  Clip, surgery-level     identity at a/2 (as above) and CVS at a/2 (video max of P(CVS)
                          over non-CVS CVS-rated keyframes, calibrated on all of them)
                          -> P(any wrong clip permission in a new operation) <= a

Evaluation: pool the official val videos (minus the 10 selection videos) and test videos,
draw 200 random video-level halves (calibration / evaluation); report means and the
2.5-97.5 % range over splits.

    python scripts/surgery_level.py --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from cdss import anatomy as A                       # noqa: E402
from cdss import riskcontrol as RC                  # noqa: E402
from cdss.data import keyframes                     # noqa: E402
from evaluate_anatomy import fit_T, load_frames     # noqa: E402

ALPHAS = (0.05, 0.10, 0.20)


def order_stat_threshold(m: np.ndarray, alpha: float) -> float:
    m = np.sort(m); k = math.ceil((len(m) + 1) * (1 - alpha))
    return math.inf if len(m) == 0 or k > len(m) else float(m[k - 1])


def prepare(seed: int, sel_vids):
    det = ROOT / "experiments" / "detector" / f"frcnn_s{seed}"
    cvs = ROOT / "experiments" / "finetune" / f"dinov2l_b8_s{seed}"
    val, test = load_frames(det, cvs, "val"), load_frames(det, cvs, "test")
    sel = [f for f in val if f["vid"] in sel_vids]
    Ls = torch.tensor(np.concatenate([np.log(np.clip(f["P"], A.EPS, 1)) for f in sel if len(f["P"])]))
    ys = torch.tensor(np.concatenate([f["y"] for f in sel if len(f["P"])]))
    T_det = fit_T(lambda T: torch.nn.functional.cross_entropy(Ls / T, ys))
    L3 = torch.tensor(np.stack([f["logit3"] for f in sel]))
    Y3 = torch.tensor(np.stack([f["cvs_true"] for f in sel]), dtype=torch.float32)
    T_cvs = fit_T(lambda T: torch.nn.functional.binary_cross_entropy_with_logits(L3 / T, Y3))
    frames = [f for f in val if f["vid"] not in sel_vids] + test
    for f in frames:
        f["Pt"] = A.temper(f["P"], T_det) if len(f["P"]) else f["P"]
        f["pcvs"] = float(np.prod(1 / (1 + np.exp(-f["logit3"] / T_cvs))))
        f["pc3"] = 1 / (1 + np.exp(-f["logit3"] / T_cvs))
        cand, prob = A.projected_bits(f["Pt"], None)
        f["cand"], f["prob"] = cand, prob
        f["correct"] = A.projected_truth(f["y"], cand, None)
        f["cvs_ok"] = bool(f["cvs_true"].all())
    # CVS-rated keyframes (all of them, not only anatomy frames) for the CVS component
    va, te = keyframes("val"), keyframes("test")
    kf_v, kf_p, kf_y = [], [], []
    for split, kf in (("val", va), ("test", te)):
        z = np.load(cvs / f"logits_{split}.npz")
        idx = {(int(v), int(fr)): i for i, (v, fr) in enumerate(zip(z["vid"], z["frame"]))}
        keep = ~np.isin(kf.vid, sel_vids)
        L = np.stack([z["logits"][idx[(int(v), int(fr))]] for v, fr in zip(kf.vid[keep], kf.frame[keep])])
        kf_v.append(kf.vid[keep]); kf_p.append(np.prod(1 / (1 + np.exp(-L / T_cvs)), 1))
        kf_y.append((kf.raters[keep].sum(1) >= 2).all(1))
    return frames, np.concatenate(kf_v), np.concatenate(kf_p), np.concatenate(kf_y)


def overlays(frames, lam_id, lam_cvs=None):
    """list of {target: region or None} for a pair of thresholds."""
    out = []
    for f in frames:
        cv = lam_cvs is None or f["pcvs"] > lam_cvs
        out.append({t: (f["cand"][t] if f["cand"][t] is not None and f["prob"][t] > lam_id and cv else None)
                    for t in A.CLIP_TARGETS})
    return out


def score(frames, ovs, clip: bool):
    wrong_f, n_over, n_correct, by_video = 0, 0, 0, {}
    for f, ov in zip(frames, ovs):
        bad = False
        for t, r in ov.items():
            if r is None:
                continue
            n_over += 1
            ok = f["correct"][t] and (not clip or f["cvs_ok"])
            n_correct += ok; bad |= not ok
        wrong_f += bad
        by_video[f["vid"]] = by_video.get(f["vid"], False) or bad
    certifiable = sum(int(f["correct"][t] and (not clip or f["cvs_ok"])) for f in frames for t in A.CLIP_TARGETS)
    return {"surgeries_with_error": float(np.mean(list(by_video.values()))),
            "frame_error": wrong_f / len(frames), "overlays": n_over,
            "precision": n_correct / max(n_over, 1), "recall": n_correct / max(certifiable, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--splits", type=int, default=200)
    a = ap.parse_args()
    sel_vids = np.random.default_rng(0).choice(np.unique(keyframes("val").vid), 10, replace=False)
    rows = []
    for seed in a.seeds:
        frames, kv, kp, ky = prepare(seed, sel_vids)
        vids = np.array([f["vid"] for f in frames]); uv = np.unique(vids)
        for task in ("identify", "clip"):
            clip = task == "clip"
            # point world (fixed; no calibration)
            map_ovs = []
            for f in frames:
                y = A.map_world(f["Pt"]) if len(f["Pt"]) else np.zeros(0, int)
                cv = not clip or f["pcvs"] > 0.5
                map_ovs.append({t: (int(np.flatnonzero(y == t)[0]) if (y == t).any() and cv else None)
                                for t in A.CLIP_TARGETS})
            for rep in range(a.splits):
                rng = np.random.default_rng(rep)
                cv_ = set(rng.choice(uv, len(uv) // 2, replace=False).tolist())
                cal = [f for f in frames if f["vid"] in cv_]
                tst = [f for f in frames if f["vid"] not in cv_]
                tst_map = [o for f, o in zip(frames, map_ovs) if f["vid"] not in cv_]
                kc = np.isin(kv, list(cv_))
                rows.append({"seed": seed, "task": task, "rep": rep, "alpha": None, "method": "MAP (point world)",
                             **score(tst, tst_map, clip)})
                for alpha in ALPHAS:
                    a_id = alpha / 2 if clip else alpha
                    # surgery-level identity threshold: per calibration video, max confidence of a WRONG overlay
                    m_id = []
                    for v in cv_:
                        wrong = [f["prob"][t] for f in cal if f["vid"] == v for t in A.CLIP_TARGETS
                                 if f["cand"][t] is not None and not f["correct"][t]]
                        m_id.append(max(wrong) if wrong else -np.inf)
                    lam_id = order_stat_threshold(np.array(m_id), a_id)
                    lam_cvs = None
                    if clip:
                        m_c = [kp[kc & (kv == v) & ~ky].max() if (kc & (kv == v) & ~ky).any() else -np.inf
                               for v in cv_]
                        lam_cvs = order_stat_threshold(np.array(m_c), alpha / 2)
                    rows.append({"seed": seed, "task": task, "rep": rep, "alpha": alpha,
                                 "method": "Surgery-level", **score(tst, overlays(tst, lam_id, lam_cvs), clip)})
                    # frame-level comparators, frames pooled
                    if not clip:
                        s_cal = np.array([A.projected_score(f["prob"], f["correct"]) for f in cal])
                        k = math.ceil((len(s_cal) + 1) * (1 - alpha))
                        q = math.inf if k > len(s_cal) else float(np.sort(s_cal)[k - 1])
                        ovs = [A.projected_certify(f["cand"], f["prob"], q) for f in tst]
                        rows.append({"seed": seed, "task": task, "rep": rep, "alpha": alpha,
                                     "method": "Frame-level ours (frames pooled)", **score(tst, ovs, clip)})
                    allp = np.concatenate([list(f["prob"].values()) + ([f["pcvs"]] if clip else []) for f in cal])
                    grid = RC.threshold_grid(allp)
                    Lc = np.array([[any(f["cand"][t] is not None and f["prob"][t] > lam
                                        and (not clip or f["pcvs"] > lam)
                                        and (not f["correct"][t] or (clip and not f["cvs_ok"]))
                                        for t in A.CLIP_TARGETS) for lam in grid] for f in cal])
                    g = RC.ltt_index(Lc, alpha); lam = np.inf if g is None else grid[g]
                    rows.append({"seed": seed, "task": task, "rep": rep, "alpha": alpha,
                                 "method": "LTT (frames pooled)",
                                 **score(tst, overlays(tst, lam, lam if clip else None), clip)})
            print(f"seed {seed} {task} done", flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "experiments" / "surgery_level"; out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "anatomy_surgery_level.csv", index=False)
    df["alpha"] = df.alpha.fillna(-1)
    agg = df.groupby(["task", "alpha", "method"]).agg(
        surgeries_with_error=("surgeries_with_error", "mean"),
        surg_lo=("surgeries_with_error", lambda x: x.quantile(0.025)),
        surg_hi=("surgeries_with_error", lambda x: x.quantile(0.975)),
        frame_error=("frame_error", "mean"), precision=("precision", "mean"), recall=("recall", "mean"),
        overlays=("overlays", "mean")).round(4)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
    print(agg.to_string())
    (out / "anatomy_summary.json").write_text(agg.reset_index().to_json(orient="records", indent=1))


if __name__ == "__main__":
    main()
