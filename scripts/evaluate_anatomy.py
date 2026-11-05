"""Certified structure identification (and clip permission) on Endoscapes-BBox201.

Two tasks per detection keyframe:
  identify   overlay "this region is the cystic duct / cystic artery"
  clip       the same, and only if CVS holds (Safe Cholecystectomy guideline)
Guarantee targeted by ours: frame-level — P(any overlay in the frame is wrong) <= alpha.

Splits as everywhere else: 10 val videos select/temperature, 31 val videos calibrate,
40 test videos evaluate.

    python scripts/evaluate_anatomy.py --det experiments/detector/frcnn_s0 \
        --cvs experiments/finetune/dinov2l_b8_s0
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cdss import anatomy as A                     # noqa: E402
from cdss.data import keyframes                   # noqa: E402
from cdss import riskcontrol as RC                # noqa: E402

ALPHAS = (0.05, 0.10, 0.20)


def load_frames(det_dir: Path, cvs_dir: Path, split: str):
    reg = json.loads((det_dir / f"regions_{split}.json").read_text())
    z = np.load(cvs_dir / f"logits_{split}.npz")
    idx = {(int(v), int(f)): i for i, (v, f) in enumerate(zip(z["vid"], z["frame"]))}
    kf = keyframes(split)
    raters = {(int(v), int(f)): r for v, f, r in zip(kf.vid, kf.frame, kf.raters)}
    # video/frame from the image file name ("{vid}_{frame}.jpg"); the COCO image id does not
    # encode them consistently
    names = {str(i["id"]): i["file_name"] for i in
             json.loads((ROOT / "data" / "endoscapes" / split / "annotation_coco.json").read_text())["images"]}
    frames = []
    for iid, d in reg.items():
        vid, fr = (int(x) for x in Path(names[iid]).stem.split("_"))
        B = np.array(d["boxes"]).reshape(-1, 4); P = np.array(d["probs"]).reshape(-1, 7)
        y = A.region_truth(B, np.array(d["gt_boxes"]).reshape(-1, 4), np.array(d["gt_labels"], int))
        R = raters[(vid, fr)]
        frames.append({"vid": vid, "frame": fr, "P": P, "y": y, "logit3": z["logits"][idx[(vid, fr)]],
                       "cvs_true": (R.sum(0) >= 2), "gt_labels": np.array(d["gt_labels"], int)})
    return frames


def fit_T(nll_fn):
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad(); l = nll_fn(logT.exp()); l.backward(); return l
    opt.step(closure)
    return float(logT.exp())


def _cert(args):
    P, p3, q, require_cvs, use_K = args
    return A.certify_frame(P, p3, q, require_cvs, use_K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True); ap.add_argument("--cvs", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    det, cvs = ROOT / a.det, ROOT / a.cvs
    val, test = load_frames(det, cvs, "val"), load_frames(det, cvs, "test")
    sel_vids = np.random.default_rng(0).choice(np.unique(keyframes("val").vid), 10, replace=False)
    sel = [f for f in val if f["vid"] in sel_vids]; cal = [f for f in val if f["vid"] not in sel_vids]

    # temperatures (selection videos only)
    Ls = torch.tensor(np.concatenate([np.log(np.clip(f["P"], A.EPS, 1)) for f in sel if len(f["P"])]))
    ys = torch.tensor(np.concatenate([f["y"] for f in sel if len(f["P"])]))
    T_det = fit_T(lambda T: torch.nn.functional.cross_entropy(Ls / T, ys))
    L3 = torch.tensor(np.stack([f["logit3"] for f in sel])); Y3 = torch.tensor(np.stack([f["cvs_true"] for f in sel]), dtype=torch.float32)
    T_cvs = fit_T(lambda T: torch.nn.functional.binary_cross_entropy_with_logits(L3 / T, Y3))
    for f in val + test:
        f["Pt"] = A.temper(f["P"], T_det) if len(f["P"]) else f["P"]
        f["p3"] = 1 / (1 + np.exp(-f["logit3"] / T_cvs))
        assert A.in_K(f["y"])                                       # truth always satisfies K
    # CVS calibration on EVERY rated calibration keyframe (not only detection frames)
    from cdss.decide import joint_from_marginals
    va = keyframes("val")
    zc = np.load(cvs / "logits_val.npz")
    ic = {(int(v), int(fr)): i for i, (v, fr) in enumerate(zip(zc["vid"], zc["frame"]))}
    cm = ~np.isin(va.vid, sel_vids)
    Lk = np.stack([zc["logits"][ic[(int(v), int(fr))]] for v, fr in zip(va.vid[cm], va.frame[cm])])
    cvs_cal_p = joint_from_marginals(1 / (1 + np.exp(-Lk / T_cvs)))[:, 7]
    cvs_cal_y = (va.raters[cm].sum(1) >= 2).all(1)
    info = {"T_det": T_det, "T_cvs": T_cvs, "n_cal": len(cal), "n_test": len(test),
            "n_cvs_cal_keyframes": int(cm.sum()),
            "regions_per_test_frame": float(np.mean([len(f["P"]) for f in test]))}

    # how often is each target structure actually held by some region (detector recall)?
    for t, name in ((A.DUCT, "duct"), (A.ARTERY, "artery")):
        present = [f for f in test if (f["gt_labels"] == t).any()]
        info[f"{name}_frames"] = len(present)
        info[f"{name}_detected"] = float(np.mean([(f["y"] == t).any() for f in present]))

    rows = []

    def evaluate(method, alpha, task, overlays):
        """overlays: list (per test frame) of dict target -> region or None."""
        n_over = wrong_frames = n_wrong = correct = 0
        for f, ov in zip(test, overlays):
            bad = False
            for t, r in ov.items():
                if r is None:
                    continue
                n_over += 1
                ok = f["y"][r] == t and (task == "identify" or f["cvs_true"].all())
                correct += ok; n_wrong += (not ok); bad |= (not ok)
            wrong_frames += bad
        held = sum(int((f["y"] == t).any() and (task == "identify" or f["cvs_true"].all()))
                   for f in test for t in A.CLIP_TARGETS)
        rows.append({"task": task, "method": method, "alpha": alpha, "overlays": n_over, "wrong": n_wrong,
                     "frame_error_rate": wrong_frames / len(test), "precision": correct / max(n_over, 1),
                     "recall_of_certifiable": correct / max(held, 1), "certifiable": held})

    for task in ("identify", "clip"):
        req = task == "clip"
        # point world
        ovs = []
        for f in test:
            y = A.map_world(f["Pt"]) if len(f["Pt"]) else np.zeros(0, int)
            cvs_ok = (f["p3"] > 0.5).all() or not req
            ovs.append({t: (int(np.flatnonzero(y == t)[0]) if (y == t).any() and cvs_ok else None) for t in A.CLIP_TARGETS})
        evaluate("MAP (point world)", None, task, ovs)

        def cal_score(f, with_K=True):
            return A.score(f["Pt"], f["y"], f["p3"] if req else None, f["cvs_true"] if req else None)
        s_cal = np.array([cal_score(f) for f in cal])
        for alpha in ALPHAS:
            k = math.ceil((len(s_cal) + 1) * (1 - alpha))
            q = math.inf if k > len(s_cal) else float(np.sort(s_cal)[k - 1])
            cov = np.mean([cal_score(f) <= q for f in test])
            with Pool(a.workers) as pool:
                for use_K, name in ((True, "full-world (Gamma ∩ K)"), (False, "full-world w/o K")):
                    ovs = pool.map(_cert, [(f["Pt"], f["p3"] if req else None, q, req, use_K) for f in test])
                    evaluate(name, alpha, task, ovs)
                    rows[-1]["coverage"] = cov; rows[-1]["q_hat"] = q

            # ours on the decision-sufficient projection
            def pc(f):
                return float(np.prod(f["p3"])) if req else None
            s_proj = np.array([A.projected_score(A.projected_bits(f["Pt"], pc(f))[1],
                                                 A.projected_truth(f["y"], A.projected_bits(f["Pt"], pc(f))[0],
                                                                   f["cvs_true"].all() if req else None))
                               for f in cal])
            kp = math.ceil((len(s_proj) + 1) * (1 - alpha))
            qp = math.inf if kp > len(s_proj) else float(np.sort(s_proj)[kp - 1])
            ovs = [A.projected_certify(*A.projected_bits(f["Pt"], pc(f)), qp) for f in test]
            evaluate("projected", alpha, task, ovs); rows[-1]["q_hat"] = qp

            # risk-control baselines on the frame-level error: overlay target t on its
            # candidate region iff P(candidate is t) > lam (and P(CVS) > lam for clip)
            def conf_of(f):
                cand, prob = A.projected_bits(f["Pt"], pc(f))
                return cand, prob
            def frame_loss(f, lam):
                cand, prob = conf_of(f)
                cv = prob.get("cvs", 1.0) > lam
                return any(cand[t] is not None and prob[t] > lam and cv and
                           (f["y"][cand[t]] != t or (req and not f["cvs_true"].all())) for t in A.CLIP_TARGETS)
            allp = np.concatenate([[p for p in conf_of(f)[1].values()] for f in cal])
            grid = RC.threshold_grid(allp)
            Lc = np.array([[frame_loss(f, lam) for lam in grid] for f in cal])
            for nm, fn in (("CRC", RC.crc_index), ("LTT", RC.ltt_index)):
                g = fn(Lc, alpha); lam = np.inf if g is None else grid[g]
                ovs = []
                for f in test:
                    cand, prob = conf_of(f); cv = prob.get("cvs", 1.0) > lam
                    ovs.append({t: (cand[t] if cand[t] is not None and prob[t] > lam and cv else None) for t in A.CLIP_TARGETS})
                evaluate(nm, alpha, task, ovs); rows[-1]["lambda"] = float(lam)

            # composed certificate (clip only): identity at alpha/2 on the projection WITHOUT the
            # CVS bit, CVS at alpha/2 calibrated on every rated calibration keyframe (CRC on the
            # joint risk P(confirm and not CVS)). Union bound: P(wrong clip) <= alpha.
            if req:
                s_id = np.array([A.projected_score(A.projected_bits(f["Pt"], None)[1],
                                                   A.projected_truth(f["y"], A.projected_bits(f["Pt"], None)[0], None))
                                 for f in cal])
                ki = math.ceil((len(s_id) + 1) * (1 - alpha / 2))
                qi = math.inf if ki > len(s_id) else float(np.sort(s_id)[ki - 1])
                grid = RC.threshold_grid(cvs_cal_p)
                Lj = (cvs_cal_p[:, None] > grid[None, :]) & ~cvs_cal_y[:, None]
                g = RC.crc_index(Lj, alpha / 2); lam = np.inf if g is None else grid[g]
                ovs = []
                for f in test:
                    ov = A.projected_certify(*A.projected_bits(f["Pt"], None), qi)
                    ok_cvs = float(np.prod(f["p3"])) > lam
                    ovs.append({t: (r if ok_cvs else None) for t, r in ov.items()})
                evaluate("composed (id ∧ CVS)", alpha, task, ovs)
                rows[-1]["lambda_cvs"] = float(lam); rows[-1]["q_id"] = qi

            # tuned threshold: smallest tau with calibration frame-error <= alpha
            def thr_overlays(frames, tau):
                out = []
                for f in frames:
                    ov = {}
                    for t in A.CLIP_TARGETS:
                        if len(f["Pt"]) == 0:
                            ov[t] = None; continue
                        r = int(np.argmax(f["Pt"][:, t]))
                        ok = f["Pt"][r, t] >= tau and (not req or np.prod(f["p3"]) >= tau)
                        ov[t] = r if ok else None
                    out.append(ov)
                return out

            def frame_err(frames, ovs):
                return np.mean([any(r is not None and (f["y"][r] != t or (req and not f["cvs_true"].all()))
                                    for t, r in ov.items()) for f, ov in zip(frames, ovs)])
            taus = np.linspace(0.3, 0.999, 300)
            ok = [t for t in taus if frame_err(cal, thr_overlays(cal, t)) <= alpha]
            tau = min(ok) if ok else 1.01
            evaluate("Tuned threshold", alpha, task, thr_overlays(test, tau)); rows[-1]["tau"] = tau

            # per-region conformal (LAC) — guarantees each overlay, not the frame
            reg_s = np.concatenate([-np.log(np.clip(f["Pt"][np.arange(len(f["y"])), f["y"]], A.EPS, 1)) for f in cal if len(f["y"])])
            kk = math.ceil((len(reg_s) + 1) * (1 - alpha)); qr = float(np.sort(reg_s)[kk - 1])
            if req:
                c_s = np.array([-np.log(np.clip(np.where(f["cvs_true"], f["p3"], 1 - f["p3"]), A.EPS, 1)).sum() for f in cal])
                kc = math.ceil((len(c_s) + 1) * (1 - alpha)); qc = float(np.sort(c_s)[kc - 1])
            ovs = []
            for f in test:
                ov = {}
                for t in A.CLIP_TARGETS:
                    if len(f["Pt"]) == 0:
                        ov[t] = None; continue
                    r = int(np.argmax(f["Pt"][:, t]))
                    single = (-np.log(np.clip(f["Pt"][r], A.EPS, 1)) <= qr)
                    okr = single[t] and single.sum() == 1
                    if req:
                        p = np.clip(f["p3"], A.EPS, 1 - A.EPS)
                        # CVS set excludes every non-CVS configuration
                        okc = all(-np.log(np.where(np.array(c, bool), p, 1 - p)).sum() > qc
                                  for c in np.ndindex(2, 2, 2) if not all(c))
                        okr = okr and okc and -np.log(p).sum() <= qc
                    ov[t] = r if okr else None
                ovs.append(ov)
            evaluate("Per-region conformal", alpha, task, ovs)
            print(f"{task} alpha={alpha} done", flush=True)

    df = pd.DataFrame(rows)
    out = det / "anatomy_decisions.csv"
    df.to_csv(out, index=False)
    (det / "anatomy_info.json").write_text(json.dumps(info, indent=1))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    print(json.dumps(info, indent=1))
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
