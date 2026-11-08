"""External multicentre evaluation on annotated pre-clipping windows (SAGES CVS Challenge 2024).

Dataset: huggingface.co/datasets/CAMMA-public/SAGES_CVS_Challenge_2024 (CC BY-NC 4.0; 1,000 90-s
clips ending at clipping of the cystic duct or artery, 54 institutions, 24 countries; 18 frames per
clip, one every 5 s, rated on C1-C3 by three raters).  Nothing is trained, tuned or selected on it.

Stages (run in order; each caches its output):
  lock      choose 100 official-train cases (calibration) and 100 official-test cases (evaluation)
            uniformly without replacement, seed 2027 over sorted case IDs, BEFORE any label or
            prediction is read; compute the source (Endoscapes) thresholds; write protocol.json
  fetch     download labels and the selected videos only  (--limit N for the benchmark)
  extract   decode the 1-fps frames the causal 10 s window needs (t = 0, 1, ..., 85 s), resized to
            854x480 (INTER_AREA) and stored as JPEG q95, exactly as the Cholec80 external frames
  infer     run the frozen Endoscapes CVS model (DINOv2-L, seed 0, selection-video temperature) once
            and cache per-frame logits
  evaluate  policies A/B/C at alpha 0.05 and 0.10, metrics with exact and bootstrap intervals,
            table, figure, numbers.json

    python scripts/external/sages_external.py lock
    python scripts/external/sages_external.py fetch --limit 5 && ... extract --limit 5 && ... infer --limit 5
    python scripts/external/sages_external.py fetch && ... extract && ... infer && ... evaluate
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "scripts", "scripts/review"):
    sys.path.insert(0, str(ROOT / p))

REPO = "CAMMA-public/SAGES_CVS_Challenge_2024"
DATA = ROOT / "data" / "sages"
OUT = ROOT / "experiments" / "external_sages"
SEED = 2027
N_CAL = N_TEST = 100
ALPHAS = (0.05, 0.10)
WINDOWS = (10, 1)            # primary: 10 s causal smoothing (the paper's smoothed certificate); secondary: none
CKPT_SEED = 0                # existing checkpoint, chosen before this experiment (the paper's default seed)
CLIP_FPS = 30                # frame_id is 0-indexed at the clip FPS (dataset card)
T_MAX = 85                   # last annotated timestamp (frame_id 2550)


def protocol():
    return json.loads((OUT / "protocol.json").read_text())


def cases(split, limit=None):
    """Locked cases of a split; with --limit (benchmark) only the first `limit` calibration cases."""
    sel = protocol()["cases"]
    if limit is not None:
        return sel["calibration"][:limit] if split == "train" else []
    return sel["calibration"] if split == "train" else sel["evaluation"]


# ----------------------------------------------------------------------------------------- lock
def lock(a):
    from huggingface_hub import hf_hub_download
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        sys.exit("protocol.json exists: the selection is locked and is not rerun")
    ids = {}
    for split in ("train", "test"):
        meta = pd.read_csv(hf_hub_download(REPO, f"{split}/labels/metadata.csv", repo_type="dataset"))
        ids[split] = sorted(meta.video_name.astype(str).tolist())
    rng = np.random.default_rng(SEED)
    cal = sorted(np.array(ids["train"])[rng.choice(len(ids["train"]), N_CAL, replace=False)].tolist())
    tst = sorted(np.array(ids["test"])[rng.choice(len(ids["test"]), N_TEST, replace=False)].tolist())
    # source thresholds: the paper's operation-level certificate calibrated on all 71 Endoscapes
    # calibration/evaluation operations (convention of the paper: operations without unsafe frames left out)
    from common import endoscapes_ops, order_stat, temperature
    run = ROOT / "experiments" / "finetune" / f"dinov2l_b8_s{CKPT_SEED}"
    ops = endoscapes_ops("dinov2l_b8", CKPT_SEED, WINDOWS)
    src = {}
    for w in WINDOWS:
        m = [o["p"][w][~o["cvs"]].max() for o in ops if (~o["cvs"]).any()]
        src[str(w)] = {str(al): order_stat(m, al) for al in ALPHAS}
        src[str(w)]["V"] = len(m)
    prot = {
        "title": "External multicentre evaluation on annotated pre-clipping windows",
        "locked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": REPO, "dataset_release": "2024 edition (public); later annotations not included",
        "selection": {"seed": SEED, "rule": "numpy default_rng(2027): choice(sorted official-train IDs, 100, "
                      "replace=False), then choice(sorted official-test IDs, 100, replace=False); IDs only, "
                      "no labels, image quality, confidence or outcomes read before selection",
                      "pool_sizes": {k: len(v) for k, v in ids.items()}},
        "cases": {"calibration": cal, "evaluation": tst},
        "model": {"checkpoint": f"experiments/finetune/dinov2l_b8_s{CKPT_SEED}/best.pt",
                  "temperature": temperature(run), "temperature_fit": "10 Endoscapes selection videos",
                  "score": "p = prod_c sigmoid(mean of logits over the causal window / T)",
                  "preprocessing": "frame decoded from the clip, resized to 854x480 (cv2.INTER_AREA), JPEG q95, "
                                   "then the training-time evaluation transform (resize 336x602, ImageNet "
                                   "normalisation), bf16 autocast; identical to the Cholec80 transfer"},
        "smoothing": {"windows_s": list(WINDOWS), "primary_s": WINDOWS[0],
                      "rule": "mean of the logits of the 1-fps frames at t-w+1..t within the clip; the "
                              "window is truncated at the clip start (no pre-clip history exists in the "
                              "release); state is reset at every clip"},
        "reference": "per criterion majority of the three raters (>= 2 of 3), then CVS = C1 and C2 and C3 "
                     "(aggregation before combining, as for Endoscapes); unsafe scored frame = CVS absent",
        "scored_frames": "the 18 annotated timestamps per clip (frame_id 0, 150, ..., 2550 at 30 fps)",
        "alphas": list(ALPHAS),
        "policies": {
            "A": "source-calibrated operation threshold (Endoscapes, 71 operations) transferred unchanged",
            "B": "frame-level CRC recalibrated on the SAGES calibration cases: ceil((n0+1)(1-alpha))-th smallest "
                 "score over all pooled unsafe annotated frames (frames treated as exchangeable)",
            "C": "operation-unit calibration on the SAGES calibration cases: m_v = maximum score over the "
                 "case's unsafe annotated frames, -inf if it has none (all 100 cases retained); threshold = "
                 "ceil((V+1)(1-alpha))-th smallest m_v, +inf if that rank exceeds V"},
        "issuance": "confirm CVS iff p > threshold (strict); ties therefore never confirm",
        "source_thresholds": src,
        "endpoint": "case-level any false CVS confirmation among the 18 annotated timestamps of the 90-s clip",
    }
    (OUT / "protocol.json").write_text(json.dumps(prot, indent=1))
    pd.DataFrame({"case": cal + tst, "role": ["calibration"] * len(cal) + ["evaluation"] * len(tst)}).to_csv(
        OUT / "locked_cases.csv", index=False)
    print("locked", len(cal), len(tst), "source thresholds", src)


# ---------------------------------------------------------------------------------------- fetch
def fetch(a):
    from huggingface_hub import hf_hub_download
    log = []
    for split in ("train", "test"):
        for c in cases(split, a.limit):
            t0 = time.time()
            for f in (f"{split}/labels/{c}/frame.csv", f"{split}/labels/{c}/video.csv", f"{split}/videos/{c}.mp4"):
                hf_hub_download(REPO, f, repo_type="dataset", local_dir=DATA)
            mb = (DATA / split / "videos" / f"{c}.mp4").stat().st_size / 1e6
            log.append({"split": split, "case": c, "MB": mb, "s": time.time() - t0})
            print(split, c, f"{mb:.0f} MB {time.time() - t0:.1f} s", flush=True)
    pd.DataFrame(log).to_csv(OUT / f"fetch_log{'' if a.limit is None else '_bench'}.csv", index=False)


# -------------------------------------------------------------------------------------- extract
def extract_one(args):
    import cv2
    split, c = args
    od = DATA / "frames" / split / c
    if (od / "done").exists():
        return split, c, 0.0, None
    od.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cap = cv2.VideoCapture(str(DATA / split / "videos" / f"{c}.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or CLIP_FPS
    targets, k, idx, got = list(range(T_MAX + 1)), 0, 0, []
    while k < len(targets):
        ok, img = cap.read()
        if not ok:
            break
        if idx / fps >= targets[k] - 0.5 / fps:            # first decoded frame at or after second k
            img = cv2.resize(img, (854, 480), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(od / f"{targets[k] * CLIP_FPS}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            got.append(targets[k]); k += 1
        idx += 1
    cap.release()
    (od / "done").write_text(json.dumps({"fps": fps, "n": len(got)}))
    return split, c, time.time() - t0, fps


def extract(a):
    from multiprocessing import Pool
    jobs = [(s, c) for s in ("train", "test") for c in cases(s, a.limit)]
    rows = []
    with Pool(12) as pool:
        for split, c, dt, fps in pool.imap_unordered(extract_one, jobs):
            rows.append({"split": split, "case": c, "s": dt, "fps": fps}); print(split, c, f"{dt:.1f} s fps {fps}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / f"extract_log{'' if a.limit is None else '_bench'}.csv", index=False)


# ---------------------------------------------------------------------------------------- infer
def infer(a):
    import torch
    from finetune import build_model, predict
    rows = []
    for split in ("train", "test"):
        for c in cases(split, a.limit):
            for f in sorted((DATA / "frames" / split / c).glob("*.jpg"), key=lambda p: int(p.stem)):
                rows.append((split, c, int(f.stem), str(f)))
    df = pd.DataFrame(rows, columns=["split", "case", "frame_id", "file"])
    model = build_model("dinov2l").cuda()
    model.load_state_dict(torch.load(ROOT / "experiments" / "finetune" / f"dinov2l_b8_s{CKPT_SEED}" / "best.pt",
                                     map_location="cuda"))
    t0 = time.time()
    L = predict(model, list(df.file))
    dt = time.time() - t0
    name = "logits.npz" if a.limit is None else "logits_bench.npz"
    np.savez(OUT / name, split=df.split.to_numpy(), case=df.case.to_numpy(), frame_id=df.frame_id.to_numpy(), logits=L)
    print(f"{len(df)} frames in {dt:.1f} s ({len(df) / dt:.0f} frames/s) -> {name}", flush=True)
    (OUT / f"infer_time{'' if a.limit is None else '_bench'}.json").write_text(json.dumps({"frames": len(df), "s": dt}))


# ------------------------------------------------------------------------------------- evaluate
def load_cases(T, w):
    z = np.load(OUT / "logits.npz", allow_pickle=True)
    out = {}
    for split in ("train", "test"):
        for c in cases(split):
            m = np.flatnonzero((z["case"] == c) & (z["split"] == split))
            m = m[np.argsort(z["frame_id"][m])]
            fid, L = z["frame_id"][m], z["logits"][m].astype(float)
            Ls = np.stack([L[max(0, i - w + 1):i + 1].mean(0) for i in range(len(L))])   # causal, clip-local
            pc = 1 / (1 + np.exp(-Ls / T))
            lab = pd.read_csv(DATA / split / "labels" / c / "frame.csv").sort_values("frame_id")
            # (n frames, 3 raters, 3 criteria)
            R = np.stack([np.stack([lab[f"c{x}_rater{y}"].to_numpy() for x in (1, 2, 3)], 1) for y in (1, 2, 3)], 1)
            idx = {f: i for i, f in enumerate(fid)}
            keep = [idx[f] for f in lab.frame_id]
            maj = (R.sum(1) >= 2).all(1)                      # majority per criterion, then C1 and C2 and C3
            alt = (R.all(2).sum(1) >= 2)                      # alternative: per rater C1&C2&C3, then majority
            out[(split, c)] = {"p": np.prod(pc[keep], 1), "pc": pc[keep], "cvs": maj, "alt": alt,
                               "frame_id": lab.frame_id.to_numpy()}
    return out


def order_stat(m, alpha):
    m = np.sort(np.asarray(m, float)); k = math.ceil((len(m) + 1) * (1 - alpha))
    return math.inf if len(m) == 0 or k > len(m) else float(m[k - 1])


def cp(k, n):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return lo, hi


def case_stats(tst, lam):
    rows = []
    for o in tst:
        c = o["p"] > lam; y = o["cvs"]
        rows.append({"err": bool((c & ~y).any()), "tp": int((c & y).sum()), "fp": int((c & ~y).sum()),
                     "pos": int(y.sum()), "n": len(y), "cvs_case": bool(y.any()), "hit": bool((c & y).any())})
    return pd.DataFrame(rows)


def summarise(d, rng, B=2000):
    tp, fp, pos, n = d.tp.sum(), d.fp.sum(), d.pos.sum(), d.n.sum()
    k = int(d.err.sum()); lo, hi = cp(k, len(d))
    out = {"n_test": len(d), "cases_with_false_conf": k, "case_error": k / len(d), "case_error_lo": lo, "case_error_hi": hi,
           "sens": tp / pos if pos else np.nan, "precision": tp / (tp + fp) if tp + fp else np.nan,
           "fdp": fp / (tp + fp) if tp + fp else np.nan, "issuance": (tp + fp) / n, "confirmations": int(tp + fp),
           "cvs_frames": int(pos), "frames": int(n), "cvs_cases": int(d.cvs_case.sum()),
           "cvs_cases_confirmed": int((d.cvs_case & d.hit).sum())}
    bs = {"sens": [], "precision": [], "issuance": []}
    for _ in range(B):                                       # case-level bootstrap
        s = d.iloc[rng.integers(0, len(d), len(d))]
        tp_, fp_, pos_, n_ = s.tp.sum(), s.fp.sum(), s.pos.sum(), s.n.sum()
        bs["sens"].append(tp_ / pos_ if pos_ else np.nan)
        bs["precision"].append(tp_ / (tp_ + fp_) if tp_ + fp_ else np.nan)
        bs["issuance"].append((tp_ + fp_) / n_)
    for k_, v in bs.items():
        v = np.array(v, float); v = v[~np.isnan(v)]
        out[f"{k_}_lo"], out[f"{k_}_hi"] = (np.quantile(v, 0.025), np.quantile(v, 0.975)) if len(v) else (np.nan, np.nan)
    return out


def evaluate(a):
    prot = protocol(); T = prot["model"]["temperature"]
    rows, info = [], {}
    for w in WINDOWS:
        D = load_cases(T, w)
        cal = [D[("train", c)] for c in cases("train")]; tst = [D[("test", c)] for c in cases("test")]
        info[f"w{w}"] = {"cal_cases": len(cal), "test_cases": len(tst),
                         "cal_cases_without_unsafe": int(sum((~o["cvs"]).sum() == 0 for o in cal)),
                         "test_cases_without_unsafe": int(sum((~o["cvs"]).sum() == 0 for o in tst)),
                         "cal_cvs_frames": int(sum(o["cvs"].sum() for o in cal)), "test_cvs_frames": int(sum(o["cvs"].sum() for o in tst)),
                         "cal_frames": int(sum(len(o["cvs"]) for o in cal)), "test_frames": int(sum(len(o["cvs"]) for o in tst)),
                         "reference_disagreement_frames": int(sum((o["cvs"] != o["alt"]).sum() for o in cal + tst))}
        for al in ALPHAS:
            negs = np.concatenate([o["p"][~o["cvs"]] for o in cal])
            m = [o["p"][~o["cvs"]].max() if (~o["cvs"]).any() else -np.inf for o in cal]
            pols = {"A: source operation threshold (unchanged)": prot["source_thresholds"][str(w)][str(al)],
                    "B: frame-level CRC, recalibrated": order_stat(negs, al),
                    "C: operation-unit, recalibrated (ours)": order_stat(m, al)}
            for name, lam in pols.items():
                rng = np.random.default_rng(SEED)
                rows.append({"w": w, "alpha": al, "policy": name, "threshold": lam, "n_cal": len(cal),
                             "n0_cal_frames": len(negs) if name.startswith("B") else np.nan,
                             "V_cal": len(m) if name.startswith("C") else (prot["source_thresholds"][str(w)]["V"] if name.startswith("A") else np.nan),
                             **summarise(case_stats(tst, lam), rng)})
    df = pd.DataFrame(rows); df.to_csv(OUT / "results.csv", index=False)
    (OUT / "numbers.json").write_text(json.dumps({"info": info, "rows": rows}, indent=1, default=float))
    pd.set_option("display.width", 250)
    print(json.dumps(info, indent=1))
    print(df[["w", "alpha", "policy", "threshold", "cases_with_false_conf", "case_error", "case_error_lo", "case_error_hi",
              "sens", "precision", "issuance", "cvs_cases", "cvs_cases_confirmed"]].round(4).to_string())
    table(df); figure(df)


def fmt(x, d=1):
    return "--" if x is None or (isinstance(x, float) and (np.isnan(x))) else f"{100 * x:.{d}f}"


def table(df):
    L = []
    for w in WINDOWS:
        for _, r in df[df.w == w].iterrows():
            lam = "$+\\infty$" if math.isinf(r.threshold) else f"{r.threshold:.3f}"
            L.append(f"{w} & {r.alpha:.2f}".replace("0.", ".") + f" & {r.policy.split(':')[0]} & {lam} & "
                     f"{r.cases_with_false_conf}/{r.n_test} & {fmt(r.case_error)} [{fmt(r.case_error_lo)}, {fmt(r.case_error_hi)}] & "
                     f"{fmt(r.sens)} [{fmt(r.sens_lo)}, {fmt(r.sens_hi)}] & {fmt(r.precision)} & {fmt(r.issuance)} & "
                     f"{r.cvs_cases_confirmed}/{r.cvs_cases} \\\\")
        L.append("\\midrule")
    p = ROOT / "paper" / "tables" / "supp" / "sages.tex"; p.write_text("\n".join(L[:-1]) + "\n")
    L = []
    for _, r in df[df.w == WINDOWS[0]].iterrows():
        L.append(f"{r.policy.split(':')[0]} & {r.alpha:.2f}".replace("0.", ".") + f" & {r.cases_with_false_conf}/{r.n_test} & "
                 f"[{fmt(r.case_error_lo, 0)}, {fmt(r.case_error_hi, 0)}] & {fmt(r.sens)} & {fmt(r.precision, 0)} & {r.cvs_cases_confirmed}/{r.cvs_cases} \\\\")
    (ROOT / "paper" / "tables" / "sages.tex").write_text("\n".join(L) + "\n")


def figure(df):
    sys.path.insert(0, str(ROOT / "scripts" / "figures"))
    import style as ST
    ST.setup()
    import matplotlib.pyplot as plt
    full = S_full = None
    pf = OUT / "results_full.csv"
    if pf.exists():
        full = pd.read_csv(pf); full = full[(full.w == WINDOWS[0]) & full.analysis.str.startswith("F1")]
    d = df[df.w == WINDOWS[0]]
    cols = {"A": (ST.POINT, "D", "A  Endoscapes threshold"), "B": (ST.CRC, "s", "B  frame-level CRC, recalibrated"),
            "C": (ST.OURS, "o", "C  per operation (ours), recalibrated")}
    sets = [("100 → 100 (locked)", d, lambda x: x.policy.str[0])]
    if full is not None:
        sets.append(("700 → 300", full, lambda x: x.policy))
    fig, ax = plt.subplots(1, 2, figsize=(ST.DOUBLE, 2.2))
    for si, (sname, x, polkey) in enumerate(sets):
        for j, al in enumerate(ALPHAS):
            base = si * 2.6 + j * 1.15
            ax[0].plot([base - 0.45, base + 0.45], [100 * al] * 2, c=ST.INK, ls=(0, (1.2, 1.4)), lw=0.8, label="target (nominal level)" if (si == 0 and j == 0) else None)
            for i, p in enumerate("ABC"):
                r = x[(polkey(x) == p) & np.isclose(x.alpha, al)].iloc[0]
                c, mk, lab = cols[p]; xx = base + (i - 1) * 0.28
                ax[0].errorbar(xx, 100 * r.case_error, yerr=[[100 * (r.case_error - r.case_error_lo)], [100 * (r.case_error_hi - r.case_error)]],
                               fmt=mk, c=c, mec="white", mew=0.5, label=lab if (si == 0 and j == 0) else None, zorder=3)
                ax[1].errorbar(xx, 100 * r.sens, yerr=[[100 * (r.sens - r.sens_lo)], [100 * (r.sens_hi - r.sens)]],
                               fmt=mk, c=c, mec="white", mew=0.5, zorder=3)
    ticks = [si * 2.6 + j * 1.15 for si in range(len(sets)) for j in range(len(ALPHAS))]
    labs = [f"α={al:.2f}".replace("0.", ".") for _ in sets for al in ALPHAS]
    for k in range(2):
        ax[k].set_xticks(ticks); ax[k].set_xticklabels(labs); ax[k].grid(axis="x", visible=False)
        for si, (sname, _, _) in enumerate(sets):
            ax[k].text(si * 2.6 + 0.575, -0.2, sname, transform=ax[k].get_xaxis_transform(), ha="center", fontsize=7,
                       color=ST.MUTED)
    ax[0].set_ylim(0, 55); ax[0].set_ylabel("cases with ≥1 false confirmation (%)")
    ax[1].set_ylim(0, 100); ax[1].set_ylabel("sensitivity on CVS frames (%)")
    ax[0].set_title("a  case-level error, exact 95% CI"); ax[1].set_title("b  sensitivity, case-bootstrap 95% CI")
    fig.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    ST.save(fig, ROOT / "paper" / "figures" / "fig_sages")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["lock", "fetch", "extract", "infer", "evaluate"])
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    {"lock": lock, "fetch": fetch, "extract": extract, "infer": infer, "evaluate": evaluate}[a.stage](a)


if __name__ == "__main__":
    main()
