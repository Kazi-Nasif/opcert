"""Second, pre-declared SAGES analysis (review round 4): all 1,000 cases.

Declared in experiments/external_sages/protocol_full.json BEFORE any of the additional 800 cases was
downloaded or scored; the locked 100/100 analysis (sages_external.py) stays the primary result.
  F1  official split in full: calibrate on all 700 official-train cases, evaluate on all 300 official-test
      cases.
  F2  leave-one-country-out: for every anonymised country with >= 30 cases (from metadata counts only),
      calibrate on all cases of the other countries (both official splits), evaluate on that country.
  F3  exchangeable reference: 200 random 700/300 re-splits of the 1,000 cases (official split ignored).
Same model, preprocessing, smoothing (10 s primary, none secondary), reference, policies A/B/C, levels
(0.05, 0.10) and metrics as the locked analysis.

    python scripts/external/sages_full.py declare
    python scripts/external/sages_full.py fetch      # downloads, extracts and scores the missing cases
    python scripts/external/sages_full.py infer
    python scripts/external/sages_full.py evaluate
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "external"))
import sages_external as S  # noqa: E402

PF = S.OUT / "protocol_full.json"
MIN_COUNTRY = 30


def metadata():
    from huggingface_hub import hf_hub_download
    return pd.concat([pd.read_csv(hf_hub_download(S.REPO, f"{s}/labels/metadata.csv", repo_type="dataset")).assign(split=s)
                      for s in ("train", "test")], ignore_index=True)


def declare(a):
    if PF.exists():
        sys.exit("protocol_full.json exists; not redeclared")
    m = metadata()
    cnt = m.country.value_counts()
    groups = sorted(int(c) for c in cnt[cnt >= MIN_COUNTRY].index)
    prot = {"declared_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "primary": "the locked 100/100 analysis of protocol.json remains the primary result",
            "F1": "calibrate on all 700 official-train cases, evaluate on all 300 official-test cases",
            "F2": f"leave-one-country-out for anonymised countries with >= {MIN_COUNTRY} cases (counts from metadata only): "
                  "calibrate on all cases of the other countries (both splits), evaluate on the held-out country",
            "F2_countries": groups, "F2_country_sizes": {str(c): int(cnt[c]) for c in groups},
            "F3": "200 random 700/300 re-splits of all 1,000 cases (exchangeable reference), seed 0",
            "country_ids_note": "country IDs are anonymised; we assume one mapping across the two splits (not verifiable)",
            "unchanged": "model, preprocessing, smoothing (10 s primary, none secondary), reference, policies A/B/C, "
                         "alphas, metrics and conventions exactly as protocol.json"}
    PF.write_text(json.dumps(prot, indent=1))
    print(json.dumps(prot, indent=1))


def all_cases():
    m = metadata()
    return [(r.split, str(r.video_name)) for r in m.itertuples()]


def fetch_one(args):
    from huggingface_hub import hf_hub_download
    split, c = args
    for f in (f"{split}/labels/{c}/frame.csv", f"{split}/labels/{c}/video.csv", f"{split}/videos/{c}.mp4"):
        for attempt in range(3):
            try:
                hf_hub_download(S.REPO, f, repo_type="dataset", local_dir=S.DATA); break
            except Exception as e:                      # transient network errors: retry
                if attempt == 2:
                    raise
                time.sleep(10)
    return S.extract_one((split, c))


def fetch(a):
    assert PF.exists(), "declare first"
    jobs = all_cases()
    t0 = time.time()
    with Pool(8) as pool:
        for i, (split, c, dt, fps) in enumerate(pool.imap_unordered(fetch_one, jobs)):
            if i % 25 == 0:
                print(f"{i}/{len(jobs)} {time.time() - t0:.0f} s", flush=True)
    print("fetched", len(jobs), flush=True)


def infer(a):
    import torch
    from finetune import build_model, predict
    rows = []
    for split, c in all_cases():
        for f in sorted((S.DATA / "frames" / split / c).glob("*.jpg"), key=lambda p: int(p.stem)):
            rows.append((split, c, int(f.stem), str(f)))
    df = pd.DataFrame(rows, columns=["split", "case", "frame_id", "file"])
    model = build_model("dinov2l").cuda()
    model.load_state_dict(torch.load(ROOT / "experiments" / "finetune" / f"dinov2l_b8_s{S.CKPT_SEED}" / "best.pt", map_location="cuda"))
    t0 = time.time(); L = predict(model, list(df.file)); dt = time.time() - t0
    np.savez(S.OUT / "logits_full.npz", split=df.split.to_numpy(), case=df.case.to_numpy(), frame_id=df.frame_id.to_numpy(), logits=L)
    print(f"{len(df)} frames in {dt:.0f} s", flush=True)


def load_all(T, w):
    z = np.load(S.OUT / "logits_full.npz", allow_pickle=True)
    key = pd.Series(np.arange(len(z["case"]))).groupby([z["split"], z["case"]]).apply(list)
    out = {}
    for (split, c), idx in key.items():
        idx = np.array(idx); idx = idx[np.argsort(z["frame_id"][idx])]
        L = z["logits"][idx].astype(float)
        Ls = np.stack([L[max(0, i - w + 1):i + 1].mean(0) for i in range(len(L))])
        p = np.prod(1 / (1 + np.exp(-Ls / T)), 1)
        lab = pd.read_csv(S.DATA / split / "labels" / c / "frame.csv").sort_values("frame_id")
        R = np.stack([np.stack([lab[f"c{x}_rater{y}"].to_numpy() for x in (1, 2, 3)], 1) for y in (1, 2, 3)], 1)
        pos = {f: i for i, f in enumerate(z["frame_id"][idx])}
        keep = [pos[f] for f in lab.frame_id]
        out[(split, c)] = {"p": p[keep], "cvs": (R.sum(1) >= 2).all(1)}
    return out


def policies(cal, prot, w, al):
    negs = np.concatenate([o["p"][~o["cvs"]] for o in cal])
    m = [o["p"][~o["cvs"]].max() if (~o["cvs"]).any() else -np.inf for o in cal]
    return {"A": prot["source_thresholds"][str(w)][str(al)], "B": S.order_stat(negs, al), "C": S.order_stat(m, al)}


def evaluate(a):
    prot = S.protocol(); pf = json.loads(PF.read_text()); T = prot["model"]["temperature"]
    meta = metadata(); country = {(r.split, str(r.video_name)): int(r.country) for r in meta.itertuples()}
    rows, ref = [], []
    for w in S.WINDOWS:
        D = load_all(T, w)
        splits = {"F1 official 700 -> 300": ([k for k in D if k[0] == "train"], [k for k in D if k[0] == "test"])}
        for g in pf["F2_countries"]:
            splits[f"F2 country {g}"] = ([k for k in D if country[k] != g], [k for k in D if country[k] == g])
        for name, (ck, tk) in splits.items():
            cal, tst = [D[k] for k in ck], [D[k] for k in tk]
            for al in S.ALPHAS:
                for pol, lam in policies(cal, prot, w, al).items():
                    rows.append({"w": w, "analysis": name, "alpha": al, "policy": pol, "threshold": lam, "n_cal": len(cal),
                                 **S.summarise(S.case_stats(tst, lam), np.random.default_rng(S.SEED), B=500)})
        keys = list(D); rng = np.random.default_rng(0)
        for rep in range(200):
            perm = rng.permutation(len(keys)); ck = [keys[i] for i in perm[:700]]; tk = [keys[i] for i in perm[700:]]
            cal, tst = [D[k] for k in ck], [D[k] for k in tk]
            for al in S.ALPHAS:
                lam = policies(cal, prot, w, al)
                for pol in ("B", "C"):
                    ref.append({"w": w, "rep": rep, "alpha": al, "policy": pol,
                                "case_error": float(np.mean([((o["p"] > lam[pol]) & ~o["cvs"]).any() for o in tst]))})
        print("w", w, "done", flush=True)
    df = pd.DataFrame(rows); df.to_csv(S.OUT / "results_full.csv", index=False)
    rf = pd.DataFrame(ref); rf.to_csv(S.OUT / "resplits_full.csv", index=False)
    tables(df, rf)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    print(df[["w", "analysis", "alpha", "policy", "n_cal", "n_test", "cases_with_false_conf", "case_error", "case_error_lo",
              "case_error_hi", "sens", "precision", "cvs_cases", "cvs_cases_confirmed"]].round(3).to_string())
    print(rf.groupby(["w", "alpha", "policy"]).case_error.describe(percentiles=[.05, .5, .95]).round(3).to_string())


def tables(df, rf):
    f = lambda x, d=1: "--" if pd.isna(x) else f"{100 * x:.{d}f}"
    d = df[df.w == S.WINDOWS[0]]
    L = []
    for al in S.ALPHAS:
        for pol, lab in (("A", "A: Endoscapes threshold"), ("B", "B: frame-level CRC"), ("C", "C: ours")):
            r = d[(d.analysis.str.startswith("F1")) & np.isclose(d.alpha, al) & (d.policy == pol)].iloc[0]
            L.append(f"{al:.2f}".replace("0.", ".") + f" & {lab} & {r.cases_with_false_conf}/{r.n_test} & {f(r.case_error)} "
                     f"[{f(r.case_error_lo)}, {f(r.case_error_hi)}] & {f(r.sens)} [{f(r.sens_lo, 0)}, {f(r.sens_hi, 0)}] & "
                     f"{f(r.precision, 0)} & {r.cvs_cases_confirmed}/{r.cvs_cases} \\\\")
    (ROOT / "paper" / "tables" / "supp" / "sages_full.tex").write_text("\n".join(L) + "\n")
    L = []
    x = d[d.analysis.str.startswith("F2") & np.isclose(d.alpha, 0.10)]
    for name in sorted(x.analysis.unique(), key=lambda n: int(n.split()[-1])):
        c = x[(x.analysis == name) & (x.policy == "C")].iloc[0]; b = x[(x.analysis == name) & (x.policy == "B")].iloc[0]
        L.append(f"{name.split()[-1]} & {c.n_test} & {c.cases_with_false_conf} & {f(c.case_error)} [{f(c.case_error_lo, 0)}, {f(c.case_error_hi, 0)}] & "
                 f"{f(c.sens)} & {f(b.case_error)} \\\\")
    (ROOT / "paper" / "tables" / "supp" / "sages_country.tex").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["declare", "fetch", "infer", "evaluate"])
    a = ap.parse_args()
    {"declare": declare, "fetch": fetch, "infer": infer, "evaluate": evaluate}[a.stage](a)


if __name__ == "__main__":
    main()
