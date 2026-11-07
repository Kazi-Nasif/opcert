"""Review round 4 (second evaluation), three analyses on existing cached predictions:

  timeout_equiv  At the time-out there is one eligible frame per operation, so class-conditional CRC
                 fitted on the time-out scores of the calibration operations WITHOUT CVS (grid of
                 observed scores) must reproduce our CONFIRM threshold.  Checked on the same halvings
                 as Table 3 (Endoscapes, 200 x 3 seeds; Cholec80 recalibrated, 100 x 3), 5 s smoothing.
  calsize_sages  Calibration size with the test set held fixed: draw V in {100, 200, 400} cases from the
                 700 SAGES official-training cases (500 draws each) and evaluate policies B and C on
                 the same 300 official-test cases; compare with V = 700.
  outcomes       Three mutually exclusive outcomes per test operation on fixed splits, before the first
                 clip: (F) >= 1 false confirmation; (S) >= 1 correct confirmation and no false one;
                 (N) no confirmation.  Time remaining before the first clip at the first correct
                 confirmation for (S).  Endoscapes official split (31 -> 40, 3 seeds, 10 s), SAGES
                 700 -> 300 (10 s; clip at 90 s, last rated timestamp 85 s).

    python scripts/review4/review4.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "scripts", "scripts/review", "scripts/review2", "scripts/review3", "scripts/external"):
    sys.path.insert(0, str(ROOT / p))
from cdss import riskcontrol as RC                                           # noqa: E402
from cdss.data import keyframes                                              # noqa: E402
from common import endoscapes_ops, halves, order_stat, temperature           # noqa: E402
import sages_external as S                                                   # noqa: E402
import sages_full as SF                                                      # noqa: E402
from timeout_final import cholec80_ops as c80_ops                            # noqa: E402

OUT = ROOT / "experiments" / "review4"; OUT.mkdir(parents=True, exist_ok=True)
ALPHAS = (0.05, 0.10)


def crc_cc(neg, a):
    """Class-conditional CRC on the grid of observed scores, loss 1[p > lam]."""
    grid = np.r_[np.unique(neg), np.inf]
    L = (neg[:, None] > grid[None, :]).astype(float)
    g = RC.crc_index(L, a)
    return np.inf if g is None else float(grid[g])


def timeout_equiv():
    rows = []
    for seed in (0, 1, 2):
        T = temperature(ROOT / "experiments" / "finetune" / f"dinov2l_b8_s{seed}")
        E = endoscapes_ops("dinov2l_b8", seed, (5,))
        E = [{"p": o["p"][5][-1], "b": bool(o["cvs"][-1])} for o in E]
        C = [{"p": o["p"][-1], "b": bool(o["ref"]["dataset"][-1])} for o in c80_ops(seed, 5, T)]
        for name, ops, reps in (("Endoscapes", E, 200), ("Cholec80 (recal.)", C, 100)):
            for rep, ci, ti in halves(len(ops), reps):
                cal = [ops[i] for i in ci]; tst = [ops[i] for i in ti]
                neg = np.array([o["p"] for o in cal if not o["b"]]); pt = np.array([o["p"] for o in tst])
                for a in (0.05, 0.10, 0.20):
                    lo, lc = order_stat(neg, a), crc_cc(neg, a)
                    rows.append({"dataset": name, "seed": seed, "rep": rep, "alpha": a, "lam_ours": lo, "lam_crc": lc,
                                 "same_threshold": bool(lo == lc), "same_decisions": bool(((pt > lo) == (pt > lc)).all())})
    d = pd.DataFrame(rows); d.to_csv(OUT / "timeout_equiv.csv", index=False)
    g = d.groupby(["dataset", "alpha"])[["same_threshold", "same_decisions"]].mean()
    print(g.to_string()); return g


def calsize_sages():
    prot = S.protocol(); T = prot["model"]["temperature"]
    D = SF.load_all(T, 10)
    cal = [D[k] for k in D if k[0] == "train"]; tst = [D[k] for k in D if k[0] == "test"]
    rng = np.random.default_rng(0); rows = []
    for V in (100, 200, 400, 700):
        for rep in range(500 if V < 700 else 1):
            sub = cal if V == 700 else [cal[i] for i in rng.choice(len(cal), V, replace=False)]
            for a in ALPHAS:
                lam = SF.policies(sub, prot, 10, a)
                for pol in ("B", "C"):
                    err = np.mean([((o["p"] > lam[pol]) & ~o["cvs"]).any() for o in tst])
                    tp = sum(((o["p"] > lam[pol]) & o["cvs"]).sum() for o in tst); pos = sum(o["cvs"].sum() for o in tst)
                    rows.append({"V": V, "rep": rep, "alpha": a, "policy": pol, "case_error": err, "sens": tp / pos,
                                 "finite": math.isfinite(lam[pol])})
    d = pd.DataFrame(rows); d.to_csv(OUT / "calsize_sages.csv", index=False)
    g = d.groupby(["alpha", "policy", "V"]).agg(err=("case_error", "mean"), err_q05=("case_error", lambda x: x.quantile(.05)),
                                                err_q95=("case_error", lambda x: x.quantile(.95)),
                                                above=("case_error", lambda x: np.nan), sens=("sens", "mean"), finite=("finite", "mean"))
    for (a, pol, V), _ in g.iterrows():
        x = d[(d.alpha == a) & (d.policy == pol) & (d.V == V)]
        g.loc[(a, pol, V), "above"] = float((x.case_error > a).mean())
    print(g.round(3).to_string()); return g


def outcomes_table(tst_ops, lam, clip_time):
    """tst_ops: list of dicts with p (scores), cvs (bool), t (seconds); clip_time(o) -> seconds of first clip."""
    rows = []
    for o in tst_ops:
        c = o["p"] > lam; y = o["cvs"]
        false = bool((c & ~y).any()); hit = np.flatnonzero(c & y)
        if false:
            k = "F"
        elif len(hit):
            k = "S"
        else:
            k = "N"
        rows.append({"has_cvs": bool(y.any()), "outcome": k,
                     "lead_s": float(clip_time(o) - o["t"][hit[0]]) if (k == "S" and len(hit)) else np.nan})
    return pd.DataFrame(rows)


def summarise_outcomes(df, label):
    out = []
    for grp, g in (("with CVS", df[df.has_cvs]), ("without CVS", df[~df.has_cvs]), ("all", df)):
        n = len(g); cnt = g.outcome.value_counts()
        leads = g.lead_s.dropna()
        out.append({"split": label, "group": grp, "n": n, "F": int(cnt.get("F", 0)), "S": int(cnt.get("S", 0)), "N": int(cnt.get("N", 0)),
                    "lead_median_s": float(leads.median()) if len(leads) else np.nan,
                    "lead_q25_s": float(leads.quantile(.25)) if len(leads) else np.nan,
                    "lead_q75_s": float(leads.quantile(.75)) if len(leads) else np.nan})
    return out


def outcomes():
    rows = []
    te = set(keyframes("test").vid.tolist())
    for a in ALPHAS:
        agg = []
        for seed in (0, 1, 2):
            ops = endoscapes_ops("dinov2l_b8", seed, (10,))
            cal = [o for o in ops if o["vid"] not in te]; tst = [o for o in ops if o["vid"] in te]
            m = [o["p"][10][~o["cvs"]].max() for o in cal if (~o["cvs"]).any()]
            lam = order_stat(m, a)
            T = [{"p": o["p"][10], "cvs": o["cvs"], "t": o["frame"] / 25.0} for o in tst]
            agg.append(outcomes_table(T, lam, lambda o: o["t"][-1]))
        d = pd.concat(agg)                                   # 3 seeds x 40 operations
        for r in summarise_outcomes(d, f"Endoscapes official (3 seeds)|{a}"):
            r["alpha"] = a; rows.append(r)
    prot = S.protocol(); T = prot["model"]["temperature"]; D = SF.load_all(T, 10)
    cal = [D[k] for k in D if k[0] == "train"]; tst = [D[k] for k in D if k[0] == "test"]
    lab = {}
    for (split, c) in D:
        if split == "test":
            lab[c] = pd.read_csv(S.DATA / split / "labels" / c / "frame.csv").sort_values("frame_id").frame_id.to_numpy() / S.CLIP_FPS
    tk = [k for k in D if k[0] == "test"]
    for a in ALPHAS:
        lam = SF.policies(cal, prot, 10, a)["C"]
        Tt = [{"p": D[k]["p"], "cvs": D[k]["cvs"], "t": lab[k[1]]} for k in tk]
        d = outcomes_table(Tt, lam, lambda o: 90.0)
        for r in summarise_outcomes(d, f"SAGES 700 -> 300|{a}"):
            r["alpha"] = a; rows.append(r)
    d = pd.DataFrame(rows); d.to_csv(OUT / "outcomes.csv", index=False)
    pd.set_option("display.width", 200); print(d.round(1).to_string()); return d


def tables(eq, cs, oc):
    T = ROOT / "paper" / "tables" / "supp"
    # outcomes table: one block per split, rows: with CVS / without CVS
    L = []
    for sp in dict.fromkeys(x.split("|")[0] for x in oc.split):
        for a in ALPHAS:
            for grp in ("with CVS", "without CVS"):
                r = oc[oc.split.str.startswith(sp) & np.isclose(oc.alpha, a) & (oc.group == grp)].iloc[0]
                lead = "--" if np.isnan(r.lead_median_s) else f"{r.lead_median_s:.0f} [{r.lead_q25_s:.0f}, {r.lead_q75_s:.0f}]"
                name = sp.replace(" -> ", " $\\to$ ")
                L.append(f"{name} & {a:.2f}".replace("0.", ".") + f" & {grp} & {r.n} & {r.F} & {r.S} & {r.N} & {lead} \\\\")
        L.append("\\midrule")
    (T / "outcomes.tex").write_text("\n".join(L[:-1]) + "\n")
    L = []
    for a in ALPHAS:
        for pol, lab in (("C", "ours"), ("B", "frame-level CRC")):
            cells = []
            for V in (100, 200, 400, 700):
                r = cs.loc[(a, pol, V)]
                cells.append(f"{100 * r.err:.1f}" if V == 700 else f"{100 * r.err:.1f} [{100 * r.err_q05:.1f}, {100 * r.err_q95:.1f}]")
            L.append(f"{a:.2f}".replace("0.", ".") + f" & {lab} & " + " & ".join(cells) + " \\\\")
    (T / "calsize_sages.tex").write_text("\n".join(L) + "\n")
    json.dump({"timeout_equiv": {f"{k[0]}|{k[1]}": v for k, v in eq.to_dict("index").items()},
               "calsize": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in cs.to_dict("index").items()},
               "outcomes": oc.to_dict("records")}, open(OUT / "numbers_r4.json", "w"), indent=1, default=float)


def main():
    if "--tables" in sys.argv:                      # rebuild tables from the saved CSVs
        eq = pd.read_csv(OUT / "timeout_equiv.csv").groupby(["dataset", "alpha"])[["same_threshold", "same_decisions"]].mean()
        d = pd.read_csv(OUT / "calsize_sages.csv")
        cs = d.groupby(["alpha", "policy", "V"]).agg(err=("case_error", "mean"), err_q05=("case_error", lambda x: x.quantile(.05)),
                                                    err_q95=("case_error", lambda x: x.quantile(.95)), sens=("sens", "mean"))
        for (a, pol, V), _ in cs.iterrows():
            x = d[(d.alpha == a) & (d.policy == pol) & (d.V == V)]; cs.loc[(a, pol, V), "above"] = float((x.case_error > a).mean())
        tables(eq, cs, pd.read_csv(OUT / "outcomes.csv")); return
    eq = timeout_equiv(); cs = calsize_sages(); oc = outcomes(); tables(eq, cs, oc)


if __name__ == "__main__":
    main()
