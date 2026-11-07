"""LaTeX tables for review round 3 from experiments/review3/*.csv and experiments/phase/*.csv.

    python scripts/review3/make_tables_r3.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta as B

ROOT = Path(__file__).resolve().parents[2]
R3 = ROOT / "experiments" / "review3"
TAB = ROOT / "paper" / "tables"
NUM = {}


def pct(x, d=1):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}"


def write(name, lines):
    p = TAB / f"{name}.tex"; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n"); print("wrote", p)


def timeout_agg(d):
    g = d.groupby(["dataset", "calib", "eval", "policy", "alpha_c"], dropna=False).agg(
        conf_noCVS=("conf_noCVS", "mean"), conf_CVS=("conf_CVS", "mean"), stop_CVS=("stop_CVS", "mean"),
        stop_noCVS=("stop_noCVS", "mean"), defer=("defer", "mean"), fdr=("fdp", "mean"), prec=("precision", "mean"),
        any_conf=("any_conf", "mean"), stop_prec=("stop_precision", "mean"), tp=("tp", "mean"), fp=("fp", "mean"),
        n_cvs=("n_cvs", "mean"), n_ops=("n_ops", "mean"), stop_possible=("stop_possible", "mean")).reset_index()
    g["prec_pooled"] = g.tp / (g.tp + g.fp)
    return g


def get(g, ds, policy, a, calib=None, ev=None):
    calib = calib or ("-" if policy == "point world" else ("majority" if ds == "Endoscapes" else "dataset"))
    ev = ev or ("majority" if ds == "Endoscapes" else "dataset")
    m = (g.dataset == ds) & (g.policy == policy) & (g.calib == calib) & (g["eval"] == ev)
    m &= g.alpha_c.isna() if a is None else np.isclose(g.alpha_c, a)
    r = g[m]
    assert len(r) == 1, (ds, policy, a, calib, ev, len(r))
    return r.iloc[0]


def timeout_tables():
    df = pd.read_csv(R3 / "timeout_final.csv"); d = df[df.smooth_s == 5]
    g = timeout_agg(d)
    rows = [("Point world", "point world", None, "--"), ("Frame CRC", "frame CRC", 0.05, ".05"), (None, "frame CRC", 0.10, ".10"),
            ("Frame LTT", "frame LTT", 0.10, ".10"), ("Conf.\\ sel.", "conf. selection", 0.20, ".20"),
            ("\\textbf{Ours}", "ours", 0.05, ".05"), (None, "ours", 0.10, ".10"), (None, "ours", 0.20, ".20")]
    L = []
    for label, pol, a, astr in rows:
        e, c = get(g, "Endoscapes", pol, a), get(g, "Cholec80 (recal.)", pol, a)
        cells = []
        for r in (e, c):
            bold = pol == "ours"
            cn = pct(r.conf_noCVS)
            cells += [f"\\textbf{{{cn}}}" if bold else cn, pct(r.conf_CVS), pct(r.prec_pooled, 0),
                      pct(r.stop_CVS) if pol in ("ours", "point world") else "--"]
        L.append(f"{label or ''} & {astr} & " + " & ".join(cells) + " \\\\")
        if pol == "frame LTT" or pol == "conf. selection":
            pass
    write("timeout", L)
    for k, (ds, pol, a) in {"e_ours10": ("Endoscapes", "ours", .10), "e_ours05": ("Endoscapes", "ours", .05),
                            "c_ours10": ("Cholec80 (recal.)", "ours", .10), "e_crc10": ("Endoscapes", "frame CRC", .10),
                            "e_crc05": ("Endoscapes", "frame CRC", .05), "e_crc20": ("Endoscapes", "frame CRC", .20),
                            "e_ltt10": ("Endoscapes", "frame LTT", .10), "c_crc10": ("Cholec80 (recal.)", "frame CRC", .10),
                            "e_sel20": ("Endoscapes", "conf. selection", .20), "e_pw": ("Endoscapes", "point world", None),
                            "c_pw": ("Cholec80 (recal.)", "point world", None)}.items():
        r = get(g, ds, pol, a)
        NUM[k] = {c: round(float(r[c]), 4) for c in ("conf_noCVS", "conf_CVS", "stop_CVS", "stop_noCVS", "defer", "prec",
                                                         "prec_pooled", "fdr", "any_conf", "stop_prec", "tp", "fp", "n_cvs", "n_ops")}
    # full table: ours and baselines by alpha, both datasets, with counts and aggregation variants
    L = []
    for ds in ("Endoscapes", "Cholec80 (recal.)"):
        for pol in ("point world", "frame CRC", "frame LTT", "conf. selection", "ours"):
            for a in ([None] if pol == "point world" else [0.05, 0.10, 0.20]):
                r = get(g, ds, pol, a)
                L.append(f"{ds.replace(' (recal.)', '')} & {pol} & {'--' if a is None else a} & {r.n_ops:.0f} & {r.n_cvs:.1f} & "
                         f"{r.tp:.1f} & {r.fp:.1f} & {pct(r.conf_noCVS)} & {pct(r.conf_CVS)} & {pct(r.prec)} & {pct(r.prec_pooled)} & "
                         f"{pct(r.fdr)} & {pct(r.any_conf, 0)} & {pct(r.stop_CVS) if pol in ('ours', 'point world') else '--'} & "
                         f"{pct(r.stop_noCVS) if pol in ('ours', 'point world') else '--'} & "
                         f"{pct(r.stop_prec) if pol in ('ours', 'point world') else '--'} & {pct(r.defer)} \\\\")
        L.append("\\midrule")
    write("supp/timeout_full", L[:-1])
    # reference robustness (alpha_c = 0.10)
    L = []
    for ds, cals, evs in (("Endoscapes", ("majority", "unanimous"), ("majority", "surgeon1", "surgeon2", "surgeon3", "unanimous")),
                          ("Cholec80 (recal.)", ("dataset", "strict"), ("dataset", "strict"))):
        for cal in cals:
            for ev in evs:
                m = (g.dataset == ds) & (g.policy == "ours") & (g.calib == cal) & (g["eval"] == ev) & np.isclose(g.alpha_c, 0.10)
                if not m.any():
                    continue
                r = g[m].iloc[0]
                L.append(f"{ds.replace(' (recal.)', '')} & {cal} & {ev} & {r.n_cvs:.1f} & {pct(r.conf_noCVS)} & {pct(r.conf_CVS)} & "
                         f"{pct(r.prec_pooled)} & {pct(r.stop_CVS)} \\\\")
        L.append("\\midrule")
    write("supp/timeout_refs", L[:-1])
    # fixed official split, per seed
    fx = pd.read_csv(R3 / "timeout_fixed.csv"); f = fx[(fx.smooth_s == 5) & (fx["eval"] == "majority") & fx.calib.isin(["majority", "-"])]
    L = []
    for pol, a in (("point world", None), ("frame CRC", .10), ("frame LTT", .10), ("ours", .05), ("ours", .10), ("ours", .20)):
        s = f[(f.policy == pol) & (f.alpha_c.isna() if a is None else np.isclose(f.alpha_c, a))].sort_values("seed")
        cells = []
        for _, r in s.iterrows():
            n0 = int(r.n_ops - r.n_cvs); k = int(r.fp)
            hi = 1.0 if k == n0 else B.ppf(0.975, k + 1, n0 - k)
            cells.append(f"{k}/{n0} [{pct(hi, 0)}]; {int(r.tp)}/{int(r.n_cvs)}; {int(r.stop_on_cvs)}")
        L.append(f"{pol} & {'--' if a is None else a} & " + " & ".join(cells) + " \\\\")
    write("supp/timeout_fixed", L)


def stream_tables():
    df = pd.read_csv(R3 / "stream_extra.csv")
    a = df[(df.part == "a")].groupby(["w", "alpha", "method"])[
        ["ops_err", "ops_any_conf", "ops_err_given_conf", "frame_precision", "sens", "cvs_ops_confirmed", "delay_med", "lead_med"]].mean().reset_index()
    L = []
    for w in (1, 10):
        for al in (0.05, 0.10, 0.20):
            for m in ("ours", "ours PAC", "CRC frame-level", "LTT frame-level"):
                r = a[(a.w == w) & np.isclose(a.alpha, al) & (a.method == m)].iloc[0]
                L.append(f"{w} & {al} & {m} & {pct(r.ops_err)} & {pct(r.ops_any_conf)} & {pct(r.ops_err_given_conf)} & "
                         f"{pct(r.frame_precision)} & {pct(r.cvs_ops_confirmed)} & "
                         f"{'--' if np.isnan(r.delay_med) else f'{r.delay_med:.0f}'} & {'--' if np.isnan(r.lead_med) else f'{r.lead_med:.0f}'} \\\\")
            L.append("\\midrule")
    write("supp/stream_precision", L[:-1])
    r = a[(a.w == 10) & np.isclose(a.alpha, 0.10) & (a.method == "ours")].iloc[0]
    NUM["stream_ours10_w10"] = {k: round(float(r[k]), 4) for k in r.index if k not in ("method",)}
    r = a[(a.w == 10) & np.isclose(a.alpha, 0.10) & (a.method == "CRC frame-level")].iloc[0]
    NUM["stream_crc10_w10"] = {k: round(float(r[k]), 4) for k in r.index if k not in ("method",)}
    b = df[df.part == "b"].groupby(["alpha", "method", "V"])[["finite", "ops_err", "sens"]].mean().reset_index()
    L = []
    for al in (0.05, 0.10, 0.20):
        for m in ("ours marginal", "ours pac"):
            s = b[np.isclose(b.alpha, al) & (b.method == m)].sort_values("V")
            L.append(f"{al} & {m.replace('ours ', '')} & " + " & ".join(
                f"{pct(r.ops_err)}/{pct(r.sens, 0)}" if r.finite > 0 else "never" for _, r in s.iterrows()) + " \\\\")
    write("supp/calsize", L)
    fx = pd.read_csv(R3 / "fixed_splits.csv")
    g = fx.groupby(["split", "w", "alpha", "method"])[["V", "n_test", "k_err", "ops_err", "cp_hi", "sens"]].mean().reset_index()
    L = []
    for sp in ("official (val->test)", "official reversed", "video ID: lower->upper", "video ID: upper->lower"):
        for m in ("ours", "ours PAC", "CRC frame-level", "LTT frame-level"):
            s = g[(g.split == sp) & (g.w == 10) & (g.method == m)].sort_values("alpha")
            name = sp.replace("->", "$\\to$")
            L.append(f"{name} & {int(s.V.iloc[0])}/{int(s.n_test.iloc[0])} & {m} & " + " & ".join(
                f"{pct(r.ops_err)} [{pct(r.cp_hi, 0)}] / {pct(r.sens, 0)}" for _, r in s.iterrows()) + " \\\\")
        L.append("\\midrule")
    write("supp/nonrandom", L[:-1])
    NUM["nonrandom"] = {f"{sp}|{m}|{al}": round(float(g[(g.split == sp) & (g.w == 10) & (g.method == m) & np.isclose(g.alpha, al)].ops_err.iloc[0]), 4)
                        for sp in g.split.unique() for m in ("ours", "ours PAC", "CRC frame-level") for al in (0.05, 0.10, 0.20)}
    of = pd.read_csv(R3 / "onefps.csv")
    o = of.groupby(["w", "label", "alpha", "method"])[["ops_err_1fps", "ops_err_keyframes", "sens_1fps"]].mean().reset_index()
    L = []
    for w in (1, 10):
        for lab in ("hold", "cons"):
            for m in ("ours, calibrated on keyframes", "ours, calibrated at 1 fps", "CRC frame-level at 1 fps"):
                s = o[(o.w == w) & (o.label == lab) & (o.method == m)].sort_values("alpha")
                L.append(f"{w} & {'hold' if lab == 'hold' else 'conservative'} & {m} & " + " & ".join(
                    f"{pct(r.ops_err_1fps)} / {pct(r.sens_1fps, 0)}" for _, r in s.iterrows()) + " \\\\")
        L.append("\\midrule")
    write("supp/onefps", L[:-1])
    NUM["onefps"] = {f"{w}|{lab}|{m}|{al}": [round(float(x), 4) for x in o[(o.w == w) & (o.label == lab) & (o.method == m) & np.isclose(o.alpha, al)][["ops_err_1fps", "sens_1fps"]].iloc[0]]
                     for w in (1, 10) for lab in ("hold", "cons") for m in o.method.unique() for al in (0.05, 0.10, 0.20)}


def phase_tables():
    p = ROOT / "experiments" / "phase" / "phase_certify.csv"
    if not p.exists():
        print("phase results missing"); return
    df = pd.read_csv(p); df["alpha"] = df.alpha.fillna(-1)
    perc = pd.read_csv(ROOT / "experiments" / "phase" / "phase_perception.csv")
    NUM["phase_acc"] = {"mean": round(float(perc.frame_acc.mean()), 4), "sd": round(float(perc.frame_acc.std()), 4)}
    g = df.groupby(["task", "w", "alpha", "method"]).mean(numeric_only=True).reset_index()
    L = []
    for m, al, lab in (("point world", -1, "Point world"), ("CRC frame-level", 0.05, "CRC"), ("CRC frame-level", 0.10, ""),
                       ("LTT frame-level", 0.10, "LTT"), ("surgery-level (ours)", 0.05, "\\textbf{Ours}"),
                       ("surgery-level (ours)", 0.10, ""), ("surgery-level (ours)", 0.20, "")):
        A = g[(g.task == "A") & (g.w == 10) & (g.method == m) & np.isclose(g.alpha, al)].iloc[0]
        Bm = g[(g.task == "B") & (g.w == 10) & (g.method == m) & np.isclose(g.alpha, al)].iloc[0]
        bold = m.startswith("surgery")
        e1, e2 = pct(A.ops_err), pct(Bm.ops_err)
        if bold:
            e1, e2 = f"\\textbf{{{e1}}}", f"\\textbf{{{e2}}}"
        L.append(f"{lab} & {'--' if al < 0 else f'{al:.2f}'.lstrip('0')} & {e1} & {pct(A.frame_err)} & {pct(A.sens)} & "
                 f"{'--' if np.isnan(A.delay_med_s) else f'{A.delay_med_s:.0f}'} & {e2} & {pct(Bm.coverage)} & {pct(Bm.issued_acc)} \\\\")
    write("phase", L)
    L1 = []
    for m, al, lab in (("point world", -1, "Point world"), ("CRC frame-level", 0.05, "CRC"), ("CRC frame-level", 0.10, ""),
                       ("LTT frame-level", 0.10, "LTT"), ("surgery-level (ours)", 0.05, "\\textbf{Ours}"),
                       ("surgery-level (ours)", 0.10, ""), ("surgery-level (ours)", 0.20, "")):
        A = g[(g.task == "A") & (g.w == 1) & (g.method == m) & np.isclose(g.alpha, al)].iloc[0]
        Bm = g[(g.task == "B") & (g.w == 1) & (g.method == m) & np.isclose(g.alpha, al)].iloc[0]
        L1.append(f"{lab} & {'--' if al < 0 else f'{al:.2f}'.lstrip('0')} & {pct(A.ops_err)} & {pct(A.frame_err)} & {pct(A.sens)} & "
                  f"{'--' if np.isnan(A.delay_med_s) else f'{A.delay_med_s:.0f}'} & {pct(Bm.ops_err)} & {pct(Bm.coverage)} & {pct(Bm.issued_acc)} \\\\")
    write("supp/phase_w1", L1)
    NUM["phase"] = {f"{t}|{w}|{m}|{al}": {k: round(float(v), 4) for k, v in r.items() if k not in ("task", "w", "alpha", "method")}
                    for (t, w, al, m), r in g.set_index(["task", "w", "alpha", "method"]).iterrows()}


def main():
    timeout_tables(); stream_tables(); phase_tables()
    (R3 / "numbers_r3.json").write_text(json.dumps(NUM, indent=1))


if __name__ == "__main__":
    main()
