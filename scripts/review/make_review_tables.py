"""LaTeX tables for the supplementary material and paper/review_numbers.json, from the CSVs
in experiments/review (no hand-typed numbers).

    python scripts/review/make_review_tables.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "experiments" / "review"
OUT = ROOT / "paper" / "tables" / "supp"; OUT.mkdir(parents=True, exist_ok=True)
NUM = {}
pc = lambda x: "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}"


def write(name, lines):
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n")


def selective():
    d = pd.read_csv(R / "selective.csv"); d = d[d.arch == "dinov2l_b8"]
    s = pd.read_csv(R / "summaries.csv"); s = s[(s.arch == "dinov2l_b8") & (s.method == "surgery-level") & (s.k == 0) & (s.r == 1)]
    lines = []
    for a in (0.05, 0.10, 0.20):
        for w in (1, 10):
            b = s[(s.alpha == a) & (s.w == w)][["ops_false_ge1", "ops_false_gt1", "sens", "fdr"]].mean()
            lines.append(f"{a:.2f} & Surgery-level, {w}\\,s (ours) & -- & 100 & {pc(b.ops_false_ge1)} & {pc(b.ops_false_gt1)} & {pc(b.sens)} & {pc(b.fdr)} \\\\")
        for xi in (0.5, 0.75, 0.9):
            x = d[(d.alpha == a) & (d.method == "SCRC-I (frames)") & (d.xi == xi)].mean(numeric_only=True)
            lines.append(f" & SCRC-I~\\cite{{xu2025selective}}, frames & {xi:.2f} & 100 & {pc(x.ops_false_ge1)} & {pc(x.ops_false_gt1)} & {pc(x.sens)} & {pc(x.fdr)} \\\\")
            NUM[f"scrc_i|{a}|{xi}"] = {"ops": x.ops_false_ge1, "sens": x.sens, "fdr": x.fdr}
        for xi in (0.75, 0.9):
            for k in (0, 1):
                x = d[(d.alpha == a) & (d.method == "Selective surgery-level (ours)") & (d.xi == xi) & (d.w == 10) & (d.k == k)].mean(numeric_only=True)
                lines.append(f" & Frame-selective surgery-level, $k$={k} (ours) & {xi:.2f} & 100 & {pc(x.ops_false_ge1)} & {pc(x.ops_false_gt1)} & {pc(x.sens)} & {pc(x.fdr)} \\\\")
                NUM[f"sel_surg|{a}|{xi}|k{k}"] = {"ops": x.ops_false_ge1, "ops_gt1": x.ops_false_gt1, "sens": x.sens}
        for xi in (0.5, 0.75, 0.9):
            x = d[(d.alpha == a) & (d.method == "Operation-selective surgery-level (ours)") & (d.xi == xi) & (d.w == 10)].mean(numeric_only=True)
            lines.append(f" & Operation-selective (ours) & {xi:.2f} & {pc(x.op_accept)} & {pc(x.ops_false_ge1_given_acc)}$^\\dagger$ & -- & {pc(x.sens_given_acc)}$^\\dagger$ & {pc(x.fdr)} \\\\")
            NUM[f"op_sel|{a}|{xi}"] = {"accept": x.op_accept, "ops_given_acc": x.ops_false_ge1_given_acc,
                                       "sens_given_acc": x.sens_given_acc, "ops_marginal": x.ops_false_ge1, "sens": x.sens}
        lines.append("\\midrule")
    write("selective", lines[:-1])


def summaries():
    s = pd.read_csv(R / "summaries.csv")
    lines = []
    d = s[s.arch == "dinov2l_b8"]
    for a in (0.10, 0.20):
        for (w, r), g in d[(d.alpha == a) & (d.method == "surgery-level") & (d.k == 0)].groupby(["w", "r"]):
            x = g.mean(numeric_only=True)
            lines.append(f"{a:.2f} & max, smooth {w}\\,s, persist {r} & {pc(x.ops_false_ge1)} & {pc(x.ops_false_gt1)} & {pc(x.sens)} & {pc(x.fdr)} \\\\")
            NUM[f"summ|{a}|w{w}|r{r}"] = {"ops": x.ops_false_ge1, "sens": x.sens}
        for k in (1, 2):
            x = d[(d.alpha == a) & (d.method == "surgery-level") & (d.k == k) & (d.w == 10) & (d.r == 1)].mean(numeric_only=True)
            lines.append(f" & $(k{{+}}1)$-th largest, $k$={k}, smooth 10\\,s & {pc(x.ops_false_ge1)} & {pc(x.ops_false_gt1)} & {pc(x.sens)} & {pc(x.fdr)} \\\\")
        for m in ("surgery-level, summary chosen (split)", "window, one per op", "window-pooled (CRC)"):
            x = d[(d.alpha == a) & (d.method == m)].mean(numeric_only=True)
            lab = {"surgery-level, summary chosen (split)": "summary chosen on half the calibration ops",
                   "window, one per op": "60\\,s windows, one per operation",
                   "window-pooled (CRC)": "60\\,s windows pooled (CRC)"}[m]
            lines.append(f" & {lab} & {pc(x.ops_false_ge1)} & {pc(x.ops_false_gt1)} & {pc(x.sens)} & {pc(x.fdr)} \\\\")
            NUM[f"summ|{a}|{m}"] = {"ops": x.ops_false_ge1, "sens": x.sens}
        lines.append("\\midrule")
    write("summaries", lines[:-1])
    # best variant per backbone at alpha = 0.1
    b = s[(s.method == "surgery-level") & (s.k == 0) & (s.alpha == 0.1)].groupby(["arch", "w", "r"])[["ops_false_ge1", "sens"]].mean()
    NUM["summ_backbones_max_ops"] = float(b.ops_false_ge1.max())


def annotators():
    d = pd.read_csv(R / "annotators.csv")
    d = d[(d.alpha == 0.1) & (d.w == 1)]
    evals = ("majority", "rater1", "rater2", "rater3", "every surgeon")
    lines = []
    for c in ("majority", "rater1", "rater2", "rater3", "any", "unanimous", "random rater"):
        if c == "random rater":
            x = d[(d.calib == c)].mean(numeric_only=True)
            lines.append(f"random surgeon per operation & \\multicolumn{{5}}{{c}}{{{pc(x.ops_false_ge1)} (vs.\\ a random surgeon)}} & {pc(x.sens)} & -- \\\\")
            NUM["ann|random"] = x.ops_false_ge1
            continue
        cells = []
        for e in evals:
            x = d[(d.calib == c) & (d["eval"] == e)].mean(numeric_only=True)
            v = pc(x.ops_false_ge1)
            cells.append(f"\\textbf{{{v}}}" if x.ops_false_ge1 <= 0.1 + 1e-9 and (e == c or (c == "unanimous" and e == "every surgeon")) else v)
            NUM[f"ann|{c}|{e}"] = x.ops_false_ge1
        sens = d[(d.calib == c) & (d["eval"] == "majority")].sens.mean()
        share = d[(d.calib == c) & (d["eval"] == "every surgeon")].share_conf_on_disagreement.mean()
        NUM[f"ann|{c}|sens"] = sens; NUM[f"ann|{c}|share_dis"] = share
        lab = {"unanimous": "all three surgeons (every-surgeon)", "any": "lenient (any surgeon per criterion)"}.get(c, c)
        lines.append(f"{lab} & " + " & ".join(cells) + f" & {pc(sens)} & {pc(share)} \\\\")
    write("annotators", lines)


def shift():
    d = pd.read_csv(R / "shift.csv")
    lines = []
    order = [("As-is (Endoscapes)", 0), ("Weighted (covariate shift)", 0), ("Few-shot recalibration", 5),
             ("Few-shot recalibration", 10), ("Few-shot recalibration", 20), ("Pooled (unweighted)", 10),
             ("Pooled + weighted", 10), ("Full recalibration (37)", 37)]
    for m, n in order:
        cells = []
        for dec, w in (("stream", 1), ("stream", 10), ("time-out", 1), ("time-out", 10)):
            for a in (0.1, 0.2):
                x = d[(d.decision == dec) & (d.w == w) & (d.alpha == a) & (d.method == m) & (d.n == n)].mean(numeric_only=True)
                v = pc(x.ops_false); cells.append(f"\\textbf{{{v}}}" if x.ops_false > a + 1e-9 else v)
                NUM[f"shift|{dec}|{w}|{a}|{m}|{n}"] = {"ops": x.ops_false, "sens": x.sens}
        lab = m if n in (0, 37) else f"{m}, $n$={n}"
        lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    write("shift", lines)


def anatomy():
    d = pd.read_csv(R / "anatomy_robustness.csv"); d["alpha"] = d.alpha.fillna(-1)
    key = ["variant", "level", "degrade_cal", "assign", "K"]
    g = d.groupby(key + ["method", "alpha"])[["ops_err", "precision", "recall", "recall_e2e"]].mean().reset_index()
    order = [("base", 0, True, "argmax", True, "base (all regions, IoU 0.5, argmax per structure)"),
             ("K", 0, True, "argmax", False, "point world without one-region-per-structure"),
             ("assign", 0, True, "all", True, "overlay every region above threshold"),
             ("assign", 0, True, "map", True, "candidate = point-world region"),
             ("min_fg", 0.3, True, "argmax", True, "foreground $\\ge$ 0.3"),
             ("min_fg", 0.5, True, "argmax", True, "foreground $\\ge$ 0.5"),
             ("iou", 0.3, True, "argmax", True, "truth at IoU 0.3"),
             ("iou", 0.7, True, "argmax", True, "truth at IoU 0.7"),
             ("drop", 0.2, False, "argmax", True, "drop 20\\% of regions, test only"),
             ("drop", 0.3, False, "argmax", True, "drop 30\\% of regions, test only"),
             ("drop", 0.3, True, "argmax", True, "drop 30\\% of regions, recalibrated"),
             ("noise", 0.1, False, "argmax", True, "class-prob.\\ noise 0.1, test only"),
             ("noise", 0.3, True, "argmax", True, "class-prob.\\ noise 0.3, recalibrated"),
             ("jitter", 0.1, True, "argmax", True, "box jitter 10\\%"),
             ("jitter", 0.2, True, "argmax", True, "box jitter 20\\%")]
    lines = []
    worst = 0
    for v, l, c, a, K, lab in order:
        sel = g[(g.variant == v) & (g.level == l) & (g.degrade_cal == c) & (g["assign"] == a) & (g.K == K)]
        pw = sel[sel.method == "Point world"].iloc[0]
        fl = sel[(sel.method == "Frame-level (projected, Tab. 2)") & (sel.alpha == 0.1)].iloc[0]
        ft = sel[(sel.method == "Frame-level (threshold)") & (sel.alpha == 0.1)].iloc[0]
        cells = [pc(pw.ops_err), pc(fl.ops_err), pc(ft.ops_err)]
        for al in (0.05, 0.1, 0.2):
            s = sel[(sel.method == "Surgery-level (ours)") & (sel.alpha == al)].iloc[0]
            cells.append(pc(s.ops_err)); worst = max(worst, s.ops_err - al)
        s = sel[(sel.method == "Surgery-level (ours)") & (sel.alpha == 0.1)].iloc[0]
        cells += [pc(s.precision), pc(s.recall_e2e)] if s.recall > 0 else ["\\multicolumn{2}{c}{abstains}"]
        lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
        NUM[f"anat|{v}|{l}|{c}|{a}|{K}"] = {"pw": pw.ops_err, "frame": fl.ops_err, "surg": s.ops_err,
                                            "prec": s.precision, "rec": s.recall}
    NUM["anat|worst_excess"] = worst
    allv = g[(g.method == "Surgery-level (ours)")]
    NUM["anat|surg_range_0.1"] = [float(allv[allv.alpha == 0.1].ops_err.min()), float(allv[allv.alpha == 0.1].ops_err.max())]
    NUM["anat|pw_range"] = [float(g[g.method == "Point world"].ops_err.min()), float(g[g.method == "Point world"].ops_err.max())]
    fp = g[(g.method == "Frame-level (projected, Tab. 2)") & (g.alpha == 0.1)].ops_err
    NUM["anat|frame_range_0.1"] = [float(fp.min()), float(fp.max())]
    write("anatomy", lines)


def timeout():
    d = pd.read_csv(R / "timeout_stability.csv")
    d["mu_f"] = d.mu.replace([-np.inf, np.inf], np.nan)
    lines = []
    for w in (1, 5):
        for a in (0.05, 0.1, 0.2):
            for gk in ("fine", "coarse"):
                x = d[(d.w == w) & (d.alpha == a) & (d.grid == gk)]
                q = lambda s, p: s.quantile(p)
                mu_inf = (x.mu == np.inf).mean(); mu_ninf = (x.mu == -np.inf).mean()
                lines.append(f"{w} & {a:.2f} & {gk} & {x.lam_c.mean():.3f} [{q(x.lam_c, .05):.3f}, {q(x.lam_c, .95):.3f}] & "
                             f"{x.mu_f.mean():.3f} [{q(x.mu_f, .05):.3f}, {q(x.mu_f, .95):.3f}] & {pc(mu_ninf)} & {pc(mu_inf)} & "
                             f"{pc(x.false_confirm_given_noCVS.mean())} & {pc(x.false_alarm_joint.mean())} & "
                             f"{pc(x.confirm_given_CVS.mean())} & {pc(x.defer.mean())} & {pc(x.flip_rate_bootstrap.mean())} \\\\")
                NUM[f"to|{w}|{a}|{gk}"] = {"lam_c": x.lam_c.mean(), "lam_c_q05": q(x.lam_c, .05), "lam_c_q95": q(x.lam_c, .95),
                                          "mu": x.mu_f.mean(), "mu_inf": mu_inf, "fc": x.false_confirm_given_noCVS.mean(),
                                          "fa": x.false_alarm_joint.mean(), "conf": x.confirm_given_CVS.mean(),
                                          "defer": x.defer.mean(), "flip": x.flip_rate_bootstrap.mean(),
                                          "n_pos": x.n_pos_cal.mean()}
    write("timeout", lines)


def temporal():
    ca = pd.read_csv(ROOT / "experiments" / "compare_all" / "compare_all.csv")
    ct = pd.read_csv(ROOT / "experiments" / "compare_all" / "compare_temporal.csv")
    d = pd.concat([ca[ca.run.str.startswith("dinov2l_b8")], ct])
    d["arch"] = d.run.str.replace(r"_s\d+$", "", regex=True)
    rows = [("dinov2l_b8", "DINOv2-L, per frame"), ("temporal_lstm_ft", "+ causal LSTM"),
            ("temporal_tcn_ft", "+ causal TCN"), ("temporal_transformer_ft", "+ causal transformer"),
            ("temporal_lstm_frozen", "frozen DINOv2-L + LSTM"), ("temporal_tcn_frozen", "frozen + TCN"),
            ("temporal_transformer_frozen", "frozen + transformer")]
    lines = []
    for arch, lab in rows:
        ap = json.loads((ROOT / "paper" / "numbers.json").read_text()).get(f"cvs_ap_{arch}" if arch != "dinov2l_b8" else "cvs_ap_dinov2")
        cells = [f"{100 * ap['mAP']:.1f}"]
        for a in (0.05, 0.1, 0.2):
            x = d[(d.arch == arch) & (d.alpha == a)]
            s_ = x[x.method == "Surgery-level (ours)"][["ops_false_ge1", "sens"]].mean()
            c_ = x[x.method == "CRC (= frame-level)"].ops_false_ge1.mean()
            cells += [pc(s_.ops_false_ge1), pc(s_.sens), pc(c_)]
            NUM[f"temporal|{arch}|{a}"] = {"surg_ops": s_.ops_false_ge1, "surg_sens": s_.sens, "crc_ops": c_}
        lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
        if arch in ("dinov2l_b8", "temporal_transformer_ft"):
            lines.append("\\midrule")
    write("temporal", lines)


def rnd(x):
    if isinstance(x, dict):
        return {k: rnd(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return round(float(x), 4)
    return x


if __name__ == "__main__":
    selective(); summaries(); annotators(); shift(); anatomy(); timeout(); temporal()
    (ROOT / "paper" / "review_numbers.json").write_text(json.dumps(rnd(NUM), indent=1))
    for f in sorted(OUT.glob("*.tex")):
        print("==", f.name); print(f.read_text())
