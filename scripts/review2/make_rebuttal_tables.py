"""LaTeX tables for the supplementary analyses added after the pre-submission reviews, generated
from the stored result files (no hand-typed numbers).

    python scripts/review2/make_rebuttal_tables.py   -> paper/tables/supp/{retarget,holdout,utility,
                                                       timeout_rules,cert_training,cholec80_strict}.tex
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "experiments" / "review"; OUT = ROOT / "paper" / "tables" / "supp"; OUT.mkdir(parents=True, exist_ok=True)
pc = lambda x: "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}"


def write(name, lines):
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n"); print("==", name); print("\n".join(lines))


def retarget():
    d = pd.read_csv(R / "rebuttal_stream.csv"); d = d[d.arch == "dinov2l_b8"]
    order = [("surgery-level", "Surgery-level (ours)"), ("op-CRC", "CRC over operations"),
             ("op-LTT", "LTT over operations, PAC (ours)"), ("op-PAC", "Beta order statistic, PAC (ours)"),
             ("CRC Bonferroni (N max)", "CRC at $\\alpha/N_{\\max}$"), ("LTT Bonferroni (N max)", "LTT at $\\alpha/N_{\\max}$"),
             ("CRC Bonferroni (N mean)", "CRC at $\\alpha/\\bar N$ (no guarantee)"), ("CRC frame-level", "CRC at $\\alpha$ (frame)")]
    lines = []
    for w in (1, 10):
        for m, lab in order:
            cells = []
            for a in (0.05, 0.1, 0.2):
                x = d[(d.w == w) & (d.method == m) & (d.alpha == a)][["ops_false_ge1", "sens"]].mean()
                cells.append(f"{pc(x.ops_false_ge1)} / {pc(x.sens)}" if x.sens > 0 else "nothing")
            lines.append(f"{w}\\,s & {lab} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    write("retarget", lines[:-1])


def holdout():
    d = pd.read_csv(R / "rebuttal_holdout.csv"); d = d[d.arch == "dinov2l_b8"]
    lines = []
    for w in (1, 10):
        for m, lab in (("surgery-level", "Surgery-level (ours)"), ("op-LTT", "PAC (ours)"),
                       ("CRC frame-level", "CRC (frame)"), ("LTT frame-level", "LTT (frame)"),
                       ("CRC Bonferroni (N max)", "CRC at $\\alpha/N_{\\max}$")):
            cells = []
            for a in (0.05, 0.1, 0.2):
                x = d[(d.w == w) & (d.method == m) & (d.alpha == a)].mean(numeric_only=True)
                cells.append(f"{pc(x.ops_err)} [{pc(x.ci_lo)}, {pc(x.ci_hi)}] / {pc(x.sens)}")
            lines.append(f"{w}\\,s & {lab} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    write("holdout", lines[:-1])


def utility():
    d = pd.read_csv(R / "rebuttal_stream.csv"); d = d[(d.arch == "dinov2l_b8") & (d.w == 10)]
    lines = []
    for m, a, lab in (("surgery-level", 0.05, "Surgery-level (ours), $\\alpha$=.05"), ("surgery-level", 0.1, "\\quad $\\alpha$=.10"),
                      ("surgery-level", 0.2, "\\quad $\\alpha$=.20"), ("op-LTT", 0.1, "PAC (ours), $\\alpha$=.10"),
                      ("CRC Bonferroni (N mean)", 0.2, "CRC at $\\alpha/\\bar N$, $\\alpha$=.20"),
                      ("CRC frame-level", 0.05, "CRC (frame), $\\alpha$=.05")):
        x = d[(d.method == m) & (d.alpha == a)].mean(numeric_only=True)
        lines.append(f"{lab} & {pc(x.ops_false_ge1)} & {pc(x.ops_cvs_confirmed)} & {x.delay_s_median:.0f} & {pc(x.persistence)} \\\\")
    write("utility", lines)


def timeout_rules():
    d = pd.read_csv(R / "timeout_alarm.csv"); d = d[d.smooth_s == 5]
    lines = []
    for ds in ("Endoscapes", "Cholec80 (recal.)"):
        for rule, a, lab in (("point world", None, "Point world"), ("joint", 0.1, "joint STOP, $\\alpha$=.10"),
                             ("conditional", 0.1, "conditional STOP, $\\alpha$=.10"),
                             ("conditional", 0.2, "conditional STOP, $\\alpha$=.20"),
                             ("asymmetric (confirm .10, STOP .20)", 0.1, "\\textbf{ours: confirm .10, STOP .20}")):
            x = d[(d.dataset == ds) & (d.rule == rule)]
            x = x[x.alpha.isna()] if a is None else x[x.alpha == a]
            x = x.mean(numeric_only=True)
            stop = pc(x.stop_possible) if rule not in ("point world", "joint") else "--"
            lines.append(f"{ds.split(' ')[0]} & {lab} & {pc(x.confirm_given_noCVS)} & {pc(x.confirm_given_CVS)} & "
                         f"{pc(x.alarm_given_CVS)} & {pc(x.alarm_given_noCVS)} & {pc(x.defer)} & {stop} \\\\")
        lines.append("\\midrule")
    write("timeout_rules", lines[:-1])


def cert_training():
    a = pd.read_csv(ROOT / "experiments" / "compare_all" / "compare_temporal.csv")
    b = pd.read_csv(ROOT / "experiments" / "compare_all" / "compare_temporal_cert.csv")
    d = pd.concat([a, b]); d["arch"] = d.run.str.replace(r"_s\d+$", "", regex=True)
    lines = []
    for arch, lab in (("temporal_tcn_ft", "causal TCN"), ("temporal_tcn_ft_cert", "\\quad + certification loss"),
                      ("temporal_lstm_ft", "causal LSTM"), ("temporal_lstm_ft_cert", "\\quad + certification loss")):
        ap = np.mean([json.loads((ROOT / "experiments" / "finetune" / f"{arch}_s{s}" / "test_ap.json").read_text())["mAP"]
                      for s in (0, 1, 2)])
        x = d[d.arch == arch]
        cells = [f"{100 * ap:.1f}"]
        for al in (0.05, 0.1, 0.2):
            cells.append(pc(x[(x.method == "Surgery-level (ours)") & (x.alpha == al)].sens.mean()))
        cells.append(pc(x[(x.method == "CRC (= frame-level)") & (x.alpha == 0.1)].ops_false_ge1.mean()))
        lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    write("cert_training", lines)


def cholec80_strict():
    c = pd.read_csv(ROOT / "experiments" / "surgery_level" / "cholec80_surgery_level.csv")
    lines = []
    for sc, slab in (("A: Endoscapes calibration", "as-is"), ("B: recalibrate on 20 Cholec80 videos", "recal.\\ 20")):
        for m, mlab in (("Surgery-level", "Surgery-level (ours)"), ("CRC-Mondrian (frames pooled)", "CRC (frame)"),
                        ("LTT (frames pooled)", "LTT (frame)")):
            cells = []
            for ref in ("cvs_authors", "cvs_strict"):
                for a in (0.05, 0.1, 0.2):
                    x = c[(c.scenario == sc) & (c.method == m) & (c.alpha == a) & (c.ref == ref)]
                    cells.append(pc(x.surgeries_with_false_confirm.mean()))
            lines.append(f"{slab} & {mlab} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    write("cholec80_strict", lines[:-1])


if __name__ == "__main__":
    retarget(); holdout(); utility(); timeout_rules(); cert_training(); cholec80_strict()
