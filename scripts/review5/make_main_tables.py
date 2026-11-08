"""Condensed main-paper tables (review round 6), built from the generated full tables so that every
number has one source:
  tables/compare_main.tex   streaming CVS comparison: frame FR, ops >= 1 false, sensitivity, FDR at
                            alpha = .05 / .10; main frame-level baselines and ours
  tables/timeout_main.tex   time-out decisions with separate confirm / stop targets and the defer rate
  tables/decisions.tex      other decisions at alpha = .10, full width: one block of three explicit
                            columns per decision (identification, clip/cut target, clipping onset);
                            point world, frame-level CRC, ours, ours + smoothing
Run after make_tables.py, make_tables_r3.py and surgery_level.py; then colorize_tables.py.

    python scripts/review5/make_main_tables.py
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "paper" / "tables"


def clean(line):
    line = re.sub(r"\\rowcolor\{[^}]*\}|\\cellcolor\{[^}]*\}|\$\^\\dagger\$", "", line)
    return [c.strip() for c in line.rstrip().rstrip("\\").split("&")]


def rows(name):
    return [clean(l) for l in (T / name).read_text().splitlines() if "&" in l]


def expand(cells):
    """Logical columns: a \\multicolumn{n} cell occupies n slots (the cell, then n-1 None)."""
    out = []
    for c in cells:
        m = re.search(r"\\multicolumn\{(\d+)\}", c)
        out.append(c); out += [None] * (int(m.group(1)) - 1 if m else 0)
    return out


def compare_main():
    keep = ["Point world", "CRC~", "CRC + smoothing", "Learn-then-Test", "ACI", "One frame/video", "CRC at $\\alpha/N_{\\max}$",
            "Surgery-level (ours)", "+ smoothing (ours)", "window selected on half", "PAC"]
    out = []
    for r in rows("compare.tex"):
        if not any(r[0].startswith(k) or k in r[0] for k in keep):
            continue
        e = expand(r); cells = [e[0], e[1]]
        for block in (e[2:7], e[7:12]):
            if block[0] and "multicolumn" in block[0]:
                cells.append(block[0].replace("multicolumn{5}", "multicolumn{4}")); continue
            ops1, ops2, fr, sens, fdr = block
            cells += [fr, ops1, sens, fdr]
        if r[0].startswith("Surgery-level (ours)") and out and not out[-1].startswith("\\midrule"):
            out.append("\\midrule")
        if "Surgery-level (ours)" in cells[0]:   # same procedure as CRC with operations as data points
            cells[0] = cells[0].replace("Surgery-level (ours)", "Per operation (ours)") + " $\\equiv$ CRC over operations"
        line = " & ".join(cells) + " \\\\"
        out.append(line)
    (T / "compare_main.tex").write_text("\n".join(out) + "\n")


def timeout_main():
    # defer rate per (dataset, policy, alpha_c) from the full time-out table (last column)
    defer = {}
    for r in rows("supp/timeout_full.tex"):
        pol = {"point world": "Point world", "frame CRC": "Frame CRC", "frame LTT": "Frame LTT", "ours": "Ours"}.get(r[1])
        if pol:
            defer[(r[0], pol, "--" if r[2] == "--" else f"{float(r[2]):.2f}".lstrip("0"))] = r[-1]
    out, last = [], ""
    for r in rows("timeout.tex"):
        if r[0].startswith("Conf."):
            continue
        lab = r[0].replace("\\textbf{", "").replace("}", "") or last
        last = lab
        a_s = ".20" if "Ours" in lab else "--"
        ak = "--" if r[1] == "--" else f"{float(r[1]):.2f}".lstrip("0")
        de, dc = defer[("Endoscapes", lab, ak)], defer[("Cholec80", lab, ak)]
        out.append(" & ".join([r[0], r[1], a_s] + r[2:6] + [de] + r[6:10] + [dc]) + " \\\\")
    (T / "timeout_main.tex").write_text("\n".join(out) + "\n")


def decisions():
    """Full-width table: rows = methods, three blocks of three explicit columns.
    identification (Endoscapes): ops with >= 1 wrong duct/artery label, precision, recall;
    clip/cut target (CholecT50): ops with >= 1 wrong statement, precision, share of clipping frames certified;
    clipping onset (Cholec80, 10 s smoothing): ops announced before the true onset, sensitivity, median delay (s).
    The displayed-phase-label statement is reported in the supplementary only."""
    pc = lambda x: f"{100 * x:.1f}"
    blk = {"Point world": [], "Frame-level CRC": [], "Ours": [], "Ours + smoothing": []}
    # (a) duct / artery identification, Endoscapes (recall over all annotated ducts and arteries); no smoothing variant
    a = pd.read_csv(ROOT / "experiments" / "surgery_level" / "anatomy_surgery_level.csv"); a = a[a.task == "identify"]
    g = a.groupby(["method", a.alpha.fillna(-1)])[["surgeries_with_error", "precision", "recall_e2e"]].mean()
    for m, lab in (("MAP (point world)", "Point world"), ("Frame-level ours (frames pooled)", "Frame-level CRC"), ("Surgery-level", "Ours")):
        r = g.loc[(m, -1 if m.startswith("MAP") else 0.1)]
        blk[lab] += [pc(r.surgeries_with_error), pc(r.precision), pc(r.recall_e2e)]
    blk["Ours + smoothing"] += ["", "", ""]
    # (b) clip / cut target, CholecT50: ops >= 1 wrong, precision, coverage
    for r in rows("cholect50.tex"):
        lab = r[0].replace("\\textbf{", "").replace("}", "").replace(" (ours)", "")
        lab = {"CRC": "Frame-level CRC", "Surg.-level": "Ours", "+ smoothing": "Ours + smoothing"}.get(lab, lab)
        if lab in blk:
            blk[lab] += [r[1], r[5], r[4]]
    # (c) clipping onset, Cholec80 (10 s smoothing block): ops premature, sensitivity, delay
    last = ""
    for r in rows("phase.tex"):
        lab = r[0].replace("\\textbf{", "").replace("}", "") or last
        last = lab
        if r[1] not in ("--", ".10") or lab == "LTT":
            continue
        lab = {"CRC": "Frame-level CRC"}.get(lab, lab)
        blk[lab] += [r[2], r[4], r[5]]
    blk["Ours + smoothing"] += ["", "", ""]
    L = []
    for lab, cells in blk.items():
        assert len(cells) == 9, (lab, cells)
        name = f"\\textbf{{{lab}}}" if lab.startswith("Ours") else lab
        L.append(" & ".join([name] + cells) + " \\\\")
    (T / "decisions.tex").write_text("\n".join(L) + "\n")


def main():
    compare_main(); timeout_main(); decisions()
    for f in ("compare_main", "timeout_main", "decisions"):
        print(f"--- {f}"); print((T / f"{f}.tex").read_text())


if __name__ == "__main__":
    main()
