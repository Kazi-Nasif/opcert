"""Condensed main-paper tables (review round 6), built from the generated full tables so that every
number has one source:
  tables/compare_main.tex   streaming CVS comparison: frame FR, ops >= 1 false, sensitivity, FDR at
                            alpha = .05 / .10; main frame-level baselines and ours
  tables/timeout_main.tex   time-out decisions with separate confirm / stop targets
  tables/decisions.tex      other decisions at alpha = .10: duct/artery identification, clip/cut
                            target, surgical phase; frame-level comparator vs ours with the utility
                            measure of each task
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
        line = " & ".join(cells) + " \\\\"
        if r[0].startswith("Surgery-level (ours)") and out and not out[-1].startswith("\\midrule"):
            out.append("\\midrule")
        out.append(line)
    (T / "compare_main.tex").write_text("\n".join(out) + "\n")


def timeout_main():
    out = []
    for r in rows("timeout.tex"):
        if r[0].startswith("Conf."):
            continue
        ours = "Ours" in r[0] or (r[0] == "" and out and ("Ours" in out[-1] or "Ours" in out[-2] if len(out) > 1 else False))
        a_s = ".20" if ("Ours" in r[0] or ours) else "--"
        out.append(" & ".join([r[0], r[1], a_s] + r[2:]) + " \\\\")
    (T / "timeout_main.tex").write_text("\n".join(out) + "\n")


def decisions():
    pc = lambda x: f"{100 * x:.1f}"
    L = []
    # (a) duct / artery identification, Endoscapes (recall over all annotated ducts and arteries)
    a = pd.read_csv(ROOT / "experiments" / "surgery_level" / "anatomy_surgery_level.csv"); a = a[a.task == "identify"]
    g = a.groupby(["method", a.alpha.fillna(-1)])[["surgeries_with_error", "precision", "recall_e2e"]].mean()
    L.append("\\multicolumn{5}{@{}l}{\\emph{Duct / artery label, Endoscapes: (i) precision, (ii) recall}} \\\\")
    for m, lab in (("MAP (point world)", "Point world"), ("Frame-level ours (frames pooled)", "Frame-level CRC"), ("Surgery-level", "\\textbf{Ours}")):
        al = -1 if m.startswith("MAP") else 0.1
        r = g.loc[(m, al)]
        L.append(f"{lab} & {pc(r.surgeries_with_error)} & {pc(r.precision)} & {pc(r.recall_e2e)} & -- \\\\")
    L.append("\\midrule")
    # (b) clip / cut target, CholecT50: ops >= 1 wrong, coverage, precision
    L.append("\\multicolumn{5}{@{}l}{\\emph{Clip / cut target, CholecT50: (i) precision, (ii) frames certified}} \\\\")
    for r in rows("cholect50.tex"):
        lab = r[0].replace("\\textbf{", "").replace("}", "").replace(" (ours)", "")
        if lab not in ("Point world", "CRC", "Surg.-level", "+ smoothing"):
            continue
        lab = {"CRC": "Frame-level CRC", "Surg.-level": "\\textbf{Ours}", "+ smoothing": "\\textbf{Ours + smoothing}"}.get(lab, lab)
        L.append(f"{lab} & {r[1]} & {r[5]} & {r[4]} & -- \\\\")
    L.append("\\midrule")
    # (c) surgical phase, Cholec80: "clipping has begun" ops err / sens / delay; displayed label ops err / accuracy / coverage
    L.append("\\multicolumn{5}{@{}l}{\\emph{Phase ``clipping has begun'', Cholec80: (i) sensitivity, (ii) delay (s)}} \\\\")
    last = ""
    for r in rows("phase.tex"):
        lab = r[0].replace("\\textbf{", "").replace("}", "") or last
        last = lab
        if r[1] not in ("--", ".10") or lab == "LTT":
            continue
        lab = {"CRC": "Frame-level CRC", "Ours": "\\textbf{Ours}"}.get(lab, lab)
        L.append(f"{lab} & {r[2]} & {r[4]} & {r[5]} & {r[6]} / {r[8]} / {r[7]} \\\\")
    (T / "decisions.tex").write_text("\n".join(L) + "\n")


def main():
    compare_main(); timeout_main(); decisions()
    for f in ("compare_main", "timeout_main", "decisions"):
        print(f"--- {f}"); print((T / f"{f}.tex").read_text())


if __name__ == "__main__":
    main()
