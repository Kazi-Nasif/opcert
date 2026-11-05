"""LaTeX bodies of the generated paper tables, from paper/numbers.json (no hand-typed numbers).

    python scripts/make_tables.py   ->  paper/tables/{compare,backbones,cholect50}.tex
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
N = json.loads((ROOT / "paper" / "numbers.json").read_text())
OUT = ROOT / "paper" / "tables"; OUT.mkdir(exist_ok=True)
pc = lambda x: f"{100 * x:.1f}"

# method key in compare_all, row label, controlled quantity
COMPARE = [
    ("Point world (MAP)", "Point world", "none"),
    ("CRC (= frame-level)", "CRC~\\cite{angelopoulos2024conformal} ($\\equiv$ frame-lvl.)", "frame"),
    ("CRC + smoothing (10 s)", "CRC + smoothing", "frame"),
    ("LTT", "Learn-then-Test~\\cite{angelopoulos2021learn}", "frame, w.h.p."),
    ("ACI (label feedback)", "ACI~\\cite{gibbs2021adaptive}", "frame, long run"),
    ("SGR (selective risk)", "SGR~\\cite{geifman2017selective}", "sel.\\ risk, w.h.p."),
    ("Conformal selection (BH)", "Conf.\\ selection~\\cite{jin2023selection}", "FDR"),
    ("One frame / video (Dunn)", "One frame/video~\\cite{dunn2023distribution}", "frame"),
    ("Repeated subsampling (Dunn, a/2)", "Rep.\\ subsampling~\\cite{dunn2023distribution}", "frame"),
    ("SCRC", "SCRC-I~\\cite{xu2025selective} ($\\xi$=0.9)", "sel.\\ frame, w.h.p."),
    ("R|CRC Bonferroni (N max)", "CRC at $\\alpha/N_{\\max}$", "op.\\ (Bonferroni)"),
    ("R|CRC Bonferroni (N mean)", "CRC at $\\alpha/\\bar N$", "none"),
    (None, None, None),
    ("Surgery-level (ours)", "\\textbf{Surgery-level (ours)}", "op.\\ ($\\ge$1)"),
    ("Surgery-level, k = 1 (ours)", "\\textbf{$k$=1 (ours)}", "op.\\ ($>$1)"),
    ("Surgery-level + smoothing (ours)", "\\textbf{+ smoothing (ours)}", "op.\\ ($\\ge$1)"),
    ("R|op-LTT", "\\textbf{PAC, $\\delta$=0.1 (ours)}", "op., w.h.p."),
]
COLS = ("ops_false_ge1", "ops_false_gt1", "frame_fr", "sens", "fdr")


def compare():
    lines = []
    for key, lab, ctrl in COMPARE:
        if key is None:
            lines.append("\\midrule"); continue
        cells = []
        if key.startswith("R|"):                       # experiments/review/rebuttal_stream.csv (same splits)
            import pandas as pd
            d = pd.read_csv(ROOT / "experiments" / "review" / "rebuttal_stream.csv")
            d = d[(d.arch == "dinov2l_b8") & (d.w == 1) & (d.method == key[2:])]
            for a in (0.05, 0.1):
                x = d[d.alpha == a][list(COLS)].mean()
                if x["sens"] == 0:
                    cells += ["\\multicolumn{5}{c}{certifies nothing}"]; continue
                v = [pc(x[c]) for c in COLS]
                if "ours" in lab:
                    v[0] = f"\\textbf{{{v[0]}}}"
                cells += v
            lines.append(f"{lab} & {ctrl} & " + " & ".join(cells) + " \\\\"); continue
        if key == "SCRC":                              # experiments/review/selective.csv (same protocol)
            import pandas as pd
            d = pd.read_csv(ROOT / "experiments" / "review" / "selective.csv")
            d = d[(d.arch == "dinov2l_b8") & (d.method == "SCRC-I (frames)") & (d.xi == 0.9)]
            for a in (0.05, 0.1):
                x = d[d.alpha == a][list(COLS)].mean()
                cells += [pc(x[c]) for c in COLS]
            lines.append(f"{lab} & {ctrl} & " + " & ".join(cells) + " \\\\"); continue
        for a in ("0.05", "0.1"):
            r = N["compare_all"].get(f"dinov2l_b8|{key}|{'-1.0' if key.startswith('Point') else a}")
            if key.startswith("Point") and a == "0.1":
                cells += ["\\multicolumn{5}{c}{(no level)}"]; break
            vals = [pc(r[c]) for c in COLS]
            # bold the quantity a method controls when it is met
            if "ours" in key:
                j = 1 if "k = 1" in key else 0
                vals[j] = f"\\textbf{{{vals[j]}}}"
            if r["sens"] == 0:
                vals = ["\\multicolumn{5}{c}{certifies nothing}"]
            cells += vals
        lines.append(f"{lab} & {ctrl} & " + " & ".join(cells) + " \\\\")
    (OUT / "compare.tex").write_text("\n".join(lines) + "\n")


def backbones():
    archs = [("dinov2l_b8", "dinov2"), ("resnet50", "resnet50"), ("surgenetxl", "surgenetxl"), ("endovit", "endovit")]
    lines = ["CVS mAP & " + " & ".join(f"{100 * N['cvs_ap_' + t]['mAP']:.1f}" for _, t in archs) + " \\\\", "\\midrule"]
    rows = [("Point world (MAP)", "-1.0", "Point world"), ("CRC (= frame-level)", "0.1", "CRC"),
            ("LTT", "0.1", "LTT"), ("ACI (label feedback)", "0.1", "ACI"),
            ("One frame / video (Dunn)", "0.1", "1 frame/video"), ("Conformal selection (BH)", "0.1", "Conf.\\ sel."),
            ("Surgery-level (ours)", "0.1", "\\textbf{Surg.-level (ours)}"),
            ("Surgery-level + smoothing (ours)", "0.1", "\\textbf{+ smooth. (ours)}")]
    for key, a, lab in rows:
        r = [N["compare_all"][f"{arch}|{key}|{a}"] for arch, _ in archs]
        cell = lambda x: (f"\\textbf{{{pc(x['ops_false_ge1'])}}}" if "ours" in key else pc(x["ops_false_ge1"])) \
            + f"/{pc(x['sens'])}"
        lines.append(f"{lab} & " + " & ".join(cell(x) for x in r) + " \\\\")
    (OUT / "backbones.tex").write_text("\n".join(lines) + "\n")


def cholect50():
    rows = [("Point world|-1.0", "Point world"), ("CRC (frames pooled)|0.1", "CRC"),
            ("LTT (frames pooled)|0.1", "Learn-then-Test"), (None, None),
            ("Surgery-level|0.1", "\\textbf{Surg.-level (ours)}"),
            ("Surgery-level + smoothing|0.1", "\\textbf{+ smoothing (ours)}"),
            ("Surgery-level, k = 1|0.1", "\\textbf{$k$=1 (ours)}"),
            ("Surgery-level, k = 1 + smoothing|0.1", "\\textbf{$k$=1 + sm. (ours)}")]
    lines = []
    for key, lab in rows:
        if key is None:
            lines.append("\\midrule"); continue
        r = N["cholect50"][key]
        v = [pc(r[c]) for c in ("ops_wrong_ge1", "ops_wrong_gt1", "frame_error", "coverage", "precision")]
        if "ours" in lab:
            j = 1 if "k = 1" in key else 0
            v[j] = f"\\textbf{{{v[j]}}}"
        lines.append(f"{lab} & " + " & ".join(v) + " \\\\")
    (OUT / "cholect50.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    compare(); backbones(); cholect50()
    for f in sorted(OUT.glob("*.tex")):
        print("==", f.name); print(f.read_text())
