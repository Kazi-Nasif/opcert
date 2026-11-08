"""Colour the generated LaTeX tables with one rule set (run after every table generator):
  * rows of our method get a light-blue background (\\rowcolor{oursbg});
  * every per-operation error that exceeds its nominal level gets a light-red cell
    (\\cellcolor{violbg}), whichever method it belongs to.
Colours are defined in paper/main.tex and paper/supp.tex (colortbl). Idempotent.

    python scripts/review3/colorize_tables.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "paper" / "tables"
# file -> (indices of per-operation error cells, alpha per cell: float or "row:<col>")
SPEC = {   # file -> (error cells, their levels, column holding the method label)
    "compare.tex": ([2, 7], [0.05, 0.10], 0),
    "timeout.tex": ([2, 6], ["row:1", "row:1"], 0),
    "phase.tex": ([2, 6], ["row:1", "row:1"], 0),
    "sages.tex": ([2], ["row:1"], 0),
    "supp/sages.tex": ([4], ["row:1"], 2),
    "supp/sages_full.tex": ([2], ["row:0"], 1),
    "supp/sages_country.tex": ([3, 5], [0.10, 0.10], 0),
    "compare_main.tex": ([3, 7], [0.05, 0.10], 0),
    "timeout_main.tex": ([3, 8], ["row:1", "row:1"], 0),
    "decisions.tex": ([1, 4, 7], [0.10, 0.10, 0.10], 0),
}
OURS = re.compile(r"\(ours\)|\bOurs\b|\bours\b|C: ours|^C$|^C &")


def num(cell):
    m = re.search(r"(\d+(?:\.\d+)?)", re.sub(r"\\[a-zA-Z]+\{|\}", "", cell.split("[")[0].split("/")[0]))
    return float(m.group(1)) if m else None


def frac(cell):
    m = re.match(r"\s*(\d+)/(\d+)", cell)
    return 100 * int(m.group(1)) / int(m.group(2)) if m else None


def alpha(cell):
    m = re.search(r"(\d*\.\d+)", cell)
    return float(m.group(1)) if m else None


def colorize(path, cols, alphas, lab_col):
    lines, out, in_ours = path.read_text().splitlines(), [], False
    for ln in lines:
        ln = ln.replace("\\rowcolor{oursbg}", "").replace("\\cellcolor{violbg}", "").replace("$^\\dagger$", "")
        if not ln.strip() or ln.lstrip().startswith("\\midrule") or "&" not in ln:
            in_ours = False if ln.lstrip().startswith("\\midrule") else in_ours
            out.append(ln); continue
        cells = ln.split("&")
        label = cells[lab_col].strip()
        if label:
            in_ours = bool(OURS.search(label)) or label in ("C", "C: ours")
        # logical column -> physical cell (\multicolumn{n} spans n logical columns)
        phys, pos = {}, 0
        for k, c in enumerate(cells):
            m = re.search(r"\\multicolumn\{(\d+)\}", c); n = int(m.group(1)) if m else 1
            if n == 1:
                phys[pos] = k
            pos += n
        skip_row = "$k$=1" in label                      # bounds ">1 error", not ">=1"
        for j, a in zip(cols, alphas):
            if j not in phys or skip_row:
                continue
            k = phys[j]
            lev = alpha(cells[phys.get(int(a.split(":")[1]), 0)]) if isinstance(a, str) else a
            if lev is None:
                continue
            v = frac(cells[k]) if "/" in cells[k] and "[" not in cells[k].split("/")[0] else num(cells[k])
            if v is not None and v > 100 * lev + 0.05:
                m2 = re.match(r"^(.*?)(\s*\\\\\s*)?$", cells[k], re.S)
                body, tail = m2.group(1).rstrip(), cells[k][len(m2.group(1).rstrip()):]
                cells[k] = " \\cellcolor{violbg}" + body.lstrip() + "$^\\dagger$" + tail
        ln = "&".join(cells)
        if in_ours:
            ln = "\\rowcolor{oursbg}" + ln
        out.append(ln)
    path.write_text("\n".join(out) + "\n")
    print("coloured", path.relative_to(ROOT))


def main():
    for f, (cols, alphas, lab) in SPEC.items():
        p = T / f
        if p.exists():
            colorize(p, cols, alphas, lab)


if __name__ == "__main__":
    main()
