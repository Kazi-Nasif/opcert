"""Main-paper qualitative figure (full text width, three panels) from the stored evaluation artifacts.

Same configuration as supplementary Figs. S2/S3 (scripts/figures/fig_qualitative.py): official Endoscapes
split, thresholds calibrated at alpha = 0.10 on the 31 validation operations outside the 10 selection
videos, detector and CVS model seed 0, raw (unsmoothed) keyframe scores, anatomy candidates and
projected statements exactly as evaluated.  Nothing is re-trained or re-inferred: detector regions come
from experiments/detector/frcnn_s0/regions_test.json, CVS logits from experiments/finetune/dinov2l_b8_s0.

Cases (each verified against the recomputed decisions; a deterministic search over the test keyframes
replaces a candidate that fails its condition):
  a  reference CVS absent, frame-level CRC confirms CVS (false), ours defers and issues no wrong label
  b  frame-level CRC certifies a wrong duct/artery label, ours certifies a correct one and no wrong one
  c  reference CVS achieved, ours confirms CVS and issues no wrong label

Outputs: paper/figures/qualitative_main.{pdf,png} and paper/figures/qualitative_main_manifest.csv.

    python scripts/generate_qualitative_main.py
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "scripts" / "figures"))
from cdss import anatomy as A                            # noqa: E402
from cdss.data import keyframes                          # noqa: E402
from surgery_level import order_stat_threshold, prepare  # noqa: E402
import style as ST                                       # noqa: E402

ST.setup()
plt.rcParams.update({"axes.grid": False, "font.size": 7.5})
OUT = ROOT / "paper" / "figures"
ALPHA, SEED = 0.10, 0
CROP = (0.13, 0.87)                       # horizontal crop of the black endoscope border (as in Figs. S2/S3)
NAMES = {A.DUCT: "duct", A.ARTERY: "artery"}
GOOD, BAD, DEFER, INK, MUTED = "#1B7F79", "#C0392B", "#6B6B6B", "#1F2329", "#5c6068"
OK, NO = r"$\checkmark$", "×"
BOX_GT, BOX_WRONG, BOX_OURS = "#F2C230", "#E2401C", "#1B7F79"
CANDIDATES = {"a": (166, 13700), "b": (190, 45675), "c": (192, 24575)}   # supplementary Fig. S2(b), S2(c), S3(d)
TITLES = {"a": "Premature CVS confirmation avoided", "b": "Wrong anatomy label avoided",
          "c": "Correct CVS confirmation kept"}


def regions():
    det = ROOT / "experiments" / "detector" / f"frcnn_s{SEED}"
    reg = json.loads((det / "regions_test.json").read_text())
    names = {str(i["id"]): i["file_name"] for i in
             json.loads((ROOT / "data" / "endoscapes" / "test" / "annotation_coco.json").read_text())["images"]}
    return {names[k]: v for k, v in reg.items()}


def thresholds(frames, kv, kp, ky, test_vids):
    cal = [f for f in frames if f["vid"] not in test_vids]
    kc = ~np.isin(kv, list(test_vids))

    def wrong_conf(f):
        w = [f["prob"][t] for t in A.CLIP_TARGETS if f["cand"][t] is not None and not f["correct"][t]]
        return max(w) if w else -np.inf
    lam_id_frame = order_stat_threshold(np.array([wrong_conf(f) for f in cal]), ALPHA)
    per_vid = {}
    for f in cal:
        per_vid[f["vid"]] = max(per_vid.get(f["vid"], -np.inf), wrong_conf(f))
    lam_id_surg = order_stat_threshold(np.array(list(per_vid.values())), ALPHA)
    lam_cvs_frame = order_stat_threshold(kp[kc & ~ky], ALPHA)
    m = [kp[kc & (kv == v) & ~ky].max() if (kc & (kv == v) & ~ky).any() else -np.inf for v in np.unique(kv[kc])]
    lam_cvs_surg = order_stat_threshold(np.array(m), ALPHA)
    return dict(id_frame=lam_id_frame, id_surg=lam_id_surg, cvs_frame=lam_cvs_frame, cvs_surg=lam_cvs_surg)


def decisions(f, lam):
    """Certified labels {target: region or None} and CVS decision for frame-level CRC and ours."""
    def labels(l):
        return {t: (f["cand"][t] if f["cand"][t] is not None and f["prob"][t] > l else None) for t in A.CLIP_TARGETS}
    return {"frame": (labels(lam["id_frame"]), bool(f["pcvs"] > lam["cvs_frame"])),
            "ours": (labels(lam["id_surg"]), bool(f["pcvs"] > lam["cvs_surg"]))}


def wrong(f, lab):
    return [t for t, r in lab.items() if r is not None and f["y"][r] != t]


def right(f, lab):
    return [t for t, r in lab.items() if r is not None and f["y"][r] == t]


CONDITIONS = {
    "a": lambda f, o: o["frame"][1] and not f["cvs_ok"] and not o["ours"][1] and not wrong(f, o["ours"][0]),
    "b": lambda f, o: bool(wrong(f, o["frame"][0])) and bool(right(f, o["ours"][0])) and not wrong(f, o["ours"][0]),
    "c": lambda f, o: o["ours"][1] and f["cvs_ok"] and not wrong(f, o["ours"][0]),
}


def describe(f, lab, cvs, who):
    """Row text and colour for one method on one frame."""
    parts = []
    if cvs:
        parts.append((f"CVS confirmed {OK}", GOOD) if f["cvs_ok"] else (f"CVS confirmed {NO}", BAD))
    else:
        parts.append(("CVS deferred", DEFER))
    lab_txt = []
    for t, r in lab.items():
        if r is None:
            continue
        ok = f["y"][r] == t
        lab_txt.append((f"{NAMES[t]} {OK}" if ok else f"{NAMES[t]} {NO}", GOOD if ok else BAD))
    if not lab_txt:
        lab_txt = [("no label", DEFER)]
    return parts + lab_txt


def main():
    sel_vids = np.random.default_rng(0).choice(np.unique(keyframes("val").vid), 10, replace=False)
    frames, kv, kp, ky = prepare(SEED, sel_vids)
    test_vids = set(keyframes("test").vid.tolist())
    lam = thresholds(frames, kv, kp, ky, test_vids)
    print("thresholds", {k: round(v, 4) for k, v in lam.items()})
    tst = sorted([f for f in frames if f["vid"] in test_vids], key=lambda f: (f["vid"], f["frame"]))
    by_key = {(f["vid"], f["frame"]): f for f in tst}
    R = regions()

    chosen, used = {}, set()
    for tag in "abc":
        cand = by_key.get(CANDIDATES[tag])
        if cand is not None and CONDITIONS[tag](cand, decisions(cand, lam)) and cand["vid"] not in used:
            chosen[tag] = (cand, "documented candidate (supplementary Figs. S2/S3), condition verified")
        else:
            rep = next((f for f in tst if f["vid"] not in used and CONDITIONS[tag](f, decisions(f, lam))), None)
            if rep is None:
                raise SystemExit(f"no test keyframe satisfies condition {tag}")
            chosen[tag] = (rep, f"candidate {CANDIDATES[tag]} failed its condition; first test keyframe (by operation, frame) satisfying it")
        used.add(chosen[tag][0]["vid"])
        print(tag, chosen[tag][0]["vid"], chosen[tag][0]["frame"], "|", chosen[tag][1])

    # ---- figure: three panels, image on top, three aligned decision rows underneath
    W = 6.875; n = 3; gap = 0.10; pw = (W - (n - 1) * gap) / n
    img_asp = 480 / (854 * (CROP[1] - CROP[0]))
    ih = pw * img_asp                                        # image height in inches
    title_h, row_h, rows = 0.24, 0.135, 3
    H = title_h + ih + rows * row_h + 0.04
    fig = plt.figure(figsize=(W, H))
    rows_out = []
    for j, tag in enumerate("abc"):
        f, reason = chosen[tag]
        o = decisions(f, lam)
        fname = f"{f['vid']}_{f['frame']}.jpg"
        img = Image.open(ROOT / "data" / "endoscapes" / "test" / fname).convert("RGB")
        Wi, Hi = img.size; x0, x1 = int(CROP[0] * Wi), int(CROP[1] * Wi)
        img = img.crop((x0, 0, x1, Hi))
        d = R[fname]; B = np.array(d["boxes"]).reshape(-1, 4)
        G = np.array(d["gt_boxes"]).reshape(-1, 4); GL = np.array(d["gt_labels"], int)
        xl = j * (pw + gap)
        # title
        fig.text(xl / W, 1 - 0.07 / H, f"{tag}  {TITLES[tag]}", fontsize=8, fontweight="bold", color=INK, va="center")
        fig.text(xl / W, 1 - 0.18 / H, f"operation {f['vid']} · frame {f['frame']} · α = {ALPHA:.2f}", fontsize=6.3,
                 color=MUTED, va="center")
        ax = fig.add_axes([xl / W, 1 - (title_h + ih) / H, pw / W, ih / H])
        ax.imshow(np.asarray(img)); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        # reference boxes (surgeons), thin, class colour; label at the top-left corner of the box
        for bx, l in zip(G, GL):
            if l in NAMES:
                ax.add_patch(patches.Rectangle((bx[0] - x0, bx[1]), bx[2] - bx[0], bx[3] - bx[1], fill=False, lw=1.0, ec=BOX_GT))
                ax.text(bx[0] - x0 + 3, bx[1] + 3, NAMES[l], fontsize=6, color="k", va="top", ha="left",
                        bbox=dict(fc=BOX_GT, ec="none", pad=0.3))
        # wrong frame-CRC labels: dashed red box with the label CRC put on it
        for t in wrong(f, o["frame"][0]):
            bx = B[o["frame"][0][t]]
            ax.add_patch(patches.Rectangle((bx[0] - x0, bx[1]), bx[2] - bx[0], bx[3] - bx[1], fill=False, lw=1.1, ec=BOX_WRONG, ls=(0, (3, 1.5))))
            ax.text(bx[2] - x0, bx[1] - 4, f"CRC: {NAMES[t]} {NO}", fontsize=6, color="w", va="bottom", ha="right",
                    bbox=dict(fc=BOX_WRONG, ec="none", pad=0.3))      # just above the box, clear of the reference tags
        # labels ours certifies (all correct by construction of the conditions): solid teal box
        for t in right(f, o["ours"][0]):
            bx = B[o["ours"][0][t]]
            ax.add_patch(patches.Rectangle((bx[0] - x0, bx[1]), bx[2] - bx[0], bx[3] - bx[1], fill=False, lw=1.3, ec=BOX_OURS))
            ax.text(bx[0] - x0 + 3, bx[3] - 3, f"ours: {NAMES[t]} {OK}", fontsize=6, color="w", va="bottom", ha="left",
                    bbox=dict(fc=BOX_OURS, ec="none", pad=0.3))
        # decision rows
        ref_cvs = ("CVS achieved", GOOD) if f["cvs_ok"] else ("CVS absent", DEFER)
        ann = [NAMES[l] for l in GL if l in NAMES]
        ref_row = [ref_cvs] + ([("boxes: " + ", ".join(ann), INK)] if ann else [])
        table = [("Reference", ref_row), ("Frame CRC", describe(f, *o["frame"], "frame")), ("Ours", describe(f, *o["ours"], "ours"))]
        y_top = 1 - (title_h + ih) / H
        for i, (name, cells) in enumerate(table):
            yc = y_top - (i + 0.5) * row_h / H
            fig.text(xl / W, yc, name, fontsize=6.5, color=ST.OURS if name == "Ours" else INK,
                     fontweight="bold" if name == "Ours" else "normal", va="center")
            x = xl + 0.56
            for txt, col in cells:
                t = fig.text(x / W, yc, txt, fontsize=6.5, color=col, va="center", fontweight="bold" if col in (GOOD, BAD) else "normal")
                fig.canvas.draw()
                x += t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi + 0.07
            assert x - 0.07 <= xl + pw + 0.02, f"row '{name}' of panel {tag} overflows its panel by {x - 0.07 - xl - pw:.2f} in"
        fr_lab, fr_cvs = o["frame"]; ou_lab, ou_cvs = o["ours"]
        rows_out.append(dict(
            panel=tag, title=TITLES[tag], operation=f["vid"], frame=f["frame"], image=f"data/endoscapes/test/{fname}",
            split="official Endoscapes test (31 val calibration operations)", seed=SEED, alpha=ALPHA, smoothing="none (raw keyframe scores)",
            reference_cvs=f["cvs_ok"], annotated_structures=";".join(ann), p_cvs=round(float(f["pcvs"]), 4),
            thr_cvs_frame=round(lam["cvs_frame"], 4), thr_cvs_ours=round(lam["cvs_surg"], 4),
            thr_label_frame=round(lam["id_frame"], 4), thr_label_ours=round(lam["id_surg"], 4),
            frame_crc_cvs="confirm" if fr_cvs else "defer",
            frame_crc_labels=";".join(f"{NAMES[t]}:region{r}:p={f['prob'][t]:.3f}:{'correct' if f['y'][r] == t else 'wrong'}" for t, r in fr_lab.items() if r is not None) or "none",
            ours_cvs="confirm" if ou_cvs else "defer",
            ours_labels=";".join(f"{NAMES[t]}:region{r}:p={f['prob'][t]:.3f}:{'correct' if f['y'][r] == t else 'wrong'}" for t, r in ou_lab.items() if r is not None) or "none",
            selection=reason))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"qualitative_main.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.01)
    with open(OUT / "qualitative_main_manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys())); w.writeheader(); w.writerows(rows_out)
    print("figure", W, "x", round(H, 2), "in; wrote", OUT / "qualitative_main.pdf")


if __name__ == "__main__":
    main()
