"""Qualitative grid (full text width): six test operations (columns) x four rows
  Reference      the surgeons' boxes and CVS (majority of three)
  Point world    MAP labels; "CVS achieved" if all three criterion probabilities exceed 0.5
  Frame-level    CRC (= frame-level) at alpha = 0.10, frames pooled
  Surgery-level  ours at alpha = 0.10 (ours)
Official split: thresholds calibrated on the 31 val calibration videos, test videos shown;
detector and CVS model seed 0.  Each panel shows the duct / artery labels the system displays
and, when it confirms CVS, a CVS banner.  Columns are chosen automatically, one test
operation each, in this order of case types:
  a  point world and frame-level both show a wrong label; ours defers
  b  frame-level falsely confirms CVS; ours does not
  c  point world shows a wrong label; ours certifies the correct one
  d  ours certifies both structures correctly
  e  ours certifies CVS correctly
  f  any further test operation where ours certifies a correct label
  g  point world wrong, ours makes no wrong statement

    python scripts/figures/fig_qualitative.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from cdss import anatomy as A                       # noqa: E402
from cdss.data import keyframes                     # noqa: E402
from surgery_level import order_stat_threshold, prepare  # noqa: E402

OUT = ROOT / "paper" / "figures"
NAMES = {A.DUCT: "duct", A.ARTERY: "artery"}
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as ST  # noqa: E402

ST.setup()
plt.rcParams["axes.grid"] = False
GOOD, BAD, NEUTRAL, DEFER = "#138a60", "#d03b3b", "#52514e", "#6f6e69"
COL = {A.DUCT: "#ffd21f", A.ARTERY: "#ff4d4d"}
ALPHA = 0.10
CROP = (0.13, 0.87)                                  # horizontal crop of the black endoscope border


def put_label(ax, placed, b, x0, text, color, fs, img_w, img_h):
    """Label just above its box; if that collides with a placed label, try above that label,
    then below the box; always kept inside the image."""
    w = len(text) * fs * 0.62 * img_w / 80.0          # rough label size in pixels at this scale
    h = fs * 1.9 * img_w / 80.0
    x = min(b[0] - x0, img_w - w)

    def free(y):
        return all(not (abs(py - y) < h and x < px + w and px < x + w) for px, py in placed)
    cands = [b[1] - 2] + [py - h for _, py in placed] + [b[3] + h + 2]
    y = next((c for c in cands if h <= c <= img_h - 2 and free(c)), max(h, min(b[1] - 2, img_h - 2)))
    placed.append((x, y))
    ax.text(x, y, text, fontsize=fs, color="k", va="bottom", bbox=dict(fc=color, ec="none", pad=0.4))


def regions():
    det = ROOT / "experiments" / "detector" / "frcnn_s0"
    reg = json.loads((det / "regions_test.json").read_text())
    names = {str(i["id"]): i["file_name"] for i in
             json.loads((ROOT / "data" / "endoscapes" / "test" / "annotation_coco.json").read_text())["images"]}
    return {names[k]: v for k, v in reg.items()}


def main():
    sel_vids = np.random.default_rng(0).choice(np.unique(keyframes("val").vid), 10, replace=False)
    frames, kv, kp, ky = prepare(0, sel_vids)
    test_vids = set(keyframes("test").vid.tolist())
    cal = [f for f in frames if f["vid"] not in test_vids]
    tst = [f for f in frames if f["vid"] in test_vids]
    kc = ~np.isin(kv, list(test_vids))                # CVS-rated calibration keyframes

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
    print(f"thresholds id frame {lam_id_frame:.3f} surg {lam_id_surg:.3f} | "
          f"cvs frame {lam_cvs_frame:.3f} surg {lam_cvs_surg:.3f}")

    R = regions()

    def outputs(f):
        y = A.map_world(f["Pt"]) if len(f["Pt"]) else np.zeros(0, int)
        pw = {t: (int(np.flatnonzero(y == t)[0]) if (y == t).any() else None) for t in A.CLIP_TARGETS}
        thr = lambda lam: {t: (f["cand"][t] if f["cand"][t] is not None and f["prob"][t] > lam else None)
                           for t in A.CLIP_TARGETS}
        return {"pw": (pw, bool((f["pc3"] > 0.5).all())),
                "frame": (thr(lam_id_frame), f["pcvs"] > lam_cvs_frame),
                "surg": (thr(lam_id_surg), f["pcvs"] > lam_cvs_surg)}

    def wrong(f, lab):
        return [t for t, r in lab.items() if r is not None and f["y"][r] != t]

    def right(f, lab):
        return [t for t, r in lab.items() if r is not None and f["y"][r] == t]

    tests = [
        ("a", lambda f, o: wrong(f, o["pw"][0]) and wrong(f, o["frame"][0]) and not any(
            r is not None for r in o["surg"][0].values()) and not o["surg"][1]),
        ("b", lambda f, o: o["frame"][1] and not f["cvs_ok"] and not o["surg"][1] and not wrong(f, o["surg"][0])),
        ("c", lambda f, o: wrong(f, o["pw"][0]) and right(f, o["surg"][0]) and not wrong(f, o["surg"][0])),
        ("d", lambda f, o: len(right(f, o["surg"][0])) == 2 and not wrong(f, o["surg"][0])),
        ("e", lambda f, o: o["surg"][1] and f["cvs_ok"] and not wrong(f, o["surg"][0])),
        ("f", lambda f, o: right(f, o["surg"][0]) and not wrong(f, o["surg"][0])),
        ("g", lambda f, o: wrong(f, o["pw"][0]) and not wrong(f, o["surg"][0])),
    ]
    outs = [(f, outputs(f)) for f in tst]
    chosen, used = [], set()
    for tag, test in tests:
        for f, o in outs:
            if f["vid"] in used or not test(f, o):
                continue
            chosen.append((tag, f, o)); used.add(f["vid"]); break
        if len(chosen) == 6:
            break
    print("chosen:", [(t, f["vid"], f["frame"]) for t, f, _ in chosen])

    rows = [("Reference\n(surgeons)", None), ("Point world", "pw"), ("Frame-level\nCRC", "frame"), ("Ours\n(per operation)", "surg")]
    # two blocks of three cases: larger frames, the same crop in every row, outcome labels outside the image
    for blk, block in enumerate((chosen[:3], chosen[3:6])):
        n = len(block)
        cell_w = 6.875 / n; img_h = cell_w * 0.97 * (480 / (854 * (CROP[1] - CROP[0])))
        fig = plt.figure(figsize=(6.875, len(rows) * (img_h + 0.30) + 0.3))
        gs = fig.add_gridspec(2 * len(rows), n, height_ratios=[img_h, 0.30] * len(rows), wspace=0.03, hspace=0.05,
                              left=0.06, right=0.995, top=0.965, bottom=0.01)
        for j, (tag, f, o) in enumerate(block):
            fname = f"{f['vid']}_{f['frame']}.jpg"
            img = Image.open(ROOT / "data" / "endoscapes" / "test" / fname).convert("RGB")
            W, H = img.size; x0, x1 = int(CROP[0] * W), int(CROP[1] * W)
            img = img.crop((x0, 0, x1, H))
            d = R[fname]; B = np.array(d["boxes"]).reshape(-1, 4)
            G = np.array(d["gt_boxes"]).reshape(-1, 4); GL = np.array(d["gt_labels"], int)
            for i, (rname, key) in enumerate(rows):
                ax = fig.add_subplot(gs[2 * i, j]); ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
                lab_ax = fig.add_subplot(gs[2 * i + 1, j]); lab_ax.axis("off")
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if j == 0:
                    ax.set_ylabel(rname, fontsize=7, rotation=90, labelpad=3,
                                  color=ST.OURS if key == "surg" else ST.INK, weight="bold" if key == "surg" else "normal")
                if i == 0:
                    ax.set_title(f"{'abcdef'[3 * blk + j]}  operation {f['vid']}, frame {f['frame']}", fontsize=7.5, pad=3, loc="left")
                fs = 6.5; placed = []
                if key is None:
                    for bx, l in zip(G, GL):
                        if l in NAMES:
                            ax.add_patch(patches.Rectangle((bx[0] - x0, bx[1]), bx[2] - bx[0], bx[3] - bx[1], fill=False, lw=1.0, ec=COL[l]))
                            put_label(ax, placed, bx, x0, NAMES[l], COL[l], fs, img.size[0], img.size[1])
                    txt, fc = ("CVS achieved", GOOD) if f["cvs_ok"] else ("CVS not achieved", NEUTRAL)
                    lab_ax.text(0.0, 0.5, txt, fontsize=6.6, color="w", va="center", ha="left", transform=lab_ax.transAxes,
                                bbox=dict(fc=fc, ec="none", pad=1.2))
                    continue
                lab, cvs = o[key]
                shown = False
                for t, r in lab.items():
                    if r is None:
                        continue
                    ok = f["y"][r] == t; bx = B[r]
                    ax.add_patch(patches.Rectangle((bx[0] - x0, bx[1]), bx[2] - bx[0], bx[3] - bx[1], fill=False, lw=1.0,
                                                   ec=COL[t], ls="-" if ok else (0, (2, 1.2))))
                    put_label(ax, placed, bx, x0, f"{NAMES[t]} {f['Pt'][r, t]:.2f}" + ("" if ok else " ×"), COL[t], fs,
                              img.size[0], img.size[1])
                    shown = True
                ok = f["cvs_ok"]
                if cvs:
                    txt, fc = ("CVS confirmed", GOOD) if ok else ("CVS confirmed ×", BAD)
                elif key == "pw":
                    txt, fc = ("CVS not achieved", NEUTRAL) if not ok else ("CVS not achieved ×", BAD)
                else:
                    txt, fc = ("CVS: defer to surgeon", DEFER)
                lab_ax.text(0.0, 0.5, txt, fontsize=6.6, color="w", va="center", ha="left", transform=lab_ax.transAxes,
                            bbox=dict(fc=fc, ec="none", pad=1.2))
                if not shown:
                    lab_ax.text(1.0, 0.5, "no label certified" if key != "pw" else "no label", fontsize=6.6, color=ST.MUTED,
                                va="center", ha="right", transform=lab_ax.transAxes)
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"fig_qualitative_{'ab'[blk]}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    (OUT / "fig_qualitative_cases.json").write_text(json.dumps(
        {"thresholds": {"id_frame": float(lam_id_frame), "id_surg": float(lam_id_surg),
                        "cvs_frame": float(lam_cvs_frame), "cvs_surg": float(lam_cvs_surg)},
         "cases": [{"tag": t, "vid": int(f["vid"]), "frame": int(f["frame"])} for t, f, _ in chosen]}, indent=1))
    print("wrote", OUT / "fig_qualitative_a.pdf", OUT / "fig_qualitative_b.pdf")


if __name__ == "__main__":
    main()
