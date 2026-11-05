"""Real Endoscapes2023 frames for the architecture diagram (Fig. 1).

  past_op_{1,2,3}_{raw,labelled}.png  three CALIBRATION operations (official val split, not the
                                      10 selection videos); 'labelled' adds the surgeons' duct /
                                      artery boxes and their CVS rating (majority of three)
  new_op_raw.png                      a held-out TEST operation, no labels (labels unavailable)
  new_op_strip.png                    5 frames of that operation, 20 s apart (frame stream)
  past_op_strip.png                   5 frames of a calibration operation (one score per operation)
All images are cropped to the circular endoscope view (black borders removed) and saved at
full resolution.  Source: Endoscapes2023 (Murali et al., 2023), CC BY-NC-SA 4.0 -- cite it in
the caption.

    python scripts/figures/export_arch_assets.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cdss.data import all_frames, keyframes      # noqa: E402

OUT = ROOT / "paper" / "figures" / "arch_assets"; OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data" / "endoscapes"
COL = {4: (255, 210, 31), 3: (255, 77, 77)}       # duct, artery
NAME = {4: "cystic duct", 3: "cystic artery"}
CROP = (0.12, 0.88)
try:
    FONT = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
except OSError:
    FONT = ImageFont.load_default()


def crop(im):
    w, h = im.size
    return im.crop((int(CROP[0] * w), 0, int(CROP[1] * w), h)), int(CROP[0] * w)


def strip(paths):
    tiles = []
    for f in paths:
        im = crop(Image.open(f).convert("RGB"))[0]
        if np.asarray(im).mean() > 40:                   # skip frames with the camera outside the body
            tiles.append(im.resize((320, 270)))
        if len(tiles) == 5:
            break
    out = Image.new("RGB", (5 * 320 + 4 * 8, 270), "white")
    for j, t in enumerate(tiles):
        out.paste(t, (j * 328, 0))
    return out


def boxed_frames(split):
    coco = json.loads((DATA / split / "annotation_coco.json").read_text())
    anns = {}
    for a in coco["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    out = {}
    for im in coco["images"]:
        labs = {a["category_id"] for a in anns.get(im["id"], [])}
        if {3, 4} <= labs:                                      # both duct and artery annotated
            out[im["file_name"]] = anns[im["id"]]
    return out


def main():
    va, te = keyframes("val"), keyframes("test")
    sel = set(np.random.default_rng(0).choice(np.unique(va.vid), 10, replace=False).tolist())
    cvs = {(int(v), int(f)): bool(c) for v, f, c in zip(va.vid, va.frame, va.cvs("majority"))}
    bf = boxed_frames("val")
    picks, used = [], set()
    for want in (True, False, True):                            # mix of CVS achieved / not achieved
        for fn, anns in sorted(bf.items()):
            v, f = (int(x) for x in Path(fn).stem.split("_"))
            if v in sel or v in used or cvs.get((v, f)) is not want:
                continue
            picks.append((fn, anns, v, f)); used.add(v); break
    for k, (fn, anns, v, f) in enumerate(picks, 1):
        im = Image.open(DATA / "val" / fn).convert("RGB"); c, x0 = crop(im)
        c.save(OUT / f"past_op_{k}_raw.png")
        d = ImageDraw.Draw(c)
        for a in anns:
            if a["category_id"] in COL:
                bx, by, bw, bh = a["bbox"]
                d.rectangle([bx - x0, by, bx - x0 + bw, by + bh], outline=COL[a["category_id"]], width=4)
                d.text((bx - x0 + 2, max(by - 26, 2)), NAME[a["category_id"]], fill=COL[a["category_id"]], font=FONT)
        txt = "CVS achieved" if cvs[(v, f)] else "CVS not achieved"
        d.rectangle([8, 8, 8 + 12 * len(txt) + 16, 42], fill=(46, 125, 50) if cvs[(v, f)] else (85, 85, 85))
        d.text((16, 12), txt, fill="white", font=FONT)
        c.save(OUT / f"past_op_{k}_labelled.png")
        print("past op", k, "video", v, "frame", f, txt)
    # new (test) operation: raw frames, no labels
    vid = 196
    df = all_frames("test"); df = df[df.vid == vid].sort_values("frame").reset_index(drop=True)
    mid = df.iloc[int(0.93 * len(df))]
    crop(Image.open(mid.file).convert("RGB"))[0].save(OUT / "new_op_raw.png")
    strip([df.iloc[int(i)].file for i in np.linspace(0.15, 0.95, 9) * (len(df) - 1)]).save(OUT / "new_op_strip.png")
    pv = picks[0][2]
    dp = all_frames("val"); dp = dp[dp.vid == pv].sort_values("frame").reset_index(drop=True)
    strip([dp.iloc[int(i)].file for i in np.linspace(0.1, 0.9, 9) * (len(dp) - 1)]).save(OUT / "past_op_strip.png")
    print("new op video", vid, "frame", int(mid.frame))
    (OUT / "SOURCE.txt").write_text(
        "Frames from Endoscapes2023 (Murali et al., arXiv:2312.12429, 2023; CAMMA / IHU Strasbourg),\n"
        "licence CC BY-NC-SA 4.0. Past operations: official validation (calibration) videos "
        + ", ".join(str(p[2]) for p in picks) + f". New operation: official test video {vid}.\n")


def detector_asset():
    """anatomy_detector_output.png: real output of our Faster R-CNN (seed 0) on a held-out test frame:
    each region drawn with its most likely class and probability (all seven classes); and
    anatomy_detector_duct_artery.png: only the cystic duct / artery candidates."""
    sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "scripts" / "demo"))
    from cdss import anatomy as A
    from make_demo import det_temperature
    va = keyframes("val")
    T_det = det_temperature(set(np.random.default_rng(0).choice(np.unique(va.vid), 10, replace=False).tolist()))
    det = ROOT / "experiments" / "detector" / "frcnn_s0"
    reg = json.loads((det / "regions_test.json").read_text())
    names = {str(i["id"]): i["file_name"] for i in json.loads((DATA / "test" / "annotation_coco.json").read_text())["images"]}
    cls_col = {1: (120, 200, 255), 2: (180, 120, 255), 3: (255, 77, 77), 4: (255, 210, 31), 5: (90, 200, 120), 6: (200, 200, 200)}
    best = None
    for iid, d in reg.items():
        B = np.array(d["boxes"]).reshape(-1, 4); P = np.array(d["probs"]).reshape(-1, 7)
        if not len(P):
            continue
        y = A.region_truth(B, np.array(d["gt_boxes"]).reshape(-1, 4), np.array(d["gt_labels"], int))
        ok = [P[:, t].max() for t in (3, 4) if y[int(np.argmax(P[:, t]))] == t]
        if len(ok) == 2 and (P.argmax(1) > 0).sum() >= 4:
            sc = min(ok)
            if best is None or sc > best[0]:
                best = (sc, iid, B, P)
    _, iid, B, P = best
    P = A.temper(P, T_det)                                  # calibrated probabilities (what the certifier uses)
    im = Image.open(DATA / "test" / names[iid]).convert("RGB"); c, x0 = crop(im); c2 = c.copy()
    for img, only in ((c, False), (c2, True)):
        dr = ImageDraw.Draw(img)
        for b, p in zip(B, P):
            k = int(p.argmax())
            if k == 0 or (only and k not in (3, 4)):
                continue
            dr.rectangle([b[0] - x0, b[1], b[2] - x0, b[3]], outline=cls_col[k], width=4)
            lab = f"{A.CLASSES[k].replace('_', ' ')} {p[k]:.2f}"
            dr.rectangle([b[0] - x0, max(b[1] - 28, 0), b[0] - x0 + 13 * len(lab) + 8, max(b[1] - 28, 0) + 26], fill=cls_col[k])
            dr.text((b[0] - x0 + 4, max(b[1] - 26, 2)), lab, fill="black", font=FONT)
    c.save(OUT / "anatomy_detector_output.png"); c2.save(OUT / "anatomy_detector_duct_artery.png")
    crop(Image.open(DATA / "test" / names[iid]).convert("RGB"))[0].save(OUT / "new_op_raw_same_frame.png")
    v = int(names[iid].split("_")[0])
    df = all_frames("test"); df = df[df.vid == v].sort_values("frame").reset_index(drop=True)
    strip([df.iloc[int(i)].file for i in np.linspace(0.15, 0.95, 9) * (len(df) - 1)]).save(OUT / "new_op_strip_same_video.png")
    print("detector asset:", names[iid])
    with open(OUT / "SOURCE.txt", "a") as fh:
        fh.write(f"Anatomy detector output: our Faster R-CNN (seed 0) on held-out test frame {names[iid]}.\n")


if __name__ == "__main__":
    main()
    detector_asset()
