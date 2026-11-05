"""Demo video: one held-out Endoscapes test operation played back with three decision systems
side by side, all running causally on the trained models (CVS model and anatomy detector,
seed 0), thresholds calibrated on the 31 official calibration videos at alpha = 0.10:

  Point world      most probable interpretation (no guarantee)
  Frame-level CRC  conformal risk control per frame (frames pooled)
  Surgery-level    our certificate: P(any false statement in the operation) <= alpha

Each panel shows the CVS decision (confirm / not achieved / defer) and the duct / artery labels
it would display.  Below: P(CVS) over time with both thresholds, the surgeons' reference, and
running counts of false CVS confirmations.  Title, explanation and summary cards bracket it.

    python scripts/demo/make_demo.py            -> experiments/demo/demo.mp4
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from cdss import anatomy as A                                     # noqa: E402
from cdss.data import all_frames, keyframes                       # noqa: E402
from evaluate import causal_logits, fit_temperature               # noqa: E402
from evaluate_anatomy import fit_T                                # noqa: E402
from surgery_level import order_stat_threshold, prepare           # noqa: E402
from train_detector import build, regions                         # noqa: E402

ALPHA, SMOOTH, FPS, W, H = 0.10, 10, 10, 1920, 1080
OUT = ROOT / "experiments" / "demo"; OUT.mkdir(parents=True, exist_ok=True)
NAMES = {A.DUCT: "cystic duct", A.ARTERY: "cystic artery"}
COL = {A.DUCT: "#ffd21f", A.ARTERY: "#ff4d4d"}
C_FRAME, C_OURS, C_PW = "#e07b1a", "#1f4e9c", "#6d6d6d"
plt.rcParams.update({"font.family": "DejaVu Sans"})


def sig(x):
    return 1 / (1 + np.exp(-x))


# ------------------------------------------------------------------ thresholds (official split)
def cvs_thresholds(run, T):
    va = keyframes("val"); sel = np.isin(va.vid, np.random.default_rng(0).choice(np.unique(va.vid), 10, replace=False))
    p = np.prod(sig(causal_logits(run / "logits_val.npz", va, SMOOTH)[~sel] / T), 1)
    y = va.cvs("majority")[~sel]; v = va.vid[~sel]
    lam_frame = order_stat_threshold(p[~y], ALPHA)
    lam_op = order_stat_threshold(np.array([p[(v == u) & ~y].max() for u in np.unique(v) if ((v == u) & ~y).any()]), ALPHA)
    return lam_frame, lam_op


def id_thresholds(frames, test_vids):
    cal = [f for f in frames if f["vid"] not in test_vids]
    def wc(f):
        w = [f["prob"][t] for t in A.CLIP_TARGETS if f["cand"][t] is not None and not f["correct"][t]]
        return max(w) if w else -np.inf
    lam_frame = order_stat_threshold(np.array([wc(f) for f in cal]), ALPHA)
    per = {}
    for f in cal:
        per[f["vid"]] = max(per.get(f["vid"], -np.inf), wc(f))
    return lam_frame, order_stat_threshold(np.array(list(per.values())), ALPHA)


def det_temperature(sel_vids):
    reg = json.loads((ROOT / "experiments" / "detector" / "frcnn_s0" / "regions_val.json").read_text())
    names = {str(i["id"]): i["file_name"] for i in
             json.loads((ROOT / "data" / "endoscapes" / "val" / "annotation_coco.json").read_text())["images"]}
    Ls, ys = [], []
    for iid, d in reg.items():
        if int(names[iid].split("_")[0]) in sel_vids and len(d["probs"]):
            P = np.array(d["probs"]).reshape(-1, 7)
            Ls.append(np.log(np.clip(P, A.EPS, 1)))
            ys.append(A.region_truth(np.array(d["boxes"]).reshape(-1, 4), np.array(d["gt_boxes"]).reshape(-1, 4),
                                     np.array(d["gt_labels"], int)))
    Ls, ys = torch.tensor(np.concatenate(Ls)), torch.tensor(np.concatenate(ys))
    return fit_T(lambda T: torch.nn.functional.cross_entropy(Ls / T, ys))


# ------------------------------------------------------------------ choose the operation
def pick_video(run, T, lam_frame, lam_op):
    te = keyframes("test"); z = np.load(run / "logits_test.npz"); yt = te.cvs("majority")
    best = None
    for v in np.unique(te.vid):
        rows = np.flatnonzero(z["vid"] == v); rows = rows[np.argsort(z["frame"][rows])]
        L = z["logits"][rows]
        Ls = np.stack([L[max(0, i - SMOOTH + 1):i + 1].mean(0) for i in range(len(L))])
        p = np.prod(sig(Ls / T), 1)
        km = te.vid == v; idx = np.searchsorted(z["frame"][rows], te.frame[km]); ky = yt[km]; kp = p[idx]
        ff, fo, to = ((kp > lam_frame) & ~ky).sum(), ((kp > lam_op) & ~ky).sum(), ((kp > lam_op) & ky).sum()
        if ff >= 2 and fo == 0 and to >= 1 and len(rows) <= 900:
            score = ff + to
            if best is None or score > best[0]:
                best = (score, int(v))
    return best[1]


# ------------------------------------------------------------------ drawing helpers
def card(lines, sub=None):
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100); fig.patch.set_facecolor("#0f1720")
    y = 0.72
    for txt, size, col, weight in lines:
        fig.text(0.5, y, txt, ha="center", va="center", fontsize=size, color=col, weight=weight, wrap=True)
        y -= 0.06 + size / 900
    if sub:
        fig.text(0.5, 0.06, sub, ha="center", fontsize=13, color="#9aa5b1")
    return fig


def fig_to_rgb(fig):
    fig.canvas.draw()
    a = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return a


def draw_panel(ax, img, title, title_col, cvs_state, labels, boxes, probs):
    ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(title_col); s.set_linewidth(3)
    ax.set_title(title, fontsize=17, color=title_col, weight="bold", pad=8)
    txt, fc = cvs_state
    ax.text(0.02, 0.96, txt, transform=ax.transAxes, va="top", fontsize=16, color="w", weight="bold",
            bbox=dict(fc=fc, ec="none", pad=4))
    shown = False
    for t, r in labels.items():
        if r is None:
            continue
        b = boxes[r]
        ax.add_patch(patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, lw=2.5, ec=COL[t]))
        ax.text(b[0], max(b[1] - 6, 18), f"{NAMES[t]} {probs[r, t]:.2f}", fontsize=13, color="k",
                bbox=dict(fc=COL[t], ec="none", pad=2))
        shown = True
    if not shown:
        ax.text(0.5, 0.05, "no anatomy label certified" if "Point" not in title else "no anatomy label",
                transform=ax.transAxes, ha="center", fontsize=13, color="w",
                bbox=dict(fc="#1f4e9c" if "Point" not in title else "#555555", ec="none", pad=3))


def op_series(run, T, vid):
    """1-fps P(CVS) with 10 s causal smoothing, criterion probabilities, surgeons' reference held
    between rated keyframes, and time in seconds, for one test operation."""
    z = np.load(run / "logits_test.npz")
    rows = np.flatnonzero(z["vid"] == vid); rows = rows[np.argsort(z["frame"][rows])]
    L = z["logits"][rows]; fr = z["frame"][rows]
    Ls = np.stack([L[max(0, i - SMOOTH + 1):i + 1].mean(0) for i in range(len(L))])
    pc = sig(Ls / T); p = pc.prod(1)
    te = keyframes("test"); km = te.vid == vid; kf, ky = te.frame[km], te.cvs("majority")[km]
    ref = np.array([bool(ky[np.searchsorted(kf, f, side="right") - 1]) if f >= kf[0] else False for f in fr])
    return fr, pc, p, ref, (fr - fr[0]) / 25.0


def test_set_summary(run, T, lam_cf, lam_co):
    out = {"ops": 0, "ops_cvs": 0, "ours_conf_ops": 0, "no_cvs_ops": 0, "crc_false_ops_no_cvs": 0,
           "ours_false_ops_no_cvs": 0, "crc_false_ops": 0, "ours_false_ops": 0}
    for v in np.unique(keyframes("test").vid):
        fr, pc, p, ref, t = op_series(run, T, int(v))
        out["ops"] += 1
        has = ref.any(); out["ops_cvs"] += has; out["no_cvs_ops"] += not has
        out["ours_conf_ops"] += bool(((p > lam_co) & ref).any())
        out["crc_false_ops"] += bool(((p > lam_cf) & ~ref).any()); out["ours_false_ops"] += bool(((p > lam_co) & ~ref).any())
        if not has:
            out["crc_false_ops_no_cvs"] += bool((p > lam_cf).any()); out["ours_false_ops_no_cvs"] += bool((p > lam_co).any())
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--videos", type=int, nargs="+", default=[196, 193])
    a = ap.parse_args()
    run = ROOT / "experiments" / "finetune" / "dinov2l_b8_s0"
    va = keyframes("val")
    sel_vids = set(np.random.default_rng(0).choice(np.unique(va.vid), 10, replace=False).tolist())
    sel = np.isin(va.vid, list(sel_vids))
    T = fit_temperature(causal_logits(run / "logits_val.npz", va, 1)[sel], va.soft[sel])
    lam_cf, lam_co = cvs_thresholds(run, T)
    frames, *_ = prepare(0, np.array(sorted(sel_vids)))
    lam_if, lam_io = id_thresholds(frames, set(keyframes("test").vid.tolist()))
    T_det = det_temperature(sel_vids)
    summ = test_set_summary(run, T, lam_cf, lam_co)
    print(f"CVS thresholds frame {lam_cf:.3f} ours {lam_co:.3f} | id thresholds frame {lam_if:.3f} ours {lam_io:.3f}")
    print("test-set summary", summ)
    det = build().cuda(); det.load_state_dict(torch.load(ROOT / "experiments" / "detector" / "frcnn_s0" / "model.pt",
                                                         map_location="cuda")); det.eval()

    writer = imageio_ffmpeg.write_frames(str(OUT / "demo.mp4"), (W, H), fps=FPS, codec="libx264",
                                         pix_fmt_out="yuv420p", quality=8, macro_block_size=8)
    writer.send(None)
    def hold(fig, seconds):
        arr = fig_to_rgb(fig)
        for _ in range(int(seconds * FPS)):
            writer.send(arr)

    hold(card([("ours: certified safety checks for surgical video", 34, "white", "bold"),
               ("Guarantees must hold per operation", 24, "#7fb3ff", "normal"),
               ("Laparoscopic cholecystectomy · Critical View of Safety (CVS) · anatomy identification", 16, "#c9d1d9", "normal")],
              "Research prototype on retrospective, public data (Endoscapes2023). Not for clinical use."), 4)
    hold(card([("Before clipping the cystic duct and artery, surgeons must achieve the Critical View of Safety.", 18, "white", "normal"),
               ("An AI check can confirm CVS and label the duct and artery - but a false 'all clear' is dangerous.", 18, "white", "normal"),
               ("Frame-level guarantees bound errors per image: over hundreds of frames, most operations still get a false 'CVS confirmed'.", 18, "#ffb366", "normal"),
               ("Our surgery-level certificate bounds the chance that an operation receives ANY false statement by α = 10%,", 18, "#7fb3ff", "normal"),
               ("confirms CVS when the evidence is strong enough, and otherwise defers to the surgeon.", 18, "#7fb3ff", "normal"),
               (f"Next: {len(a.videos)} held-out test operations, 1 frame per second played at {FPS}x speed; all three systems use the same model outputs.", 16, "#c9d1d9", "normal")]), 9)

    intro = {True: "CVS is achieved during this operation: does each system confirm it, and only then?",
             False: "CVS is (almost) never achieved in this operation: does any system give a false 'all clear'?"}
    for vid in a.videos:
        fr, pc, p, ref, t = op_series(run, T, vid)
        df = all_frames("test"); df = df[df.vid == vid].sort_values("frame")
        files = dict(zip(df.frame.to_numpy(), df.file))
        dets = []
        with torch.no_grad():
            for f in fr:
                x = torchvision.transforms.functional.to_tensor(Image.open(files[f]).convert("RGB")).cuda()
                B, P = regions(det, x)
                dets.append((B, A.temper(P, T_det) if len(P) else P))
        has_cvs = ref.mean() > 0.1
        hold(card([(f"Test operation {vid}", 30, "white", "bold"), (intro[has_cvs], 20, "#c9d1d9", "normal"),
                   (f"{len(fr) // 60} min of dissection, surgeons rate CVS achieved for {int(ref.sum())} s", 16, "#9aa5b1", "normal")]), 3.5)
        cnt = {k: [0, 0] for k in ("pw", "frame", "ours")}            # [correct confirmations, false confirmations]
        false_pts = []
        for i, f in enumerate(fr):
            img = Image.open(files[f]).convert("RGB")
            B, Pt = dets[i]
            cand = {tt: (int(np.argmax(Pt[:, tt])) if len(Pt) else None) for tt in A.CLIP_TARGETS}
            def lab(lam):
                return {tt: (r if r is not None and Pt[r, tt] > lam else None) for tt, r in cand.items()}
            y = A.map_world(Pt) if len(Pt) else np.zeros(0, int)
            pw_lab = {tt: (int(np.flatnonzero(y == tt)[0]) if (y == tt).any() else None) for tt in A.CLIP_TARGETS}
            dec = {"pw": bool((pc[i] > 0.5).all()), "frame": p[i] > lam_cf, "ours": p[i] > lam_co}
            for k, c in dec.items():
                if c:
                    cnt[k][0 if ref[i] else 1] += 1
            if dec["frame"] and not ref[i]:
                false_pts.append((t[i], p[i]))
            def state(conf, pw=False):
                if conf:
                    return ("CVS CONFIRMED", "#2e7d32") if ref[i] else ("CVS CONFIRMED  ✗ false", "#c62828")
                return ("CVS not achieved", "#555555") if pw else ("CVS: defer to surgeon", "#1f4e9c")

            fig = plt.figure(figsize=(W / 100, H / 100), dpi=100); fig.patch.set_facecolor("white")
            mm, ss = divmod(int(t[i]), 60)
            fig.text(0.03, 0.955, f"Test operation {vid}  ·  {mm:02d}:{ss:02d} into dissection  ·  α = {ALPHA:.2f}",
                     fontsize=18, weight="bold")
            fig.text(0.985, 0.955, "Surgeons' reference: " + ("CVS ACHIEVED" if ref[i] else "CVS not achieved"),
                     ha="right", fontsize=18, weight="bold", color="#2e7d32" if ref[i] else "#555555")
            pan = [fig.add_axes([0.015 + k * 0.33, 0.50, 0.32, 0.39]) for k in range(3)]
            draw_panel(pan[0], img, "Point world (no guarantee)", C_PW, state(dec["pw"], True), pw_lab, B, Pt)
            draw_panel(pan[1], img, "Frame-level guarantee (CRC)", C_FRAME, state(dec["frame"]), lab(lam_if), B, Pt)
            draw_panel(pan[2], img, "Surgery-level certificate (ours)", C_OURS, state(dec["ours"]), lab(lam_io), B, Pt)

            ax = fig.add_axes([0.05, 0.075, 0.66, 0.36])
            ax.fill_between(t / 60, 0, 1.05, where=ref, color="#2e8b57", alpha=0.15, step="mid", lw=0,
                            label="surgeons: CVS achieved")
            ax.plot(t[:i + 1] / 60, p[:i + 1], c="k", lw=2, label="model: P(CVS)")
            ax.axhline(lam_cf, c=C_FRAME, ls="--", lw=2, label=f"frame-level threshold {lam_cf:.2f}")
            ax.axhline(min(lam_co, 1.0), c=C_OURS, lw=2.5, label=f"surgery-level threshold (ours) {lam_co:.2f}")
            if false_pts:
                a_ = np.array(false_pts); ax.scatter(a_[:, 0] / 60, a_[:, 1], marker="x", c="#c62828", s=60, lw=2,
                                                     zorder=5, label="false CVS confirmation (frame-level)")
            ax.axvline(t[i] / 60, c="#999", lw=1)
            ax.set_xlim(0, t[-1] / 60); ax.set_ylim(0, 1.05)
            ax.set_xlabel("time in dissection phase (min)", fontsize=14); ax.set_ylabel("P(CVS achieved)", fontsize=14)
            ax.tick_params(labelsize=12); ax.spines[["top", "right"]].set_visible(False)
            ax.legend(loc="upper left", fontsize=11.5, frameon=False, ncol=2)
            fig.text(0.735, 0.43, "Seconds of 'CVS confirmed' so far", fontsize=16, weight="bold", va="top")
            fig.text(0.905, 0.375, "✓ correct", fontsize=14, color="#2e7d32", weight="bold", ha="right")
            fig.text(0.985, 0.375, "✗ false", fontsize=14, color="#c62828", weight="bold", ha="right")
            for k, (nm, c, key) in enumerate((("Point world", C_PW, "pw"), ("Frame-level (CRC)", C_FRAME, "frame"),
                                               ("Surgery-level (ours)", C_OURS, "ours"))):
                yk = 0.30 - k * 0.075
                fig.text(0.735, yk, nm, fontsize=16, color=c, weight="bold", va="center")
                fig.text(0.905, yk, f"{cnt[key][0]}", fontsize=22, color="#2e7d32", weight="bold", va="center", ha="right")
                fig.text(0.985, yk, f"{cnt[key][1]}", fontsize=22, color="#c62828", weight="bold", va="center", ha="right")
            writer.send(fig_to_rgb(fig))
            if i % 100 == 0:
                print(vid, "frame", i, "/", len(fr), flush=True)
        print(vid, "counts [correct, false]", cnt, flush=True)
        hold(card([(f"Operation {vid}: seconds of 'CVS confirmed' (correct / false)", 24, "white", "bold"),
                   (f"Point world {cnt['pw'][0]} / {cnt['pw'][1]}     ·     Frame-level CRC {cnt['frame'][0]} / {cnt['frame'][1]}", 20, "#ffb366", "normal"),
                   (f"Surgery-level (ours) {cnt['ours'][0]} / {cnt['ours'][1]}", 22, "#7fb3ff", "bold")]), 3.5)

    n = json.loads((ROOT / "paper" / "numbers.json").read_text())["compare_all"]
    g = lambda m, a_: 100 * n[f"dinov2l_b8|{m}|{a_}"]["ops_false_ge1"]
    s_ = lambda m, a_: 100 * n[f"dinov2l_b8|{m}|{a_}"]["sens"]
    hold(card([(f"All {summ['ops']} test operations, same thresholds (α = 10%):", 22, "white", "bold"),
               (f"{summ['no_cvs_ops']} operations never reach CVS: frame-level CRC falsely confirms in {summ['crc_false_ops_no_cvs']}, ours in {summ['ours_false_ops_no_cvs']}", 19, "#c9d1d9", "normal"),
               (f"{summ['ops_cvs']} operations reach CVS: ours confirms it in {summ['ours_conf_ops']} of them, and defers in the rest", 19, "#c9d1d9", "normal"),
               ("Across 71 held-out operations (200 random calibration/evaluation splits x 3 seeds):", 18, "#9aa5b1", "normal"),
               (f"operations with at least one false CVS confirmation - frame-level CRC {g('CRC + smoothing (10 s)', 0.1):.0f}%,  Learn-then-Test {g('LTT', 0.1):.0f}%,  ours {g('Surgery-level + smoothing (ours)', 0.1):.1f}% (guarantee: 10%)", 18, "#7fb3ff", "bold"),
               (f"The price: ours confirms {s_('Surgery-level + smoothing (ours)', 0.1):.0f}% of CVS frames (frame-level: {s_('CRC + smoothing (10 s)', 0.1):.0f}%) and defers otherwise.", 17, "#c9d1d9", "normal")],
              "Endoscapes2023 · Cholec80-CVS · CholecT50 · four backbones · nine competing methods (see paper)"), 10)
    writer.close()
    print("wrote", OUT / "demo.mp4")


if __name__ == "__main__":
    main()
