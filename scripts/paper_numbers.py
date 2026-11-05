"""Every number quoted in the paper, computed from the result files (single source of truth).

    python scripts/paper_numbers.py   ->  paper/numbers.json (+ printed)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "experiments"
N = {}

# perception
for arch, tag in (("dinov2l_b8", "dinov2"), ("resnet50", "resnet50")):
    aps = np.array([json.loads((E / "finetune" / f"{arch}_s{s}" / "test_ap.json").read_text())["AP"] for s in (0, 1, 2)])
    N[f"cvs_ap_{tag}"] = {"C1": aps[:, 0].mean(), "C2": aps[:, 1].mean(), "C3": aps[:, 2].mean(),
                          "mAP": aps.mean(1).mean(), "mAP_std": aps.mean(1).std(ddof=1),
                          "per_seed_mAP": aps.mean(1).tolist()}
det = [json.loads((E / "detector" / f"frcnn_s{s}" / "det_metrics_test.json").read_text()) for s in (0, 1, 2)]
N["det_map"] = {"mean": float(np.mean([d["mAP"] for d in det])), "per_seed": [d["mAP"] for d in det],
                "per_class": {k: float(np.mean([d["per_class_AP"][k] for d in det])) for k in det[0]["per_class_AP"]}}

# reduction: frame-level CRC-Mondrian vs CRC (official split, DINOv2, 3 seeds)
rows = pd.concat([pd.read_csv(E / "finetune" / f"dinov2l_b8_s{s}" / "decisions.csv") for s in (0, 1, 2)])
rows = rows[rows.window_s == 1]
red = {}
for alpha in (0.01, 0.02, 0.05, 0.10):
    r = rows[rows.alpha == alpha]
    red[str(alpha)] = {m: float(r[r.method == m].n_confirm.mean())
                       for m in ("CRC-Mondrian", "CRC (given not CVS)", "CRC (joint)", "LTT (joint)")}
N["reduction_n_confirm"] = red
m = rows[rows.method == "MAP (point world)"]
N["official_map"] = {"n_confirm": float(m.n_confirm.mean()), "fdr": float(m.fdr_majority.mean())}

# operation-level, Endoscapes CVS (DINOv2, 200 splits x 3 seeds)
r = pd.read_csv(E / "resplit" / "resplit.csv"); r = r[r.run.str.startswith("dinov2l")]
def agg(d, col, extra=()):
    g = d.groupby(["method", "alpha"]).agg(mean=(col, "mean"), lo=(col, lambda x: x.quantile(.025)),
                                            hi=(col, lambda x: x.quantile(.975)),
                                            **{e: (e, "mean") for e in extra})
    return {f"{k[0]}|{k[1]}": {kk: float(vv) for kk, vv in v.items()} for k, v in g.iterrows()}
N["ops_cvs_endoscapes"] = agg(r, "video_false_confirm", ("sens", "fr_given_notCVS", "fdr"))
a = pd.read_csv(E / "surgery_level" / "anatomy_surgery_level.csv")
a["alpha"] = a.alpha.fillna(-1)
for task in ("identify", "clip"):
    N[f"ops_{task}"] = agg(a[a.task == task], "surgeries_with_error", ("precision", "recall", "frame_error"))
c = pd.read_csv(E / "surgery_level" / "cholec80_surgery_level.csv"); c["alpha"] = c.alpha.fillna(-1)
for sc in c.scenario.unique():
    d = c[(c.ref == "cvs_authors") & (c.scenario == sc)]
    N[f"ops_cholec80|{sc}"] = agg(d, "surgeries_with_false_confirm", ("sens", "frame_fr_given_notCVS"))
info = json.loads((E / "cholec80" / "info.json").read_text())
N["cholec80"] = {"frames": info["frames"], "videos": info["videos"],
                 "AP_cvs_authors": float(np.mean([info[f"seed{s}_AP_cvs_cvs_authors"] for s in (0, 1, 2)]))}
c80 = pd.read_csv(ROOT / "data" / "cholec80" / "cholec80_cvs_frames.csv")
N["cholec80"].update({"cvs_authors_rate": float(c80.cvs_authors.mean()), "cvs_strict_rate": float(c80.cvs_strict.mean()),
                      "videos_with_cvs": int(c80.groupby("video").cvs_authors.any().sum())})

# all backbones (perception)
for arch, tag in (("surgenetxl", "surgenetxl"), ("endovit", "endovit")):
    aps = np.array([json.loads((E / "finetune" / f"{arch}_s{s}" / "test_ap.json").read_text())["AP"] for s in (0, 1, 2)])
    N[f"cvs_ap_{tag}"] = {"C1": aps[:, 0].mean(), "C2": aps[:, 1].mean(), "C3": aps[:, 2].mean(),
                          "mAP": aps.mean(1).mean(), "mAP_std": aps.mean(1).std(ddof=1)}

# full comparison: every method x backbone (Endoscapes CVS, 200 splits x 3 seeds)
ca = pd.read_csv(E / "compare_all" / "compare_all.csv")
ca["arch"] = ca.run.str.replace(r"_s\d+$", "", regex=True); ca["alpha"] = ca.alpha.fillna(-1)
g = ca.groupby(["arch", "method", "alpha"])[["ops_false_ge1", "ops_false_gt1", "frame_fr", "sens", "fdr"]].mean()
N["compare_all"] = {f"{a}|{m}|{al}": {k: float(v) for k, v in row.items()} for (a, m, al), row in g.iterrows()}

# causal temporal models on frame embeddings (frozen / fine-tuned DINOv2-L), same comparison
ct_ = pd.read_csv(E / "compare_all" / "compare_temporal.csv")
ct_["arch"] = ct_.run.str.replace(r"_s\d+$", "", regex=True); ct_["alpha"] = ct_.alpha.fillna(-1)
g = ct_.groupby(["arch", "method", "alpha"])[["ops_false_ge1", "ops_false_gt1", "frame_fr", "sens", "fdr"]].mean()
N["compare_temporal"] = {f"{a}|{m}|{al}": {k: float(v) for k, v in row.items()} for (a, m, al), row in g.iterrows()}
for arch in sorted(ct_.arch.unique()):
    aps = np.array([json.loads((E / "finetune" / f"{arch}_s{s}" / "test_ap.json").read_text())["AP"] for s in (0, 1, 2)])
    N[f"cvs_ap_{arch}"] = {"C1": aps[:, 0].mean(), "C2": aps[:, 1].mean(), "C3": aps[:, 2].mean(),
                           "mAP": aps.mean(1).mean(), "mAP_std": aps.mean(1).std(ddof=1)}

# CholecT50 clip / divide target certification
ct = pd.read_csv(E / "cholect50" / "target_certification.csv"); ct["alpha"] = ct.alpha.fillna(-1)
g = ct.groupby(["method", "alpha"])[["ops_wrong_ge1", "ops_wrong_gt1", "frame_error", "coverage", "precision"]].mean()
N["cholect50"] = {f"{m}|{a}": {k: float(v) for k, v in row.items()} for (m, a), row in g.iterrows()}
N["cholect50_acc"] = [json.loads((E / "cholect50" / f"target_dinov2l_s{s}" / "train_log.json").read_text())["eval_acc"]
                      for s in (0, 1, 2)]

# time-out (clip-time) certificates
pc = pd.read_csv(E / "preclip" / "preclip.csv"); pc["alpha"] = pc.alpha.fillna(-1)
g = pc.groupby(["dataset", "smooth_s", "method", "alpha"])[
    ["cvs_at_clip_rate", "false_confirm_given_noCVS", "confirm_given_CVS", "alarm_given_noCVS",
     "false_alarm_joint", "false_alarm_given_CVS", "defer", "window_ops_false_confirm", "window_sensitivity"]].mean()
N["timeout"] = {f"{d}|{s}|{m}|{a}": {k: float(v) for k, v in row.items()} for (d, s, m, a), row in g.iterrows()}


def rnd(x):
    if isinstance(x, dict):
        return {k: rnd(v) for k, v in x.items()}
    if isinstance(x, list):
        return [rnd(v) for v in x]
    return round(x, 6) if isinstance(x, float) else x
out = ROOT / "paper" / "numbers.json"; out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(rnd(N), indent=1))
print(json.dumps(rnd(N), indent=1))
