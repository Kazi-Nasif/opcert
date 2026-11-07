"""POST-HOC diagnostics for the SAGES external evaluation (not part of the locked protocol; they use
test outcomes and therefore select nothing). They ask why the operation-unit certificate recalibrated
on official-train cases exceeds its level on official-test cases:
  1. per-split distribution of the case statistic m_v (10 s), Kolmogorov-Smirnov test
  2. per-split annotation characteristics: rater agreement on all criteria, CVS-positive frames,
     video-level annotator confidence, country identifiers (anonymised)
  3. exchangeability check: 500 random 100/100 re-halvings of the pooled 200 locked cases (official
     split ignored); if the pooled cases were exchangeable the mean error would be <= alpha

    python scripts/external/sages_diagnostics.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "external"))
import sages_external as S  # noqa: E402


def main():
    prot = S.protocol(); T = prot["model"]["temperature"]
    out = {}
    for w in S.WINDOWS:
        D = S.load_cases(T, w)
        cal = [D[("train", c)] for c in S.cases("train")]; tst = [D[("test", c)] for c in S.cases("test")]
        mc = np.array([o["p"][~o["cvs"]].max() for o in cal]); mt = np.array([o["p"][~o["cvs"]].max() for o in tst])
        ks = ks_2samp(mc, mt); mw = mannwhitneyu(mc, mt)
        pooled = cal + tst; rng = np.random.default_rng(0); res = {a: [] for a in S.ALPHAS}
        for _ in range(500):
            ci = rng.choice(200, 100, replace=False); ti = np.setdiff1d(np.arange(200), ci)
            m = [pooled[i]["p"][~pooled[i]["cvs"]].max() for i in ci]
            for a in S.ALPHAS:
                lam = S.order_stat(m, a)
                res[a].append(np.mean([((pooled[i]["p"] > lam) & ~pooled[i]["cvs"]).any() for i in ti]))
        out[f"w{w}"] = {"m_cal_quantiles": np.quantile(mc, [.5, .75, .9, .95]).round(4).tolist(),
                        "m_test_quantiles": np.quantile(mt, [.5, .75, .9, .95]).round(4).tolist(),
                        "ks_stat": ks.statistic, "ks_p": ks.pvalue, "mannwhitney_p": mw.pvalue,
                        "pooled_rehalving_mean_error": {str(a): float(np.mean(v)) for a, v in res.items()},
                        "pooled_rehalving_share_above_alpha": {str(a): float(np.mean(np.array(v) > a)) for a, v in res.items()},
                        "pooled_rehalving_q95": {str(a): float(np.quantile(v, 0.95)) for a, v in res.items()},
                        "locked_error": {str(a): float(np.mean([((o["p"] > S.order_stat(mc, a)) & ~o["cvs"]).any() for o in tst])) for a in S.ALPHAS},
                        "pooled_rehalving_share_at_least_locked": {str(a): float(np.mean(np.array(v) >= np.mean([((o["p"] > S.order_stat(mc, a)) & ~o["cvs"]).any() for o in tst]) - 1e-9)) for a, v in res.items()}}
    ann = {}
    for split, role in (("train", "calibration"), ("test", "evaluation")):
        agree, conf, cvs_cases = [], [], 0
        for c in S.cases(split):
            f = pd.read_csv(S.DATA / split / "labels" / c / "frame.csv")
            R = np.stack([np.stack([f[f"c{x}_rater{y}"] for x in (1, 2, 3)], 1) for y in (1, 2, 3)], 1)
            agree.append(((R.sum(1) % 3) == 0).all(1).mean())
            v = pd.read_csv(S.DATA / split / "labels" / c / "video.csv")
            conf.append(float(v[[f"confidence_rater{y}" for y in (1, 2, 3)]].astype(float).mean(axis=1).iloc[0]))
            cvs_cases += int(((R.sum(1) >= 2).all(1)).any())
        meta = pd.read_csv(S.DATA / split / "labels" / "metadata.csv") if (S.DATA / split / "labels" / "metadata.csv").exists() else None
        if meta is None:
            from huggingface_hub import hf_hub_download
            meta = pd.read_csv(hf_hub_download(S.REPO, f"{split}/labels/metadata.csv", repo_type="dataset"))
        mm = meta[meta.video_name.isin(S.cases(split))]
        ann[role] = {"frames_all_raters_agree": float(np.mean(agree)), "video_confidence_mean": float(np.mean(conf)),
                     "cases_with_cvs_frame": cvs_cases, "countries": int(mm.country.nunique()),
                     "top_country_share": float(mm.country.value_counts(normalize=True).iloc[0]),
                     "robotic": int(mm.robotic.sum()), "icg": int(mm.icg.sum()), "ioc": int(mm.ioc.sum())}
    from huggingface_hub import hf_hub_download
    mtr = pd.read_csv(hf_hub_download(S.REPO, "train/labels/metadata.csv", repo_type="dataset"))
    mte = pd.read_csv(hf_hub_download(S.REPO, "test/labels/metadata.csv", repo_type="dataset"))
    cc = set(mtr[mtr.video_name.isin(S.cases("train"))].country); tc = mte[mte.video_name.isin(S.cases("test"))].country
    ann["evaluation"]["cases_from_countries_absent_in_calibration"] = int((~tc.isin(cc)).sum())
    out["annotation"] = ann
    (S.OUT / "diagnostics_posthoc.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
