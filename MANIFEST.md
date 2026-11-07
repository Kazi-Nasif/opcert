# Experiment manifest: every table and figure, its script and its result file

All Endoscapes stream results use the same 200 video-level halvings (`scripts/review/common.halves`,
`np.random.default_rng(rep).choice(71, 35)`), the 10 selection videos `default_rng(0).choice(val, 10)`,
DINOv2-L CVS seeds 0/1/2, and the majority reference unless stated.

| Paper item | Script | Result file |
|---|---|---|
| Main Tab. 1 (perception) | `scripts/evaluate.py`, `scripts/temporal/train_temporal.py` | `experiments/finetune/*/metrics.json`, `experiments/temporal/*` |
| Main Fig. 3, Tab. 3 | `scripts/compare_all.py` (+ `scripts/review2/rebuttal.py` for Bonferroni/PAC rows) | `experiments/compare_all/compare_all.csv`, `experiments/review/rebuttal_stream.csv` |
| Main Tab. 2 (anatomy, Cholec80) | `scripts/surgery_level.py`, `scripts/cholec80_surgery_level.py` | `experiments/surgery_level/*.csv` |
| Main Tab. 4 (time-out) | `scripts/review3/timeout_final.py` | `experiments/review3/timeout_final.csv`, `timeout_fixed.csv` |
| Main Tab. 5 (CholecT50) | `scripts/cholect50_target.py` | `experiments/cholect50/*.csv` |
| Main Tab. 6 (phase) | `scripts/phase/extract_cholec80.py`, `scripts/phase/phase_certify.py` | `experiments/phase/phase_certify.csv` |
| Main text: non-random splits, calibration size, 1 fps, stream precision | `scripts/review3/stream_extra.py` | `experiments/review3/{stream_extra,fixed_splits,onefps}.csv` |
| Supp. per-operation audit table | `scripts/review3/audit_table.py` | `experiments/review3/audit_official_split.csv` |
| Supp. summaries, selective, annotators, shift, anatomy pipeline | `scripts/review/*.py` | `experiments/review/*.csv` |
| Supp. retarget, holdout, utility, alternative stop rules, cert. training, Cholec80 strict | `scripts/review2/*.py` | `experiments/review/*.csv` |
| All LaTeX tables | `scripts/make_tables.py`, `scripts/review/make_review_tables.py`, `scripts/review2/make_rebuttal_tables.py`, `scripts/review3/make_tables_r3.py` | `paper/tables/**` |
| Numbers quoted in the text | `scripts/paper_numbers.py`, `scripts/review3/make_tables_r3.py` | `experiments/numbers.json`, `experiments/review3/numbers_r3.json` |

Conventions checked by the scripts: statements are issued iff score > threshold; order statistic
`ceil((V+1)(1-alpha))`-th smallest, +inf if it exceeds V; operations without unsafe scored frames are
left out of calibration and counted (as error-free) in every test denominator; CRC uses the floor
`floor((n+1) alpha) - 1`; conditional STOP never fires when `floor(alpha_s (V1+1)) = 0`.
