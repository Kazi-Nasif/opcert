# Certified safety checks for surgical video: guarantees must hold per operation

Anonymous code for the CVPR 2027 submission. Research prototype for decision *support* on public,
retrospective data; not a medical device.

## Data (public, CC BY-NC-SA 4.0; see DATASETS.md)
- Endoscapes2023: `bash scripts/get_endoscapes.sh` (CVS keyframes rated by 3 surgeons, anatomy boxes).
- Cholec80 + Cholec80-CVS: download from the dataset pages, then `python scripts/prepare_cholec80.py`.
- CholecT50: download from the dataset page into `data/cholect50/`.

## Environment
Python 3.12, PyTorch >= 2.4, torchvision, transformers, timm, scikit-learn, scipy, pandas,
ortools, matplotlib (`pip install -r requirements.txt`).

## Pipeline (all results are produced by these scripts from stored outputs)

`MANIFEST.md` maps every table and figure of the paper and supplement to its script and result file.
`scripts/review3/audit_table.py` writes a per-operation audit table for the fixed official split.

| Step | Command |
|---|---|
| CVS models (4 backbones x 3 seeds) | `python scripts/finetune.py --arch {dinov2l,resnet50,surgenetxl,endovit} --seed S` |
| Anatomy detector | `python scripts/train_detector.py --seed S` |
| Temporal heads | `python scripts/temporal/extract_finetuned.py --seed S`; `python scripts/temporal/train_temporal.py --model {lstm,tcn,transformer} --feat {ft,frozen} --seed S [--cert 1]` |
| Main comparison (Table 3, Fig. 3, Fig. 5) | `python scripts/compare_all.py`; `python scripts/resplit_check.py` |
| Anatomy and Cholec80 (Table 2) | `python scripts/surgery_level.py`; `python scripts/cholec80_surgery_level.py` |
| CholecT50 (Table 5) | `python scripts/cholect50_target.py --seed S`; `python scripts/cholect50_eval.py` |
| Time-out (Fig. 6) | `python scripts/preclip.py`; `python scripts/review2/timeout_alarm.py` |
| Re-targeted baselines, PAC, fixed split | `python scripts/review2/rebuttal.py`; `python scripts/review2/calib_conditional.py` |
| Robustness (supplementary) | `python scripts/review/{summaries,selective,annotators,anatomy_robustness,shift,timeout_stability}.py` |
| Numbers, tables, figures | `python scripts/paper_numbers.py`; `python scripts/make_tables.py`; `python scripts/review/make_review_tables.py`; `python scripts/review2/make_rebuttal_tables.py`; `python scripts/figures/*.py` |
| Tests | `python -m pytest tests` |

Core certificates: `src/cdss/decide.py` (frame-level and per-operation CVS certificates and baselines),
`src/cdss/anatomy.py` (plausible worlds, projection, anatomy certificates), `src/cdss/riskcontrol.py`
(CRC, Learn-then-Test). Splits: 10 fixed selection videos (seed 0); 200 video-level halvings
(`numpy.random.default_rng(rep)`, rep = 0..199) of the remaining 71 Endoscapes operations.
