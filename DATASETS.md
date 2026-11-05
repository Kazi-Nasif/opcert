# Datasets used in Clinical ours

All three come from the CAMMA group (University of Strasbourg / IHU Strasbourg / IRCAD, France),
recorded in real laparoscopic cholecystectomies, and are released for non-commercial research
under **CC BY-NC-SA 4.0**. Cite the dataset papers below and credit the dataset in any figure
that shows frames.

## 1. Endoscapes2023 (main dataset: CVS + anatomy)
- **What:** 201 laparoscopic cholecystectomy videos. 11,090 keyframes (one per 5 s of the
  dissection phase) rated on the three CVS criteria by three surgeons; 1,933 frames with tool and
  anatomy boxes; 422 frames with segmentation masks (50 videos). Subsets: Endoscapes-CVS201
  (58,813 dissection frames at 1 fps), Endoscapes-BBox201, Endoscapes-Seg50.
- **Official splits:** 120 / 41 / 40 videos (train / val / test).
- **Source:** Strasbourg (IHU / IRCAD), CAMMA group.
- **Download:** https://github.com/CAMMA-public/Endoscapes (public zip, used here via
  `scripts/get_endoscapes.sh`: https://s3.unistra.fr/camma_public/datasets/endoscapes/endoscapes.zip).
  Also on PhysioNet (credentialed access): https://www.physionet.org/content/endoscapes-2023/
- **Dataset paper (cite):** P. Mascagni, D. Alapatt, A. Murali, et al., N. Padoy. *Endoscapes, a
  critical view of safety and surgical scene segmentation dataset for laparoscopic
  cholecystectomy.* Scientific Data 12, 331 (2025). doi:10.1038/s41597-025-04642-4.
  Preprint: arXiv:2312.12429 (*The Endoscapes Dataset for Surgical Scene Segmentation, Object
  Detection, and Critical View of Safety Assessment: Official Splits and Benchmark*).
- **Also requested by the authors:** Murali et al., *Latent graph representations for critical
  view of safety assessment*, IEEE TMI 2023; Alapatt et al., *Temporally constrained neural
  networks (TCNN)*, arXiv:2112.13815, 2021.
- **Papers using it (verified):** the benchmark paper above; LG-CVS (Murali et al., IEEE TMI 2023);
  SV2LSTG (Murali et al., *Encoding surgical videos as latent spatiotemporal graphs*, MICCAI 2023);
  this work. Code for the benchmark models: https://github.com/CAMMA-public/SurgLatentGraph

## 2. Cholec80 + Cholec80-CVS (external test set)
- **Cholec80 videos:** 80 cholecystectomies by 13 surgeons, University Hospital of Strasbourg /
  IRCAD, 25 fps, with 7 surgical phases and 7 tool-presence labels.
  Download: CAMMA datasets page, http://camma.u-strasbg.fr/datasets (see also
  https://github.com/CAMMA-public/TF-Cholec80).
  **Cite:** A. P. Twinanda, S. Shehata, D. Mutter, J. Marescaux, M. de Mathelin, N. Padoy.
  *EndoNet: A deep architecture for recognition tasks on laparoscopic videos.* IEEE TMI 36(1), 2017.
- **Cholec80-CVS labels:** CVS criteria scored 0/1/2 by surgeons on 572 video segments of all 80
  Cholec80 videos; unannotated time counts as 0; CVS achieved = total score >= 5 (the dataset's
  definition). Distributed as an XLSX file linked from the paper.
  **Cite:** M. S. Ríos, M. A. Molina-Rodriguez, D. Londoño, et al. *Cholec80-CVS: An open dataset
  with an evaluation of Strasberg's critical view of safety for AI.* Scientific Data 10, 194 (2023).
  doi:10.1038/s41597-023-02073-7.
- **Our use:** calot-triangle-dissection phase at 1 fps; videos 66, 67, 68, 70, 71, 72 removed
  (they also appear in Endoscapes) -> 74 videos, 69,610 frames, CVS in 406 frames (0.58%, 12 videos).
- **Papers using Cholec80 (verified, many more exist):** EndoNet (TMI 2017); SV-RCNet (Jin et al.,
  TMI 2018); TeCNO (Czempiel et al., MICCAI 2020); Trans-SVNet (Gao et al., MICCAI 2021).
  For Cholec80-CVS, check "cited by" of the Sci Data paper on Google Scholar.

## 3. CholecT50 (third task: what is being clipped / cut)
- **What:** 50 cholecystectomy videos (45 from Cholec80 + 5 in-house), every frame (1 fps)
  labelled with <instrument, verb, target> triplets, 100 triplet classes.
- **Download:** https://github.com/CAMMA-public/cholect50 (and http://camma.u-strasbg.fr/datasets).
  Variants: CholecT45 (without the challenge test set); official splits in arXiv:2204.05235.
- **Cite:** C. I. Nwoye, T. Yu, C. Gonzalez, B. Seeliger, P. Mascagni, D. Mutter, J. Marescaux,
  N. Padoy. *Rendezvous: Attention mechanisms for the recognition of surgical action triplets in
  endoscopic videos.* Medical Image Analysis 78, 102433 (2022). arXiv:2109.03223.
  The dataset README also asks to cite: Nwoye et al., *CholecTriplet2021: A benchmark challenge
  for surgical action triplet recognition*, Medical Image Analysis (arXiv:2204.04746); Nwoye et al.,
  *CholecTriplet2022* (arXiv:2302.06294); Nwoye & Padoy, *Data splits and metrics for method
  benchmarking on surgical action triplet datasets* (arXiv:2204.05235).
- **Papers using it (verified):** Rendezvous (MedIA 2022); Tripnet (CAMMA, MICCAI 2020, on the
  earlier CholecT40); the CholecTriplet2021 / 2022 challenges; EndoViT (Batić et al., IJCARS 2024,
  action-triplet recognition); this work.
- **Our use:** clipper-clip and scissors-cut frames; target = cystic duct / cystic artery / other;
  20 training videos (incl. the 6 overlapping with Endoscapes), 4 for model selection, 30 for
  calibration/evaluation (3,031 frames).

## Overlaps handled
Cholec80 videos 66-72 (except 69) also appear in Endoscapes; they are excluded from the Cholec80
test set, and the 6 CholecT50 videos that appear in Endoscapes are placed in CholecT50 training,
so no operation is evaluated after being trained on.
