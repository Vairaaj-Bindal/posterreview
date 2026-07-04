# Phase 4 — The Scoring Head, and an Important Negative Result

**Date:** 2026-07-04
**Modules:** `research/scripts/extract_features.py`, `train_scoring_head.py`
**Data:** `research/data/features_iclr2024.parquet` (293 posters)

## What we did

Built the scoring-head harness (Stanford's method: features → regression onto
real ICLR reviewer scores) and a **design-only baseline** that needs no API key.
Extracted deterministic design metrics + metadata text features for a stratified
293-poster sample of the Phase-1 linkage (150 poster / 75 spotlight / 56 oral /
12 reject), each carrying its real OpenReview mean rating and decision tier.
Evaluated with repeated stratified cross-validation (held-out).

## Result: design + text features do NOT predict the reviewer rating

| Target | Model | Held-out metric |
|---|---|---|
| Reviewer mean rating | Ridge (design+text) | **Spearman 0.01 ± 0.10** |
| Reviewer mean rating | GBM (design+text) | Spearman 0.05 ± 0.14 |
| Reviewer mean rating | Ridge (design-only) | Spearman 0.06 ± 0.12 |
| High tier (spotlight/oral) | LogReg | **ROC-AUC 0.52 ± 0.05** (chance = 0.50) |

Strongest single feature: `is_portrait`, |Spearman| = 0.17. Every other design
metric is < 0.15. This is a **clean null** — not a bug:
- The label is valid and well-formed: mean rating separates tiers monotonically
  (reject 5.27 → poster 6.31 → spotlight 7.00 → oral 7.59).
- The metrics engine is validated (Phase 2). It's simply that these features
  carry almost no information about the reviewer score.

## Why — the insight that reshapes the project

**ICLR reviewer scores measure the *paper's* scientific merit, not the *poster's*
quality.** The poster is a communication artifact produced *after* review, and its
design (font sizes, contrast, columns, whitespace) is largely independent of
whether the underlying research was judged important. A beautiful poster of a
rejected paper and an ugly poster of an oral both inherit their paper's score.

So the Phase-1 assumption — "use PosterSum→OpenReview ratings as the training
signal for a poster scoring head" — is only **half right**:

- ❌ **Not valid** for *design / communication* dimensions (readability, hierarchy,
  figure quality, whitespace). These are orthogonal to paper merit by
  construction, which is exactly what the null result shows.
- ✅ **Still valid** for *content* dimensions a reader infers from the poster
  (importance, contribution, contextualization). A VLM that reads the poster's
  research claims can predict paper merit the way Stanford predicts it from paper
  text — because the poster conveys the research. That path needs the API key
  (LLM dimension scores), so it's untested here; the harness is built to accept
  those columns the moment they exist.

## Consequence — corrected architecture

The overall poster-quality score must come from **two ground-truth sources, each
matched to what it actually measures**, not from OpenReview alone:

1. **Design / communication quality** → validate against **poster-native** ground
   truth: P2PEval's 121 human checklists (content fidelity) + Paper2Poster's 6
   VLM-judge rubrics (aesthetics). These score the *poster*. (Phase-1 assets,
   already in hand.)
2. **Content merit** → the OpenReview linkage remains the right target, but only
   for LLM-read content dimensions (Stanford's mechanism). Design metrics stay as
   *descriptive, grounded facts in the review* (their real value — Phase 3), not
   as predictors of a paper score they were never going to predict.

This is a course-correction, and a credibility win: the honest negative result
tells us precisely which signal trains which head, instead of shipping a scoring
model that silently correlates design with the wrong target.

## What's reusable

- `extract_features.py` — resumable feature extraction over the linked posters
  (design metrics via the image/OCR engine + metadata text features + labels).
- `train_scoring_head.py` — the full train/eval harness (repeated-CV Spearman,
  AUC, calibration, ablation, feature importance). Adding LLM dimension scores is
  a one-line change to the feature list.

## Next

- **Phase 4b (needs key):** generate LLM content-dimension scores for the 293
  posters, add them as features, and re-run — test whether content dimensions
  read from the poster predict the reviewer rating (the valid half of the signal).
- **Phase 4c:** fit + calibrate a *design* scoring head against P2PEval /
  Paper2Poster rubric labels (poster-native, no key for the labels themselves).
- Then the web app, with an honest tech-overview page reporting all of the above.
