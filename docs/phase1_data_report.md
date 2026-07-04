# Phase 1 Data Report — Training Set Shape & Feasibility

**Date:** 2026-07-03
**Verdict:** Feasible. We have real ground truth for all three scoring pillars,
at scale exceeding the Stanford paperreview.ai training set.

## What we set out to prove

paperreview.ai's credibility rests on a scoring head trained on 300 ICLR 2025
submissions with real human review scores (150 train / 147 test). The open
question for a *poster* version was ground truth: posters have no OpenReview
page of their own. Phase 1 tested whether we can manufacture that ground truth.
We can — three independent sources, one per scoring pillar.

## Pillar 1 — Overall quality (content signal): PosterSum → OpenReview

- **PosterSum** (`rohitsaxena/PosterSum`, HF): 16,305 real conference posters,
  ICLR/ICML/NeurIPS 2022–2024. Fields: conference, year, paper_id, title,
  abstract, topics, `image_url` (direct PNG on the venue CDN).
  Full metadata pulled → `research/data/postersum_metadata.parquet`.
- **Linkage** to OpenReview reviewer scores via title-normalization join
  against `smallari/openreview-iclr-peer-reviews` (19,076 rows, ICLR 2024+2025,
  full per-review ratings + decision tier).

### Result (ICLR 2024 prototype)

| Metric | Value |
|---|---|
| PosterSum ICLR 2024 posters | 1,799 |
| Matched to OpenReview by title | **1,558 (86.6%)** |
| Posters with ≥1 extractable reviewer rating | 1,558 (100% of matched) |
| Mean-rating range / median | 4.0–8.5 / 6.33 |

**Ratings track decision tier cleanly** (this is the signal the scoring head learns):

| Tier | n | mean reviewer rating (std) |
|---|---|---|
| Reject | 12 | 5.27 (0.61) |
| Accept (poster) | 1,243 | 6.27 (0.58) |
| Accept (spotlight) | 247 | 7.08 (0.55) |
| Accept (oral) | 56 | 7.59 (0.59) |

Spearman(mean_rating, tier_rank) = **0.514**.

Output: `research/data/linkage_iclr2024.parquet` (poster metadata + decision +
per-reviewer ratings + mean_rating).

> **Caveat / bias to document publicly:** PosterSum contains *accepted* posters
> only, so the overall-quality label is dominated by the 6.0–7.5 band (few true
> rejects). This compresses the low end. Stanford had the same issue (accepted
> papers benefit from decisions already made on their scores). Mitigations for
> Phase 3: (a) pull rejected-submission reviews from the same mirror to
> synthesize low-quality posters, (b) lean on P2PEval + augmentation for the
> tails, (c) report calibration honestly like they do.

### Scale-up path (not yet executed, low-risk)

- ICLR 2022/2023 posters (2,243 more) need a 2022/2023 review mirror —
  `AlgorithmicResearchGroup/openreview-papers-with-reviews` covers ICLR 2022+
  (per-review rows, needs aggregation). Est. +~1,900 linked.
- NeurIPS 2022/2023 (7,138 posters) and ICML — reviews are public on
  OpenReview for recent years; same join. Biggest volume lever.
- The live OpenReview **API v2 is now behind a bot challenge (HTTP 403)**, so
  direct calls are out; HF mirrors are the reliable path. Documented in
  `scripts/link_openreview.py` (kept for reference; superseded by the mirror
  join baked into the linkage step).

## Pillar 2 — Design quality: Paper2Poster VLM-judge rubrics

Pulled from the NeurIPS-2025 Paper2Poster repo (MIT) →
`research/data/p2p_assets/vlm_judge_rubrics/`. **Six behaviorally-anchored
5-point rubrics**, ready to use as VLM-as-judge dimensions:

- **Aesthetic:** `aesthetic_element`, `aesthetic_engagement`, `aesthetic_layout`
- **Information:** `information_low_level`, `information_logic`, `information_content`

Each is a full system-prompt + 1–5 scale with explicit anchors per score and a
strict `{"reason","score"}` JSON output contract. Example (`aesthetic_layout`):
scans for alignment/spacing/white-space/reading-path, "5 very rarely granted."
These map directly onto our *communication* dimensions and, crucially, we can
**ground them in the deterministic design metrics** (measured font sizes,
contrast, text coverage) so the judge cites facts instead of vibes.

Also saved: `poster_eval_utils.py` (judge harness), `eval_poster_pipeline.py`
(stats/qa/judge/aesthetic metrics), `create_paper_questions.py` (PaperQuiz gen).

## Pillar 3 — Content-fidelity ground truth: P2PEval human checklists

`ASC8384/P2PEval` (HF): **121 paper–poster pairs**, each folder has `paper.pdf`,
`poster.pdf`, `poster.png`, figure PNGs, and a **human-authored
`checklist.yaml`** — weighted content-fidelity items (`description` +
`max_score` 3–5, some tied to a specific `figure`). Sample confirmed
(`research/data/p2peval_samples/10434_checklist.yaml`): ~14 items covering
title, sections, each figure, contact info, key claims. This is the ground
truth for the *"does the poster faithfully convey the work"* dimension and to
validate our PaperQuiz adaptation.

## Bottom line for the scoring head

| Pillar | Label | Source | N | Status |
|---|---|---|---|---|
| Overall quality | reviewer mean rating + tier | PosterSum→OpenReview | 1,558 (ICLR'24) → ~10k with scale-up | ✅ working |
| Design quality | 6× 1–5 rubric scores | Paper2Poster judges | rubrics in hand; scores generated at inference | ✅ assets in hand |
| Content fidelity | weighted checklist | P2PEval | 121 pairs | ✅ available |

Stanford trained a 7-dim→score regression on 300 papers. We start with **1,558
labeled posters from one conference-year alone** and a clear path to ~10k.
Feasibility is not the risk anymore; the acceptance-only label bias is the thing
to manage, and we have three mitigations.

## Recommended next phase

**Phase 2 — parsing & deterministic design-metrics engine.** Given a poster
PDF/PNG: extract text blocks with exact font sizes & colors (PDF vector data;
OCR-box fallback for image-only), compute contrast ratios, text-coverage %,
column/grid structure, figure count & area, white-space balance. This is the
piece paperreview.ai *can't* do (papers aren't visual) and is our moat — the
review grounds every design claim in a measured number. Validate metrics
against the PosterSum PNGs and the P2PEval posters.
