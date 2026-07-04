# Phase 4c — The Design Score (poster-native, key-free)

**Date:** 2026-07-04
**Modules:** `research/scripts/design_score.py`, `validate_design_score.py`

## Why a rubric, not a learned model

Phase 4 established that OpenReview ratings can't label poster *design* (they score
the paper). And there is **no key-free, poster-native quality-label dataset** for
scientific posters — the P2PEval checklists and Paper2Poster rubrics are *grading
instruments* that need a VLM to apply, and the one public "PosterReward" benchmark
is for advertising/graphic-design posters, a different domain.

So the design score is a **transparent, interpretable rubric** over the
deterministic metrics — every sub-score is a defensible function of a measured
quantity, anchored to established large-format poster guidelines. A reviewer can
see exactly why a poster scored what it did. This is key-free and usable in the
product today.

## The rubric (`design_score.py`)

0–100, weighted sum of five sub-scores:

| Sub-score | Weight | Driven by |
|---|---|---|
| legibility | 30% | body/title font size vs floors, % fine print |
| accessibility | 25% | WCAG contrast (median, % below AA) |
| density_balance | 20% | text coverage (ideal ~30%) + open space (ideal ~45%) |
| structure | 15% | column count (2–4 ideal; 1 or ≥5 penalized) |
| palette | 10% | saturated-hue restraint + palette size |

On real posters it differentiates sensibly — e.g. a busy, dense poster (49% text,
8 saturated colors) scores 70, a clean well-structured one scores 95.

## Validation by controlled degradation (`validate_design_score.py`)

With no labels to regress against, we test the score the way you test any
label-free quality metric: **perturb real posters in known-bad directions, re-run
the full metrics engine on the perturbed pixels, and check the score drops.**
Non-circular — the score never sees the perturbation, only the re-measured pixels.
Both baseline and degraded go through the same image/OCR path, isolating the
perturbation. 6 posters × 4 perturbations:

| Perturbation | Target sub-score | Score dropped | Mean target Δ |
|---|---|---|---|
| contrast_down | accessibility | **5/6 (83%)** | −15.1 |
| palette_clutter | palette | **5/6 (83%)** | −61.4 |
| blur | accessibility | **6/6 (100%)** | −15.1 |
| overcrowd | density_balance | 4/6 (67%) | +9.5 |

**The three monotonic quality axes validate strongly** (83–100%): lowering
contrast, cluttering the palette, or blurring the poster reliably lowers the
score, and the targeted sub-score moves the right way by a large margin. If it
didn't, the score would be meaningless; it does, so the rubric is capturing real
design quality.

**`overcrowd` is the honest exception, and it's correct behavior:**
`density_balance` is a *two-sided* metric — both too-sparse and too-dense posters
are penalized (it peaks at ~30% text / ~45% open space). Several test posters
were already too sparse (e.g. one with 83% open space), so *adding* content moved
them **toward** the ideal and raised the sub-score. A one-directional perturbation
therefore can't cleanly validate a two-sided axis — that's a property of the test,
not a flaw in the score.

## Integration

`review.py` now emits the deterministic `design_score` (key-free) alongside the
LLM content/communication review (key-dependent), matching the Phase-4 corrected
architecture: **design quality from the transparent rubric, content merit from the
LLM.** The design score prints as a 0–100 with its five sub-scores and the
measured fact behind each.

## Limitations / next

- Weights and anchors are v1, hand-set from poster guidelines — reasonable but not
  yet tuned against human design preferences (would need a labeled set).
- The density axis can't be validated by one-directional degradation; a paired
  too-sparse / too-dense test set would validate it directly.
- Legibility on image-flattened posters uses OCR font estimates (Phase 2b), which
  overshoot for large display text — the score notes when a poster is OCR-sourced.
