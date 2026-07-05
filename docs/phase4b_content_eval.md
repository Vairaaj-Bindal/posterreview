# Phase 4b — Do LLM content scores predict the reviewer rating? (No — and why that's the real finding)

**Date:** 2026-07-04
**Modules:** `research/scripts/phase4b_content_scores.py` (Spark 72B), evaluation over
the 293 labeled ICLR'24 posters.

## Setup

Phase 4 showed *design* metrics don't predict the reviewer rating (they measure
the paper, not the poster). The remaining hope was Stanford's actual mechanism:
LLM-judged *content* dimensions. So we scored 5 content dimensions (importance,
claim support, contextualization, takeaway clarity, self-containedness) for all
293 labeled posters with the Spark's `qwen2.5:72b` — zero cost — and tested
whether they predict the real OpenReview mean rating and decision tier.

## Result — a second clean null

Held-out, repeated cross-validation:

| Features → reviewer mean rating | Spearman |
|---|---|
| design-only | +0.06 |
| text-only | −0.11 |
| **content-only (LLM)** | **−0.04** |
| content + design + text | +0.03 |

| Features → high tier (spotlight/oral) | ROC-AUC |
|---|---|
| design-only | 0.53 |
| **content-only (LLM)** | **0.44** (below chance) |
| all | 0.49 |

Per-dimension Spearman with the rating maxes out at 0.06. The content scores are
also low-variance (means 3.3–4.0 — the LLM rates most accepted abstracts "pretty
good"). Nothing predicts the rating.

## Why — the fundamental reason, and it reshapes the project

Stanford got AI-vs-human Spearman 0.42 on *papers*. It doesn't transfer to
posters, for a structural reason:

**Posters only exist for *accepted* papers.** A rejected paper never gets a poster
slot, so a poster dataset (PosterSum) is ~96% accepted by construction. The only
label variation left is *within* accepted work — poster vs spotlight vs oral — a
narrow, noisy band that even two human reviewers barely separate (their agreement
is 0.41 *with the full paper in hand*). Predicting that fine gradation from a
poster's abstract, or from its design, is not achievable — there is essentially
**no negative class to learn from.**

This is not a modeling failure to fix with a better model or more data. It's
inherent to the medium. Stanford could build a calibrated acceptance predictor
because papers have rejects; posters don't.

## What this means for PosterReview (the honest repositioning)

The product's value is **not** a calibrated "will this be accepted" score — that
signal is structurally unavailable for posters, and we won't fake it. The value
is what authors actually want and what we *can* deliver and validate:

1. **A deterministic design score** — measured, interpretable, and validated by
   controlled degradation (Phase 4c). Real and trustworthy.
2. **Grounded, specific content feedback** — the 72B review, which reads the
   research and gives actionable critique, grounded in retrieved related work and
   the measured design facts. Validated by inspection; genuinely strong.

So the "trained scoring head that predicts acceptance" ambition is retired, on
evidence. The reviewer reports a **provisional content score (mean of LLM
dimensions)** and a **design score**, both clearly labeled as quality signals for
*improving the poster* — not as an acceptance oracle. That's the honest, defensible
framing, and it's the one paperreview.ai's own value ultimately rests on too
(actionable feedback, fast).

## Artifacts

- `content_scores_iclr2024.parquet` — 293 posters × 5 LLM content scores (Spark,
  $0). Reusable if a score-diverse poster corpus ever appears.
- The evaluation is reproducible from the two parquets + the harness.

## Bottom line

Two honest nulls (design in Phase 4, content here) converge on one truth: you
can't predict paper-acceptance from a poster, because posters are an
accepted-only medium. The tech-overview page will say exactly this — it's more
credible than a manufactured correlation, and it correctly points users at what
the tool is actually good for.
