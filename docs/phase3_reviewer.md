# Phase 3 — The Reviewer (retrieval-grounded + metrics-grounded)

**Date:** 2026-07-03
**Modules:** `research/scripts/arxiv_retrieval.py`, `research/scripts/review.py`
**Status:** Pipeline complete. Every deterministic stage validated on real
posters. The two LLM calls are wired to the current Anthropic SDK but **untested
pending an API key** (no `ANTHROPIC_API_KEY` and no `ant` CLI on this machine).

## What it does — the full paperreview.ai recipe, for posters

`review.py` runs five stages end to end:

1. **Parse** — `poster_metrics.analyze_poster(pdf)` → text + deterministic design
   metrics (Phase 2/2b).
2. **Render** — poster → PNG (≤1600px) for the vision model.
3. **Ground in prior work** — generate 4 multi-angle arXiv queries from the
   poster's claims (Claude), retrieve + dedup related papers (`arxiv_retrieval`,
   free arXiv API). This is paperreview.ai Stage 2, adapted.
4. **Review** — a single Claude (`claude-opus-4-8`, vision + adaptive thinking)
   call that **fuses three inputs**: the poster image, the *measured* design
   metrics (passed as ground truth), and the retrieved related work. Structured
   output (`messages.parse` + Pydantic) across 8 dimensions:
   - content: importance, claim_support, contextualization
   - communication: visual_hierarchy, readability, figure_quality,
     takeaway_clarity, self_containedness
   Each dimension gets a 1–5 score, a rationale that must cite measured facts,
   and concrete suggestions. Plus strengths, weaknesses, and grounded design
   notes.
5. **Score** — provisional overall = unweighted mean of the LLM dimension scores.
   **The trained scoring head (regression on the PosterSum→OpenReview labels
   from Phase 1) is Phase 4** — this is labeled provisional everywhere so the
   through-line stays honest.

## Why it's grounded, not a wrapper

The review prompt embeds the measured metrics verbatim, e.g. for poster 10434:

```
Font sizes normalized to a 48in poster: body ~32pt, heading ~46pt, title ~91pt.
82% of text is below the body legibility floor (fine print).
Text contrast: median 21.0:1, min 10.86:1, 0% below WCAG AA.
Text blocks cover 36% of the poster; 83% is background/open space.
```

The model is instructed to treat these as ground truth and cite them by number,
so readability feedback is anchored to measurements the model can't fudge — the
capability a paper reviewer fundamentally lacks (papers aren't visual).

## Validated without a key (real posters)

- **arXiv retrieval** — runs; returns relevant papers (glaucoma/surgical for the
  medical poster 10434; will be far richer for AI/ML posters, per Stanford's
  arXiv-coverage caveat).
- **Title + reading-order text extraction** — fixed a real bug: raw PDF text
  order made 10434's title come out as "PUBLICATIONS" (the hidden 6pt list leads
  the byte stream). Now uses the metrics engine's artifact-filtered spans →
  correct title ("Rapid learning curve assessment…") and clean column-major
  reading order. Works for vector and OCR posters.
- **Prompt assembly** — the full grounded prompt (7.5k chars + base64 PNG)
  assembles correctly and contains the measured facts.

## The gap (be explicit)

The two Claude calls — query generation and review synthesis — have **not been
run**, because there's no API key or `ant` login on this machine. They're built
against the current SDK (`messages.parse`, vision image blocks, adaptive
thinking) but should be smoke-tested once a key is available.

## How to run it

```bash
export ANTHROPIC_API_KEY=sk-ant-...
cd research/scripts
../../.venv/bin/python review.py ../assets/p2peval_posters/10434.pdf          # pretty
../../.venv/bin/python review.py ../assets/p2peval_posters/10434.pdf --json    # machine
../../.venv/bin/python review.py <poster.pdf> --no-arxiv                        # skip retrieval
```

## Next — Phase 4

Train the scoring head: featurize (LLM dimension scores + deterministic metrics)
→ regress onto the PosterSum→OpenReview reviewer ratings (1,558 ICLR'24 labels
in hand), report Spearman / AUC / calibration on held-out posters, exactly as
Stanford did. That replaces `provisional_score` with a calibrated one and gives
us the honest evaluation page that makes the whole thing credible.
