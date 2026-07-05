# PosterReview

An agentic reviewer for research posters, modeled on the architecture of
[paperreview.ai](https://paperreview.ai) (Stanford ML Group's Agentic Reviewer),
not on its surface. Upload a poster, get a structured, evidence-grounded review
and a calibrated score.

## Why this isn't a wrapper

paperreview.ai's credibility comes from three things (per their
[tech overview](https://paperreview.ai/tech-overview)):

1. **Structured extraction** — agentic PDF→Markdown parsing with validation.
2. **Retrieval grounding** — multi-query arXiv search + adaptive summarization,
   so novelty/context claims cite actual prior work.
3. **A trained scoring head** — the LLM scores 7 quality dimensions; a linear
   regression *trained on 300 ICLR 2025 submissions with real human review
   scores* maps dimensions → final 1–10. Result: Spearman 0.42 with a human
   reviewer (human–human is 0.41), AUC 0.75 predicting acceptance.

PosterReview replicates that recipe for posters:

| Stage | paperreview.ai | PosterReview |
|---|---|---|
| Extraction | LandingAI ADE, PDF→MD | Vision parse + **deterministic design metrics** (exact font sizes, contrast ratios, text coverage, whitespace from PDF vector data / OCR boxes) |
| Grounding | Tavily→arXiv retrieval | arXiv API retrieval from poster claims (same design) |
| Dimensions | 7 paper-quality dims | ~8 dims split content (claim support, contextualization, importance) vs communication (hierarchy, readability, figure quality, takeaway) |
| Scoring head | Linear regression on ICLR 2025 human scores | Regression on **PosterSum→OpenReview linkage** (16,305 real conference posters → reviewer scores + oral/spotlight/poster tier) + P2PEval human checklist annotations |
| Evaluation | Spearman, AUC, calibration, published | Same, published on a public tech-overview page |

## Data sources

- **PosterSum** ([HF](https://huggingface.co/datasets/rohitsaxena/PosterSum),
  [paper](https://arxiv.org/abs/2502.17540)) — 16,305 poster image URLs from
  ICLR/ICML/NeurIPS 2022–2024 with title/abstract/topics. Linkable to
  OpenReview by title → reviewer scores, decision tier.
- **P2PEval** ([paper](https://arxiv.org/abs/2505.17104)) — 121 paper–poster
  pairs, 1,738 triple-annotated human checklist items (775 visual).
- **Paper2Poster** ([repo](https://github.com/paper2poster/paper2poster), MIT)
  — eval rubric + PaperQuiz (VLM answers quizzes from the poster alone).

## Layout

```
research/
  data/       # datasets, linkage tables (gitignored where large)
  scripts/    # data acquisition + linkage + training pipeline
  notebooks/  # analysis
docs/         # tech overview drafts, eval reports
app/          # (later) web app
```

## Status

- **Phase 1 ✓** — data acquisition + OpenReview linkage (1,558 ICLR'24 posters
  → reviewer ratings/tiers) + P2PEval / Paper2Poster assets. (`docs/phase1_data_report.md`)
- **Phase 2 ✓** — deterministic design-metrics engine (font sizes, contrast,
  coverage, columns, palette), validated on real posters. (`docs/phase2_metrics_engine.md`)
- **Phase 2b ✓** — OCR fallback (RapidOCR) for image-flattened poster PDFs.
- **Phase 3 ✓** — retrieval-grounded + metrics-grounded reviewer (8 scored
  dimensions). (`docs/phase3_reviewer.md`)
- **Phase 5 ✓** — **zero-cost local backend**: runs the whole reviewer with no
  API key on open models (MLX on Apple Silicon, or Ollama with a 70B on a DGX
  Spark / any box). LLM scores content only; design is the deterministic rubric.
  (`docs/phase5_zero_cost.md`)
- **Phase 4 ✓** — scoring-head harness + design-only baseline. **Key finding:**
  OpenReview reviewer scores measure the *paper's* merit, so design metrics don't
  predict them (Spearman ≈ 0) — they're the right target only for LLM-read
  *content* dimensions, while *design* quality must be validated against
  poster-native ground truth (P2PEval / Paper2Poster rubrics). Corrected
  architecture in `docs/phase4_scoring_head.md`.

### The scoring-head ambition, retired on evidence (Phase 4 + 4b)

Two honest null results converged on a structural truth: **you can't predict paper
acceptance from a poster.** Design metrics don't predict the reviewer rating
(Phase 4: Spearman ≈ 0), and neither do LLM-judged content scores (Phase 4b:
Spearman −0.04, high-tier AUC 0.44). The reason: **posters only exist for accepted
papers**, so a poster corpus is ~96% accepted — there's no negative class to learn
from, and the within-accepted gradation (poster/spotlight/oral) is a band even two
humans barely separate. Stanford's paper-scoring result (0.42) doesn't transfer,
because papers have rejects and posters don't. So PosterReview reports a **design
score** (measured, validated) and a **content review + provisional score** (useful
feedback) — quality signals for *improving* a poster, not an acceptance oracle.
Full write-up: `docs/phase4b_content_eval.md`.

### Scoring-head target, corrected (Phase 4)

| Dimension type | Ground truth |
|---|---|
| Content merit (importance, contribution, contextualization) | OpenReview reviewer scores — but only via **LLM dimensions read from the poster** (needs key); design metrics alone are orthogonal to it |
| Design / communication (readability, hierarchy, figure quality) | **Poster-native** labels: P2PEval human checklists + Paper2Poster VLM-judge rubrics — *not* OpenReview |
| Grounded design facts in the review | Deterministic metrics engine (Phase 2) — descriptive, cited by number |

Next: Phase 4b (LLM content-dimension features, needs key), Phase 4c (design head
vs rubric labels), then the web app.
