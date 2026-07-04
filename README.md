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

Phase 1: data acquisition + OpenReview linkage — in progress.
