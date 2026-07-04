# Phase 5 — Zero-Cost Local Reviewer (no API key)

**Date:** 2026-07-04
**Modules:** `research/scripts/local_llm.py`, refactored `review.py`
**Status:** Working end-to-end on Apple Silicon with **zero paid API calls**.

## Goal

Run the whole reviewer on compute alone — no Anthropic/OpenAI key, no metered
service. (Note: paperreview.ai is free *to users* but does pay for inference —
LandingAI extraction, Tavily search, LLM calls. "Zero cost to run" is a stronger
goal, and it's achievable here.)

## What made it easy

The deterministic engine already extracts every *visual* fact (font sizes,
contrast, coverage, columns, palette) and the design score (Phase 2/4c) already
grades design. So the LLM only has to reason over **text + measured metrics +
related work** — a text task. Text models are far cheaper to run locally than
vision models, and most of the "hard" grounding is done deterministically before
the LLM is called.

## Architecture (all local, all free)

| Stage | Engine | Cost |
|---|---|---|
| Parse + design metrics + OCR | PyMuPDF / RapidOCR | $0 local |
| Search queries | **heuristic** (title + salient terms) — no LLM | $0 |
| Related-work retrieval | free arXiv API | $0 |
| Review synthesis (content dims) | **local open model** (MLX or Ollama) | $0 |
| Design score | deterministic rubric (Phase 4c) | $0 |

Query generation was moved off the LLM entirely (deterministic keyword/title
heuristic), so the *only* model call is the review synthesis.

## Engine-agnostic backend (`local_llm.py`)

One interface, two interchangeable engines, auto-selected:

- **`mlx`** — Apple Silicon (this M4). `Qwen2.5-7B-Instruct-4bit`, ~4.3GB, runs on
  the Metal GPU. Full review in ~2 min including cold model load. Dev/testing.
- **`ollama`** — anywhere Ollama runs (esp. a **DGX Spark**, 128GB). Talks to the
  local Ollama HTTP API, so it runs a big model (`llama3.3` 70B, `qwen2.5:72b`)
  for near-paid-API quality. Production.

Selection order: `POSTERREVIEW_ENGINE` env → an Ollama server reachable at
`OLLAMA_HOST` → else MLX. Moving from Mac to Spark is a config switch, no code
change. An optional `--backend anthropic` path remains for when a key *is*
available (highest quality).

## Division of labor (fixes a real bug)

First run exposed the quality gap of a 7B: it **contradicted the measured
metrics** — called a poster's contrast "very poor" and fonts "too large" when the
measurements said contrast 16.3:1 and body 31pt (both good). Fix, aligned with
the Phase-4 correction: **the LLM scores only research-content dimensions**
(importance, claim support, contextualization, takeaway clarity,
self-containedness) — the things judgeable from text — while **design is owned
entirely by the deterministic rubric**. The system prompt tells the model to
defer to the measurements and not judge visual design. After the split, the
contradictions are gone; the review reads cleanly.

## Honest quality note

The local 7B produces coherent, specific, useful content feedback, but it's below
Opus-tier: shallower reasoning, occasional generic suggestions. The two levers:
- **DGX Spark + 70B via Ollama** — the intended production path; a 70B closes most
  of the gap at $0.
- **`--backend anthropic`** — Opus 4.8, highest quality, paid (opt-in).

The deterministic half (metrics, design score, retrieval) is identical regardless
of engine — that's where the novel grounding lives.

## Running it

```bash
# Mac (default, MLX, zero cost):
.venv/bin/python research/scripts/review.py <poster.pdf>

# DGX Spark / Linux box (zero cost, big model):
#   on the Spark:  curl -fsSL https://ollama.com/install.sh | sh
#                  ollama pull llama3.3
#   then point the reviewer at it:
export OLLAMA_HOST=<spark-ip>:11434
export POSTERREVIEW_OLLAMA_MODEL=llama3.3
.venv/bin/python research/scripts/review.py <poster.pdf>

# Opus (highest quality, needs key):
export ANTHROPIC_API_KEY=...
.venv/bin/python research/scripts/review.py <poster.pdf> --backend anthropic
```

## Live on the DGX Spark (done)

The Spark path is working end-to-end. Setup: passwordless SSH key from the Mac,
Ollama already serving on the Spark (GB10, 121GB RAM), reached from the Mac over
an encrypted `ssh -f -N -L 11434:localhost:11434` tunnel (nothing exposed to the
network). One command: `run_review_spark.sh <poster.pdf>` (auto-opens the tunnel).

Model: **qwen2.5:72b** (Vairaaj rejected gemma). Quality is a large jump over the
Mac's 7B — real reviewer-grade, specific critiques that correctly cite the
measured metrics (e.g. *"the alignment of nociceptive maps with tactile maps in
SI is not as strongly supported; the fMRI phase-encoded mapping could be more
clearly explained"*). All at $0.

Speed on the GB10: ~5 min cold (first call loads the 47GB model), **~2.4 min warm**
per review. Async-friendly (paperreview.ai is async too). `keep_alive: 30m` on
the Ollama call keeps the model resident so back-to-back reviews stay warm. If
snappier latency is wanted, a mid-size non-gemma model (e.g. `qwen2.5:32b`) trades
a little quality for speed.

## Next
- Phase 4b (content scoring head) can now run key-free too — generate the LLM
  content-dimension scores with the local model over the 293 labeled posters.
- Web app front door (upload → review), served from whichever engine is available.
