# Phase 2 — Deterministic Design-Metrics Engine

**Date:** 2026-07-03
**Module:** `research/scripts/poster_metrics.py` — `analyze_poster(pdf) -> PosterMetrics`
**Status:** Working and validated on 8 real P2PEval poster PDFs.

## Why this is the moat

paperreview.ai grounds *content* claims in retrieved arXiv work. Posters are
visual, so we additionally ground *design* claims in measured numbers: the review
can say "body text is 32pt-equivalent, title 91pt, black-on-white at 21:1
contrast, 3 columns, 82% of the text is fine print" — facts, not a VLM's guess.
Papers have no equivalent; this is what makes PosterReview more than a wrapper.

## What it measures

Auto-detects two regimes and computes what each supports:
- **vector** (real PDF text): exact font sizes, colors, everything below.
- **image** (flattened poster): render-based metrics only (coverage, contrast,
  columns, palette work; font sizes need OCR — flagged, deferred to Phase 2b).

Metrics: poster size + nearest standard + orientation; body/heading/title font
size (normalized to a 48-inch long edge so it's authoring-scale- and
orientation-invariant); % fine print below the legibility floor; text-block
footprint; background/open fraction (background-color aware); raster-figure
count; column count; palette size + saturated-hue count; WCAG contrast
(median/min/% below AA). Emits human-readable **flags** for every threshold
breach.

## Validation — bugs found by looking at real posters

Every metric was checked against the rendered poster image. Naive versions were
wrong; these are the fixes that made outputs trustworthy:

1. **Font size hijacked by non-body text.** A char-weighted mode called body
   text "6pt" on poster 10434 — it was a *hidden 6pt publications list rendered
   under the figures* (21k of 25k chars). Fix: body = dominant size among
   **readable, wordy** spans (real words, ≥ legibility floor). Now 32pt (correct).
   The hidden text is surfaced as an honest "82% is fine print" flag.
2. **Figure coverage counted the full-page background fill** (→ 64–100% figures,
   0% whitespace everywhere). Fix: exclude page-spanning elements; report
   background/open via **dominant-background-color match** (handles colored
   posters like 10629's gray, where near-white gave a false 0%).
3. **Contrast was implausibly bad** (1.4:1 on a black-on-white poster) because
   tiny chart labels over busy backgrounds dominated. Fix: contrast over
   readable wordy spans only → 21:1 (correct).
4. **Columns detected as 1** on a clean 3-column poster (thin gutters bridged in
   x-projection). Fix: **visual gutter detection** — vertical strips that stay
   background down the body height. Gets 10434→3 and even the image poster
   10488→3 (pixel-based, so it works in both regimes).
5. **"Text covers 99%"** on poster 118740 — a diagonal "Creative Commons"
   **watermark** whose rotated bbox was as tall as the page. Fix: drop artifact
   spans (bbox height > 30% page or area > 4% page) everywhere.

## Known limitations (documented, not silently ignored)

- **Column count** is exact for clean grids, returns 1 for irregular/mixed-grid
  posters (e.g., 10629 is 3-col over 2-col; no single full-height gutter). This
  is a defensible abstention, not a silent wrong answer.
- **White chart interiors count as background**, so "open %" overstates true
  whitespace on white posters with big charts. We label the metric as
  "background-color area," not "whitespace," to stay honest. A region-based
  figure mask (Phase 2b) would fix this.
- **Image-regime posters** get no font metrics until OCR is wired in (Phase 2b:
  tesseract or the VLM itself). Coverage/contrast/columns/palette still work.

## Phase 2b — OCR fallback for image-flattened posters (done)

Some poster PDFs are flattened images (10488 was 1,337 image tiles, 8 chars of
real text). These now get font metrics + text content via **RapidOCR**
(`rapidocr-onnxruntime`, ONNX — pip-only, no system binary; Homebrew/tesseract
weren't available). OCR spans reuse the exact same `Span` shape, so all
downstream classification (body/heading/title, contrast, coverage, columns) runs
identically. Font size = OCR box height × 0.95 (calibrated). `text_source` field
records `vector | ocr | none`; if RapidOCR isn't installed the engine degrades
gracefully to render-only metrics.

**Controlled validation** — flattened a vector poster (10434) whose true font
sizes are known, ran the OCR path, compared:

| metric | vector truth | OCR | |
|---|---|---|---|
| body | 32pt | 33pt | ✓ |
| text coverage | 36% | 36% | ✓ |
| background | 83% | 82% | ✓ |
| columns | 4 | 3 | ✓ |
| contrast | 21:1 | 10:1 | ✓ same verdict |
| title | 91pt | 166pt | ✗ ~1.8× over |
| heading | 46pt | 76pt | ✗ over |

- **Body size (the key legibility signal) is accurate** — single-line boxes
  calibrate cleanly.
- **Large display text (titles/headings) overshoots** because OCR merges stacked
  lines into one tall box. The error is conservative for the "title too small"
  flag (we under-claim smallness, never false-alarm). Documented, not hidden.
- OCR sees only *visible* text: it recovered the 4.7k readable chars and ignored
  10434's 21k hidden fine-print — a feature.
- ~6s/poster (one-off, acceptable). Model is a lazy singleton.

## Next (Phase 3)

- Feed these metrics as grounded evidence into the review generator, and as
  features into the scoring head alongside the PosterSum→OpenReview labels.
