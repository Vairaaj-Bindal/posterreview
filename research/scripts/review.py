"""PosterReview end-to-end reviewer.

Pipeline (mirrors paperreview.ai, adapted for posters):
  1. Parse the poster PDF -> text + deterministic design metrics (poster_metrics)
  2. Render the poster to an image for the vision model
  3. Generate arXiv search queries from the poster's claims (Claude)
  4. Retrieve related prior work (arxiv_retrieval)
  5. Generate a structured review that FUSES the poster image, the measured
     design facts, and the retrieved related work (Claude, structured output)

The design metrics are passed in as grounding so the review cites measured facts
("body text is 32pt-equivalent; 82% of the text is fine print") instead of
guessing — the thing a paper reviewer can't do because papers aren't visual.

Usage:  python review.py <poster.pdf> [--json] [--no-arxiv]
Requires an Anthropic API key (ANTHROPIC_API_KEY) for steps 3 and 5.
"""
from __future__ import annotations

import argparse
import base64
import dataclasses
import json
import sys
from typing import List

import fitz
from pydantic import BaseModel, Field

import poster_metrics
import arxiv_retrieval

MODEL = "claude-opus-4-8"
IMAGE_LONG_EDGE_PX = 1600  # under the 2576px high-res cap; enough for legibility

# ---- dimensions: content (what paperreview.ai scores) + communication (our add) ----
DIMENSIONS = [
    ("importance", "content", "Importance of the research question / problem."),
    ("claim_support", "content", "Are the claims adequately supported by the shown results?"),
    ("contextualization", "content", "Positioning vs prior work / novelty, given the retrieved related papers."),
    ("visual_hierarchy", "communication", "Does the layout guide the eye and establish clear reading order?"),
    ("readability", "communication", "Legibility: font sizes, contrast, text density (use the measured metrics)."),
    ("figure_quality", "communication", "Are figures clear, well-labeled, and doing real explanatory work?"),
    ("takeaway_clarity", "communication", "Can a passerby grab the main message in ~30 seconds?"),
    ("self_containedness", "communication", "Does the poster stand alone without the author narrating it?"),
]


class Dimension(BaseModel):
    name: str
    category: str
    score: int = Field(description="1 (poor) to 5 (excellent)")
    rationale: str = Field(description="2-4 sentences; cite measured design facts where relevant")
    suggestions: List[str] = Field(description="1-3 concrete, actionable fixes")


class PosterReview(BaseModel):
    one_line_summary: str
    dimensions: List[Dimension]
    top_strengths: List[str]
    top_weaknesses: List[str]
    grounded_design_notes: List[str] = Field(
        description="Specific claims tied to the measured metrics (e.g. 'body ~18pt is below the legibility floor')"
    )


class Queries(BaseModel):
    queries: List[str]


# ---------- steps ----------
def _poster_spans(page, m):
    """Artifact-filtered, wordy spans (vector or OCR) — excludes hidden fine print."""
    if m.text_source == "ocr":
        spans = poster_metrics._ocr_spans(page, m.scale_to_48in)
    else:
        spans = poster_metrics._extract_spans(page, m.scale_to_48in)
    return [s for s in spans if poster_metrics._wordy(s.text)]


def extract_title_and_text(page, m):
    """Return (title, reading-ordered text). Title = spans near the title font size;
    text = readable spans sorted column-major (poster reading order)."""
    spans = _poster_spans(page, m)
    if not spans:
        return "(untitled)", page.get_text("text")[:6000]

    # Title: the large spans near the top, at ~title font size.
    tcut = 0.8 * m.title_pt if m.title_pt else max(s.norm_pt for s in spans)
    title_spans = sorted((s for s in spans if s.norm_pt >= tcut),
                         key=lambda s: (s.bbox[1], s.bbox[0]))
    title = " ".join(s.text.strip() for s in title_spans)[:200] or "(untitled)"

    # Reading order: bucket by column (x), then top-to-bottom within column.
    W = page.rect.width
    ncol = max(m.n_columns, 1)
    def col(s):
        return int(((s.bbox[0] + s.bbox[2]) / 2) / W * ncol)
    ordered = sorted(spans, key=lambda s: (col(s), s.bbox[1], s.bbox[0]))
    text = " ".join(s.text.strip() for s in ordered)
    return title, text[:6000]


def render_image_b64(page) -> str:
    scale = IMAGE_LONG_EDGE_PX / max(page.rect.width, page.rect.height)
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    return base64.standard_b64encode(pix.tobytes("png")).decode()


def metrics_summary(m) -> str:
    lines = [
        f"Poster size: {m.print_w_in}x{m.print_h_in} in ({m.orientation}, ~{m.nearest_std_size}).",
        f"Text source: {m.text_source}. Columns: {m.n_columns}. "
        f"{'Raster figures: %d.' % m.n_figures if m.n_figures >= 0 else 'Figure count: n/a (flattened).'}",
    ]
    if m.text_source in ("vector", "ocr"):
        est = " (OCR-estimated)" if m.text_source == "ocr" else ""
        lines.append(f"Font sizes normalized to a 48in poster{est}: body ~{m.body_pt:.0f}pt, "
                     f"heading ~{m.heading_pt:.0f}pt, title ~{m.title_pt:.0f}pt. "
                     f"Legibility floor for body is {poster_metrics.BODY_MIN_PT:.0f}pt.")
        lines.append(f"{m.pct_body_below_floor:.0f}% of text is below the body legibility floor (fine print).")
    lines.append(f"Text blocks cover {m.text_coverage*100:.0f}% of the poster area; "
                 f"{m.whitespace*100:.0f}% is background/open space.")
    lines.append(f"Text contrast: median {m.median_contrast}:1, min {m.min_contrast}:1, "
                 f"{m.pct_text_below_wcag:.0f}% below WCAG AA (4.5:1).")
    lines.append(f"Palette: {m.n_palette_colors} dominant colors, {m.saturated_hues} strongly saturated.")
    if m.flags:
        lines.append("Automated flags: " + " | ".join(m.flags))
    return "\n".join(lines)


def generate_queries(client, title, text, topics) -> list[str]:
    prompt = (
        "You are helping find related prior work for a research poster. "
        "Generate 4 arXiv search queries at varying specificity to surface: the same problem, "
        "competing methods/baselines, and related techniques. Keep each query 3-8 words.\n\n"
        f"Title: {title}\nTopics: {', '.join(topics) if topics else 'n/a'}\n\n"
        f"Poster text (excerpt):\n{text[:2500]}"
    )
    resp = client.messages.parse(
        model=MODEL, max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
        output_format=Queries,
    )
    return (resp.parsed_output.queries if resp.parsed_output else [])[:4]


SYSTEM = """You are an expert reviewer of academic research posters, in the spirit of a \
top-conference reviewer who also has a designer's eye. You give specific, constructive, \
honest feedback — not praise. You are given three things: (1) an image of the poster, \
(2) MEASURED design metrics extracted deterministically from the poster file (font sizes, \
contrast, text coverage, columns — treat these as ground truth and cite them by number), \
and (3) retrieved related work from arXiv to judge novelty and contextualization.

Score each dimension 1-5 (1 poor, 5 excellent; 5 is rare). Ground every design claim in the \
measured metrics rather than guessing. For contextualization, refer to the specific related \
papers. Be concrete in suggestions — an author should know exactly what to change."""


def build_review(pdf_path: str, use_arxiv: bool = True) -> dict:
    import anthropic
    client = anthropic.Anthropic()

    doc = fitz.open(pdf_path)
    page = doc[0]
    m = poster_metrics.analyze_poster(pdf_path)
    title, text = extract_title_and_text(page, m)
    img_b64 = render_image_b64(page)

    related = []
    if use_arxiv:
        print("  generating search queries...", file=sys.stderr)
        queries = generate_queries(client, title, text, [])
        print(f"  queries: {queries}", file=sys.stderr)
        print("  retrieving related work from arXiv...", file=sys.stderr)
        related = arxiv_retrieval.retrieve_related(queries) if queries else []
        print(f"  {len(related)} related papers", file=sys.stderr)

    related_block = "\n".join(
        f"- [{p['year']}] {p['title']} ({p['arxiv_id']}): {p['abstract'][:300]}"
        for p in related
    ) or "(no related work retrieved)"

    dims_spec = "\n".join(f"- {n} [{cat}]: {desc}" for n, cat, desc in DIMENSIONS)
    user_text = (
        f"Review this poster across exactly these dimensions:\n{dims_spec}\n\n"
        f"=== MEASURED DESIGN METRICS (ground truth) ===\n{metrics_summary(m)}\n\n"
        f"=== POSTER TEXT (extracted) ===\n{text}\n\n"
        f"=== RETRIEVED RELATED WORK (arXiv) ===\n{related_block}"
    )

    print("  generating review...", file=sys.stderr)
    resp = client.messages.parse(
        model=MODEL, max_tokens=12000,
        thinking={"type": "adaptive"},
        system=SYSTEM,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
            {"type": "text", "text": user_text},
        ]}],
        output_format=PosterReview,
    )
    review = resp.parsed_output

    scores = [d.score for d in review.dimensions] if review else []
    provisional = round(sum(scores) / len(scores), 2) if scores else None
    return {
        "poster": pdf_path,
        "metrics": dataclasses.asdict(m),
        "related_work": related,
        "review": review.model_dump() if review else None,
        # NOTE: provisional = unweighted mean of LLM dimension scores. The trained
        # scoring head (regression on PosterSum->OpenReview labels) is Phase 4.
        "provisional_score": provisional,
    }


def print_review(out: dict):
    r = out["review"]
    if not r:
        print("No review produced."); return
    print(f"\n{'='*70}\n{out['poster']}\n{'='*70}")
    print(f"\n▶ {r['one_line_summary']}")
    print(f"\nProvisional score: {out['provisional_score']}/5  "
          f"(unweighted mean of dimensions — trained scoring head is Phase 4)")
    print("\nDimensions:")
    for d in r["dimensions"]:
        print(f"  [{d['score']}/5] {d['name']} ({d['category']})")
        print(f"        {d['rationale']}")
        for s in d["suggestions"]:
            print(f"        → {s}")
    print("\nTop strengths:")
    for s in r["top_strengths"]:
        print(f"  + {s}")
    print("\nTop weaknesses:")
    for w in r["top_weaknesses"]:
        print(f"  - {w}")
    print("\nGrounded design notes (tied to measured metrics):")
    for n in r["grounded_design_notes"]:
        print(f"  • {n}")
    if out["related_work"]:
        print(f"\nGrounded against {len(out['related_work'])} arXiv papers.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-arxiv", action="store_true")
    a = ap.parse_args()
    out = build_review(a.pdf, use_arxiv=not a.no_arxiv)
    if a.json:
        print(json.dumps(out, indent=2, default=str))
    else:
        print_review(out)


if __name__ == "__main__":
    main()
