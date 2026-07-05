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

try:  # pydantic is only needed for the optional --backend anthropic path
    from pydantic import BaseModel, Field
    _HAS_PYDANTIC = True
except ImportError:
    _HAS_PYDANTIC = False

import poster_metrics
import arxiv_retrieval
import design_score as design_score_mod

MODEL = "claude-opus-4-8"
IMAGE_LONG_EDGE_PX = 1600  # under the 2576px high-res cap; enough for legibility

# The LLM scores only CONTENT/messaging dimensions — things judgeable from the poster's
# text. DESIGN (legibility, contrast, density, structure, palette) is owned by the
# deterministic design score (Phase 4c), so the LLM never re-judges (or contradicts) the
# measured visual facts. This is the Phase-4 split: rubric for design, LLM for content.
DIMENSIONS = [
    ("importance", "content", "Importance of the research question / problem."),
    ("claim_support", "content", "Are the claims adequately supported by the results described?"),
    ("contextualization", "content", "Positioning vs prior work / novelty, given the retrieved related papers."),
    ("takeaway_clarity", "communication", "Is there a clear main message/finding a passerby could grab quickly?"),
    ("self_containedness", "communication", "Does the text stand alone, or does it seem to rely on the author narrating?"),
]


if _HAS_PYDANTIC:  # schema classes used only by the Anthropic structured-output path
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
            description="Specific claims tied to the measured metrics (e.g. 'body ~18pt below legibility floor')"
        )


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


_STOP = set("the a an of for and to in on with using via from into over under this that "
            "we our their its is are be can via toward towards based novel new approach "
            "method model results using paper poster study analysis via".split())


def heuristic_queries(title, text) -> list[str]:
    """LLM-free arXiv queries: the title, plus salient multiword terms from the text.
    Keeps retrieval zero-cost and deterministic."""
    import re as _re
    from collections import Counter
    queries = []
    t = " ".join(title.split())
    if t and t != "(untitled)":
        queries.append(t[:120])
    # salient terms: frequent capitalized phrases + frequent long words
    words = _re.findall(r"[A-Za-z][A-Za-z\-]{3,}", text)
    freq = Counter(w.lower() for w in words if w.lower() not in _STOP)
    top = [w for w, _ in freq.most_common(8)]
    if len(top) >= 4:
        queries.append(" ".join(top[:4]))
        queries.append(" ".join(top[2:6]))
    # capitalized multiword phrases (likely method/dataset names)
    phrases = _re.findall(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,3})\b", text)
    ph = [p for p in phrases if p.lower() not in {title.lower()}][:2]
    queries.extend(ph)
    # dedup, cap
    seen, out = set(), []
    for q in queries:
        k = q.lower().strip()
        if k and k not in seen:
            seen.add(k); out.append(q)
    return out[:4]


SYSTEM = """You are an expert reviewer of academic research posters — a top-conference \
reviewer giving specific, constructive, honest feedback, not praise. You are given: (1) the \
poster's extracted text, (2) retrieved related work from arXiv, and (3) MEASURED design metrics \
for context. Score ONLY the research-content and messaging dimensions asked for. \

Do NOT judge or comment on visual design, readability, contrast, font sizes, or layout — those \
are measured separately and objectively; anything you'd say about them would be a guess. If the \
measured metrics contradict an instinct (e.g. you assume a dense poster is unreadable but the \
measurements say contrast and font sizes are fine), defer to the measurements. \

Score each dimension 1-5 (1 poor, 5 excellent; 5 is rare). Be concrete about the research and \
messaging — an author should know exactly what to change."""

_CATEGORY = {n: c for n, c, _ in DIMENSIONS}
_DIM_NAMES = [n for n, _, _ in DIMENSIONS]


def _review_inputs(pdf_path, use_arxiv):
    doc = fitz.open(pdf_path)
    page = doc[0]
    m = poster_metrics.analyze_poster(pdf_path)
    title, text = extract_title_and_text(page, m)
    related = []
    if use_arxiv:
        print("  search queries (heuristic)...", file=sys.stderr)
        queries = heuristic_queries(title, text)
        print(f"  queries: {queries}", file=sys.stderr)
        related = arxiv_retrieval.retrieve_related(queries) if queries else []
        print(f"  {len(related)} related papers", file=sys.stderr)
    related_block = "\n".join(
        f"- [{p['year']}] {p['title']} ({p['arxiv_id']}): {p['abstract'][:300]}" for p in related
    ) or "(no related work retrieved)"
    dims_spec = "\n".join(f"- {n} [{cat}]: {desc}" for n, cat, desc in DIMENSIONS)
    user_text = (
        f"Review this poster across exactly these dimensions:\n{dims_spec}\n\n"
        f"=== MEASURED DESIGN METRICS (ground truth) ===\n{metrics_summary(m)}\n\n"
        f"=== POSTER TEXT (extracted) ===\n{text[:5000]}\n\n"
        f"=== RETRIEVED RELATED WORK (arXiv) ===\n{related_block}"
    )
    return m, page, related, user_text


def _normalize(raw: dict) -> dict | None:
    """Coerce a local model's JSON into the standard review shape."""
    if not raw:
        return None
    dims_in = raw.get("dimensions", {})
    dimensions = []
    for name in _DIM_NAMES:
        d = dims_in.get(name, {}) if isinstance(dims_in, dict) else {}
        try:
            score = int(round(float(d.get("score", 3))))
        except (TypeError, ValueError):
            score = 3
        sugg = d.get("suggestions") or ([d["suggestion"]] if d.get("suggestion") else [])
        dimensions.append({
            "name": name, "category": _CATEGORY[name],
            "score": max(1, min(5, score)),
            "rationale": str(d.get("rationale", "")).strip(),
            "suggestions": [str(s) for s in (sugg if isinstance(sugg, list) else [sugg]) if s],
        })
    aslist = lambda v: [str(x) for x in v] if isinstance(v, list) else ([str(v)] if v else [])
    return {
        "one_line_summary": str(raw.get("one_line_summary", "")).strip(),
        "dimensions": dimensions,
        "top_strengths": aslist(raw.get("top_strengths")),
        "top_weaknesses": aslist(raw.get("top_weaknesses")),
        "grounded_design_notes": aslist(raw.get("grounded_design_notes")),
    }


def _review_local(user_text: str) -> dict | None:
    import local_llm
    print(f"  generating review (local: {local_llm.engine_info()})...", file=sys.stderr)
    schema = (
        '{\n'
        '  "one_line_summary": "string",\n'
        '  "dimensions": {\n'
        + ",\n".join(f'    "{n}": {{"score": 1-5, "rationale": "string", "suggestion": "string"}}'
                     for n in _DIM_NAMES)
        + '\n  },\n'
        '  "top_strengths": ["string"],\n'
        '  "top_weaknesses": ["string"],\n'
        '  "grounded_design_notes": ["string (cite a measured number)"]\n'
        '}'
    )
    prompt = (user_text + "\n\n"
              "Respond with ONLY this JSON object (no prose, no code fences):\n" + schema)
    return _normalize(local_llm.generate_json(prompt, system=SYSTEM, max_tokens=3500))


def _review_anthropic(page, user_text: str) -> dict | None:
    import anthropic
    client = anthropic.Anthropic()
    print("  generating review (anthropic: claude-opus-4-8)...", file=sys.stderr)
    img_b64 = render_image_b64(page)
    resp = client.messages.parse(
        model=MODEL, max_tokens=12000, thinking={"type": "adaptive"}, system=SYSTEM,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
            {"type": "text", "text": user_text},
        ]}],
        output_format=PosterReview,
    )
    return resp.parsed_output.model_dump() if resp.parsed_output else None


def build_review(pdf_path: str, backend: str = "local", use_arxiv: bool = True) -> dict:
    m, page, related, user_text = _review_inputs(pdf_path, use_arxiv)
    review = _review_anthropic(page, user_text) if backend == "anthropic" else _review_local(user_text)
    scores = [d["score"] for d in review["dimensions"]] if review else []
    provisional = round(sum(scores) / len(scores), 2) if scores else None
    return {
        "poster": pdf_path,
        "backend": backend,
        "metrics": dataclasses.asdict(m),
        # Deterministic, key-free design score (Phase 4c) — validated by controlled
        # degradation, NOT trained on OpenReview (which measures the paper, not the poster).
        "design_score": design_score_mod.design_score(m),
        "related_work": related,
        "review": review,
        # provisional = unweighted mean of LLM dimension scores (trained content head = Phase 4b)
        "provisional_score": provisional,
    }


def print_review(out: dict):
    r = out["review"]
    if not r:
        print("No review produced."); return
    print(f"\n{'='*70}\n{out['poster']}\n{'='*70}")
    print(f"\n▶ {r['one_line_summary']}")
    ds = out.get("design_score")
    if ds:
        print(f"\nDesign score (deterministic, key-free): {ds['overall']}/100")
        for k, v in ds["subscores"].items():
            print(f"    {k:16s} {v:5.1f}  — {ds['notes'][k]}")
    print(f"\nContent/communication (LLM) provisional score: {out['provisional_score']}/5  "
          f"(unweighted mean of dimensions — trained content head is Phase 4b)")
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
    ap.add_argument("--backend", choices=["local", "anthropic"], default="local",
                    help="local = open model via MLX/Ollama, zero cost (default); anthropic = needs key")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-arxiv", action="store_true")
    a = ap.parse_args()
    out = build_review(a.pdf, backend=a.backend, use_arxiv=not a.no_arxiv)
    if a.json:
        print(json.dumps(out, indent=2, default=str))
    else:
        print_review(out)


if __name__ == "__main__":
    main()
