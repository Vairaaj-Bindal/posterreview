"""Deterministic design-metrics engine for research posters.

The moat: paperreview.ai grounds *content* claims in retrieved prior work; we
additionally ground *design* claims in measured numbers. Given a poster PDF this
computes exact font sizes, text/figure/whitespace coverage, column structure,
WCAG contrast, and palette size — facts a review can cite instead of vibes.

Two extraction regimes, auto-detected:
  - "vector": real text spans in the PDF -> exact font/color/bbox (deterministic)
  - "image":  flattened/image poster -> render-based metrics only; text needs OCR
Render-based metrics (coverage via masks, contrast, palette) work for BOTH.

Usage:  python poster_metrics.py <poster.pdf> [--json]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, field, asdict

import fitz  # PyMuPDF
import numpy as np

_WORD = re.compile(r"[A-Za-z]{4,}")

# OCR box height overshoots the true point size slightly (calibrated on vector
# posters with known font sizes: 36px-derived vs 34pt true).
OCR_PT_FACTOR = 0.95
OCR_RENDER_PX = 3000      # long-edge px for the OCR render (~higher DPI than metrics render)
OCR_MIN_CONF = 0.5

_OCR_ENGINE = None        # lazy singleton — model load is expensive


def _wordy(t: str) -> bool:
    return bool(_WORD.search(t))


def _get_ocr():
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR  # optional dep; only for image posters
        _OCR_ENGINE = RapidOCR()
    return _OCR_ENGINE

# Legibility thresholds, expressed in points on a poster whose long edge is
# normalized to 48 inches (a standard large-format poster). Orientation- and
# authoring-scale-invariant. Rule-of-thumb large-format poster guidance.
NORM_LONG_EDGE_IN = 48.0
BODY_MIN_PT = 24.0     # body text readable from ~4-6 ft
HEADING_MIN_PT = 36.0
TITLE_MIN_PT = 60.0
WCAG_AA = 4.5          # contrast floor for normal text
RENDER_LONG_EDGE_PX = 1600

STD_SIZES_IN = {  # (w, h) inches -> label, for print-size guess (portrait/landscape agnostic)
    "A0": (33.1, 46.8), "A1": (23.4, 33.1),
    "48x36": (48, 36), "42x36": (42, 36), "36x24": (36, 24), "24x18": (24, 18),
}


@dataclass
class Span:
    text: str
    size_pt: float          # raw pt in the PDF
    norm_pt: float          # pt normalized to a 48in long-edge poster
    font: str
    bold: bool
    color: tuple            # (r,g,b) 0-255 foreground
    bbox: tuple             # (x0,y0,x1,y1) in pt


@dataclass
class PosterMetrics:
    regime: str = "vector"
    text_source: str = "vector"     # vector | ocr | none
    page_w_pt: float = 0.0
    page_h_pt: float = 0.0
    print_w_in: float = 0.0
    print_h_in: float = 0.0
    aspect: float = 0.0
    orientation: str = ""
    nearest_std_size: str = ""
    scale_to_48in: float = 1.0

    n_text_spans: int = 0
    n_chars: int = 0
    n_words: int = 0
    n_text_blocks: int = 0
    n_figures: int = 0

    body_pt: float = 0.0            # normalized pt of dominant body text
    title_pt: float = 0.0
    heading_pt: float = 0.0
    font_size_hist: dict = field(default_factory=dict)  # norm_pt(rounded) -> chars
    pct_body_below_floor: float = 0.0  # share of body chars below BODY_MIN_PT

    text_coverage: float = 0.0      # fraction of poster area under text bboxes
    visual_density: float = 0.0     # non-background area (text + figures + decoration)
    whitespace: float = 0.0         # background/open area

    n_columns: int = 0

    n_fonts: int = 0
    bg_color: tuple = (255, 255, 255)
    n_palette_colors: int = 0
    saturated_hues: int = 0

    median_contrast: float = 0.0
    min_contrast: float = 0.0
    pct_text_below_wcag: float = 0.0

    flags: list = field(default_factory=list)


# ---------- helpers ----------
def _int_to_rgb(c: int) -> tuple:
    return ((c >> 16) & 255, (c >> 8) & 255, c & 255)


def _rel_luminance(rgb) -> float:
    def lin(v):
        v /= 255.0
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = rgb
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def _contrast(fg, bg) -> float:
    l1, l2 = _rel_luminance(fg), _rel_luminance(bg)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def _nearest_std(w_in, h_in) -> str:
    long_e, short_e = max(w_in, h_in), min(w_in, h_in)
    best, bestd = "", 1e9
    for name, (a, b) in STD_SIZES_IN.items():
        L, S = max(a, b), min(a, b)
        d = abs(L - long_e) + abs(S - short_e)
        if d < bestd:
            bestd, best = d, name
    return best if bestd < 8 else "non-standard"


# ---------- extraction ----------
def _extract_spans(page, scale_to_48in) -> list[Span]:
    """Extract text spans, skipping artifacts (rotated watermarks, giant bboxes).

    A single line of poster text never spans a large fraction of the page. Spans
    whose bbox is oversized (rotated Creative-Commons watermarks report an
    axis-aligned bbox as tall as the whole page) are dropped so they don't
    pollute coverage, columns, contrast, or font stats.
    """
    r = page.rect
    page_area = r.width * r.height
    spans = []
    d = page.get_text("dict")
    for b in d["blocks"]:
        if b.get("type") != 0:
            continue
        for line in b["lines"]:
            for s in line["spans"]:
                txt = s["text"]
                if not txt.strip():
                    continue
                bb = s["bbox"]
                bw, bh = bb[2] - bb[0], bb[3] - bb[1]
                if bh > 0.3 * r.height or bw * bh > 0.04 * page_area:
                    continue  # rotated/watermark/artifact span
                size = s["size"]
                font = s.get("font", "")
                spans.append(Span(
                    text=txt, size_pt=size, norm_pt=size * scale_to_48in,
                    font=font, bold=bool(s.get("flags", 0) & 2**4) or "bold" in font.lower(),
                    color=_int_to_rgb(s.get("color", 0)), bbox=tuple(s["bbox"]),
                ))
    return spans


def _ocr_spans(page, scale_to_48in):
    """Build Spans for an image-flattened poster via OCR.

    Renders at higher DPI, runs RapidOCR, and produces the same Span shape as the
    vector path so all downstream classification is identical. Font size comes
    from the OCR box height (calibrated); text color is sampled from the render.
    """
    r = page.rect
    scale = OCR_RENDER_PX / max(r.width, r.height)   # px per pt
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
    result, _ = _get_ocr()(img)
    spans = []
    for box, text, conf in (result or []):
        if conf < OCR_MIN_CONF or not text.strip():
            continue
        xs = [pt[0] for pt in box]
        ys = [pt[1] for pt in box]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        h_px = ((y1 - y0) + (abs(box[3][1] - box[0][1]) + abs(box[2][1] - box[1][1])) / 2) / 2
        size_pt = (h_px / scale) * OCR_PT_FACTOR
        if (y1 - y0) / scale > 0.3 * r.height:
            continue  # oversized/merged artifact box
        # sample text color: pixels in the box furthest from the box's dominant (bg) color
        patch = img[int(y0):int(y1), int(x0):int(x1)].reshape(-1, 3).astype(float)
        if len(patch) < 4:
            continue
        bg = np.median(patch, axis=0)
        far = patch[np.abs(patch - bg).max(axis=1) > 40]
        fg = np.median(far, axis=0) if len(far) > 2 else np.array([0, 0, 0])
        bbox_pt = (x0 / scale, y0 / scale, x1 / scale, y1 / scale)
        spans.append(Span(
            text=text, size_pt=size_pt, norm_pt=size_pt * scale_to_48in,
            font="", bold=False, color=tuple(int(v) for v in fg), bbox=bbox_pt,
        ))
    return spans


def _classify_fonts(spans, out: PosterMetrics):
    """Body/heading/title from *readable, wordy* text.

    Chart axis labels (numeric), hidden fine-print, and references pollute a
    naive char-weighted mode. We restrict the body estimate to wordy spans at or
    above the legibility floor; if none clear it, the whole poster is tiny and we
    fall back to the overall dominant (and a flag is raised elsewhere).
    """
    if not spans:
        return
    wordy = [s for s in spans if _wordy(s.text)]
    if not wordy:
        wordy = spans
    hist = Counter()
    for s in wordy:
        hist[round(s.norm_pt)] += len(s.text.strip())
    out.font_size_hist = dict(sorted(hist.items()))

    readable = Counter({sz: c for sz, c in hist.items() if sz >= BODY_MIN_PT})
    if readable and max(readable.values()) >= 100:
        out.body_pt = float(max(readable.items(), key=lambda kv: kv[1])[0])
    else:
        out.body_pt = float(max(hist.items(), key=lambda kv: kv[1])[0])

    # title = largest size carrying a short phrase; heading = largest sustained
    # size strictly between body and title.
    sizes_with_phrase = [round(s.norm_pt) for s in wordy if len(s.text.strip()) >= 6]
    out.title_pt = float(max(sizes_with_phrase)) if sizes_with_phrase else out.body_pt
    mids = [sz for sz, c in hist.items() if out.body_pt < sz < out.title_pt and c >= 30]
    out.heading_pt = float(max(mids)) if mids else out.title_pt

    total = sum(hist.values()) or 1
    below = sum(c for sz, c in hist.items() if sz < BODY_MIN_PT)
    out.pct_body_below_floor = round(100 * below / total, 1)


def _readable_wordy(spans):
    return [s for s in spans if _wordy(s.text) and s.norm_pt >= BODY_MIN_PT]


def _text_coverage(page, spans):
    """Fraction of poster area under text bboxes (coarse union mask)."""
    r = page.rect
    GW = 400
    GH = max(1, int(GW * r.height / r.width))
    sx, sy = GW / r.width, GH / r.height
    mask = np.zeros((GH, GW), bool)
    for s in spans:
        x0, y0, x1, y1 = s.bbox
        ix0, iy0 = max(0, int(x0 * sx)), max(0, int(y0 * sy))
        ix1, iy1 = min(GW, int(x1 * sx) + 1), min(GH, int(y1 * sy) + 1)
        if ix1 > ix0 and iy1 > iy0:
            mask[iy0:iy1, ix0:ix1] = True
    return round(float(mask.mean()), 4)


def _detect_bg(img):
    """Dominant background color (coarse-quantized mode). Handles colored posters."""
    sample = (img[::3, ::3].reshape(-1, 3) // 16 * 16)
    return np.array(Counter(map(tuple, sample)).most_common(1)[0][0], int)


def _whitespace(img, bg):
    """Openness = fraction of pixels matching the background color (any hue)."""
    matches = np.abs(img.astype(int) - bg).max(axis=2) < 28
    return round(float(matches.mean()), 4)


def _column_count_visual(img, bg) -> int:
    """Columns from vertical gutters: x-strips that stay background down the body.

    Pixel-based, so it works for both vector and image posters. Returns the
    number of content bands separated by full-height background gutters; 1 for
    single-column or irregular/mixed-grid layouts (no consistent gutter).
    """
    H, W = img.shape[:2]
    isbg = np.abs(img.astype(int) - bg).max(axis=2) < 28
    body = isbg[int(0.15 * H):, :]           # skip title banner (usually full width)
    col_bg = body.mean(axis=0)
    gutter = col_bg > 0.93
    runs, start = [], None
    for i, g in enumerate(gutter):
        if not g and start is None:
            start = i
        elif g and start is not None:
            runs.append((start, i)); start = None
    if start is not None:
        runs.append((start, W))
    cols = [(s, e) for s, e in runs if (e - s) / W >= 0.08]
    return max(len(cols), 1)


def _count_figures(page):
    """Raster figures = non-background image blocks (area < 40% of page)."""
    r = page.rect
    A = r.width * r.height
    n = 0
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") == 1:
            bb = b["bbox"]
            if (bb[2] - bb[0]) * (bb[3] - bb[1]) < 0.4 * A:
                n += 1
    return n




def _render(page):
    scale = RENDER_LONG_EDGE_PX / max(page.rect.width, page.rect.height)
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    return arr[:, :, :3], scale


def _palette_stats(img):
    small = img[::4, ::4].reshape(-1, 3)
    q = (small // 32) * 32  # coarse quantize
    counts = Counter(map(tuple, q))
    dom = [c for c, n in counts.items() if n > 0.01 * len(small)]
    sat = 0
    for c in dom:
        mx, mn = max(c), min(c)
        s = 0 if mx == 0 else (mx - mn) / mx
        if s > 0.4 and mx > 60:
            sat += 1
    return len(dom), sat


def _contrast_stats(img, spans, scale):
    """Contrast of *readable* text: fg=span color, bg=median of pixels behind it.
    Restricted to readable wordy spans so fine-print/label noise doesn't dominate.
    """
    ratios, weights = [], []
    H, W = img.shape[:2]
    for s in spans:
        x0, y0, x1, y1 = [int(v * scale) for v in s.bbox]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        patch = img[y0:y1, x0:x1].reshape(-1, 3).astype(float)
        fg = np.array(s.color, float)
        dist = np.linalg.norm(patch - fg, axis=1)
        bg_pixels = patch[dist > 60]  # pixels unlike the text = background
        bg = np.median(bg_pixels, axis=0) if len(bg_pixels) > 4 else np.median(patch, axis=0)
        ratios.append(_contrast(s.color, tuple(bg)))
        weights.append(len(s.text.strip()))
    if not ratios:
        return 0.0, 0.0, 0.0
    ratios = np.array(ratios); weights = np.array(weights)
    med = float(np.median(ratios))
    mn = float(ratios.min())
    below = float(100 * weights[ratios < WCAG_AA].sum() / max(weights.sum(), 1))
    return round(med, 2), round(mn, 2), round(below, 1)


# ---------- main entry ----------
def analyze_poster(pdf_path: str) -> PosterMetrics:
    doc = fitz.open(pdf_path)
    page = doc[0]
    r = page.rect
    out = PosterMetrics()
    out.page_w_pt, out.page_h_pt = round(r.width, 1), round(r.height, 1)
    out.print_w_in, out.print_h_in = round(r.width / 72, 1), round(r.height / 72, 1)
    out.aspect = round(r.width / r.height, 2)
    out.orientation = "landscape" if r.width >= r.height else "portrait"
    out.nearest_std_size = _nearest_std(out.print_w_in, out.print_h_in)
    long_edge_in = max(out.print_w_in, out.print_h_in)
    out.scale_to_48in = round(NORM_LONG_EDGE_IN / long_edge_in, 3) if long_edge_in else 1.0

    spans = _extract_spans(page, out.scale_to_48in)
    vector_chars = sum(len(s.text.strip()) for s in spans)
    out.regime = "vector" if vector_chars > 200 else "image"
    if out.regime == "image":
        try:
            spans = _ocr_spans(page, out.scale_to_48in)
            out.text_source = "ocr"
        except ImportError:
            out.text_source = "none"   # rapidocr not installed; render metrics only
    else:
        out.text_source = "vector"

    out.n_text_spans = len(spans)
    out.n_chars = sum(len(s.text.strip()) for s in spans)
    out.n_words = sum(len(s.text.split()) for s in spans)
    out.n_text_blocks = len(page.get_text("blocks"))
    out.n_fonts = len({s.font for s in spans if s.font})

    if spans:
        _classify_fonts(spans, out)
    readable = _readable_wordy(spans)

    img, scale = _render(page)
    bg = _detect_bg(img)
    out.bg_color = tuple(int(v) for v in bg)
    out.text_coverage = _text_coverage(page, spans)
    out.whitespace = _whitespace(img, bg)
    # everything that isn't background = text + figures + decoration
    out.visual_density = round(1 - out.whitespace, 4)
    # raster-figure count is only meaningful for vector PDFs; a flattened poster
    # is one big image (or many tiles) and can't be segmented into figures here.
    out.n_figures = _count_figures(page) if out.regime == "vector" else -1
    out.n_columns = _column_count_visual(img, bg)

    out.n_palette_colors, out.saturated_hues = _palette_stats(img)
    if readable:
        out.median_contrast, out.min_contrast, out.pct_text_below_wcag = _contrast_stats(img, readable, scale)

    _add_flags(out)
    return out


def _add_flags(out: PosterMetrics):
    f = out.flags
    if out.regime == "image":
        if out.text_source == "ocr":
            f.append("IMAGE-BASED PDF: text was recovered via OCR; font sizes are "
                     "estimated from OCR box heights (±a few pt).")
        else:
            f.append("IMAGE-BASED PDF: text is flattened and OCR is unavailable; "
                     "coverage/contrast/palette still valid, font metrics are not.")
    if out.text_source in ("vector", "ocr"):
        if out.body_pt and out.body_pt < BODY_MIN_PT:
            f.append(f"Body text ~{out.body_pt:.0f}pt (48in-normalized) is below the "
                     f"{BODY_MIN_PT:.0f}pt legibility floor for viewing from a few feet.")
        if out.title_pt and out.title_pt < TITLE_MIN_PT:
            f.append(f"Title ~{out.title_pt:.0f}pt is small for a poster title "
                     f"(aim >= {TITLE_MIN_PT:.0f}pt normalized).")
        if out.pct_body_below_floor > 25:
            f.append(f"{out.pct_body_below_floor:.0f}% of the text is fine print below the "
                     f"legibility floor (references, embedded labels, or hidden text).")
    if out.text_coverage > 0.42:
        f.append(f"Text covers {out.text_coverage*100:.0f}% of the poster — dense; "
                 f"posters read best under ~40% text coverage.")
    if out.whitespace < 0.15:
        f.append(f"Only {out.whitespace*100:.0f}% of the poster is open/background space "
                 f"— little breathing room.")
    if out.n_columns > 4:
        f.append(f"{out.n_columns} columns detected — many narrow columns can fracture the reading path.")
    if out.pct_text_below_wcag > 10:
        f.append(f"{out.pct_text_below_wcag:.0f}% of text is below WCAG AA contrast ({WCAG_AA}:1).")
    if out.saturated_hues > 5:
        f.append(f"{out.saturated_hues} strongly saturated colors — palette may read as busy.")
    if out.nearest_std_size == "non-standard":
        f.append(f"Poster is {out.print_w_in}x{out.print_h_in}in — not a standard print size.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    m = analyze_poster(a.pdf)
    d = asdict(m)
    if a.json:
        print(json.dumps(d, indent=2))
        return
    print(f"\n=== {a.pdf} ===")
    print(f"regime: {m.regime} | {m.print_w_in}x{m.print_h_in}in {m.orientation} "
          f"(~{m.nearest_std_size}, aspect {m.aspect})")
    figs = "n/a" if m.n_figures < 0 else str(m.n_figures)
    print(f"text: {m.n_words} words / {m.n_chars} chars in {m.n_text_spans} spans "
          f"(source: {m.text_source}), {figs} figures, {m.n_columns} columns")
    if m.text_source in ("vector", "ocr"):
        est = " (OCR-estimated)" if m.text_source == "ocr" else ""
        print(f"fonts (48in-norm pt){est}: body ~{m.body_pt:.0f} | heading ~{m.heading_pt:.0f} | title ~{m.title_pt:.0f}")
    figline = f" | {m.n_figures} raster figures" if m.n_figures >= 0 else ""
    print(f"layout: text blocks cover {m.text_coverage*100:.0f}% of area | "
          f"{m.whitespace*100:.0f}% is background color {m.bg_color}{figline}")
    print(f"contrast: median {m.median_contrast}:1 | min {m.min_contrast}:1 | {m.pct_text_below_wcag}% below WCAG AA")
    print(f"palette: {m.n_palette_colors} dominant colors, {m.saturated_hues} strongly saturated")
    if m.flags:
        print("\nflags:")
        for fl in m.flags:
            print(f"  • {fl}")


if __name__ == "__main__":
    main()
