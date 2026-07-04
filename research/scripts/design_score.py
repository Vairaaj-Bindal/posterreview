"""Transparent design-score rubric for research posters (Phase 4c).

Phase 4 showed OpenReview ratings can't supply a *design* label (they score the
paper, not the poster). And no key-free, poster-native quality-label dataset
exists for scientific posters. So the design score is a transparent, interpretable
rubric over the deterministic metrics — anchored to established large-format
poster guidelines — NOT a black-box learned model. Every sub-score is a defensible
function of a measured quantity, so a reviewer can see exactly why a poster scored
what it did. Validated separately by controlled degradation (validate_design_score.py).

design_score(PosterMetrics) -> {overall, subscores{...}, notes[...]}
Scores are 0-100, higher = better.
"""
from __future__ import annotations

import poster_metrics


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _tent(x, peak, half_width):
    """1.0 at peak, linearly to 0 at peak±half_width."""
    return max(0.0, 1.0 - abs(x - peak) / half_width)


# sub-score weights (sum to 1). Legibility + contrast are the biggest poster sins.
WEIGHTS = {
    "legibility": 0.30,
    "accessibility": 0.25,
    "density_balance": 0.20,
    "structure": 0.15,
    "palette": 0.10,
}


def _legibility(m) -> tuple[float, str]:
    if m.text_source not in ("vector", "ocr") or not m.body_pt:
        return 60.0, "font sizes unavailable (image poster, no OCR) — neutral 60"
    body_ok = _clamp((m.body_pt - 16) / (poster_metrics.BODY_MIN_PT - 16))     # 16pt→0, 24pt→1
    title_ok = _clamp((m.title_pt - 40) / (poster_metrics.TITLE_MIN_PT - 40))  # 40pt→0, 60pt→1
    fineprint_pen = _clamp(m.pct_body_below_floor / 60)                        # 60%→full penalty
    s = 100 * (0.5 * body_ok + 0.25 * title_ok + 0.25 * (1 - fineprint_pen))
    est = " (OCR-estimated)" if m.text_source == "ocr" else ""
    return s, f"body ~{m.body_pt:.0f}pt, title ~{m.title_pt:.0f}pt{est}, {m.pct_body_below_floor:.0f}% fine print"


def _accessibility(m) -> tuple[float, str]:
    if m.median_contrast <= 0:
        return 60.0, "no measurable text contrast — neutral 60"
    con = _clamp((m.median_contrast - 3) / (7 - 3))            # 3:1→0, 7:1(AAA)→1
    below_pen = _clamp(m.pct_text_below_wcag / 25)
    s = 100 * (0.7 * con + 0.3 * (1 - below_pen))
    return s, f"median contrast {m.median_contrast}:1, {m.pct_text_below_wcag:.0f}% below WCAG AA"


def _density_balance(m) -> tuple[float, str]:
    dens = _tent(m.text_coverage, peak=0.30, half_width=0.30)   # ideal ~30% text
    ws = _tent(m.whitespace, peak=0.45, half_width=0.45)        # ideal ~45% open
    s = 100 * (0.6 * dens + 0.4 * ws)
    return s, f"text covers {m.text_coverage*100:.0f}%, {m.whitespace*100:.0f}% open space"


def _structure(m) -> tuple[float, str]:
    nc = m.n_columns
    if 2 <= nc <= 4:
        s = 100.0
    elif nc == 1:
        s = 60.0
    else:
        s = 40.0
    return s, f"{nc} column(s) detected"


def _palette(m) -> tuple[float, str]:
    sat_pen = _clamp((m.saturated_hues - 3) / 5)
    pal_pen = _clamp((m.n_palette_colors - 12) / 12)
    s = 100 * _clamp(1 - 0.7 * sat_pen - 0.3 * pal_pen)
    return s, f"{m.n_palette_colors} dominant colors, {m.saturated_hues} strongly saturated"


def design_score(m) -> dict:
    parts = {
        "legibility": _legibility(m),
        "accessibility": _accessibility(m),
        "density_balance": _density_balance(m),
        "structure": _structure(m),
        "palette": _palette(m),
    }
    subscores = {k: round(parts[k][0], 1) for k in parts}
    notes = {k: parts[k][1] for k in parts}
    overall = round(sum(subscores[k] * WEIGHTS[k] for k in WEIGHTS), 1)
    return {"overall": overall, "subscores": subscores, "notes": notes}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    a = ap.parse_args()
    m = poster_metrics.analyze_poster(a.pdf)
    d = design_score(m)
    print(f"\n=== Design score: {d['overall']}/100 ===  (source: {m.text_source})")
    for k in WEIGHTS:
        print(f"  {k:16s} {d['subscores'][k]:5.1f}  (w={WEIGHTS[k]:.0%})  — {d['notes'][k]}")


if __name__ == "__main__":
    main()
