"""Validate the design score by controlled degradation (key-free, non-circular).

There's no poster-native quality-label dataset to regress against, so we test the
score the way you'd test any quality metric without labels: perturb real posters
in KNOWN-BAD directions, re-extract metrics from the perturbed pixels, and check
the score drops. If lowering contrast / cluttering the palette / removing
whitespace / blurring doesn't lower the score, the score is meaningless. If it
reliably does, the rubric is capturing real design quality.

Non-circular: perturbations are applied to the rendered IMAGE, then the FULL
metrics engine re-measures the perturbed pixels — the score never sees the
perturbation directly. Baseline and degraded both go through the image/OCR path,
so the only difference is the perturbation.

Usage:  python validate_design_score.py
"""
from __future__ import annotations

import glob
import tempfile

import fitz
import numpy as np
from PIL import Image, ImageFilter

import poster_metrics
import design_score

RENDER_PX = 2000


def render_rgb(pdf_path) -> np.ndarray:
    doc = fitz.open(pdf_path)
    p = doc[0]
    scale = RENDER_PX / max(p.rect.width, p.rect.height)
    pix = p.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].copy()


def rgb_to_pdf(img: np.ndarray) -> str:
    """Write an RGB ndarray as a one-page image PDF; return the temp path."""
    h, w = img.shape[:2]
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    Image.fromarray(img).save(tmp.name)
    doc = fitz.open()
    page = doc.new_page(width=w, height=h)
    page.insert_image(page.rect, filename=tmp.name)
    out = tmp.name.replace(".png", ".pdf")
    doc.save(out)
    return out


# ---- perturbations (each returns a worsened image + the subscore it should hurt) ----
def contrast_down(img):
    return np.clip(128 + (img.astype(float) - 128) * 0.35, 0, 255).astype(np.uint8), "accessibility"


def palette_clutter(img):
    out = img.copy()
    h, w = out.shape[:2]
    rng = list(range(12))
    hues = [(230, 30, 30), (30, 30, 230), (240, 200, 20), (20, 200, 60),
            (230, 20, 230), (20, 220, 220)]
    for i in rng:  # deterministic placement (no RNG — grid-ish scatter)
        cx = int(w * ((i * 137) % 100) / 100)
        cy = int(h * ((i * 89) % 100) / 100)
        c = hues[i % len(hues)]
        r = min(h, w) // 14
        out[max(0, cy - r):cy + r, max(0, cx - r):cx + r] = c
    return out, "palette"


def overcrowd(img):
    """Overlay a grid of gray blocks (~1/3 of area) -> less whitespace, denser look."""
    out = img.copy()
    h, w = out.shape[:2]
    sy, sx = h // 9, w // 9
    for gy in range(9):
        for gx in range(9):
            if (gy + gx) % 2:  # checkerboard: ~half the cells, each ~half-filled
                y, x = gy * sy, gx * sx
                out[y:y + sy // 2, x:x + sx] = 120  # dense gray band
    return out, "density_balance"


def blur(img):
    return np.array(Image.fromarray(img).filter(ImageFilter.GaussianBlur(2.5))), "accessibility"


PERTURBATIONS = [("contrast_down", contrast_down), ("palette_clutter", palette_clutter),
                 ("overcrowd", overcrowd), ("blur", blur)]


def score_path(pdf_path) -> dict:
    return design_score.design_score(poster_metrics.analyze_poster(pdf_path))


def main():
    posters = sorted(glob.glob("research/assets/p2peval_posters/*.pdf"))
    # keep vector posters only (skip already-flattened)
    posters = [p for p in posters if "flattened" not in p][:6]
    print(f"Validating on {len(posters)} posters × {len(PERTURBATIONS)} perturbations\n")

    per_pert = {name: {"wins": 0, "n": 0, "d_overall": [], "d_target": []}
                for name, _ in PERTURBATIONS}
    for pdf in posters:
        base_img = render_rgb(pdf)
        base = score_path(rgb_to_pdf(base_img))
        name = pdf.split("/")[-1]
        print(f"{name}: baseline design score {base['overall']}")
        for pname, fn in PERTURBATIONS:
            dimg, target = fn(base_img)
            deg = score_path(rgb_to_pdf(dimg))
            d_overall = deg["overall"] - base["overall"]
            d_target = deg["subscores"][target] - base["subscores"][target]
            per_pert[pname]["n"] += 1
            per_pert[pname]["wins"] += int(deg["overall"] < base["overall"])
            per_pert[pname]["d_overall"].append(d_overall)
            per_pert[pname]["d_target"].append(d_target)
            print(f"    {pname:16s} overall {deg['overall']:5.1f} (Δ{d_overall:+.1f}) | "
                  f"{target} Δ{d_target:+.1f}")

    print("\n=== Summary (does degradation lower the score?) ===")
    total_wins = total_n = 0
    for pname, s in per_pert.items():
        wr = s["wins"] / s["n"] if s["n"] else 0
        total_wins += s["wins"]; total_n += s["n"]
        print(f"  {pname:16s} score dropped {s['wins']}/{s['n']} ({wr:.0%}) | "
              f"mean Δoverall {np.mean(s['d_overall']):+.1f} | "
              f"mean Δtarget {np.mean(s['d_target']):+.1f}")
    print(f"\n  OVERALL: degradation lowered the score in {total_wins}/{total_n} "
          f"cases ({total_wins/total_n:.0%}). A meaningful score should be near 100%.")


if __name__ == "__main__":
    main()
