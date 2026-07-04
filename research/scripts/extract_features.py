"""Extract design + text features for the linked ICLR'24 posters (scoring-head inputs).

PosterSum posters are raster PNGs, so every poster runs the image-regime metrics
engine (render metrics + OCR font sizes). Slow (~5s each) → resumable: caches
per-poster rows to a parquet and skips ones already done. Stratified by tier so
all four decision tiers are represented.

Usage:  python extract_features.py [N]   (default N≈300)
"""
from __future__ import annotations

import pathlib
import sys
import tempfile
import urllib.request

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import poster_metrics

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
OUT = DATA / "features_iclr2024.parquet"
TIER_RANK = {"Reject": 0, "Accept (poster)": 1, "Accept (spotlight)": 2, "Accept (oral)": 3}

NUM_FIELDS = [
    "aspect", "n_text_spans", "n_chars", "n_words", "n_text_blocks",
    "body_pt", "heading_pt", "title_pt", "pct_body_below_floor",
    "text_coverage", "visual_density", "whitespace", "n_columns",
    "n_palette_colors", "saturated_hues", "median_contrast", "min_contrast",
    "pct_text_below_wcag",
]


def stratified(df, n):
    df = df[df.mean_rating.notna() & df.image_url.notna()].copy()
    # take all rare tiers, sample the common ones, aiming for ~n total
    parts, quota = [], {"Reject": 999, "Accept (oral)": 999,
                        "Accept (spotlight)": max(1, n // 4), "Accept (poster)": max(1, n // 2)}
    for tier, q in quota.items():
        sub = df[df.decision == tier]
        parts.append(sub if len(sub) <= q else sub.sample(q, random_state=7))
    return pd.concat(parts).reset_index(drop=True)


def features_for(row) -> dict | None:
    req = urllib.request.Request(row.image_url, headers={"User-Agent": "posterreview/0.1"})
    try:
        data = urllib.request.urlopen(req, timeout=90).read()
    except Exception as e:
        print(f"  download failed {row.or_id}: {e}")
        return None
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tf:
        tf.write(data); tf.flush()
        try:
            m = poster_metrics.analyze_poster(tf.name)
        except Exception as e:
            print(f"  analyze failed {row.or_id}: {e}")
            return None
    feat = {f: getattr(m, f) for f in NUM_FIELDS}
    feat["is_portrait"] = int(m.orientation == "portrait")
    feat["title_chars"] = len(str(row.title))
    feat["abstract_chars"] = len(str(row.abstract))
    feat["abstract_words"] = len(str(row.abstract).split())
    feat["n_topics"] = len(row.topics) if hasattr(row.topics, "__len__") else 0
    feat["or_id"] = row.or_id
    feat["title"] = row.title
    feat["mean_rating"] = row.mean_rating
    feat["decision"] = row.decision
    feat["tier_rank"] = TIER_RANK.get(row.decision)
    feat["is_high_tier"] = int(row.decision in ("Accept (spotlight)", "Accept (oral)"))
    return feat


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    posters = pd.read_parquet(DATA / "linkage_iclr2024.parquet")
    sample = stratified(posters, n)
    print(f"target sample: {len(sample)} posters", flush=True)
    print(sample.decision.value_counts().to_dict(), flush=True)

    done = set()
    rows = []
    if OUT.exists():
        prev = pd.read_parquet(OUT)
        rows = prev.to_dict("records")
        done = set(prev.or_id)
        print(f"resuming: {len(done)} already done", flush=True)

    todo = sample[~sample.or_id.isin(done)]
    for i, (_, row) in enumerate(todo.iterrows(), 1):
        f = features_for(row)
        if f:
            rows.append(f)
        if i % 10 == 0 or i == len(todo):
            pd.DataFrame(rows).to_parquet(OUT)
            print(f"  {i}/{len(todo)} processed, {len(rows)} rows saved", flush=True)
    pd.DataFrame(rows).to_parquet(OUT)
    print(f"DONE: {len(rows)} feature rows -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
