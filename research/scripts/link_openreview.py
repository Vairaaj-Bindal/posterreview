"""Link PosterSum posters to OpenReview: decision tier + reviewer scores.

Prototype on ICLR 2024 (API v2). Accepted papers carry content.venue like
"ICLR 2024 oral|spotlight|poster"; reviewer ratings live in Official_Review
reply notes on each forum.
"""
import json
import pathlib
import re
import sys
import time
import urllib.parse
import urllib.request

import pandas as pd

DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data"
API2 = "https://api2.openreview.net"


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "posterreview-research/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def norm_title(t):
    return re.sub(r"[^a-z0-9]", "", t.lower())


def fetch_venue_notes(venueid):
    """All accepted-paper notes for a venue (paginated)."""
    notes, offset = [], 0
    while True:
        q = urllib.parse.urlencode({"content.venueid": venueid, "limit": 1000, "offset": offset})
        batch = get(f"{API2}/notes?{q}").get("notes", [])
        notes.extend(batch)
        if len(batch) < 1000:
            return notes
        offset += 1000
        time.sleep(0.5)


def fetch_ratings(forum_id):
    """Extract numeric reviewer ratings from a forum's Official_Review replies."""
    notes = get(f"{API2}/notes?forum={forum_id}").get("notes", [])
    ratings = []
    for n in notes:
        invs = n.get("invitations", [])
        if not any("Official_Review" in i for i in invs):
            continue
        val = (n.get("content", {}).get("rating", {}) or {}).get("value")
        if val is None:
            continue
        m = re.match(r"\s*(\d+)", str(val))
        if m:
            ratings.append(int(m.group(1)))
    return ratings


def main():
    posters = pd.read_parquet(DATA_DIR / "postersum_metadata.parquet")
    iclr24 = posters[(posters.conference == "ICLR") & (posters.year == 2024)].copy()
    iclr24["norm_title"] = iclr24.title.map(norm_title)
    print(f"PosterSum ICLR 2024 posters: {len(iclr24)}")

    notes = fetch_venue_notes("ICLR.cc/2024/Conference")
    print(f"OpenReview ICLR 2024 accepted notes: {len(notes)}")

    or_rows = []
    for n in notes:
        c = n["content"]
        or_rows.append({
            "or_id": n["id"],
            "or_title": (c.get("title", {}) or {}).get("value", ""),
            "venue": (c.get("venue", {}) or {}).get("value", ""),
        })
    ordf = pd.DataFrame(or_rows)
    ordf["norm_title"] = ordf.or_title.map(norm_title)
    ordf["tier"] = ordf.venue.str.extract(r"(oral|spotlight|poster)", flags=re.I)[0].str.lower()

    merged = iclr24.merge(ordf, on="norm_title", how="left")
    matched = merged[merged.or_id.notna()]
    print(f"\nTitle match rate: {len(matched)}/{len(iclr24)} ({len(matched)/len(iclr24):.1%})")
    print("Tier distribution of matched posters:")
    print(matched.tier.value_counts(dropna=False))

    # Validate score extraction on a small sample across tiers
    sample = matched.groupby("tier").head(10)
    print(f"\nFetching reviews for {len(sample)} sample papers...")
    recs = []
    for _, row in sample.iterrows():
        ratings = fetch_ratings(row.or_id)
        recs.append({"title": row.title[:60], "tier": row.tier, "ratings": ratings,
                     "mean_rating": sum(ratings) / len(ratings) if ratings else None})
        time.sleep(0.3)
    rdf = pd.DataFrame(recs)
    print(rdf.to_string())
    print("\nMean rating by tier (sample):")
    print(rdf.groupby("tier").mean_rating.agg(["mean", "count"]))

    out = DATA_DIR / "linkage_iclr2024.parquet"
    merged.drop(columns=["norm_title"]).to_parquet(out)
    print(f"\nSaved linkage table -> {out}")


if __name__ == "__main__":
    sys.exit(main())
