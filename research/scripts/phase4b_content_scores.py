"""Phase 4b — LLM content-dimension scores for the 293 labeled ICLR posters.

Phase 4 showed design metrics DON'T predict the reviewer rating (they measure the
paper, not the poster). The open question: do *content* dimensions — judged by an
LLM from the poster's research — predict it? (Stanford's actual mechanism.) This
generates those scores with the local Spark model (zero cost), so the harness can
test the "valid half" of the OpenReview signal.

Scores-only (no rationales) → tiny output → fast per poster. Resumable.
Content is judged from the poster's title + abstract + topics (the readily
available research content; poster text derives from the paper anyway).

Usage:  POSTERREVIEW_OLLAMA_MODEL=qwen2.5:72b OLLAMA_HOST=127.0.0.1:11434 \
        python phase4b_content_scores.py [N]
"""
from __future__ import annotations

import os
import pathlib
import sys

import pandas as pd

os.environ.setdefault("OLLAMA_HOST", "127.0.0.1:11434")
os.environ.setdefault("POSTERREVIEW_OLLAMA_MODEL", "qwen2.5:72b")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import local_llm

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
OUT = DATA / "content_scores_iclr2024.parquet"
DIMS = ["importance", "claim_support", "contextualization", "takeaway_clarity", "self_containedness"]

SYSTEM = ("You are an expert peer reviewer judging research quality from a poster's abstract. "
          "Score each dimension 1-5 (1 poor, 5 excellent; 5 is rare, reserved for standout work). "
          "Output ONLY a JSON object, no prose.")


def score_row(row) -> dict | None:
    topics = ", ".join(row.topics) if hasattr(row.topics, "__len__") and not isinstance(row.topics, str) else ""
    prompt = (
        f"Title: {row.title}\nTopics: {topics}\nAbstract: {str(row.abstract)[:1800]}\n\n"
        "Score these dimensions of the research (integers 1-5):\n"
        "- importance: how important/impactful is the research question?\n"
        "- claim_support: how well would the claims likely be supported?\n"
        "- contextualization: novelty and positioning vs the field.\n"
        "- takeaway_clarity: is there a clear, crisp main finding?\n"
        "- self_containedness: is the work clearly and completely described?\n\n"
        'Output ONLY: {"importance":N,"claim_support":N,"contextualization":N,'
        '"takeaway_clarity":N,"self_containedness":N}'
    )
    raw = local_llm.generate_json(prompt, system=SYSTEM, max_tokens=200)
    if not raw:
        return None
    out = {"or_id": row.or_id}
    for d in DIMS:
        try:
            out[f"c_{d}"] = max(1, min(5, int(round(float(raw.get(d, 3))))))
        except (TypeError, ValueError):
            out[f"c_{d}"] = 3
    out["c_mean"] = round(sum(out[f"c_{d}"] for d in DIMS) / len(DIMS), 2)
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    feats = pd.read_parquet(DATA / "features_iclr2024.parquet")
    link = pd.read_parquet(DATA / "linkage_iclr2024.parquet")[["or_id", "abstract", "topics"]]
    df = feats.merge(link, on="or_id", how="left").drop_duplicates("or_id").head(n)
    print(f"scoring {len(df)} posters with {local_llm.engine_info()}", flush=True)

    rows, done = [], set()
    if OUT.exists():
        prev = pd.read_parquet(OUT)
        rows = prev.to_dict("records"); done = set(prev.or_id)
        print(f"resuming: {len(done)} done", flush=True)
    todo = df[~df.or_id.isin(done)]
    for i, (_, row) in enumerate(todo.iterrows(), 1):
        r = score_row(row)
        if r:
            rows.append(r)
        if i % 10 == 0 or i == len(todo):
            pd.DataFrame(rows).to_parquet(OUT)
            print(f"  {i}/{len(todo)} scored ({len(rows)} total)", flush=True)
    pd.DataFrame(rows).to_parquet(OUT)
    print(f"DONE: {len(rows)} content-score rows -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
