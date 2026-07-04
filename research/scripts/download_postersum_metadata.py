"""Download PosterSum metadata (all 16,305 records, ~24MB — URLs only, no images)."""
import json
import pathlib
import urllib.request

import pandas as pd

DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

API = "https://huggingface.co/api/datasets/rohitsaxena/PosterSum/parquet/default"

frames = []
for split in ["train", "validation", "test"]:
    with urllib.request.urlopen(f"{API}/{split}") as r:
        urls = json.load(r)
    for url in urls:
        df = pd.read_parquet(url)
        df["split"] = split
        frames.append(df)
        print(f"{split}: {len(df)} rows from {url.split('/')[-1]}")

full = pd.concat(frames, ignore_index=True)
out = DATA_DIR / "postersum_metadata.parquet"
full.to_parquet(out)
print(f"\nTotal: {len(full)} posters -> {out}")
print(full.groupby(["conference", "year"]).size())
