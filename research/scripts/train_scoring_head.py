"""Train + evaluate the scoring head (paperreview.ai's method, for posters).

Stanford maps 7 LLM dimension scores -> final rating via linear regression trained
on ICLR human review scores. We do the same structurally, but on the features we
have WITHOUT an API key: deterministic design metrics + metadata text features ->
real OpenReview reviewer rating / accept-tier.

This is the honest DESIGN-ONLY baseline and the reusable harness: when LLM
dimension scores are added (needs a key), they slot in as extra feature columns.

Reports (held-out, repeated CV):
  - Spearman(predicted rating, human mean rating)   [vs Stanford: human-human 0.41]
  - ROC-AUC for predicting high-tier (spotlight/oral)
  - Calibration of predicted vs actual rating
  - Which features carry signal

Usage:  python train_scoring_head.py
"""
from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, mean_absolute_error

warnings.filterwarnings("ignore")
DATA = pathlib.Path(__file__).resolve().parents[1] / "data"

DESIGN = [
    "aspect", "is_portrait", "n_text_spans", "n_chars", "n_words", "n_text_blocks",
    "body_pt", "heading_pt", "title_pt", "pct_body_below_floor",
    "text_coverage", "visual_density", "whitespace", "n_columns",
    "n_palette_colors", "saturated_hues", "median_contrast", "min_contrast",
    "pct_text_below_wcag",
]
TEXT = ["title_chars", "abstract_chars", "abstract_words", "n_topics"]
FEATURES = DESIGN + TEXT


def load():
    df = pd.read_parquet(DATA / "features_iclr2024.parquet")
    df = df[df.mean_rating.notna() & df.tier_rank.notna()].copy()
    for c in FEATURES:
        if c not in df:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())
    return df


def cv_regression(X, y, strat, model_fn, n_splits=5, n_repeats=6):
    """Repeated CV; return out-of-fold Spearman and MAE aggregated across repeats."""
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)
    rhos, maes = [], []
    for tr, te in rkf.split(X, strat):
        sc = StandardScaler().fit(X[tr])
        m = model_fn().fit(sc.transform(X[tr]), y[tr])
        pred = m.predict(sc.transform(X[te]))
        if np.std(pred) > 1e-9:
            rhos.append(spearmanr(pred, y[te]).correlation)
        maes.append(mean_absolute_error(y[te], pred))
    return np.nanmean(rhos), np.nanstd(rhos), np.mean(maes)


def cv_auc(X, y_bin, model_fn, n_splits=5, n_repeats=6):
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=2)
    aucs = []
    for tr, te in rkf.split(X, y_bin):
        if len(np.unique(y_bin[te])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        m = model_fn().fit(sc.transform(X[tr]), y_bin[tr])
        p = m.predict_proba(sc.transform(X[te]))[:, 1]
        aucs.append(roc_auc_score(y_bin[te], p))
    return np.mean(aucs), np.std(aucs)


def main():
    df = load()
    print(f"N = {len(df)} posters")
    print("tier distribution:", df.decision.value_counts().to_dict())
    print(f"rating range {df.mean_rating.min():.1f}-{df.mean_rating.max():.1f}, "
          f"median {df.mean_rating.median():.2f}\n")

    X = df[FEATURES].to_numpy(float)
    y = df.mean_rating.to_numpy(float)
    strat = df.tier_rank.to_numpy(int)
    y_high = df.is_high_tier.to_numpy(int)

    print("=== Predicting reviewer mean rating (Spearman vs human, held-out) ===")
    print("  reference: human-vs-human Spearman ≈ 0.41 (Stanford, ICLR)")
    for name, fn in [("Ridge (design+text)", lambda: Ridge(alpha=10.0)),
                     ("GBM (design+text)", lambda: GradientBoostingRegressor(
                         n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8))]:
        rho, sd, mae = cv_regression(X, y, strat, fn)
        print(f"  {name:24s} Spearman {rho:.3f} ± {sd:.3f} | MAE {mae:.2f}")

    # ablation: design-only vs text-only
    for label, cols in [("design-only", DESIGN), ("text-only", TEXT)]:
        Xa = df[cols].to_numpy(float)
        rho, sd, mae = cv_regression(Xa, y, strat, lambda: Ridge(alpha=10.0))
        print(f"  Ridge {label:20s} Spearman {rho:.3f} ± {sd:.3f} | MAE {mae:.2f}")

    print("\n=== Predicting high-tier (spotlight/oral) — ROC-AUC, held-out ===")
    auc, asd = cv_auc(X, y_high, lambda: LogisticRegression(max_iter=1000, C=0.5))
    print(f"  LogReg (design+text)     AUC {auc:.3f} ± {asd:.3f}  "
          f"(base rate {y_high.mean():.2f})")

    print("\n=== Feature signal (GBM importance, full fit) ===")
    sc = StandardScaler().fit(X)
    gbm = GradientBoostingRegressor(n_estimators=200, max_depth=2,
                                    learning_rate=0.05, subsample=0.8, random_state=0)
    gbm.fit(sc.transform(X), y)
    imp = sorted(zip(FEATURES, gbm.feature_importances_), key=lambda t: -t[1])
    for f, v in imp[:10]:
        print(f"  {f:22s} {v:.3f}")

    print("\n=== Calibration (single 60/40 split) ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)
    tr, te = next(skf.split(X, strat))
    scc = StandardScaler().fit(X[tr])
    reg = Ridge(alpha=10.0).fit(scc.transform(X[tr]), y[tr])
    pred = reg.predict(scc.transform(X[te]))
    dfc = pd.DataFrame({"pred": pred, "true": y[te]})
    dfc["bin"] = pd.qcut(dfc.pred, min(4, dfc.pred.nunique()), duplicates="drop")
    print(dfc.groupby("bin", observed=True).agg(
        n=("true", "size"), pred_mean=("pred", "mean"), true_mean=("true", "mean")).round(2))


if __name__ == "__main__":
    main()
