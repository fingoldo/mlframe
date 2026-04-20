"""Diagnose categorical-feature drift between train/val/test splits.

Motivation
----------
2026-04-20: ``train_mlframe_models_suite`` crashed silently (kernel die
without traceback) during XGBoost's val IterativeDMatrix construction.
Timing diff was tight: train DMatrix was built at T+~4min, kernel
died ~1min later — before val DMatrix finished. Removing ALL 20
categorical features (``skills_text``, ``job_type``, ``occupation``,
etc.) from the config bypassed the crash entirely. So the trigger is
one of those cats.

XGB 3.x with ``enable_categorical=True`` builds categorical dictionaries
during train-DMatrix construction, then replays them during val-DMatrix
construction using ``ref=train_dmatrix``. If val contains category
values that don't exist in train (or vice versa — unlikely since train
is usually larger), native code can mishandle the lookup. On Windows
this surfaces as a silent process kill.

This script compares, per categorical feature:

  * cardinality in train / val / test
  * number of categories present in one split but missing from another
  * null fraction drift
  * most common value % in each split (skew)

A "drift score" is computed to rank suspicious features; columns with
high score + high cardinality are the most likely crash triggers.

Usage
-----
    python -m mlframe.scripts.diagnose_cat_drift \\
        --parquet D:/path/to/prod_jobsdetails.parquet \\
        --timestamp-col job_posted_at \\
        --val-size 0.09 --test-size 0.10 \\
        --cat-features job_type category category_group contractor_tier \\
                       workload occupation skills_text ontology_skills_text \\
                       hourly_budget_type _raw_segmentation job_post_type \\
                       job_post_device job_post_browser job_post_source \\
                       job_post_flow_type job_urgency desc_ai_opted_in \\
                       qual_type job_req_english job_local_flexibility

Output is a table sorted by drift_score desc. Columns with
``val_minus_train > 0`` are the most interesting — those contain
category values XGB has never seen at fit time.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

import polars as pl

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _unique_values_set(series: pl.Series, drop_nulls: bool = True) -> set:
    """Return a Python set of distinct values in the series. For very
    high-cardinality columns (>5M uniques), caps at 5M to avoid OOM —
    the drift metric is approximate in that case."""
    vals = series.drop_nulls() if drop_nulls else series
    n = vals.n_unique()
    if n > 5_000_000:
        # Sample: take 1M random values' uniques. Approximate.
        vals = vals.sample(n=1_000_000, seed=42, with_replacement=False)
    return set(vals.to_list())


def _most_common_fraction(series: pl.Series) -> float:
    """Fraction of the modal (most-frequent) value among non-null rows.
    High values (>0.9) indicate severe skew."""
    n = series.drop_nulls().len()
    if n == 0:
        return 0.0
    vc = series.drop_nulls().value_counts(sort=True)
    if vc.height == 0:
        return 0.0
    top_count = vc["count"][0] if "count" in vc.columns else vc[vc.columns[-1]][0]
    return float(top_count) / n


def _summarize_column(name: str, train, val, test) -> dict:
    """Per-feature drift summary. All three inputs are pl.Series."""
    train_uniques = _unique_values_set(train)
    val_uniques = _unique_values_set(val)
    test_uniques = _unique_values_set(test)

    val_minus_train = val_uniques - train_uniques
    train_minus_val = train_uniques - val_uniques
    test_minus_train = test_uniques - train_uniques
    train_minus_test = train_uniques - test_uniques

    n_tr, n_v, n_te = train.len(), val.len(), test.len()
    null_tr = train.null_count() / max(n_tr, 1)
    null_v = val.null_count() / max(n_v, 1)
    null_te = test.null_count() / max(n_te, 1)

    # Drift score: high when val has many categories train never saw
    # (the primary crash hypothesis), weighted by cardinality and
    # relative size of the unseen set.
    train_card = max(len(train_uniques), 1)
    score_val_drift = len(val_minus_train) / train_card
    score_test_drift = len(test_minus_train) / train_card
    # Bias toward val drift (that's the crash path); test drift secondary.
    drift_score = 2.0 * score_val_drift + 1.0 * score_test_drift

    return {
        "feature": name,
        "card_train": len(train_uniques),
        "card_val": len(val_uniques),
        "card_test": len(test_uniques),
        "val_minus_train": len(val_minus_train),
        "train_minus_val": len(train_minus_val),
        "test_minus_train": len(test_minus_train),
        "train_minus_test": len(train_minus_test),
        "null_train": null_tr,
        "null_val": null_v,
        "null_test": null_te,
        "skew_train_top1": _most_common_fraction(train),
        "skew_val_top1": _most_common_fraction(val),
        "skew_test_top1": _most_common_fraction(test),
        "drift_score": drift_score,
    }


def split_timeordered(
    df: pl.DataFrame,
    timestamp_col: str,
    val_size: float,
    test_size: float,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Time-ordered train/val/test split matching mlframe's default
    sequential behaviour (wholeday_splitting=False)."""
    df_sorted = df.sort(timestamp_col)
    n = df_sorted.height
    n_test = int(n * test_size)
    n_val = int(n * val_size)
    n_train = n - n_val - n_test
    train = df_sorted[:n_train]
    val = df_sorted[n_train : n_train + n_val]
    test = df_sorted[n_train + n_val :]
    return train, val, test


def run_diagnostic(
    parquet_path: Path,
    cat_features: Sequence[str],
    timestamp_col: str,
    val_size: float = 0.09,
    test_size: float = 0.10,
    out_csv: Optional[Path] = None,
) -> pl.DataFrame:
    print(f"Loading {parquet_path}...")
    t0 = time.perf_counter()
    df = pl.read_parquet(parquet_path)
    print(f"  loaded {df.shape[0]:_} rows x {df.shape[1]} cols in {time.perf_counter()-t0:.1f}s")

    missing = [c for c in cat_features if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: cat features not in parquet: {missing}")
    if timestamp_col not in df.columns:
        sys.exit(f"ERROR: timestamp column {timestamp_col!r} not in parquet")

    print(f"Splitting by {timestamp_col} (val={val_size:.0%}, test={test_size:.0%})...")
    train, val, test = split_timeordered(df, timestamp_col, val_size, test_size)
    print(f"  train: {train.height:_}  val: {val.height:_}  test: {test.height:_}")

    print("Computing per-feature drift...")
    rows = []
    for i, name in enumerate(cat_features):
        t1 = time.perf_counter()
        summary = _summarize_column(name, train[name], val[name], test[name])
        rows.append(summary)
        print(
            f"  [{i+1:2d}/{len(cat_features)}] {name}: "
            f"card_tr={summary['card_train']:_}, "
            f"val-tr={summary['val_minus_train']:_}, "
            f"test-tr={summary['test_minus_train']:_}, "
            f"null_tr={summary['null_train']:.3f}, "
            f"drift={summary['drift_score']:.4f}  ({time.perf_counter()-t1:.1f}s)"
        )

    result = pl.DataFrame(rows).sort("drift_score", descending=True)
    print("\n=== DRIFT REPORT (sorted by suspicion) ===")
    with pl.Config(
        tbl_rows=100,
        tbl_cols=20,
        tbl_width_chars=200,
        float_precision=4,
    ):
        print(result)

    if out_csv:
        result.write_csv(out_csv)
        print(f"\nSaved to {out_csv}")

    # Call out the top 3 suspects explicitly.
    print("\n=== TOP 3 CRASH-TRIGGER SUSPECTS ===")
    for row in result.head(3).iter_rows(named=True):
        print(
            f"  {row['feature']}: card_train={row['card_train']:_}, "
            f"val-only={row['val_minus_train']:_} categories not seen in train, "
            f"drift_score={row['drift_score']:.4f}"
        )
    print()

    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", required=True, type=Path, help="Path to parquet with the full dataframe.")
    ap.add_argument("--timestamp-col", required=True, help="Column to sort by for time-ordered split.")
    ap.add_argument("--cat-features", nargs="+", required=True, help="Categorical feature names to diagnose.")
    ap.add_argument("--val-size", type=float, default=0.09, help="Fraction for validation split (default: 0.09, matches prod log).")
    ap.add_argument("--test-size", type=float, default=0.10, help="Fraction for test split (default: 0.10).")
    ap.add_argument("--out-csv", type=Path, default=None, help="Optional: also save report to this CSV.")
    args = ap.parse_args()

    run_diagnostic(
        parquet_path=args.parquet,
        cat_features=args.cat_features,
        timestamp_col=args.timestamp_col,
        val_size=args.val_size,
        test_size=args.test_size,
        out_csv=args.out_csv,
    )


if __name__ == "__main__":
    main()
