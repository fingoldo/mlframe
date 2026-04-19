"""Profile the four "medium-impact" blocks from the 2026-04-19 timing
analysis:

  - ``compute_split_metrics``          — 3.0 s/call × 4
  - ``fast_calibration_report``        — 1.9 s/call × 4
  - ``plot_feature_importances``       — 1.5 s/call × 2
  - ``report_probabilistic_model_perf``— 2.3 s/call × 4

Synthetic data is sized and shaped to match the production prod_jobsdetails
frame as closely as is reasonable for a single-host profile run:

  - 810_000 × 100 columns
  - mixed dtypes: Float32, Int16, Boolean, Categorical (some with nulls),
    one or two long-string "text blob" columns

We run the full suite through ``train_mlframe_models_suite`` with cb + xgb,
capture with cProfile, then pull the per-function cumulative time for each
target block AND its direct callees. That pinpoints which sub-function
dominates inside each block.

Usage:
    python -m mlframe.profiling.profile_metrics_blocks
"""
from __future__ import annotations

import sys
import os
import time
import tempfile
import cProfile
import pstats
import io
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import polars as pl

# Windows cp1251 stdout is incompatible with our log arrows/emoji — be safe.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import logging
logging.basicConfig(level=logging.WARNING)  # suppress INFO noise, keep WARN+

from mlframe.training.core import train_mlframe_models_suite
from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor
from mlframe.metrics import prewarm_numba_cache


# ============================================================================
# Synthetic dataset matching production prod_jobsdetails shape
# ============================================================================

def create_synthetic_data(n_rows: int = 810_000) -> pl.DataFrame:
    """Build a Polars DataFrame with prod-like shape.

    Column mix (matches 2026-04-19 prod log schema):
      - 40 Float32 numeric
      - 30 Int16 numeric
      - 15 Boolean
      - 14 Categorical (6 of them with 10-50% nulls, 8 clean)
      - 1 high-cardinality text column (skills-text analogue, ~150k uniques)
      - 1 timestamp column (for time-based split)
      - 2 targets (one binary, one regression)
    """
    print(f"Building synthetic {n_rows:,} × ~100 frame...")
    np.random.seed(42)
    cols = {}

    # 40 Float32
    for i in range(40):
        cols[f"num_f{i}"] = np.random.randn(n_rows).astype(np.float32)

    # 30 Int16
    for i in range(30):
        cols[f"num_i{i}"] = np.random.randint(-1000, 1000, size=n_rows).astype(np.int16)

    # 15 Boolean
    for i in range(15):
        cols[f"bool_{i}"] = np.random.choice([True, False], size=n_rows)

    # 8 "clean" Categoricals (no nulls, 3-15 unique values — enum-like)
    for i in range(8):
        k = np.random.randint(3, 15)
        pool = [f"cat{i}_v{j}" for j in range(k)]
        cols[f"cat_clean_{i}"] = pl.Series(
            f"cat_clean_{i}",
            np.random.choice(pool, size=n_rows),
        ).cast(pl.Categorical)

    # 6 "null-heavy" Categoricals (10-70% nulls) — matches prod where
    # hourly_budget_type, contractor_tier, etc. have null_count in
    # the hundreds-of-thousands range.
    for i in range(6):
        k = int(np.random.randint(3, 10))
        pool = [f"ncat{i}_v{j}" for j in range(k)]
        null_frac = 0.1 + 0.6 * (i / 5)  # 0.1 … 0.7
        picks = np.random.choice(pool, size=n_rows).tolist()
        null_mask = np.random.random(n_rows) < null_frac
        for idx, is_null in enumerate(null_mask):
            if is_null:
                picks[idx] = None
        cols[f"cat_null_{i}"] = pl.Series(
            f"cat_null_{i}", picks, dtype=pl.String
        ).cast(pl.Categorical)

    # 1 high-cardinality text column: synthesize "skills_text"-like blob.
    # ~150k uniques over 810k rows so CB's text estimator has real work.
    vocab = [f"skill_{i:05d}" for i in range(15_000)]
    n_words_per_row = 8
    text_vals = [
        " ".join(np.random.choice(vocab, size=n_words_per_row))
        for _ in range(n_rows)
    ]
    cols["skills_text"] = pl.Series("skills_text", text_vals, dtype=pl.String)

    # Timestamp for ordered split — must span enough distinct days for
    # wholeday_splitting to produce non-empty train/val/test. At 100k rows
    # we spread over ~35 days (30 s/row).
    start = datetime(2020, 1, 1)
    step_s = max(1, int(3_000_000 / max(n_rows, 1)))
    cols["timestamp"] = [start + timedelta(seconds=i * step_s) for i in range(n_rows)]

    # Targets (one binary-ish 0/1 with some signal from num_f0)
    logits = cols["num_f0"] + 0.3 * cols["num_f1"]
    probs = 1.0 / (1.0 + np.exp(-logits))
    cols["target"] = (np.random.random(n_rows) < probs).astype(np.float32)
    cols["target2"] = (np.random.random(n_rows) < probs).astype(np.float32)

    df = pl.DataFrame(cols)
    print(f"Built. Shape={df.shape}, size≈{df.estimated_size()/1e9:.2f} GB")
    return df


# ============================================================================
# Profiling driver
# ============================================================================

TARGET_BLOCKS = [
    "compute_split_metrics",
    "report_probabilistic_model_perf",
    "fast_calibration_report",
    "plot_feature_importances",
]


def _capture_print(stats: pstats.Stats, call, *args) -> str:
    """Redirect stats.<call>() output to a string. pstats writes to
    stats.stream (or sys.stdout if unset), so we swap it temporarily."""
    buf = io.StringIO()
    old_stream = stats.stream
    stats.stream = buf
    try:
        call(*args)
    finally:
        stats.stream = old_stream
    return buf.getvalue()


def print_block_stats(stats: pstats.Stats, target_fn_name: str, top_callees: int = 15) -> None:
    """Print the block's own cumulative/total time and its heaviest callees.

    Uses pstats' built-in regex restriction — first arg to print_stats /
    print_callees is a regex matched against the full "path:line(name)"
    label. The function-name substring is a reliable match.
    """
    print("\n" + "=" * 84)
    print(f"BLOCK: {target_fn_name}")
    print("=" * 84)

    stats.sort_stats("cumulative")

    # 1) The block itself — matches wrapper + the real function.
    itself = _capture_print(stats, stats.print_stats, target_fn_name, 5)
    print(itself)

    # 2) Callees — what each instance of the function spent time in.
    callees = _capture_print(stats, stats.print_callees, target_fn_name, top_callees)
    lines = callees.splitlines()
    max_lines = 80
    if len(lines) > max_lines:
        callees = "\n".join(lines[:max_lines]) + f"\n  ... (truncated, {len(lines) - max_lines} more lines)"
    print(callees)


def main():
    print("Pre-warming Numba JIT cache...")
    prewarm_numba_cache()
    print("Numba cache warmed.\n")

    # Use a smaller frame (100k) for tractable local profiling while
    # still exercising every block ≥ 4× (2 targets × 2 splits).
    n_rows = int(os.environ.get("PROFILE_N_ROWS", "100000"))
    df = create_synthetic_data(n_rows=n_rows)

    extractor = SimpleFeaturesAndTargetsExtractor(
        classification_targets=["target", "target2"],
        columns_to_drop={"timestamp"},
        ts_field="timestamp",
        verbose=0,
    )

    with tempfile.TemporaryDirectory() as tmpdir:

        def run():
            return train_mlframe_models_suite(
                df=df,
                target_name="profile_metrics_blocks",
                model_name="profile_run",
                features_and_targets_extractor=extractor,
                mlframe_models=["cb", "xgb"],
                use_ordinary_models=True,
                use_mlframe_ensembles=False,
                use_mrmr_fs=False,
                data_dir=tmpdir,
                models_dir="models",
                verbose=0,
                behavior_config={"prefer_gpu_configs": False},
            )

        print("\n" + "=" * 84)
        print("RUNNING TRAIN SUITE UNDER cProfile (cb + xgb, 2 targets)")
        print("=" * 84)

        profiler = cProfile.Profile()
        t0 = time.perf_counter()
        profiler.enable()
        run()
        profiler.disable()
        elapsed = time.perf_counter() - t0

        print(f"\nTotal wall-clock: {elapsed:.2f} s\n")

        # Save the raw profile to disk so we can re-analyze without
        # re-running the full suite.
        prof_path = os.environ.get(
            "PROFILE_DUMP_PATH",
            os.path.join(tempfile.gettempdir(), "profile_metrics_blocks.prof"),
        )
        try:
            profiler.dump_stats(prof_path)
            print(f"Raw profile saved to: {prof_path}\n")
        except OSError as e:
            print(f"(couldn't save raw profile: {e})\n")

        # Dump the full profile into pstats for analysis.
        stats = pstats.Stats(profiler)

        # High-level top 20 by cumulative time — gives the big picture.
        print("=" * 84)
        print("TOP 25 FUNCTIONS BY CUMULATIVE TIME")
        print("=" * 84)
        ctx = io.StringIO()
        stats_for_top = pstats.Stats(profiler, stream=ctx)
        stats_for_top.sort_stats("cumulative").print_stats(25)
        print(ctx.getvalue())

        # Per-block breakdown.
        for block in TARGET_BLOCKS:
            print_block_stats(stats, block)


if __name__ == "__main__":
    main()
