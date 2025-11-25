"""
Profile train_mlframe_models_suite with cProfile.

Usage:
    python profile_training.py
"""

import time
import cProfile
import pstats
import io
import numpy as np
import polars as pl
import tempfile
import os
from datetime import datetime, timedelta

# Setup logging
import logging

logging.basicConfig(level=logging.INFO)

from mlframe.training.core import train_mlframe_models_suite
from mlframe.training_old import SimpleFeaturesAndTargetsExtractor
from mlframe.metrics import prewarm_numba_cache


def create_synthetic_data(n_rows: int = 1_000_000, n_cols: int = 50) -> pl.DataFrame:
    """Create synthetic dataset for profiling using Polars."""
    print(f"Creating synthetic data: {n_rows:,} rows x {n_cols} columns...")

    np.random.seed(42)

    # Create feature columns
    data = {}

    # Numeric features (40 columns)
    for i in range(40):
        data[f"num_{i}"] = np.random.randn(n_rows).astype(np.float32)

    # Categorical features (8 columns)
    for i in range(8):
        data[f"cat_{i}"] = np.random.choice(["A", "B", "C", "D", "E"], n_rows)

    # Target columns (binary classification)
    data["target"] = np.random.randint(0, 2, n_rows).astype(np.float32)
    data["target2"] = np.random.randint(0, 2, n_rows).astype(np.float32)

    # Timestamp column
    start = datetime(2020, 1, 1)
    data["timestamp"] = [start + timedelta(seconds=i) for i in range(n_rows)]

    df = pl.DataFrame(data)
    memory_gb = df.estimated_size() / 1e9
    print(f"Data created. Shape: {df.shape}, Memory: {memory_gb:.2f} GB")

    return df


def run_profiling():
    """Run profiling on train_mlframe_models_suite."""

    # Pre-warm Numba JIT cache
    print("Pre-warming Numba JIT cache...")
    prewarm_numba_cache()
    print("Numba cache warmed.\n")

    # Create synthetic data (500k rows)
    df = create_synthetic_data(n_rows=5_000_000, n_cols=50)

    # Create features and targets extractor (with two targets)
    ft_extractor = SimpleFeaturesAndTargetsExtractor(regression_targets=["target", "target2"], columns_to_drop={"timestamp"}, verbose=1)

    # Create temp directory for outputs
    with tempfile.TemporaryDirectory() as tmpdir:

        def run_training():
            return train_mlframe_models_suite(
                df=df,
                target_name="profiling_test",
                model_name="profile_run",
                features_and_targets_extractor=ft_extractor,
                mlframe_models=["cb"],  # Single CatBoost model
                use_ordinary_models=True,
                use_mlframe_ensembles=False,
                use_mrmr_fs=False,
                data_dir=tmpdir,
                models_dir="models",
                verbose=1,
                control_params_override={"prefer_gpu_configs": False},  # Force CPU
            )

        # Profile the function
        print("\n" + "=" * 80)
        print("STARTING PROFILING")
        print("=" * 80 + "\n")

        profiler = cProfile.Profile()
        start_time = time.time()
        profiler.enable()
        models, metadata = run_training()
        profiler.disable()
        elapsed = time.time() - start_time

        print("\n" + "=" * 80)
        print(f"TOTAL TIME: {elapsed:.2f} seconds")
        print("=" * 80)

        # Print top 40 by cumulative time
        print("\nTOP 40 HOTSPOTS (by cumulative time):")
        print("=" * 80)
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats("cumulative")
        stats.print_stats(40)
        print(stream.getvalue())


if __name__ == "__main__":
    run_profiling()
