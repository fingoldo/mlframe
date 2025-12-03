"""
Profile score_ensemble function to identify bottlenecks.

Usage:
    python -m mlframe.profiling.profile_ensembling
    python -m mlframe.profiling.profile_ensembling --n_samples 10000000
"""

import argparse
import cProfile
import io
import pstats
import time
import numpy as np
import pandas as pd

from mlframe.ensembling import score_ensemble, SIMPLE_ENSEMBLING_METHODS
from mlframe.training.configs import PredictionsContainer

# Pre-import to avoid counting import time in profiling
from mlframe.training import train_and_evaluate_model  # noqa: F401


def create_mock_model_results(n_samples: int, n_models: int = 3, seed: int = 42):
    """Create mock PredictionsContainer objects simulating trained model outputs.

    Args:
        n_samples: Total number of samples
        n_models: Number of models to simulate
        seed: Random seed for reproducibility

    Returns:
        List of PredictionsContainer objects with synthetic predictions
    """
    np.random.seed(seed)

    # Split samples into train/val/test (60/20/20)
    n_train = int(n_samples * 0.6)
    n_val = int(n_samples * 0.2)
    n_test = n_samples - n_train - n_val

    models = []
    for i in range(n_models):
        # Generate synthetic binary classification probabilities
        # Each model has slightly different predictions
        np.random.seed(seed + i)

        train_probs = np.clip(np.random.beta(2, 2, n_train) + np.random.randn(n_train) * 0.1, 0.01, 0.99)
        val_probs = np.clip(np.random.beta(2, 2, n_val) + np.random.randn(n_val) * 0.1, 0.01, 0.99)
        test_probs = np.clip(np.random.beta(2, 2, n_test) + np.random.randn(n_test) * 0.1, 0.01, 0.99)

        # Make 2D arrays (n_samples, 2) for binary classification probabilities
        train_probs_2d = np.column_stack([1 - train_probs, train_probs])
        val_probs_2d = np.column_stack([1 - val_probs, val_probs])
        test_probs_2d = np.column_stack([1 - test_probs, test_probs])

        model_result = PredictionsContainer(
            train_preds=(train_probs > 0.5).astype(np.int32),
            train_probs=train_probs_2d,
            val_preds=(val_probs > 0.5).astype(np.int32),
            val_probs=val_probs_2d,
            test_preds=(test_probs > 0.5).astype(np.int32),
            test_probs=test_probs_2d,
        )
        models.append(model_result)

    return models, n_train, n_val, n_test


def create_mock_targets(n_train: int, n_val: int, n_test: int, seed: int = 42):
    """Create mock target arrays."""
    np.random.seed(seed)

    train_target = pd.Series(np.random.randint(0, 2, n_train).astype(np.float64))
    val_target = pd.Series(np.random.randint(0, 2, n_val).astype(np.float64))
    test_target = pd.Series(np.random.randint(0, 2, n_test).astype(np.float64))

    return train_target, val_target, test_target


def profile_score_ensemble(n_samples: int, n_models: int = 3, top_n: int = 50, with_charts: bool = False):
    """Profile score_ensemble with cProfile.

    Args:
        n_samples: Number of samples to use
        n_models: Number of models to ensemble
        top_n: Number of top functions to show in profile output
        with_charts: If True, generate charts (adds significant time)
    """
    import tempfile
    import shutil

    print(f"\n{'='*70}")
    print(f"Profiling score_ensemble")
    print(f"  Samples: {n_samples:,}")
    print(f"  Models: {n_models}")
    print(f"  Ensembling methods: {SIMPLE_ENSEMBLING_METHODS}")
    print(f"  With charts: {with_charts}")
    print(f"{'='*70}\n")

    # Create mock data
    print("Creating mock data...")
    start = time.perf_counter()
    models, n_train, n_val, n_test = create_mock_model_results(n_samples, n_models)
    train_target, val_target, test_target = create_mock_targets(n_train, n_val, n_test)
    print(f"  Mock data created in {time.perf_counter() - start:.2f}s")
    print(f"  Train: {n_train:,}, Val: {n_val:,}, Test: {n_test:,}")

    # Create indices
    train_idx = np.arange(n_train)
    val_idx = np.arange(n_val)
    test_idx = np.arange(n_test)

    # Create dummy DataFrames to trigger full metrics computation
    # (_compute_split_metrics returns early if df is None, skipping fast_calibration_report)
    train_df = pd.DataFrame({'dummy': np.zeros(n_train)})
    val_df = pd.DataFrame({'dummy': np.zeros(n_val)})
    test_df = pd.DataFrame({'dummy': np.zeros(n_test)})

    # Setup chart output directory
    temp_dir = None
    plot_file = None
    if with_charts:
        temp_dir = tempfile.mkdtemp(prefix="profile_ensembling_")
        plot_file = temp_dir + "/"
        print(f"  Charts output: {temp_dir}")

    # Profile with cProfile
    print("\nRunning cProfile...")
    profiler = cProfile.Profile()

    start = time.perf_counter()
    profiler.enable()

    try:
        result = score_ensemble(
            models_and_predictions=models,
            ensemble_name="ProfileTest",
            train_target=train_target,
            val_target=val_target,
            test_target=test_target,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            plot_file=plot_file,
            verbose=False,
            uncertainty_quantile=0,  # Disable to simplify profiling
        )
    finally:
        profiler.disable()
        # Cleanup temp dir
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)

    elapsed = time.perf_counter() - start

    print(f"\nTotal time: {elapsed:.2f}s")
    print(f"\n{'='*70}")
    print(f"Top {top_n} functions by cumulative time:")
    print(f"{'='*70}\n")

    # Print stats
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative')
    stats.print_stats(top_n)
    print(stream.getvalue())

    return result, elapsed


def profile_by_ensemble_type(n_samples: int, n_models: int = 3, with_charts: bool = False):
    """Profile each ensemble method separately to compare performance.

    Args:
        n_samples: Number of samples to use
        n_models: Number of models to ensemble
        with_charts: If True, generate charts
    """
    import tempfile
    import shutil

    print(f"\n{'='*70}")
    print(f"Profiling by ensemble type")
    print(f"  Samples: {n_samples:,}")
    print(f"  Models: {n_models}")
    print(f"  Methods: {SIMPLE_ENSEMBLING_METHODS}")
    print(f"{'='*70}\n")

    # Create mock data once
    print("Creating mock data...")
    start = time.perf_counter()
    models, n_train, n_val, n_test = create_mock_model_results(n_samples, n_models)
    train_target, val_target, test_target = create_mock_targets(n_train, n_val, n_test)
    print(f"  Mock data created in {time.perf_counter() - start:.2f}s")
    print(f"  Train: {n_train:,}, Val: {n_val:,}, Test: {n_test:,}")

    # Create indices
    train_idx = np.arange(n_train)
    val_idx = np.arange(n_val)
    test_idx = np.arange(n_test)

    # Create dummy DataFrames
    train_df = pd.DataFrame({'dummy': np.zeros(n_train)})
    val_df = pd.DataFrame({'dummy': np.zeros(n_val)})
    test_df = pd.DataFrame({'dummy': np.zeros(n_test)})

    # Setup chart output directory
    temp_dir = None
    plot_file = None
    if with_charts:
        temp_dir = tempfile.mkdtemp(prefix="profile_ensembling_")
        plot_file = temp_dir + "/"

    results = {}

    # Profile each ensemble method separately
    for method in SIMPLE_ENSEMBLING_METHODS:
        print(f"\n--- Profiling '{method}' ensemble ---")

        start = time.perf_counter()
        try:
            _ = score_ensemble(
                models_and_predictions=models,
                ensemble_name=f"Profile_{method}",
                train_target=train_target,
                val_target=val_target,
                test_target=test_target,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                plot_file=plot_file,
                verbose=False,
                uncertainty_quantile=0,
                ensembling_methods=[method],  # Only this method
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            results[method] = -1
            continue

        elapsed = time.perf_counter() - start
        results[method] = elapsed
        print(f"  Time: {elapsed:.2f}s")

    # Cleanup
    if temp_dir:
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: Time per ensemble method")
    print(f"{'='*70}")

    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    total = sum(v for v in results.values() if v > 0)

    for method, elapsed in sorted_results:
        if elapsed > 0:
            pct = (elapsed / total) * 100
            print(f"  {method:10s}: {elapsed:7.2f}s ({pct:5.1f}%)")
        else:
            print(f"  {method:10s}: ERROR")

    print(f"  {'TOTAL':10s}: {total:7.2f}s")

    return results


def main():
    parser = argparse.ArgumentParser(description="Profile score_ensemble function")
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1_000_000,
        help="Number of samples (default: 1,000,000)"
    )
    parser.add_argument(
        "--n_models",
        type=int,
        default=3,
        help="Number of models to ensemble (default: 3)"
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=50,
        help="Number of top functions to show (default: 50)"
    )
    parser.add_argument(
        "--with-charts",
        action="store_true",
        help="Enable chart generation (slower but more realistic)"
    )
    parser.add_argument(
        "--by-type",
        action="store_true",
        help="Profile each ensemble method separately"
    )
    args = parser.parse_args()

    if args.by_type:
        profile_by_ensemble_type(
            n_samples=args.n_samples,
            n_models=args.n_models,
            with_charts=args.with_charts,
        )
    else:
        profile_score_ensemble(
            n_samples=args.n_samples,
            n_models=args.n_models,
            top_n=args.top_n,
            with_charts=args.with_charts,
        )


if __name__ == "__main__":
    main()
