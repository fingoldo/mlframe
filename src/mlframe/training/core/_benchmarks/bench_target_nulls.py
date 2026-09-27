"""Targets with missing labels: one suite call with ``target_null_policy="drop_rows"`` against an outer loop.

The outer loop is what a user did before: for each target, drop its unlabelled rows and call the suite on that target
alone. Both train the same targets; the in-suite run shares the split, the preprocessing and the frames across targets.

Usage::

    python -m mlframe.training.core._benchmarks.bench_target_nulls --rows 200000 --features 60 [--profile out.prof]

Prints wall time, peak private commit (Windows; RSS elsewhere) and each target's test RMSE for both modes.
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import tempfile
import threading
import time
from typing import Callable

import numpy as np
import pandas as pd

# (name, share of rows without a label, mask seed): targets sharing a seed share their missing rows (one row group).
TARGETS = (("full_a", 0.0, None), ("full_b", 0.0, None), ("g1_a", 0.3, 1), ("g1_b", 0.3, 1), ("g1_c", 0.3, 1), ("g2_a", 0.6, 2), ("g2_b", 0.6, 2), ("g3_a", 0.15, 3))


def make_frame(n_rows: int, n_features: int, seed: int = 0) -> pd.DataFrame:
    """Features plus one regression target per TARGETS entry, each a different linear mix, with its rows missing."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, n_features)).astype(np.float32)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    for k, (name, missing_share, mask_seed) in enumerate(TARGETS):
        w = rng.normal(size=5)
        y = X[:, k % n_features : k % n_features + 5] @ w[: min(5, n_features - k % n_features)] + rng.normal(0, 0.3, n_rows)
        if mask_seed is not None:
            y[np.random.default_rng(mask_seed).random(n_rows) < missing_share] = np.nan
        df[name] = y
    return df


class PeakMemory:
    """Samples this process's private commit (RSS where unavailable) every 50 ms; ``peak_gb`` after ``stop``."""

    def __init__(self) -> None:
        import psutil

        self._proc = psutil.Process()
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> int:
        info = self._proc.memory_info()
        return int(getattr(info, "private", 0) or info.rss)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self._sample())
            time.sleep(0.05)

    def __enter__(self) -> "PeakMemory":
        self.peak = self._sample()
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()

    @property
    def peak_gb(self) -> float:
        return self.peak / 1024**3


def _suite(df: pd.DataFrame, targets: list, data_dir: str, policy: str, quiet: bool = False) -> dict:
    """One suite call on ``targets``, LightGBM only, optional analyses off; returns ``{target: test RMSE}``.

    ``quiet=True`` also turns off BaselineDiagnostics and every post-fit diagnostic render, isolating the row-narrowing
    machinery's own cost from the (much larger, pre-existing, unrelated to missing labels) chart/diagnostics cost.
    """
    from mlframe.training.configs import (
        BaselineDiagnosticsConfig, CompositeTargetDiscoveryConfig, DummyBaselinesConfig, OutputConfig, ReportingConfig, TargetTypes,
        TrainingBehaviorConfig,
    )
    from mlframe.training.core import train_mlframe_models_suite
    from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor

    reporting_kwargs = dict(show_perf_chart=False, show_fi=False)
    extra: dict = {}
    if quiet:
        reporting_kwargs.update(
            adversarial_validation=False, interaction_strength_charts=False, engineered_separability_charts=False,
            class_structure_charts=False, category_discriminability_charts=False,
        )
        extra["baseline_diagnostics_config"] = BaselineDiagnosticsConfig(enabled=False)
        extra["dummy_baselines_config"] = DummyBaselinesConfig(plot_strongest=False)
    models, _ = train_mlframe_models_suite(
        df=df,
        target_name="bench",
        model_name="m",
        features_and_targets_extractor=SimpleFeaturesAndTargetsExtractor(regression_targets=targets),
        mlframe_models=["lgb"],
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        behavior_config=TrainingBehaviorConfig(prefer_gpu_configs=False, target_null_policy=policy),
        composite_target_discovery_config=CompositeTargetDiscoveryConfig(enabled=False),
        reporting_config=ReportingConfig(**reporting_kwargs),
        output_config=OutputConfig(data_dir=data_dir, models_dir="models"),
        verbose=0,
        **extra,
    )[:2]
    out = {}
    for name, entries in models[TargetTypes.REGRESSION].items():
        entry = entries[0] if isinstance(entries, list) else entries
        out[name] = float((entry.metrics.get("test") or {}).get("RMSE", float("nan")))
    return out


def run_in_suite(df: pd.DataFrame, quiet: bool = False) -> dict:
    """Every target in one call, each trained on its labelled rows."""
    with tempfile.TemporaryDirectory() as d:
        return _suite(df, [t[0] for t in TARGETS], d, "drop_rows", quiet=quiet)


def run_outer_loop(df: pd.DataFrame) -> dict:
    """One call per target on the frame reduced to that target's labelled rows."""
    out = {}
    names = [t[0] for t in TARGETS]
    for name in names:
        subset = df[df[name].notna()].drop(columns=[n for n in names if n != name]).reset_index(drop=True)
        with tempfile.TemporaryDirectory() as d:
            out.update(_suite(subset, [name], d, "raise"))
    return out


def measure(label: str, fn: Callable[[pd.DataFrame], dict], df: pd.DataFrame, profile: str | None = None) -> dict:
    """Run ``fn`` once, timed and memory-sampled, optionally under cProfile; print and return the result row."""
    profiler = cProfile.Profile() if profile else None
    with PeakMemory() as mem:
        t0 = time.perf_counter()
        if profiler:
            profiler.enable()
        rmse = fn(df)
        if profiler:
            profiler.disable()
        wall = time.perf_counter() - t0
    if profiler:
        profiler.dump_stats(profile)
    print(f"{label:>10}: wall {wall:7.1f}s  peak {mem.peak_gb:5.2f} GB  " + "  ".join(f"{k}={v:.4f}" for k, v in sorted(rmse.items())), flush=True)
    return {"wall": wall, "peak_gb": mem.peak_gb, "rmse": rmse}


def top_hotspots(profile: str, n: int = 40, needle: str = "mlframe") -> None:
    """Print the ``n`` largest own-time (tottime) functions from ``needle`` files in a saved profile."""
    stats = pstats.Stats(profile)
    rows = [(v[2], v[3], k) for k, v in stats.stats.items() if needle in k[0]]
    for tottime, cumtime, (file, line, func) in sorted(rows, reverse=True)[:n]:
        print(f"{tottime:8.2f} {cumtime:8.2f}  {file.split(needle)[-1]}:{line} {func}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--features", type=int, default=60)
    parser.add_argument("--modes", default="suite,outer")
    parser.add_argument("--profile", default=None, help="write a cProfile of the in-suite run here and print its mlframe hotspots")
    parser.add_argument("--quiet", action="store_true", help="turn off BaselineDiagnostics and post-fit diagnostic charts (isolates the row-narrowing cost)")
    args = parser.parse_args()
    df = make_frame(args.rows, args.features)
    modes = args.modes.split(",")
    if "suite" in modes:
        measure("drop_rows", lambda d: run_in_suite(d, quiet=args.quiet), df, args.profile)
        if args.profile:
            top_hotspots(args.profile)
    if "outer" in modes:
        measure("outer", run_outer_loop, df)


if __name__ == "__main__":
    main()
