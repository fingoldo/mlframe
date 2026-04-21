"""Shared helpers: synth data, timing, metrics, leakage scorers."""
from __future__ import annotations

import gc
import time
import tracemalloc
from dataclasses import dataclass, asdict
from typing import Any, Callable

import numpy as np
import polars as pl


@dataclass
class Timing:
    name: str
    seconds: float
    peak_mem_mb: float
    result_hash: str | None = None

    def dict(self) -> dict[str, Any]:
        return asdict(self)


def time_and_mem(fn: Callable[..., Any], name: str, *args, repeats: int = 3, **kwargs) -> Timing:
    """Run fn repeats times, return median time + peak memory."""
    durations = []
    peak = 0.0
    last = None
    for _ in range(repeats):
        gc.collect()
        tracemalloc.start()
        t0 = time.perf_counter()
        last = fn(*args, **kwargs)
        durations.append(time.perf_counter() - t0)
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak = max(peak, peak_bytes / (1024 * 1024))
    return Timing(name=name, seconds=float(np.median(durations)), peak_mem_mb=peak)


def make_numeric_data(
    n: int = 10_000,
    n_features: int = 20,
    missing_rate: float = 0.05,
    outlier_rate: float = 0.02,
    seed: int = 0,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float64)
    mask = rng.random((n, n_features)) < missing_rate
    X[mask] = np.nan
    out_mask = rng.random((n, n_features)) < outlier_rate
    X[out_mask] = X[out_mask] + rng.choice([-1, 1], size=out_mask.sum()) * 50
    y = (X[:, :3].sum(axis=1) + rng.standard_normal(n) * 0.5 > 0).astype(np.int8)
    cols = {f"f{i}": X[:, i] for i in range(n_features)}
    cols["y"] = y
    return pl.DataFrame(cols)


def make_high_card_cat(
    n: int = 10_000,
    n_cat_cols: int = 3,
    cardinality: int = 200,
    signal_strength: float = 0.6,
    seed: int = 0,
) -> pl.DataFrame:
    """Synthetic data with high-cardinality categoricals that strongly predict y.

    Used to expose target-leakage in naive target/woe encoders: fit-on-train
    without OOF will memorize per-category mean y from train itself, which
    inflates training metric while offering nothing on holdout.
    """
    rng = np.random.default_rng(seed)
    cat_effects = {}
    data: dict[str, Any] = {}
    logit = np.zeros(n)
    for c in range(n_cat_cols):
        cats = rng.integers(0, cardinality, size=n)
        effect = rng.standard_normal(cardinality) * signal_strength
        cat_effects[f"c{c}"] = effect
        logit += effect[cats]
        data[f"c{c}"] = [f"cat_{v}" for v in cats]
    noise = rng.standard_normal((n, 5))
    for i in range(5):
        data[f"n{i}"] = noise[:, i]
    logit += noise[:, 0] * 0.1
    p = 1.0 / (1.0 + np.exp(-logit))
    data["y"] = (rng.random(n) < p).astype(np.float64)
    return pl.DataFrame(data)


def train_test_split_frame(df: pl.DataFrame, frac: float = 0.7, seed: int = 42) -> tuple[pl.DataFrame, pl.DataFrame]:
    df2 = df.with_row_index("__idx").sample(fraction=1.0, seed=seed, shuffle=True)
    n_train = int(len(df2) * frac)
    train = df2.slice(0, n_train).drop("__idx")
    test = df2.slice(n_train, len(df2) - n_train).drop("__idx")
    return train, test


def auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, y_score))


def save_result(path: str, payload: dict[str, Any]) -> None:
    import json
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str, sort_keys=True)
