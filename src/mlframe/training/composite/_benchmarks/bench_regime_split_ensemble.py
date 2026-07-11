"""cProfile harness for ``training.composite.RegimeSplitEnsemble``.

Run: ``python -m mlframe.training.composite._benchmarks.bench_regime_split_ensemble``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from mlframe.training.composite import RegimeSplitEnsemble


def _regime_fn(X):
    trend = X["trend"].to_numpy() if hasattr(X, "columns") else np.asarray(X)[:, 0]
    return np.where(trend > 0.3, "bull", np.where(trend < -0.3, "bear", "stable"))


def _make_dataset(n: int, seed: int):
    rng = np.random.default_rng(seed)
    trend = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = trend * 2 + x2 + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({"trend": trend, "x2": x2}), y


def _regime_proba_fn(X, tau: float = 0.15):
    trend = X["trend"].to_numpy() if hasattr(X, "columns") else np.asarray(X)[:, 0]
    p_bull = 1.0 / (1.0 + np.exp(-(trend - 0.3) / tau))
    p_bear = 1.0 / (1.0 + np.exp((trend + 0.3) / tau))
    p_stable = np.clip(1.0 - p_bull - p_bear, 0.0, None)
    return {"bull": p_bull, "bear": p_bear, "stable": p_stable}


def _run(n: int, combine: Literal["route", "average", "blend"] = "route") -> None:
    X, y = _make_dataset(n, seed=0)
    kwargs = {"regime_proba_fn": _regime_proba_fn} if combine == "blend" else {}
    ensemble = RegimeSplitEnsemble(estimator_factory=lambda: LinearRegression(), regime_fn=_regime_fn, combine=combine, **kwargs)
    ensemble.fit(X, y)
    ensemble.predict(X)


if __name__ == "__main__":
    _combine_modes: list[Literal["route", "blend"]] = ["route", "blend"]
    for combine_mode in _combine_modes:
        for n in [1_000, 10_000, 100_000]:
            t0 = time.perf_counter()
            _run(n, combine=combine_mode)
            wall = time.perf_counter() - t0
            print(f"combine={combine_mode:<6} n={n:>7,} -> {wall * 1000:9.2f} ms")

    for combine_mode in _combine_modes:
        profiler = cProfile.Profile()
        profiler.enable()
        _run(100_000, combine=combine_mode)
        profiler.disable()
        buf = StringIO()
        stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
        stats.print_stats(15)
        print(f"--- combine={combine_mode} ---")
        print(buf.getvalue())
