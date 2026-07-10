"""cProfile harness for ``training.composite.CountWeightedBlendEnsemble``.

Run: ``python -m mlframe.training.composite._benchmarks.bench_count_weighted_blend``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from mlframe.training.composite import CountWeightedBlendEnsemble


def _make_dataset(n_entities: int, seed: int):
    rng = np.random.default_rng(seed)
    counts = rng.integers(1, 4, n_entities)
    counts[: max(1, n_entities // 15)] = rng.integers(60, 100, max(1, n_entities // 15))
    entity_effect = rng.normal(scale=8.0, size=n_entities)
    rows = []
    for e in range(n_entities):
        n_obs = counts[e] + 5
        x = rng.normal(size=n_obs)
        y = x * 2.0 + entity_effect[e] + rng.normal(scale=12.0, size=n_obs)
        for i in range(n_obs):
            rows.append({"entity": float(e), "x": x[i], "y": y[i]})
    df = pd.DataFrame(rows)
    return df[["entity", "x"]], df["y"].to_numpy()


def _entity_pipeline():
    return make_pipeline(ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore"), ["entity"])], remainder="passthrough"), LinearRegression())


def _run(n_entities: int) -> None:
    X, y = _make_dataset(n_entities, seed=0)
    blend = CountWeightedBlendEnsemble(entity_estimator=_entity_pipeline(), global_estimator=LinearRegression(), entity_col="entity", metadata_cols=["x"], k=10.0)
    blend.fit(X, y)
    blend.predict(X)


if __name__ == "__main__":
    for n_entities in [100, 1_000, 5_000]:
        t0 = time.perf_counter()
        _run(n_entities)
        wall = time.perf_counter() - t0
        print(f"n_entities={n_entities:>6,} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(5_000)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
