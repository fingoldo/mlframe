"""Bench: how to feed a text column to ``TfidfVectorizer.fit_transform`` in ``apply_preprocessing_extensions``.

A2-05: the current path does ``train[col].fillna("").astype(str).values`` -- materialising a dense object ndarray. Compare three feed paths:

  A. ``.values``                          (current)
  B. ``.to_numpy(dtype=object, na_value="")`` (skip the .fillna pass; numpy handles NA fill)
  C. pass the pandas Series directly        (TfidfVectorizer iterates any iterable of strings)

Run: ``python -m mlframe.training._benchmarks.bench_tfidf_input_path``
Verdict (see ``_results/tfidf_input_path.json``): kept the ``.values`` path -- no measurable win; details in the JSON + the CHANGELOG row.
"""
from __future__ import annotations

import json
import os
from timeit import timeit

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def _make_series(n: int, frac_na: float = 0.1) -> pd.Series:
    rng = np.random.default_rng(0)
    vocab = [f"tok{i}" for i in range(200)]
    rows = []
    for _ in range(n):
        k = rng.integers(3, 12)
        rows.append(" ".join(rng.choice(vocab, size=k)))
    s = pd.Series(rows, dtype=object)
    na_idx = rng.choice(n, size=int(n * frac_na), replace=False)
    s.iloc[na_idx] = None
    return s


def _path_values(s: pd.Series):
    vec = TfidfVectorizer(max_features=500, ngram_range=(1, 1))
    return vec.fit_transform(s.fillna("").astype(str).values)


def _path_to_numpy(s: pd.Series):
    vec = TfidfVectorizer(max_features=500, ngram_range=(1, 1))
    return vec.fit_transform(s.astype(str).to_numpy(dtype=object, na_value=""))


def _path_series(s: pd.Series):
    vec = TfidfVectorizer(max_features=500, ngram_range=(1, 1))
    return vec.fit_transform(s.fillna("").astype(str))


def main() -> dict:
    results = {}
    for n in (5_000, 50_000):
        s = _make_series(n)
        reps = 5
        results[str(n)] = {
            "values_ms": 1000 * timeit(lambda: _path_values(s), number=reps) / reps,
            "to_numpy_ms": 1000 * timeit(lambda: _path_to_numpy(s), number=reps) / reps,
            "series_ms": 1000 * timeit(lambda: _path_series(s), number=reps) / reps,
        }
    out_dir = os.path.join(os.path.dirname(__file__), "_results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "tfidf_input_path.json"), "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(json.dumps(results, indent=2, sort_keys=True))
    return results


if __name__ == "__main__":
    main()
