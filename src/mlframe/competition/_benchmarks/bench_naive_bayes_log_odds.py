"""cProfile benchmark for ``mlframe.competition.naive_bayes_log_odds``.

COMPETITION/EXPLORATORY USE ONLY -- see ``mlframe.competition`` package docstring.

Run directly: ``python -m mlframe.competition._benchmarks.bench_naive_bayes_log_odds``
"""
from __future__ import annotations

import functools

import numpy as np

from mlframe.competition._benchmarks._cprofile_bench_shared import cprofile_main
from mlframe.competition.naive_bayes_log_odds import NaiveBayesLogOddsEnsembler


def _make_dataset(n: int, n_features: int, n_informative: int, seed: int):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    X = np.empty((n, n_features))
    for i in range(n_features):
        mean = np.where(y == 1, 0.7, -0.7) if i < n_informative else np.zeros(n)
        X[:, i] = mean + rng.normal(0, 1.0, size=n)
    return X, y


def _run_once() -> None:
    for n, n_features, n_informative in [(2_000, 20, 3), (5_000, 50, 5), (8_000, 80, 3)]:
        X_train, y_train = _make_dataset(n, n_features, n_informative, seed=0)
        X_test, _ = _make_dataset(n, n_features, n_informative, seed=1)
        ens = NaiveBayesLogOddsEnsembler(calibrate=False)
        ens.fit(X_train, y_train)
        ens.predict_proba(X_test)
        ens.predict_proba_average_baseline(X_test)


main = functools.partial(cprofile_main, _run_once)


if __name__ == "__main__":
    main()
