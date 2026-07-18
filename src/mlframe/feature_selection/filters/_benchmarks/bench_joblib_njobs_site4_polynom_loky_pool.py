"""Measurement-only bench for joblib site 4: ``polynom_pair_fe.py:482``'s loky CPU pool
running ``_eval_one_pair`` (per-pair Optuna/CMA-ES Hermite polynomial search) via
``run_polynom_pair_fe``.

Gated by ``_PARALLEL_PAIR_THRESHOLD = 16`` pairs (``polynom_pair_fe.py:433``). Uses the
default ``fe_optimizer="cupy_kernel"`` is NOT exercised here (GPU); this bench targets the
CPU-only loky branch as the 7 sites all concern CPU joblib overhead. Sweeps
``fe_smart_polynom_iters`` (default production value context: docstring
``_joblib_safe.py:93`` cites "100 trials x 5 restarts x ~11 hard pairs") and n_pairs at/around
the 16-pair threshold plus a larger pool.

Run: PYTHONPATH=src python src/mlframe/feature_selection/filters/_benchmarks/bench_joblib_njobs_site4_polynom_loky_pool.py
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import polynom_pair_fe as ppf


def _make_data(n, n_cols, seed=0):
    rng = np.random.default_rng(seed)
    cols = [f"c{i}" for i in range(n_cols)]
    X = pd.DataFrame({c: rng.standard_normal(n) for c in cols})
    a, b = X["c0"].values, X["c1"].values
    logit = 1.5 * (a * a - b * b) + 0.5 * X["c2"].values
    y = (1.0 / (1.0 + np.exp(-logit)) > rng.random(n)).astype(np.int64)
    return X, cols, y


def _run(n_jobs, n_cols, fe_smart_polynom_iters, n=5000, opt_steps=15):
    X, cols, y = _make_data(n, n_cols)
    n_cols_ = len(cols)
    pairs = [(i, j) for i in range(n_cols_) for j in range(i + 1, n_cols_)]
    prospective_pairs = {((i, j), 0.5): None for (i, j) in pairs}
    eng_feats: set[str] = set()
    eng_recipes: dict[str, Any] = {}
    herm: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    ppf.run_polynom_pair_fe(
        X=X, is_polars_input=False, prospective_pairs=prospective_pairs,
        classes_y=y, cols=list(cols), nbins=np.full(n_cols_, 8, dtype=np.int64),
        data=X.values.copy(), engineered_features=eng_feats,
        engineered_recipes=eng_recipes, hermite_features_list=herm,
        feature_names_in=list(cols),
        fe_smart_polynom_iters=fe_smart_polynom_iters, fe_smart_polynom_optimization_steps=opt_steps,
        fe_min_polynom_degree=1, fe_max_polynom_degree=4,
        fe_min_polynom_coeff=-3.0, fe_max_polynom_coeff=3.0,
        fe_min_engineered_mi_prevalence=0.1, fe_hermite_l2_penalty=0.0,
        fe_polynomial_basis="hermite", fe_mi_estimator="plugin",
        fe_optimizer="cma", fe_warm_start=False, fe_multi_fidelity=False,
        quantization_nbins=8, quantization_method="quantile",
        quantization_dtype=np.int16, n_jobs=n_jobs, verbose=0,
        subsample_n=0,
    )
    return time.perf_counter() - t0


def main():
    for n_cols, fe_iters, label in (
        (6, 1, "below threshold (C(6,2)=15 pairs < 16 -- forced serial by code)"),
        (7, 1, "just above threshold (C(7,2)=21 pairs)"),
        (9, 2, "larger pool (C(9,2)=36 pairs, fe_smart_polynom_iters=2)"),
    ):
        print(f"\n=== {label} ===")
        _run(1, n_cols, fe_iters)  # warm numba/imports
        t1 = _run(1, n_cols, fe_iters)
        for n_jobs in (2, 4):
            _run(n_jobs, n_cols, fe_iters)  # warm pool spawn
            t = _run(n_jobs, n_cols, fe_iters)
            print(f"n_jobs=1: {t1:6.2f}s   n_jobs={n_jobs}: {t:6.2f}s   speedup={t1/t:.2f}x")


if __name__ == "__main__":
    main()
