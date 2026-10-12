"""The polynomial pair FE's engineered-MI prevalence gate must report its drops to the caller.

Before this fix the polynom reduce loop dropped a candidate under ``fe_min_engineered_mi_prevalence`` with a bare
``continue``, so the rejection ledger could not say which candidates the gate killed or by what margin.
"""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import polynom_pair_fe as ppf


def _run(rejection_records, prevalence):
    """Call the polynomial pair FE on a small synthetic product frame with the given engineered-MI prevalence."""
    n, n_cols = 400, 4
    rng = np.random.default_rng(0)
    X = pd.DataFrame({f"c{i}": rng.standard_normal(n) for i in range(n_cols)})
    cols = list(X.columns)
    classes_y = (X["c0"].to_numpy() * X["c1"].to_numpy() + 0.1 * rng.standard_normal(n) > 0).astype(np.int64)
    pairs = {((i, j), 0.5): None for i in range(n_cols) for j in range(i + 1, n_cols)}
    return ppf.run_polynom_pair_fe(
        X=X,
        is_polars_input=False,
        prospective_pairs=pairs,
        classes_y=classes_y,
        cols=cols,
        nbins=np.full(n_cols, 8, dtype=np.int64),
        data=X.values.copy(),
        engineered_features=set(),
        engineered_recipes={},
        hermite_features_list=[],
        feature_names_in=cols,
        fe_smart_polynom_iters=1,
        fe_smart_polynom_optimization_steps=1,
        fe_min_polynom_degree=1,
        fe_max_polynom_degree=3,
        fe_min_polynom_coeff=-2.0,
        fe_max_polynom_coeff=2.0,
        fe_min_engineered_mi_prevalence=prevalence,
        fe_hermite_l2_penalty=0.0,
        fe_polynomial_basis="hermite",
        fe_mi_estimator="plugin",
        fe_optimizer="cma",
        fe_warm_start=False,
        fe_multi_fidelity=False,
        quantization_nbins=8,
        quantization_method="quantile",
        quantization_dtype=np.int16,
        n_jobs=1,
        verbose=0,
        rejection_records=rejection_records,
        poly_cheap_skip_ratio=2.0,
    )


def test_polynom_engineered_prevalence_drops_are_reported():
    """A prevalence gate that rejects every candidate must hand each dropped candidate to the caller's record list."""
    records = []
    _run(records, prevalence=1e9)
    assert records, "a prevalence gate that rejects every candidate must report its drops, not drop them silently"
    for rec in records:
        assert rec["gate"] == "engineered_mi_prevalence"
        assert rec["reason"] == "polynom_mi_below_baseline_times_prevalence"
        assert rec["observed"] <= rec["threshold"]
