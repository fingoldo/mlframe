"""A polynom pair whose X column assignment raises must leave no trace in the survivor lists.

``run_polynom_pair_fe`` used to append the binned column to ``data``/``cols``/``nbins`` before the
``X[_new_col_name] = _t_vals`` assignment; a raise there then left a column in the survivor lists with no
matching X column and no recipe. The assignment now runs first. This test makes the assignment fail for
every engineered polynom column and checks that the returned survivor lists still match the original
columns, so a regression to the old order fails here instead of in a silent column mismatch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import polynom_pair_fe as ppf


class _RejectsPolynomAssignment(pd.DataFrame):
    """A frame whose engineered-column assignment raises, so the survivor commit has to be rolled back by ordering."""

    attempts: list = []

    @property
    def _constructor(self):
        """Keep the subclass through pandas operations."""
        return _RejectsPolynomAssignment

    def __setitem__(self, key, value):
        """Reject ``_polynom_*`` columns and record the attempt; other columns assign normally."""
        if isinstance(key, str) and key.startswith("_polynom_"):
            _RejectsPolynomAssignment.attempts.append(key)
            raise ValueError("simulated X assignment failure")
        super().__setitem__(key, value)


def test_failed_x_assignment_leaves_survivor_lists_untouched():
    """Every polynom survivor whose X assignment raises is absent from data, cols, nbins and the engineered sets."""
    _RejectsPolynomAssignment.attempts.clear()
    n, n_cols = 2000, 6
    rng = np.random.default_rng(1)
    base = rng.standard_normal((n, n_cols))
    X = _RejectsPolynomAssignment(pd.DataFrame(base, columns=[f"c{i}" for i in range(n_cols)]))
    cols = list(X.columns)
    # A circle in (c0, c1): the raw product does not explain it, so the polynom search has a survivor to try to commit.
    classes_y = ((base[:, 0] ** 2 + base[:, 1] ** 2) > 1.0).astype(np.int64)
    pairs = [(i, j) for i in range(n_cols) for j in range(i + 1, n_cols)]
    prospective_pairs = {((i, j), 0.5): None for (i, j) in pairs}

    data = np.column_stack([np.digitize(base[:, k], np.quantile(base[:, k], [0.25, 0.5, 0.75])) for k in range(n_cols)]).astype(np.int16)
    nbins = np.full(n_cols, 4, dtype=np.int64)
    engineered_features: set = set()
    engineered_recipes: dict = {}
    hermite_features_list: list = []

    out_data, out_nbins, out_cols, _ = ppf.run_polynom_pair_fe(
        X=X,
        is_polars_input=False,
        prospective_pairs=prospective_pairs,
        classes_y=classes_y,
        cols=cols,
        nbins=nbins,
        data=data,
        engineered_features=engineered_features,
        engineered_recipes=engineered_recipes,
        hermite_features_list=hermite_features_list,
        feature_names_in=cols,
        fe_smart_polynom_iters=1,
        fe_smart_polynom_optimization_steps=1,
        fe_min_polynom_degree=1,
        fe_max_polynom_degree=3,
        fe_min_polynom_coeff=-2.0,
        fe_max_polynom_coeff=2.0,
        fe_min_engineered_mi_prevalence=0.0,
        fe_hermite_l2_penalty=0.0,
        fe_polynomial_basis="hermite",
        fe_mi_estimator="plugin",
        fe_optimizer="cma",
        fe_warm_start=False,
        fe_multi_fidelity=False,
        quantization_nbins=4,
        quantization_method="quantile",
        quantization_dtype=np.int16,
        n_jobs=1,
        verbose=0,
    )

    assert _RejectsPolynomAssignment.attempts, "no polynom survivor reached the X assignment, so the rollback was not exercised"
    assert len(out_cols) == out_data.shape[1] == len(out_nbins), "survivor lists diverged: a column reached data/nbins without a commit"
    assert not any(name.startswith("_polynom_") for name in out_cols), "a rejected polynom column was committed to the survivor lists"
    assert not engineered_features, "a rejected polynom column was marked engineered"
    assert not hermite_features_list, "a rejected polynom column got a hermite feature record"
