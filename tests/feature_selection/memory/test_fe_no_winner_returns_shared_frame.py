"""FE entry points that engineer nothing must return a column-buffer-sharing frame, not a deep copy of the input."""
from __future__ import annotations

import importlib
import warnings

import numpy as np
import pandas as pd
import pytest

_PKG = "mlframe.feature_selection.filters."
_SPECS = [
    ("_hinge_basis_fe", "hybrid_hinge_fe_with_recipes"),
    ("_numeric_decompose_fe", "hybrid_numeric_decompose_fe"),
    ("_orthogonal_bootstrap_mi_fe", "hybrid_orth_mi_bootstrap_fe"),
    ("_orthogonal_cluster_basis_fe", "hybrid_orth_mi_cluster_basis_fe"),
    ("_orthogonal_cmim_fe", "hybrid_orth_mi_cmim_fe"),
    ("_orthogonal_copula_mi_fe", "hybrid_orth_mi_copula_fe"),
    ("_orthogonal_dcor_fe", "hybrid_orth_mi_dcor_fe"),
    ("_orthogonal_diff_basis_fe", "hybrid_orth_mi_diff_basis_fe"),
    ("_orthogonal_elasticnet_fe", "hybrid_orth_mi_elasticnet_fe"),
    ("_orthogonal_hsic_fe", "hybrid_orth_mi_hsic_fe"),
    ("_orthogonal_jmim_fe", "hybrid_orth_mi_jmim_fe"),
    ("_orthogonal_ksg_mi_fe", "hybrid_orth_mi_ksg_fe"),
    ("_orthogonal_lasso_fe", "hybrid_orth_mi_lasso_fe"),
    ("_orthogonal_routing_fe", "hybrid_orth_mi_conditional_routing_fe"),
    ("_orthogonal_three_gate_mi_fe", "hybrid_orth_mi_three_gate_fe"),
    ("_orthogonal_total_correlation_fe", "hybrid_orth_mi_tc_fe"),
    ("_periodic_fe", "hybrid_modular_fe"),
    ("_wavelet_basis_fe_recipes", "hybrid_wavelet_fe_with_recipes"),
]


@pytest.mark.parametrize("module,fn_name", _SPECS)
def test_no_winner_return_shares_buffers_and_leaves_input_unchanged(module, fn_name):
    """top_k=0 engineers nothing: the result has the input's columns, shares its memory and the input is bit-identical afterwards."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((600, 5)), columns=list("abcde"))
    y = rng.integers(0, 2, 600)
    snap = X.copy(deep=True)
    fn = getattr(importlib.import_module(_PKG + module), fn_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_aug = fn(X, y, top_k=0)[0]
    assert list(X_aug.columns) == list(X.columns)
    assert np.shares_memory(X_aug.to_numpy(), X.to_numpy())
    pd.testing.assert_frame_equal(X, snap)


def test_missingness_augmented_frame_shares_buffers_and_adds_column_only_to_the_copy():
    """return_augmented=True shares the input's buffers; the new column exists on the result only."""
    from mlframe.feature_selection.filters._missingness_fe import missingness_count_with_recipes

    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.random((200, 3)), columns=list("abc"))
    X.iloc[::7, 0] = np.nan
    snap = X.copy(deep=True)
    X_aug, names, _ = missingness_count_with_recipes(X, cols=list("abc"), return_augmented=True)
    assert names[0] in X_aug.columns and names[0] not in X.columns
    assert np.shares_memory(X_aug["b"].to_numpy(), X["b"].to_numpy())
    pd.testing.assert_frame_equal(X, snap)
