"""DEVICE-BORN OOF binned-aggregate parity (2026-06-30).

Asserts the device-born OOF binned-aggregate matrix reproduces the host ``fit_binned_numeric_agg`` feat columns:
* STRUCTURAL bit-exact: the fold-id vector, the per-row substitution mask (where the global fallback applies),
  and which rows took a per-cell value vs the global are IDENTICAL host vs device;
* per-cell moment VALUES match within ~1e-10 (raw-moment ULP, the approved selection-equivalent trade);
* the resulting per-column MI ranking / survivor selection is identical.

Skips when cupy is unavailable (CI without a GPU)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._binned_numeric_agg_fe import (
    SUPPORTED_STATS,
    fit_binned_numeric_agg,
)
from mlframe.feature_selection.filters._binned_numeric_agg_resident import (
    binagg_fold_ids,
    build_binagg_oof_matrix_gpu,
)


def _make_frame(seed: int = 7, n: int = 8000):
    rng = np.random.default_rng(seed)
    g = rng.uniform(0.0, 1.0, n)
    sigma = 0.5 + 2.0 * np.abs(g - 0.5)
    aux = rng.normal(0.0, sigma, n)
    g2 = rng.uniform(-2.0, 2.0, n)
    X = pd.DataFrame({"g": g, "aux": aux, "g2": g2})
    y = sigma
    return X, y, sigma


def _col_specs_from_recipes(feat_df, recipes):
    return [
        {
            "name": c,
            "group_col": recipes[c]["group_col"],
            "agg_col": recipes[c]["agg_col"],
            "stat": recipes[c]["stat"],
            "edges": recipes[c]["edges"],
            "global": recipes[c]["global"],
        }
        for c in feat_df.columns
    ]


def test_foldids_bit_identical_to_host():
    # The resident fold ids MUST reproduce fit_binned_numeric_agg's host RNG fold assignment EXACTLY.
    n, n_folds, rs = 8000, 5, 0
    rng = np.random.default_rng(rs)
    host = np.empty(n, dtype=np.int64)
    host[rng.permutation(n)] = np.arange(n) % n_folds
    dev = binagg_fold_ids(n, n_folds, rs)
    np.testing.assert_array_equal(host, dev)


def test_device_oof_matrix_matches_host_columns():
    X, y, _ = _make_frame()
    n_folds, rs = 5, 0
    feat_df, recipes = fit_binned_numeric_agg(
        X,
        y,
        group_num_cols=["g", "g2"],
        agg_num_cols=["aux"],
        stats=SUPPORTED_STATS,
        nbins_base=10,
        n_folds=n_folds,
        random_state=rs,
    )
    assert feat_df.shape[1] > 0
    col_specs = _col_specs_from_recipes(feat_df, recipes)
    fold_ids = binagg_fold_ids(len(X), n_folds, rs)
    mat = build_binagg_oof_matrix_gpu(cp, X, col_specs, fold_ids, n_folds)
    dev = cp.asnumpy(mat)
    host = feat_df.to_numpy(dtype=np.float64)
    assert dev.shape == host.shape
    # ULP-level value parity (raw-moment from bincount vs numpy njit accumulator).
    maxdiff = float(np.max(np.abs(dev - host)))
    assert maxdiff < 1e-10, f"device OOF columns differ from host by {maxdiff} (> 1e-10 ULP band)"


def test_structural_fold_gather_fallback_bit_exact():
    """The STRUCTURE -- which rows took the global fallback vs a per-cell value -- must be byte-exact, even
    though the per-cell moment values are only ULP-close. Reconstruct the host's per-column fallback mask and
    assert the device reproduces the SAME rows-at-global set per column."""
    X, y, _ = _make_frame(seed=11)
    n_folds, rs = 5, 0
    feat_df, recipes = fit_binned_numeric_agg(
        X,
        y,
        group_num_cols=["g", "g2"],
        agg_num_cols=["aux"],
        stats=SUPPORTED_STATS,
        nbins_base=10,
        n_folds=n_folds,
        random_state=rs,
    )
    col_specs = _col_specs_from_recipes(feat_df, recipes)
    fold_ids = binagg_fold_ids(len(X), n_folds, rs)
    mat = cp.asnumpy(build_binagg_oof_matrix_gpu(cp, X, col_specs, fold_ids, n_folds))
    host = feat_df.to_numpy(dtype=np.float64)
    for j, spec in enumerate(col_specs):
        g = float(spec["global"])
        # Rows sitting EXACTLY on the global constant = the fallback set (init + empty-cell substitution).
        host_at_global = host[:, j] == g
        dev_at_global = mat[:, j] == g
        np.testing.assert_array_equal(
            host_at_global,
            dev_at_global,
            err_msg=f"column {spec['name']}: device fallback-mask differs from host (structural)",
        )


def test_resident_gate_selection_identical_to_host():
    """The device-born resident gate must return the SAME survivor set as the host local_mi_gate."""
    import os

    os.environ["MLFRAME_FE_GPU_STRICT"] = "1"
    os.environ["MLFRAME_FE_GPU_STRICT_RESIDENT"] = "1"
    os.environ["MLFRAME_CMI_GPU"] = "1"
    from mlframe.feature_selection.filters._binned_numeric_agg_resident import local_mi_gate_binagg_resident
    from mlframe.feature_selection.filters._unified_fe_gate import local_mi_gate

    X, y, _ = _make_frame(seed=13, n=6000)
    n_folds, rs = 5, 0
    feat_df, recipes = fit_binned_numeric_agg(
        X,
        y,
        group_num_cols=["g", "g2"],
        agg_num_cols=["aux"],
        stats=SUPPORTED_STATS,
        nbins_base=10,
        n_folds=n_folds,
        random_state=rs,
    )
    # Continuous y -> _coerce_y_classes quantile-bins it; pass the same y to both gates.
    host_keep = set(local_mi_gate(feat_df, y, raw_X=X))
    dev_keep = local_mi_gate_binagg_resident(
        feat_df,
        y,
        raw_X=X,
        recipes=recipes,
        n_folds=n_folds,
        random_state=rs,
    )
    assert dev_keep is not None, "resident gate returned None (GPU path unavailable under STRICT)"
    assert set(dev_keep) == host_keep, f"device survivors {set(dev_keep)} != host {host_keep}"
