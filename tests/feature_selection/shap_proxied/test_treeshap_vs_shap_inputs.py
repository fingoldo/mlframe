"""Our TreeSHAP against the official shap for input containers, values on thresholds, row counts and tree-shape extremes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.feature_selection.shap_proxied.treeshap_vs_shap_common import (
    assert_matches_shap,
    fit_lgb,
    fit_xgb,
    make_data,
    ours,
    shap_ref,
    ts,
)

pl = pytest.importorskip("polars")
FITS = [("lgb", fit_lgb), ("xgb", fit_xgb)]


@pytest.fixture(params=["lgb", "xgb"])
def model_and_data(request):
    """A fitted small model of each family with its training matrix."""
    X, y = make_data()
    return dict(FITS)[request.param](X, y), X


def test_float64_input(model_and_data):
    """float64 input is cast to float32 for routing and matches shap scored on the same float32 values."""
    m, X = model_and_data
    assert_matches_shap(m, X.astype(np.float64), ref_X=X)


def test_float64_values_off_the_float32_grid():
    """float64 rows that are not float32-representable are routed after the cast, as shap's LightGBM and xgboost paths see them."""
    X, y = make_data()
    m = fit_lgb(X.astype(np.float64), y)
    Xd = X.astype(np.float64) * (1 + 1e-9)
    assert_matches_shap(m, Xd, ref_X=Xd.astype(np.float32))


def test_integer_input():
    """An integer matrix."""
    X, y = make_data()
    Xi = (X * 3).astype(np.int32)
    assert_matches_shap(fit_lgb(Xi, y), Xi)


def test_pandas_dataframe(model_and_data):
    """A DataFrame scored via ``.to_numpy()`` (as the dispatcher does) equals shap on the DataFrame."""
    m, X = model_and_data
    df = pd.DataFrame(X)
    if type(m).__module__.startswith("xgboost"):
        m = fit_xgb(X, make_data()[1])
    assert_matches_shap(m, df.to_numpy(), ref_X=X)


def test_polars_frame(model_and_data):
    """A polars frame converted to numpy."""
    m, X = model_and_data
    assert_matches_shap(m, pl.DataFrame(X).to_numpy(), ref_X=X)


def test_fortran_ordered(model_and_data):
    """Fortran-ordered input."""
    m, X = model_and_data
    assert_matches_shap(m, np.asfortranarray(X))


def test_non_contiguous_view(model_and_data):
    """A strided column view (every second column of a doubled matrix)."""
    m, X = model_and_data
    big = np.hstack([X, X])[:, ::2]
    assert not big.flags["C_CONTIGUOUS"]
    assert_matches_shap(m, big)


def test_row_slice_non_contiguous(model_and_data):
    """A reversed-row view."""
    m, X = model_and_data
    assert_matches_shap(m, X[::-2])


def test_single_row(model_and_data):
    """n=1."""
    m, X = model_and_data
    phi, _ = assert_matches_shap(m, X[:1])
    assert phi.shape == (1, X.shape[1])


def test_zero_rows_ours_is_empty_and_shap_cannot(model_and_data):
    """n=0: ours returns an empty (0, f) matrix; the official shap fails with an unrelated internal error (ZeroDivisionError / IndexError)."""
    m, X = model_and_data
    phi, _ = ours(m, X[:0])
    assert phi.shape == (0, X.shape[1])
    with pytest.raises((ZeroDivisionError, IndexError)):
        shap_ref(m, X[:0])


@pytest.mark.parametrize("which", ["numba", "path", "table"])
def test_values_exactly_on_thresholds(which):
    """Rows built from every split threshold of the model (float32-exact) route like shap for lightgbm (x <= t) and xgboost (x < t)."""
    X, y = make_data()
    for fit in (fit_lgb, fit_xgb):
        m = fit(X, y, n_estimators=5)
        ens = ts.extract_ensemble(m)
        thr = ens.thresholds[ens.features >= 0].astype(np.float32)
        feat = ens.features[ens.features >= 0]
        rows = np.tile(np.median(X, 0), (len(thr), 1)).astype(np.float32)
        rows[np.arange(len(thr)), feat] = thr
        assert_matches_shap(m, rows, which)


def test_adjacent_float32_values_around_a_threshold():
    """Rows with the float32 just below, at and just above each xgboost threshold."""
    X, y = make_data()
    m = fit_xgb(X, y, n_estimators=5)
    ens = ts.extract_ensemble(m)
    sel = ens.features >= 0
    thr, feat = ens.thresholds[sel].astype(np.float32), ens.features[sel]
    rows = []
    for t, f in zip(thr, feat):
        for v in (np.nextafter(t, np.float32(-np.inf)), t, np.nextafter(t, np.float32(np.inf))):
            r = np.median(X, 0).astype(np.float32)
            r[f] = v
            rows.append(r)
    assert_matches_shap(m, np.array(rows))


def test_constant_columns_in_scoring_data(model_and_data):
    """Scoring rows where one column is constant and another is all NaN."""
    m, X = model_and_data
    Xs = X.copy()
    Xs[:, 0] = 1.5
    Xs[:, 1] = np.nan
    assert_matches_shap(m, Xs)
