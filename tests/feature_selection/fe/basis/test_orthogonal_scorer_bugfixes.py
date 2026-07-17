"""Regression tests for the orthogonal-FE scorer bug cluster.

Each test pins one confirmed bug fix in the ``_orthogonal_*_fe`` scorer family:

1. near-zero-baseline uplift explosion (adaptive_degree / hsic / ksg / lasso)
2. silent continuous-y truncation in ``_coerce_y_*``
3. missing NaN guard before rankdata/subsample (copula / dcor)
4. int64 Horner-key overflow in ``_factorize_pack`` (cmim / total_correlation)
5. bare ``except Exception: return 0.0`` swallowing programming errors (meta_scorer)
6. blanket ``simplefilter("ignore")`` hiding ConvergenceWarning (elasticnet)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import _orthogonal_adaptive_degree_fe as adeg
from mlframe.feature_selection.filters import _orthogonal_bootstrap_mi_fe as boot
from mlframe.feature_selection.filters import _orthogonal_cmim_fe as cmim
from mlframe.feature_selection.filters import _orthogonal_copula_mi_fe as copula
from mlframe.feature_selection.filters import _orthogonal_dcor_fe as dcor
from mlframe.feature_selection.filters import _orthogonal_elasticnet_fe as enet
from mlframe.feature_selection.filters import _orthogonal_hsic_fe as hsic
from mlframe.feature_selection.filters import _orthogonal_ksg_mi_fe as ksg
from mlframe.feature_selection.filters import _orthogonal_lasso_fe as lasso
from mlframe.feature_selection.filters import _orthogonal_meta_scorer_fe as meta
from mlframe.feature_selection.filters import _orthogonal_total_correlation_fe as tc


# --------------------------------------------------------------------------- #
# Bug 1: near-zero-baseline uplift explosion                                   #
# --------------------------------------------------------------------------- #
def _near_zero_baseline_frames():
    """A constant source column -> baseline MI/|coef| ~= 0. The engineered
    column carries a faint amount of structure but no real signal. Pre-fix the
    uplift = emi / (baseline + 1e-12) ratio explodes to ~1e9 and passes the gate."""
    rng = np.random.default_rng(0)
    n = 400
    raw_X = pd.DataFrame({"a": np.ones(n)})  # constant -> zero baseline
    eng = pd.DataFrame({"a__hr2": rng.normal(size=n) * 1e-3})
    y = (rng.random(n) > 0.5).astype(np.int64)
    return raw_X, eng, y


def test_ksg_near_zero_baseline_uplift_not_exploded(monkeypatch):
    """Ksg near zero baseline uplift not exploded."""
    raw_X, eng, y = _near_zero_baseline_frames()
    # Force a genuinely near-zero baseline (KSG's noise floor would otherwise
    # estimate a constant source above eps) and a small engineered MI.
    monkeypatch.setattr(ksg, "_ksg_mi_batch", lambda X, *a, **k: np.full(X.shape[1], 1e-10))
    df = ksg.score_features_by_ksg_mi_uplift(raw_X, eng, y)
    up = float(df["uplift"].iloc[0])
    # Pre-fix this divided 1e-10 / (1e-10 + 1e-12) -> ~0.99 OR a huge ratio when
    # emi >> baseline; either way the +1e-12 path produces a finite ratio that
    # can pass the gate. Post-fix the near-zero baseline suppresses the ratio.
    assert up == 0.0 or up == float("inf"), up
    assert not (0.0 < up < 1e6)


def test_hsic_near_zero_baseline_uplift_not_exploded(monkeypatch):
    """Hsic near zero baseline uplift not exploded."""
    raw_X, eng, y = _near_zero_baseline_frames()
    calls = {"n": 0}

    def _fake_hsic(X, *a, **k):
        # First call = raw baseline (near zero), second = engineered (signal).
        """Fake hsic."""
        calls["n"] += 1
        return np.full(X.shape[1], 1e-10 if calls["n"] == 1 else 0.5)

    monkeypatch.setattr(hsic, "_hsic_batch", _fake_hsic)
    df = hsic.score_features_by_hsic_uplift(raw_X, eng, y)
    up = float(df["uplift"].iloc[0])
    assert up == 0.0 or up == float("inf"), up
    assert not (0.0 < up < 1e6)


def test_lasso_near_zero_baseline_uplift_not_exploded(monkeypatch):
    """Lasso near zero baseline uplift not exploded."""
    raw_X, eng, y = _near_zero_baseline_frames()

    def _fake_coefs(stack_arr, y_arr, **k):
        # raw col baseline ~0, engineered col carries weight.
        """Fake coefs."""
        return np.array([1e-10, 0.5])

    monkeypatch.setattr(lasso, "_fit_lasso_abs_coefs", _fake_coefs)
    df = lasso.score_features_by_lasso_coef(raw_X, eng, y)
    up = float(df["uplift"].iloc[0])
    assert up == 0.0 or up == float("inf"), up
    assert not (0.0 < up < 1e6)


def test_adaptive_degree_near_zero_baseline_does_not_pass_gate():
    # A constant source column produces an engineered column with no real
    # signal; pre-fix the exploded uplift cleared min_uplift and emitted it.
    """Adaptive degree near zero baseline does not pass gate."""
    rng = np.random.default_rng(1)
    n = 400
    X = pd.DataFrame({"a": np.ones(n), "b": rng.normal(size=n)})
    y = (rng.random(n) > 0.5).astype(np.int64)
    _eng, mlmeta = adeg.generate_adaptive_degree_basis_features(
        X,
        y,
        cols=["a"],
        degree_range=(2, 3),
        min_uplift=1.05,
    )
    # No engineered column off the constant source should survive via an
    # exploded uplift.
    for info in mlmeta.values():
        assert info["uplift"] != float("inf") or info["engineered_mi"] >= adeg._ABS_MI_FLOOR


# --------------------------------------------------------------------------- #
# Bug 2: silent continuous-y truncation                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "coerce",
    [
        adeg._coerce_y_classif,
        boot._coerce_y_int64,
        cmim._coerce_y_int64,
        tc._coerce_y_int64,
    ],
)
def test_coerce_y_does_not_truncate_float_labels(coerce):
    # Two distinct float labels inside [0, 1) collapse to class 0 under
    # plain .astype(int64) -- destroying a binary signal entirely.
    """Coerce y does not truncate float labels."""
    y = np.array([0.2, 0.8, 0.2, 0.8, 0.2, 0.8], dtype=np.float64)
    out = coerce(y)
    assert len(np.unique(out)) == 2, "distinct float labels must survive coercion"
    # The mapping must respect the value ordering (0.2 -> 0, 0.8 -> 1).
    assert out[0] != out[1]


def test_coerce_y_continuous_signal_preserved_for_mi():
    # A continuous-y signal correlated with x must remain detectable: pre-fix
    # all y in [0, 1) -> single class 0 -> MI(x; y) == 0.
    """Coerce y continuous signal preserved for mi."""
    rng = np.random.default_rng(2)
    n = 300
    x = rng.normal(size=n)
    y = 0.3 + 0.4 * (x > 0)  # values 0.3 / 0.7, both inside [0,1)
    out = cmim._coerce_y_int64(y)
    assert len(np.unique(out)) == 2


# --------------------------------------------------------------------------- #
# Bug 3: NaN guard before rankdata / subsample                                 #
# --------------------------------------------------------------------------- #
def test_copula_mi_nan_rows_dropped_not_ranked():
    # The fix must DROP non-finite rows, not feed them to rankdata (which would
    # assign NaN the largest rank -> a valid high uniform bin masquerading as
    # real signal). Proof: copula_mi on the NaN-injected array must equal
    # copula_mi on the explicitly-dropped subset. Pre-fix the NaN rows are
    # ranked and included, so the two values differ.
    """Copula mi nan rows dropped not ranked."""
    rng = np.random.default_rng(3)
    n = 500
    x = rng.normal(size=n)
    y = rng.integers(0, 3, size=n)
    x_nan = x.copy()
    nan_mask = np.zeros(n, dtype=bool)
    nan_mask[::7] = True
    x_nan[nan_mask] = np.nan
    mi_masked = copula.copula_mi(x_nan, y)
    mi_dropped = copula.copula_mi(x[~nan_mask], y[~nan_mask])
    assert np.isfinite(mi_masked)
    assert mi_masked == pytest.approx(mi_dropped, abs=1e-12)


def test_copula_batch_nan_masked():
    """Copula batch nan masked."""
    rng = np.random.default_rng(4)
    n = 400
    x = rng.normal(size=n)
    y = (x > 0).astype(np.int64)
    X = x.copy()
    X[:50] = np.nan
    out = copula._copula_mi_batch(X.reshape(-1, 1), y)
    assert np.isfinite(out).all()


def test_dcor_nan_guard():
    """Dcor nan guard."""
    rng = np.random.default_rng(5)
    n = 400
    x = rng.normal(size=n)
    y = x + rng.normal(size=n) * 0.1
    x_nan = x.copy()
    x_nan[:20] = np.nan
    d = dcor.distance_correlation(x_nan, y, n_sample=300)
    # Pre-fix, NaN poisons the distance matrices -> NaN dCor.
    assert np.isfinite(d)
    assert 0.0 <= d <= 1.0


# --------------------------------------------------------------------------- #
# Bug 4: int64 Horner overflow in _factorize_pack                              #
# --------------------------------------------------------------------------- #
def _overflow_cols():
    """Two columns whose Horner radix (cmax0 = 2**63) overflows int64: the
    distinct source rows (0, 2, 2**63-1) map their col0 contribution 2 * 2**63
    to a wrapped 0, colliding with the col0 == 0 row. Pre-fix the silent wrap
    merges distinct rows (corrupting the joint count multiset); post-fix the
    radix-product guard falls back to a sort-based renumber that cannot overflow."""
    big = (1 << 63) - 1
    col0 = np.array([0, 2, big], dtype=np.int64)  # cmax0 = 2**63
    col1 = np.array([5, 5, 5], dtype=np.int64)
    return col0, col1


def test_cmim_factorize_pack_overflow_no_silent_collision():
    """Cmim factorize pack overflow no silent collision."""
    col0, col1 = _overflow_cols()
    codes, k = cmim._factorize_pack(col0, col1)
    # Three distinct (col0, col1) rows -> exactly 3 classes.
    assert k == 3, k
    assert len(np.unique(codes)) == 3


def test_tc_factorize_pack_overflow_no_silent_collision():
    """Tc factorize pack overflow no silent collision."""
    col0, col1 = _overflow_cols()
    codes = tc._factorize_pack(col0, col1)
    assert len(np.unique(codes)) == 3


def test_factorize_pack_matches_reference_on_small_input():
    # The overflow fallback must produce the same count multiset as the
    # Horner path on a small (non-overflowing) input.
    """Factorize pack matches reference on small input."""
    a = np.array([0, 1, 0, 1, 2], dtype=np.int64)
    b = np.array([0, 0, 1, 1, 0], dtype=np.int64)
    _codes, k = cmim._factorize_pack(a, b)
    # 4 distinct (a, b) pairs: (0,0),(1,0),(0,1),(1,1),(2,0) -> 5.
    assert k == 5


# --------------------------------------------------------------------------- #
# Bug 5: meta_scorer must not swallow programming errors to 0.0                #
# --------------------------------------------------------------------------- #
def test_meta_scorer_propagates_programming_error():
    """Meta scorer propagates programming error."""
    rng = np.random.default_rng(6)
    n = 200
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = (rng.random(n) > 0.5).astype(np.int64)

    # A genuine programming error inside the inter_x_max_corr path
    # (AttributeError, not a numeric error) must propagate, not be coerced to
    # 0.0. That branch calls ``DataFrame.corr`` with no inner nan-catch.
    orig = pd.DataFrame.corr

    def _broken_corr(self, *a, **k):
        """Broken corr."""
        raise AttributeError("simulated programming error in corr")

    pd.DataFrame.corr = _broken_corr
    try:
        with pytest.raises(AttributeError):
            meta.fingerprint_signal(X, y)
    finally:
        pd.DataFrame.corr = orig


def test_meta_scorer_numeric_failure_still_yields_zero(caplog):
    # A numeric failure (legit "no signal") still degrades to 0.0 but logs.
    """Meta scorer numeric failure still yields zero."""
    rng = np.random.default_rng(7)
    n = 50
    X = pd.DataFrame({"a": rng.normal(size=n)})
    y = np.zeros(n, dtype=np.int64)  # constant y -> pearson path yields 0.0
    fp = meta.fingerprint_signal(X, y)
    assert fp["mean_abs_pearson"] == 0.0


# --------------------------------------------------------------------------- #
# Bug 6: elasticnet must surface ConvergenceWarning                            #
# --------------------------------------------------------------------------- #
def test_elasticnet_non_convergence_logged(caplog):
    """Elasticnet non convergence logged."""
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(8)
    # A pathological, heavily collinear design with near-zero regularization
    # does not converge even at max_iter=5000; pre-fix the blanket
    # simplefilter("ignore") swallowed the ConvergenceWarning silently.
    n, p = 80, 200
    base = rng.normal(size=(n, 5))
    X = np.column_stack([base[:, i % 5] + 1e-4 * rng.normal(size=n) for i in range(p)])
    y = rng.normal(size=n)
    with caplog.at_level(logging.WARNING, logger=enet.logger.name):
        enet._fit_elasticnet_abs_coefs(X, y, alpha=1e-9, l1_ratio=0.5, standardize=True)
    assert any("did not converge" in r.message for r in caplog.records), "non-convergence must be surfaced, not silently ignored"
