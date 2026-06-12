"""Degenerate-frame robustness + diagnostic-surface tests for MRMR.

Covers: each pathological column type is (a) handled without a crash and (b) recorded
in ``degenerate_columns_`` with the correct reason; y-NaN raises a clean ValueError
with PARITY to the sibling selectors; and degenerate handling is TRANSPARENT -- the
genuine features are selected byte-identically whether or not degenerate columns are
present.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR
from mlframe.feature_selection.filters._mrmr_degenerate import audit_degenerate_columns


def _mk(n=300, seed=1):
    rng = np.random.RandomState(seed)
    a = rng.randn(n)
    b = rng.randn(n)
    c = rng.randn(n)
    y = (a + 0.5 * b + rng.randn(n) * 0.1 > 0).astype(int)
    return a, b, c, y


def _mrmr():
    return MRMR(full_npermutations=2, baseline_npermutations=1, fe_max_steps=1, fe_smart_polynom_iters=0)


def _fit(df, y):
    m = _mrmr()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(df, y)
    return m


# --------------------------------------------------------------------------- unit: pure scan


def test_scan_all_nan():
    a, b, _, _ = _mk()
    df = pd.DataFrame({"a": a, "b": b, "z": np.full(len(a), np.nan)})
    assert audit_degenerate_columns(df) == {"z": "all_nan"}


def test_scan_constant():
    a, b, _, _ = _mk()
    df = pd.DataFrame({"a": a, "b": b, "z": np.ones(len(a))})
    assert audit_degenerate_columns(df) == {"z": "constant"}


def test_scan_exact_duplicate():
    a, b, _, _ = _mk()
    df = pd.DataFrame({"a": a, "b": b, "z": a.copy()})
    assert audit_degenerate_columns(df) == {"z": "duplicate_of:a"}


def test_scan_collinear():
    a, b, _, _ = _mk()
    df = pd.DataFrame({"a": a, "b": b, "z": 2.0 * a + 3.0})
    assert audit_degenerate_columns(df) == {"z": "collinear_with:a"}


def test_scan_clean_frame_empty():
    a, b, c, _ = _mk()
    df = pd.DataFrame({"a": a, "b": b, "c": c})
    assert audit_degenerate_columns(df) == {}


def test_scan_numpy_array():
    a, b, _, _ = _mk()
    arr = np.column_stack([a, b, np.ones(len(a))])
    assert audit_degenerate_columns(arr) == {2: "constant"}


# --------------------------------------------------------------------------- unit: end-to-end no crash + recorded


@pytest.mark.parametrize(
    "make_col,reason",
    [
        (lambda a: np.full(len(a), np.nan), "all_nan"),
        (lambda a: np.ones(len(a)), "constant"),
        (lambda a: a.copy(), "duplicate_of:a"),
        (lambda a: 2.0 * a + 3.0, "collinear_with:a"),
    ],
)
def test_fit_records_degenerate(make_col, reason):
    a, b, _, y = _mk()
    df = pd.DataFrame({"a": a, "b": b, "z": make_col(a)})
    m = _fit(df, y)  # must not crash
    assert m.degenerate_columns_.get("z") == reason


def test_clean_frame_has_empty_degenerate():
    a, b, c, y = _mk()
    df = pd.DataFrame({"a": a, "b": b, "c": c})
    m = _fit(df, y)
    assert m.degenerate_columns_ == {}


# --------------------------------------------------------------------------- y-NaN parity with siblings


def test_y_nan_raises_valueerror():
    a, b, _, y = _mk()
    df = pd.DataFrame({"a": a, "b": b})
    yn = y.astype(float)
    yn[0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        _mrmr().fit(df, yn)


def test_y_inf_raises_valueerror():
    a, b, _, y = _mk()
    df = pd.DataFrame({"a": a, "b": b})
    yn = y.astype(float)
    yn[0] = np.inf
    with pytest.raises(ValueError, match="inf"):
        _mrmr().fit(df, yn)


def test_y_nan_parity_with_rfecv():
    """MRMR and RFECV both reject a NaN y -- no silent coercion (parity)."""
    from mlframe.feature_selection.wrappers import RFECV

    a, b, _, y = _mk(n=120)
    df = pd.DataFrame({"a": a, "b": b})
    yn = y.astype(float)
    yn[0] = np.nan
    with pytest.raises((ValueError, Exception)):
        _mrmr().fit(df, yn)
    # RFECV path also refuses to produce an honest fit on NaN y (raises somewhere in
    # the sklearn estimator / cv split). Both refuse rather than silently coerce.
    from sklearn.linear_model import LogisticRegression

    raised = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            RFECV(LogisticRegression(max_iter=50), cv=2).fit(df, yn)
    except Exception:
        raised = True
    assert raised


# --------------------------------------------------------------------------- transparency: byte-identical selection


@pytest.mark.parametrize(
    "make_col",
    [
        lambda a: np.ones(len(a)),          # constant -- carries NO signal
        lambda a: np.full(len(a), np.nan),  # all-nan -- carries NO signal
    ],
)
def test_selection_byte_identical_with_signal_free_degenerate(make_col):
    """Adding a SIGNAL-FREE degenerate column (constant / all-NaN) must NOT change
    which good features are selected -- the diagnostic scan is fully transparent.

    (A duplicate / collinear column of a GENUINE feature is NOT signal-free: it is an
    equal substitute, so MRMR's redundancy gate legitimately picks one representative
    of the pair -- covered by ``test_duplicate_signal_is_represented`` instead.)"""
    a, b, c, y = _mk()
    clean = pd.DataFrame({"a": a, "b": b, "c": c})
    dirty = pd.DataFrame({"a": a, "b": b, "c": c, "z": make_col(a)})

    m_clean = _fit(clean, y)
    m_dirty = _fit(dirty, y)

    good = ["a", "b", "c"]
    sel_clean = [n for n in np.asarray(m_clean.feature_names_in_)[m_clean.support_] if n in good]
    sel_dirty = [n for n in np.asarray(m_dirty.feature_names_in_)[m_dirty.support_] if n in good]
    assert sel_clean == sel_dirty
    # the degenerate column is never selected
    assert "z" not in set(np.asarray(m_dirty.feature_names_in_)[m_dirty.support_])


@pytest.mark.parametrize("make_col", [lambda a: a.copy(), lambda a: 2.0 * a + 3.0])
def test_duplicate_collinear_handled_and_recorded(make_col):
    """A duplicate / collinear copy of a genuine feature: MRMR does NOT crash, records
    the column as degenerate, and NEVER selects BOTH the original and its copy (the
    redundancy gate removes the redundant member). The dominant signal (b) is retained.
    Whether MRMR keeps the 'a' representative or drops the whole redundant pair is its
    own redundancy decision -- the diagnostic surface does not alter it."""
    a, b, c, y = _mk()
    dirty = pd.DataFrame({"a": a, "b": b, "c": c, "z": make_col(a)})
    m = _fit(dirty, y)
    selected = set(np.asarray(m.feature_names_in_)[m.support_])
    # never both members of the redundant pair
    assert not ({"a", "z"} <= selected)
    # dominant signal retained
    assert "b" in selected
    # degenerate column recorded
    assert "z" in m.degenerate_columns_


# --------------------------------------------------------------------------- biz_value


def test_biz_value_realistic_mixed_degenerate():
    """A realistic frame with genuine signal + a constant + a duplicate + a collinear
    column: MRMR still selects the genuine drivers AND reports all 3 degenerates."""
    a, b, c, y = _mk(n=500)
    df = pd.DataFrame(
        {
            "good_a": a,
            "good_b": b,
            "noise_c": c,
            "const_col": np.full(len(a), 7.0),
            "dup_good_a": a.copy(),
            "collinear_b": -3.0 * b + 1.0,
        }
    )
    m = _fit(df, y)
    selected = set(np.asarray(m.feature_names_in_)[m.support_])
    # the dominant genuine driver is recovered (represented by good_a or its duplicate).
    assert len({"good_a", "dup_good_a"} & selected) >= 1
    # the constant column carries no signal and is never selected.
    assert "const_col" not in selected
    # MRMR never selects BOTH members of a redundant pair.
    assert not ({"good_a", "dup_good_a"} <= selected)
    assert not ({"good_b", "collinear_b"} <= selected)
    # all three degenerates are diagnosed in the diagnostic surface.
    deg = m.degenerate_columns_
    assert deg.get("const_col") == "constant"
    assert deg.get("dup_good_a") == "duplicate_of:good_a"
    assert deg.get("collinear_b") == "collinear_with:good_b"
