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
    """Build a synthetic (a, b, c, y) fixture with a and b genuinely predictive of y."""
    rng = np.random.RandomState(seed)
    a = rng.randn(n)
    b = rng.randn(n)
    c = rng.randn(n)
    y = (a + 0.5 * b + rng.randn(n) * 0.1 > 0).astype(int)
    return a, b, c, y


def _mrmr():
    """Construct a cheap MRMR instance for the degenerate-frame tests."""
    return MRMR(full_npermutations=2, baseline_npermutations=1, fe_max_steps=1, fe_smart_polynom_iters=0)


def _fit(df, y):
    """Fit a fresh MRMR instance on (df, y), silencing the accuracy-degrading-params warning."""
    m = _mrmr()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(df, y)
    return m


# --------------------------------------------------------------------------- unit: pure scan


def test_scan_all_nan():
    """Scan all nan."""
    a, b, _, _ = _mk()
    df = pd.DataFrame({"a": a, "b": b, "z": np.full(len(a), np.nan)})
    assert audit_degenerate_columns(df) == {"z": "all_nan"}


def test_scan_constant():
    """Scan constant."""
    a, b, _, _ = _mk()
    df = pd.DataFrame({"a": a, "b": b, "z": np.ones(len(a))})
    assert audit_degenerate_columns(df) == {"z": "constant"}


def test_scan_exact_duplicate():
    """Scan exact duplicate."""
    a, b, _, _ = _mk()
    df = pd.DataFrame({"a": a, "b": b, "z": a.copy()})
    assert audit_degenerate_columns(df) == {"z": "duplicate_of:a"}


def test_scan_collinear():
    """Scan collinear."""
    a, b, _, _ = _mk()
    df = pd.DataFrame({"a": a, "b": b, "z": 2.0 * a + 3.0})
    assert audit_degenerate_columns(df) == {"z": "collinear_with:a"}


def test_scan_clean_frame_empty():
    """Scan clean frame empty."""
    a, b, c, _ = _mk()
    df = pd.DataFrame({"a": a, "b": b, "c": c})
    assert audit_degenerate_columns(df) == {}


def test_scan_numpy_array():
    """Scan numpy array."""
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
    """Fit records degenerate."""
    a, b, _, y = _mk()
    df = pd.DataFrame({"a": a, "b": b, "z": make_col(a)})
    m = _fit(df, y)  # must not crash
    assert m.degenerate_columns_.get("z") == reason


def test_clean_frame_has_empty_degenerate():
    """Clean frame has empty degenerate."""
    a, b, c, y = _mk()
    df = pd.DataFrame({"a": a, "b": b, "c": c})
    m = _fit(df, y)
    assert m.degenerate_columns_ == {}


# --------------------------------------------------------------------------- y-NaN parity with siblings


def test_y_nan_raises_valueerror():
    """Y nan raises valueerror."""
    a, b, _, y = _mk()
    df = pd.DataFrame({"a": a, "b": b})
    yn = y.astype(float)
    yn[0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        _mrmr().fit(df, yn)


def test_y_inf_raises_valueerror():
    """Y inf raises valueerror."""
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
        lambda a: np.ones(len(a)),  # constant -- carries NO signal
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


# --------------------------------------------------------------------------- regression: content-hash + M-layout perf fix


def test_regression_wide_frame_matches_reference_slow_content_hash():
    """2026-07-17: ``_content_key`` was switched from ``pandas.util.hash_array(...).tobytes()`` to a
    copy-free xxh3 hash of the raw buffer (the dominant self-time line of ``audit_degenerate_columns`` on
    wide frames -- measured ~17ms/column), and the collinearity matrix ``M`` was rebuilt row-major (was a
    column-into-row-major-array write antipattern, ~13ms/column). Both are perf-only changes; this pins
    exact output equivalence against the ORIGINAL (slow) implementation on a wide (p=60), NaN-and-duplicate-
    and-collinear-laden frame -- the shape most likely to expose an off-by-something in the row/column swap
    or a NaN-bit-pattern mismatch between the two hashing schemes."""
    from mlframe.feature_selection.filters import _mrmr_degenerate as m

    def _reference_content_key(values):
        """Original (pre-2026-07-17) pandas.util.hash_array-based content key, kept as the ground truth."""
        try:
            return pd.util.hash_array(np.asarray(values)).tobytes()
        except Exception:
            try:
                return np.asarray(values).tobytes()
            except Exception:
                return None

    def _reference_audit(X):
        """Original (pre-2026-07-17) audit_degenerate_columns implementation, kept as the ground truth."""
        degenerate: dict = {}
        seen_content: dict = {}
        numeric_cols: list = []
        for name, values in m._column_arrays(X):
            if m._is_all_nan(values):
                degenerate[name] = "all_nan"
                continue
            if m._is_constant(values):
                degenerate[name] = "constant"
                continue
            key = _reference_content_key(values)
            if key is not None:
                if key in seen_content:
                    degenerate[name] = f"duplicate_of:{seen_content[key]}"
                    continue
                seen_content[key] = name
            if values.dtype.kind in "fiu":
                v = values.astype(np.float64)
                finite = np.isfinite(v)
                if finite.sum() >= 2:
                    numeric_cols.append((name, v, finite))
        live = [(n, v, f) for (n, v, f) in numeric_cols if n not in degenerate]
        if len(live) >= 2:
            names = [n for (n, _, _) in live]
            n_rows = live[0][1].shape[0]
            M = np.empty((n_rows, len(live)), dtype=np.float64)
            for k, (_, v, fin) in enumerate(live):
                col = v.copy()
                if not fin.all():
                    col_mean = float(np.nanmean(col)) if fin.any() else 0.0
                    col = np.where(fin, col, col_mean)
                M[:, k] = col
            with np.errstate(invalid="ignore"):
                M -= M.mean(axis=0, keepdims=True)
                stds = np.sqrt((M * M).sum(axis=0))
            good = stds > 0
            with np.errstate(invalid="ignore", divide="ignore"):
                M = np.where(good, M / np.where(stds == 0, 1.0, stds), 0.0)
            corr = M.T @ M
            np.fill_diagonal(corr, 0.0)
            abs_corr = np.abs(corr)
            for j in range(len(live)):
                if not good[j]:
                    continue
                row = abs_corr[j, :j]
                hits = np.where(np.abs(row - 1.0) <= m._COLLINEAR_TOL)[0]
                for i in hits:
                    if good[i] and names[i] not in degenerate:
                        degenerate[names[j]] = f"collinear_with:{names[i]}"
                        break
        return degenerate

    rng = np.random.RandomState(7)
    n = 400
    cols: dict = {}
    base: dict = {}
    for j in range(60):
        kind = j % 6
        if kind == 0:
            v = rng.randn(n)
        elif kind == 1:
            v = np.full(n, float(j))
        elif kind == 2:
            v = np.full(n, np.nan)
        elif kind == 3 and base:
            v = base[list(base)[j % len(base)]].copy()
        elif kind == 4 and base:
            v = base[list(base)[j % len(base)]] * (j - 30) + 1.0
        elif kind == 5:
            v = rng.randn(n)
            v[rng.random(n) < 0.15] = np.nan
        else:
            v = rng.randn(n)
        cols[f"c{j}"] = v
        base[f"c{j}"] = v
    X = pd.DataFrame(cols)

    assert audit_degenerate_columns(X) == _reference_audit(X)
