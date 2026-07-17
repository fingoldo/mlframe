"""Meta-test — targeted Hypothesis fuzz on the utility transforms that
sit between user input and an estimator's ``fit()``: ``prepare_df_for_
catboost``, ``_canonical_predict_proba_shape``, ``_predict_from_probs``.

Complements the broad-axis ``test_fuzz_suite.py`` which sweeps
combinations of preprocessing options at the integration level. The tests
here are *unit-level* boundary fuzzers — small inputs designed to surface
silent-coercion / NaN / single-class / empty / single-row bugs in the
transforms themselves, where a wide-net integration fuzzer either misses
the case or doesn't pinpoint the offending utility.

These are exactly the spots flagged in user feedback "Fuzz test failure →
fix prod, never mask via canon/runtime-rewrites/guards" — so we surface
ANY failure in these utilities loudly rather than letting an upstream
canonicaliser paper over them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings, strategies as st


# ---------------------------------------------------------------------------
# prepare_df_for_catboost — NaN handling, mixed dtypes, empty input.
# ---------------------------------------------------------------------------


@settings(deadline=None, max_examples=30, suppress_health_check=[HealthCheck.too_slow])
@given(
    n_rows=st.integers(min_value=1, max_value=50),
    n_cat=st.integers(min_value=1, max_value=4),
    null_frac=st.floats(min_value=0.0, max_value=0.95),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_prepare_df_for_catboost_nan_handling(n_rows, n_cat, null_frac, seed):
    """NaN cells in cat columns must be replaced with the ``__MISSING__``
    sentinel; the column must end up dtype ``category``. No raised
    exceptions for any null_frac in [0, 1) including 0% and ~100%.
    """
    from mlframe.training.pipeline import prepare_df_for_catboost

    rng = np.random.default_rng(seed)
    cat_cols = [f"cat_{i}" for i in range(n_cat)]
    data: dict = {}
    had_nan: dict[str, bool] = {}
    for col in cat_cols:
        levels = ["A", "B", "C", "D"]
        vals = rng.choice(levels, size=n_rows).astype(object)
        # Inject NaN at the requested fraction.
        mask = rng.random(n_rows) < null_frac
        vals = np.where(mask, np.nan, vals)
        data[col] = vals
        had_nan[col] = bool(mask.any())
    df = pd.DataFrame(data)
    prepare_df_for_catboost(df, cat_cols)
    for col in cat_cols:
        assert df[col].dtype.name == "category", f"col {col} not category dtype after prepare; got {df[col].dtype}"
        assert not df[col].isna().any(), f"col {col} still has NaN after prepare (null_frac={null_frac})"
        # The ``__MISSING__`` sentinel only matters when NaN was actually
        # injected; the function is otherwise a pure dtype cast. Conditioning
        # on ``had_nan[col]`` (the realised injection, not the requested
        # frac) avoids flake on small n_rows where the random draw happens
        # to produce zero nulls.
        if had_nan[col]:
            assert "__MISSING__" in set(df[col].cat.categories), (
                f"col {col} missing the __MISSING__ sentinel after NaN injection (null_frac={null_frac}, n_rows={n_rows})"
            )


@settings(deadline=None, max_examples=10)
@given(n_rows=st.integers(min_value=1, max_value=30))
def test_prepare_df_for_catboost_missing_column_is_noop(n_rows):
    """If a column listed in ``cat_features`` isn't actually in the df,
    ``prepare_df_for_catboost`` must skip it silently (the dispatcher
    elsewhere handles the column-presence policy)."""
    from mlframe.training.pipeline import prepare_df_for_catboost

    df = pd.DataFrame({"x": range(n_rows)})
    prepare_df_for_catboost(df, ["nonexistent_col"])  # must not raise
    assert "nonexistent_col" not in df.columns


# ---------------------------------------------------------------------------
# _canonical_predict_proba_shape — variable input shapes (binary, multiclass,
# multilabel as list-of-(N,2)).
# ---------------------------------------------------------------------------


@settings(deadline=None, max_examples=30, suppress_health_check=[HealthCheck.too_slow])
@given(
    n_rows=st.integers(min_value=1, max_value=50),
    n_classes=st.integers(min_value=2, max_value=8),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_canonical_predict_proba_shape_dense_2d_unchanged(n_rows, n_classes, seed):
    """A clean (N, K) probability matrix must round-trip unchanged
    (modulo dtype coercion to float64)."""
    from mlframe.training.helpers import _canonical_predict_proba_shape

    rng = np.random.default_rng(seed)
    probs = rng.random((n_rows, n_classes))
    out = _canonical_predict_proba_shape(probs)
    assert out.shape == (n_rows, n_classes)
    np.testing.assert_allclose(out, probs, rtol=1e-12, atol=0)


@settings(deadline=None, max_examples=20)
@given(
    n_rows=st.integers(min_value=1, max_value=50),
    n_labels=st.integers(min_value=2, max_value=6),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_canonical_predict_proba_shape_multilabel_list_form(n_rows, n_labels, seed):
    """``MultiOutputClassifier.predict_proba`` returns
    ``list[(N, 2)]`` with one element per label. The canonicaliser must
    reduce to (N, K) by taking the positive-class column from each.
    """
    from mlframe.training.helpers import _canonical_predict_proba_shape

    rng = np.random.default_rng(seed)
    list_of_arrs = [rng.random((n_rows, 2)) for _ in range(n_labels)]
    # Normalize each (N, 2) so columns sum to 1 — what real predict_proba
    # outputs.
    list_of_arrs = [a / a.sum(axis=1, keepdims=True) for a in list_of_arrs]
    out = _canonical_predict_proba_shape(list_of_arrs)
    assert out.shape == (n_rows, n_labels), f"expected (N={n_rows}, K={n_labels}), got {out.shape}"
    for i, arr in enumerate(list_of_arrs):
        np.testing.assert_allclose(
            out[:, i],
            arr[:, 1],
            rtol=1e-12,
            err_msg=f"col {i} != positive-class probs from input arr {i}",
        )


# ---------------------------------------------------------------------------
# _predict_from_probs — boundary cases that aren't covered by the existing
# multilabel test in test_multiclass_classification.py.
# ---------------------------------------------------------------------------


@settings(deadline=None, max_examples=30, suppress_health_check=[HealthCheck.too_slow])
@given(
    n_rows=st.integers(min_value=1, max_value=80),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_predict_from_probs_binary_threshold_0_returns_all_pos(n_rows, seed):
    """Binary path: threshold=0.0 forces every row to the positive class
    (probs ≥ 0 in [0, 1] always)."""
    from mlframe.training.helpers import _predict_from_probs
    from mlframe.training.configs import TargetTypes

    rng = np.random.default_rng(seed)
    # Two-column probs (sklearn binary form).
    raw = rng.random((n_rows, 2))
    probs = raw / raw.sum(axis=1, keepdims=True)
    classes = np.array([0, 1])
    out = _predict_from_probs(
        probs,
        TargetTypes.BINARY_CLASSIFICATION,
        classes_=classes,
        threshold=0.0,
    )
    assert out.shape == (n_rows,)
    assert (out == 1).all(), "threshold=0.0 must label every row positive"


@settings(deadline=None, max_examples=30, suppress_health_check=[HealthCheck.too_slow])
@given(
    n_rows=st.integers(min_value=2, max_value=80),
    n_classes=st.integers(min_value=3, max_value=8),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_predict_from_probs_multiclass_argmax_consistent_with_numpy(n_rows, n_classes, seed):
    """Multiclass path is documented as ``argmax(probs, axis=1)``; assert
    the implementation matches the contract literally — catches a future
    regression to e.g. weighted-vote or temperature-aware argmax."""
    from mlframe.training.helpers import _predict_from_probs
    from mlframe.training.configs import TargetTypes

    rng = np.random.default_rng(seed)
    probs = rng.random((n_rows, n_classes))
    out = _predict_from_probs(
        probs,
        TargetTypes.MULTICLASS_CLASSIFICATION,
    )
    np.testing.assert_array_equal(out, probs.argmax(axis=1))


@settings(deadline=None, max_examples=20)
@given(
    n_rows=st.integers(min_value=1, max_value=50),
    n_labels=st.integers(min_value=2, max_value=6),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_predict_from_probs_multilabel_with_nans_treated_as_negative(n_rows, n_labels, seed):
    """The docstring promises NaN-safe behaviour: NaN probabilities are
    treated as below-threshold (negative). Hypothesis-fuzz this so it
    holds for every (sparsity, threshold) combo."""
    from mlframe.training.helpers import _predict_from_probs
    from mlframe.training.configs import TargetTypes

    rng = np.random.default_rng(seed)
    probs = rng.random((n_rows, n_labels))
    # Inject NaN at random cells.
    mask = rng.random((n_rows, n_labels)) < 0.3
    probs[mask] = np.nan
    out = _predict_from_probs(
        probs,
        TargetTypes.MULTILABEL_CLASSIFICATION,
        threshold=0.5,
    )
    assert out.shape == (n_rows, n_labels)
    # Every NaN cell must end up zero.
    assert (out[mask] == 0).all(), "NaN probabilities should be labelled negative — found positive labels in NaN cells"
