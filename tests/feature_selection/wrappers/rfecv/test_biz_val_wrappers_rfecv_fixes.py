"""biz_val + unit tests for fixes landed 2026-05-12:

1. ``RFECV.max_nfeatures`` now hard-caps the final selection (was
   silently violated when the all-features baseline had the highest
   CV score; the post-optimization argmax ignored max_nfeatures).
2. ``RFECV.must_exclude`` works correctly with DataFrames (the
   earlier "didn't exclude" observation was a TEST bug, not a
   feature bug -- the test indexed ``df.columns`` with support
   indices that are aligned with ``feature_names_in_`` AFTER the
   exclusion).
3. ``OptimumSearch.ScipyLocal`` / ``ScipyGlobal`` /
   ``ExhaustiveDichotomic`` implemented (previously raised
   ``NotImplementedError``).

Naming: ``test_biz_val_rfecv_<feature>_<scenario>`` and
``test_unit_rfecv_<helper>_<scenario>``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_df(n=600, p_signal=5, p_noise=8, seed=42):
    """Make df."""
    rng = np.random.default_rng(seed)
    X_sig = rng.normal(size=(n, p_signal))
    X_noise = rng.normal(size=(n, p_noise))
    X = np.column_stack([X_sig, X_noise])
    y = (X_sig.sum(axis=1) + 0.3 * rng.normal(size=n) > 0).astype(np.int64)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(p_signal + p_noise)])
    return df, y


def _support_to_indices(sel):
    """Support to indices."""
    s = sel.support_
    if hasattr(s, "dtype") and s.dtype == bool:
        return [int(i) for i in np.flatnonzero(s)]
    return [int(i) for i in s]


def _support_to_names(sel):
    """CORRECT way: map support_ through feature_names_in_ (not
    through the original DataFrame.columns -- the two diverge when
    must_exclude or other column-dropping options fire)."""
    return [sel.feature_names_in_[i] for i in _support_to_indices(sel)]


# ---------------------------------------------------------------------------
# Fix 1: max_nfeatures hard cap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cap", [2, 3, 5])
def test_biz_val_rfecv_max_nfeatures_hard_caps_final_selection(cap):
    """``max_nfeatures=N`` must STRICTLY cap ``n_features_`` to <= N
    on the final selection. Before the 2026-05-12 fix, the
    all-features baseline could win the argmax even when ``cap``
    was set; the final selection then violated the user's cap."""
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier
    from mlframe.feature_selection.wrappers import RFECV

    df, y = _make_df(n=600, p_signal=5, p_noise=8, seed=42)
    sel = RFECV(
        estimator=RandomForestClassifier(random_state=42, n_estimators=15),
        cv=2,
        max_refits=4,
        verbose=0,
        random_state=42,
        max_noimproving_iters=3,
        max_nfeatures=cap,
    )
    sel.fit(df, y)
    assert sel.n_features_ <= cap, f"max_nfeatures={cap} must hard-cap n_features_; got {sel.n_features_}"
    assert len(_support_to_indices(sel)) <= cap


def test_biz_val_rfecv_max_nfeatures_none_unconstrained():
    """``max_nfeatures=None`` (default) must NOT artificially cap."""
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier
    from mlframe.feature_selection.wrappers import RFECV

    df, y = _make_df(n=600, p_signal=5, p_noise=8, seed=42)
    sel = RFECV(
        estimator=RandomForestClassifier(random_state=42, n_estimators=15),
        cv=2,
        max_refits=4,
        verbose=0,
        random_state=42,
        max_noimproving_iters=3,
        max_nfeatures=None,
    )
    sel.fit(df, y)
    assert sel.n_features_ >= 1


@pytest.mark.parametrize("rule", ["argmax", "one_se_max", "one_se_min"])
def test_biz_val_rfecv_max_nfeatures_respected_across_selection_rules(rule):
    """``max_nfeatures`` must be respected by ALL selection rules
    (argmax / one_se_max / one_se_min). Parametrize over the 3
    documented rules."""
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier
    from mlframe.feature_selection.wrappers import RFECV

    df, y = _make_df(n=600, p_signal=5, p_noise=8, seed=42)
    sel = RFECV(
        estimator=RandomForestClassifier(random_state=42, n_estimators=15),
        cv=2,
        max_refits=4,
        verbose=0,
        random_state=42,
        max_noimproving_iters=3,
        max_nfeatures=4,
        n_features_selection_rule=rule,
    )
    sel.fit(df, y)
    assert sel.n_features_ <= 4, f"rule={rule} must respect max_nfeatures=4; got {sel.n_features_}"


# ---------------------------------------------------------------------------
# Fix 2: must_exclude works correctly (verifies the actual contract,
# guards against future regressions, AND documents the right way
# to read selected names)
# ---------------------------------------------------------------------------


def test_biz_val_rfecv_must_exclude_drops_named_column():
    """``must_exclude=['x0']`` must drop x0 from ``feature_names_in_``
    and therefore from any final ``support_``-derived selection."""
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier
    from mlframe.feature_selection.wrappers import RFECV

    df, y = _make_df(n=600, p_signal=3, p_noise=6, seed=42)
    sel = RFECV(
        estimator=RandomForestClassifier(random_state=42, n_estimators=20),
        cv=2,
        max_refits=3,
        verbose=0,
        random_state=42,
        max_noimproving_iters=2,
        must_exclude=["x0"],
    )
    sel.fit(df, y)
    # x0 must NOT appear in feature_names_in_ (it was dropped at fit
    # entry; feature_names_in_ reflects post-drop columns).
    assert "x0" not in sel.feature_names_in_, f"must_exclude=['x0'] must drop x0 from feature_names_in_; got {sel.feature_names_in_}"
    # Map support_ through feature_names_in_ (THE correct way; using
    # df.columns instead would index the WRONG column because
    # feature_names_in_ has one fewer entry than df).
    selected = _support_to_names(sel)
    assert "x0" not in selected, f"must_exclude=['x0'] must keep x0 out of selection; got selected={selected}"


def test_biz_val_rfecv_must_exclude_silently_ignored_on_numpy():
    """``must_exclude`` requires a DataFrame input -- on numpy arrays
    the flag is silently ignored. Document this contract; if the
    behaviour ever changes (e.g. raising or normalizing), this test
    detects it."""
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier
    from mlframe.feature_selection.wrappers import RFECV

    df, y = _make_df(n=400, p_signal=3, p_noise=5, seed=42)
    X_np = df.values  # numpy array, no column names
    sel = RFECV(
        estimator=RandomForestClassifier(random_state=42, n_estimators=15),
        cv=2,
        max_refits=3,
        verbose=0,
        random_state=42,
        max_noimproving_iters=2,
        must_exclude=["x0"],
    )
    # Must not raise even though x0 is a name and X is unnamed numpy.
    sel.fit(X_np, y)
    # All original columns must be in feature_names_in_ (none were
    # dropped -- must_exclude is silently a no-op on numpy).
    assert len(sel.feature_names_in_) == X_np.shape[1]


# ---------------------------------------------------------------------------
# Fix 3: OptimumSearch -- all 5 variants now functional
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "search_method",
    [
        "ScipyLocal",
        "ScipyGlobal",
        "ModelBasedHeuristic",
        "ExhaustiveRandom",
        "ExhaustiveDichotomic",
    ],
)
def test_biz_val_rfecv_optimum_search_all_variants_complete(search_method):
    """All 5 OptimumSearch enum values must produce a valid
    ``support_`` without raising. Pre-fix only ExhaustiveRandom +
    ModelBasedHeuristic worked; the other 3 raised
    ``NotImplementedError``."""
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier
    from mlframe.feature_selection.wrappers import RFECV
    from mlframe.feature_selection.wrappers._enums import OptimumSearch

    df, y = _make_df(n=400, p_signal=3, p_noise=5, seed=42)
    sel = RFECV(
        estimator=RandomForestClassifier(random_state=42, n_estimators=15),
        cv=2,
        max_refits=4,
        verbose=0,
        random_state=42,
        max_noimproving_iters=2,
        top_predictors_search_method=OptimumSearch(search_method),
    )
    sel.fit(df, y)
    assert 1 <= len(_support_to_indices(sel)) <= df.shape[1]


# ---------------------------------------------------------------------------
# Unit tests for the OptimumSearch suggester helpers
# ---------------------------------------------------------------------------


def test_unit_helpers_suggest_dichotomic_probes_midpoint_when_empty():
    """``_suggest_dichotomic`` with empty/single-evaluation history
    probes the midpoint of the full feature range."""
    from mlframe.feature_selection.wrappers._helpers import _suggest_dichotomic

    n_total = 20
    remaining = list(range(1, n_total + 1))
    # 1 evaluation -> probe N/2 = 10
    result = _suggest_dichotomic(remaining, evaluated_scores_mean={20: 0.5}, n_total=n_total)
    assert 8 <= result <= 12, f"midpoint probe should target ~10; got {result}"


def test_unit_helpers_suggest_dichotomic_narrows_around_best():
    """With history, dichotomic probes between best_evaluated N and
    nearest unevaluated neighbour."""
    from mlframe.feature_selection.wrappers._helpers import _suggest_dichotomic

    # best is at N=10 (score 0.8); 5 and 15 evaluated lower
    remaining = [3, 4, 6, 7, 8, 12, 13, 14, 17, 18]
    history = {5: 0.4, 10: 0.8, 15: 0.3}
    result = _suggest_dichotomic(remaining, history, n_total=20)
    # Should probe near (5+10)/2 = 7 or (10+15)/2 = 12 -- both are
    # in remaining.
    assert result in {6, 7, 8, 12, 13}, f"dichotomic should probe around midpoint of [5,10] or [10,15]; got {result}"


def test_unit_helpers_suggest_scipy_local_falls_back_on_short_history():
    """``_suggest_scipy_local`` requires >= 3 evaluations to fit a
    bracket; shorter history must fall back to dichotomic."""
    from mlframe.feature_selection.wrappers._helpers import _suggest_scipy_local

    remaining = list(range(1, 11))
    # 2 evals -> not enough for scipy bracket; should fall back
    result = _suggest_scipy_local(remaining, {1: 0.1, 10: 0.5}, n_total=10)
    assert result is not None
    assert result in remaining


def test_unit_helpers_suggest_scipy_local_converges_to_peak():
    """``_suggest_scipy_local`` with a clear single peak must
    propose a probe NEAR that peak."""
    from mlframe.feature_selection.wrappers._helpers import _suggest_scipy_local

    # Peak at N=5: history has scores forming an upside-down U.
    history = {1: 0.0, 3: 0.3, 5: 0.9, 7: 0.4, 9: 0.0}
    remaining = [2, 4, 6, 8, 10]
    result = _suggest_scipy_local(remaining, history, n_total=10)
    # The proposed probe should be near the peak (5) -- 4 or 6
    # are the closest unevaluated points.
    assert result in {4, 6}, f"scipy_local should converge to peak neighbour; got {result}"


def test_unit_helpers_suggest_scipy_global_falls_back_on_short_history():
    """``_suggest_scipy_global`` requires >= 4 evaluations; shorter
    history must fall back to dichotomic."""
    from mlframe.feature_selection.wrappers._helpers import _suggest_scipy_global

    remaining = list(range(1, 11))
    result = _suggest_scipy_global(remaining, {1: 0.1, 5: 0.5, 10: 0.3}, n_total=10)
    assert result in remaining


def test_unit_helpers_suggest_scipy_global_finds_dominant_peak():
    """``_suggest_scipy_global`` on a bimodal score curve must find
    the DOMINANT peak, not a secondary one."""
    from mlframe.feature_selection.wrappers._helpers import _suggest_scipy_global

    # Bimodal: small peak at N=3 (0.5), dominant peak at N=8 (0.9)
    history = {1: 0.0, 3: 0.5, 5: 0.1, 8: 0.9, 10: 0.4}
    remaining = [2, 4, 6, 7, 9]
    result = _suggest_scipy_global(remaining, history, n_total=10)
    # Dominant peak is at 8; probe should be a neighbour of 8 (7 or 9)
    assert result in {7, 9}, f"scipy_global should find dominant peak neighbour; got {result}"


def test_unit_helpers_suggest_dichotomic_handles_no_remaining():
    """When ``remaining`` is empty, suggesters must return None
    cleanly (no IndexError, no crash)."""
    from mlframe.feature_selection.wrappers._helpers import (
        _suggest_dichotomic,
        _suggest_scipy_local,
        _suggest_scipy_global,
    )

    assert _suggest_dichotomic([], {5: 0.5}, n_total=10) is None
    assert _suggest_scipy_local([], {1: 0.1, 5: 0.5, 10: 0.3}, n_total=10) is None
    assert _suggest_scipy_global([], {1: 0.1, 3: 0.2, 5: 0.5, 10: 0.3}, n_total=10) is None
