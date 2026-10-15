"""Behavioural tests for MRMR stability selection, Boruta, null importance and selector wrappers."""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import pytest


def test_regression_cluster_stability_selection_raises_on_total_failure():
    """Pre-fix: 100% bootstrap failure silently returned an empty selection. Post-fix: raises RuntimeError."""
    from mlframe.feature_selection.filters._stability_cluster import cluster_stability_selection

    rng = np.random.default_rng(0)
    n, p = 40, 5
    X = rng.standard_normal((n, p))
    y = rng.standard_normal(n)

    def _always_fails(Xb, yb):
        """Stand-in that raises ValueError, so the caller's failure path is the one under test."""
        raise ValueError("simulated systematic selector_fn failure")

    with pytest.raises(RuntimeError, match="all .* bootstraps failed"):
        cluster_stability_selection(X, y, _always_fails, n_bootstrap=5, rng_seed=0)


def test_regression_stability_select_fe_raises_on_total_failure(monkeypatch):
    """Pre-fix: 100% bootstrap failure silently returned an empty per-bootstrap list. Post-fix: raises."""
    import pandas as pd

    from mlframe.feature_selection.filters import _stability_fe as sfe

    class _FakeMRMR:
        """Minimal MRMR stand-in: carries just the attributes the code under test reads, nothing else."""

        def __init__(self, **kwargs):
            """Minimal stand-in constructor for the double used below."""
            pass

        def fit(self, X, y):
            """Stand-in that raises ValueError, so the caller's failure path is the one under test."""
            raise ValueError("simulated systematic MRMR.fit failure")

    monkeypatch.setattr(sfe, "_resolve_mrmr_cls", lambda: _FakeMRMR)

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((40, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(rng.standard_normal(40))

    with pytest.raises(RuntimeError, match="all .* bootstraps failed"):
        sfe._run_bootstraps(X, y, {}, n_bootstraps=5, sample_fraction=0.5, rng=rng)


# ---------------------------------------------------------------------------
# CLUSTERING_STABILITY-5 (P2): _calibrate_tau_auto's diagnostics (n_pairs_sampled/n_pairs_finite) were
# populated AFTER the len(su_scores)<10 fallback check, so they stayed at 0 whenever that fallback fired.
# ---------------------------------------------------------------------------


def test_regression_calibrate_tau_auto_diagnostics_populated_before_fallback():
    """Pre-fix: n_pairs_sampled/n_pairs_finite stayed at their 0 init value when the <10-scores fallback
    fired. Post-fix: they reflect what was actually sampled/scored even on the fallback path."""
    from mlframe.feature_selection.filters._dcd_tau_auto import _calibrate_tau_auto

    rng = np.random.default_rng(0)
    n_cols = 4  # small n_cols -> C(4,2)=6 pairs, well under the len(su_scores)<10 fallback floor
    n_rows = 200
    factors_data = rng.integers(0, 5, size=(n_rows, n_cols)).astype(np.int32)
    factors_nbins = np.full(n_cols, 5, dtype=np.int64)

    _tau, diagnostics = _calibrate_tau_auto(factors_data=factors_data, factors_nbins=factors_nbins, seed=0)
    assert diagnostics["n_pairs_sampled"] > 0, "diagnostics must reflect the actual sampled-pair count, not 0"


# ---------------------------------------------------------------------------
# CLUSTERING_STABILITY-6 (P2): _binarize_aggregate excluded NaN/Inf rows only from edge computation, not
# from the actual binned output -- a non-finite row silently landed in a real bin instead of a dedicated code.
# ---------------------------------------------------------------------------


def test_regression_ks_stability_filter_rejects_invalid_split_frac():
    """Pre-fix: split_frac=1.5 crashed deep inside rng.choice with an opaque numpy ValueError.
    Post-fix: a clear, named ValueError raises immediately."""
    import pandas as pd

    from mlframe.feature_selection.filters._ks_stability import ks_stability_filter

    rng = np.random.default_rng(0)
    train_df = pd.DataFrame({"f": rng.standard_normal(100)})
    test_df = pd.DataFrame({"f": rng.standard_normal(100)})

    with pytest.raises(ValueError, match="split_frac"):
        ks_stability_filter(train_df, test_df, n_splits=3, split_frac=1.5)


# ---------------------------------------------------------------------------
# CLUSTERING_STABILITY-10 (P2): commit_swap's is_member_swap check used a getattr default that could never
# be reached (SwapDecision.branch always exists, defaulting to "none").
# ---------------------------------------------------------------------------


def test_regression_cluster_basis_no_noop_index_lookup():
    """Post-fix: the no-op dense_names.index(name) re-lookup is removed; mean_corr's names are used directly."""
    import numpy as np
    import pandas as pd

    from mlframe.feature_selection.filters._orthogonal_cluster_basis_fe import detect_clusters_by_correlation

    rng = np.random.default_rng(0)
    n = 400
    base = rng.standard_normal(n)
    X = pd.DataFrame(
        {
            "a1": base + 0.05 * rng.standard_normal(n),
            "a2": base + 0.05 * rng.standard_normal(n),
            "a3": base + 0.1 * rng.standard_normal(n),
            "z": rng.standard_normal(n),
            "w": rng.standard_normal(n),
        }
    )
    clusters = detect_clusters_by_correlation(X)
    # The three correlated columns form one cluster whose members are the column names themselves, sorted; the independent columns are not clustered.
    assert clusters == {"a1": ["a1", "a2", "a3"]}


# ---------------------------------------------------------------------------
# ORTH_SCORING_B-7 (P2): _oracle_scorer_select.py's _quality_objective except-Exception had zero logging.
# ---------------------------------------------------------------------------


def test_regression_boruta_select_rejects_zero_iterations():
    """Regression: boruta_select(n_iterations=0) silently produced a NaN win_rate then an unhelpful scipy ValueError instead of a clear parameter-
    validation message.
    """
    from mlframe.feature_selection.filters._boruta import boruta_select

    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 3))
    y = (X[:, 0] > 0).astype(int)

    with pytest.raises(ValueError, match="n_iterations"):
        boruta_select(X, y, importance_fn=lambda Xs, ys: np.abs(Xs).mean(axis=0), n_iterations=0)


# ---------------------------------------------------------------------------
# USABILITY_A-3 (P2): null_importance_filter(n_shuffles=0) raised an unhelpful IndexError instead of a
# clear parameter-validation message.
# ---------------------------------------------------------------------------


def test_regression_null_importance_filter_rejects_zero_shuffles():
    """Regression: null_importance_filter(n_shuffles=0) raised an unhelpful IndexError instead of a clear parameter-validation message."""
    from mlframe.feature_selection.filters._null_importance import null_importance_filter

    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 3))
    y = (X[:, 0] > 0).astype(int)

    with pytest.raises(ValueError, match="n_shuffles"):
        null_importance_filter(X, y, importance_fn=lambda Xs, ys: np.abs(Xs).mean(axis=0), n_shuffles=0)


# ---------------------------------------------------------------------------
# USABILITY_A-4 (P2): compose_pair_fe's single_mi_cache was keyed only by column name; a duplicate name
# silently aliased the second column's MI to the first one's cached value.
# ---------------------------------------------------------------------------


def test_regression_boruta_select_bh_correction_scales_with_rounds():
    """Post-fix: the BH branch's per-round critical values shrink with rounds_run the same way the bonferroni branch's
    corrected_alpha does (both / rounds_run), instead of staying fixed at alpha. A feature hit 9 times in 10 rounds has a
    two-sided p of about 0.0215: under the scaled level alpha/rounds_run = 0.005 it must stay undecided, while a feature
    hit 10 times in 10 rounds (p of about 0.00195) is still confirmed."""
    from mlframe.feature_selection.filters._boruta import _resolve_undecided_features_for_round

    borderline = [None]
    _resolve_undecided_features_for_round(borderline, np.array([9]), 10, 1, 0.05, "bh")
    assert borderline == [None], f"a p of about 0.0215 must not clear the rounds-corrected level 0.005; got {borderline}"

    clear = [None]
    _resolve_undecided_features_for_round(clear, np.array([10]), 10, 1, 0.05, "bh")
    assert clear == ["confirmed"], f"a p of about 0.00195 must clear the rounds-corrected level 0.005; got {clear}"


# ---------------------------------------------------------------------------
# USABILITY_A-14 (P2): warn_accuracy_suboptimal_params's one-shot latch never re-fired after a
# set_params() call degraded a param post-first-fit.
# ---------------------------------------------------------------------------


def test_regression_explain_selection_logs_section_failures(caplog):
    """Regression: explain_selection's per-section except-Exception blocks never called logger, even at debug level."""
    import logging

    from mlframe.feature_selection.filters import _mrmr_explain as explain_mod

    class _BrokenMRMR:
        # A non-empty DataFrame missing the "origin" column: _survivor_section's
        # `prov.groupby("origin", ...)` raises KeyError, exercising the except-Exception path.
        """Deliberately broken MRMR stand-in: exercises the caller's error handling, not a happy path."""
        fe_provenance_ = pd.DataFrame({"feature_name": ["a", "b"]})

    with caplog.at_level(logging.DEBUG, logger="mlframe.feature_selection.filters._mrmr_explain"):
        report = explain_mod.explain_selection(_BrokenMRMR())
    assert isinstance(report, str)
    assert any("failed" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# USABILITY_B-9 (P2, licensing): no LICENSE/NOTICE existed for the vendored InfoNet tree.
# ---------------------------------------------------------------------------


def test_regression_stability_mrmr_has_get_feature_names_out():
    """Regression: stabilityMRMR / StabilityFESelector were missing get_feature_names_out(), unlike their siblings MRMR / CorrelatedFeaturesSelector in the same
    module.
    """
    from mlframe.feature_selection.filters.stability import StabilityMRMR
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(rng.standard_normal((n, 5)), columns=[f"x{i}" for i in range(5)])
    y = (X["x0"] > 0).astype(int)
    sel = StabilityMRMR(MRMR(verbose=0, quantization_nbins=5), n_bootstraps=3, sample_fraction=0.8, random_state=0)
    sel.fit(X, y)
    names = sel.get_feature_names_out()
    assert isinstance(names, np.ndarray)
    assert len(names) == len(sel.support_)
    assert all(n in X.columns for n in names)


def test_regression_stability_fe_selector_has_get_feature_names_out():
    """Regression guard for stability fe selector has get feature names out."""
    from mlframe.feature_selection.filters._stability_fe import StabilityFESelector

    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(rng.standard_normal((n, 5)), columns=[f"x{i}" for i in range(5)])
    y = (X["x0"] > 0).astype(int)
    sel = StabilityFESelector(base_mrmr_params={"verbose": 0, "quantization_nbins": 5}, n_bootstraps=3, sample_fraction=0.8, random_state=0)
    sel.fit(X, y)
    names = sel.get_feature_names_out()
    out = sel.transform(X)
    assert isinstance(names, np.ndarray)
    assert list(names) == list(out.columns)


# ---------------------------------------------------------------------------
# X_SECURITY_API_PACKAGING-3 (P2): CorrelatedFeaturesSelector.fit had zero validation of corr_threshold /
# min_reduction, unlike StabilityMRMR's analogous knobs in the same module.
# ---------------------------------------------------------------------------


def test_regression_correlated_features_selector_rejects_invalid_corr_threshold():
    """Regression: groupAwareMRMR.fit had zero validation of corr_threshold / min_reduction, unlike StabilityMRMR's analogous knobs in the same module."""
    from mlframe.feature_selection.filters.correlated_features import CorrelatedFeaturesSelector
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((50, 4)), columns=["a", "b", "c", "d"])
    y = (X["a"] > 0).astype(int)
    sel = CorrelatedFeaturesSelector(MRMR(verbose=0), corr_threshold=5.0)
    with pytest.raises(ValueError, match="corr_threshold"):
        sel.fit(X, y)


def test_regression_correlated_features_selector_rejects_invalid_min_reduction():
    """Regression guard for group aware mrmr rejects invalid min reduction."""
    from mlframe.feature_selection.filters.correlated_features import CorrelatedFeaturesSelector
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((50, 4)), columns=["a", "b", "c", "d"])
    y = (X["a"] > 0).astype(int)
    sel = CorrelatedFeaturesSelector(MRMR(verbose=0), min_reduction=1.0)
    with pytest.raises(ValueError, match="min_reduction"):
        sel.fit(X, y)


def test_regression_correlated_features_selector_default_params_still_fit():
    """Sanity: the new validation must not reject the class's own defaults."""
    from mlframe.feature_selection.filters.correlated_features import CorrelatedFeaturesSelector
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((50, 4)), columns=["a", "b", "c", "d"])
    y = (X["a"] > 0).astype(int)
    sel = CorrelatedFeaturesSelector(MRMR(verbose=0))
    sel.fit(X, y)
    assert "a" in list(sel.get_feature_names_out())


# ---------------------------------------------------------------------------
# X_SECURITY_API_PACKAGING-5 (P2): pyutilz/py-ci-shared were declared as bare git URLs with no
# pinned commit SHA -- pip install resolved to whatever the default branch happened to be.
# ---------------------------------------------------------------------------


def test_regression_selection_stability_report_honours_random_state():
    """Post-fix: an MRMR seeded via random_state= (canonical) and one seeded via the equivalent
    random_seed= (deprecated alias) produce IDENTICAL bootstrap selection-frequency reports --
    pinning that selection_stability_report resolves random_state, not just the deprecated alias."""

    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame(rng.standard_normal((n, 6)), columns=[f"x{i}" for i in range(6)])
    y = (X["x0"] + 0.5 * X["x1"] > 0).astype(int)

    m_state = MRMR(verbose=0, quantization_nbins=5, random_state=42)
    m_seed = MRMR(verbose=0, quantization_nbins=5, random_seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        m_state.fit(X, y)
        m_seed.fit(X, y)

    report_state = m_state.selection_stability_report(n_boot=10, as_text=False)
    report_seed = m_seed.selection_stability_report(n_boot=10, as_text=False)
    assert report_state, "expected a non-empty replay-state report"
    assert report_state["feature_selection_frequency"] == report_seed["feature_selection_frequency"]


def test_regression_selection_stability_report_random_state_changes_bootstrap():
    """Sanity: two DIFFERENT random_state values must (with overwhelming probability on this fixture)
    produce a different bootstrap RNG draw -- proves the seed is actually being consumed, not ignored
    entirely (which would make the equality test above pass vacuously)."""
    import warnings

    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame(rng.standard_normal((n, 6)), columns=[f"x{i}" for i in range(6)])
    y = (X["x0"] + 0.5 * X["x1"] > 0).astype(int)
    m = MRMR(verbose=0, quantization_nbins=5, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        m.fit(X, y)
    report_a = m.selection_stability_report(n_boot=10, as_text=False, random_state=1)
    report_b = m.selection_stability_report(n_boot=10, as_text=False, random_state=2)
    assert report_a["feature_selection_frequency"] != report_b["feature_selection_frequency"]


# ---------------------------------------------------------------------------
# X_TEST_COVERAGE_QUALITY-2 (P1): commit f067e0d44 patched 14 scorer-zoo FE modules for the identical
# B-17 replay-drift bug (recipe doesn't freeze fit-time basis-preprocess mean/std, so replaying on a
# distributionally-shifted slice silently refits and drifts), but only 1 of the 14 got a named
# regression test. Port the same slice-replay pattern (test_jmim.py::TestRecipeFreezesPreprocessParams)
# to the remaining 13.
# ---------------------------------------------------------------------------
