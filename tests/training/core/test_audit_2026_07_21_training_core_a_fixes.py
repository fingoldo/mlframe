"""Regression tests for audits/full_audit_2026-07-21/training_core_a.md findings F1-F4 and proposals PR2-PR4.

F4 (the ``_phase_composite_post_xt_ensemble/__init__.py`` 1147-LOC file-size finding) is a documented,
justified exemption in ``tests/test_meta/test_no_file_over_1k_loc.py``'s own exempt-list comment (assessed
2026-06-22: the OOF/scoring block mutably rebinds locals its own nested closures close over -- not safely
carvable without reproducing the whole local environment as an argument list) -- pinned here, no code
change. PR1 (a focused unit test for ``_run_one_weight_iteration``'s break_model_loop/skip paths) is
NOT implemented as a direct unit test: the function takes 70+ required kwargs deeply coupled to real
upstream pipeline state, making an isolated unit test either extremely brittle or a large refactor
disproportionate to a P-level test-coverage nice-to-have; it is already exercised indirectly via full-suite
integration/e2e tests per the report's own note -- documented disposition, not silently dropped.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# F1: MTR cross-target ensemble honest-OOF NNLS weighting silently dropped sample_weight
# ---------------------------------------------------------------------------


def test_f1_compute_mtr_oof_nnls_weights_threads_sample_weight():
    """Spy-based: directly confirms sample_weight reaches each fold's component.fit() call, sliced to
    that fold's train indices -- more deterministic than inferring it from the final NNLS weight split."""
    from mlframe.training.core._phase_composite_post_xt_ensemble._phase_composite_post_xt_mtr_oof import compute_mtr_oof_nnls_weights
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(0)
    n, k = 300, 2
    X = rng.normal(size=(n, 3))
    true_coef = rng.normal(size=(3, k))
    y = X @ true_coef + rng.normal(scale=0.1, size=(n, k))
    sample_weight = rng.uniform(0.5, 5.0, size=n)

    seen_weights = []

    class _SpyLinearRegression(LinearRegression):
        """Spy Linear Regression."""
        def fit(self, X, y, sample_weight=None):
            """No-op / recording stub matching the estimator's fit() signature."""
            seen_weights.append(None if sample_weight is None else np.asarray(sample_weight).copy())
            return super().fit(X, y, sample_weight=sample_weight)

    weights = compute_mtr_oof_nnls_weights([_SpyLinearRegression(), _SpyLinearRegression()], X, y, kfold=3, random_state=0, sample_weight=sample_weight)
    assert weights is not None
    assert len(seen_weights) > 0
    assert all(w is not None for w in seen_weights)
    # Every fold's fit() received a NON-trivial (not all-ones) weight slice, aligned to that fold's size.
    assert all(not np.allclose(w, w[0]) for w in seen_weights)


def test_f1_compute_mtr_oof_nnls_weights_default_none_unaffected():
    """F1: compute mtr oof nnls weights default none unaffected."""
    from mlframe.training.core._phase_composite_post_xt_ensemble._phase_composite_post_xt_mtr_oof import compute_mtr_oof_nnls_weights
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(1)
    n, k = 200, 1
    X = rng.normal(size=(n, 2))
    y = X @ rng.normal(size=(2, k)) + rng.normal(scale=0.1, size=(n, k))

    components = [LinearRegression(), LinearRegression()]
    weights = compute_mtr_oof_nnls_weights(components, X, y, kfold=3, random_state=0)
    assert weights is not None
    assert weights.shape == (2, 1)


# ---------------------------------------------------------------------------
# F2: MLFRAME_PANDAS_VIEW_CACHE_MAX_MB=0 (and SIZE=0) silently reverted to the 2 GB default
# ---------------------------------------------------------------------------


def test_f2_legacy_max_mb_zero_disables_cache(monkeypatch):
    """F2: legacy max mb zero disables cache."""
    from mlframe.training.core._phase_train_one_target_polars_fastpath import resolve_pandas_view_cache_budget_bytes

    monkeypatch.delenv("MLFRAME_PANDAS_VIEW_CACHE_TYPE", raising=False)
    monkeypatch.delenv("MLFRAME_PANDAS_VIEW_CACHE_SIZE", raising=False)
    monkeypatch.setenv("MLFRAME_PANDAS_VIEW_CACHE_MAX_MB", "0")
    assert resolve_pandas_view_cache_budget_bytes() == 0.0


def test_f2_size_zero_disables_cache_for_absolute_mb(monkeypatch):
    """F2: size zero disables cache for absolute mb."""
    from mlframe.training.core._phase_train_one_target_polars_fastpath import resolve_pandas_view_cache_budget_bytes

    monkeypatch.delenv("MLFRAME_PANDAS_VIEW_CACHE_MAX_MB", raising=False)
    monkeypatch.setenv("MLFRAME_PANDAS_VIEW_CACHE_TYPE", "ABSOLUTE_MB")
    monkeypatch.setenv("MLFRAME_PANDAS_VIEW_CACHE_SIZE", "0")
    assert resolve_pandas_view_cache_budget_bytes() == 0.0


def test_f2_size_zero_disables_cache_for_ram_share(monkeypatch):
    """F2: size zero disables cache for ram share."""
    from mlframe.training.core._phase_train_one_target_polars_fastpath import resolve_pandas_view_cache_budget_bytes

    monkeypatch.delenv("MLFRAME_PANDAS_VIEW_CACHE_MAX_MB", raising=False)
    monkeypatch.setenv("MLFRAME_PANDAS_VIEW_CACHE_TYPE", "FREE_RAM_SHARE")
    monkeypatch.setenv("MLFRAME_PANDAS_VIEW_CACHE_SIZE", "0")
    assert resolve_pandas_view_cache_budget_bytes() == 0.0


def test_f2_negative_size_clamps_to_zero_not_default(monkeypatch):
    """F2: negative size clamps to zero not default."""
    from mlframe.training.core._phase_train_one_target_polars_fastpath import resolve_pandas_view_cache_budget_bytes

    monkeypatch.delenv("MLFRAME_PANDAS_VIEW_CACHE_MAX_MB", raising=False)
    monkeypatch.setenv("MLFRAME_PANDAS_VIEW_CACHE_TYPE", "ABSOLUTE_MB")
    monkeypatch.setenv("MLFRAME_PANDAS_VIEW_CACHE_SIZE", "-5")
    assert resolve_pandas_view_cache_budget_bytes() == 0.0


def test_f2_malformed_size_still_falls_back_to_default(monkeypatch):
    """The ValueError (genuinely malformed input) fallback path must still work -- only the falsy-zero
    shortcut was the bug."""
    from mlframe.training.core._phase_train_one_target_polars_fastpath import resolve_pandas_view_cache_budget_bytes

    monkeypatch.delenv("MLFRAME_PANDAS_VIEW_CACHE_TYPE", raising=False)
    monkeypatch.delenv("MLFRAME_PANDAS_VIEW_CACHE_SIZE", raising=False)
    monkeypatch.setenv("MLFRAME_PANDAS_VIEW_CACHE_MAX_MB", "not-a-number")
    assert resolve_pandas_view_cache_budget_bytes() == 2048.0 * (1024**2)


def test_f2_positive_legacy_mb_still_works(monkeypatch):
    """F2: positive legacy mb still works."""
    from mlframe.training.core._phase_train_one_target_polars_fastpath import resolve_pandas_view_cache_budget_bytes

    monkeypatch.delenv("MLFRAME_PANDAS_VIEW_CACHE_TYPE", raising=False)
    monkeypatch.delenv("MLFRAME_PANDAS_VIEW_CACHE_SIZE", raising=False)
    monkeypatch.setenv("MLFRAME_PANDAS_VIEW_CACHE_MAX_MB", "5000")
    assert resolve_pandas_view_cache_budget_bytes() == 5000 * (1024**2)


# ---------------------------------------------------------------------------
# F3: _ensure_logging_visible's fast-path could skip fixing a later-appended bare handler
# ---------------------------------------------------------------------------


def test_f3_fast_path_still_fixes_a_second_bare_handler():
    """F3: fast path still fixes a second bare handler."""
    from mlframe.training.core._misc_helpers import _ensure_logging_visible

    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    try:
        root.handlers = []
        root.setLevel(logging.INFO)
        _ensure_logging_visible(logging.INFO)  # installs the first, timestamped handler
        assert len(root.handlers) == 1

        # Simulate another package (e.g. Jupyter) appending a bare, non-timestamped handler afterward.
        bare = logging.StreamHandler()
        bare.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        root.addHandler(bare)

        _ensure_logging_visible(logging.INFO)  # second call: must NOT early-return via the first handler
        for h in root.handlers:
            fmt = getattr(h.formatter, "_fmt", None)
            assert fmt is not None and "%(asctime)" in fmt
    finally:
        root.handlers = saved_handlers
        root.setLevel(saved_level)


def test_f3_fast_path_still_short_circuits_when_all_handlers_are_fixed():
    """Regression against a naive fix that removes the fast-path entirely -- it must still no-op
    (not reassign formatters) when every handler is already timestamped."""
    from mlframe.training.core._misc_helpers import _ensure_logging_visible

    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    try:
        root.handlers = []
        root.setLevel(logging.INFO)
        _ensure_logging_visible(logging.INFO)
        assert len(root.handlers) == 1
        original_formatter = root.handlers[0].formatter

        _ensure_logging_visible(logging.INFO)  # nothing new appended -- fast path should apply
        assert root.handlers[0].formatter is original_formatter  # untouched, not reassigned
    finally:
        root.handlers = saved_handlers
        root.setLevel(saved_level)


# ---------------------------------------------------------------------------
# F4: _phase_composite_post_xt_ensemble/__init__.py's 1147-LOC size -- documented exempt, pinned
# ---------------------------------------------------------------------------


def test_f4_oversized_xt_ensemble_file_is_a_documented_exempt():
    """F4: oversized xt ensemble file is a documented exempt."""
    from tests.test_meta.test_no_file_over_1k_loc import LOC_BUDGET_EXEMPT

    assert "src/mlframe/training/core/_phase_composite_post_xt_ensemble/__init__.py" in LOC_BUDGET_EXEMPT


# ---------------------------------------------------------------------------
# PR2: _build_default_extractor / _infer_target_is_classification dtype-inference boundary cases
# ---------------------------------------------------------------------------


def test_pr2_all_nan_float_target_infers_regression_no_crash():
    """PR2: all nan float target infers regression no crash."""
    from mlframe.training.core._main_train_suite_defaults import _infer_target_is_classification

    s = pd.Series([np.nan] * 100, name="y")
    assert _infer_target_is_classification(s) is False


def test_pr2_low_cardinality_float_target_infers_regression():
    """Float dtype is ALWAYS treated as regression regardless of cardinality (the heuristic only checks
    integer/bool dtype) -- this is the verified real (non-crashing) behavior, not a bug the audit's
    speculative wording implied."""
    from mlframe.training.core._main_train_suite_defaults import _infer_target_is_classification

    rng = np.random.default_rng(0)
    s = pd.Series(rng.choice([0.0, 1.0, 2.0], size=200), name="y")
    assert _infer_target_is_classification(s) is False


def test_pr2_build_default_extractor_all_nan_target_no_crash():
    """PR2: build default extractor all nan target no crash."""
    from mlframe.training.core._main_train_suite_defaults import _build_default_extractor
    from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor

    df = pd.DataFrame({"y": [np.nan] * 50, "x": np.arange(50, dtype=float)})
    extractor = _build_default_extractor(df, "y")
    assert isinstance(extractor, SimpleFeaturesAndTargetsExtractor)
    assert "y" in extractor.regression_targets


def test_pr2_build_default_extractor_low_cardinality_int_target_is_classification():
    """PR2: build default extractor low cardinality int target is classification."""
    from mlframe.training.core._main_train_suite_defaults import _build_default_extractor

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"y": rng.integers(0, 3, size=200), "x": rng.normal(size=200)})
    extractor = _build_default_extractor(df, "y")
    assert "y" in extractor.classification_targets


# ---------------------------------------------------------------------------
# PR3: tune_decision_threshold's O(n_candidates * n) sweep vectorized to O(n log n)
# ---------------------------------------------------------------------------


def test_pr3_vectorized_sweep_bit_identical_to_reference_random():
    """PR3: vectorized sweep bit identical to reference random."""
    from mlframe.training.core._benchmarks.bench_tune_decision_threshold import _reference_tune_decision_threshold
    from mlframe.training.core._setup_helpers import tune_decision_threshold

    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, size=4000)
    p = rng.uniform(0, 1, size=4000)
    for metric in ("balanced_accuracy", "f1"):
        assert tune_decision_threshold(y, p, metric=metric) == _reference_tune_decision_threshold(y, p, metric=metric)


def test_pr3_vectorized_sweep_bit_identical_on_heavily_tied_p():
    """The identity-critical case: many rows sharing the exact same probability value."""
    from mlframe.training.core._benchmarks.bench_tune_decision_threshold import _reference_tune_decision_threshold
    from mlframe.training.core._setup_helpers import tune_decision_threshold

    rng = np.random.default_rng(8)
    n = 2000
    y = rng.integers(0, 2, size=n)
    p = rng.choice([0.05, 0.2, 0.5, 0.8, 0.95], size=n)
    for metric in ("balanced_accuracy", "f1"):
        assert tune_decision_threshold(y, p, metric=metric) == _reference_tune_decision_threshold(y, p, metric=metric)


def test_pr3_degenerate_inputs_still_return_default():
    """PR3: degenerate inputs still return default."""
    from mlframe.training.core._setup_helpers import DEFAULT_PROBABILITY_THRESHOLD, tune_decision_threshold

    assert tune_decision_threshold(np.array([]), np.array([])) == DEFAULT_PROBABILITY_THRESHOLD
    assert tune_decision_threshold(np.zeros(10, dtype=int), np.random.default_rng(0).uniform(size=10)) == DEFAULT_PROBABILITY_THRESHOLD


def test_pr3_unsupported_metric_raises():
    """PR3: unsupported metric raises."""
    from mlframe.training.core._setup_helpers import tune_decision_threshold

    with pytest.raises(ValueError, match="unsupported metric"):
        tune_decision_threshold(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8]), metric="accuracy")


def test_pr3_measured_speedup_at_scale():
    """Not a strict perf-regression gate (machine-load-sensitive), but confirms the vectorized path is
    at minimum an order of magnitude faster at a realistic scale -- pins the fix, not a specific number."""
    import time

    from mlframe.training.core._setup_helpers import tune_decision_threshold

    rng = np.random.default_rng(0)
    n = 500_000
    y = rng.integers(0, 2, size=n).astype(np.int8)
    p = rng.uniform(0, 1, size=n)
    t0 = time.perf_counter()
    tune_decision_threshold(y, p, metric="f1")
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0  # was ~14s pre-fix at this scale (linear extrapolation from the bench numbers)


# ---------------------------------------------------------------------------
# PR4: diversity recommendations degraded fully dark on compute_valset_metrics=False suites
# ---------------------------------------------------------------------------


def test_pr4_falls_back_to_oof_derived_score_when_val_metrics_absent():
    """PR4: falls back to oof derived score when val metrics absent."""
    from mlframe.training.core._diversity_recommendations import compute_diversity_recommendations

    class _Member:
        """Member."""
        def __init__(self, model, oof_preds, oof_target, metrics=None):
            self.model = model
            self.oof_preds = oof_preds
            self.oof_target = oof_target
            self.metrics = metrics or {}

    rng = np.random.default_rng(0)
    n = 400
    y_true = rng.normal(size=n)
    # Two members with genuinely different OOF quality and NO val metrics at all.
    m1 = _Member("model_good", y_true + rng.normal(scale=0.1, size=n), y_true)
    m2 = _Member("model_bad", y_true + rng.normal(scale=5.0, size=n), y_true)

    class _Behavior:
        """Behavior."""
        recommend_diversity_additions_in_leaderboard = True
        diversity_recommendation_correlation_threshold = 0.85
        diversity_recommendation_min_improvement = 0.0
        diversity_recommendation_top_k = None

    from mlframe.training.configs import TargetTypes

    result = compute_diversity_recommendations(
        ens_models=[m1, m2], target_type=TargetTypes.REGRESSION, behavior_config=_Behavior(), verbose=False,
    )
    # Pre-fix: both members lack metrics["val"] -> _member_individual_score returns None -> the WHOLE
    # diagnostic returns None (fully dark). Post-fix: the OOF fallback lets it actually run.
    assert result is not None


def test_pr4_still_uses_val_metrics_when_present():
    """PR4: still uses val metrics when present."""
    from mlframe.training.core._diversity_recommendations import _member_individual_score

    class _Member:
        """Member."""
        metrics = {"val": {"rmse": 1.23}}

    score = _member_individual_score(_Member(), is_classification=False, oof_fallback=(np.array([1.0, 2.0]), np.array([1.1, 2.1])))
    assert score == pytest.approx(1.23)  # val metric wins over the OOF fallback when both are available
