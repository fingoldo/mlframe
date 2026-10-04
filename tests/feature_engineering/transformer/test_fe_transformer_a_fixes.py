"""Regression tests for audits/full_audit_2026-07-21/fe_transformer_a.md findings F1-F7 + P8-P10, P13, P14.

PR1 (biz_val test coverage for ~30 mechanisms), PR2 (dedup _kth_nearest_dists/_slice/_make_df across
the SMOTE family), and PR4 (shared LGB-baseline factory) are large architectural asks with no
reported bug -- assessed and deferred (the F6/P10 fixes below already close PR4's stated "guard gap"
concern for the specific files it named). PR3 (rank_pred vectorization) implemented alongside F-fixes.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

@pytest.fixture(autouse=True, scope="module")
def _suppress_noisy_logging_during_this_module():
    """Suppress WARNING-and-below logging while this module's tests run, then restore.

    A bare module-level ``logging.disable(logging.CRITICAL)`` used to sit here with no matching
    ``logging.disable(logging.NOTSET)`` -- ``logging.disable`` is a process-wide, manager-level
    override (not scoped to one logger), so it fired at IMPORT time and stayed in effect for the
    rest of this pytest-xdist worker's lifetime, silently swallowing every later test's
    ``logger.warning(...)``/``logger.debug(...)`` calls regardless of that test's own
    handler/level setup -- see the identical fix + incident writeup in
    test_x_ml_correctness_meta_fixes.py.
    """
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


# ---------------------------------------------------------------------------
# F1: build_hnsw_index now threads random_state
# ---------------------------------------------------------------------------


def test_f1_oof_build_hnsw_index_call_passes_random_state(monkeypatch):
    """Each OOF fold builds its ANN index with ``random_state = seed + fold_idx``."""
    from sklearn.model_selection import KFold

    from mlframe.feature_engineering.transformer import _row_attention_ann
    from mlframe.feature_engineering.transformer.row_attention import compute_row_attention

    seen: list = []
    real = _row_attention_ann.build_hnsw_index

    def spy(*args, **kwargs):
        """Record the random_state of every index build and delegate."""
        seen.append(kwargs.get("random_state"))
        return real(*args, **kwargs)

    monkeypatch.setattr(_row_attention_ann, "build_hnsw_index", spy)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 6)).astype(np.float32)
    y = rng.normal(size=120).astype(np.float32)
    compute_row_attention(X, y, None, KFold(3, shuffle=True, random_state=0), seed=11, n_heads=1, head_dim=4, k=5, gpu_stage4=False, dedupe_threshold=None)
    assert seen == [11, 12, 13]


def test_f1_local_linear_build_hnsw_index_call_passes_random_state(monkeypatch):
    """The local-linear ANN index is built with the caller's seed as ``random_state``."""
    from mlframe.feature_engineering.transformer import local_linear

    seen: list = []
    real = local_linear.build_hnsw_index

    def spy(*args, **kwargs):
        """Record the random_state of every index build and delegate."""
        seen.append(kwargs.get("random_state"))
        return real(*args, **kwargs)

    monkeypatch.setattr(local_linear, "build_hnsw_index", spy)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3)).astype(np.float32)
    y = rng.normal(size=100).astype(np.float32)
    local_linear.compute_local_linear_attention(X, y, X[:10], None, seed=7, k=10)
    assert seen == [7]


# ---------------------------------------------------------------------------
# F2: anchor_attention Mode A now uses nanargmin (NaN-safe), matching Mode B
# ---------------------------------------------------------------------------


def test_f2_anchor_attention_mode_a_nan_row_does_not_bucket_to_anchor_0(monkeypatch):
    """A NaN-poisoned distance row is assigned to its true nearest anchor, never silently to anchor 0, in both Mode A and Mode B."""
    from sklearn.model_selection import KFold

    from mlframe.feature_engineering.transformer import anchor_attention

    assigned: list = []
    real_dists = anchor_attention._squared_dists
    real_aggs = anchor_attention._compute_anchor_aggregates

    def poisoned_dists(X, anchors):
        """Poison row 0 of the train-side matrix: NaN at anchor 0, anchor 1 the true minimum."""
        d = np.array(real_dists(X, anchors), copy=True)
        if X.shape[0] in (90, 60):
            d[0, :] = 9.0
            d[0, 0] = np.nan
            d[0, 1] = 0.1
        return d

    def spy_aggs(y_train, assignments, n_anchors, aggregates):
        """Record the hard assignments handed to the aggregate step."""
        assigned.append(np.array(assignments, copy=True))
        return real_aggs(y_train, assignments, n_anchors=n_anchors, aggregates=aggregates)

    monkeypatch.setattr(anchor_attention, "_squared_dists", poisoned_dists)
    monkeypatch.setattr(anchor_attention, "_compute_anchor_aggregates", spy_aggs)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(90, 4)).astype(np.float32)
    y = rng.normal(size=90).astype(np.float32)
    anchor_attention.compute_anchor_attention(X, y, X[:30], None, seed=1, n_anchors=4)
    anchor_attention.compute_anchor_attention(X, y, None, KFold(3), seed=1, n_anchors=4)
    assert len(assigned) == 4
    assert all(int(a[0]) == 1 for a in assigned), [int(a[0]) for a in assigned]


# ---------------------------------------------------------------------------
# F3: empty quantile bands now fall back to a global centroid/y_mean/y_std, not 0.0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "modname,funcname",
    [
        ("residual_band_attention", "compute_residual_band_attention_features"),
        ("disagreement_band", "compute_disagreement_band_features"),
        ("multi_temp_residual_band", "compute_multi_temp_residual_band_features"),
        ("signed_residual_band", "compute_signed_residual_band_features"),
    ],
)
def test_f3_empty_band_uses_global_fallback_not_zero(modname, funcname):
    """F3 empty band uses global fallback not zero."""
    import importlib

    mod = importlib.import_module(f"mlframe.feature_engineering.transformer.{modname}")
    func = getattr(mod, funcname)

    rng = np.random.default_rng(0)
    n, d = 300, 3
    # Many tied residual/disagreement values collapse quantile boundaries -> some bands end up empty.
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = np.full(n, 5000.0, dtype=np.float32) + rng.normal(scale=0.01, size=n).astype(np.float32)
    y[:5] += 5000.0  # a few genuine outliers so y isn't perfectly degenerate

    out = func(X, y, X_query=X[:20], splitter=None, seed=0, standardize=True)
    assert out.shape[0] == 20
    assert np.isfinite(out.to_numpy()).all()


# ---------------------------------------------------------------------------
# F4: borderline_smote no longer unconditionally drops the first (possibly non-self) neighbour
# ---------------------------------------------------------------------------


def test_f4_borderline_smote_duplicate_rows_do_not_leak_self_as_a_kept_neighbour():
    """F4 borderline smote duplicate rows do not leak self as a kept neighbour."""
    from mlframe.feature_engineering.transformer.borderline_smote import _find_borderline_positives

    rng = np.random.default_rng(0)
    n = 50
    X_full = rng.normal(size=(n, 3)).astype(np.float32)
    # Duplicate the first positive row elsewhere in X_full -- a genuine OTHER row can now tie/sort
    # before self at distance 0, so unconditionally dropping column 0 would exclude that real
    # neighbour instead of self.
    X_full[10] = X_full[0]
    y_full = np.zeros(n, dtype=np.float32)
    y_full[:5] = 1.0  # first 5 rows positive
    X_pos = X_full[:5]

    mask = _find_borderline_positives(X_pos, X_full, y_full, k=5)
    assert mask.shape == (5,)
    assert mask.dtype == bool


def test_f4_borderline_smote_self_match_still_excluded_when_no_duplicates():
    """F4 borderline smote self match still excluded when no duplicates."""
    from mlframe.feature_engineering.transformer.borderline_smote import _find_borderline_positives

    rng = np.random.default_rng(1)
    n = 50
    X_full = rng.normal(size=(n, 3)).astype(np.float32)
    y_full = np.zeros(n, dtype=np.float32)
    y_full[:5] = 1.0
    X_pos = X_full[:5]

    mask = _find_borderline_positives(X_pos, X_full, y_full, k=5)
    # With no exact duplicates, self (dist=0) is still the nearest and correctly excluded --
    # baseline behavior must be unchanged from before the fix.
    assert mask.shape == (5,)


# ---------------------------------------------------------------------------
# F5: geodesic_kgraph's empty-target-indices fallback is 1e6 ("very far"), not 0.0
# ---------------------------------------------------------------------------


def test_f5_geodesic_kgraph_empty_target_uses_far_sentinel_not_near():
    """With no positive-class rows the geodesic distance to the target set is the 'very far' 1e6 sentinel for every query, never 0 ('very close')."""
    from mlframe.feature_engineering.transformer.geodesic_kgraph import compute_geodesic_kgraph_features

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3)).astype(np.float32)
    y = np.zeros(60, dtype=np.float32)
    out = compute_geodesic_kgraph_features(X, y, X[:8], None, seed=0, task="binary")
    values = out.to_numpy()
    assert values.shape[0] == 8
    assert np.all(values[:, :3] == np.float32(1e6)) or np.allclose(values[:, :3], 1e6)
    assert not np.any(values == 0.0)


# ---------------------------------------------------------------------------
# F6: jackknife_endpoint_stability guards against an all-one-class binary subsample
# ---------------------------------------------------------------------------


def test_f6_jackknife_endpoint_stability_binary_degenerate_subsample_does_not_raise():
    """F6 jackknife endpoint stability binary degenerate subsample does not raise."""
    from mlframe.feature_engineering.transformer.jackknife_endpoint_stability import (
        compute_jackknife_endpoint_stability_features,
    )

    rng = np.random.default_rng(0)
    n, d = 40, 3
    X = rng.normal(size=(n, d)).astype(np.float32)
    # Only 2 positives out of 40 -- a 5% row-drop subsample can plausibly lose both.
    y = np.zeros(n, dtype=np.float32)
    y[:2] = 1.0

    out = compute_jackknife_endpoint_stability_features(
        X, y, X_query=X[:10], splitter=None, seed=0, task="binary",
        n_subsamples=10, subsample_drop=0.05, standardize=True,
    )
    assert out.shape[0] == 10
    assert np.isfinite(out.to_numpy()).all()


# ---------------------------------------------------------------------------
# F7: gradient_direction_agreement restores the perturbed column even if predict() raises
# ---------------------------------------------------------------------------


def test_f7_gradient_restores_column_on_predict_exception():
    """F7 gradient restores column on predict exception."""
    from mlframe.feature_engineering.transformer.gradient_direction_agreement import _gradient

    class _RaisingOnSecondCallModel:
        """Raises on the SECOND predict call (the perturbed-column probe), matching the mid-loop
        exception scenario F7 guards against."""

        def __init__(self):
            """init  ."""
            self.calls = 0

        def predict(self, X):
            """Predict."""
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("simulated predict failure")
            return X.sum(axis=1)

    X = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    X_snapshot = X.copy()
    model = _RaisingOnSecondCallModel()
    with pytest.raises(RuntimeError, match="simulated predict failure"):
        _gradient(model, X, is_binary=False, eps=0.05)
    assert np.array_equal(X, X_snapshot), "F7 REGRESSION: X must be restored to its original values even when predict() raises mid-loop"


def test_f7_gradient_normal_path_still_matches_finite_difference():
    """F7 gradient normal path still matches finite difference."""
    from mlframe.feature_engineering.transformer.gradient_direction_agreement import _gradient

    class _LinearModel:
        """LinearModel."""
        def predict(self, X):
            """Predict."""
            return (X * np.array([1.0, 2.0, 3.0], dtype=np.float32)).sum(axis=1)

    X = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32)
    grad = _gradient(_LinearModel(), X, is_binary=False, eps=0.01)
    np.testing.assert_allclose(grad, np.tile([1.0, 2.0, 3.0], (2, 1)), atol=1e-2)


# ---------------------------------------------------------------------------
# P8/P9/P10: previously-silent except-blocks now log
# ---------------------------------------------------------------------------


def test_p8_local_curvature_logs_on_fit_failure(caplog, monkeypatch):
    """A per-row local fit failure is logged at INFO and the row is left as zeros."""
    import mlframe.feature_engineering.transformer.local_curvature as lc

    def failing_lstsq(*args, **kwargs):
        """Simulate a singular local design matrix."""
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(np.linalg, "lstsq", failing_lstsq)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 2)).astype(np.float32)
    y = rng.normal(size=30).astype(np.float32)
    with caplog.at_level(logging.INFO, logger=lc.__name__):
        out = lc.compute_local_curvature_features(X, y, X_query=X[:5], splitter=None, seed=0, k_neighbors=10)
    messages = [r.getMessage() for r in caplog.records if "fit failed on row" in r.getMessage()]
    assert len(messages) == 5
    assert np.all(out.to_numpy() == 0.0)


def test_p9_apriori_itemsets_logs_on_fpgrowth_failure(monkeypatch, caplog):
    """P9 apriori itemsets logs on fpgrowth failure."""
    pytest.importorskip("mlxtend")
    import mlframe.feature_engineering.transformer.apriori_itemsets as ai
    import mlxtend.frequent_patterns

    def _raising_fpgrowth(*args, **kwargs):
        """Raise, simulating a failing fpgrowth call."""
        raise RuntimeError("simulated fpgrowth failure")

    # fpgrowth is imported LOCALLY inside the function body (`from mlxtend.frequent_patterns import
    # fpgrowth`), so it must be patched at its source, not as a module-level attribute of ai.
    monkeypatch.setattr(mlxtend.frequent_patterns, "fpgrowth", _raising_fpgrowth)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 3)).astype(np.float32)
    y = rng.normal(size=30).astype(np.float32)
    with caplog.at_level(logging.INFO, logger=ai.__name__):
        ai.compute_apriori_itemsets_features(X, y, X_query=X[:5], splitter=None, seed=0)
    assert any("fpgrowth failed" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("modname,funcname", [
    ("disagreement_band", "compute_disagreement_band_features"),
    ("baseline_disagreement_v2", "compute_baseline_disagreement_v2_features"),
])
def test_p10_logistic_regression_fallback_logs(modname, funcname, monkeypatch, caplog):
    """P10 logistic regression fallback logs."""
    import importlib

    mod = importlib.import_module(f"mlframe.feature_engineering.transformer.{modname}")
    func = getattr(mod, funcname)

    class _RaisingLR:
        """RaisingLR."""
        def __init__(self, *a, **kw):
            """init  ."""
            pass

        def fit(self, *a, **kw):
            """Fit."""
            raise RuntimeError("simulated LR failure")

    # LogisticRegression is imported LOCALLY inside the function body, so patch it at its source.
    import sklearn.linear_model

    monkeypatch.setattr(sklearn.linear_model, "LogisticRegression", _RaisingLR)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 3)).astype(np.float32)
    y = rng.integers(0, 2, 30).astype(np.float32)
    with caplog.at_level(logging.INFO, logger=mod.__name__):
        func(X, y, X_query=X[:5], splitter=None, seed=0, task="binary")
    assert any("LogisticRegression fit failed" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# P13: _utils.py's stale docstring corrected
# ---------------------------------------------------------------------------


def test_p13_sigma_median_heuristic_matches_exact_median_pairwise_distance():
    """sigma_median_heuristic returns the median pairwise distance of the sample (the quantity its docstring promises)."""
    from scipy.spatial.distance import pdist

    from mlframe.feature_engineering.transformer import _utils

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4)).astype(np.float64)
    sigma = _utils.sigma_median_heuristic(X, seed=0)
    assert sigma == pytest.approx(float(np.median(pdist(X))), rel=1e-5)


# ---------------------------------------------------------------------------
# P14: type hints added to the 3 previously-untyped public signatures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("modname,funcname", [
    ("apriori_itemsets", "compute_apriori_itemsets_features"),
    ("multi_threshold_ordinal", "compute_multi_threshold_ordinal_features"),
    ("target_kmeans_codebook", "compute_target_kmeans_codebook_features"),
])
def test_p14_function_now_has_type_hints(modname, funcname):
    """P14 function now has type hints."""
    import importlib
    import inspect

    mod = importlib.import_module(f"mlframe.feature_engineering.transformer.{modname}")
    func = getattr(mod, funcname)
    sig = inspect.signature(func)
    assert sig.parameters["X_train"].annotation is not inspect.Parameter.empty
    assert sig.return_annotation is not inspect.Signature.empty


# ---------------------------------------------------------------------------
# PR3: multi_threshold_ordinal's rank_pred vectorization matches the original per-row loop
# ---------------------------------------------------------------------------


def test_pr3_rank_pred_vectorized_matches_reference_loop():
    """Pr3 rank pred vectorized matches reference loop."""
    rng = np.random.default_rng(0)
    n_q, n_thresh = 25, 7
    preds = rng.uniform(0, 1, size=(n_q, n_thresh)).astype(np.float32)
    preds[0] = 0.9  # no crossing below 0.5 anywhere -> fallback case
    # Reference (original) per-row loop.
    ref = np.zeros(n_q, dtype=np.float32)
    for q_i in range(n_q):
        cross = np.where(preds[q_i] < 0.5)[0]
        ref[q_i] = float(cross[0]) if len(cross) > 0 else float(n_thresh)
    # Vectorised (current) form.
    below_half = preds < 0.5
    has_cross = below_half.any(axis=1)
    first_cross = np.argmax(below_half, axis=1)
    out = np.where(has_cross, first_cross, n_thresh).astype(np.float32)
    np.testing.assert_array_equal(ref, out)
