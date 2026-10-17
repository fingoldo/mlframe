"""Regression tests for audits/full_audit_2026-07-21/fe_transformer_b.md findings F1-F22, F25.

F18 (class_conditional_anchor.py) already logs a warning before its degenerate-fold skip (the
audit's own note: a lower-severity variant of F3-F8, kept for completeness of the pattern sweep,
not because it was undocumented) -- no code change needed.

F23/F24 (cross-file dedup: _softmax/_topk_within_subset, _K_SCALES/_slice/_kth_nearest_dists),
F26-F28 (perf: per-row loops in local_classifier.py/cluster_smote.py/cutmix.py/persistence_diagram.py),
and PR2 (systematic <2-unique-y guards across 12 MORE files beyond quantile_spread_fan.py, which F16
already covers with a concrete crash mechanism) are large architectural/perf asks with no reported
bug for the other files -- assessed and deferred, documented here only (no code to cite a "won't fix"
comment against).
"""

from __future__ import annotations

import importlib
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
# F1: y_quintile_baseline_knn no longer excludes baseline-prob==1.0 rows from every stratum
# ---------------------------------------------------------------------------


def test_f1_prob_exactly_one_row_included_in_top_stratum(monkeypatch):
    """A baseline probability of exactly 1.0 must land in the top stratum, so its neighbour mean is 1.0.

    With an epsilon-shifted top edge the row drops out, the empty stratum falls back to the whole pool, and the
    mean over k=8 neighbours becomes 0.125 (one 1.0 among eight values).
    """
    from mlframe.feature_engineering.transformer import y_quintile_baseline_knn as mod

    rng = np.random.default_rng(0)
    n, d = 150, 4
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = rng.integers(0, 2, n).astype(np.float32)
    pred = np.zeros(n, dtype=np.float32)
    pred[0] = 1.0
    monkeypatch.setattr(mod, "_fit_baseline_predict", lambda *a, **kw: pred)
    out = mod.compute_y_quintile_baseline_knn_features(X, y, X_query=X[:1], splitter=None, seed=0, task="binary", standardize=False)
    assert out.shape[0] == 1
    assert np.isfinite(out.to_numpy()).all()
    assert out["yqbk_q4_mean"].to_numpy()[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# F2: pairwise_kl_divergence's Gaussian JS keeps per-row mixture sigma, not a query-wide scalar
# ---------------------------------------------------------------------------


def test_f2_js_sigma_is_per_row_not_scalar():
    """F2 js sigma is per row not scalar: a row's JS value must not depend on which other rows share its query batch."""
    from mlframe.feature_engineering.transformer.pairwise_kl_divergence import compute_pairwise_kl_features

    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4)).astype(np.float32)
    y = rng.normal(size=300).astype(np.float32)
    batch = compute_pairwise_kl_features(X, y, X_query=X[:60], splitter=None, seed=0, task="regression")
    single = compute_pairwise_kl_features(X, y, X_query=X[:1], splitter=None, seed=0, task="regression")
    js_col = next(c for c in batch.columns if c.endswith("_js"))
    # a query-set-wide scalar sigma (the pre-fix behaviour) would make row 0's value shift with the batch around it
    assert batch[js_col].to_numpy()[0] == pytest.approx(single[js_col].to_numpy()[0])


def test_f2_js_varies_across_rows_with_genuinely_different_spread():
    """F2 js varies across rows with genuinely different spread."""
    from mlframe.feature_engineering.transformer.pairwise_kl_divergence import compute_pairwise_kl_features

    rng = np.random.default_rng(0)
    n, d = 300, 4
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = rng.normal(size=n).astype(np.float32)
    out = compute_pairwise_kl_features(X, y, X_query=X[:60], splitter=None, seed=0, task="regression")
    js = out["pklkl_js"].to_numpy() if "pklkl_js" in out.columns else out[next(c for c in out.columns if c.endswith("_js"))].to_numpy()
    assert np.isfinite(js).all()
    assert js.std() > 0, "F2 REGRESSION: js must retain per-row structure, not collapse to a near-constant scalar-derived value"


# ---------------------------------------------------------------------------
# F3-F8: degenerate-fold fallback uses a consistent sentinel, not a misleading 0.0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("modname,funcname", [
    ("cluster_smote", "compute_cluster_smote_features"),
    ("bgmm_virtual", "compute_bgmm_virtual_features"),
    ("diffusion_noise", "compute_diffusion_noise_features"),
    ("cutmix", "compute_cutmix_features"),
])
def test_f3_f6_degenerate_fold_uses_far_sentinel_not_zero(modname, funcname):
    """F3 f6 degenerate fold uses far sentinel not zero."""
    mod = importlib.import_module(f"mlframe.feature_engineering.transformer.{modname}")
    func = getattr(mod, funcname)

    rng = np.random.default_rng(0)
    n, d = 20, 3
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = np.zeros(n, dtype=np.float32)
    y[0] = 1.0  # only 1 positive -> degenerate fold (< 2 positives)

    out = func(X, y, X_query=X[:5], splitter=None, seed=0, task="binary")
    pos_cols = [c for c in out.columns if "loggap" not in c.lower()]
    loggap_cols = [c for c in out.columns if "loggap" in c.lower()]
    assert pos_cols, f"{modname} must emit distance columns, got {list(out.columns)}"
    assert loggap_cols, f"{modname} must emit loggap columns, got {list(out.columns)}"
    pos_vals = out[pos_cols].to_numpy()
    assert np.all(pos_vals >= 1e5), f"F3-F6 REGRESSION: {modname} degenerate-fold distance columns must use the far sentinel, not 0.0: {pos_vals}"
    loggap_vals = out[loggap_cols].to_numpy()
    assert np.allclose(loggap_vals, 0.0)


def test_f7_bgmm_density_ratio_degenerate_fold_uses_low_density_sentinel():
    """F7 bgmm density ratio degenerate fold uses low density sentinel."""
    from mlframe.feature_engineering.transformer.bgmm_density_ratio import compute_bgmm_density_ratio_features

    rng = np.random.default_rng(0)
    n, d = 20, 3
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = np.zeros(n, dtype=np.float32)
    y[0] = 1.0

    out = compute_bgmm_density_ratio_features(X, y, X_query=X[:5], splitter=None, seed=0, task="binary")
    logp_cols = [c for c in out.columns if "log_ratio" not in c and ("logp" in c or "log_p" in c)]
    assert logp_cols, f"bgmm_density_ratio must emit log-density columns, got {list(out.columns)}"
    vals = out[logp_cols].to_numpy()
    assert np.allclose(vals, -30.0), f"F7 REGRESSION: degenerate-fold log-density columns must use the -30.0 sentinel, not 0.0: {vals}"


def test_f8_class_mahalanobis_degenerate_fold_uses_far_sentinel():
    """F8 class mahalanobis degenerate fold uses far sentinel."""
    from mlframe.feature_engineering.transformer.class_mahalanobis import compute_class_mahalanobis_features

    rng = np.random.default_rng(0)
    n, d = 20, 3
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = np.zeros(n, dtype=np.float32)
    y[0] = 1.0

    out = compute_class_mahalanobis_features(X, y, X_query=X[:5], splitter=None, seed=0)
    m_pos = out["mahcc_m_pos"].to_numpy() if "mahcc_m_pos" in out.columns else out[next(c for c in out.columns if "m_pos" in c)].to_numpy()
    assert np.all(m_pos >= 1e5), f"F8 REGRESSION: degenerate-fold m_pos/m_neg must use the far sentinel, not 0.0 ('at the class centroid'): {m_pos}"


# ---------------------------------------------------------------------------
# F9/F10: empty-side anchor fallback no longer repeats an arbitrary real row (index 0)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("modname,funcname", [
    ("multi_baseline_hard_row", "compute_multi_baseline_hard_row_features"),
    ("class_balanced_hard_row", "compute_class_balanced_hard_row_features"),
])
def test_f9_f10_empty_side_no_longer_repeats_index_zero(modname, funcname):
    """F9 f10 empty side no longer repeats index zero: an empty side's anchors are the far sentinel, not a real row."""
    mod = importlib.import_module(f"mlframe.feature_engineering.transformer.{modname}")
    func = getattr(mod, funcname)

    rng = np.random.default_rng(0)
    n, d = 40, 3
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = np.zeros(n, dtype=np.float32)  # ALL negative -> the "positive" side is fully empty
    out = func(X, y, X_query=X[:10], splitter=None, seed=0, task="binary")
    assert out.shape[0] == 10
    assert np.isfinite(out.to_numpy()).all()
    mindist_empty = out[next(c for c in out.columns if c.endswith("mindist_side1"))].to_numpy()
    mindist_real = out[next(c for c in out.columns if c.endswith("mindist_side2"))].to_numpy()
    # repeating real row 0 as the empty side's anchors would give it a near-zero distance for the query row X[0]
    assert (mindist_empty > 1e6).all()
    assert (mindist_real < mindist_empty).all()


# ---------------------------------------------------------------------------
# F11-F14: previously-silent except-blocks now log
# ---------------------------------------------------------------------------


def test_f11_tree_path_boolean_logs_on_extraction_failure(monkeypatch, caplog):
    """F11 tree path boolean logs on extraction failure."""
    import mlframe.feature_engineering.transformer.tree_path_boolean as mod

    def _raising(*a, **kw):
        """Raise, simulating a failing call."""
        raise RuntimeError("simulated path extraction failure")

    monkeypatch.setattr(mod, "_extract_top_paths", _raising)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 3)).astype(np.float32)
    y = rng.normal(size=30).astype(np.float32)
    with caplog.at_level(logging.INFO, logger=mod.__name__):
        mod.compute_tree_path_boolean_features(X, y, X_query=X[:5], splitter=None, seed=0, task="regression")
    assert any("path extraction failed" in r.getMessage() for r in caplog.records)


def test_f13_fca_closed_concepts_logs_on_lattice_failure(monkeypatch, caplog):
    """A failing lattice build is logged at INFO and degrades to zero concepts: top_k + 2 columns, n_concepts == 0, no indicator fires."""
    pytest.importorskip("concepts")
    import concepts
    import mlframe.feature_engineering.transformer.fca_closed_concepts as mod

    def _raising_context(*args, **kwargs):
        """Raise, simulating a failing lattice construction."""
        raise RuntimeError("simulated concepts lib failure")

    monkeypatch.setattr(concepts, "Context", _raising_context)
    rng = np.random.default_rng(0)
    X = (rng.random((30, 3)) > 0.5).astype(np.float32)
    y = rng.normal(size=30).astype(np.float32)
    with caplog.at_level(logging.INFO, logger=mod.__name__):
        out = mod.compute_fca_closed_concepts_features(X, y, X_query=X[:5], splitter=None, seed=0)
    messages = [r.getMessage() for r in caplog.records if "lattice construction failed" in r.getMessage()]
    assert messages, "the lattice failure must be logged"
    assert "simulated concepts lib failure" in messages[0]
    arr = out.to_numpy()
    assert arr.shape[0] == 5
    n_concepts = [c for c in out.columns if c.endswith("_n_concepts")]
    assert len(n_concepts) == 1
    assert np.all(out[n_concepts[0]].to_numpy() == 0.0)
    assert np.all(arr == 0.0)


def test_f14_multi_baseline_hard_row_logs_on_logreg_failure(monkeypatch, caplog):
    """F14 multi baseline hard row logs on logreg failure."""
    import mlframe.feature_engineering.transformer.multi_baseline_hard_row as mod
    import sklearn.linear_model

    class _RaisingLR:
        """RaisingLR."""
        def __init__(self, *a, **kw):
            """init  ."""
            pass

        def fit(self, *a, **kw):
            """Fit."""
            raise RuntimeError("simulated LR failure")

    monkeypatch.setattr(sklearn.linear_model, "LogisticRegression", _RaisingLR)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 3)).astype(np.float32)
    y = rng.integers(0, 2, 30).astype(np.float32)
    with caplog.at_level(logging.INFO, logger=mod.__name__):
        mod.compute_multi_baseline_hard_row_features(X, y, X_query=X[:5], splitter=None, seed=0, task="binary")
    assert any("LogisticRegression fit failed" in r.getMessage() for r in caplog.records)


def test_f12_persistence_diagram_logs_on_gudhi_row_failure(monkeypatch, caplog):
    """F12: a per-row gudhi failure must be logged (row left zero-filled but visibly so), not silently
    swallowed by a bare ``except Exception: pass``."""
    gudhi = pytest.importorskip("gudhi")
    import mlframe.feature_engineering.transformer.persistence_diagram as mod

    rng = np.random.default_rng(0)
    n, d = 20, 3
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = rng.normal(size=n).astype(np.float32)

    def _raising_rips_complex(*a, **kw):
        """Raise, simulating a gudhi RipsComplex construction failure on every row."""
        raise RuntimeError("simulated gudhi failure")

    monkeypatch.setattr(gudhi, "RipsComplex", _raising_rips_complex)
    with caplog.at_level(logging.INFO, logger=mod.__name__):
        out = mod.compute_persistence_diagram_features(X, y, X_query=X[:5], splitter=None, seed=0)
    assert any("gudhi computation failed" in r.getMessage() for r in caplog.records)
    # Rows still come back zero-filled (not raised through to the caller).
    assert out.shape[0] == 5
    assert np.allclose(out.to_numpy(), 0.0)


# ---------------------------------------------------------------------------
# F15/F17: mdl_binning_pairwise honours the caller's explicit `task` param
# ---------------------------------------------------------------------------


def test_f15_mdl_binning_honours_explicit_binary_task_for_nonstandard_labels():
    """F15 mdl binning honours explicit binary task for nonstandard labels."""
    from mlframe.feature_engineering.transformer.mdl_binning_pairwise import compute_mdl_binning_pairwise_features

    rng = np.random.default_rng(0)
    n, d = 100, 4
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = rng.choice([1.0, 2.0], size=n).astype(np.float32)  # {1,2}-coded binary, NOT {0,1}

    out_binary = compute_mdl_binning_pairwise_features(X, y, X_query=X[:20], splitter=None, seed=0, task="binary", max_bins_per_feat=4)
    # With task="binary" honoured, y_class has only 2 distinct values (n_classes=2), which bounds
    # the MDL bin-edge count differently than the auto-detected 5-class quantile path would.
    assert out_binary.shape[0] == 20
    assert np.isfinite(out_binary.to_numpy()).all()

    # task="binary" must map the caller's label CODES to {0,1} class indices, not use them as
    # indices directly -- the {0,1}-relabelled target carries the identical partition, so the
    # binning result must be identical regardless of which two raw values encode the classes.
    # Regression for a real bug: y_t.astype(np.int32) fed {1,2}-coded labels straight into a
    # fixed-size (n_classes=2,) count array as an index, silently overrunning it under real JIT
    # (no bounds check) and raising IndexError only with NUMBA_DISABLE_JIT=1.
    y_relabelled = np.where(y == 1.0, 0.0, 1.0).astype(np.float32)
    out_relabelled = compute_mdl_binning_pairwise_features(X, y_relabelled, X_query=X[:20], splitter=None, seed=0, task="binary", max_bins_per_feat=4)
    np.testing.assert_array_equal(out_binary.to_numpy(), out_relabelled.to_numpy())


def test_f17_no_dead_else_branch_remains():
    """F17 no dead else branch remains: the module no longer compares ``.dtype`` against ``np.int32`` (checked on the AST)."""
    import ast
    from pathlib import Path

    from mlframe.feature_engineering.transformer import mdl_binning_pairwise as mod

    tree = ast.parse(Path(mod.__file__).read_bytes())
    dtype_vs_int32 = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Attribute)
        and node.left.attr == "dtype"
        and any(isinstance(c, ast.Attribute) and c.attr == "int32" for c in node.comparators)
    ]
    assert dtype_vs_int32 == []


# ---------------------------------------------------------------------------
# F16: quantile_spread_fan guards against a single-class binary fold
# ---------------------------------------------------------------------------


def test_f16_quantile_spread_fan_single_class_fold_does_not_raise():
    """F16 quantile spread fan single class fold does not raise."""
    from mlframe.feature_engineering.transformer.quantile_spread_fan import compute_quantile_spread_fan_features

    rng = np.random.default_rng(0)
    n, d = 30, 3
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = np.zeros(n, dtype=np.float32)  # single-class fold

    out = compute_quantile_spread_fan_features(X, y, X_query=X[:10], splitter=None, seed=0, task="binary")
    assert out.shape[0] == 10
    assert np.isfinite(out.to_numpy()).all()


# ---------------------------------------------------------------------------
# F19-F22: empty quantile band/bin fallback uses a neutral global value, not 0.0
# ---------------------------------------------------------------------------


def test_f19_prediction_band_attention_empty_band_uses_global_fallback():
    """F19 prediction band attention empty band uses global fallback."""
    from mlframe.feature_engineering.transformer.prediction_band_attention import compute_prediction_band_attention_features

    rng = np.random.default_rng(0)
    n, d = 300, 3
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = np.full(n, 5000.0, dtype=np.float32) + rng.normal(scale=0.01, size=n).astype(np.float32)
    out = compute_prediction_band_attention_features(X, y, X_query=X[:20], splitter=None, seed=0, task="regression")
    assert np.isfinite(out.to_numpy()).all()


def test_f21_predictive_info_delta_empty_bin_uses_marginal_entropy(monkeypatch):
    """F21 predictive info delta empty bin uses marginal entropy: a query landing in a bin with no train rows gets delta 0."""
    import lightgbm

    from mlframe.feature_engineering.transformer.predictive_info_delta import compute_predictive_info_delta_features

    class _TwoClusterRegressor:
        """Fake baseline: train predictions equal y (two clusters), every query prediction lies above all train predictions."""

        def __init__(self, **kwargs):
            """Accept and ignore the LightGBM hyper-parameters."""

        def fit(self, X, y):
            """Remember the train targets so ``predict`` can tell train rows from query rows."""
            self._n_train = len(y)
            self._y = np.asarray(y, dtype=np.float32)
            return self

        def predict(self, X):
            """Return y for the train matrix and 2.0 (beyond every train prediction) for the query matrix."""
            return self._y if len(X) == self._n_train else np.full(len(X), 2.0, dtype=np.float32)

    monkeypatch.setattr(lightgbm, "LGBMRegressor", _TwoClusterRegressor)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3)).astype(np.float32)
    y = (np.arange(100) % 2).astype(np.float32)
    out = compute_predictive_info_delta_features(X, y, X_query=X[:5], splitter=None, seed=0, task="regression", n_bins=4, standardize=False)
    # the top bin (p > last edge) is empty in train, so its conditional entropy falls back to the marginal one
    assert (out["pinfo_bin"].to_numpy() == 3).all()
    np.testing.assert_allclose(out["pinfo_H_given_bin"].to_numpy(), out["pinfo_H_marginal"].to_numpy())
    np.testing.assert_allclose(out["pinfo_delta"].to_numpy(), 0.0, atol=1e-6)


def test_f22_ib_baseline_codes_empty_cell_uses_global_fallback(monkeypatch):
    """F22 ib baseline codes empty cell uses global fallback: a query whose baseline code has no train rows gets the global y mean/std."""
    from mlframe.feature_engineering.transformer import ib_baseline_codes as mod

    def _fake_baselines(Xt, y_t, Xq, task, seed):
        """Train predictions mirror y in all 3 baselines (codes 0 or 7 only); query predictions set only bit 0 (code 1, unseen)."""
        tp = np.column_stack([y_t, y_t, y_t]).astype(np.float32)
        qp = np.column_stack([np.full(len(Xq), 2.0), np.zeros(len(Xq)), np.zeros(len(Xq))]).astype(np.float32)
        return tp, qp

    monkeypatch.setattr(mod, "_fit_3baselines_two", _fake_baselines)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3)).astype(np.float32)
    y = (np.arange(100) % 4 == 0).astype(np.float32)
    out = mod.compute_ib_baseline_codes_features(X, y, X_query=X[:5], splitter=None, seed=0, task="regression", standardize=False)
    cols = out.columns
    assert (out[cols[0]].to_numpy() == 1).all()
    np.testing.assert_allclose(out[cols[1]].to_numpy(), y.mean(), rtol=1e-5)
    np.testing.assert_allclose(out[cols[2]].to_numpy(), y.std() + 1e-9, rtol=1e-4)


def test_f20_fisher_weighted_residual_empty_band_uses_global_fallback(monkeypatch):
    """F20: an empty Fisher-weighted-residual band (tied values collapsing a quantile boundary) must
    fall back to the global target mean, not a misleading ``0.0`` that reads as "genuinely low"."""
    import lightgbm

    from mlframe.feature_engineering.transformer import fisher_weighted_residual as mod

    class _FakeRegressor:
        """Fake baseline model: every query prediction sits 2.0 above the train-target median."""

        def __init__(self, **kwargs):
            """Accept and ignore the LightGBM hyper-parameters."""

        def fit(self, X, y):
            """Remember the train median."""
            self._median = float(np.median(y))
            return self

        def predict(self, X):
            """Return a constant prediction far above the train targets."""
            return np.full(len(X), self._median + 2.0, dtype=np.float32)

    # residuals of 0 / 1 tie heavily, so the top quantile band (weighted residual > max train value) is empty in train
    monkeypatch.setattr(lightgbm, "LGBMRegressor", _FakeRegressor)
    monkeypatch.setattr(mod, "_oof_train_predictions", lambda Xt, y_t, is_binary, seed: (y_t - (y_t > 0.5)).astype(np.float32))
    monkeypatch.setattr(mod, "_finite_difference_grad_norm", lambda model, X, is_binary, eps: np.ones(len(X), dtype=np.float32))
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3)).astype(np.float32)
    y = (np.arange(100) % 2).astype(np.float32)
    out = mod.compute_fisher_weighted_residual_features(X, y, X_query=X[:5], splitter=None, seed=0, task="regression", standardize=False)
    assert (out["fishres_band"].to_numpy() == 4).all()
    # a 0.0 fallback for the empty band would read as a genuinely low target rather than "no data"
    np.testing.assert_allclose(out["fishres_band_y_mean"].to_numpy(), y.mean())


def test_f20_fisher_weighted_residual_runs_clean_on_tied_residuals():
    """A dataset with many tied weighted-residual values (likely to collapse a quantile boundary and
    produce at least one empty band) must still return finite output using the global-mean fallback."""
    from mlframe.feature_engineering.transformer.fisher_weighted_residual import compute_fisher_weighted_residual_features

    rng = np.random.default_rng(0)
    n, d = 200, 3
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = np.where(rng.random(n) < 0.9, 0.0, 1.0).astype(np.float32)  # heavily tied -> empty-band-prone
    out = compute_fisher_weighted_residual_features(X, y, X_query=X[:20], splitter=None, seed=0, task="binary")
    assert np.isfinite(out.to_numpy()).all()


# ---------------------------------------------------------------------------
# F29: trust_score_oof's empty-subset sentinel is already documented in the docstring -- lower-severity
# by the report's own verdict, pinned here so the disposition is tracked, not merely implied.
# ---------------------------------------------------------------------------


def test_f29_nn_dist_empty_subset_sentinel_is_documented():
    """F29: the ``nearest=0.0``-for-empty-subset sentinel must stay explicitly documented in
    ``_nn_dist``'s own docstring (the report's basis for treating this as a documented tradeoff, not a
    silent one, unlike F3-F8's undocumented zero-sentinels)."""
    import ast
    from pathlib import Path

    from mlframe.feature_engineering.transformer import trust_score_oof as mod

    tree = ast.parse(Path(mod.__file__).read_bytes())
    nn_dist = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_nn_dist"]
    assert nn_dist
    assert all("sentinel fill values" in (ast.get_docstring(node) or "") for node in nn_dist)


# ---------------------------------------------------------------------------
# F25: swap_noise never assigns a swapped cell its own original value
# ---------------------------------------------------------------------------


def test_f25_swap_noise_never_self_assigns():
    """F25 swap noise never self assigns."""
    from mlframe.feature_engineering.transformer.swap_noise import swap_noise_augment

    n, d = 20, 3
    X = np.arange(n * d, dtype=np.float64).reshape(n, d)  # every cell unique -> self-match is detectable

    # swap_prob=1.0 forces every cell to be a swap candidate, maximizing self-collision probability
    # for this seed across many repeated calls.
    any_self_match = False
    for trial_seed in range(50):
        out = swap_noise_augment(X, swap_prob=1.0, rng=np.random.default_rng(trial_seed))
        if np.any(out == X):
            any_self_match = True
            break
    assert not any_self_match, "F25 REGRESSION: a swapped cell must never retain its own original value"
