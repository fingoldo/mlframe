"""F-34 E3: NNLS per-column weight learning for the MTR CT_ENSEMBLE.

Verifies that:
  * MTRPerColumnEqualMeanEnsemble with strategy='nnls' + .fit(X, y)
    learns per-column weights that sum to 1 per column
  * predict() applies the learned weights correctly (verified by
    constructing a problem where the optimal weights are known)
  * Degenerate NNLS solution (all-zero components for a column) falls
    back to equal-mean for THAT column
  * Pre-supplied weights via __init__(weights=...) take precedence
  * Dispatcher wiring: when filtered_val_df + val y are available,
    the ensemble is NNLS; otherwise equal_mean.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mlframe.training import TargetTypes
from mlframe.training.core._phase_composite_post_xt_ensemble import (
    MTRPerColumnEqualMeanEnsemble,
    _build_cross_target_ensemble_for_target,
)


class _FixedComponent:
    """Returns a fixed (N, K) prediction independent of X (so we can
    test weight-learning without confounding from fit-time noise)."""

    def __init__(self, preds):
        self._preds = np.asarray(preds, dtype=np.float64)

    def predict(self, X):
        n = len(X) if hasattr(X, "__len__") else self._preds.shape[0]
        if self._preds.shape[0] == n:
            return self._preds
        # If X size differs (eg val vs train), tile the first row.
        return np.tile(self._preds[:1], (n, 1))


def test_nnls_learns_optimal_weights_two_components():
    """Two components emitting (0, 0, ...) and (2, 2, ...) on each
    column. True y = (1, 1, ...). NNLS finds weights such that
    w1 * 0 + w2 * 2 = 1, i.e. w2 = 0.5 per column (w1 is degenerate
    since multiplying by zero contributes nothing).

    Verifies the OUTPUT (predictions recover y), not the weights
    directly -- weight under-determination is normal when components'
    columns are collinear or zero."""
    n, k = 30, 2
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 4))
    c1 = _FixedComponent(np.zeros((n, k)))
    c2 = _FixedComponent(np.full((n, k), 2.0))
    y = np.ones((n, k))

    ens = MTRPerColumnEqualMeanEnsemble(
        components=[c1, c2], component_names=["c1", "c2"],
        n_targets=k, strategy="nnls",
    )
    ens.fit(X, y)
    # predict should recover ~1 everywhere (the only assertion that
    # makes sense when one component column is all-zero).
    preds = ens.predict(X)
    np.testing.assert_allclose(preds, y, atol=0.01)
    # The c2 weight should be ~0.5 (only contributor).
    w = ens.weights
    assert w.shape == (2, k)
    np.testing.assert_allclose(w[1, :], np.full(k, 0.5), atol=0.05)


def test_nnls_per_column_independence():
    """Three components, K=2 targets. Each target's true y favours a
    DIFFERENT component subset. Verifies the per-column independence
    of the NNLS solve."""
    n, k = 40, 2
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 3))
    # Component 1: emits (5, 0) -- helps column 0 only.
    c1 = _FixedComponent(np.tile([5.0, 0.0], (n, 1)))
    # Component 2: emits (0, 3) -- helps column 1 only.
    c2 = _FixedComponent(np.tile([0.0, 3.0], (n, 1)))
    # Component 3: emits noise (0, 0) -- irrelevant.
    c3 = _FixedComponent(np.zeros((n, k)))
    # True y: column 0 = 5, column 1 = 3.
    y = np.tile([5.0, 3.0], (n, 1))

    ens = MTRPerColumnEqualMeanEnsemble(
        components=[c1, c2, c3], component_names=["c1", "c2", "c3"],
        n_targets=k, strategy="nnls",
    )
    ens.fit(X, y)
    w = ens.weights
    # Column 0 should be dominated by c1.
    assert w[0, 0] > 0.9
    assert w[1, 0] < 0.1
    # Column 1 should be dominated by c2.
    assert w[1, 1] > 0.9
    assert w[0, 1] < 0.1
    # predict should recover y exactly.
    preds = ens.predict(X)
    np.testing.assert_allclose(preds, y, atol=0.01)


def test_nnls_degenerate_zero_column_falls_back_to_equal_mean():
    """When all components emit 0 for a column and y for that column
    is nonzero, NNLS returns all-zero weights. The class falls back to
    equal_mean for THAT column."""
    n, k = 20, 2
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 3))
    # Both components emit (0, 1) -- column 0 has no signal.
    c1 = _FixedComponent(np.tile([0.0, 1.0], (n, 1)))
    c2 = _FixedComponent(np.tile([0.0, 1.0], (n, 1)))
    # True y: column 0 = 7 (impossible to fit), column 1 = 1.
    y = np.tile([7.0, 1.0], (n, 1))

    ens = MTRPerColumnEqualMeanEnsemble(
        components=[c1, c2], component_names=["c1", "c2"],
        n_targets=k, strategy="nnls",
    )
    ens.fit(X, y)
    w = ens.weights
    # Column 0: NNLS returns all-zero -> fallback to (0.5, 0.5).
    np.testing.assert_allclose(w[:, 0], [0.5, 0.5], atol=1e-6)
    # Column 1: NNLS finds (0.5, 0.5) (both components emit same value).
    np.testing.assert_allclose(w[:, 1].sum(), 1.0, atol=1e-6)


def test_supplied_weights_take_precedence():
    """Explicit weights via __init__(weights=...) skip the NNLS step
    AND skip the equal_mean default. Useful when external code (e.g.
    a future PR's honest-OOF stack) pre-computes weights."""
    n, k = 10, 2
    c1 = _FixedComponent(np.full((n, k), 1.0))
    c2 = _FixedComponent(np.full((n, k), 3.0))
    custom_w = np.array([[0.25, 0.75], [0.75, 0.25]])  # (n_comp=2, K=2)
    ens = MTRPerColumnEqualMeanEnsemble(
        components=[c1, c2], component_names=["c1", "c2"],
        n_targets=k, weights=custom_w,
    )
    assert ens.strategy == "nnls"
    np.testing.assert_array_equal(ens.weights, custom_w)
    # predict applies the supplied weights.
    preds = ens.predict(np.zeros((n, 3)))
    # col 0: 0.25*1 + 0.75*3 = 2.5; col 1: 0.75*1 + 0.25*3 = 1.5
    expected = np.tile([2.5, 1.5], (n, 1))
    np.testing.assert_allclose(preds, expected, atol=1e-6)


def test_supplied_weights_wrong_shape_raises():
    n, k = 5, 2
    c1 = _FixedComponent(np.zeros((n, k)))
    with pytest.raises(ValueError, match=r"weights shape"):
        MTRPerColumnEqualMeanEnsemble(
            components=[c1], component_names=["c1"],
            n_targets=k, weights=np.zeros((3, 5)),
        )


def test_invalid_strategy_raises():
    n, k = 5, 2
    c1 = _FixedComponent(np.zeros((n, k)))
    with pytest.raises(ValueError, match=r"strategy must be"):
        MTRPerColumnEqualMeanEnsemble(
            components=[c1], component_names=["c1"],
            n_targets=k, strategy="not_a_strategy",
        )


def test_equal_mean_fit_is_noop():
    """fit() on equal_mean strategy is a no-op (no held-out data
    needed for equal weights). Returns self."""
    n, k = 5, 2
    c1 = _FixedComponent(np.zeros((n, k)))
    c2 = _FixedComponent(np.ones((n, k)))
    ens = MTRPerColumnEqualMeanEnsemble(
        components=[c1, c2], component_names=["c1", "c2"],
        n_targets=k, strategy="equal_mean",
    )
    w_before = ens.weights
    ret = ens.fit(np.zeros((5, 3)), np.zeros((5, k)))
    assert ret is ens
    np.testing.assert_array_equal(ens.weights, w_before)


# --- Dispatcher: NNLS path when val data is available ------------------------


def _make_models_dict_for_dispatcher(n_components=3, n_targets=2, n=30):
    target_type = TargetTypes.MULTI_TARGET_REGRESSION
    target_name = "mtr_target"
    rng = np.random.default_rng(0)
    entries = []
    # Each component emits a fixed prediction; component i emits column
    # values (i, i*2) tiled.
    for i in range(n_components):
        comp_preds = np.tile([float(i), float(i * 2)], (n, 1))
        entries.append(SimpleNamespace(
            model=_FixedComponent(comp_preds), pre_pipeline=None,
        ))
    return {target_type: {target_name: entries}}, target_type, target_name


def test_dispatcher_uses_nnls_oof_when_train_data_available():
    """Dispatcher derives per-column NNLS weights from an honest train-K-fold OOF stack.

    The legacy val-fit path was removed because fitting per-column weights on the val fold double-dipped the
    components' early-stopping surface. The dispatcher now needs ``filtered_train_df`` + ``filtered_train_idx`` +
    train y to run the leak-free OOF refit; given those it builds a ``per_column_nnls_oof`` ensemble with
    non-negative weights. Components must be sklearn-cloneable for the OOF refit, so use real regressors.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.multioutput import MultiOutputRegressor

    tt = TargetTypes.MULTI_TARGET_REGRESSION
    tn = "mtr_target"
    n, k, p = 60, 2, 4
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, p))
    coef = rng.normal(size=(p, k))
    y = X @ coef + 0.01 * rng.normal(size=(n, k))

    entries = []
    for _ in range(3):
        m = MultiOutputRegressor(LinearRegression())
        m.fit(X, y)
        entries.append(SimpleNamespace(model=m, pre_pipeline=None))
    models = {tt: {tn: entries}}
    metadata = {}
    cache = {}
    train_idx = np.arange(n)
    target_by_type = {tt: {tn: y}}

    _build_cross_target_ensemble_for_target(
        _tt_e=tt, _orig_tname=tn,
        _spec_list=[], _ce_strategy="weighted_mean",
        models=models, metadata=metadata,
        target_by_type=target_by_type,
        composite_target_discovery_config=None,
        target_name=tn, model_name="test",
        filtered_train_df=X,
        filtered_val_df=None,
        test_df_pd=None,
        filtered_train_idx=train_idx, filtered_val_idx=None,
        test_idx=None,
        train_df_pd=None, val_df_pd=None,
        train_idx=None, val_idx=None,
        reporting_config=None, plot_file=None,
        _train_pred_cache=cache,
    )
    # 3 originals + 1 ensemble = 4.
    assert len(models[tt][tn]) == 4
    ens_entry = models[tt][tn][-1]
    assert ens_entry.ensemble_strategy == "per_column_nnls_oof"
    assert ens_entry.model.strategy == "nnls"
    # Weights are non-negative (NNLS constraint) per column.
    assert (ens_entry.model.weights >= 0).all()


def test_dispatcher_falls_back_to_equal_mean_without_train_data():
    """Without train data the honest OOF refit cannot run -> equal_mean (no val-fit fallback)."""
    n, k = 30, 2
    models, tt, tn = _make_models_dict_for_dispatcher(n_components=3, n_targets=k, n=n)
    metadata = {}
    cache = {}
    rng = np.random.default_rng(0)
    val_df = rng.normal(size=(n, 4))
    target_by_type = {tt: {tn: np.tile([2.0, 4.0], (n, 1))}}

    _build_cross_target_ensemble_for_target(
        _tt_e=tt, _orig_tname=tn,
        _spec_list=[], _ce_strategy="weighted_mean",
        models=models, metadata=metadata,
        target_by_type=target_by_type,
        composite_target_discovery_config=None,
        target_name=tn, model_name="test",
        filtered_train_df=None,
        filtered_val_df=val_df,
        test_df_pd=None,
        filtered_train_idx=None, filtered_val_idx=None,
        test_idx=None,
        train_df_pd=None, val_df_pd=None,
        train_idx=None, val_idx=None,
        reporting_config=None, plot_file=None,
        _train_pred_cache=cache,
    )
    assert len(models[tt][tn]) == 4
    ens_entry = models[tt][tn][-1]
    assert ens_entry.ensemble_strategy == "per_column_equal_mean"
    assert ens_entry.model.strategy == "equal_mean"


def test_dispatcher_falls_back_to_equal_mean_without_val_data():
    """No val data -> equal_mean (the legacy default)."""
    n, k = 20, 2
    models, tt, tn = _make_models_dict_for_dispatcher(n_components=3, n_targets=k, n=n)
    metadata = {}
    cache = {}
    target_by_type = {tt: {tn: np.zeros((n, k))}}

    _build_cross_target_ensemble_for_target(
        _tt_e=tt, _orig_tname=tn,
        _spec_list=[], _ce_strategy="weighted_mean",
        models=models, metadata=metadata,
        target_by_type=target_by_type,
        composite_target_discovery_config=None,
        target_name=tn, model_name="test",
        filtered_train_df=None,
        filtered_val_df=None,  # <- no val data
        test_df_pd=None,
        filtered_train_idx=None, filtered_val_idx=None,
        test_idx=None,
        train_df_pd=None, val_df_pd=None,
        train_idx=None, val_idx=None,
        reporting_config=None, plot_file=None,
        _train_pred_cache=cache,
    )
    ens_entry = models[tt][tn][-1]
    assert ens_entry.ensemble_strategy == "per_column_equal_mean"
    assert ens_entry.model.strategy == "equal_mean"


def test_repr_includes_strategy():
    """repr should surface strategy so logs / debugger output is
    informative."""
    n, k = 5, 2
    c1 = _FixedComponent(np.zeros((n, k)))
    c2 = _FixedComponent(np.ones((n, k)))
    ens_em = MTRPerColumnEqualMeanEnsemble(
        components=[c1, c2], component_names=["c1", "c2"], n_targets=k,
    )
    assert "strategy='equal_mean'" in repr(ens_em)
    ens_nnls = MTRPerColumnEqualMeanEnsemble(
        components=[c1, c2], component_names=["c1", "c2"], n_targets=k,
        strategy="nnls",
    )
    assert "strategy='nnls'" in repr(ens_nnls)
