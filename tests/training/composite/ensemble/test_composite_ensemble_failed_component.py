"""Regression sensors for two composite cross-target ensemble audit items.

I23 (``_phase_composite_post_xt_ensemble/__init__.py``): when a component's
train-predict FAILED, the train-RMSE proxy imputed its row with the MEDIAN of
the surviving components' RMSEs, so ``from_train_metrics`` granted the broken
component mid-pack ensemble weight on a fabricated score. The fix leaves the
failed row NaN and drops the component from the pool entirely (zero weight).

N9 (``composite/ensemble/__init__.py``): ``external_holdout_base_per_spec`` was
accepted, passed through, and documented as REQUIRED but never READ in
``_compute_oof_with_external_holdout`` -- the ``CompositeTargetEstimator``
wrapper re-extracts its base column from ``external_holdout_X`` itself at
predict time. The fix removes the dead param from the internal helper and
corrects the docstring; the public param is retained (ignored) for back-compat.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin

from mlframe.training.composite import (
    CompositeTargetEstimator,
    compute_oof_holdout_predictions,
)
from mlframe.training.composite import ensemble as _ens_mod
from mlframe.training.core._phase_composite_post_xt_ensemble import (
    _build_cross_target_ensemble_for_target,
)


# ---------------------------------------------------------------------------
# I23 -- a component whose train-predict FAILS must get ZERO ensemble weight
# (dropped from the pool), NOT a median-imputed mid-pack weight.
# ---------------------------------------------------------------------------


class _GoodRaw(BaseEstimator, RegressorMixin):
    """A raw regressor that predicts a fixed offset from a single feature."""

    def __init__(self, offset: float = 0.0):
        self.offset = offset

    def fit(self, X, y, **kw):
        """Fit."""
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        """Predict."""
        n = len(X)
        return np.full(n, self.offset, dtype=np.float64)


class _ExplodingRaw(BaseEstimator, RegressorMixin):
    """A raw regressor whose predict ALWAYS raises -- mimics a component whose
    train-predict fails (e.g. a frame-mismatch / inner-estimator crash)."""

    def fit(self, X, y, **kw):
        """Fit."""
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        """Predict."""
        raise RuntimeError("deliberate predict failure (I23 sensor)")


def _make_models_and_targets():
    """Make models and targets."""
    from mlframe.training.configs import TargetTypes as TT

    rng = np.random.default_rng(0)
    n = 400
    feat = rng.normal(0.0, 1.0, size=n)
    y = 5.0 + 0.1 * feat  # near-constant target ~5.0

    X = pd.DataFrame({"feat": feat})
    # Two GOOD components close to the target, one BAD (explodes on predict).
    good_a = _GoodRaw(offset=5.0).fit(X, y)  # excellent
    good_b = _GoodRaw(offset=5.2).fit(X, y)  # decent
    bad = _ExplodingRaw().fit(X, y)  # predict() raises
    models = {
        TT.REGRESSION: {
            "y": [
                SimpleNamespace(model=good_a, pre_pipeline=None),
                SimpleNamespace(model=good_b, pre_pipeline=None),
                SimpleNamespace(model=bad, pre_pipeline=None),
            ],
        }
    }
    target_by_type = {TT.REGRESSION: {"y": y}}
    return TT, models, target_by_type, X, y, n


def _make_config_proxy_path():
    """Config that forces the TRAIN-RMSE proxy weighting path (the I23 site):
    ``oof_holdout_frac=0`` skips the honest-OOF block entirely, and the
    dummy floor / lag failsafe are disabled so the weights reflect ONLY the
    proxy + drop logic."""
    from mlframe.training._composite_target_discovery_config import (
        CompositeTargetDiscoveryConfig,
    )

    return CompositeTargetDiscoveryConfig(
        enabled=True,
        oof_holdout_frac=0.0,
        ct_ensemble_dummy_floor_enabled=False,
        lag_predict_failsafe_tolerance=0.0,
    )


def _run_builder(strategy: str):
    """Run builder."""
    TT, models, target_by_type, X, _y, n = _make_models_and_targets()
    metadata: dict = {}
    _build_cross_target_ensemble_for_target(
        _tt_e=TT.REGRESSION,
        _orig_tname="y",
        _spec_list=[],
        _ce_strategy=strategy,
        models=models,
        metadata=metadata,
        target_by_type=target_by_type,
        composite_target_discovery_config=_make_config_proxy_path(),
        target_name="t",
        model_name="m",
        filtered_train_df=X,
        filtered_val_df=None,
        test_df_pd=None,
        filtered_train_idx=np.arange(n),
        filtered_val_idx=None,
        test_idx=None,
        train_df_pd=X,
        val_df_pd=None,
        train_idx=np.arange(n),
        val_idx=None,
        reporting_config=SimpleNamespace(
            compute_valset_metrics=False,
            compute_testset_metrics=False,
        ),
        plot_file=None,
        _train_pred_cache={},
        ctx=None,
    )
    return TT, models, metadata


def test_failed_component_excluded_from_oof_weighted_ensemble():
    """I23: the exploding component must NOT appear in the built ensemble and
    must NOT receive any weight; the two good components remain.

    Pre-fix, the failed component's NaN proxy was median-imputed and it
    received mid-pack ``oof_weighted`` weight on a fabricated score.
    """
    TT, models, _metadata = _run_builder("oof_weighted")
    ens_entry = models[TT.REGRESSION].get("_CT_ENSEMBLE__y")
    assert ens_entry is not None, "no CT_ENSEMBLE was built"
    ens = ens_entry[0].model
    from mlframe.training.composite.ensemble import CompositeCrossTargetEnsemble

    assert isinstance(ens, CompositeCrossTargetEnsemble), f"expected a real ensemble (>=2 surviving components), got {type(ens).__name__}"
    names = list(ens.component_names)
    weights = np.asarray(ens.weights, dtype=np.float64)
    # The exploding component (3rd entry -> "raw#2") is dropped entirely.
    assert "raw#2" not in names, f"failed component still present in ensemble: {names}"
    # Both good components survive and split all the weight.
    assert {"raw#0", "raw#1"} <= set(names)
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-9)
    # Sanity: every surviving weight is a real positive number (no NaN that a
    # median-impute-then-rank could have leaked through).
    assert np.all(np.isfinite(weights))


def test_failed_component_gets_zero_weight_in_nnls_stack():
    """I23: under nnls_stack the failed component must contribute zero to the
    stacked weights (it is dropped before the stacker sees the matrix)."""
    TT, models, _metadata = _run_builder("nnls_stack")
    ens_entry = models[TT.REGRESSION].get("_CT_ENSEMBLE__y")
    assert ens_entry is not None
    ens = ens_entry[0].model
    # nnls_stack may collapse to the single best component; either way the
    # exploding component must never be a member.
    _names = getattr(ens, "component_names", None)
    if _names is not None:
        assert "raw#2" not in list(_names)


# ---------------------------------------------------------------------------
# N9 -- external_holdout_base_per_spec is dead in the external-holdout path.
# ---------------------------------------------------------------------------


def test_internal_external_holdout_helper_drops_dead_base_param():
    """N9: the internal helper no longer accepts the unused param."""
    params = inspect.signature(_ens_mod._compute_oof_with_external_holdout).parameters
    assert "external_holdout_base_per_spec" not in params, "dead param must be removed from _compute_oof_with_external_holdout"
    # The train-side base dict is still needed (drives transform.forward).
    assert "base_train_full_per_spec" in params


def test_public_param_retained_but_ignored_for_back_compat():
    """N9: the public function keeps the param (back-compat) but it is unused;
    passing it or omitting it yields identical results on a composite external
    holdout."""
    params = inspect.signature(compute_oof_holdout_predictions).parameters
    assert "external_holdout_base_per_spec" in params

    rng = np.random.default_rng(3)
    n_train, n_val = 300, 80
    b_tr = rng.normal(0.0, 1.0, size=n_train)
    feat_tr = rng.normal(0.0, 1.0, size=n_train)
    y_tr = b_tr + 0.5 * feat_tr + rng.normal(0.0, 0.05, size=n_train)
    X_tr = pd.DataFrame({"b": b_tr, "feat": feat_tr})

    b_val = rng.normal(0.0, 1.0, size=n_val)
    feat_val = rng.normal(0.0, 1.0, size=n_val)
    y_val = b_val + 0.5 * feat_val + rng.normal(0.0, 0.05, size=n_val)
    X_val = pd.DataFrame({"b": b_val, "feat": feat_val})

    # Build a composite-target component (diff transform on base column "b").
    inner = _GoodRaw(offset=0.0)
    # Fit the inner on the T = (y - b) target so it learns the residual.
    inner.fit(X_tr, y_tr - b_tr)
    spec = {
        "transform_name": "diff",
        "base_column": "b",
        "fitted_params": {},
        "extra_base_columns": (),
    }
    wrapped = CompositeTargetEstimator.from_fitted_inner(
        fitted_inner=inner,
        transform_name="diff",
        base_column="b",
        transform_fitted_params={},
        y_train=y_tr,
    )

    def _call(pass_base):
        """Call."""
        kwargs = dict(
            component_models=[wrapped, wrapped],
            component_names=["c0", "c1"],
            component_specs=[spec, spec],
            train_X=X_tr,
            y_train_full=y_tr,
            base_train_full_per_spec={"b": b_tr},
            holdout_frac=0.2,
            random_state=0,
            external_holdout_X=X_val,
            external_holdout_y=y_val,
        )
        if pass_base:
            # A WRONG base dict here would change results IF it were consulted;
            # it must be ignored, so results are identical to omitting it.
            kwargs["external_holdout_base_per_spec"] = {
                "b": np.full(n_val, 1e6),
            }
        return compute_oof_holdout_predictions(**kwargs)

    preds_with, y_with, surv_with = _call(pass_base=True)
    preds_without, y_without, surv_without = _call(pass_base=False)

    # The composite wrapper re-extracts the holdout base from X_val itself, so
    # the bogus per-spec base dict has no effect: results are bit-identical.
    assert surv_with == surv_without
    np.testing.assert_array_equal(preds_with, preds_without)
    np.testing.assert_array_equal(y_with, y_without)
    # And the predictions are honest (base re-extracted from val -> ~y_val).
    assert preds_with.shape == (n_val, 2)
    assert np.isfinite(preds_with).all()
    assert abs(float(np.mean(preds_with[:, 0] - y_val))) < 0.5
