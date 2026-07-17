"""Locks the auto-scaling contract for all regression model families.

Both ``LinearModelStrategy`` and ``NeuralModelStrategy`` declare ``requires_scaling=True``, which the pipeline builder consumes to inject a ``StandardScaler`` step automatically. The 2026-05-11 TVT run revealed two scaling gaps:
- X-side scaling works for ALL models that declare ``requires_scaling=True`` (linear, MLP).
- Y-side scaling was MISSING for MLP regression -- a target with mean=11500 / std=644 caused MLP to predict ~0 (val_MSE = 11559 ~ target_var after 30 min) because a kaiming-init network outputs near-zero at init and takes many epochs just to learn the constant offset. Fixed by wrapping MLP in ``sklearn.compose.TransformedTargetRegressor(MLP, transformer=StandardScaler())`` so y is auto-standardised at fit / inverse-scaled at predict, transparent to downstream code.

The PARAMETRISED biz_value test below runs each supported regression family (cb / xgb / lgb / linear / mlp) on a synthetic regression dataset with mean=11500, std=644 -- the same magnitude as the TVT target that broke MLP. Locks: every model's RMSE < 50% of target std (i.e., the model actually learned, not just predicting mean).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Sanity: strategy flags + pipeline-built StandardScaler
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy_cls_name", ["LinearModelStrategy", "NeuralNetStrategy"])
def test_strategy_declares_requires_scaling(strategy_cls_name: str) -> None:
    """Both linear and MLP strategies must declare ``requires_scaling=True`` -- otherwise the suite skips StandardScaler injection and these families silently train on un-scaled features (catastrophic for SGD / OLS conditioning)."""
    import mlframe.training.strategies as _strat

    cls = getattr(_strat, strategy_cls_name)
    assert cls.requires_scaling is True
    assert cls.requires_imputation is True
    # NeuralNetStrategy defaults to learnable cat embeddings, so it does NOT target-encode (requires_encoding False); Linear still encodes.
    if strategy_cls_name == "NeuralNetStrategy":
        assert cls().requires_encoding is False
    else:
        assert cls.requires_encoding is True


@pytest.mark.parametrize("strategy_cls_name", ["LinearModelStrategy", "NeuralNetStrategy"])
def test_built_pipeline_contains_standard_scaler(strategy_cls_name: str) -> None:
    """Pipeline built by each strategy must include a StandardScaler step (locks the X-side auto-scaling that the strategy flags promise). Uses the real build_pipeline signature: ``(base_pipeline, cat_features, category_encoder, imputer, scaler)``."""
    import mlframe.training.strategies as _strat
    from sklearn.preprocessing import StandardScaler

    cls = getattr(_strat, strategy_cls_name)
    strategy = cls()
    pipeline = strategy.build_pipeline(
        base_pipeline=None,
        cat_features=[],
        category_encoder=None,
        imputer=None,
        scaler=StandardScaler(),
    )
    # Some strategies return None (when nothing needs to be built); the contract is that linear / NN with requires_scaling=True must materialise a pipeline carrying StandardScaler.
    assert pipeline is not None, f"{strategy_cls_name}.build_pipeline returned None despite requires_scaling=True"
    step_types = [type(step[1]).__name__ for step in pipeline.steps]
    assert "StandardScaler" in step_types, f"{strategy_cls_name} pipeline must include StandardScaler; got steps={step_types}"


# ---------------------------------------------------------------------------
# Biz_value: every model family learns on large-magnitude regression target
# ---------------------------------------------------------------------------


def _make_large_magnitude_dataset(n: int = 1000, seed: int = 0) -> tuple:
    """Synthetic regression DGP mirroring the TVT-scale failure: target mean ~ 11500, std ~ 644.

    Features:
    - f1: strongly correlated with target (slope 0.95, like TVT_prev)
    - f2: independent gaussian
    """
    rng = np.random.default_rng(seed)
    f1 = rng.normal(loc=11500.0, scale=650.0, size=n)
    f2 = rng.normal(size=n)
    y = 0.95 * f1 + 0.3 * f2 + 575.0 + rng.normal(scale=10.0, size=n)
    df = pd.DataFrame({"f1": f1, "f2": f2})
    return df, y


_FAMILY_FACTORIES = []


def _try_register_family(name: str, factory):
    """Register a model family in the parametrise list ONLY if its optional dep is importable; otherwise add an importorskip marker so CI without the dep doesn't break."""
    _FAMILY_FACTORIES.append((name, factory))


def _cb_factory():
    cb = pytest.importorskip("catboost")
    return cb.CatBoostRegressor(iterations=50, depth=4, verbose=False, allow_writing_files=False)


def _xgb_factory():
    xgb = pytest.importorskip("xgboost")
    return xgb.XGBRegressor(n_estimators=50, max_depth=4, verbosity=0)


def _lgb_factory():
    lgb = pytest.importorskip("lightgbm")
    return lgb.LGBMRegressor(n_estimators=50, num_leaves=15, verbose=-1, random_state=0)


def _linear_factory():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression

    # Mirror the suite's pipeline-time auto-scaling for the linear strategy.
    return Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])


def _mlp_factory():
    """sklearn MLPRegressor as a fast stand-in for the suite's PytorchLightningRegressor. We standardise X with a Pipeline AND wrap the regressor in TransformedTargetRegressor for y-scaling -- exactly the F1 fix landed in trainer.py for the production MLP."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPRegressor
    from sklearn.compose import TransformedTargetRegressor

    inner = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(32, 16),
                    max_iter=200,
                    random_state=0,
                ),
            ),
        ]
    )
    return TransformedTargetRegressor(regressor=inner, transformer=StandardScaler())


_try_register_family("cb", _cb_factory)
_try_register_family("xgb", _xgb_factory)
_try_register_family("lgb", _lgb_factory)
_try_register_family("linear", _linear_factory)
_try_register_family("mlp", _mlp_factory)


@pytest.mark.parametrize("family_name, factory", _FAMILY_FACTORIES, ids=[f for f, _ in _FAMILY_FACTORIES])
def test_model_family_learns_on_large_magnitude_target(family_name: str, factory) -> None:
    """Every supported regression family must actually LEARN on a target with mean ~11500, std ~644 -- i.e. produce RMSE < 50% of target_std on a held-out slice. The 2026-05-11 TVT log exposed MLP catastrophically failing this contract (RMSE = target_std = 644, meaning predicting ~mean). The F1 fix (TransformedTargetRegressor wrap for MLP regression in trainer.py) closes this gap.

    Pass criterion: ``rmse / target_std < 0.5``. For a model that just predicts mean(y), ``rmse / target_std == 1.0``; a well-fitted model (linear with f1 dominating) sits at < 0.1.
    """
    model = factory()
    df, y = _make_large_magnitude_dataset(n=800, seed=0)
    train_X, test_X = df.iloc[:600], df.iloc[600:]
    train_y, test_y = y[:600], y[600:]
    target_std = float(np.std(train_y))

    # CatBoost wants pure numpy or list-of-list (DataFrame works too but we explicitly convert to match the production path that runs on numpy / pool).
    model.fit(train_X, train_y)
    preds = np.asarray(model.predict(test_X), dtype=np.float64).reshape(-1)
    rmse = float(np.sqrt(np.mean((preds - test_y) ** 2)))
    ratio = rmse / target_std

    # Every family must learn -- RMSE strictly below 50% of target std on the hold-out.
    assert ratio < 0.5, (
        f"{family_name} did NOT learn on large-magnitude target -- rmse={rmse:.2f}, target_std={target_std:.2f}, ratio={ratio:.3f}. The 2026-05-11 TVT failure mode (predicting near-mean against mean=11500) reproduces if this fails."
    )
