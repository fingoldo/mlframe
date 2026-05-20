"""Regression tests for MLPRanker auto-imputation + auto-scaling.

The LTR suite calls MLPRanker.fit directly (bypassing the strategy.build_pipeline
machinery that adds SimpleImputer + StandardScaler for sklearn-pipeline-based
neural strategies). Before the fix, a single NaN cell in X (legitimate for fuzz
combos that disable fillna_value_cfg) propagated through MLP.forward to give
NaN scores, NaN val_loss, and a Lightning silent-halt after epoch 0. Unscaled
features caused gradient bouncing that pinned val_loss at ln(2) even for clean
inputs. These tests pin both safety nets so a future refactor can't silently
re-introduce the bypass.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pytorch_lightning")


@pytest.fixture
def rng():
    return np.random.default_rng(20260521)


def _make_data(rng, n_rows=300, n_features=8, n_queries=30, nan_frac=0.0,
               scale_range: tuple[float, float] | None = None):
    X = rng.standard_normal((n_rows, n_features)).astype(np.float32)
    if scale_range is not None:
        # Per-column magnitude spread; mimics raw tabular features
        # (binary 0/1 next to USD revenue in [1e3, 1e6]).
        scales = rng.uniform(scale_range[0], scale_range[1], size=n_features).astype(np.float32)
        X *= scales
    if nan_frac > 0:
        mask = rng.random(X.shape) < nan_frac
        X[mask] = np.nan
    group_ids = np.repeat(np.arange(n_queries), n_rows // n_queries)
    # Pad/truncate to exact n_rows
    if len(group_ids) < n_rows:
        group_ids = np.concatenate([group_ids, np.full(n_rows - len(group_ids), n_queries - 1)])
    group_ids = group_ids[:n_rows]
    y = rng.integers(0, 4, size=n_rows).astype(np.float32)
    return X, y, group_ids


def test_fit_with_nan_input_does_not_crash(rng):
    from mlframe.training.neural.ranker import MLPRanker
    X, y, g = _make_data(rng, nan_frac=0.1)
    model = MLPRanker(n_estimators=2, early_stopping_patience=None, verbose=0, seed=1)
    model.fit(X, y, g)
    pred = model.predict(X)
    assert np.all(np.isfinite(pred)), "predictions must be finite after NaN imputation"


def test_imputer_uses_train_means_not_inf(rng):
    from mlframe.training.neural.ranker import MLPRanker
    X, y, g = _make_data(rng, nan_frac=0.05)
    model = MLPRanker(n_estimators=1, early_stopping_patience=None, verbose=0, seed=2)
    model.fit(X, y, g)
    assert np.all(np.isfinite(model._impute_means_)), "imputer means must be finite"
    assert model._impute_means_.shape == (X.shape[1],)


def test_inf_input_also_imputed(rng):
    from mlframe.training.neural.ranker import MLPRanker
    X, y, g = _make_data(rng)
    X[5, 0] = np.inf
    X[6, 1] = -np.inf
    model = MLPRanker(n_estimators=2, early_stopping_patience=None, verbose=0, seed=3)
    model.fit(X, y, g)
    pred = model.predict(X)
    assert np.all(np.isfinite(pred)), "Inf must also be imputed (not just NaN)"


def test_all_nan_column_falls_back_to_zero(rng):
    from mlframe.training.neural.ranker import MLPRanker
    X, y, g = _make_data(rng)
    X[:, 0] = np.nan  # entire column NaN
    model = MLPRanker(n_estimators=1, early_stopping_patience=None, verbose=0, seed=4)
    model.fit(X, y, g)
    # All-NaN col -> mean falls back to 0.0, then z-scaled with std=1.0
    assert model._impute_means_[0] == 0.0
    pred = model.predict(X)
    assert np.all(np.isfinite(pred))


def test_imputer_does_not_mutate_caller_buffer(rng):
    from mlframe.training.neural.ranker import MLPRanker
    X, y, g = _make_data(rng, nan_frac=0.1)
    nan_count_before = int(np.isnan(X).sum())
    model = MLPRanker(n_estimators=1, early_stopping_patience=None, verbose=0, seed=5)
    model.fit(X, y, g)
    nan_count_after = int(np.isnan(X).sum())
    assert nan_count_before == nan_count_after, (
        "MLPRanker.fit must not mutate the caller's X buffer "
        f"(before={nan_count_before}, after={nan_count_after})"
    )


def test_scaler_normalises_disparate_magnitude_features(rng):
    from mlframe.training.neural.ranker import MLPRanker
    # Feature 0 in [0, 1], feature 1 in [1e4, 1e6] — without scaling AdamW
    # bounces gradients and val_loss never moves off random-init.
    X, y, g = _make_data(rng, scale_range=(0.5, 1e5))
    model = MLPRanker(n_estimators=1, early_stopping_patience=None, verbose=0, seed=6)
    model.fit(X, y, g)
    # Post-fit means/stds match the (n_features,) shape and have no zero stds
    assert model._scaler_mean_.shape == (X.shape[1],)
    assert model._scaler_std_.shape == (X.shape[1],)
    assert np.all(model._scaler_std_ > 0), "constant cols should have std substituted to 1.0"


def test_constant_column_does_not_break_scaler(rng):
    from mlframe.training.neural.ranker import MLPRanker
    X, y, g = _make_data(rng)
    X[:, 2] = 7.0  # constant col
    model = MLPRanker(n_estimators=1, early_stopping_patience=None, verbose=0, seed=7)
    model.fit(X, y, g)
    pred = model.predict(X)
    assert np.all(np.isfinite(pred)), (
        "Constant columns must not yield inf/nan when std=0 is substituted to 1.0"
    )


def test_val_loss_not_nan_with_dirty_input(rng):
    """The core regression: fuzz combo c0063 with NaN input produced NaN val_loss
    and a Lightning silent-halt after epoch 0. With auto-imputation, val_loss
    must be a finite float."""
    from mlframe.training.neural.ranker import MLPRanker, MLPRankerLightningModule
    X_tr, y_tr, g_tr = _make_data(rng, n_rows=200, n_queries=20, nan_frac=0.08)
    X_va, y_va, g_va = _make_data(rng, n_rows=100, n_queries=10, nan_frac=0.08)
    losses = []
    orig_vs = MLPRankerLightningModule.validation_step

    def _spy(self, batch, batch_idx):
        out = orig_vs(self, batch, batch_idx)
        if batch_idx == 0:
            losses.append(float(out))
        return out

    MLPRankerLightningModule.validation_step = _spy
    try:
        model = MLPRanker(n_estimators=2, early_stopping_patience=None, verbose=0, seed=8)
        model.fit(X_tr, y_tr, g_tr, X_val=X_va, y_val=y_va, group_ids_val=g_va)
    finally:
        MLPRankerLightningModule.validation_step = orig_vs
    assert losses, "no validation_step ran -- trainer halted before val"
    assert all(np.isfinite(loss) for loss in losses), f"val_loss has NaN/Inf: {losses}"
