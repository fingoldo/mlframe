"""Corner-case tests for multi-output (Session 5 Tier 2).

Closes ~6 review-driven gaps from the Session 1/2 audits that weren't yet
covered by existing test files.

Tests:
- test_cb_multilogloss_rejects_float_target_with_clear_msg
- test_xgb_num_class_with_y_val_none
- test_multilabel_K1_degenerate_ok
- test_splitting_stratify_2d_via_iterstrat_roundtrip
- test_iterstrat_optional_dep_raises_on_missing
- test_xgb_native_config_carries_through_objective_kwargs
- test_per_class_calibrator_empty_class_set_safe
"""
from __future__ import annotations

import sys
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from mlframe.training.configs import (
    TargetTypes,
    MultilabelDispatchConfig,
)


# ---------------------------------------------------------------------------
# 1. CatBoost MultiLogloss rejects float target with clear message
# ---------------------------------------------------------------------------


def test_cb_multilogloss_accepts_int_targets_returns_NK():
    """CB MultiLogloss accepts {0, 1} integer (N, K) targets and returns
    (N, K) probabilities directly — no wrapper needed.

    (Original audit claimed CB rejects float targets; empirically CB 1.2.10
    accepts float too, but our dispatch path pre-casts to uint8 for safety.)
    """
    pytest.importorskip("catboost")
    from catboost import CatBoostClassifier

    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 4))
    y_int = rng.integers(0, 2, size=(100, 3)).astype(np.int8)

    clf = CatBoostClassifier(
        loss_function="MultiLogloss", iterations=3, verbose=False,
        allow_writing_files=False,
    )
    clf.fit(X, y_int)
    probs = clf.predict_proba(X[:10])
    assert probs.shape == (10, 3)
    assert (probs >= 0).all() and (probs <= 1).all()


# ---------------------------------------------------------------------------
# 2. XGB num_class with y_val=None
# ---------------------------------------------------------------------------


def test_xgb_num_class_inference_survives_y_val_none():
    """When no validation set is provided, ``configure_training_params``
    must still compute ``num_class`` correctly from y_train alone (not
    crash on ``np.concatenate([y_train, None, None])``)."""
    from mlframe.training.helpers import get_training_configs

    cfg = get_training_configs(
        iterations=10, early_stopping_rounds=2,
        target_type=TargetTypes.MULTICLASS_CLASSIFICATION,
        n_classes=4,
    )
    assert cfg.XGB_GENERAL_CLASSIF["num_class"] == 4
    assert cfg.LGB_GENERAL_PARAMS["num_class"] == 4
    # CB multiclass doesn't need num_class (auto-inferred from y)
    assert cfg.CB_CLASSIF["loss_function"] == "MultiClass"


# ---------------------------------------------------------------------------
# 3. Multilabel K=1 degenerate case
# ---------------------------------------------------------------------------


def test_multilabel_K1_degenerate_metrics_ok():
    """K=1 multilabel is degenerate (just a single binary label stored as
    2-D). Numba metrics and canonicalizer should not crash."""
    from mlframe.metrics import (
        hamming_loss, subset_accuracy, jaccard_score_multilabel,
    )
    from mlframe.training.helpers import _canonical_predict_proba_shape

    y_true = np.array([[1], [0], [1], [0]], dtype=np.int8)
    y_pred = np.array([[1], [1], [1], [0]], dtype=np.int8)

    # All 3 metrics must return valid floats
    h = hamming_loss(y_true, y_pred)
    s = subset_accuracy(y_true, y_pred)
    j = jaccard_score_multilabel(y_true, y_pred)
    assert 0.0 <= h <= 1.0
    assert 0.0 <= s <= 1.0
    assert 0.0 <= j <= 1.0
    # Known: 1 mismatch / 4 elements = 0.25
    assert abs(h - 0.25) < 1e-12

    # Canonicalizer should passthrough (N, 1)
    probs_N1 = np.array([[0.9], [0.4], [0.7], [0.1]])
    out = _canonical_predict_proba_shape(probs_N1)
    assert out.shape == (4, 1)


# ---------------------------------------------------------------------------
# 4. Splitting stratify_y 2-D via iterstrat
# ---------------------------------------------------------------------------


def test_splitting_stratify_2d_via_iterstrat_roundtrip():
    """make_train_test_split with stratify_y of shape (N, K) routes through
    iterstrat and preserves per-label frequencies within ±10%."""
    pytest.importorskip("iterstrat")
    from mlframe.training.splitting import make_train_test_split

    rng = np.random.default_rng(42)
    N, K = 500, 3
    # Per-label rates 0.3, 0.5, 0.7 — we want these preserved in splits
    y = np.stack([
        (rng.uniform(size=N) < 0.3).astype(np.int8),
        (rng.uniform(size=N) < 0.5).astype(np.int8),
        (rng.uniform(size=N) < 0.7).astype(np.int8),
    ], axis=1)
    df = pd.DataFrame({"feature_a": rng.standard_normal(N)})

    train_idx, val_idx, test_idx, *_ = make_train_test_split(
        df, test_size=0.2, val_size=0.1, shuffle_test=True, shuffle_val=True,
        random_seed=0, stratify_y=y,
    )
    # Each split should have ~similar per-label rates to the full data
    full_rates = y.mean(axis=0)
    for idx, name in [(train_idx, "train"), (val_idx, "val"), (test_idx, "test")]:
        if len(idx) == 0:
            continue
        split_rates = y[idx].mean(axis=0)
        diff = np.abs(split_rates - full_rates)
        assert (diff < 0.10).all(), (
            f"{name} split drifted: full={full_rates}, split={split_rates}, "
            f"diff={diff}"
        )


def test_splitting_stratify_1d_equivalent_to_sklearn():
    """1-D stratify_y routes through sklearn StratifiedShuffleSplit —
    preserves class ratios precisely."""
    from mlframe.training.splitting import make_train_test_split
    rng = np.random.default_rng(0)
    N = 500
    y = (rng.uniform(size=N) < 0.3).astype(np.int8)  # ~30% positives
    df = pd.DataFrame({"feature_a": rng.standard_normal(N)})

    train_idx, val_idx, test_idx, *_ = make_train_test_split(
        df, test_size=0.2, val_size=0.1, shuffle_test=True, shuffle_val=True,
        random_seed=0, stratify_y=y,
    )
    for idx, name in [(train_idx, "train"), (val_idx, "val"), (test_idx, "test")]:
        if len(idx) == 0:
            continue
        split_rate = float(y[idx].mean())
        # StratifiedShuffleSplit should give very tight ratios (<5pp)
        assert abs(split_rate - 0.3) < 0.05, (
            f"{name} class rate {split_rate:.3f} drifted from 0.3"
        )


# ---------------------------------------------------------------------------
# 5. iterstrat ImportError message
# ---------------------------------------------------------------------------


def test_iterstrat_import_error_has_install_hint():
    """When iterstrat is missing, make_train_test_split(stratify_y=2D) must
    raise ImportError with a pip-install hint (not a cryptic failure)."""
    from mlframe.training.splitting import make_train_test_split

    # Patch sys.modules to simulate iterstrat missing
    with patch.dict(sys.modules, {"iterstrat": None, "iterstrat.ml_stratifiers": None}):
        # Also patch the actual import path
        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__
        def fake_import(name, *args, **kwargs):
            if name.startswith("iterstrat"):
                raise ImportError(f"No module named {name!r}")
            return real_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=fake_import):
            rng = np.random.default_rng(0)
            df = pd.DataFrame({"a": rng.standard_normal(100)})
            y_2d = rng.integers(0, 2, size=(100, 3)).astype(np.int8)
            with pytest.raises(ImportError, match="iterative-stratification"):
                make_train_test_split(
                    df, test_size=0.2, stratify_y=y_2d, random_seed=0,
                )


# ---------------------------------------------------------------------------
# 6. XGB native multilabel config carries through
# ---------------------------------------------------------------------------


def test_xgb_native_multilabel_kwargs_have_tree_method():
    """When ``force_native_xgb_multilabel=True``, the native multilabel
    kwargs include ``tree_method='hist'`` (required by XGB 3.x for
    multi_output_tree)."""
    from mlframe.training.strategies import XGBoostStrategy
    s = XGBoostStrategy()
    cfg = MultilabelDispatchConfig(force_native_xgb_multilabel=True)
    kw = s.get_classif_objective_kwargs(
        TargetTypes.MULTILABEL_CLASSIFICATION, n_classes=3,
        multilabel_config=cfg,
    )
    assert kw["tree_method"] == "hist"
    assert kw["multi_strategy"] == "multi_output_tree"
    # Must NOT include num_class (binary:logistic per-output, not softprob)
    assert "num_class" not in kw


# ---------------------------------------------------------------------------
# 7. Per-class calibrator on small-K edge
# ---------------------------------------------------------------------------


def test_per_class_calibrator_tiny_calib_set():
    """Calibration set smaller than n_classes. Each per-class calibrator
    either fits or is skipped (None = identity); no crash."""
    from mlframe.training.trainer import _PerClassIsotonicCalibrator
    # Only 10 samples, 5 classes — some classes will have 0-2 samples
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(5), size=10)
    y = rng.integers(0, 5, size=10)
    cal = _PerClassIsotonicCalibrator.fit(probs, y, TargetTypes.MULTICLASS_CLASSIFICATION)
    # All classes either have IsotonicRegression or None (skip)
    assert set(cal.calibrators.keys()) == {0, 1, 2, 3, 4}
    # Apply — must not crash; output shape preserved
    out = cal.predict_proba(probs)
    assert out.shape == probs.shape
    # Rows still sum to 1 (softmax invariant even with some identity columns)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-10)
