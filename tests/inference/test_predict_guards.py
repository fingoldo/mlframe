"""Regression tests for the predict-guard fixes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Dummy-model stand-ins + 200-row pandas frames; guard logic only, no real estimator fits, wall <0.5s.
pytestmark = [pytest.mark.fast]


class _DummyModel:
    """Stand-in model that records the X handed to fn() so we can assert the imputed values."""

    def __init__(self):
        self.last_X = None
        self.last_X_means = None
        self.last_X_stds = None

    def __call__(self, X):
        self.last_X = X
        arr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        self.last_X_means = arr.mean(axis=0)
        self.last_X_stds = arr.std(axis=0)
        return arr[:, 0]


def _make_train_predict_frames():
    """Train frame: monotone-increasing column 'x' over n_train rows; predict frame: shifted (much higher)
    values for the SAME column. If the NaN guard fits on the predict frame, the imputed mean reflects the
    predict-shifted distribution; if it correctly uses the persisted-train stats, the imputed mean stays at
    the train mean."""
    n_train = 200
    n_pred = 50
    train_x = np.arange(n_train, dtype=np.float64)
    # Inject a NaN row in the predict frame so the guard fires.
    pred_x = 10_000.0 + np.arange(n_pred, dtype=np.float64)
    pred_x[5] = np.nan
    return train_x, pred_x


def test_nan_guard_uses_persisted_stats_no_leakage():
    """The guard, after the model has _mlframe_nan_imputer/_mlframe_nan_scaler attached, must transform with
    the TRAIN statistics. Verified by: the imputed value for a NaN row in the predict frame equals the TRAIN
    mean (not the predict-frame mean)."""
    from mlframe.training._predict_guards import _apply_nan_guard, prime_nan_guard_stats

    train_x, pred_x = _make_train_predict_frames()
    train_df = pd.DataFrame({"x": train_x})
    pred_df = pd.DataFrame({"x": pred_x})

    model = _DummyModel()
    prime_nan_guard_stats(model, train_df)
    # Now the predict-time call must transform via persisted stats.
    _apply_nan_guard(model, pred_df, model, n_rows=len(pred_df))

    # The persisted imputer's statistics_ should be the train mean (~99.5 for arange(200)).
    train_mean = float(train_x.mean())
    persisted_imp = model._mlframe_nan_imputer
    assert abs(float(persisted_imp.statistics_[0]) - train_mean) < 1e-6, (
        f"persisted imputer mean must equal train mean; got {persisted_imp.statistics_[0]} vs {train_mean}"
    )


def test_nan_guard_refuses_when_no_persisted_stats_and_fit_at_predict_false():
    """Audit 2026-05-17 (C10): when ``fit_at_predict=False`` (default)
    and the model has no persisted stats, the guard MUST REFUSE rather
    than silently fit on the current frame. Pre-fix the path emitted a
    leakage WARN and proceeded; post-fix it raises
    :class:`NanGuardNotPrimedError` so silent leakage is impossible.
    Callers that explicitly want fit-on-current-frame semantics must
    opt in via ``fit_at_predict=True``."""
    from mlframe.training._predict_guards import _apply_nan_guard, NanGuardNotPrimedError

    _, pred_x = _make_train_predict_frames()
    pred_df = pd.DataFrame({"x": pred_x})
    model = _DummyModel()

    with pytest.raises(NanGuardNotPrimedError, match="prime_nan_guard_stats"):
        _apply_nan_guard(model, pred_df, model, n_rows=len(pred_df), fit_at_predict=False)


def test_nan_guard_opt_in_fit_at_predict_still_works():
    """Audit 2026-05-17 (C10): the legacy fit-on-current-frame path is
    still available via ``fit_at_predict=True`` so callers that
    genuinely want it (e.g. one-shot scoring without a training tail)
    aren't blocked."""
    from mlframe.training._predict_guards import _apply_nan_guard

    _, pred_x = _make_train_predict_frames()
    pred_df = pd.DataFrame({"x": pred_x})
    model = _DummyModel()

    _ = _apply_nan_guard(model, pred_df, model, n_rows=len(pred_df), fit_at_predict=True)
    assert hasattr(model, "_mlframe_nan_imputer")
    assert hasattr(model, "_mlframe_nan_scaler")


def test_nan_guard_skips_when_no_nan_present():
    """The fast NaN check must short-circuit the guard for NaN-free input -- the persisted stats must NOT be
    consulted (would force an unnecessary transform copy)."""
    from mlframe.training._predict_guards import _apply_nan_guard

    clean = pd.DataFrame({"x": np.arange(50.0, dtype=np.float64)})
    model = _DummyModel()
    _apply_nan_guard(model, clean, model, n_rows=len(clean))
    # The dummy model received the original frame (not the transformed one).
    assert model.last_X is clean


def test_prime_nan_guard_stats_primes_even_on_clean_train():
    """The priming helper attaches imputer + scaler unconditionally. A clean train frame doesn't need
    imputation but DOES need its mean/std for the predict-time scaler (when predict has NaN). Skipping the
    prime on clean-train would force the predict-time guard to refit on predict, the original leakage bug."""
    from mlframe.training._predict_guards import prime_nan_guard_stats

    train = pd.DataFrame({"x": np.arange(50.0, dtype=np.float64)})
    model = _DummyModel()
    prime_nan_guard_stats(model, train)
    assert hasattr(model, "_mlframe_nan_imputer")
    # Mean of arange(50) == 24.5 (statistics_ holds the train mean even though the frame had no NaN).
    assert abs(float(model._mlframe_nan_imputer.statistics_[0]) - 24.5) < 1e-6


def test_persisted_stats_path_reuses_train_mean_for_nan_imputation():
    """End-to-end: train mean is 99.5; predict frame has shifted distribution + a NaN. After running the
    guard, the value imputed into the NaN slot must reflect a train-mean-derived offset, NOT the predict-
    frame mean."""
    from mlframe.training._predict_guards import prime_nan_guard_stats, _apply_nan_guard

    train_x, pred_x = _make_train_predict_frames()
    train_df = pd.DataFrame({"x": train_x})
    pred_df = pd.DataFrame({"x": pred_x})

    captured = {}

    def _fn(X):
        captured["X"] = X.copy() if hasattr(X, "copy") else np.array(X)
        return np.zeros(len(X))

    model = _DummyModel()
    prime_nan_guard_stats(model, train_df)
    _apply_nan_guard(model, pred_df, _fn, n_rows=len(pred_df))

    # The transformed frame post-imputation row for the NaN slot should NOT be the predict-frame mean.
    transformed = captured["X"]
    # train_mean=99.5; train_std~57.7; predict mean (non-NaN cells) ~10000+something. Persisted scaler used
    # train stats so the standardised values will be far from zero (predict frame is shifted ~+10000 vs
    # train).
    arr = transformed.to_numpy() if hasattr(transformed, "to_numpy") else np.asarray(transformed)
    # The NaN row had value=nan; imputed-then-scaled value should be (train_mean - train_mean) / train_std = 0.
    nan_row_scaled = float(arr[5, 0])
    assert abs(nan_row_scaled - 0.0) < 1e-6, f"NaN row should be imputed to train_mean then standardised to 0.0; got {nan_row_scaled}"
    # Non-NaN predict rows are FAR from zero because they were shifted +10000 from the train distribution.
    nonnan_row_scaled = float(arr[0, 0])
    assert nonnan_row_scaled > 10.0, f"shifted predict row should standardise far from zero; got {nonnan_row_scaled}"
