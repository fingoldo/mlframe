"""post_calibrate_model must mask the NaN warm-up rows of a temporal OOF vector before fitting the calibrator."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression

from mlframe.training._calibration_oof_mask import mask_nonfinite_oof_rows
from mlframe.training.evaluation import post_calibrate_model


class _IsoMeta:
    """Isotonic meta-model with the (N, 1) fit / predict_proba surface post_calibrate_model drives; rejects NaN like sklearn."""

    def __init__(self):
        """Create the unfitted isotonic regressor and the fit-size record."""
        self.iso = IsotonicRegression(out_of_bounds="clip")
        self.n_fit = None

    def fit(self, X, y, **kwargs):
        """Fit isotonic on the first column; sklearn raises ValueError on NaN input."""
        self.iso.fit(np.asarray(X)[:, 0], np.asarray(y))
        self.n_fit = int(np.asarray(X).shape[0])
        return self

    def predict_proba(self, X):
        """Return an (N, 2) probability matrix from the isotonic fit."""
        p1 = np.clip(self.iso.predict(np.asarray(X)[:, 0]), 0.0, 1.0)
        return np.stack([1.0 - p1, p1], axis=1)


def _call(oof_probs, oof_target, metrics):
    """Run post_calibrate_model with an OOF-stamped model and return (result, meta_model)."""
    rng = np.random.default_rng(0)
    tp = rng.uniform(size=(20, 2))
    vp = rng.uniform(size=(20, 2))
    model = SimpleNamespace(oof_probs=oof_probs, oof_target=oof_target)
    meta = _IsoMeta()
    res = post_calibrate_model(
        original_model=(model, (tp[:, 1] > 0.5).astype(int), tp, (vp[:, 1] > 0.5).astype(int), vp, ["c0", "c1"], None, metrics),
        target_series=pd.Series(rng.integers(0, 2, size=100)),
        target_label_encoder=None,
        val_idx=np.arange(60, 80),
        test_idx=np.arange(80, 100),
        configs=SimpleNamespace(integral_calibration_error=lambda *a, **k: 0.0, calibration=SimpleNamespace(policy_auto_pick=False)),
        meta_model=meta,
    )
    return res, meta


def test_temporal_oof_with_nan_warmup_rows_calibrates_on_finite_rows_only():
    """Leading NaN warm-up rows no longer crash the isotonic fit; only finite rows are used and the masked count is stamped."""
    rng = np.random.default_rng(1)
    m = 60
    oof = rng.uniform(size=m)
    oof[:15] = np.nan
    y = (rng.uniform(size=m) < oof.clip(0, 1)).astype(int)
    metrics: dict = {}
    res, meta = _call(oof, y, metrics)
    assert meta.n_fit == 45
    assert metrics["oof_nonfinite_rows_masked"] == 15
    assert len(res) == 8


def test_mask_helper_drops_rows_jointly_with_labels_and_raises_when_nothing_finite():
    """Rows with a non-finite probability in any column are dropped together with their label; an all-NaN source raises."""
    x = np.array([[0.2, 0.8], [np.nan, np.nan], [0.6, 0.4]])
    y = np.array([0, 1, 1])
    xm, ym = mask_nonfinite_oof_rows(x, y)
    assert xm.shape == (2, 2)
    assert ym.tolist() == [0, 1]
    with pytest.raises(ValueError, match="non-finite"):
        mask_nonfinite_oof_rows(np.full(4, np.nan), np.zeros(4))


def test_multiclass_oof_with_nan_warmup_rows_calibrates_on_finite_rows():
    """The per-class isotonic path also masks NaN warm-up rows and stamps the masked count."""
    rng = np.random.default_rng(2)
    m = 90
    oof = rng.dirichlet(np.ones(3), size=m)
    oof[:12] = np.nan
    y = rng.integers(0, 3, size=m)
    tp = rng.dirichlet(np.ones(3), size=20)
    vp = rng.dirichlet(np.ones(3), size=20)
    metrics: dict = {}
    model = SimpleNamespace(oof_probs=oof, oof_target=y)
    res = post_calibrate_model(
        original_model=(model, tp.argmax(1), tp, vp.argmax(1), vp, ["c0", "c1", "c2"], None, metrics),
        target_series=pd.Series(rng.integers(0, 3, size=100)),
        target_label_encoder=None,
        val_idx=np.arange(60, 80),
        test_idx=np.arange(80, 100),
        configs=SimpleNamespace(integral_calibration_error=lambda *a, **k: 0.0, calibration=SimpleNamespace(policy_auto_pick=False)),
        meta_model=_IsoMeta(),
    )
    assert metrics["oof_nonfinite_rows_masked"] == 12
    assert np.isfinite(res[2]).all()
