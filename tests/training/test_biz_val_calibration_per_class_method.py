"""biz_value + unit tests for the per-class post-hoc calibration METHOD default.

The multiclass / multilabel post-hoc calibrator ``_PerClassIsotonicCalibrator`` (reached in production
from ``training.evaluation.post_calibrate_model`` for any ``(N, K!=2)`` probability matrix) defaults its
per-class map to ``method="sigmoid"`` (Platt). On the small per-class OOF / calibration slices this path
typically sees, free-form isotonic interpolates the training calibration and generalises worse on held-out
data. These tests pin (a) the default is sigmoid and (b) sigmoid's held-out mean-per-class ECE win over
isotonic on miscalibrated multiclass scores. A silent revert to an isotonic default fails the win.

Bench: src/mlframe/calibration/_benchmarks/bench_per_class_method_isotonic_vs_sigmoid.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.training._calibration_models import (
    _PerClassIsotonicCalibrator,
    _SigmoidLogitAdapter,
)
from mlframe.training.configs import TargetTypes

N_BINS = 10


def _heldout_ece_binary(y, p, n_bins=N_BINS):
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    ece, n = 0.0, len(p)
    for b in range(n_bins):
        m = idx == b
        c = int(m.sum())
        if c:
            ece += (c / n) * abs(p[m].mean() - y[m].mean())
    return ece


def _overconfident_multiclass(seed, n, K, temp=2.2):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(K, 4))
    Xf = rng.normal(size=(n, 4))
    logits = Xf @ W.T
    pt = np.exp(logits - logits.max(1, keepdims=True))
    pt /= pt.sum(1, keepdims=True)
    y = np.array([rng.choice(K, p=pt[i]) for i in range(n)])
    s = np.exp(temp * logits)
    s /= s.sum(1, keepdims=True)
    return s, y


def _mean_per_class_heldout_ece(calibrator, s_ho, y_ho, K):
    out = calibrator.predict_proba(s_ho)
    return float(np.mean([_heldout_ece_binary((y_ho == k).astype(float), out[:, k]) for k in range(K)]))


def test_per_class_default_method_is_sigmoid():
    """Default fit must build Platt (_SigmoidLogitAdapter) per-class maps, NOT isotonic."""
    s, y = _overconfident_multiclass(0, 400, 4)
    c = _PerClassIsotonicCalibrator.fit(s, y, TargetTypes.MULTICLASS_CLASSIFICATION)
    kinds = {type(v).__name__ for v in c.calibrators.values() if v is not None}
    assert kinds == {"_SigmoidLogitAdapter"}, f"default per-class method should be sigmoid, got {kinds}"


def test_per_class_isotonic_method_selectable():
    """method='isotonic' must still build isotonic maps (REJECTED != DELETED)."""
    s, y = _overconfident_multiclass(0, 400, 4)
    c = _PerClassIsotonicCalibrator.fit(s, y, TargetTypes.MULTICLASS_CLASSIFICATION, method="isotonic")
    kinds = {type(v).__name__ for v in c.calibrators.values() if v is not None}
    assert kinds == {"IsotonicRegression"}, f"method='isotonic' should build isotonic, got {kinds}"


def test_predict_proba_rows_sum_to_one_multiclass():
    s, y = _overconfident_multiclass(1, 400, 5)
    c = _PerClassIsotonicCalibrator.fit(s, y, TargetTypes.MULTICLASS_CLASSIFICATION)
    out = c.predict_proba(s)
    assert out.shape == s.shape
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-9)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_biz_val_per_class_sigmoid_beats_isotonic_heldout_ece(seed):
    """Sigmoid per-class calibration must achieve LOWER held-out mean-per-class ECE than isotonic on
    overconfident multiclass scores with a small calibration slice. Measured deltas (n_cal=200): iso
    ~0.038-0.052 vs sig ~0.014-0.031; floor the required improvement at 15% of the isotonic ECE to
    absorb seed noise while still catching a regression to an isotonic default."""
    K, n_cal, n_ho = 5, 200, 4000
    s_all, y_all = _overconfident_multiclass(seed, n_cal + n_ho, K)
    s_cal, y_cal, s_ho, y_ho = s_all[:n_cal], y_all[:n_cal], s_all[n_cal:], y_all[n_cal:]

    c_sig = _PerClassIsotonicCalibrator.fit(s_cal, y_cal, TargetTypes.MULTICLASS_CLASSIFICATION, method="sigmoid")
    c_iso = _PerClassIsotonicCalibrator.fit(s_cal, y_cal, TargetTypes.MULTICLASS_CLASSIFICATION, method="isotonic")
    ece_sig = _mean_per_class_heldout_ece(c_sig, s_ho, y_ho, K)
    ece_iso = _mean_per_class_heldout_ece(c_iso, s_ho, y_ho, K)

    assert ece_sig < ece_iso * 0.85, f"seed={seed}: sigmoid ECE {ece_sig:.4f} should beat isotonic {ece_iso:.4f} by >=15%"
