"""Unit + biz_value tests for classification conformal prediction sets (LAC/APS).

Covers the standalone ``conformal_classification_report`` (marginal coverage >= 1-alpha, set-size
efficiency) and the finalize hook's classification branch stamping ``metadata["conformal"]``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mlframe.training._conformal_finalize import conformal_classification_report
from mlframe.training.core._phase_finalize import _conformal_on_calib_slice


def _softmax_probs(rng, n, k=3, signal=2.0):
    """Well-formed k-class probs: true class gets a +signal logit boost, then softmax."""
    y = rng.integers(0, k, size=n)
    logits = rng.standard_normal((n, k))
    logits[np.arange(n), y] += signal
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    return probs, y


def test_lac_sets_achieve_marginal_coverage():
    rng = np.random.default_rng(0)
    cp, ct = _softmax_probs(rng, 6000)
    tp, tt = _softmax_probs(rng, 6000)
    rep = conformal_classification_report(
        test_probs=tp,
        test_target=tt,
        calib_probs=cp,
        calib_target=ct,
        classes=np.array([0, 1, 2]),
        alphas=(0.1,),
        score="lac",
    )
    assert rep["method"] == "conformal_set"
    pa = rep["per_alpha"][0.1]
    assert pa["achieved_coverage"] >= 0.87  # >= 1-alpha within finite-sample noise
    assert 1.0 <= pa["mean_set_size"] <= 3.0


def test_aps_sets_achieve_marginal_coverage():
    rng = np.random.default_rng(1)
    cp, ct = _softmax_probs(rng, 6000)
    tp, tt = _softmax_probs(rng, 6000)
    rep = conformal_classification_report(
        test_probs=tp,
        test_target=tt,
        calib_probs=cp,
        calib_target=ct,
        classes=np.array([0, 1, 2]),
        alphas=(0.1,),
        score="aps",
    )
    assert rep["score"] == "aps"
    assert rep["per_alpha"][0.1]["achieved_coverage"] >= 0.87


def test_classification_report_rejects_shape_mismatch():
    rng = np.random.default_rng(2)
    cp, ct = _softmax_probs(rng, 100)
    import pytest

    with pytest.raises(ValueError):
        conformal_classification_report(
            test_probs=cp[:, :2],
            test_target=ct,
            calib_probs=cp,
            calib_target=ct,
            classes=np.array([0, 1, 2]),
            alphas=(0.1,),
        )


def test_finalize_hook_stamps_classification_sets():
    rng = np.random.default_rng(3)
    cp, ct = _softmax_probs(rng, 4000)
    tp, tt = _softmax_probs(rng, 4000)
    e = SimpleNamespace(
        model=SimpleNamespace(classes_=np.array([0, 1, 2])),
        test_probs=tp,
        test_target=tt,
        calib_probs=cp,
        calib_target=ct,
        model_name="clf",
    )
    ctx = SimpleNamespace(
        models={"MULTICLASS": {"y": [e]}},
        metadata={},
        verbose=0,
        split_config=None,
        configs=None,
        conformal_config=None,
    )
    _conformal_on_calib_slice(ctx)
    assert "conformal" in ctx.metadata
    rep = ctx.metadata["conformal"]["MULTICLASS/y/clf"]
    assert rep["method"] == "conformal_set"
    assert rep["n_classes"] == 3
    assert rep["per_alpha"][0.1]["achieved_coverage"] >= 0.85


def test_finalize_hook_classification_mode_off_skips():
    rng = np.random.default_rng(4)
    cp, ct = _softmax_probs(rng, 2000)
    tp, tt = _softmax_probs(rng, 2000)
    e = SimpleNamespace(
        model=SimpleNamespace(classes_=np.array([0, 1, 2])),
        test_probs=tp,
        test_target=tt,
        calib_probs=cp,
        calib_target=ct,
        model_name="clf",
    )
    ctx = SimpleNamespace(
        models={"MULTICLASS": {"y": [e]}},
        metadata={},
        verbose=0,
        split_config=None,
        configs=None,
        conformal_config=SimpleNamespace(enabled=True, classification_mode="off"),
    )
    _conformal_on_calib_slice(ctx)
    assert "conformal" not in ctx.metadata
