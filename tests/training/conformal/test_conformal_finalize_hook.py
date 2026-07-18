"""Unit tests for the finalize hook `_conformal_on_calib_slice` (regression conformal stamping).

Drives the hook with a synthetic ``ctx`` (no full suite run): verifies split-conformal when a calib
slice is present, CV+ fallback from OOF residuals otherwise, classification skip, and the no-op path.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mlframe.training.core._phase_finalize import _conformal_on_calib_slice


def _ctx(entry, **over):
    """Ctx."""
    base = dict(
        models={"REGRESSION": {"y": [entry]}},
        metadata={},
        verbose=0,
        split_config=None,
        configs=None,
        conformal_config=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _reg_entry(rng, n=3000, with_calib=True, with_oof=False):
    """Reg entry."""
    test_pred = rng.standard_normal(n)
    test_true = test_pred + rng.standard_normal(n)
    e = SimpleNamespace(
        test_preds=test_pred,
        test_target=test_true,
        test_probs=None,
        calib_preds=None,
        calib_target=None,
        oof_preds=None,
        train_target=None,
        model_name="m",
    )
    if with_calib:
        cp = rng.standard_normal(n)
        e.calib_preds = cp
        e.calib_target = cp + rng.standard_normal(n)
    if with_oof:
        op = rng.standard_normal(n)
        e.oof_preds = op
        e.train_target = op + rng.standard_normal(n)
    return e


def test_hook_stamps_split_conformal_from_calib_slice():
    """Hook stamps split conformal from calib slice."""
    rng = np.random.default_rng(0)
    ctx = _ctx(_reg_entry(rng, with_calib=True))
    _conformal_on_calib_slice(ctx)
    assert "conformal" in ctx.metadata
    rep = ctx.metadata["conformal"]["REGRESSION/y/m"]
    assert rep["method"] == "split_conformal"
    assert rep["guarantee"] == "marginal>=1-alpha"
    assert 0.1 in rep["per_alpha"] and 0.2 in rep["per_alpha"]
    assert "intervals" not in rep  # per-row arrays dropped to keep metadata small
    assert 0.84 <= rep["per_alpha"][0.1]["achieved_coverage"] <= 0.96


def test_hook_falls_back_to_cv_plus_from_oof():
    """Hook falls back to cv plus from oof."""
    rng = np.random.default_rng(1)
    ctx = _ctx(_reg_entry(rng, with_calib=False, with_oof=True))
    _conformal_on_calib_slice(ctx)
    rep = ctx.metadata["conformal"]["REGRESSION/y/m"]
    assert rep["method"] == "cv_plus"
    assert rep["guarantee"] == "marginal>=1-2alpha"


def test_hook_skips_classification_entries():
    """Hook skips classification entries."""
    rng = np.random.default_rng(2)
    e = _reg_entry(rng, with_calib=True)
    e.test_probs = rng.uniform(size=(3000, 2))  # classification marker -> skip (sets/venn-abers later)
    ctx = _ctx(e)
    _conformal_on_calib_slice(ctx)
    assert "conformal" not in ctx.metadata


def test_hook_noop_without_test_or_calibration_source():
    """Hook noop without test or calibration source."""
    rng = np.random.default_rng(3)
    e = _reg_entry(rng, with_calib=False, with_oof=False)  # no calib, no oof -> nothing to calibrate from
    ctx = _ctx(e)
    _conformal_on_calib_slice(ctx)
    assert "conformal" not in ctx.metadata

    e2 = _reg_entry(rng, with_calib=True)
    e2.test_preds = None  # no test side -> cannot report coverage
    ctx2 = _ctx(e2)
    _conformal_on_calib_slice(ctx2)
    assert "conformal" not in ctx2.metadata


def test_hook_respects_disabled_config():
    """Hook respects disabled config."""
    rng = np.random.default_rng(4)
    ctx = _ctx(_reg_entry(rng, with_calib=True), conformal_config=SimpleNamespace(enabled=False))
    _conformal_on_calib_slice(ctx)
    assert "conformal" not in ctx.metadata


def test_hook_structure_from_split_config_flags_coverage_validity():
    """Hook structure from split config flags coverage validity."""
    rng = np.random.default_rng(5)
    ctx = _ctx(
        _reg_entry(rng, with_calib=True),
        split_config=SimpleNamespace(time_column="ts", cv_strategy=None, use_groups=False, bucket_stratify=False, wholeday_splitting=False),
    )
    _conformal_on_calib_slice(ctx)
    rep = ctx.metadata["conformal"]["REGRESSION/y/m"]
    assert rep["structure"] == "temporal"
    assert rep["split_conformal_valid_for_structure"] is False
