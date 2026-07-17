"""Regression test: compare_postcalibrators / train_postcalibrators no longer silently drop calib-set metrics.

Pre-fix: compare_postcalibrators skipped ALL metric computation whenever oos_probs/oos_target were
None, returning metrics_df=None unconditionally. train_postcalibrators (the only in-repo caller) ALWAYS
calls it with oos_probs=None (by design -- see its docstring on the calib/test split), so its
calib_test_metrics local was `None` on every real call: computed nowhere, then silently discarded
(only test_calibrators reached the return value). Post-fix: compare_postcalibrators falls back to a
self-evaluation on the calib set itself when no OOS set is supplied, and train_postcalibrators logs
+ returns those metrics alongside the fitted calibrators.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import numpy as np
import pytest


def _synth(seed: int = 0, n: int = 300):
    rng = np.random.default_rng(seed)
    p1 = rng.random(n)
    probs = np.column_stack([1 - p1, p1])
    target = (p1 + rng.normal(0, 0.1, n) > 0.5).astype(int)
    return probs, target


def test_compare_postcalibrators_self_evaluates_when_no_oos_set():
    from mlframe.calibration.post import compare_postcalibrators, named_calibrator
    from sklearn.isotonic import IsotonicRegression

    probs, target = _synth()
    fake_calibrators = [named_calibrator(IsotonicRegression(out_of_bounds="clip"), name="Iso", lib="sklearn")]

    with patch("mlframe.calibration.post.get_postcalibrators", return_value=fake_calibrators):
        metrics_df, calibrators, failed = compare_postcalibrators(
            model_name="m",
            columns=["y"],
            calib_probs=probs,
            calib_target=target,
            oos_probs=None,
            oos_target=None,
            calib_type="calib",
            include_patterns=["sklearn"],
            selection="self_eval",
        )

    assert metrics_df is not None, "pre-fix: metrics_df was unconditionally None when oos_probs is None"
    assert "sklearn.Iso" in metrics_df.index, "the fitted calibrator's self-eval row must be present"
    assert "oos" in metrics_df.index, "the baseline (pre-calibration) self-eval row must be present"
    assert calibrators, "fitted calibrators must still be returned"
    assert failed == {}, "no calibrator should have failed in this synthetic scenario"


def test_train_postcalibrators_returns_and_logs_metrics(caplog, tmp_path):
    from mlframe.calibration.post import train_postcalibrators, named_calibrator
    from sklearn.isotonic import IsotonicRegression

    n = 300
    probs, target = _synth(n=n)

    class _FakeModel:
        columns = ["y"]

    fake_calibrators = [named_calibrator(IsotonicRegression(out_of_bounds="clip"), name="Iso", lib="sklearn")]

    # train_postcalibrators writes calibrator dumps under a slugified target/featureset/task/model
    # subdir it does not create itself (expects the caller's model-directory setup to have done so).
    from pyutilz.strings import slugify
    from mlframe.training import TargetTypes

    (tmp_path / slugify("t") / slugify("fs") / slugify(str(TargetTypes.BINARY_CLASSIFICATION)) / slugify("m")).mkdir(parents=True)

    with patch("mlframe.calibration.post.get_postcalibrators", return_value=fake_calibrators):
        with caplog.at_level(logging.INFO, logger="mlframe.calibration.post"):
            result = train_postcalibrators(
                models={"m1": _FakeModel()},
                model_name="m",
                models_dir=str(tmp_path),
                target_name="t",
                featureset_name="fs",
                include_patterns=["sklearn"],
                ensembling_method="harm",
                verbose=0,
                calib_probs_per_model=[probs],
                calib_target=target,
            )

    assert isinstance(result, dict) and set(result.keys()) == {"calibrators", "metrics", "failed_calibrators"}, (
        f"pre-fix: train_postcalibrators returned the raw calibrators dict directly, dropping metrics; got {result!r}"
    )
    assert result["metrics"] is not None, "calib_test_metrics must no longer be unconditionally None"
    assert result["calibrators"], "fitted calibrators must still be present"
    assert any("calib-set comparison metrics" in rec.message for rec in caplog.records), "the calib-set comparison metrics must be logged, not only returned"
