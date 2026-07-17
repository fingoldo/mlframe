"""Regression tests for P0-3 (see audit/mlframe_audit_2026_07/calibration.md): the calib==test
leakage guard in train_postcalibrators only caught EXACT array equality (same shape, same values, same
order). It missed the SAME rows reshuffled (different order), a subset/superset of the same rows
(different shape), and probability-row reuse when the target vector itself was independently
reshuffled/relabelled. Fixed by adding a sorted-multiset (reorder) check, a row-index overlap check,
and a probability-value overlap check (via the new ``_values_overlap_fraction`` helper).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _synth(seed: int = 0, n: int = 300):
    rng = np.random.default_rng(seed)
    p1 = rng.random(n)
    probs = np.column_stack([1 - p1, p1])
    target = (p1 + rng.normal(0, 0.1, n) > 0.5).astype(int)
    return probs, target


def test_train_postcalibrators_raises_on_reordered_calib_target_reusing_test_rows():
    """A calib_target that is model.test_target SHUFFLED (same values, different order) must be
    caught -- pre-fix, np.array_equal returned False on the reordered array and the guard silently
    passed."""
    from mlframe.calibration.post import train_postcalibrators, named_calibrator
    from unittest.mock import patch
    from sklearn.isotonic import IsotonicRegression

    n = 300
    probs, target = _synth(n=n)
    rng = np.random.default_rng(1)
    perm = rng.permutation(n)
    shuffled_target = target[perm]  # same multiset of values, different order -- same underlying rows

    class _FakeModel:
        columns = ["y"]
        test_target = target

    fake_calibrators = [named_calibrator(IsotonicRegression(out_of_bounds="clip"), name="Iso", lib="sklearn")]

    with patch("mlframe.calibration.post.get_postcalibrators", return_value=fake_calibrators):
        with pytest.raises(ValueError, match="reordered|identical"):
            train_postcalibrators(
                models={"m1": _FakeModel()},
                model_name="m",
                models_dir="unused",
                target_name="t",
                featureset_name="fs",
                include_patterns=["sklearn"],
                ensembling_method="harm",
                verbose=0,
                calib_probs_per_model=[probs],
                calib_target=shuffled_target,
            )


def test_train_postcalibrators_raises_on_index_subset_overlap_with_test_target():
    """calib_target sharing most of its row INDEX with model.test_target (e.g. a subset reuse via a
    different-but-overlapping train_test_split) must be caught even though the shapes differ and
    array_equal never applies."""
    from mlframe.calibration.post import train_postcalibrators, named_calibrator
    from unittest.mock import patch
    from sklearn.isotonic import IsotonicRegression

    n = 300
    probs, target = _synth(n=n)
    _outer_test_target = pd.Series(target, index=np.arange(n))
    # calib_target reuses 95% of the SAME row index as test_target, just fewer rows (a subset).
    calib_idx = np.arange(int(n * 0.95))
    calib_target = pd.Series(target[calib_idx], index=calib_idx)
    calib_probs = probs[calib_idx]

    class _FakeModel:
        columns = ["y"]
        test_target = _outer_test_target

    fake_calibrators = [named_calibrator(IsotonicRegression(out_of_bounds="clip"), name="Iso", lib="sklearn")]

    with patch("mlframe.calibration.post.get_postcalibrators", return_value=fake_calibrators):
        with pytest.raises(ValueError, match="row index overlaps"):
            train_postcalibrators(
                models={"m1": _FakeModel()},
                model_name="m",
                models_dir="unused",
                target_name="t",
                featureset_name="fs",
                include_patterns=["sklearn"],
                ensembling_method="harm",
                verbose=0,
                calib_probs_per_model=[calib_probs],
                calib_target=calib_target,
            )


def test_train_postcalibrators_raises_on_probs_overlap_with_reshuffled_target():
    """calib_probs_per_model reusing model.test_probs rows must be caught even when the target vector
    itself was independently reshuffled/relabelled -- the pre-fix guard never cross-checked probs at
    all, only the target vector."""
    from mlframe.calibration.post import train_postcalibrators, named_calibrator
    from unittest.mock import patch
    from sklearn.isotonic import IsotonicRegression

    n = 300
    probs, target = _synth(n=n)
    rng = np.random.default_rng(2)
    # calib_target is an INDEPENDENT relabelling (not derived from test's target at all), so a
    # target-only guard sees no relationship whatsoever; only the probs betray the reuse. Give
    # model.test_target one extra row so its shape never matches calib_target's -- this isolates the
    # probs-overlap detector from the (already separately tested) exact/reorder target-value checks.
    independent_target = rng.integers(0, 2, size=n)
    _outer_test_target = np.concatenate([independent_target, [0]])

    class _FakeModel:
        columns = ["y"]
        test_target = _outer_test_target
        test_probs = probs  # SAME probability rows as calib_probs_per_model below

    fake_calibrators = [named_calibrator(IsotonicRegression(out_of_bounds="clip"), name="Iso", lib="sklearn")]

    with patch("mlframe.calibration.post.get_postcalibrators", return_value=fake_calibrators):
        with pytest.raises(ValueError, match="test_probs"):
            train_postcalibrators(
                models={"m1": _FakeModel()},
                model_name="m",
                models_dir="unused",
                target_name="t",
                featureset_name="fs",
                include_patterns=["sklearn"],
                ensembling_method="harm",
                verbose=0,
                calib_probs_per_model=[probs],
                calib_target=independent_target,
            )
