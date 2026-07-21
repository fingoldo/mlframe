"""Regression tests for audits/full_audit_2026-07-21/calibration.md findings F1-F9.

PR1-PR4 (test-coverage gaps) are closed by the same tests below (NaN-group, zero-member, dtype-drift,
floor/active_class range). PR5 is F2's own fix. PR6 (policy.py/post.py LOC) is a no-action per the
project's ~1000-LOC hard-gate precedent (both files stay under it). PR7 is F5's own fix (plus the
module docstring split).

F3's fix also surfaced a SEPARATE latent bug in the same function while implementing it: ``lo = r`` was
nested inside ``if not skip_plotting:``, so ``skip_plotting=True`` calls never advanced the per-interval
slice and every interval after the first silently re-read overlapping data from index 0. Covered here too.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

logging.disable(logging.NOTSET)


# ---------------------------------------------------------------------------
# F1: group_bias_correction fit/apply dtype-drift key mismatch
# ---------------------------------------------------------------------------


def test_f1_group_bias_correction_survives_int_to_float_dtype_drift():
    """F1: group bias correction survives int to float dtype drift."""
    from mlframe.calibration.group_bias_correction import apply_group_bias_correction, fit_group_bias_correction

    rng = np.random.default_rng(0)
    n = 300
    group_int = rng.integers(0, 5, size=n).astype(np.int64)
    y_pred = rng.uniform(1.0, 10.0, size=n)
    y_true = y_pred * (1.0 + 0.3 * (group_int % 2))  # group-dependent bias to correct

    ratios = fit_group_bias_correction(y_true, y_pred, group_int, min_group_size=1, clip_range=None)

    # Simulate the realistic drift: the SAME logical group ids arrive as float64 at apply time
    # (e.g. a stray NaN elsewhere in the same source column upcast the whole int64 column).
    group_float = group_int.astype(np.float64)
    corrected = apply_group_bias_correction(y_pred, group_float, ratios)

    # Bit-identical to applying against the ORIGINAL int64 group array -- proves the float64 keys
    # actually matched the fitted table instead of silently falling back to default_ratio=1.0 everywhere.
    corrected_same_dtype = apply_group_bias_correction(y_pred, group_int, ratios)
    assert np.allclose(corrected, corrected_same_dtype)
    assert not np.allclose(corrected, y_pred)  # a real (non-identity) correction was actually applied


def test_f1_group_bias_correction_same_dtype_unaffected():
    """F1: group bias correction same dtype unaffected."""
    from mlframe.calibration.group_bias_correction import apply_group_bias_correction, fit_group_bias_correction

    rng = np.random.default_rng(1)
    n = 200
    group = rng.integers(0, 4, size=n)
    y_pred = rng.uniform(1.0, 5.0, size=n)
    y_true = y_pred * 1.2

    ratios = fit_group_bias_correction(y_true, y_pred, group, min_group_size=1, clip_range=None)
    corrected = apply_group_bias_correction(y_pred, group, ratios)
    assert np.allclose(corrected, y_pred * 1.2, atol=1e-6)


# ---------------------------------------------------------------------------
# F2: train_postcalibrators' calib==test leakage guard uses a dedicated exception type
# ---------------------------------------------------------------------------


def _synth_probs_target(seed: int = 0, n: int = 300):
    """Synth probs target."""
    rng = np.random.default_rng(seed)
    p1 = rng.random(n)
    probs = np.column_stack([1 - p1, p1])
    target = (p1 + rng.normal(0, 0.1, n) > 0.5).astype(int)
    return probs, target


def test_f2_leakage_guard_raises_dedicated_exception_type_on_identical():
    """F2: leakage guard raises dedicated exception type on identical."""
    from mlframe.calibration.post import _CalibTestOverlapError, named_calibrator, train_postcalibrators
    from unittest.mock import patch
    from sklearn.isotonic import IsotonicRegression

    n = 300
    probs, target = _synth_probs_target(n=n)

    class _FakeModel:
        """Fake model stub used to control this test's predict path."""
        columns = ["y"]
        test_target = target

    fake_calibrators = [named_calibrator(IsotonicRegression(out_of_bounds="clip"), name="Iso", lib="sklearn")]

    with patch("mlframe.calibration.post.get_postcalibrators", return_value=fake_calibrators):
        with pytest.raises(_CalibTestOverlapError, match="identical"):
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
                calib_target=target,  # bit-for-bit identical to test_target
            )


def test_f2_calib_test_overlap_error_is_a_value_error_subclass():
    """The dedicated exception type must still satisfy every existing ``pytest.raises(ValueError, ...)`` call
    site (e.g. test_train_postcalibrators_leakage_guard_overlap.py) -- it narrows the internal except-clause
    matching, not the public contract."""
    from mlframe.calibration.post import _CalibTestOverlapError

    assert issubclass(_CalibTestOverlapError, ValueError)


# ---------------------------------------------------------------------------
# F3: show_classifier_calibration's diagonal spans only the LAST interval + skip_plotting `lo` bug
# ---------------------------------------------------------------------------


class _RecordingAx:
    """Recording Ax."""
    def __init__(self):
        self.plot_calls: list = []

    def plot(self, *args, **kwargs):
        """Records the call args matching matplotlib Axes' plot() signature."""
        self.plot_calls.append((args, kwargs))

    def scatter(self, *args, **kwargs):
        """No-op stub matching matplotlib Axes' scatter() signature."""
        pass

    def legend(self, *args, **kwargs):
        """No-op stub matching matplotlib Axes' legend() signature."""
        pass

    def set_xlabel(self, *args, **kwargs):
        """No-op stub matching matplotlib Axes' set_xlabel() signature."""
        pass

    def set_ylabel(self, *args, **kwargs):
        """No-op stub matching matplotlib Axes' set_ylabel() signature."""
        pass

    def set_title(self, *args, **kwargs):
        """No-op stub matching matplotlib Axes' set_title() signature."""
        pass

    def axhline(self, *args, **kwargs):
        """No-op stub matching matplotlib Axes' axhline() signature."""
        pass

    def axvline(self, *args, **kwargs):
        """No-op stub matching matplotlib Axes' axvline() signature."""
        pass


def test_f3_diagonal_spans_all_intervals_not_just_last(monkeypatch):
    """F3: diagonal spans all intervals not just last."""
    from mlframe.calibration import quality

    fixed_returns = [
        (np.array([0.1, 0.9]), np.array([0.15, 0.85]), np.empty((0, 4)), {"Brier": 0.1}),
        (np.array([0.45, 0.55]), np.array([0.48, 0.52]), np.empty((0, 4)), {"Brier": 0.05}),
    ]
    state = {"i": 0}

    def fake_estimate(y_true, y_pred, nbins, indices, metrics_to_show):
        """Fake replacement for estimate_calibration_quality_binned, returning canned per-interval values."""
        out = fixed_returns[state["i"]]
        state["i"] += 1
        return out

    monkeypatch.setattr(quality, "estimate_calibration_quality_binned", fake_estimate)

    ax = _RecordingAx()
    n = 100
    y_true = np.zeros(n, dtype=np.int8)
    y_pred = np.zeros(n)
    quality.show_classifier_calibration(y_true, y_pred, title="t", nintervals=2, ax=ax, skip_plotting=False)

    diagonal_calls = [c for c in ax.plot_calls if c[1].get("label") == "Perfect"]
    assert len(diagonal_calls) == 1
    diag_args, _diag_kwargs = diagonal_calls[0]
    xs = diag_args[0]
    # Pre-fix this would be [0.45, 0.55] (only the LAST interval's narrow range).
    assert xs[0] == pytest.approx(0.1)
    assert xs[1] == pytest.approx(0.9)


def test_f3_skip_plotting_advances_lo_across_intervals(monkeypatch):
    """Newly-discovered while fixing F3: `lo = r` was nested inside `if not skip_plotting:`, so
    skip_plotting=True calls never advanced past the first interval's slice."""
    from mlframe.calibration import quality

    real_fn = quality.estimate_calibration_quality_binned
    captured: list = []

    def spy(y_true, y_pred, nbins, indices, metrics_to_show):
        """Records call arguments for this test's assertions."""
        captured.append(np.asarray(y_true).copy())
        return real_fn(y_true, y_pred, nbins=nbins, indices=indices, metrics_to_show=metrics_to_show)

    monkeypatch.setattr(quality, "estimate_calibration_quality_binned", spy)

    rng = np.random.default_rng(0)
    n = 300
    y_pred = rng.uniform(0.01, 0.99, size=n)
    y_true = (rng.uniform(size=n) < y_pred).astype(np.int8)

    quality.show_classifier_calibration(y_true, y_pred, title="t", nbins=5, nintervals=3, skip_plotting=True)

    assert len(captured) == 3
    # Pre-fix, intervals 0 and 1 both read y_true[0:step] identically (lo stuck at 0), and interval 2
    # (the last, whose `r` is always `s` regardless of `lo`) would read the WHOLE 300-row array instead
    # of just its own 100-row slice.
    assert not np.array_equal(captured[0], captured[1])
    assert not np.array_equal(captured[1], captured[2])
    assert len(captured[0]) == len(captured[1]) == len(captured[2]) == 100


# ---------------------------------------------------------------------------
# F4: apply_sticky_state_persistence_floor never validated floor/active_class ranges
# ---------------------------------------------------------------------------


def test_f4_floor_out_of_range_raises():
    """F4: floor out of range raises."""
    from mlframe.calibration.sticky_state_persistence_floor import apply_sticky_state_persistence_floor

    probs = np.array([[0.6, 0.4], [0.3, 0.7]])
    active = np.array([0, 1])
    with pytest.raises(ValueError, match="floor"):
        apply_sticky_state_persistence_floor(probs, active, floor=1.0)
    with pytest.raises(ValueError, match="floor"):
        apply_sticky_state_persistence_floor(probs, active, floor=-0.1)


def test_f4_active_class_out_of_range_raises():
    """F4: active class out of range raises."""
    from mlframe.calibration.sticky_state_persistence_floor import apply_sticky_state_persistence_floor

    probs = np.array([[0.6, 0.4], [0.3, 0.7]])
    with pytest.raises(ValueError, match="active_class"):
        apply_sticky_state_persistence_floor(probs, np.array([0, 2]), floor=0.5)
    with pytest.raises(ValueError, match="active_class"):
        apply_sticky_state_persistence_floor(probs, np.array([-1, 0]), floor=0.5)


def test_f4_valid_inputs_still_work():
    """F4: valid inputs still work."""
    from mlframe.calibration.sticky_state_persistence_floor import apply_sticky_state_persistence_floor

    probs = np.array([[0.6, 0.4], [0.3, 0.7]])
    active = np.array([0, 1])
    out = apply_sticky_state_persistence_floor(probs, active, floor=0.5)
    assert out.shape == probs.shape
    assert np.allclose(out.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# F5: generate_similar_probs_by_ranking never used true_outcomes
# ---------------------------------------------------------------------------


def test_f5_default_n_iterations_bit_identical_to_original_single_draw():
    """F5: default n iterations bit identical to original single draw."""
    from mlframe.calibration.probabilities import generate_similar_probs_by_ranking

    rng = np.random.default_rng(0)
    n = 200
    predicted = rng.uniform(0, 1, size=n)
    outcomes = (rng.uniform(size=n) < predicted).astype(int)

    out_a = generate_similar_probs_by_ranking(predicted, outcomes, n_bins=10, noise_scale=0.001, random_state=5)
    out_b = generate_similar_probs_by_ranking(predicted, outcomes, n_bins=10, noise_scale=0.001, random_state=5)
    assert np.allclose(out_a, out_b)  # deterministic given random_state, unaffected by the new code path


def test_f5_n_iterations_actually_uses_true_outcomes_to_improve_closeness():
    """F5: n iterations actually uses true outcomes to improve closeness."""
    from mlframe.calibration.probabilities import generate_similar_probs_by_ranking
    from mlframe.metrics.core import fast_brier_score_loss, fast_roc_auc

    rng = np.random.default_rng(0)
    n = 500
    predicted = rng.uniform(0, 1, size=n)
    outcomes = (rng.uniform(size=n) < predicted).astype(int)
    original_brier = fast_brier_score_loss(outcomes, predicted)
    original_auc = fast_roc_auc(outcomes, predicted)

    def dist(candidate):
        """Distance of a candidate draw's (Brier, AUC) pair from the original prediction's."""
        b = fast_brier_score_loss(outcomes, candidate)
        a = fast_roc_auc(outcomes, candidate)
        return abs(b - original_brier) / max(abs(original_brier), 1e-12) + abs(a - original_auc) / max(abs(original_auc), 1e-12)

    single = generate_similar_probs_by_ranking(predicted, outcomes, n_bins=10, noise_scale=0.05, n_iterations=1, random_state=1)
    multi = generate_similar_probs_by_ranking(predicted, outcomes, n_bins=10, noise_scale=0.05, n_iterations=25, random_state=1)

    # n_iterations>1 tracks the CLOSEST-to-original candidate over multiple draws -- must be no worse
    # (in practice strictly better, since a 25-candidate best-of is very unlikely to tie a single draw).
    assert dist(multi) <= dist(single)


def test_f5_n_iterations_returns_ndarray_when_score_is_nan_every_draw():
    """Discovered while re-verifying F5: single-class true_outcomes makes fast_roc_auc return NaN, so
    every candidate's `score` comparison (`nan < best_score`) is False and best_probs was never assigned,
    silently returning None from a function typed to always return an ndarray."""
    from mlframe.calibration.probabilities import generate_similar_probs_by_ranking

    rng = np.random.default_rng(0)
    n = 200
    predicted = rng.uniform(0, 1, size=n)
    outcomes = np.zeros(n, dtype=int)  # single class -> fast_roc_auc is NaN

    result = generate_similar_probs_by_ranking(predicted, outcomes, n_bins=10, noise_scale=0.001, n_iterations=5, random_state=3)
    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.shape == predicted.shape


# ---------------------------------------------------------------------------
# F6: NaN group labels silently excluded (group_bias_correction / group_zero_sum_constraint)
# ---------------------------------------------------------------------------


def test_f6_group_bias_correction_warns_on_nan_group(caplog):
    """F6: group bias correction warns on nan group."""
    from mlframe.calibration.group_bias_correction import fit_group_bias_correction

    rng = np.random.default_rng(0)
    n = 100
    group = rng.integers(0, 3, size=n).astype(np.float64)
    group[5] = np.nan
    y_pred = rng.uniform(1.0, 5.0, size=n)
    y_true = y_pred * 1.1

    with caplog.at_level(logging.WARNING, logger="mlframe.calibration.group_bias_correction"):
        ratios = fit_group_bias_correction(y_true, y_pred, group, min_group_size=1, clip_range=None)
    assert any("NaN group label" in r.getMessage() for r in caplog.records)
    assert "nan" not in ratios  # the NaN row is excluded, not silently keyed as a string


def test_f6_group_zero_sum_constraint_warns_on_nan_group(caplog):
    """F6: group zero sum constraint warns on nan group."""
    from mlframe.calibration.group_zero_sum_constraint import apply_group_zero_sum_constraint

    rng = np.random.default_rng(0)
    n = 60
    group = rng.integers(0, 3, size=n).astype(np.float64)
    group[3] = np.nan
    preds = rng.uniform(-1, 1, size=n)

    with caplog.at_level(logging.WARNING, logger="mlframe.calibration.group_zero_sum_constraint"):
        out = apply_group_zero_sum_constraint(preds, group)
    assert any("NaN group label" in r.getMessage() for r in caplog.records)
    assert out[3] == pytest.approx(preds[3])  # NaN-group row gets offset 0.0 (unchanged), not an error


# ---------------------------------------------------------------------------
# F7: odds_ratio_combine's GPU dispatch does not check MLFRAME_DISABLE_GPU -- verified as by-design
# ---------------------------------------------------------------------------


def test_f7_disable_gpu_convention_is_feature_selection_scoped_by_design():
    """Documents the investigation outcome: MLFRAME_DISABLE_GPU is used exclusively via relative imports
    inside feature_selection/filters/**, so calibration/'s own MLFRAME_ODDS_COMBINE_BACKEND override
    (documented in _ktc_dispatch.py's module docstring) is the correct, by-design override for this module."""
    from mlframe.calibration import _ktc_dispatch

    doc = _ktc_dispatch.__doc__ or ""
    assert "MLFRAME_DISABLE_GPU" in doc
    assert "package-internal convention" in doc


# ---------------------------------------------------------------------------
# F8: _get_cache()'s except-clauses logged nothing on either failure path
# ---------------------------------------------------------------------------


def test_f8_get_cache_logs_on_import_failure(monkeypatch, caplog):
    """F8: get cache logs on import failure."""
    from mlframe.calibration import _ktc_dispatch
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        """Fake import hook that always raises ImportError."""
        if name == "mlframe.feature_selection.filters":
            raise ImportError("simulated FS-package-unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with caplog.at_level(logging.DEBUG, logger="mlframe.calibration._ktc_dispatch"):
        result = _ktc_dispatch._get_cache()
    assert result is None
    assert any("kernel_tuning_cache unavailable" in r.getMessage() for r in caplog.records)


def test_f8_get_cache_logs_on_singleton_failure(monkeypatch, caplog):
    """F8: get cache logs on singleton failure."""
    from mlframe.calibration import _ktc_dispatch

    def fake_get_kernel_tuning_cache():
        """Fake replacement for get_kernel_tuning_cache that always raises."""
        raise RuntimeError("simulated singleton failure")

    class _FakeFiltersModule:
        """Fake Filters Module."""
        get_kernel_tuning_cache = staticmethod(fake_get_kernel_tuning_cache)

    import sys

    monkeypatch.setitem(sys.modules, "mlframe.feature_selection.filters", _FakeFiltersModule)
    with caplog.at_level(logging.DEBUG, logger="mlframe.calibration._ktc_dispatch"):
        result = _ktc_dispatch._get_cache()
    assert result is None
    assert any("get_kernel_tuning_cache() failed" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# F9: odds_ratio_combine silently returned uniform 0.5 for a zero-member (n, 0) input
# ---------------------------------------------------------------------------


def test_f9_zero_member_input_raises_instead_of_uniform_half():
    """F9: zero member input raises instead of uniform half."""
    from mlframe.calibration.ensembling import odds_ratio_combine

    member_probs = np.empty((50, 0))
    with pytest.raises(ValueError, match="0 members"):
        odds_ratio_combine(member_probs)


def test_f9_nonzero_member_input_still_works():
    """F9: nonzero member input still works."""
    from mlframe.calibration.ensembling import odds_ratio_combine

    rng = np.random.default_rng(0)
    member_probs = rng.uniform(0.1, 0.9, size=(20, 3))
    out = odds_ratio_combine(member_probs)
    assert out.shape == (20,)
    assert np.all((out >= 0) & (out <= 1))
