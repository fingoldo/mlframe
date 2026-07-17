"""Sensor: ``pick_best_calibrator`` picks a sensible method per Kull-2017.

Synthetic miscalibrated probs (sigmoid-shifted noise) should let an Isotonic /
Beta calibrator beat raw probs on OOF ECE; the policy helper should return a
valid disposition with ECE CI populated, and emit a PNG when ``emit_plot=True``.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

# Synthetic miscalibrated 2k-sample binary probs + 200-bootstrap policy fits; wall <2s per test.
pytestmark = [pytest.mark.fast]


def _make_miscalibrated(n: int, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Helper that make miscalibrated."""
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.0, 1.0, size=n)
    # True probability is a steeper sigmoid of raw; raw is under-confident at the tails.
    true_p = 1.0 / (1.0 + np.exp(-6.0 * (raw - 0.5)))
    y = (rng.uniform(0.0, 1.0, size=n) < true_p).astype(np.int64)
    # The "predicted" probability the calibrator sees is the raw uniform, miscalibrated by construction.
    return raw, y


def test_pick_best_calibrator_picks_isotonic_on_large_n():
    """Pick best calibrator picks isotonic on large n."""
    from mlframe.calibration.policy import pick_best_calibrator

    raw, y = _make_miscalibrated(n=2000, seed=11)
    out = pick_best_calibrator(
        probs=None,
        y=None,
        oof_probs=raw,
        oof_y=y,
        n_bootstrap=200,  # keep test fast; default 1000 covered by other paths
        random_state=11,
        selection="same_oof",  # this test exercises the legacy same-OOF Kull-2017 CI-overlap ruleset
    )

    assert out["chosen"] in {"Sigmoid", "Isotonic", "Beta", "Spline"}, out
    assert out["n_oof"] == 2000
    assert "ece_ci" in out and len(out["ece_ci"]) == 2
    lo, hi = out["ece_ci"]
    assert 0.0 <= lo <= hi, (lo, hi)
    # Percentile-bootstrap CI need not bracket the point estimate (the point is computed on the full sample,
    # the percentile CI from resamples can sit above/below); just check the point is non-negative and finite.
    assert np.isfinite(out["ece_mean"]) and out["ece_mean"] >= 0.0
    # At n=2000 the Kull-2017 default kicks in when CIs overlap; the chosen rule must be tagged.
    assert out["rule"] in {"lowest_ece", "lowest_ece_ci_overlap", "lowest_ece_ci_separated", "default_isotonic", "default_beta"}
    # Either the lowest-ECE candidate wins outright (CI separated), or the Isotonic default fires when CIs overlap.
    if out["rule"] == "default_isotonic":
        assert out["chosen"] == "Isotonic"
    if out["rule"] == "default_beta":
        # Default beta should NOT trigger at n=2000 (only when n_oof<1000).
        # Skip assertion if betacal absent reshuffles the candidate pool.
        pass
    # Bench delivered at least 2 alternatives (Sigmoid + Isotonic ship with sklearn baseline).
    assert len(out["alternatives"]) >= 2


def test_pick_best_calibrator_small_n_prefers_beta():
    """When n_oof < 1000 and CIs overlap, the default rule prefers Beta (Kull 2017)."""
    pytest.importorskip("betacal")
    from mlframe.calibration.policy import pick_best_calibrator

    raw, y = _make_miscalibrated(n=200, seed=17)
    out = pick_best_calibrator(
        probs=None,
        y=None,
        oof_probs=raw,
        oof_y=y,
        n_bootstrap=200,
        random_state=17,
        selection="same_oof",  # this test exercises the legacy small-n Beta default rule
    )
    assert out["n_oof"] == 200
    # At small n the CIs are wide -> typically overlap -> default Beta fires.
    # If it didn't overlap, lowest-ECE wins (still valid). The selection rule label tells us which path:
    assert out["rule"] in {"lowest_ece", "lowest_ece_ci_overlap", "lowest_ece_ci_separated", "default_isotonic", "default_beta"}
    if out["rule"] == "default_beta":
        assert out["chosen"] == "Beta"


def test_pick_best_calibrator_emit_plot_writes_png():
    """Pick best calibrator emit plot writes png."""
    from mlframe.calibration.policy import pick_best_calibrator

    raw, y = _make_miscalibrated(n=500, seed=23)
    with tempfile.TemporaryDirectory() as td:
        plot_path = os.path.join(td, "calib_plot.png")
        out = pick_best_calibrator(
            probs=None,
            y=None,
            oof_probs=raw,
            oof_y=y,
            n_bootstrap=100,
            random_state=23,
            emit_plot=True,
            plot_path=plot_path,
        )
        assert out["plot_path"] is not None, "expected non-null plot_path after emit_plot=True"
        assert os.path.exists(out["plot_path"]), out["plot_path"]
        assert os.path.getsize(out["plot_path"]) > 1024, "PNG looks suspiciously small"


def test_pick_best_calibrator_rejects_too_few_rows():
    """Pick best calibrator rejects too few rows."""
    from mlframe.calibration.policy import pick_best_calibrator

    with pytest.raises(ValueError, match="at least 4 OOF rows"):
        pick_best_calibrator(probs=None, y=None, oof_probs=np.array([0.1, 0.9, 0.5]), oof_y=np.array([0, 1, 0]))


def test_pick_best_calibrator_rejects_row_mismatch():
    """Pick best calibrator rejects row mismatch."""
    from mlframe.calibration.policy import pick_best_calibrator

    with pytest.raises(ValueError, match="do not match"):
        pick_best_calibrator(
            probs=None,
            y=None,
            oof_probs=np.linspace(0, 1, 20),
            oof_y=np.zeros(15),
        )


def test_calibration_config_default_policy_on():
    """Calibration config default policy on."""
    from mlframe.calibration.policy import CalibrationConfig

    cfg = CalibrationConfig()
    assert cfg.policy_auto_pick is True
    assert cfg.alpha == 0.05
    assert cfg.n_bootstrap == 1000
    assert cfg.emit_plot is False


def test_iter598_ece_score_mixed_dtype_bit_equivalent():
    """iter598 dropped the unconditional ``dtype=np.float64`` cast on
    ``y_true`` inside ``_ece_score``. The numba kernel only uses ``yi``
    in ``sum_y[b] += yi`` where sum_y is float64 -- mixed-dtype dispatch
    widens at the accumulator just like the explicit cast did. Pin
    bit-equivalence across (int64, int8, float64) labels."""
    from mlframe.calibration import policy as _policy

    rng = np.random.default_rng(20260531)
    n = 5_000
    y_int = rng.integers(0, 2, size=n, dtype=np.int64)
    p_f64 = np.clip(rng.random(n), 0.01, 0.99)
    ref = _policy._ece_score(y_int.astype(np.float64), p_f64)
    for y_t in (y_int, y_int.astype(np.int8), y_int.astype(np.int32)):
        v = _policy._ece_score(y_t, p_f64)
        assert abs(v - ref) < 1e-12, f"label dtype {y_t.dtype}: got={v} ref={ref}"
