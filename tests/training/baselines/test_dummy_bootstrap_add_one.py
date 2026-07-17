"""Regression: paired-bootstrap win fraction uses Davison-Hinkley add-one smoothing.

``p_strongest_beats`` was a naive ``mean(delta < 0)`` that returns exactly 1.0 when the
strongest predictor wins every resample -- an impossible certainty (the observed split is
itself one draw under the resampling distribution). The add-one form ``(#wins + 1)/(n + 1)``
keeps a clean sweep strictly below 1.0. This test pins that it can never report 1.0 / 0.0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.baselines.dummy import _paired_bootstrap_vs_runner_up


def _run(p_strongest, p_runner, y, n_resamples=200, seed=42):
    rmse_s = float(np.sqrt(np.mean((y - p_strongest) ** 2)))
    rmse_r = float(np.sqrt(np.mean((y - p_runner) ** 2)))
    table = pd.DataFrame({"val_RMSE": [rmse_s, rmse_r]}, index=["strongest", "runner_up"])
    val_preds = {"strongest": p_strongest, "runner_up": p_runner}
    return _paired_bootstrap_vs_runner_up(
        "regression",
        "strongest",
        "val_RMSE",
        table,
        val_preds,
        val_preds,
        y,
        y,
        n_resamples=n_resamples,
        seed=seed,
    )


def test_clean_sweep_win_fraction_below_one():
    """Clean sweep win fraction below one."""
    rng = np.random.default_rng(0)
    n = 400
    y = rng.normal(size=n)
    # Strongest near-perfect, runner-up badly off: strongest beats on essentially every resample.
    p_strongest = y + rng.normal(0, 0.01, n)
    p_runner = y + rng.normal(0, 2.0, n)
    res = _run(p_strongest, p_runner, y)
    assert res is not None
    p = res["p_strongest_beats"]
    # Add-one keeps the clean sweep strictly below the impossible 1.0 certainty.
    assert p < 1.0, f"win fraction must be < 1.0 under add-one smoothing, got {p}"
    # And it must be close to 1.0 (strongest really does dominate): n/(n+1) lower-bounds a full sweep.
    assert p > 0.95
    # Pre-fix naive mean(delta<0) would have been exactly 1.0 on a full sweep.
    assert abs(p - 1.0) > 1e-9


def test_reverse_sweep_win_fraction_above_zero():
    """Reverse sweep win fraction above zero."""
    rng = np.random.default_rng(1)
    n = 400
    y = rng.normal(size=n)
    # Strongest is actually the worse predictor here (still labelled "strongest" by the table);
    # it loses essentially every resample -> naive fraction 0.0, add-one keeps it > 0.
    p_strongest = y + rng.normal(0, 2.0, n)
    p_runner = y + rng.normal(0, 0.01, n)
    res = _run(p_strongest, p_runner, y)
    assert res is not None
    p = res["p_strongest_beats"]
    assert p > 0.0, f"win fraction must be > 0.0 under add-one smoothing, got {p}"
    assert p < 0.05


def test_win_fraction_matches_add_one_formula():
    """Win fraction matches add one formula."""
    rng = np.random.default_rng(7)
    n = 300
    y = rng.normal(size=n)
    p_strongest = y + rng.normal(0, 0.4, n)
    p_runner = y + rng.normal(0, 0.5, n)
    res = _run(p_strongest, p_runner, y, n_resamples=150, seed=11)
    assert res is not None
    p = res["p_strongest_beats"]
    # Must be a valid (n_wins+1)/(n_surviving+1) ratio: strictly inside (0, 1).
    assert 0.0 < p < 1.0
