"""biz_value: inner-CV calibrator selection generalises better than same-OOF selection.

``selection="same_oof"`` (legacy) fits AND scores each candidate on the same OOF rows, so Isotonic
interpolates its in-sample ECE to ~0 and is selected on an optimistic score that does not hold on fresh
data. ``selection="inner_cv"`` (default) ranks by held-out ECE, so on a small over-fit-prone OOF it picks
a calibrator whose FRESH-HOLDOUT ECE is materially lower, and the reported ECE is honest (no longer ~0).

Measured at n=300 over 8 seeds: same_oof holdout 0.00343 -> inner_cv holdout 0.00227 (~34% lower);
same_oof reported 0.0 -> inner_cv reported ~0.01. Floors set well below the measured margins.
"""
from __future__ import annotations

import numpy as np

from mlframe.calibration.policy import _ece_score, _fit_calibrator, pick_best_calibrator


def _gen(rng, n, slope, noise):
    y = (rng.random(n) < 0.5).astype(int)
    p = np.clip(0.5 + (y - 0.5) * slope + rng.normal(0, noise, n), 0.005, 0.995)
    return p, y


def _holdout_ece(selection, oof_p, oof_y, ho_p, ho_y, seed):
    res = pick_best_calibrator(None, None, oof_p, oof_y, n_bootstrap=150, random_state=seed, selection=selection)
    cal = _fit_calibrator(res["chosen"], oof_p, oof_y)
    holdout = float(_ece_score(ho_y, cal(ho_p)))
    return float(res["ece_mean"]), holdout


def test_biz_val_policy_inner_cv_beats_same_oof_holdout_on_small_overfit():
    """inner_cv mean fresh-holdout ECE beats same_oof by >= 0.0006 on a small over-fit OOF (measured 0.00116)."""
    old_hos, new_hos = [], []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        oof_p, oof_y = _gen(rng, 300, 1.6, 0.25)
        ho_p, ho_y = _gen(rng, 300, 1.6, 0.25)
        _, old_ho = _holdout_ece("same_oof", oof_p, oof_y, ho_p, ho_y, seed)
        _, new_ho = _holdout_ece("inner_cv", oof_p, oof_y, ho_p, ho_y, seed)
        old_hos.append(old_ho)
        new_hos.append(new_ho)
    old_mean, new_mean = float(np.mean(old_hos)), float(np.mean(new_hos))
    assert new_mean <= old_mean - 0.0006, f"inner_cv holdout {new_mean:.5f} not >=0.0006 better than same_oof {old_mean:.5f}"


def test_biz_val_policy_inner_cv_reported_ece_is_honest_not_zero():
    """inner_cv reports the held-out ECE (>1e-4), not the same_oof ~0 interpolation optimism."""
    rng = np.random.default_rng(0)
    oof_p, oof_y = _gen(rng, 300, 1.6, 0.25)
    same = pick_best_calibrator(None, None, oof_p, oof_y, n_bootstrap=100, random_state=0, selection="same_oof")
    inner = pick_best_calibrator(None, None, oof_p, oof_y, n_bootstrap=100, random_state=0, selection="inner_cv")
    assert same["ece_mean"] < 1e-6, f"same_oof should report ~0 interpolation ECE; got {same['ece_mean']}"
    assert inner["ece_mean"] > 1e-4, f"inner_cv reported ECE should be honest (>1e-4); got {inner['ece_mean']}"
    assert inner["rule"] == "lowest_heldout_ece"


def test_biz_val_policy_inner_cv_no_worse_on_large_n():
    """On large n Isotonic generalises fine, so inner_cv must not regress vs same_oof holdout ECE."""
    old_hos, new_hos = [], []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        oof_p, oof_y = _gen(rng, 4000, 0.9, 0.2)
        ho_p, ho_y = _gen(rng, 4000, 0.9, 0.2)
        _, old_ho = _holdout_ece("same_oof", oof_p, oof_y, ho_p, ho_y, seed)
        _, new_ho = _holdout_ece("inner_cv", oof_p, oof_y, ho_p, ho_y, seed)
        old_hos.append(old_ho)
        new_hos.append(new_ho)
    assert float(np.mean(new_hos)) <= float(np.mean(old_hos)) + 5e-4
