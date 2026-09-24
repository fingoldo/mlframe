"""A degenerate H-statistic ratio (> 1) must not be reported as maximal interaction."""

import numpy as np

from mlframe.inspection import interaction as it

_PD = {"grid": None, "pdp": None}  # the interpolation that reads these is patched in each test


def _patch_pdps(monkeypatch, c_ij, c_i, c_j):
    monkeypatch.setattr(it, "compute_pdp_2d", lambda *a, **k: {"grid0": None, "grid1": None, "surface": None})
    monkeypatch.setattr(it, "_centered_interp_2d", lambda *a, **k: c_ij)
    seq = iter([c_i, c_j])
    monkeypatch.setattr(it, "_centered_interp_1d", lambda *a, **k: next(seq))


def test_strong_main_effects_on_a_flat_joint_surface_give_nan(monkeypatch):
    """numer/denom > 1 used to clamp to H = 1.0, putting an additive pair at the top of the heatmap."""
    c_i = np.linspace(-1.0, 1.0, 50)
    c_j = np.linspace(-1.0, 1.0, 50) * 0.9  # same direction, so the main effects add up rather than cancel
    c_ij = np.linspace(-0.05, 0.05, 50)  # nearly flat joint surface
    _patch_pdps(monkeypatch, c_ij, c_i, c_j)
    h = it._h_from_1d_pdps(None, None, 0, 1, _PD, _PD, None, None, grid=10, sample=10, seed=0)
    assert np.isnan(h)


def test_a_pure_interaction_still_scores_near_one(monkeypatch):
    rng = np.random.default_rng(0)
    c_ij = rng.normal(size=50)
    _patch_pdps(monkeypatch, c_ij, np.zeros(50), np.zeros(50))
    assert it._h_from_1d_pdps(None, None, 0, 1, _PD, _PD, None, None, grid=10, sample=10, seed=0) == 1.0


def test_a_small_overshoot_from_estimation_noise_is_still_a_full_interaction(monkeypatch):
    """A pure product's ratio lands a hair above 1 on a sampled grid; that is noise, not degeneracy."""
    rng = np.random.default_rng(1)
    c_ij = rng.normal(size=50)
    c_i = rng.normal(scale=0.05, size=50)
    _patch_pdps(monkeypatch, c_ij, c_i, -c_i * 0.5)
    assert it._h_from_1d_pdps(None, None, 0, 1, _PD, _PD, None, None, grid=10, sample=10, seed=0) > 0.95


def test_an_additive_pair_scores_near_zero(monkeypatch):
    c_i = np.linspace(-1.0, 1.0, 50)
    c_j = np.cos(np.linspace(0, 3, 50))
    c_j = c_j - c_j.mean()
    _patch_pdps(monkeypatch, c_i + c_j, c_i, c_j)
    assert it._h_from_1d_pdps(None, None, 0, 1, _PD, _PD, None, None, grid=10, sample=10, seed=0) < 1e-6
