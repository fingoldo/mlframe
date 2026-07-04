"""Regression: the noise-floor permutation null in optimise_hermite_pair runs on a STRIDED subsample (cap 30k, env
MLFRAME_FE_NOISE_FLOOR_MAX_ROWS) of the operands instead of the full n. The permutation p95 is a coarse floor compared
against a 1.5x ratio, so the accept/reject verdict must be SELECTION-EQUIVALENT to the full-n null on real targets. This
pins that equivalence (the reject decision -- r is None -- matches capped vs uncapped) on a genuine-interaction target, a
pure-noise target (rejected by the null), and a borderline-weak target; a future change that makes the cap alter which
engineered pairs survive fails here. Cap triggers only for n>30k, so n is set above the cap."""
import os
import numpy as np
import pytest

from mlframe.feature_selection.filters.hermite_fe import optimise_hermite_pair

_KW = dict(n_trials=60, min_degree=3, max_degree=5, basis="chebyshev", mi_estimator="plugin",
           plugin_n_bins=20, optimizer="cma_batch", discrete_target=True, sweep_degrees=True,
           seed=42, noise_floor_n_perms=40)


def _mk(rng, n, kind):
    a = rng.standard_normal(n).astype(np.float64)
    b = (rng.standard_normal(n) + 0.3).astype(np.float64)
    if kind == "prod":    yc = a * b
    elif kind == "cubic": yc = a ** 3 - 2 * a * b
    elif kind == "noise": yc = rng.standard_normal(n)
    else:                 yc = 0.03 * a * b + rng.standard_normal(n)
    yc = np.nan_to_num(yc) + rng.standard_normal(n) * 0.3
    y = np.digitize(yc, np.quantile(yc, np.linspace(0, 1, 11)[1:-1])).astype(np.int64)
    return np.nan_to_num(a), np.nan_to_num(b), y


@pytest.mark.parametrize("kind", ["prod", "cubic", "noise", "weak"])
def test_noise_floor_cap_is_selection_equivalent_to_full_n(kind, monkeypatch):
    rng = np.random.default_rng(0)
    n = 60000  # above the 30k cap so the strided subsample path is exercised
    xa, xb, y = _mk(rng, n, kind)
    optimise_hermite_pair(xa, xb, y, **_KW)  # warm numba

    monkeypatch.setenv("MLFRAME_FE_NOISE_FLOOR_MAX_ROWS", "0")  # disable cap -> full-n null
    r_full = optimise_hermite_pair(xa, xb, y, **_KW)
    monkeypatch.setenv("MLFRAME_FE_NOISE_FLOOR_MAX_ROWS", "30000")  # capped null
    r_cap = optimise_hermite_pair(xa, xb, y, **_KW)

    assert (r_full is None) == (r_cap is None), (
        f"noise-floor cap flipped the reject decision on '{kind}': full={r_full is None} cap={r_cap is None}")
