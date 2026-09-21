"""Stacked cross-target ensembles: the fallback gate can fire, and capping keeps the blend on its level.

NNLS scored on the matrix it was fitted on never loses to its best single component (every unit vector is feasible), so
the gate that falls back to the best single component could not fire for ``nnls_stack``; and capping a non-convex stack
kept the top components' raw weights, so the served blend lost the dropped weight mass on every row.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from mlframe.training.composite.ensemble import CompositeCrossTargetEnsemble as Ens
from mlframe.training.core._phase_composite_post_xt_ensemble._crossfit import cross_fitted_stack_rmse, refit_capped_stack


def _pool(seed: int, n: int = 60, n_noise: int = 20):
    """One good component (y + noise) and ``n_noise`` pure-noise components around the mean."""
    rng = np.random.default_rng(seed)
    y = rng.normal(100.0, 10.0, n)
    good = y + rng.normal(0.0, 5.0, n)
    P = np.column_stack([good] + [np.full(n, y.mean()) + rng.normal(0.0, 10.0, n) for _ in range(n_noise)])
    comps = [types.SimpleNamespace() for _ in range(P.shape[1])]
    return y, P, comps, [f"c{i}" for i in range(P.shape[1])]


@pytest.mark.parametrize("seed", [1, 2])
def test_the_gate_can_fire_on_a_noise_padded_pool(seed: int):
    """Scored in-sample NNLS always beats the good component; cross-fitted, the noise it fitted costs it the comparison."""
    y, P, comps, names = _pool(seed)
    best_single = float(np.sqrt(np.mean((P[:, 0] - y) ** 2)))
    ens = Ens.from_nnls_stack(component_models=comps, component_names=names, component_predictions=P, y_train=y)
    in_sample = float(np.sqrt(np.mean((P @ ens.weights - y) ** 2)))
    assert in_sample < best_single, "the in-sample stack should beat its best single by construction (the old gate's blind spot)"
    assert cross_fitted_stack_rmse(Ens, "nnls_stack", comps, names, P, y) > best_single, "cross-fitted, the noise-padded stack must lose"


def test_a_capped_stack_is_refit_on_the_columns_it_kept():
    """Capping a non-convex stack to its top components refits the weights, so the blend stays on the target's level."""
    rng = np.random.default_rng(0)
    n = 400
    y = rng.normal(100.0, 10.0, n)
    P = np.column_stack([y + rng.normal(0.0, s, n) for s in (3.0, 4.0, 5.0, 6.0)])
    comps = [types.SimpleNamespace() for _ in range(4)]
    names = [f"c{i}" for i in range(4)]
    full = Ens.from_nnls_stack(component_models=comps, component_names=names, component_predictions=P, y_train=y)
    capped = full.cap_inference_components(2)
    kept = [names.index(nm) for nm in capped.component_names]
    raw_mean = float(np.mean(P[:, kept] @ np.asarray(capped.weights)))
    refit = refit_capped_stack(Ens, "nnls_stack", capped, names, P, y)
    refit_mean = float(np.mean(P[:, kept] @ np.asarray(refit.weights)))
    assert abs(raw_mean - y.mean()) / y.mean() > 0.02, "the fixture must show the raw-weight cap losing mass"
    assert abs(refit_mean - y.mean()) / y.mean() < 0.02
