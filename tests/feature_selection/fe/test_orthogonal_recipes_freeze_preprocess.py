"""Every orthogonal-FE ``hybrid_*_with_recipes`` entry point freezes the fit-time basis preprocessing in its recipes (B-17, X_TEST_COVERAGE_QUALITY-2).

A recipe replayed on a small slice with a different mean and spread must give the fit-time value of those rows: if the recipe only held the basis name, the slice's own
mean and std would be refit at replay time and the engineered value would drift. The jmim module has its own test; this one runs the same check over the sibling modules.
"""

from __future__ import annotations

import importlib
import inspect
import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.engineered_recipes import apply_recipe

warnings.filterwarnings("ignore")

#: Modules whose recipes the generic data below produces; cluster_basis and diff_basis emit none on it, meta returns a dict, tail_dependence and xi have no recipe entry point.
_MODULES = [
    "_orthogonal_adaptive_degree_fe",
    "_orthogonal_bootstrap_mi_fe",
    "_orthogonal_cmim_fe",
    "_orthogonal_copula_mi_fe",
    "_orthogonal_dcor_fe",
    "_orthogonal_elasticnet_fe",
    "_orthogonal_hsic_fe",
    "_orthogonal_ksg_mi_fe",
    "_orthogonal_lasso_fe",
    "_orthogonal_quadruplet_fe",
    "_orthogonal_routing_fe",
    "_orthogonal_scorer_auto_fe",
    "_orthogonal_three_gate_mi_fe",
    "_orthogonal_total_correlation_fe",
    "_orthogonal_triplet_fe",
    "_orth_auto_scorer_fe",
]


def _data(n: int = 2000):
    """A frame whose first column has mean 10 and std 3, so a head slice differs materially from the whole, and a quadratic target."""
    rng = np.random.default_rng(0)
    x1 = rng.standard_normal(n) * 3 + 10
    x2 = rng.standard_normal(n)
    X = pd.DataFrame({"x1": x1, "x2": x2, "noise_0": rng.standard_normal(n), "noise_1": rng.standard_normal(n)})
    signal = (x1 - 10.0) ** 2 + 0.6 * x2**2
    y = ((signal + 0.05 * rng.standard_normal(n)) > np.median(signal)).astype(int)
    return X, y


@pytest.mark.parametrize("module_name", _MODULES)
def test_recipes_freeze_the_preprocess_params_and_replay_on_a_slice(module_name):
    """Each emitted recipe carries its frozen params and replays on the first 50 rows to the fit-time values."""
    mod = importlib.import_module(f"mlframe.feature_selection.filters.{module_name}")
    fn = getattr(mod, next(n for n in dir(mod) if n.startswith("hybrid_") and n.endswith("_with_recipes")))
    X, y = _data()
    params = inspect.signature(fn).parameters
    wanted = dict(cols=["x1", "x2"], degrees=(2,), basis="hermite", top_k=2, min_uplift=0.0, min_abs_mi_frac=0.0, nbins=10, n_bins=10, current_support=X[["x1"]])
    out = fn(X, y, **{k: v for k, v in wanted.items() if k in params})
    x_aug, recipes = out[0], out[-1]
    assert len(recipes) > 0, f"{module_name} emitted no recipe on the planted quadratic, so the slice replay was not exercised"
    for rec in recipes:
        frozen = [v for k, v in rec.extra.items() if str(k).startswith("preprocess_params") and v is not None]
        assert frozen, f"{module_name}: recipe {rec.name!r} holds no frozen preprocess params, so a slice replay would refit them"
        fit_time = x_aug[rec.name].to_numpy()[:50]
        replayed = np.asarray(apply_recipe(rec, X.iloc[:50].copy()), dtype=float)
        tol = 1e-5 * max(1.0, float(np.max(np.abs(fit_time))))
        assert np.allclose(replayed, fit_time, rtol=1e-5, atol=tol, equal_nan=True), f"{module_name}: recipe {rec.name!r} slice replay diverged from the fit-time values"
