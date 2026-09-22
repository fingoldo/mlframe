"""A group key never seen at fit time gets the global answer, and the global answer is the pooled-best one.

The MoE gate routed every row of an unseen group to ``lag_predict`` (RMSE 8.00 against the deployed ensemble's 0.98 on a
group-disjoint split), and the grouped recurrent transforms seeded unseen groups with the mean instead of the ungrouped
continuation. The unseen-group tests that existed asserted only that predictions were finite. Here the fallback is pinned to
an exact value (the global parameters, the global estimator, the global prior) or, for the gate, to the pooled-best
expert measured on the same rows.
"""

from __future__ import annotations

import ast
import re
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import mlframe
from mlframe.training.composite.transforms import TRANSFORMS_REGISTRY

_TRAINING = Path(mlframe.__file__).resolve().parent / "training"
_FALLBACK_ATTR = re.compile(r"^(global_choice_|_global_idx|global_estimator_|_global_prior)$")

# Every class that keeps a fitted-key table with a global fallback, and the test below that pins its unseen-key answer.
UNSEEN_KEY_ROUTERS = {
    "MoESelectionGate": "test_the_moe_gate_serves_unseen_groups_the_pooled_best_expert",
    "PerGroupCompositeRouter": "test_the_per_group_router_serves_unseen_groups_the_global_estimator",
    "LeakageSafeEncoder": "test_an_unseen_category_encodes_to_the_global_prior",
}
# The OOD and volatility lag routers route on the base's range and on local volatility, not on a fitted key table, so
# they have no unseen-key case.

_GROUPED = sorted(n for n, t in TRANSFORMS_REGISTRY.items() if t.requires_groups)
# The unseen-group inverse each parametric grouped transform must reproduce from its global parameters.
_GLOBAL_INVERSE = {
    "linear_residual_grouped": lambda t, b, p: t + p["alpha_global"] * b + p["beta_global"],
    "monotonic_residual_grouped": lambda t, b, p: TRANSFORMS_REGISTRY["monotonic_residual"].inverse(t, b, p["global"]),
    "quantile_residual_grouped": lambda t, b, p: TRANSFORMS_REGISTRY["quantile_residual"].inverse(t, b, p["global"]),
    "target_encoding_residual": lambda t, b, p: t + p["global_mean"],
}


def test_every_unseen_key_router_is_registered():
    """A class storing a global fallback for fitted keys must name the test that pins its unseen-key answer."""
    found = set()
    for path in _TRAINING.rglob("*.py"):
        if "_benchmarks" in path.parts:
            continue
        for cls in (n for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))) if isinstance(n, ast.ClassDef)):
            if any(isinstance(a, ast.Attribute) and isinstance(a.ctx, ast.Store) and isinstance(a.value, ast.Name) and a.value.id == "self"
                   and _FALLBACK_ATTR.match(a.attr) for a in ast.walk(cls)):
                found.add(cls.name)
    assert found == set(UNSEEN_KEY_ROUTERS), f"unregistered: {sorted(found - set(UNSEEN_KEY_ROUTERS))}; stale: {sorted(set(UNSEEN_KEY_ROUTERS) - found)}"
    assert all(name in globals() for name in UNSEEN_KEY_ROUTERS.values())


def _grouped_data(n: int = 600, seed: int = 0):
    """``y``, base and 30 group labels, the group adding a small level shift."""
    rng = np.random.default_rng(seed)
    g = np.arange(n) % 30
    b = rng.uniform(1.0, 10.0, n)
    return 2.0 * b + 0.1 * g + rng.normal(0.0, 0.3, n), b, g


@pytest.mark.parametrize("name", _GROUPED)
def test_an_unseen_group_does_not_depend_on_its_label(name: str):
    """Two different unseen labels give identical inverses: the fallback reads no per-group state."""
    t = TRANSFORMS_REGISTRY[name]
    y, b, g = _grouped_data()
    rng = np.random.default_rng(1)
    bn, tn = rng.uniform(1.0, 10.0, 40), rng.normal(0.0, 0.3, 40)
    base = bn if t.requires_base else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = t.fit(y, b if t.requires_base else None, groups=g)
        a = t.inverse(tn, base, p, groups=np.full(40, 100))
        c = t.inverse(tn, base, p, groups=np.full(40, 777))
    np.testing.assert_allclose(a, c, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("name", sorted(_GLOBAL_INVERSE))
def test_an_unseen_group_uses_the_global_parameters(name: str):
    """Unseen-group rows equal the inverse computed from the stored global parameters, to 1e-12."""
    t = TRANSFORMS_REGISTRY[name]
    y, b, g = _grouped_data()
    rng = np.random.default_rng(2)
    bn, tn = rng.uniform(1.0, 10.0, 40), rng.normal(0.0, 0.3, 40)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = t.fit(y, b if t.requires_base else None, groups=g)
        got = t.inverse(tn, bn if t.requires_base else None, p, groups=np.full(40, 100))
    np.testing.assert_allclose(got, _GLOBAL_INVERSE[name](tn, bn, p), rtol=0, atol=1e-12)


def test_the_param_table_covers_every_parametric_grouped_transform():
    """Each grouped transform either has a global-parameter identity or is recurrent (seeded, not parametric)."""
    recurrent = {"ewma_residual_grouped", "frac_diff_grouped", "rolling_quantile_ratio_grouped"}
    assert set(_GROUPED) == set(_GLOBAL_INVERSE) | recurrent, sorted(set(_GROUPED) ^ (set(_GLOBAL_INVERSE) | recurrent))


def test_the_moe_gate_serves_unseen_groups_the_pooled_best_expert():
    """Fit on groups 0-29, serve groups 100-129: the gate's RMSE there is within 2% of the pooled-best expert's on the same rows."""
    from mlframe.training.composite._moe_gate import MoESelectionGate

    rng = np.random.default_rng(0)

    def draw(groups):
        y = rng.normal(0.0, 5.0, groups.size)
        return y, {"composite": y + rng.normal(0.0, 1.0, y.size), "raw": y + rng.normal(0.0, 2.0, y.size), "lag": y + rng.normal(0.0, 8.0, y.size)}

    g_fit = np.repeat(np.arange(30), 100)
    y_fit, e_fit = draw(g_fit)
    gate = MoESelectionGate(failsafe="lag").fit(y_fit, e_fit, group_ids=g_fit)
    g_new = np.repeat(np.arange(100, 130), 100)
    y_new, e_new = draw(g_new)
    served = gate.predict(e_new, group_ids=g_new)
    rmse = lambda p: float(np.sqrt(np.mean((p - y_new) ** 2)))
    assert rmse(served) <= 1.02 * min(rmse(p) for p in e_new.values()), (rmse(served), {k: rmse(p) for k, p in e_new.items()})


def test_the_per_group_router_serves_unseen_groups_the_global_estimator():
    """Rows of a group with no submodel get exactly the global estimator's prediction."""
    from sklearn.linear_model import LinearRegression

    from mlframe.training.composite.per_group_router import PerGroupCompositeRouter

    rng = np.random.default_rng(0)
    n = 600
    X = pd.DataFrame({"b": rng.uniform(1.0, 10.0, n), "x": rng.normal(size=n), "grp": np.arange(n) % 3})
    y = 2.0 * X["b"] + 0.5 * X["x"] + X["grp"] + rng.normal(0.0, 0.1, n)
    spec = types.SimpleNamespace(transform_name="linear_residual", base_column="b")
    disc = types.SimpleNamespace(specs_=[spec], specs_by_group_={0: [spec], 1: [spec], 2: [spec]})
    router = PerGroupCompositeRouter(discovery=disc, base_estimator=LinearRegression(), group_column="grp").fit(X, y)
    X_new = X.iloc[:50].assign(grp=99)
    np.testing.assert_allclose(router.predict(X_new), router.global_estimator_.predict(X_new.drop(columns=["grp"])), rtol=0, atol=1e-12)


def test_an_unseen_category_encodes_to_the_global_prior():
    """A category absent at fit time encodes to the fitted global prior, for every encoding method."""
    from mlframe.training.feature_handling.target_encoders import LeakageSafeEncoder

    rng = np.random.default_rng(0)
    cats = rng.choice(["a", "b", "c", "d"], 400)
    y = (rng.random(400) < np.where(cats == "a", 0.8, 0.3)).astype(float)
    for method in ("target_mean", "target_m_estimate", "target_james_stein"):
        enc = LeakageSafeEncoder(method=method)
        enc.fit_transform(cats, y)
        out = enc.transform(np.array(["zzz", "never"]))
        np.testing.assert_allclose(out, enc._global_prior, rtol=0, atol=1e-12, err_msg=method)
