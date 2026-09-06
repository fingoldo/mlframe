"""Every FE roster must exist after ANY successful fit, including the multioutput one.

Tests across this suite assert that a family stayed silent by reading ``m.<family>_features_``. Reading it
through a ``getattr(..., [])`` default would make a RENAMED roster indistinguishable from an empty one, so
those reads are direct -- which is only sound if the attribute is genuinely always present. It was not: the
multioutput path fits a clone per target column and returns before the single-target body that seeds them,
so the outer estimator came back from a successful fit with none of these attributes at all.

Two tests here. One pins the contract on a real 2D fit. The other keeps the canonical list honest: a family
added to a cascade without being listed would reintroduce the same gap silently.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._mrmr_fit_impl._fe_roster_attrs import FE_ROSTER_ATTRS

_FIT_IMPL_DIR = pathlib.Path(__file__).resolve().parents[2] / "src" / "mlframe" / "feature_selection" / "filters" / "_mrmr_fit_impl"


def _seeded_roster_names() -> set:
    """Every ``self.<name>_features_ = []`` the cascade modules actually perform."""
    found = set()
    for path in sorted(_FIT_IMPL_DIR.glob("_fe_stage_cascade_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.List) or node.value.elts:
                continue  # only `= []` seeds, not appends or real values
            for target in node.targets:
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self" and target.attr.endswith("_features_"):
                    found.add(target.attr)
    return found


def test_the_canonical_list_matches_what_the_cascades_seed():
    """A family added to a cascade but missing here would silently lose its roster on the multioutput path."""
    seeded = _seeded_roster_names()
    assert seeded, "no roster seeds found; this test's AST scan needs updating, not the source"
    missing = seeded - set(FE_ROSTER_ATTRS)
    assert not missing, f"cascades seed rosters that FE_ROSTER_ATTRS does not list: {sorted(missing)}"


@pytest.mark.parametrize("kind", ["multilabel", "multitarget"])
def test_every_roster_is_present_after_a_multioutput_fit(kind):
    """The failing case: a 2D target, where the engineering happens inside per-target clones."""
    from mlframe.feature_selection.filters import MRMR

    rng = np.random.default_rng(0)
    n, p = 300, 5
    X = pd.DataFrame({f"f{i}": rng.standard_normal(n) for i in range(p)})
    y = (rng.random((n, 3)) < 0.3).astype(int) if kind == "multilabel" else rng.standard_normal((n, 3))

    m = MRMR(verbose=0, fe_max_steps=1, dcd_enable=False, cluster_aggregate_enable=False, build_friend_graph=False, cat_fe_config=None, random_seed=0)
    m.fit(X, y)

    missing = [name for name in FE_ROSTER_ATTRS if not hasattr(m, name)]
    assert not missing, f"{kind}: fit returned without these rosters, so any test reading one gets AttributeError: {missing}"
    for name in FE_ROSTER_ATTRS:
        assert list(getattr(m, name)) == [], f"{kind}: {name} is non-empty on the outer estimator, but FE ran only inside the per-target clones"
