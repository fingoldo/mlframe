"""After the group-aware final demotion drops engineered features, the public FE rosters must not keep naming them.

Roster reconciliation ran before the demotion, the last mutation of ``_engineered_features_``, so a demoted column vanished from
``get_feature_names_out()`` but stayed in e.g. ``hybrid_orth_features_``, which is documented as the engineered columns that survived.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._mrmr_fit_impl._fe_roster_attrs import FE_ROSTER_ATTRS
from mlframe.feature_selection.filters.mrmr import MRMR


def _panel(seed: int = 1, G: int = 60, per: int = 50):
    """A product-interaction target so FE keeps engineered survivors, with a group structure for group_aware_mi."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(G), per)
    n = groups.size
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    y = a * b + 0.05 * rng.normal(size=n)
    return pd.DataFrame({"a": a, "b": b, "c": rng.normal(size=n)}), y, groups


def _fit(X, y, groups):
    """Group-aware fit with FE on and the fit cache off, so a monkeypatch between fits takes effect."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MRMR(max_runtime_mins=2, verbose=0, fit_cache_max=0, fe_max_steps=1, group_aware_mi=True)
        m.fit(X, y, groups=groups)
    return m


def _roster_names(m) -> dict:
    """Every non-empty FE roster on the fitted estimator."""
    return {a: list(getattr(m, a, None) or []) for a in FE_ROSTER_ATTRS if getattr(m, a, None)}


def test_group_demotion_prunes_public_rosters(monkeypatch):
    """With every within-group relevance forced to zero, every engineered survivor is demoted, and no roster may still list one."""
    X, y, groups = _panel()
    control = _fit(X, y, groups)
    control_eng = {r.name for r in (control._engineered_recipes_ or [])}
    assert control_eng, "fixture precondition: the control fit must keep engineered features"
    assert any(set(v) & control_eng for v in _roster_names(control).values()), "fixture precondition: a roster must list a survivor"

    monkeypatch.setattr("mlframe.feature_selection.filters.info_theory._group_mi.group_relevance_mi", lambda *a, **kw: 0.0)
    m = _fit(X, y, groups)
    out = set(map(str, m.get_feature_names_out()))
    stale = {a: [c for c in v if c not in out] for a, v in _roster_names(m).items()}
    stale = {a: v for a, v in stale.items() if v}
    assert not stale, f"rosters still list demoted engineered columns absent from the output: {stale}"
