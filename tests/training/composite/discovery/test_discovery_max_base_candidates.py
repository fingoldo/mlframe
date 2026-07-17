"""Unit: ``max_base_candidates`` prunes an over-long base grid before the per-pair MI screen (G3).

Pins:
* default ``None`` leaves the explicit list untouched (current behavior preserved);
* an explicit list longer than the cap is ranked by a cheap direct MI(y, x) pass (deliberately NOT
  the full ``_auto_base`` pipeline -- see ``_rank_bases_by_mi_for_cap``'s docstring for why reusing
  ``_auto_base`` for pruning is unsound) and trimmed, keeping the signal-carrying base;
* the cap also trims the ``"auto"`` path when tighter than ``auto_base_top_k``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.composite import CompositeTargetDiscovery
from mlframe.training.configs import CompositeTargetDiscoveryConfig


def _frame(n: int = 2000, seed: int = 0):
    """Frame with one dominant signal base (``b0``) and three pure-noise base candidates."""
    rng = np.random.default_rng(seed)
    b0 = rng.uniform(0.0, 1000.0, n)  # dominant signal base
    x0 = rng.normal(size=n)
    y = b0 + 30.0 * x0 + rng.normal(0.0, 1.0, n)
    df = pd.DataFrame(
        {
            "b0": b0,
            "b1": rng.normal(size=n),
            "b2": rng.normal(size=n),
            "b3": rng.normal(size=n),
            "x0": x0,
            "y": y,
        }
    )
    return df, ["b0", "b1", "b2", "b3", "x0"]


def _cfg(**kw) -> CompositeTargetDiscoveryConfig:
    """Minimal screening="mi" discovery config for the cap tests, with ``**kw`` overrides."""
    base = dict(
        enabled=True,
        random_state=0,
        screening="mi",
        honest_holdout_frac=None,
        auto_base_null_perms=0,
        multi_base_enabled=False,
        honest_rmse_gate_enabled=False,
        interaction_base_discovery_enabled=False,
        auto_chain_discovery_enabled=False,
        transforms=["linear_residual"],
        auto_base_dedup_corr_threshold=1.0,
    )
    base.update(kw)
    return CompositeTargetDiscoveryConfig(**base)


def _fit_bases(cfg, df, feats) -> set[str]:
    """Fit discovery and return the set of base columns its surviving specs use."""
    disc = CompositeTargetDiscovery(cfg)
    disc.fit(df, "y", feats, np.arange(len(df)))
    return {s.base_column for s in disc.specs_}


def test_max_base_candidates_default_none_keeps_full_explicit_list():
    """Default ``max_base_candidates=None`` must preserve the whole explicit base grid."""
    df, feats = _frame()
    bases = _fit_bases(_cfg(base_candidates=["b0", "b1", "b2", "b3"]), df, feats)
    assert bases == {"b0", "b1", "b2", "b3"}, "default (None) must preserve the whole explicit grid"


def test_max_base_candidates_prunes_explicit_list_keeping_signal_base():
    """A cap of 1 on a shuffled 4-base explicit list must keep the MI-ranked signal base, not just the first entry."""
    df, feats = _frame()
    bases = _fit_bases(_cfg(base_candidates=["b1", "b2", "b0", "b3"], max_base_candidates=1), df, feats)
    assert bases == {"b0"}, f"cap=1 must keep the MI-ranked signal base, got {bases}"


def test_max_base_candidates_trims_auto_path():
    """A cap tighter than ``auto_base_top_k`` must also trim the ``base_candidates="auto"`` path."""
    df, feats = _frame()
    bases = _fit_bases(_cfg(base_candidates="auto", auto_base_top_k=3, max_base_candidates=1), df, feats)
    assert len(bases) == 1, f"cap must trim the auto path below auto_base_top_k, got {bases}"
    assert bases == {"b0"}
