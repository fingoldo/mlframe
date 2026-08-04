"""Regression test for a default_via_or trap in ``apply_synergy_bootstrap``: ``fe_synergy_max_sweep_cost``
resolution used to read ``float(getattr(self, "fe_synergy_max_sweep_cost", 5e8) or float("inf"))``, which
silently rewrote an explicit, legitimate ``0.0`` (meaning "disable the synergy sweep -- any real n*p^2
cost exceeds a 0 budget") into ``float("inf")`` (meaning "unlimited budget, always allow") -- the exact
opposite of the configured intent.
"""
from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._mrmr_fe_step_helpers import apply_synergy_bootstrap


class _FakeMRMR:
    """Minimal stand-in exposing only the attributes ``apply_synergy_bootstrap`` reads."""

    fe_synergy_screen_max_features = 1
    fe_synergy_max_sweep_cost = 0.0
    fe_synergy_exhaustive = "never"  # force the pre-rank/cost-gate path regardless of local CUDA availability
    factors_to_use = None
    factors_names_to_use = None
    feature_names_in_ = np.array(["a", "b", "c"])
    _fe_synergy_exhaustive_active_ = False


def test_synergy_max_sweep_cost_zero_disables_sweep_not_rewritten_to_unlimited():
    """A 0.0 cost budget must skip the synergy sweep entirely (no columns bootstrap-added), not be
    silently treated as an unlimited budget that lets the sweep run anyway."""
    self = _FakeMRMR()
    n = 500
    data = np.random.RandomState(0).rand(n, 3)
    cols = ["a", "b", "c"]

    _pool, added_idx = apply_synergy_bootstrap(
        self,
        num_fs_steps=0,
        data=data,
        cols=cols,
        target_indices=[2],
        categorical_vars=set(),
        # Empty (not {0}) -- with the bug present, the pre-rank path (wrongly reached because cost
        # was rewritten to inf) can coincidentally keep the SAME column already excluded here and
        # mask the bug; an empty set means ANY kept column shows up as "added".
        numeric_vars_to_consider=set(),
        non_numeric_idx=set(),
        verbose=False,
    )

    assert added_idx == set(), (
        "fe_synergy_max_sweep_cost=0.0 must disable the synergy bootstrap sweep (n*p^2 cost always "
        f"exceeds a 0 budget); got synergy_added_idx={added_idx!r}, indicating the sweep ran anyway."
    )


def test_synergy_max_sweep_cost_unset_still_uses_documented_default():
    """When the attribute is genuinely absent, the documented 5e8 default must still apply (not inf)."""

    class _FakeMRMRUnset:
        """Like ``_FakeMRMR`` but with ``fe_synergy_max_sweep_cost`` genuinely absent."""

        fe_synergy_screen_max_features = 1
        fe_synergy_exhaustive = "never"
        factors_to_use = None
        factors_names_to_use = None
        feature_names_in_ = np.array(["a", "b", "c"])
        _fe_synergy_exhaustive_active_ = False

    unset_self = _FakeMRMRUnset()
    n = 500
    data = np.random.RandomState(0).rand(n, 3)
    cols = ["a", "b", "c"]

    # cap=1 < n_raw=2 -> triggers the pre-rank path (not the cost-gate-first skip), since n*cap^2 =
    # 500*1 = 500 is well under the 5e8 default -- proving the default is neither 0 nor inf.
    _pool, added_idx = apply_synergy_bootstrap(
        unset_self,
        num_fs_steps=0,
        data=data,
        cols=cols,
        target_indices=[2],
        categorical_vars=set(),
        numeric_vars_to_consider=set(),
        non_numeric_idx=set(),
        verbose=False,
    )
    assert added_idx, "with the documented 5e8 default, this tiny sweep must fit under budget and add a column"
