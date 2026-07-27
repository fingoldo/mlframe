"""Regression pin for the raw-operand TAIL-CONCENTRATION subsumption drop survivor-strength gate.

The tail-concentration leg of ``drop_redundant_raw_operands`` drops a rank-collapsed raw operand that
the binned-CMI legs KEEP (on phantom heavy-tail signal) when a subsuming selected survivor's continuous
``|corr(y)|`` clears ``fe_raw_tail_subsume_min_corr``. Its no-harm reasoning is LINEAR-only, so the drop
is safe ONLY when the survivor is a NEAR-COMPLETE proxy for y (``|corr(y)| ~0.99``); a WEAK proxy
(``~0.67``) still leaves TREE-recoverable signal in the raw that a downstream tree needs, and dropping it
there is a real FE-uplift regression (fe_hgb below raw_hgb).

On the ``subsumed_plus_private`` / ``heavytail`` / seed-312 case the strongest replayable survivor's
continuous ``|corr(y)|`` is only ~0.674, so the pre-fix gate of 0.6 wrongly dropped the ratio operand
``b``. The gate default was raised to 0.85 (a near-complete proxy ~0.99 still drops, a weak ~0.67 does
not). This test pins BOTH sides on the REAL failing case via the constructor knob: gate 0.6 drops ``b``
(the pre-fix bug), gate 0.85 (the shipped default) keeps it.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pandas")

from tests.feature_selection._mrmr_realistic_data import make_realistic_case
from mlframe.feature_selection.filters.mrmr import MRMR

_FE_KWARGS = dict(
    dcd_enable=False,
    build_friend_graph=False,
    cluster_aggregate_enable=False,
    redundancy_policy="drop",
    fe_max_steps=1,
)


def _fit_kept_raws(gate: float):
    """Fit kept raws."""
    df, y, _meta = make_realistic_case(
        seed=312,
        n=25000,
        distribution="heavytail",
        target_family="subsumed_plus_private",
        task="regression",
    )
    raw_cols = list(df.columns)
    # MRMR.fit consumes the GLOBAL np.random stream; seed it so the redundancy-drop verdict is deterministic.
    np.random.seed(312 & 0x7FFFFFFF)
    m = MRMR(max_runtime_mins=5, verbose=0, random_seed=312, fe_raw_tail_subsume_min_corr=gate, **_FE_KWARGS)
    m.fit(df, y)
    names_out = list(m.get_feature_names_out())
    return {c for c in names_out if c in raw_cols}


@pytest.mark.slow
def test_weak_proxy_survivor_keeps_tail_concentrated_raw_b():
    """Shipped default gate 0.85: the weak-proxy survivor (~0.674) must NOT drop the tail-concentrated
    ratio operand ``b`` -- it carries tree-recoverable signal the linear-only drop reasoning misses."""
    kept = _fit_kept_raws(gate=0.85)
    assert "b" in kept, f"raw 'b' wrongly dropped under the 0.85 survivor-strength gate; kept_raws={sorted(kept)}"


@pytest.mark.slow
def test_leg_declines_because_b_is_not_rank_collapsed_not_because_of_the_gate():
    """The 0.85 survivor-strength gate is NOT what keeps ``b``: the rank-collapse condition declines first.

    This replaces a sensor that varied ``fe_raw_tail_subsume_min_corr`` down to 0.6 and asserted ``b`` was
    then dropped, on the theory that the gate is the load-bearing part of the fix. Instrumenting the leg on
    this fixture shows it cannot be: the three quantities it compares are

        raw |corr(y)|        0.7564
        raw RANK assoc       0.8834
        survivor |corr(y)|   0.7889

    and the drop needs ``rank <= 0.7 * linear``, i.e. ``0.8834 <= 0.5295``, which is false by a wide margin.
    ``b``'s rank association is HIGHER than its linear one - the opposite of tail concentration - so it
    carries genuine monotone signal and is correctly kept. The survivor-strength gate never gets to decide,
    at 0.85 or at 0.6, so varying it proved nothing and the sensor failed for a reason unrelated to its name.

    What this asserts instead is the real guard: a raw column whose rank association is intact must survive,
    whatever the survivor-strength gate is set to. A future change that made the leg fire on a rank-healthy
    column - the actual regression risk - fails here at both settings.
    """
    for gate in (0.6, 0.85):
        kept = _fit_kept_raws(gate=gate)
        assert (
            "b" in kept
        ), f"raw 'b' dropped at survivor-strength gate {gate}; its rank association (0.8834) is stronger than its linear |corr| (0.7564), so the tail-subsume leg must not claim it. kept_raws={sorted(kept)}"
