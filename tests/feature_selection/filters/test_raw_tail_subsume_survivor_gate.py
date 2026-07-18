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
def test_pre_fix_gate_drops_raw_b_regression_sensor():
    """Pre-fix gate 0.6 DID drop ``b`` (the confirmed regression). Pins that the sensor above catches the
    bug: if a future change makes the tail leg fire regardless of the gate, this flips and fails first."""
    kept = _fit_kept_raws(gate=0.6)
    assert "b" not in kept, (
        "pre-fix 0.6 gate no longer drops 'b'; the tail-subsume leg must fire at 0.6 on this case for the "
        f"0.85 gate to be the load-bearing fix (kept_raws={sorted(kept)})"
    )
