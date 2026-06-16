"""Regression: empty-raw-screen fallback must not be suppressed by a DROPPED engineered feature.

When the greedy screen returns 0 surviving raw features, MRMR rescues the raw feature(s)
clearing the relevance floor, deduping them against the SURVIVING engineered features (so a
raw fully captured by a composite that DID reach the output is not re-injected). The dedup
must condition only on engineered features that ACTUALLY survive to output
(``_engineered_recipes_``), NOT on every greedily-selected engineered name
(``_engineered_features_``) -- the latter still includes composites that were selected but
then DROPPED (recipeless nested parents, or ones that failed the MI-prevalence gate).

The s319 failure (``y = 1.5*a*b + 0.5*g/k``, uniform, n=25000): the interaction composite
``mul(a,b)`` was formed but prevalence-gated out of the output, yet it still suppressed raw
``b`` in the fallback dedup -- so the rescue collapsed to a single raw ``a`` and the FE space
(HGB R^2 0.245) fell far below the all-raw baseline (0.556), delta -0.311 < the -0.05 I5 bar.
PRE-FIX support = {a}; POST-FIX support = {a,b,g,k} (delta +0.0005).

This drives a real MRMR fit on the generator's smooth-interaction case (the bug lives in the
post-greedy fallback, so it needs the full fit) and asserts the support is not collapsed to a
single raw and retains the interaction operand ``b``.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

pytest.importorskip("pandas")

os.environ.setdefault("NUMBA_DISABLE_CUDA", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MLFRAME_DISABLE_HNSW", "1")

_TD = os.path.dirname(os.path.abspath(__file__))
if _TD not in sys.path:
    sys.path.insert(0, _TD)
from _mrmr_realistic_data import make_realistic_case  # noqa: E402

from mlframe.feature_selection.filters.mrmr import MRMR  # noqa: E402


@pytest.mark.timeout(300)
def test_smooth_interaction_fallback_not_suppressed_by_dropped_composite():
    """s319: the bilinear ``a*b`` operands must survive the empty-raw fallback even when the
    ``mul(a,b)`` composite that captures them is dropped from the output."""
    df, y, meta = make_realistic_case(
        seed=319, n=25000, distribution="uniform",
        target_family="smooth_interaction", task="regression",
    )
    fs = MRMR(
        verbose=0, random_seed=319, fe_max_steps=2, fe_auto_escalation_enable=True,
        dcd_enable=False, build_friend_graph=False, cluster_aggregate_enable=False,
    ).fit(df, y)
    support = set(fs.get_feature_names_out())
    raw_support = support & set(df.columns)
    # The target is 1.5*a*b + 0.5*g/k: 'a' and 'b' BOTH carry the interaction; collapsing to a
    # single raw (pre-fix {'a'}) loses the product the HGB needs. Require the interaction
    # operand 'b' is kept and the support is not degenerate.
    assert "b" in raw_support, (
        f"interaction operand 'b' dropped -> under-selection collapsed the support to "
        f"{sorted(support)}; the dropped mul(a,b) composite wrongly suppressed it in the fallback."
    )
    assert len(raw_support) >= 3, (
        f"support collapsed to {sorted(raw_support)} raws; expected the interaction + ratio "
        f"operands (a,b,g,k) to survive the empty-raw rescue."
    )
