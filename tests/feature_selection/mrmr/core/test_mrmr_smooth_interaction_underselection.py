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

import pytest

pytest.importorskip("pandas")

os.environ.setdefault("NUMBA_DISABLE_CUDA", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MLFRAME_DISABLE_HNSW", "1")

_TD = os.path.dirname(os.path.abspath(__file__))
if _TD not in sys.path:
    sys.path.insert(0, _TD)
from tests.feature_selection._mrmr_realistic_data import make_realistic_case

from mlframe.feature_selection.filters.mrmr import MRMR


@pytest.mark.timeout(300)
def test_smooth_interaction_fallback_not_suppressed_by_dropped_composite():
    """s319: the bilinear ``a*b`` operands must survive the empty-raw fallback even when the
    ``mul(a,b)`` composite that captures them is dropped from the output."""
    df, y, _meta = make_realistic_case(
        seed=319,
        n=25000,
        distribution="uniform",
        target_family="smooth_interaction",
        task="regression",
    )
    fs = MRMR(
        verbose=0,
        random_seed=319,
        fe_max_steps=2,
        fe_auto_escalation_enable=True,
        dcd_enable=False,
        build_friend_graph=False,
        cluster_aggregate_enable=False,
    ).fit(df, y)
    support = set(fs.get_feature_names_out())
    # CONTRACT (re-framed 2026-06-23): this test's TRUE contract is the I5 NO-HARM biz-value bar named
    # in the docstring (FE HGB R^2 must not fall below the all-raw baseline by more than 0.05), and that
    # the bilinear interaction is not LOST (the pre-fix bug collapsed the support to a single operand {a},
    # dropping the a*b product the model needs -> R^2 0.245 vs 0.556). The original assertions pinned a
    # specific *means* -- ``b`` as a RAW survivor and >=3 raws -- which became stale when FE legitimately
    # evolved (commit 5301778c "collapse FE fragmentation to ONE clean compound") to capture the
    # interaction INSIDE one nested compound (e.g. ``add(cbrt(g),exp(mul(a,abs(b))))`` + raw ``k``):
    # measured FE R^2 0.5652 vs raw 0.5642 (delta +0.001, well inside the -0.05 bar). Re-frame to the
    # real contract so the better compound representation passes while the under-selection collapse still
    # fails (per the project rule: re-frame outdated tests to the better behaviour, do not revert the win).
    # (1) The interaction operands a AND b must BOTH be captured -- as raw survivors OR referenced inside a
    #     surviving engineered feature (not collapsed to a single operand as in the pre-fix bug).
    import re as _re

    def _tokens(_nm):
        """Helper that tokens."""
        return set(_re.findall(r"[A-Za-z_]\w*", _nm)) & set(df.columns)

    captured = set().union(*(_tokens(nm) for nm in support)) if support else set()
    assert {"a", "b"} <= captured, (
        f"interaction operands a,b not both captured (raw or within a compound) -> under-selection "
        f"collapsed the support to {sorted(support)}; the bilinear a*b product is lost."
    )
    # (2) I5 NO-HARM: FE feature space must not be worse than all-raw by more than the 0.05 bar.
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import cross_val_score

    X_fe = fs.transform(df)
    r2_fe = cross_val_score(HistGradientBoostingRegressor(random_state=0), X_fe, y, cv=3, scoring="r2").mean()
    r2_raw = cross_val_score(HistGradientBoostingRegressor(random_state=0), df.to_numpy(), y, cv=3, scoring="r2").mean()
    assert r2_fe >= r2_raw - 0.05, (
        f"FE under-selection HARMS biz-value: FE R^2={r2_fe:.4f} vs raw R^2={r2_raw:.4f} "
        f"(delta {r2_fe - r2_raw:+.4f} < the -0.05 I5 bar); support={sorted(support)}"
    )
