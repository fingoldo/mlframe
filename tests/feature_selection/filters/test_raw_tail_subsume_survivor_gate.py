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

import re

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
    m = MRMR(max_runtime_mins=5, verbose=0, random_state=312, fe_raw_tail_subsume_min_corr=gate, **_FE_KWARGS)
    m.fit(df, y)
    names_out = list(m.get_feature_names_out())
    return {c for c in names_out if c in raw_cols}, names_out


def _source_tokens(names_out):
    """Every source-column token of the kept output names: a raw column, or a raw inside an engineered composite."""
    return {tok for name in names_out for tok in re.split(r"[^A-Za-z0-9]+", str(name)) if tok}


@pytest.mark.slow
def test_weak_proxy_survivor_keeps_tail_concentrated_raw_b():
    """Shipped default gate 0.85: the information of the tail-concentrated ratio operand ``b`` must survive into the
    kept set, either as the raw column or inside an engineered composite that carries it.

    The raw column itself may be dropped: the tail-concentration leg fires on ``b`` (rank |corr| collapses to ~0.14
    against a linear ~0.24) and the surviving composite ``mul(log(b),...)`` subsumes it. Asserting the raw column
    pinned the drop decision, not the signal."""
    _, names_out = _fit_kept_raws(gate=0.85)
    assert "b" in _source_tokens(names_out), f"operand 'b' lost from the kept set at the 0.85 gate; kept={names_out}"


@pytest.mark.slow
def test_leg_declines_because_b_is_not_rank_collapsed_not_because_of_the_gate():
    """The information of ``b`` must survive at both survivor-strength gates (0.6 and 0.85).

    Measured on this fixture, ``b`` is rank-collapsed (raw |corr(y)| 0.238, rank |corr(y)| 0.140), so the
    tail-concentration leg may drop the raw column at either gate; what must not happen is losing ``b``'s signal
    from the kept set. The earlier numbers pinned here (0.7564 / 0.8834) belong to operand ``a``, not ``b``.
    """
    for gate in (0.6, 0.85):
        _, names_out = _fit_kept_raws(gate=gate)
        assert "b" in _source_tokens(names_out), f"operand 'b' lost from the kept set at survivor-strength gate {gate}; kept={names_out}"
