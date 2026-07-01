"""biz_value DELTA for ``warp_linear_margin`` -- the monotone-warp tie-break margin.

When two mutually-redundant candidates are strictly-monotone twins (f and g=exp(4 f),
rank-identical so binned MI / SU tie), DCD keeps exactly one. With
``warp_tiebreak_prefer_linear=True`` it biases that forced choice toward the more
linearly-usable leg -- but ONLY when the candidate's linear-usability advantage over the
anchor strictly exceeds ``warp_linear_margin`` (``_lin_c - _lin_a > margin``).

The sibling test ``test_biz_val_monotone_warp_and_ts_leak.py`` toggles the master flag
on/off but never exercises the MARGIN value itself. This file pins the margin's
decision-influencing contract: on the SAME twin pair, a LOW margin lets the linear-usable
raw f DISPLACE its exp-warp anchor g (the linear downstream recovers the signal), while a
HIGH margin (above the achievable usability gap) SUPPRESSES the displacement so the
order-decided g survives. The measured usability gap on this fixture is ~0.84
(lin(f)~0.98 vs lin(exp(4f))~0.14); floors set with margin around that.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _binq(x, nb: int = 10):
    q = np.quantile(x, np.linspace(0, 1, nb + 1))
    q[0] -= 1e-9
    return np.clip(np.searchsorted(q, x) - 1, 0, nb - 1)


def _setup(margin: float, seed: int = 0, n: int = 3000):
    """Twin pair g=exp(4 f) with g (idx 0) the already-selected anchor and f (idx 1) the
    candidate. Returns (state, selected_vars) after running the cluster-member discovery
    that contains the warp tie-break."""
    from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
        make_dcd_state, discover_cluster_members,
    )
    rng = np.random.default_rng(seed)
    f = rng.standard_normal(n)
    g = np.exp(4.0 * f)
    fac = np.column_stack([_binq(g), _binq(f)]).astype(np.int64)  # cols ['g','f']
    nbins = np.array([fac[:, 0].max() + 1, fac[:, 1].max() + 1], dtype=np.int64)
    Xraw = pd.DataFrame({"g": g, "f": f})
    state = make_dcd_state(
        X_raw=Xraw, factors_data=fac, cols=["g", "f"], nbins=nbins,
        factors_nbins=nbins, target_indices=np.array([], dtype=np.int64),
        warp_tiebreak_prefer_linear=True, tau_cluster=0.7,
        warp_linear_margin=margin,
    )
    sv = [0]  # g (idx 0) is the order-decided anchor
    discover_cluster_members(state, 0, [1], factors_data=fac,
                             factors_nbins=nbins, selected_vars=sv)
    return state, sv


def test_biz_val_warp_linear_margin_low_displaces_to_linear_f():
    """LOW margin (0.0) < the ~0.84 usability gap -> the linear-usable raw f DISPLACES the
    exp-warp anchor g. A downstream linear model recovers f's signal that g obscures."""
    state, sv = _setup(margin=0.0)
    assert sv == [1], (
        f"low warp_linear_margin must let f displace the exp-warp anchor g; selected_vars={sv}"
    )
    assert bool(state.pool_pruned_mask[0]) and not bool(state.pool_pruned_mask[1]), (
        f"g pruned, f kept; mask={state.pool_pruned_mask.tolist()}"
    )


def test_biz_val_warp_linear_margin_high_suppresses_displacement():
    """HIGH margin (0.99) > the achievable usability gap (~0.84) -> displacement is
    suppressed and the order-decided anchor g survives (legacy column-order tie-break)."""
    state, sv = _setup(margin=0.99)
    assert sv == [0], (
        f"high warp_linear_margin must suppress the displacement (gap < margin); selected_vars={sv}"
    )
    assert not bool(state.pool_pruned_mask[0]) and bool(state.pool_pruned_mask[1]), (
        f"g kept, f pruned; mask={state.pool_pruned_mask.tolist()}"
    )


def test_biz_val_warp_linear_margin_delta_flips_survivor():
    """The DELTA: holding the twin pair + master flag fixed, only changing the margin flips
    which leg survives -- proves the knob is decision-influencing on its own."""
    _, sv_low = _setup(margin=0.0)
    _, sv_high = _setup(margin=0.99)
    assert sv_low == [1] and sv_high == [0], (
        f"margin alone must flip the survivor: low->f (sv={sv_low}), high->g (sv={sv_high})"
    )
