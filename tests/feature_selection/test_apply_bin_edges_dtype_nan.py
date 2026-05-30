"""Wave 9.1 loop-iter-47 regression: ``apply_bin_edges`` MUST handle
NaN inputs and auto-pick an adequate dtype.

Pre-fix at ``supervised_binning.py:367-373``:

1. Default ``dtype=np.int8`` silently overflowed once
   ``len(edges) >= 128``. Multi-quantile / cross-feature joint binning
   routinely produces >128 edges, returning monotonic-looking but
   wrong-class codes after the int8 wrap (e.g. codes
   [0, 42, 85, -128, -86, -43, 0, 42] for what should be a monotonic
   sequence ending near 300).

2. NaN inputs got silently aliased with the highest finite bin via
   ``np.searchsorted(edges, NaN, side='right')`` -> ``len(edges)``.
   NaN rows silently merged with the largest valid bin. Downstream
   MI / WoE / MRMR computations consumed NaN rows as if they were the
   top quantile.

Severity: P0/P1 silent-corruption. Affects any supervised-binning
caller (MDLP / optimal_joint paths) under realistic bin counts AND
any caller passing NaN-bearing data.

Fix at supervised_binning.py:364:
- ``dtype`` defaults to None and auto-picks based on ``n_codes``
  (int8 if < 127, int16 if < 32767, int32 otherwise).
- Caller-forced narrow dtype now raises ``ValueError`` instead of
  silently wrapping.
- NaN inputs get a dedicated sentinel at ``n_codes`` (one past max
  real bin).
"""
from __future__ import annotations

import numpy as np
import pytest


def test_default_dtype_handles_large_edge_count():
    """300-edge input must not silently wrap on default dtype."""
    from mlframe.feature_selection.filters.supervised_binning import apply_bin_edges
    edges = np.linspace(-10, 10, 300)
    x = np.linspace(-10, 10, 8)
    out = apply_bin_edges(x, edges)
    # Monotonic in int64 (after upcast); the int8 wrap would show
    # non-monotonic codes.
    out_int64 = out.astype(np.int64)
    assert (np.diff(out_int64) >= 0).all(), (
        f"codes not monotonic - dtype overflow: {out.tolist()}"
    )
    # Codes must reach near the max bin range, not wrap.
    assert int(out_int64.max()) > 127


def test_forced_narrow_dtype_raises():
    """Caller-forced int8 dtype with 300 edges must raise instead of
    silently wrapping.
    """
    from mlframe.feature_selection.filters.supervised_binning import apply_bin_edges
    edges = np.linspace(-10, 10, 300)
    x = np.linspace(-10, 10, 8)
    with pytest.raises(ValueError, match="exceeds caller-forced dtype"):
        apply_bin_edges(x, edges, dtype=np.int8)


def test_nan_routes_to_sentinel_not_top_bin():
    """NaN inputs MUST go to a dedicated sentinel (n_codes), not
    silently alias with the highest finite bin.
    """
    from mlframe.feature_selection.filters.supervised_binning import apply_bin_edges
    edges = np.array([-3.0, -1.0, 1.0, 3.0])
    x = np.array([-2.0, 0.0, 2.0, np.nan])
    out = apply_bin_edges(x, edges)
    n_codes = len(edges) - 1  # = 3 (bins, not edges)
    assert int(out[3]) == n_codes, (
        f"NaN should map to sentinel n_codes={n_codes}; got {int(out[3])}"
    )
    # Real bin codes for finite inputs unchanged.
    assert int(out[0]) == 0
    assert int(out[1]) == 1
    assert int(out[2]) == 2


def test_no_nan_input_unchanged():
    """Negative control: no NaN -> behaviour identical to pre-fix."""
    from mlframe.feature_selection.filters.supervised_binning import apply_bin_edges
    edges = np.array([-2.0, 0.0, 2.0])
    x = np.array([-1.5, -0.5, 0.5, 1.5])
    out = apply_bin_edges(x, edges)
    # 2 inner edges => 3 bin indices possible in [0, 1, 2]
    assert set(int(c) for c in out) <= {0, 1, 2}


def test_dtype_auto_picks_int16_at_boundary():
    """Edge count just above int8 max must auto-pick int16."""
    from mlframe.feature_selection.filters.supervised_binning import apply_bin_edges
    edges = np.linspace(-10, 10, 130)  # 129 bins
    x = np.linspace(-10, 10, 5)
    out = apply_bin_edges(x, edges)
    assert out.dtype == np.int16
