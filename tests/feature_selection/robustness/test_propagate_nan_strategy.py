"""Wave 9.1 loop-iter-11 regression: ``missing_strategy='propagate'`` MUST
NOT silently merge NaN rows into the top real bin.

Pre-fix: ``_handle_missing(strategy='propagate')`` returned the NaN-bearing
array unchanged at discretization.py:394. Downstream
``np.searchsorted(ej, NaN, side='right')`` returns ``ej.size`` - the same
code as the column's TOP real bin - so NaN rows silently collapsed into
the highest-value real category. Net effect: any column whose NaN-ness
carried signal got near-zero MI under ``propagate``, while the
sibling ``separate_bin`` strategy correctly preserved it.

Concrete numeric demonstration:
  - 200 rows of N(0,1), 200 rows of NaN; y=0 for non-NaN, y=1 for NaN
  - NaN-ness perfectly predicts y -> theoretical I(x; y) = ln(2) = 0.693 nats
  - separate_bin: MI = 0.6931 (correct)
  - propagate (pre-fix): MI = 0.3804 (45% signal loss)
  - propagate (post-fix): MI = 0.6931 (correct)

Fix at discretization.py:394 + :1027 + docstring at :373:
  ``propagate`` now median-fills + lets the caller's _nan_mask reroute
  NaN positions to the dedicated NaN bin, exactly like ``separate_bin``.
  The docstring is corrected to match - it previously claimed
  ``propagate`` would "route to the lowest bin or raise" which was
  never the actual behaviour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def test_propagate_does_not_collide_nan_with_top_real_bin():
    """Pre-fix: nan_codes={3} colliding with real bins [0,1,2,3].
    Post-fix: nan_codes={4}, real [0,1,2,3], disjoint.
    """
    from mlframe.feature_selection.filters.discretization import categorize_dataset

    rng = np.random.default_rng(0)
    n = 600
    x = rng.normal(size=n)
    nan_idx = rng.choice(n, 30, replace=False)
    x[nan_idx] = np.nan

    data, cols, _nbins = categorize_dataset(
        df=pd.DataFrame({"x": x}),
        n_bins=4,
        dtype=np.int16,
        missing_strategy="propagate",
        nbins_strategy=None,
    )
    codes = data[:, cols.index("x")]
    real_codes = {int(c) for c in codes[~np.isnan(x)]}
    nan_codes = {int(c) for c in codes[np.isnan(x)]}
    collision = real_codes & nan_codes
    assert not collision, f"propagate must put NaN in a dedicated bin; got nan_codes={nan_codes}, real_codes={sorted(real_codes)}, collision={collision}"


def test_propagate_preserves_nan_as_signal():
    """When NaN-ness perfectly predicts y, MI must be ln(2) = 0.693
    nats - not the pre-fix 0.380 nats produced by the silent top-bin
    merge.
    """
    from mlframe.feature_selection.filters.discretization import categorize_dataset
    from mlframe.feature_selection.filters.info_theory import mi

    rng = np.random.default_rng(1)
    n = 400
    x = rng.normal(size=n)
    x[200:] = np.nan
    y = np.zeros(n, dtype=np.int64)
    y[200:] = 1

    data, cols, nbins = categorize_dataset(
        df=pd.DataFrame({"x": x, "y": y}),
        n_bins=4,
        dtype=np.int16,
        missing_strategy="propagate",
        nbins_strategy=None,
    )
    mi_val = mi(
        data,
        np.array([cols.index("x")], dtype=np.int64),
        np.array([cols.index("y")], dtype=np.int64),
        nbins,
    )
    # ln(2) = 0.693147; allow small slack for plug-in bias.
    assert mi_val > 0.6, (
        f"propagate must preserve NaN-as-signal: MI(x; y) = "
        f"{mi_val:.4f}, expected ~0.693 (= ln(2)) since NaN-ness "
        f"perfectly predicts y. Pre-fix value was ~0.38 due to silent "
        f"top-bin merge."
    )


def test_propagate_and_separate_bin_produce_equivalent_results():
    """Post-fix: ``propagate`` and ``separate_bin`` are functionally
    equivalent (propagate is an alias). Any divergence indicates a
    half-fix or regression.
    """
    from mlframe.feature_selection.filters.discretization import categorize_dataset
    from mlframe.feature_selection.filters.info_theory import mi

    rng = np.random.default_rng(2)
    n = 500
    x = rng.normal(size=n)
    x[rng.choice(n, 50, replace=False)] = np.nan
    y = (rng.standard_normal(n) > 0).astype(np.int64)

    miv = {}
    for strat in ("separate_bin", "propagate"):
        data, cols, nbins = categorize_dataset(
            df=pd.DataFrame({"x": x, "y": y}),
            n_bins=4,
            dtype=np.int16,
            missing_strategy=strat,
            nbins_strategy=None,
        )
        miv[strat] = float(
            mi(
                data,
                np.array([cols.index("x")], dtype=np.int64),
                np.array([cols.index("y")], dtype=np.int64),
                nbins,
            )
        )
    assert (
        abs(miv["separate_bin"] - miv["propagate"]) < 1e-10
    ), f"propagate must equal separate_bin: separate_bin={miv['separate_bin']:.6f}, propagate={miv['propagate']:.6f}"
