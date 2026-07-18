"""biz_value: ORDER-3 Westfall-Young maxT permutation-null floor on the seeded-TRIPLE joint MI.

The triple proposers (surrogate-GBM split co-occurrence seeder, CMI-lattice) rank candidate 3-way interactions by
JOINT MI(x_a, x_b, x_c; y). At any non-trivial pool width the MAX joint MI over PURE-NOISE triples is a positive order
statistic that grows with the pool size -- the same best-of-pool selection bias the order-1 / order-2 floors reject,
now at order 3. ``pooled_triple_permutation_null_joint_mi_floor`` (wired into ``_gate_seeded_triples_order3``) shuffles
the discretised target K times, takes the per-shuffle MAX 3-way joint MI over the candidate pool via the SAME
``batch_triple_mi_prange`` estimator the seeder scores with, and floors triple selection at the q-th quantile.

These gates pin the floor's measurable win directly on the null helper + the gate consumer (the full-fit triple path
needs the heavy GBM/CMI seeder, so the floor's contribution is measured here in isolation -- the same shape the proven
order-2 helper test uses):
  A. genuine 3-way synergy joint MI clears the null-max floor with margin; the overwhelming majority of pure-noise
     triples sit at/below it (the floor separates signal from best-of-pool chance-max noise).
  B. SELF-GATING / disable: ``n_permutations=0`` and sub-``min_triples`` pools return floor 0.0 (no-op).
  C. the gate consumer ``_gate_seeded_triples_order3`` keeps the genuine triple and drops noise-only triples ON,
     vs keeping ALL triples when the floor is disabled (perms=0).
"""

from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
import pandas as pd

GENUINE = {"x1", "x2", "x3"}


def _synergy_frame(n=1400, n_noise=40, seed=20260619):
    """One genuine 3-way XOR-sign synergy (x1,x2,x3) with ~zero marginal/pairwise MI, plus Gaussian noise columns."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    s = np.sign(x1) * np.sign(x2) * np.sign(x3)  # genuine 3-way interaction, no order<=2 leakage
    p = 1.0 / (1.0 + np.exp(-2.0 * s))
    y = (rng.random(n) < p).astype(int)
    d = {"x1": x1, "x2": x2, "x3": x3}
    for j in range(n_noise):
        d[f"noise_{j}"] = rng.normal(size=n)
    return pd.DataFrame(d), pd.Series(y, name="y")


def _discretize(X, y, n_bins=8):
    """Helper that discretize."""
    from mlframe.feature_selection.filters.discretization import categorize_dataset
    from mlframe.feature_selection.filters.info_theory import merge_vars

    df = X.copy()
    df["y"] = y.values if hasattr(y, "values") else y
    cols = list(df.columns)
    data, _c, nbins = categorize_dataset(df=df, method="quantile", n_bins=n_bins, dtype=np.int16)
    y_idx = cols.index("y")
    classes_y, freqs_y, _ = merge_vars(
        factors_data=data,
        vars_indices=[y_idx],
        var_is_nominal=None,
        factors_nbins=nbins,
        dtype=np.int16,
    )
    return data, nbins, cols, classes_y, freqs_y


class TestPooledTriplePermutationNullHelper:
    """Groups tests covering TestPooledTriplePermutationNullHelper."""
    def test_genuine_3way_above_null_noise_at_or_below(self):
        """Genuine 3way above null noise at or below."""
        from mlframe.feature_selection.filters.info_theory._batch_kernels import batch_triple_mi_prange
        from mlframe.feature_selection.filters._permutation_null import (
            pooled_triple_permutation_null_joint_mi_floor,
        )

        X, y = _synergy_frame(n_noise=22)
        data, nbins, cols, classes_y, freqs_y = _discretize(X, y)

        feat_idx = [cols.index(c) for c in cols if c != "y"]
        triples = list(combinations(feat_idx, 3))
        ta = np.fromiter((t[0] for t in triples), dtype=np.int64, count=len(triples))
        tb = np.fromiter((t[1] for t in triples), dtype=np.int64, count=len(triples))
        tc = np.fromiter((t[2] for t in triples), dtype=np.int64, count=len(triples))
        mis = batch_triple_mi_prange(data, ta, tb, tc, nbins, classes_y, freqs_y)

        floor = pooled_triple_permutation_null_joint_mi_floor(
            factors_data=data,
            nbins=nbins,
            triple_a=ta,
            triple_b=tb,
            triple_c=tc,
            classes_y=classes_y,
            freqs_y=freqs_y,
            n_permutations=25,
            quantile=0.95,
            random_seed=42,
        )
        assert floor > 0.0, "null floor should be positive on a non-trivial triple pool"

        genuine_key = tuple(sorted(cols.index(c) for c in ("x1", "x2", "x3")))
        gen_mi = None
        noise_mis = []
        for k, t in enumerate(triples):
            key = tuple(sorted(t))
            names = [cols[i] for i in t]
            if key == genuine_key:
                gen_mi = mis[k]
            elif all(nm.startswith("noise_") for nm in names):
                noise_mis.append(mis[k])
        noise_mis = np.array(noise_mis)
        assert gen_mi is not None
        # Genuine 3-way synergy clears the null-max floor.
        assert gen_mi > floor, f"genuine 3-way joint MI {gen_mi:.5f} not above null floor {floor:.5f}"
        # The overwhelming majority of pure-noise triples sit at/below the floor.
        below = float((noise_mis <= floor).mean())
        assert below >= 0.90, f"only {below:.2%} of noise triples at/below floor {floor:.5f}; max={noise_mis.max():.5f}"

    def test_disable_and_degenerate_return_zero_floor(self):
        """Disable and degenerate return zero floor."""
        from mlframe.feature_selection.filters._permutation_null import (
            pooled_triple_permutation_null_joint_mi_floor,
        )

        data = np.zeros((100, 4), dtype=np.int16)
        nbins = np.array([2, 2, 2, 2], dtype=np.int64)
        ta = np.array([0, 0], dtype=np.int64)
        tb = np.array([1, 1], dtype=np.int64)
        tc = np.array([2, 3], dtype=np.int64)
        cy = np.zeros(100, dtype=np.int16)
        cy[::2] = 1
        freqs_y = np.array([0.5, 0.5], dtype=np.float64)
        # n_permutations == 0 disables.
        assert (
            pooled_triple_permutation_null_joint_mi_floor(
                factors_data=data,
                nbins=nbins,
                triple_a=ta,
                triple_b=tb,
                triple_c=tc,
                classes_y=cy,
                freqs_y=freqs_y,
                n_permutations=0,
            )
            == 0.0
        )
        # too few candidate triples (1) -> 0 floor.
        assert (
            pooled_triple_permutation_null_joint_mi_floor(
                factors_data=data,
                nbins=nbins,
                triple_a=np.array([0], dtype=np.int64),
                triple_b=np.array([1], dtype=np.int64),
                triple_c=np.array([2], dtype=np.int64),
                classes_y=cy,
                freqs_y=freqs_y,
                n_permutations=25,
            )
            == 0.0
        )


class TestGateSeededTriplesOrder3Consumer:
    """Groups tests covering TestGateSeededTriplesOrder3Consumer."""
    def _gate(self, perms):
        """Helper that gate."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        from mlframe.feature_selection.filters._mrmr_fe_step_helpers import _gate_seeded_triples_order3

        X, y = _synergy_frame(n_noise=22)
        data, nbins, cols, classes_y, freqs_y = _discretize(X, y)
        gk = tuple(cols.index(c) for c in ("x1", "x2", "x3"))
        noise_idx = [cols.index(c) for c in cols if c.startswith("noise_")]
        # genuine triple + several noise-only triples (>= fe_triple_maxt_min_triples default 4 so the floor fires).
        seeded = [gk] + [tuple(noise_idx[i : i + 3]) for i in range(0, 15, 3)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = MRMR(verbose=0, random_seed=42, fe_triple_maxt_null_permutations=perms)
        kept = _gate_seeded_triples_order3(
            m,
            seeded,
            data=data,
            nbins=nbins,
            classes_y=classes_y,
            freqs_y=freqs_y,
            verbose=0,
        )
        return gk, set(kept)

    def test_floor_on_keeps_genuine_drops_noise_off_keeps_all(self):
        """Floor on keeps genuine drops noise off keeps all."""
        gk, kept_off = self._gate(perms=0)
        gk2, kept_on = self._gate(perms=25)
        # perms=0 is the no-op path: every seeded triple kept.
        assert gk in kept_off and len(kept_off) > 1, f"disabled gate dropped triples: {kept_off}"
        # Floor ON keeps the genuine triple and drops strictly more noise-only triples than OFF.
        assert gk2 in kept_on, "order-3 floor dropped the genuine 3-way synergy triple"
        assert len(kept_on) < len(kept_off), f"order-3 floor did not prune noise triples: OFF={kept_off} ON={kept_on}"
