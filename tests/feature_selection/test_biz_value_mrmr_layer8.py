"""Layer 8 biz_value MRMR contracts: polynomial-expansion explosion.

WHY THIS LAYER
--------------
Auto-feature-engineering pipelines (sklearn ``PolynomialFeatures``,
featuretools, manual cross-products) routinely take ``k`` raw columns
and expand them to ``k + k*(k-1)/2`` (pairwise) or more (cubic, higher
order). A modest ``k=10`` raw columns balloons to 55 features after
pairwise expansion, of which only a handful matter.

Why this is high-leverage in production:

* Almost every tabular ML pipeline includes some kind of cross-product
  expansion either explicitly (``PolynomialFeatures``) or implicitly
  (interaction features from domain knowledge).
* The selector sees 45+ "noise" interactions that share variance with
  the real signal column - this is the textbook redundancy trap MRMR
  was designed to solve, but only if it actually does.
* If MRMR drowns under the explosion, downstream models train on
  hundreds of redundant features, blowing memory and slowing inference.

The benchmark
-------------
10 raw Gaussian columns ``r0..r9``. We auto-generate all 45 pairwise
products ``r{i}_x_r{j}``. y depends on:

* ``r0``                          (one raw column)
* ``r1_x_r2``                     (one engineered interaction)

So the ground truth is exactly 2 informative columns out of 55.
Everything else is correlated noise: many products share a variance
component with ``r0`` (which would naively bait MRMR), but they don't
add MI conditional on r0.

CONTRACTS PINNED
----------------
1. ``r0`` (raw signal) is in support_.
2. ``r1_x_r2`` (engineered interaction signal) is in support_.
3. Decoy products that contain r0 (``r0_x_rk`` for k!=1,2) are mostly
   pruned by redundancy - we bound the count.
4. Total support stays bounded (no drowning in 55 features).
5. Holds across seeds.
"""
from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import pytest


def _build_polynom_explosion(n: int = 2500, seed: int = 8001, k_raw: int = 10):
    """Build a synthetic dataset with ``k_raw`` raw Gaussian columns and
    all pairwise products. Target depends on ``r0`` (linear) and
    ``r1 * r2`` (multiplicative interaction).
    """
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n, k_raw))
    raw_names = [f"r{i}" for i in range(k_raw)]
    raw_df = pd.DataFrame(raw, columns=raw_names)

    # Auto-generate all pairwise products (sklearn-style explosion).
    prod_cols = {}
    for i, j in combinations(range(k_raw), 2):
        name = f"r{i}_x_r{j}"
        prod_cols[name] = raw[:, i] * raw[:, j]
    prod_df = pd.DataFrame(prod_cols)

    X = pd.concat([raw_df, prod_df], axis=1)

    # Target: linear in r0 + multiplicative in r1*r2 + Gaussian noise.
    logit = 1.2 * raw[:, 0] + 1.5 * (raw[:, 1] * raw[:, 2])
    logit += 0.3 * rng.standard_normal(n)
    y = pd.Series((logit > 0).astype(np.int64), name="y")
    return X, y


class TestPolynomExplosionBasics:
    """Both ground-truth signal columns must be selected and the
    selection must stay bounded under a 55-column input."""

    def test_raw_signal_selected(self):
        """``r0`` is the raw linear signal. It must appear in support_.
        If absent, MI on the raw column was dominated by some product
        that shares variance with r0 - which would be a real bug.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _build_polynom_explosion()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, interactions_max_order=1, fe_max_steps=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert "r0" in names, (
            f"Raw linear signal r0 lost under polynomial explosion; "
            f"support={names}"
        )

    def test_engineered_interaction_selected(self):
        """``r1_x_r2`` is the multiplicative signal. It must appear in
        support_; if MRMR picked r1 and r2 instead (which carry near-
        zero marginal MI under y=sign(r1*r2)+linear-r0), it failed the
        interaction-discovery contract.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _build_polynom_explosion()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, interactions_max_order=1, fe_max_steps=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert "r1_x_r2" in names, (
            f"Engineered interaction r1_x_r2 lost; support={names}"
        )

    def test_support_size_bounded(self):
        """We have 55 input features and only 2 signal columns. A
        sane selector should NOT return more than ~15 features
        (loose bound to allow noise FPs + redundancy slack). If
        support balloons past 20, MRMR is drowning in correlated
        products and the redundancy term failed.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _build_polynom_explosion()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, interactions_max_order=1, fe_max_steps=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert len(names) <= 20, (
            f"Support drowned in polynomial explosion: {len(names)}/55 "
            f"selected; support={names}"
        )


class TestPolynomRedundancyPruning:
    """Many products share variance with r0 (``r0_x_rk``). MRMR's
    redundancy term should prune most of them - we don't need to keep
    every r0-containing product since r0 itself is selected."""

    def test_r0_redundant_products_mostly_pruned(self):
        """Among the 9 products ``r0_x_rk`` for k in 1..9, at most a
        few should remain after r0 is already picked. They share a
        large variance component with r0 + noise from the other factor.
        Bound at 4 (generous - tightened below if observation supports).
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _build_polynom_explosion()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, interactions_max_order=1, fe_max_steps=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        r0_products = [nm for nm in names if nm.startswith("r0_x_")]
        assert len(r0_products) <= 4, (
            f"Redundant r0-products not pruned: {len(r0_products)} "
            f"r0_x_* in support; full support={names}"
        )

    def test_pure_noise_products_mostly_rejected(self):
        """Products like ``r3_x_r4``, ``r5_x_r6`` etc. (neither factor
        is in {r0, r1, r2}) carry zero true signal. We allow some FPs
        because permutation power is limited at default settings, but
        bound at 6 of the ~21 such products.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _build_polynom_explosion()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, interactions_max_order=1, fe_max_steps=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        signal_factors = {"r0", "r1", "r2"}

        def _is_pure_noise_product(nm: str) -> bool:
            if "_x_" not in nm:
                return False
            a, b = nm.split("_x_")
            return a not in signal_factors and b not in signal_factors

        noise_products = [nm for nm in names if _is_pure_noise_product(nm)]
        assert len(noise_products) <= 6, (
            f"Too many pure-noise products in support: "
            f"{len(noise_products)} survived; support={names}"
        )


class TestPolynomSeedRobustness:
    """The two-ground-truth-columns contract must hold across multiple
    seeds. If it works only on the lucky seed, it's not a real win."""

    @pytest.mark.parametrize("seed", [8001, 8002, 8003, 8004, 8005])
    def test_both_signals_across_seeds(self, seed):
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _build_polynom_explosion(seed=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, interactions_max_order=1, fe_max_steps=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert "r0" in names, (
            f"seed={seed}: raw signal r0 lost; support={names}"
        )
        assert "r1_x_r2" in names, (
            f"seed={seed}: interaction r1_x_r2 lost; support={names}"
        )


class TestPolynomTopRanks:
    """Both true signals should appear in the TOP HALF of selected
    features (ranked by MRMR score)."""

    def test_signals_in_top_half(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _build_polynom_explosion()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, interactions_max_order=1, fe_max_steps=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert len(names) >= 2, f"Need at least 2 features; got {names}"
        top_half = set(names[: max(1, len(names) // 2 + 1)])
        # At least ONE of the two true signals must be in the top half;
        # ideally both, but interaction-MI estimation is noisier than
        # linear-MI so we pin the weaker, more robust contract here.
        n_top = ("r0" in top_half) + ("r1_x_r2" in top_half)
        assert n_top >= 1, (
            f"Neither true signal made the top half; top_half={top_half}, "
            f"full={names}"
        )
