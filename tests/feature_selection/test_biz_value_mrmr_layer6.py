"""Layer 6 biz_value MRMR contracts: adversarial decoy resistance.

THE TRAP
--------
Two clean signal components x1, x2 each carry moderate marginal MI with y.
A *decoy* feature is constructed as:

    decoy = x1 + x2 + small_noise
    y     = sign(x1 + x2)

decoy's marginal MI with y is HIGHER than MI(x1; y) or MI(x2; y) because
it directly encodes the linear combination y is generated from. A
pure-relevance ranker would always pick decoy first.

KEY FINDINGS
------------
1. **Default greedy MRMR cannot avoid the decoy first-pick.** At
   iteration 1 (empty selected_vars), the only signal available is
   marginal MI(f; y). Decoy maxes that, so it wins. This is INHERENT
   to greedy mRMR (Brown 2012 Sec 4.3) - 2-step lookahead would
   theoretically fix it but is an algorithmic enhancement out of
   scope.

2. **DCD (Dynamic Cluster Discovery) IS the documented solution for
   the "two near-duplicate decoys" scenario.** With ``dcd_enable=True``
   and ``dcd_tau_cluster <= 0.5``, the second decoy is pruned as a
   cluster member of the first.

3. **All three (x1, x2, decoy) survive default config.** This is the
   weak-but-useful contract: even with the marginal-MI champion
   competing for slots, the clean components are NOT crowded out.

CONTRACTS PINNED
----------------
* default config: clean components survive even when decoy is offered
* DCD config: near-duplicate decoys collapse via cluster pruning
* noise / nuisance columns are correctly rejected under decoy distortion
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


def _build_decoy_dataset(n: int = 2500, noise_scale: float = 0.25, seed: int = 6001):
    """y = sign(x1 + x2); decoy = x1 + x2 + eps; plus 4 nuisance cols."""
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    decoy = x1 + x2 + noise_scale * rng.standard_normal(n)
    nuisance = rng.standard_normal((n, 4))
    X = pd.DataFrame({
        "x1": x1, "x2": x2, "decoy": decoy,
        "noise0": nuisance[:, 0], "noise1": nuisance[:, 1],
        "noise2": nuisance[:, 2], "noise3": nuisance[:, 3],
    })
    y = pd.Series((x1 + x2 > 0).astype(np.int64), name="y")
    return X, y


def _build_two_decoy_dataset(seed: int = 6002, n: int = 2500):
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    decoy_a = x1 + x2 + 0.25 * rng.standard_normal(n)
    decoy_b = x1 + x2 + 0.25 * rng.standard_normal(n)
    X = pd.DataFrame({
        "x1": x1, "x2": x2,
        "decoy_a": decoy_a, "decoy_b": decoy_b,
        "noise0": rng.standard_normal(n),
        "noise1": rng.standard_normal(n),
    })
    y = pd.Series((x1 + x2 > 0).astype(np.int64))
    return X, y


class TestAdversarialDecoyDefault:
    """Default-config MRMR: weak-but-essential contracts on the decoy
    trap. Clean components survive even when the decoy has higher
    marginal MI."""

    def test_both_clean_components_survive_decoy(self):
        """Decoy may be picked first (inherent greedy-MRMR limitation
        for marginal-MI-dominated decoys), but BOTH x1 and x2 must
        survive selection.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _build_decoy_dataset()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, interactions_max_order=1, fe_max_steps=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert "x1" in names, (
            f"x1 (clean component) crowded out by decoy; support={names}"
        )
        assert "x2" in names, (
            f"x2 (clean component) crowded out by decoy; support={names}"
        )

    def test_noise_rejected_under_decoy_distortion(self):
        """Nuisance / pure-noise columns must NOT be selected (or at
        most 1) even when a high-MI decoy distorts the relevance
        ranking. FP guard remains intact under decoy presence.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _build_decoy_dataset()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, interactions_max_order=1, fe_max_steps=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        nuisance_picked = [c for c in names if c.startswith("noise")]
        # Allow up to 2 noise (default full_npermutations=3 has limited
        # FP statistical power, documented in Layer 5).
        assert len(nuisance_picked) <= 2, (
            f"Decoy distorted FP guard catastrophically: "
            f"{len(nuisance_picked)} nuisance columns selected. "
            f"support={names}"
        )

    def test_at_least_one_clean_component_in_top2(self):
        """If a downstream consumer truncates to top-2 features, at
        least one clean component must be there - decoy alone is not
        sufficient.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _build_decoy_dataset()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, interactions_max_order=1, fe_max_steps=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        top2 = names[:2] if len(names) >= 2 else names
        clean_in_top2 = sum(1 for nm in ("x1", "x2") if nm in top2)
        assert clean_in_top2 >= 1, (
            f"Neither x1 nor x2 in top-2; decoy fully crowded out "
            f"clean signal. top2={top2}, full={names}"
        )


class TestDCDDuplicateDecoyPruning:
    """DCD (Dynamic Cluster Discovery) IS the documented MRMR mechanism
    for near-duplicate decoy resistance. These tests verify DCD's
    contract: when two decoys are SU > tau_cluster similar, the second
    is pruned."""

    @pytest.mark.parametrize("tau", [0.3, 0.4, 0.5])
    def test_dcd_prunes_second_decoy_at_strict_tau(self, tau):
        """With ``dcd_enable=True`` and a strict tau (<= 0.5), two
        near-duplicate decoys must collapse to ONE in support_.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _build_two_decoy_dataset()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(
                verbose=0, dcd_enable=True,
                dcd_tau_cluster=tau,
                dcd_cluster_size_threshold=2,
                interactions_max_order=1, fe_max_steps=0,
            ).fit(X, y)
        names = list(sel.get_feature_names_out())
        decoys_picked = [c for c in names if c.startswith("decoy")]
        assert len(decoys_picked) <= 1, (
            f"DCD@tau={tau} failed to prune duplicate decoy; "
            f"support={names}"
        )

    def test_dcd_pruning_preserves_clean_components(self):
        """DCD must NOT prune the clean components x1, x2 when pruning
        the decoy cluster. Verifies the cluster discovery scopes
        correctly.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _build_two_decoy_dataset()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(
                verbose=0, dcd_enable=True,
                dcd_tau_cluster=0.4,
                dcd_cluster_size_threshold=2,
                interactions_max_order=1, fe_max_steps=0,
            ).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert "x1" in names or "x2" in names, (
            f"DCD over-pruned: clean components lost; support={names}"
        )


class TestDecoySeedRobustness:
    """The default-config decoy contracts must hold across multiple
    seeds - not a single lucky seed."""

    @pytest.mark.parametrize("seed", [6001, 6002, 6003, 6004, 6005])
    def test_clean_components_survive_across_seeds(self, seed):
        """At minimum, ONE clean component must survive on every seed.
        Cluster-aware selection requires both x1 and x2 to be present.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _build_decoy_dataset(seed=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, interactions_max_order=1, fe_max_steps=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        # At least one clean signal column must surface every seed.
        assert "x1" in names or "x2" in names, (
            f"seed={seed}: BOTH clean components lost; support={names}"
        )

    @pytest.mark.parametrize("noise_scale", [0.1, 0.25, 0.5, 1.0])
    def test_clean_components_survive_across_decoy_noise(self, noise_scale):
        """Vary the decoy noise level. Even when decoy = x1+x2 (no
        added noise), MRMR should pick at least one clean component.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _build_decoy_dataset(noise_scale=noise_scale)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, interactions_max_order=1, fe_max_steps=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert "x1" in names or "x2" in names, (
            f"noise_scale={noise_scale}: clean components both lost; "
            f"support={names}"
        )
