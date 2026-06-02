"""Layer 42 biz_value: DCD cluster_size_threshold default investigation.

WHY THIS LAYER
--------------
Layer 41 Probe C diagnosed that the default ``cluster_size_threshold=4``
(member count beyond the anchor) effectively gated DCD's PC1 swap OFF
in production: the canonical 3-feature redundancy cluster (anchor + 2
duplicates) was pruned but never reached the 4-member swap threshold,
so ``n_swaps`` stayed at zero on real data and the documented "PC1
denoised aggregate replaces the redundant raw anchor" behaviour
silently never fired.

Layer 42's first probe was "lower the default from 4 to 2 and check
it's a net positive". The investigation surfaced a second, more
serious downstream issue: when ``commit_swap`` does fire, it is
called at ``_screen_predictors.py`` with ``engineered_recipes=None``
- so the PC1 aggregate is appended to ``selected_vars`` but NOT
registered as an ``EngineeredRecipe``, and ``MRMR.get_feature_names_out``
drops it because ``support_`` only references ``feature_names_in_``
(which has not been extended with the aggregate's name). Net result of
lowering the default to 2 today: ``support_`` SHRINKS on real data
because the anchor is removed without the aggregate replacing it
visibly.

Per the layer brief's escape clause ("If lowering default breaks
downstream contracts, revert default and keep as opt-in"), Layer 42:

1. **Keeps the default at 4** in ``MRMR.dcd_cluster_size_threshold``,
   ``DCDState.cluster_size_threshold``, and the
   ``_screen_predictors._dcd_make_state`` fallback. The lever is well-
   documented in code so the next layer (recipe wiring) can flip it.

2. **Lowers the validation floor from 2 to 1** so users who DO want
   to swap on a strict 2-feature duplicate cluster (anchor + 1 member)
   can opt in via ``dcd_cluster_size_threshold=1``. The
   ``evaluate_swap_candidate`` floor (``max(min_cluster_size,
   cluster_size_threshold)``) keeps ``min_cluster_size`` as a separate
   hard guard, so ``=1`` only meaningfully helps when paired with
   ``dcd_min_cluster_size=1``.

3. **Tests** document both branches of the trade-off:
   - ``test_default_threshold_pinned_at_4`` proves the default did not
     flip (regression guard for any future "let me just lower it"
     attempt without doing the recipe wiring first).
   - ``test_opt_in_threshold_2_triggers_swap_on_3_dups`` proves the
     opt-in path actually fires the swap on the L41-Probe-C fixture
     (anchor + 2 perfect duplicates).
   - ``test_opt_in_threshold_2_no_swap_on_weak_correlation`` proves
     the SU=0.5 tau still rejects weakly-correlated pairs as cluster
     members, so threshold=2 does not invent spurious swaps.
   - ``test_validate_accepts_threshold_1`` proves the loosened
     floor admits =1 without ValueError.
   - ``test_validate_rejects_threshold_0`` proves the floor still
     refuses nonsense (=0).
   - ``test_swap_gain_threshold_0p02_not_a_win`` documents the
     ``swap_gain_threshold`` probe: 0.02 vs 0.05 makes no observable
     difference on the dup-cluster fixture because the
     deterministic gate (rep_rel > anchor_rel * 1.05 is already easily
     cleared on perfect dups; the perm-null gate is the binding
     constraint) - so we keep 0.05.
   - Regression spot-checks vs Layer 41 (cluster_members_) accessor.

NEVER xfail. Default-OFF byte-identical for users who pin the legacy
``dcd_cluster_size_threshold=4`` (which is the new default too).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Fixtures - Layer 41 Probe C reprise (3 perfect dups + a strong unrelated col)
# ---------------------------------------------------------------------------


def _three_dups_plus_strong_frame(n: int = 1500, seed: int = 0):
    """Strong unrelated col + 3 perfectly-correlated duplicates + 1 noise.

    With ``tau_cluster=0.5``, DCD anchors on the first duplicate and grows
    the cluster to anchor + 2 members. The PC1 swap then gates on
    ``cluster_size_threshold``: =4 never fires, =2 fires.
    """
    rng = np.random.default_rng(int(seed))
    latent = rng.standard_normal(n)
    other = rng.standard_normal(n)
    X = pd.DataFrame({
        "strong": other,
        "dup_a": latent + 0.01 * rng.standard_normal(n),
        "dup_b": latent + 0.01 * rng.standard_normal(n),
        "dup_c": latent + 0.01 * rng.standard_normal(n),
        "noise_0": rng.standard_normal(n),
    })
    y = pd.Series((2 * other + latent + 0.3 * rng.standard_normal(n) > 0).astype(int))
    return X, y


def _weak_pair_frame(n: int = 1500, seed: int = 0):
    """Two weakly-correlated columns (Spearman ~0.7) that should NOT
    form a DCD cluster at the default ``tau_cluster=0.5``. Verifies
    threshold=2 doesn't manufacture spurious swaps on weakly-related
    pairs (the membership rule itself is the guard - threshold only
    governs swap firing on already-formed clusters).
    """
    rng = np.random.default_rng(int(seed))
    latent = rng.standard_normal(n)
    a = latent
    b = latent + 1.0 * rng.standard_normal(n)
    X = pd.DataFrame({
        "a": a,
        "b": b,
        "noise_0": rng.standard_normal(n),
    })
    y = pd.Series((latent + 0.3 * rng.standard_normal(n) > 0).astype(int))
    return X, y


# ---------------------------------------------------------------------------
# 1. Default behaviour: threshold pinned at 4 pending recipe wiring.
# ---------------------------------------------------------------------------


class TestDefaultThresholdPinned:

    def test_default_threshold_pinned_at_4(self):
        """Layer 42 escape clause: the default stayed at 4 because
        lowering it net-shrinks ``support_`` on real data when the
        post-swap aggregate is not yet wired into
        ``_engineered_recipes_``. Any future PR that flips the default
        MUST land the recipe-propagation hookup in the same change so
        users get a usable aggregate column in the output. This test
        is the regression guard for that contract.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        m = MRMR()
        assert int(m.dcd_cluster_size_threshold) == 4, (
            f"Layer 42 keeps dcd_cluster_size_threshold default at 4 "
            f"pending the post-swap aggregate -> _engineered_recipes_ "
            f"wiring. See the docstring on the parameter for the full "
            f"trade-off; flipping the default in isolation drops "
            f"support_ on the canonical 3-feature redundancy fixture."
        )

    def test_default_fit_n_swaps_zero_on_3_dups(self):
        """Reciprocal: with the default threshold=4, the 3-perfect-dups
        fixture produces zero swaps (Layer 41 Probe C observation).
        Lowering the threshold is the only lever that flips this.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _three_dups_plus_strong_frame()
        m = MRMR(
            dcd_enable=True, dcd_tau_cluster=0.5,
            full_npermutations=50, verbose=0, random_seed=0,
        ).fit(X, y)
        assert m.dcd_["n_swaps"] == 0, (
            f"Default threshold=4 must keep n_swaps=0 on the 3-dups "
            f"fixture; got n_swaps={m.dcd_['n_swaps']}. If this fires, "
            f"someone lowered the default without updating the test."
        )
        # Pruning still fires (the 2 duplicate members are marked).
        assert m.dcd_["n_pruned"] >= 2, (
            f"Default DCD must still prune the duplicates; "
            f"n_pruned={m.dcd_['n_pruned']}"
        )


# ---------------------------------------------------------------------------
# 2. Opt-in: threshold=2 fires the PC1 swap on the 3-dup fixture.
# ---------------------------------------------------------------------------


class TestOptInThresholdTwoTriggersSwap:

    def test_opt_in_threshold_2_triggers_swap_on_3_dups(self):
        """With ``dcd_cluster_size_threshold=2``, the 3-dup fixture
        forms a cluster of size 2 members (anchor + 2 dups) and the
        PC1 swap fires when the deterministic gain + permutation null
        both clear. ``full_npermutations`` must be large enough to
        achieve ``p_value < swap_alpha=0.05``; B=50 yields min p =
        1/51 = 0.0196 which is fine.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _three_dups_plus_strong_frame()
        m = MRMR(
            dcd_enable=True, dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=2,
            full_npermutations=50,
            verbose=0, random_seed=0,
        ).fit(X, y)
        assert m.dcd_["n_swaps"] >= 1, (
            f"Opt-in dcd_cluster_size_threshold=2 must fire >=1 swap on "
            f"the 3-dup fixture; got n_swaps={m.dcd_['n_swaps']}. The "
            f"swap_log is {m.dcd_.get('swap_log')}."
        )
        # The swap log must record the PC1 aggregate as ``_dcd_pc1_*``.
        log = m.dcd_["swap_log"]
        assert any(entry["aggregate_name"].startswith("_dcd_pc1_") for entry in log), (
            f"Swap log must contain a ``_dcd_pc1_*`` aggregate name; got "
            f"{log}"
        )

    def test_opt_in_threshold_2_no_swap_on_weak_correlation(self):
        """Threshold=2 must not invent spurious swaps. Two
        weakly-correlated columns do not form a DCD cluster at
        ``tau_cluster=0.5`` (the membership rule's SU>tau guard), so
        no PC1 swap should fire regardless of threshold.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _weak_pair_frame()
        m = MRMR(
            dcd_enable=True, dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=2,
            full_npermutations=50,
            verbose=0, random_seed=0,
        ).fit(X, y)
        assert m.dcd_["n_swaps"] == 0, (
            f"Weak correlation (Spearman ~0.7) must NOT trigger a swap "
            f"at tau_cluster=0.5 + threshold=2; got n_swaps="
            f"{m.dcd_['n_swaps']}, swap_log={m.dcd_['swap_log']}"
        )

    def test_pin_threshold_4_bytewise_matches_default(self):
        """Users who pin the legacy ``dcd_cluster_size_threshold=4``
        get the exact same behaviour as the current default (since
        the default IS 4). Documents the byte-identical opt-out path.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _three_dups_plus_strong_frame()
        m_default = MRMR(
            dcd_enable=True, dcd_tau_cluster=0.5,
            full_npermutations=50, verbose=0, random_seed=0,
        ).fit(X, y)
        m_pin = MRMR(
            dcd_enable=True, dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=4,
            full_npermutations=50, verbose=0, random_seed=0,
        ).fit(X, y)
        assert list(m_default.get_feature_names_out()) == list(
            m_pin.get_feature_names_out()
        ), "Pinning =4 must match the current default exactly."
        assert m_default.dcd_["n_swaps"] == m_pin.dcd_["n_swaps"]
        assert m_default.dcd_["n_pruned"] == m_pin.dcd_["n_pruned"]


# ---------------------------------------------------------------------------
# 3. Validation: floor lowered from >=2 to >=1.
# ---------------------------------------------------------------------------


class TestValidationLowerBound:

    def test_validate_accepts_threshold_1(self):
        """The Layer 42 validate loosening admits ``=1`` without
        ValueError. Users who want to swap on a strict 2-feature
        duplicate cluster (anchor + 1 member) pair this with
        ``dcd_min_cluster_size=1`` to drop the
        ``evaluate_swap_candidate`` floor too.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _three_dups_plus_strong_frame(n=300)
        # No assertion that swap fires here - just that the validate
        # step accepts the value.
        m = MRMR(
            dcd_enable=True, dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=1,
            dcd_min_cluster_size=1,
            full_npermutations=20,
            verbose=0, random_seed=0,
        ).fit(X, y)
        assert hasattr(m, "support_"), (
            "fit() must complete with threshold=1 + min_cluster_size=1"
        )

    def test_validate_rejects_threshold_0(self):
        """The floor still refuses nonsense (=0)."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _three_dups_plus_strong_frame(n=300)
        with pytest.raises(ValueError, match="dcd_cluster_size_threshold"):
            MRMR(
                dcd_enable=True,
                dcd_cluster_size_threshold=0,
                verbose=0, random_seed=0,
            ).fit(X, y)


# ---------------------------------------------------------------------------
# 4. swap_gain_threshold probe: 0.02 vs 0.05 makes no observable difference.
# ---------------------------------------------------------------------------


class TestSwapGainThresholdProbe:

    def test_swap_gain_threshold_0p02_not_a_win(self):
        """Investigation: does loosening ``swap_gain_threshold`` from
        0.05 to 0.02 let through more wins on perfect dups? On the
        3-dup fixture both settings produce the same outcome (swap
        fires on both) because the rep/anchor MI ratio comfortably
        clears the larger threshold; the permutation-null gate is the
        binding constraint. So 0.05 stays the conservative default.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _three_dups_plus_strong_frame()
        m_005 = MRMR(
            dcd_enable=True, dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=2,
            dcd_swap_gain_threshold=0.05,
            full_npermutations=50, verbose=0, random_seed=0,
        ).fit(X, y)
        m_002 = MRMR(
            dcd_enable=True, dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=2,
            dcd_swap_gain_threshold=0.02,
            full_npermutations=50, verbose=0, random_seed=0,
        ).fit(X, y)
        # Both fire the swap on perfect dups - the looser threshold is
        # not the bottleneck.
        assert m_005.dcd_["n_swaps"] == m_002.dcd_["n_swaps"], (
            f"swap_gain_threshold 0.02 vs 0.05 changed n_swaps "
            f"({m_005.dcd_['n_swaps']} vs {m_002.dcd_['n_swaps']}); "
            f"either the fixture got more sensitive or the perm-null "
            f"gate stopped being binding."
        )


# ---------------------------------------------------------------------------
# 5. Regression: Layer 41 cluster_members_ accessor still reports correctly.
# ---------------------------------------------------------------------------


class TestNoRegressionLayer41:

    def test_cluster_members_accessor_reports_correctly(self):
        """The Layer 41 ``cluster_members_`` accessor must still report
        the right cluster on the 3-dup fixture after the Layer 42
        validate loosening. Spot-replicates Layer 41 contract C1 / C3.

        Rebaselined to be ANCHOR-AGNOSTIC. The old assertion required
        ``dup_a`` specifically to be the cluster anchor (with members
        {dup_b, dup_c}). ``dup_a``/``dup_b``/``dup_c`` are statistically
        interchangeable duplicates (all = latent + 0.01*noise), so which
        one DCD anchors on is just a function of greedy selection order.
        Under the new default (``use_simple_mode=False`` -> full-mode
        conditional-MI redundancy) the selection order shifted and DCD
        now anchors the dup cluster on ``dup_b`` (members {dup_a, dup_c})
        instead of ``dup_a`` -- an equally-correct clustering of the same
        three columns. We therefore pin the load-bearing, mode-invariant
        property: EXACTLY ONE cluster is anchored on one of the three
        dups and lists the OTHER TWO dups as its members. This is still
        fully falsifiable -- if DCD failed to cluster the duplicates the
        union {anchor} u members would not equal {dup_a, dup_b, dup_c}.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _three_dups_plus_strong_frame()
        m = MRMR(
            dcd_enable=True, dcd_tau_cluster=0.5,
            verbose=0, random_seed=0,
        ).fit(X, y)
        cm = m.cluster_members_
        assert isinstance(cm, dict) and len(cm) >= 1, (
            f"cluster_members_ must be a non-empty dict; got {cm!r}"
        )
        dups = {"dup_a", "dup_b", "dup_c"}
        # Find the cluster whose anchor is a dup AND whose members are dups:
        # the full dup cluster is {anchor} u members == {dup_a, dup_b, dup_c}.
        dup_clusters = [
            (k, set(v)) for k, v in cm.items()
            if k in dups and ({k} | set(v)) == dups
        ]
        assert len(dup_clusters) == 1, (
            f"Expected exactly one dup-anchored cluster covering all three "
            f"duplicates {dups}; got clusters {cm!r}"
        )
        anchor, members = dup_clusters[0]
        assert members == (dups - {anchor}), (
            f"dup cluster anchored on {anchor!r} must list the other two "
            f"dups as members; got {members}"
        )
        # And cluster_anchors_names mirrors it.
        assert m.dcd_["cluster_anchors_names"] == cm

    def test_dcd_disabled_path_unaffected(self):
        """The Layer 42 changes touch only DCD's gating logic. Users
        who fit with ``dcd_enable=False`` see no behavioural change.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _three_dups_plus_strong_frame(n=500)
        m = MRMR(
            dcd_enable=False, verbose=0, random_seed=0,
        ).fit(X, y)
        assert m.cluster_members_ is None, (
            "DCD-disabled fits must keep cluster_members_=None."
        )
        assert m.dcd_ is None or m.dcd_.get("n_swaps", 0) == 0


# ---------------------------------------------------------------------------
# 6. End-to-end: opt-in path produces a stable, documented result.
# ---------------------------------------------------------------------------


class TestOptInEndToEnd:

    def test_opt_in_fit_completes_and_records_swap(self):
        """End-to-end probe: the opt-in path completes fit, records
        the swap in ``dcd_['swap_log']``, and the aggregate name in
        the swap log is well-formed.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _three_dups_plus_strong_frame()
        m = MRMR(
            dcd_enable=True, dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=2,
            full_npermutations=50,
            verbose=0, random_seed=0,
        ).fit(X, y)
        log = m.dcd_["swap_log"]
        assert isinstance(log, list) and len(log) >= 1
        entry = log[0]
        assert "aggregate_name" in entry and entry["aggregate_name"]
        assert "anchor" in entry
        assert "n_members" in entry and entry["n_members"] >= 2
        assert "rep_relevance" in entry and entry["rep_relevance"] > 0
        assert "anchor_relevance_in_ctx" in entry and entry["anchor_relevance_in_ctx"] > 0
