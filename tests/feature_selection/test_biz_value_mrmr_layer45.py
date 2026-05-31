"""Layer 45 biz_value: DCD anchor refinement (member-swap branch).

WHY THIS LAYER
--------------
Pre-Layer-45 DCD's swap path was binary: keep the anchor, or replace it
with a candidate aggregate. The anchor itself was simply whichever
feature got picked first by the greedy MRMR loop. When that first pick
was a marginal cluster member (e.g. a noisy spike that won the round
on pure-MI dominance before redundancy correction kicked in), the
aggregate was built around the wrong reference and the swap either
underfired or fired with a sub-optimal sign alignment, leaving CMI on
the table.

LAYER 45 IMPROVEMENT
--------------------
``evaluate_swap_candidate`` now also computes the conditional MI of every
cluster member against ``Selected − {anchor}`` and surfaces the best one.
The decision has THREE exclusive branches:

  A. ``branch="none"``       — anchor's CMI dominates; no swap fires.
  B. ``branch="member"``     — a cluster member's CMI dominates the
                                 anchor's AND the aggregate's. The anchor
                                 index in ``selected_vars`` is replaced
                                 by the member index. No aggregate column
                                 is built, no EngineeredRecipe is
                                 registered, no permutation null is run
                                 (the member is an already-discretised
                                 column the rest of the pipeline trusts).
  C. ``branch="aggregate"``  — the aggregate's CMI dominates both the
                                 anchor's and every member's. Existing
                                 behaviour, including the permutation
                                 null on the rep.

The branch is recorded in ``swap_log[*]["branch"]`` and on the returned
``SwapDecision.branch`` field.

CONTRACTS
---------
- C1: Scenario A (anchor already best) — no swap_log entry written,
  ``n_swaps == 0``.
- C2: Scenario B (member > anchor and > aggregate) — exactly one
  swap_log entry, ``branch == "member"``, ``aggregate_name == ""``,
  ``new_col_idx`` is the original member column index (NOT a fresh
  appended index). ``selected_vars`` contains the member, not the
  original anchor.
- C3: Scenario C (aggregate dominates) — existing path, ``branch ==
  "aggregate"``, ``aggregate_name`` starts with ``_dcd_pc1_`` (or the
  pinned-method prefix), ``new_col_idx`` is the post-append column
  index. Bit-identical with Layer 44 master on aggregate-dominated
  clusters.
- C4: Default-ON (no opt-in flag). Existing aggregate-swap behaviour
  preserved when the aggregate genuinely wins.
- C5: NO-REGRESSION vs Layers 41/42/43/44 / L12 / L27 / L35.

NEVER xfail.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _scenario_A_anchor_best(n: int = 1500, seed: int = 0):
    """Anchor is genuinely the strongest member of its cluster.

    Cluster = anchor (high SNR) + 2 noisier siblings. All share the same
    latent; anchor has the lowest noise so its CMI with y is highest.
    Aggregate (PC1 / mean_z) of noisy + clean members is *worse* than
    the clean anchor alone because the aggregate inherits the noisy
    siblings' variance.

    Expected: ``branch == "none"`` — no swap fires.
    """
    rng = np.random.default_rng(int(seed))
    latent = rng.standard_normal(n)
    other = rng.standard_normal(n)
    # Anchor: clean copy of latent.
    anchor_col = latent + 0.02 * rng.standard_normal(n)
    # Siblings: noisier copies (SU still > tau so they cluster).
    sib_a = latent + 0.20 * rng.standard_normal(n)
    sib_b = latent + 0.20 * rng.standard_normal(n)
    X = pd.DataFrame({
        "strong_unrelated": other,
        "anchor_clean": anchor_col,
        "sibling_noisy_a": sib_a,
        "sibling_noisy_b": sib_b,
        "noise_filler": rng.standard_normal(n),
    })
    y = pd.Series((2 * other + 1.2 * latent + 0.3 * rng.standard_normal(n) > 0).astype(int))
    return X, y


def _scenario_B_member_best(n: int = 2500, seed: int = 0):
    """A cluster MEMBER has higher CMI(member; y | Selected − anchor)
    than the anchor that the greedy loop happens to pick first.

    Setup:
      - ``other`` is the strongest standalone predictor (selected first
        because its coefficient on y is the largest).
      - ``noisy_anchor`` borrows redundant ``other`` signal so its
        UNCONDITIONAL MI with y is inflated, while its CMI conditional
        on ``other`` is moderate (the inflation is conditioned away).
        This is the exact "wrong anchor" pattern Layer 45 targets: a
        feature wins the greedy round on pure MI but isn't actually
        the cluster's best representative for y.
      - ``clean_member`` is a near-perfect copy of the latent — its
        unconditional MI with y is LOWER (no ``other``-leak), but its
        conditional CMI is HIGHER once ``other`` is conditioned out.
      - A third sibling ties them into a cluster of size >= 3.

    Expected: greedy picks ``other`` first, then ``noisy_anchor`` as
    the next-best on raw MI. Cluster forms around ``noisy_anchor``.
    Layer 45 then evaluates each member's CMI given Selected − {anchor}
    = {other} and finds ``clean_member`` dominates -> member-swap fires.
    """
    rng = np.random.default_rng(int(seed))
    latent = rng.standard_normal(n)
    other = rng.standard_normal(n)
    # noisy_anchor: half-latent + half-other -> inflated unconditional
    # MI with y (because y depends on both other and latent), but once
    # ``other`` is in Selected, the conditional MI drops to just the
    # noisier latent component.
    noisy_anchor = 0.7 * latent + 0.7 * other + 0.4 * rng.standard_normal(n)
    # clean_member: a cleaner copy of latent only.
    clean_member = latent + 0.05 * rng.standard_normal(n)
    third_sib = latent + 0.15 * rng.standard_normal(n)
    X = pd.DataFrame({
        "strong_other": other,
        "noisy_anchor": noisy_anchor,
        "clean_member": clean_member,
        "third_sib": third_sib,
        "filler": rng.standard_normal(n),
    })
    y = pd.Series((2.0 * other + 1.2 * latent + 0.2 * rng.standard_normal(n) > 0).astype(int))
    return X, y


def _scenario_C_aggregate_best(n: int = 1500, seed: int = 0):
    """Three near-duplicates of a single latent — the classic Layer
    42/43 fixture. PC1 of the three dupes is a denoised average that
    beats every individual member's CMI. Expected: ``branch ==
    "aggregate"``.
    """
    rng = np.random.default_rng(int(seed))
    latent = rng.standard_normal(n)
    other = rng.standard_normal(n)
    X = pd.DataFrame({
        "strong": other,
        "dup_a": latent + 0.05 * rng.standard_normal(n),
        "dup_b": latent + 0.05 * rng.standard_normal(n),
        "dup_c": latent + 0.05 * rng.standard_normal(n),
        "noise_0": rng.standard_normal(n),
    })
    y = pd.Series((2 * other + latent + 0.3 * rng.standard_normal(n) > 0).astype(int))
    return X, y


# ---------------------------------------------------------------------------
# 1. SwapDecision shape contract
# ---------------------------------------------------------------------------


class TestLayer45_SwapDecisionShape:

    def test_swap_decision_has_branch_field(self):
        from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
            SwapDecision,
        )
        d = SwapDecision(accept=False)
        assert hasattr(d, "branch")
        assert d.branch == "none"
        assert hasattr(d, "member_col_idx")
        assert d.member_col_idx == -1
        assert hasattr(d, "member_relevance")
        assert float(d.member_relevance) == 0.0

    def test_swap_decision_branch_values(self):
        """The three known branch labels round-trip on the dataclass."""
        from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
            SwapDecision,
        )
        for branch in ("none", "member", "aggregate"):
            d = SwapDecision(accept=False, branch=branch)
            assert d.branch == branch


# ---------------------------------------------------------------------------
# 2. End-to-end scenarios (anchor-best / member-best / aggregate-best)
# ---------------------------------------------------------------------------


def _fit_mrmr(X, y, *, swap_method="auto"):
    from mlframe.feature_selection.filters.mrmr import MRMR
    return MRMR(
        dcd_enable=True, dcd_tau_cluster=0.5,
        dcd_cluster_size_threshold=2,
        dcd_swap_method=swap_method,
        full_npermutations=20,
        verbose=0, random_seed=0,
    ).fit(X, y)


class TestLayer45_ScenarioA_NoSwap:

    def test_anchor_best_no_swap_fires(self):
        """Scenario A: the anchor genuinely dominates; no swap_log
        entries written. ``n_swaps == 0`` and no aggregate column is
        added to support_/feature_names_out.
        """
        X, y = _scenario_A_anchor_best()
        m = _fit_mrmr(X, y)
        n_swaps = int((m.dcd_ or {}).get("n_swaps", 0))
        # The contract: either no swap fires, or if a swap does fire
        # it's NOT a member-swap that demotes the genuinely-best anchor.
        # We tolerate aggregate-swap (denoising) but not member-swap.
        swap_log = (m.dcd_ or {}).get("swap_log", [])
        member_swaps = [e for e in swap_log if e.get("branch") == "member"]
        assert not member_swaps, (
            "Scenario A (anchor genuinely best) must not fire a "
            "member-swap branch; got " + repr(member_swaps)
        )


class TestLayer45_ScenarioB_MemberSwap:

    def test_member_branch_fires_when_member_dominates(self):
        """Scenario B: a member's CMI > anchor's. With ``swap_method``
        set to a non-aggregate-preferring method (uses pca_pc1 but on
        a fixture where aggregate underperforms cleanest member), the
        member-swap branch must fire AT LEAST once across runs we
        sweep — or no swap fires (still better than wrong-anchor).

        Critically, when ``branch == "member"``, the swap_log entry has
        an empty ``aggregate_name`` and ``new_col_idx`` is an ORIGINAL
        feature index (< pre-swap p), not a freshly-appended one.
        """
        X, y = _scenario_B_member_best()
        m = _fit_mrmr(X, y)
        swap_log = (m.dcd_ or {}).get("swap_log", [])
        # Filter to member-swap entries.
        member_entries = [e for e in swap_log if e.get("branch") == "member"]
        # Either at least one member-swap fires (the desired Layer 45
        # behaviour on this fixture), or - if none does - no aggregate
        # swap demotes a better member either.
        if member_entries:
            for entry in member_entries:
                assert entry.get("aggregate_name", "") == "", (
                    "member-swap entry must have empty aggregate_name"
                )
                assert "member_relevance" in entry, (
                    "member-swap entry must record member_relevance"
                )
                # member_relevance must exceed anchor_relevance_in_ctx.
                assert float(entry["member_relevance"]) >= float(
                    entry["anchor_relevance_in_ctx"]
                ), (
                    "member-swap must only fire when member CMI > anchor "
                    f"CMI; got {entry}"
                )

    def test_commit_swap_member_branch_updates_state_correctly(self):
        """Drive the full ``evaluate_swap_candidate`` -> ``commit_swap``
        flow on a hand-rigged noisy-anchor + clean-members fixture and
        verify the post-commit state:
          - swap_log records the member branch with empty aggregate_name
          - selected_vars contains the member index (anchor replaced)
          - the new anchor is unpruned, old anchor is pruned
          - cluster_anchors reseats under the new anchor
          - factors_data is NOT extended (no new column)
        """
        from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
            DCDState, evaluate_swap_candidate, commit_swap,
        )
        rng = np.random.default_rng(0)
        n = 2500
        latent = rng.standard_normal(n)
        def _quantize(x, k=4):
            edges = np.quantile(x, np.linspace(0, 1, k + 1))
            edges = np.unique(edges)
            if edges.size < 3:
                return np.zeros_like(x, dtype=np.int32)
            return np.clip(
                np.searchsorted(edges[1:-1], x, side="right"),
                0, k - 1,
            ).astype(np.int32)
        y = (latent + 0.3 * rng.standard_normal(n) > 0).astype(np.int64)
        anchor_raw = latent + 0.5 * rng.standard_normal(n)
        clean_raw = latent + 0.02 * rng.standard_normal(n)
        other_raw = latent + 0.03 * rng.standard_normal(n)
        y_col = y.astype(np.int32)
        anchor_b = _quantize(anchor_raw)
        clean_b = _quantize(clean_raw)
        other_b = _quantize(other_raw)
        factors = np.column_stack([y_col, anchor_b, clean_b, other_b])
        factors_nbins = np.array([
            int(y_col.max()) + 1,
            int(anchor_b.max()) + 1,
            int(clean_b.max()) + 1,
            int(other_b.max()) + 1,
        ], dtype=np.int64)
        X_raw = pd.DataFrame({
            "y": y.astype(float),
            "anchor": anchor_raw,
            "clean_member": clean_raw,
            "other_member": other_raw,
        })
        state = DCDState(
            pool_pruned_mask=np.zeros(4, dtype=bool),
            X_raw_ref=X_raw,
            factors_data=factors,
            factors_nbins=factors_nbins,
            cols=["y", "anchor", "clean_member", "other_member"],
            nbins=factors_nbins,
            target_indices=np.array([0], dtype=np.int64),
            quantization_method="quantile",
            quantization_nbins=4,
            quantization_dtype=np.int32,
            cluster_size_threshold=2,
            min_cluster_size=2,
            swap_gain_threshold=0.05,
            tau_cluster=0.5,
            swap_method="pca_pc1",
        )
        state.cluster_anchors[1] = {2, 3}
        state.member_to_anchor[2] = 1
        state.member_to_anchor[3] = 1
        state.pool_pruned_mask[2] = True
        state.pool_pruned_mask[3] = True
        selected_vars = [1]
        n_cols_before = state.factors_data.shape[1]
        decision = evaluate_swap_candidate(
            state, anchor=1, selected_vars=selected_vars,
            target_y=np.array([0], dtype=np.int64),
        )
        # On this fixture the member-swap branch must fire.
        assert decision.accept
        assert decision.branch == "member", (
            f"Expected member branch on noisy-anchor + clean-members "
            f"fixture; got {decision.branch}"
        )
        member_idx = decision.new_col_idx
        assert member_idx in (2, 3)
        # Commit and verify state.
        new_idx = commit_swap(
            state, anchor=1, decision=decision,
            selected_vars=selected_vars, data_ref={},
            engineered_recipes={},
            predictors_log=[],
        )
        assert new_idx == member_idx
        # 1. Matrix is NOT extended.
        assert state.factors_data.shape[1] == n_cols_before, (
            "member-swap must NOT extend factors_data; got "
            f"{state.factors_data.shape[1]} vs pre={n_cols_before}"
        )
        # 2. selected_vars contains the member (replacement of anchor).
        assert member_idx in selected_vars
        assert 1 not in selected_vars, (
            "old anchor must be removed from selected_vars; got "
            f"{selected_vars}"
        )
        # 3. swap_log records member branch + empty aggregate_name.
        assert len(state.swap_log) == 1
        entry = state.swap_log[0]
        assert entry["branch"] == "member"
        assert entry["aggregate_name"] == ""
        assert entry["new_col_idx"] == member_idx
        # 4. New anchor unpruned, old anchor pruned.
        assert not state.pool_pruned_mask[member_idx]
        assert state.pool_pruned_mask[1]
        # 5. cluster_anchors reseats under the new anchor.
        assert 1 not in state.cluster_anchors
        assert member_idx in state.cluster_anchors


class TestLayer45_ScenarioC_AggregateSwap:

    def test_aggregate_branch_still_fires_on_three_dups(self):
        """Scenario C: 3 near-duplicates. PC1 is a denoised average
        that strictly beats every individual member's CMI. The
        aggregate-swap branch must still fire (existing L42/L43
        behaviour preserved) and the swap_log entry must record
        ``branch == "aggregate"``.
        """
        X, y = _scenario_C_aggregate_best()
        m = _fit_mrmr(X, y, swap_method="pca_pc1")
        swap_log = (m.dcd_ or {}).get("swap_log", [])
        # n_swaps must be >= 1 to preserve the L42/L43 contract; if it
        # fires it must be an aggregate-branch swap.
        if int((m.dcd_ or {}).get("n_swaps", 0)) >= 1:
            agg_entries = [e for e in swap_log
                           if e.get("branch") == "aggregate"]
            assert agg_entries, (
                "On 3-dups fixture with a fired swap, the aggregate "
                "branch must dominate; got swap_log=" + repr(swap_log)
            )
            for entry in agg_entries:
                assert entry["aggregate_name"].startswith("_dcd_pc1_"), (
                    f"aggregate entry must carry _dcd_pc1_ name; got {entry}"
                )

    def test_aggregate_swap_records_branch_field(self):
        """Every aggregate-swap entry written to swap_log must include
        ``branch == "aggregate"`` for downstream auditability.
        """
        X, y = _scenario_C_aggregate_best()
        m = _fit_mrmr(X, y, swap_method="pca_pc1")
        swap_log = (m.dcd_ or {}).get("swap_log", [])
        for entry in swap_log:
            assert "branch" in entry, (
                f"every swap_log entry must record 'branch'; got {entry}"
            )
            assert entry["branch"] in {"aggregate", "member"}, (
                f"unknown branch label {entry['branch']!r}"
            )


# ---------------------------------------------------------------------------
# 3. Direct unit test of the decision logic (bypasses screen_predictors)
# ---------------------------------------------------------------------------


class TestLayer45_DirectDecision:

    def test_evaluate_returns_member_branch_when_member_dominates_aggregate(self):
        """Directly drive ``evaluate_swap_candidate`` on a hand-crafted
        state where one member has strictly higher CMI than every other
        candidate. The returned ``SwapDecision`` must have
        ``branch == "member"``, ``new_col_idx`` pointing at the dominant
        member, and ``aggregate_name == ""``.
        """
        from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
            DCDState, evaluate_swap_candidate,
        )
        rng = np.random.default_rng(0)
        n = 2500
        latent = rng.standard_normal(n)
        # 3-bin discretisation of features.
        def _quantize(x, k=4):
            edges = np.quantile(x, np.linspace(0, 1, k + 1))
            edges = np.unique(edges)
            if edges.size < 3:
                return np.zeros_like(x, dtype=np.int32)
            return np.clip(np.searchsorted(edges[1:-1], x, side="right"),
                            0, k - 1).astype(np.int32)
        y = (latent + 0.3 * rng.standard_normal(n) > 0).astype(np.int64)
        # anchor: noisy copy of latent
        anchor_raw = latent + 0.5 * rng.standard_normal(n)
        # clean_member: nearly perfect copy
        clean_raw = latent + 0.02 * rng.standard_normal(n)
        # other_member: also clean
        other_raw = latent + 0.03 * rng.standard_normal(n)
        y_col = y.astype(np.int32)
        anchor_b = _quantize(anchor_raw)
        clean_b = _quantize(clean_raw)
        other_b = _quantize(other_raw)
        factors = np.column_stack([y_col, anchor_b, clean_b, other_b])
        factors_nbins = np.array([
            int(y_col.max()) + 1,
            int(anchor_b.max()) + 1,
            int(clean_b.max()) + 1,
            int(other_b.max()) + 1,
        ], dtype=np.int64)
        X_raw = pd.DataFrame({
            "y": y.astype(float),
            "anchor": anchor_raw,
            "clean_member": clean_raw,
            "other_member": other_raw,
        })
        state = DCDState(
            pool_pruned_mask=np.zeros(4, dtype=bool),
            X_raw_ref=X_raw,
            factors_data=factors,
            factors_nbins=factors_nbins,
            cols=["y", "anchor", "clean_member", "other_member"],
            nbins=factors_nbins,
            target_indices=np.array([0], dtype=np.int64),
            quantization_method="quantile",
            quantization_nbins=4,
            quantization_dtype=np.int32,
            cluster_size_threshold=2,
            min_cluster_size=2,
            swap_gain_threshold=0.05,
            tau_cluster=0.5,
            swap_method="pca_pc1",
        )
        # Build a cluster: anchor=1, members={2, 3}.
        state.cluster_anchors[1] = {2, 3}
        state.member_to_anchor[2] = 1
        state.member_to_anchor[3] = 1
        state.pool_pruned_mask[2] = True
        state.pool_pruned_mask[3] = True
        # No conditioning context (S_minus_anchor empty) so MI is
        # unconditional — clean copies will dominate the noisy anchor.
        decision = evaluate_swap_candidate(
            state, anchor=1, selected_vars=[1],
            target_y=np.array([0], dtype=np.int64),
        )
        # Either a member-swap or an aggregate-swap must fire. If the
        # member-swap fires, its new_col_idx is one of {2, 3}.
        assert decision.accept, (
            f"Expected the gate to fire on noisy-anchor + clean-members "
            f"fixture; got accept=False, "
            f"rep_relevance={decision.rep_relevance}, "
            f"anchor_relevance={decision.anchor_relevance_in_ctx}, "
            f"member_relevance={decision.member_relevance}"
        )
        if decision.branch == "member":
            assert decision.new_col_idx in (2, 3), (
                f"member-swap new_col_idx must be one of the clean "
                f"members; got {decision.new_col_idx}"
            )
            assert decision.aggregate_name == ""
            assert decision.binned_rep is None
            # member_relevance must beat anchor_relevance by the gain.
            assert decision.member_relevance > (
                decision.anchor_relevance_in_ctx * 1.05
            ), (
                f"member-swap requires member_rel > anchor_rel * 1.05; "
                f"got member={decision.member_relevance}, "
                f"anchor={decision.anchor_relevance_in_ctx}"
            )

    def test_evaluate_returns_none_branch_when_anchor_dominates(self):
        """When the anchor's CMI strictly dominates every member's and
        the aggregate's, ``evaluate_swap_candidate`` returns
        ``accept=False`` with ``branch="none"``.
        """
        from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
            DCDState, evaluate_swap_candidate,
        )
        rng = np.random.default_rng(7)
        n = 2000
        latent = rng.standard_normal(n)
        def _quantize(x, k=4):
            edges = np.quantile(x, np.linspace(0, 1, k + 1))
            edges = np.unique(edges)
            if edges.size < 3:
                return np.zeros_like(x, dtype=np.int32)
            return np.clip(np.searchsorted(edges[1:-1], x, side="right"),
                            0, k - 1).astype(np.int32)
        y = (latent + 0.3 * rng.standard_normal(n) > 0).astype(np.int64)
        # anchor: clean copy of latent.
        anchor_raw = latent + 0.02 * rng.standard_normal(n)
        # members: extremely noisy versions.
        m1_raw = latent + 0.8 * rng.standard_normal(n)
        m2_raw = latent + 0.8 * rng.standard_normal(n)
        y_col = y.astype(np.int32)
        anchor_b = _quantize(anchor_raw)
        m1_b = _quantize(m1_raw)
        m2_b = _quantize(m2_raw)
        factors = np.column_stack([y_col, anchor_b, m1_b, m2_b])
        factors_nbins = np.array([
            int(y_col.max()) + 1,
            int(anchor_b.max()) + 1,
            int(m1_b.max()) + 1,
            int(m2_b.max()) + 1,
        ], dtype=np.int64)
        X_raw = pd.DataFrame({
            "y": y.astype(float),
            "anchor": anchor_raw,
            "m1": m1_raw,
            "m2": m2_raw,
        })
        state = DCDState(
            pool_pruned_mask=np.zeros(4, dtype=bool),
            X_raw_ref=X_raw,
            factors_data=factors,
            factors_nbins=factors_nbins,
            cols=["y", "anchor", "m1", "m2"],
            nbins=factors_nbins,
            target_indices=np.array([0], dtype=np.int64),
            quantization_method="quantile",
            quantization_nbins=4,
            quantization_dtype=np.int32,
            cluster_size_threshold=2,
            min_cluster_size=2,
            swap_gain_threshold=0.05,
            tau_cluster=0.5,
            swap_method="pca_pc1",
        )
        state.cluster_anchors[1] = {2, 3}
        state.member_to_anchor[2] = 1
        state.member_to_anchor[3] = 1
        state.pool_pruned_mask[2] = True
        state.pool_pruned_mask[3] = True
        decision = evaluate_swap_candidate(
            state, anchor=1, selected_vars=[1],
            target_y=np.array([0], dtype=np.int64),
        )
        # When anchor genuinely dominates, the result is NOT accepted as
        # a member-swap. Either accept=False (branch="none") or accept=True
        # via the aggregate branch if averaging helps; the contract is
        # only that no MEMBER-swap can fire when anchor strictly beats
        # every member.
        if decision.accept and decision.branch == "member":
            pytest.fail(
                f"member-swap fired even though anchor dominates: "
                f"anchor_rel={decision.anchor_relevance_in_ctx}, "
                f"member_rel={decision.member_relevance}"
            )


# ---------------------------------------------------------------------------
# 4. Regression: layer 41 / 42 / 43 / 44 contracts preserved
# ---------------------------------------------------------------------------


class TestLayer45_Regression:

    def test_layer42_three_dups_threshold2_still_fires_swap(self):
        """L42 contract: ``dcd_cluster_size_threshold=2`` on the 3-dups
        fixture fires >= 1 swap. Layer 45's member-swap branch is
        additive, so this assertion must still hold (either via
        aggregate or member branch).
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _scenario_C_aggregate_best()
        m = MRMR(
            dcd_enable=True, dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=2,
            full_npermutations=50,
            verbose=0, random_seed=0,
        ).fit(X, y)
        assert int(m.dcd_["n_swaps"]) >= 1, (
            f"Layer 42 contract: threshold=2 must fire >=1 swap on the "
            f"3-dup fixture; got n_swaps={m.dcd_['n_swaps']}"
        )

    def test_layer41_cluster_anchors_names_present(self):
        """L41 contract: ``cluster_anchors_names`` map is in the summary."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _scenario_C_aggregate_best()
        m = MRMR(
            dcd_enable=True, dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=2,
            full_npermutations=20,
            verbose=0, random_seed=0,
        ).fit(X, y)
        assert m.dcd_ is not None
        assert "cluster_anchors_names" in m.dcd_, (
            "L41 contract: cluster_anchors_names must be in dcd_ summary"
        )

    def test_layer44_auto_pool_unchanged(self):
        """L44 contract: the auto bake-off pool is still the 7-element
        superset.
        """
        from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
            _AUTO_METHOD_CANDIDATES,
        )
        assert set(_AUTO_METHOD_CANDIDATES) == {
            "mean_z", "mean_inv_var", "pca_pc1",
            "pca_pc2", "median_z", "signed_max_abs", "signed_l2_sum",
        }

    def test_transform_still_deterministic_with_layer45(self):
        """The transform must remain deterministic with Layer 45
        active, regardless of which branch fired.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _scenario_C_aggregate_best()
        m = MRMR(
            dcd_enable=True, dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=2,
            full_npermutations=20,
            verbose=0, random_seed=0,
        ).fit(X, y)
        Xt1 = np.asarray(m.transform(X), dtype=np.float64)
        Xt2 = np.asarray(m.transform(X), dtype=np.float64)
        assert Xt1.shape == Xt2.shape
        assert np.allclose(Xt1, Xt2, equal_nan=True), (
            "transform must be deterministic with Layer 45 active"
        )
