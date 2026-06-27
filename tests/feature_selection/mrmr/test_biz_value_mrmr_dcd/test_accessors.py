"""DCD consolidation: Layer 41 biz_value: DCD cluster-membership accessor (self-describing summary).

Consolidated verbatim from test_biz_value_mrmr_layer41.py + test_biz_value_mrmr_layer42.py (per audit finding test_code_quality-16).
"""
from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Fixtures shared across the layer (clustered + StabilityMRMR-style)
# ---------------------------------------------------------------------------


def _strong_cluster_frame(n: int = 1500, n_members: int = 5, seed: int = 0):
    """Latent + n_members noisy copies + 2 noise columns. Pairwise SU
    among members is well above 0.7 (the default tau_cluster), so DCD
    forms a single cluster of size ``n_members`` reliably.
    """
    rng = np.random.default_rng(int(seed))
    latent = rng.standard_normal(n)
    cols: dict = {}
    for k in range(n_members):
        cols[f"clu_{k}"] = latent + 0.05 * (k + 1) * rng.standard_normal(n)
    cols["noise_0"] = rng.standard_normal(n)
    cols["noise_1"] = rng.standard_normal(n)
    X = pd.DataFrame(cols)
    y = pd.Series((latent > 0).astype(np.int64), name="y")
    return X, y


# ---------------------------------------------------------------------------
# C1, C2, C3, C4: the Layer 41 contracts on the new accessors.
# ---------------------------------------------------------------------------


class TestClusterMembersAccessor:

    def test_C1_cluster_members_populated_when_dcd_finds_cluster(self):
        """C1: a DCD fit with at least one detected cluster exposes
        ``cluster_members_`` keyed by valid column names; the member
        lists are non-empty and themselves valid column names.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _strong_cluster_frame()
        m = MRMR(dcd_enable=True, dcd_tau_cluster=0.5, verbose=0,
                  random_seed=0).fit(X, y)
        cm = m.cluster_members_
        assert cm is not None, (
            "cluster_members_ must be a dict (not None) when DCD ran and "
            "detected at least one cluster."
        )
        assert isinstance(cm, dict), (
            f"cluster_members_ should be a dict; got {type(cm).__name__}"
        )
        # At least one cluster found on this collinear fixture.
        assert len(cm) >= 1, (
            f"DCD should detect at least one cluster on the strong-cluster "
            f"fixture; got cluster_members_={cm}"
        )
        # Layer 41: anchor keys may be raw feature names OR engineered
        # post-swap aggregate names like ``_dcd_pc1_*``. Members must
        # always resolve against the fitted column universe (raw cols +
        # engineered aggregates).
        fitted_cols = set(X.columns)
        engineered_anchors = set()
        for anchor_name, members in cm.items():
            assert isinstance(anchor_name, str) and len(anchor_name) > 0
            assert isinstance(members, list) and len(members) >= 1
            # Anchor: either a raw column or an engineered DCD-PC1
            # aggregate. Track the latter for member resolution below.
            if anchor_name not in fitted_cols:
                assert anchor_name.startswith("_dcd_pc1_") or anchor_name.startswith("col_"), (
                    f"Unknown anchor name {anchor_name!r}: not in raw "
                    f"columns and not a known engineered prefix."
                )
                engineered_anchors.add(anchor_name)
            for member_name in members:
                assert isinstance(member_name, str) and len(member_name) > 0
                # Member must resolve against raw cols, post-swap
                # aggregates seen so far, or the col_<idx> defensive
                # fallback (which would itself indicate a bug worth
                # flagging if it ever fires on a normal fit).
                assert (
                    member_name in fitted_cols
                    or member_name in engineered_anchors
                    or member_name.startswith("_dcd_pc1_")
                    or member_name.startswith("col_")
                    or member_name.startswith("targ_")  # cat-FE encoded targets
                ), (
                    f"Member {member_name!r} does not resolve against raw "
                    f"columns nor known engineered prefixes; cluster_members_ "
                    f"map is internally inconsistent."
                )

    def test_C2_cluster_members_is_none_when_dcd_disabled(self):
        """C2: opt-out path. ``MRMR(dcd_enable=False)`` must set
        ``cluster_members_`` to ``None`` -- not raise AttributeError, not
        leave it as an empty dict (the empty-dict case is reserved for
        "DCD ran but found 0 clusters", which is meaningfully different).
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _strong_cluster_frame()
        m = MRMR(dcd_enable=False, verbose=0, random_seed=0).fit(X, y)
        assert hasattr(m, "cluster_members_"), (
            "Even DCD-disabled fits must set cluster_members_ to None "
            "(attribute-completeness, mirrors dcd_ contract)."
        )
        assert m.cluster_members_ is None, (
            f"cluster_members_ must be None when DCD is disabled; got "
            f"{m.cluster_members_!r}"
        )

    def test_C3_name_map_matches_integer_map_one_to_one(self):
        """C3: ``cluster_members_`` is a faithful rename of the integer
        ``dcd_['cluster_anchors']`` map. Each integer key -> name lookup
        must round-trip exactly. This catches a class of subtle bugs
        where the name resolution drifts from the integer columns (e.g.
        if post-swap state stays partly stale).
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _strong_cluster_frame()
        m = MRMR(dcd_enable=True, dcd_tau_cluster=0.5, verbose=0,
                  random_seed=0).fit(X, y)
        int_anchors = m.dcd_["cluster_anchors"]
        name_anchors = m.dcd_["cluster_anchors_names"]
        # cluster_members_ is the same content as cluster_anchors_names.
        assert m.cluster_members_ == name_anchors, (
            "cluster_members_ must equal dcd_['cluster_anchors_names'] "
            "exactly (both are derived from the same source)."
        )
        # Cardinalities must match.
        assert len(int_anchors) == len(name_anchors), (
            f"int-map len ({len(int_anchors)}) != name-map len "
            f"({len(name_anchors)}); the two views of DCD's cluster state "
            f"have drifted."
        )
        # Length of each member list must match.
        int_sizes = sorted(len(v) for v in int_anchors.values())
        name_sizes = sorted(len(v) for v in name_anchors.values())
        assert int_sizes == name_sizes, (
            f"member-list sizes diverged: int {int_sizes}, names {name_sizes}"
        )

    def test_C4_cluster_diagnostics_consistent_with_tau(self):
        """C4: every reported cluster has ``min_pair_su >= tau_cluster``
        (modulo a tiny float tolerance). This is precisely the
        membership rule in ``discover_cluster_members``: a candidate is
        added only when ``pair_su(member, anchor) > tau_cluster``. So
        every member-anchor SU value (which dominates ``min_pair_su``
        when n_pairs_evaluated >= 1) must be at least ``tau_cluster``.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _strong_cluster_frame()
        tau = 0.5
        m = MRMR(dcd_enable=True, dcd_tau_cluster=tau, verbose=0,
                  random_seed=0).fit(X, y)
        diag = m.dcd_.get("cluster_diagnostics", {})
        eff_tau = float(m.dcd_.get("tau_cluster", tau))
        assert diag, (
            "cluster_diagnostics must be populated on this fixture; got "
            f"empty/missing diag={diag!r}"
        )
        for anchor_name, info in diag.items():
            assert info["size"] >= 2, (
                f"cluster {anchor_name!r} reports size {info['size']} < 2; "
                f"a one-element 'cluster' should not be reported."
            )
            assert info["n_pairs_evaluated"] >= 1, (
                f"cluster {anchor_name!r} has no SU pairs evaluated"
            )
            assert info["min_pair_su"] is not None, (
                f"cluster {anchor_name!r} has no min_pair_su"
            )
            # Membership rule: SU(member, anchor) > tau. So min SU >= tau,
            # with a small float tolerance.
            assert info["min_pair_su"] >= eff_tau - 1e-6, (
                f"cluster {anchor_name!r} reports min_pair_su="
                f"{info['min_pair_su']:.4f} below tau_cluster={eff_tau:.4f}; "
                f"a member sneaked into the cluster below the membership "
                f"threshold -- discover_cluster_members invariant broken."
            )
            # Sanity: max >= min, mean between.
            assert info["max_pair_su"] >= info["min_pair_su"]
            assert info["min_pair_su"] <= info["mean_pair_su"] <= info["max_pair_su"]

    def test_C4b_tau_cluster_surfaced_in_summary(self):
        """C4 bis: ``dcd_['tau_cluster']`` is the effective tau (after any
        kernel_tuning_cache override). Lets a deserialised summary stand
        alone without needing the original MRMR ctor args.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _strong_cluster_frame()
        m = MRMR(dcd_enable=True, dcd_tau_cluster=0.55, verbose=0,
                  random_seed=0).fit(X, y)
        eff = m.dcd_.get("tau_cluster")
        # User-supplied non-default value bypasses the cache lookup, so
        # the surfaced tau matches the ctor argument exactly.
        assert eff is not None
        assert abs(float(eff) - 0.55) < 1e-9, (
            f"tau_cluster surfaced ({eff}) does not match user-supplied "
            f"non-default value (0.55); cache override fired when it "
            f"shouldn't have."
        )


# ---------------------------------------------------------------------------
# C5: NO-REGRESSION vs Layer 12 stability discrimination.
# ---------------------------------------------------------------------------


def _layer12_drift_frame(seed: int = 0):
    """Compact recreation of the Layer 12 drift fixture: stable_x +
    flaky + 5 noise columns. We pin the same 'stable > flaky'
    discrimination property at the per-fit level (single-fit MRMR, not
    StabilityMRMR -- this is a Layer 41 no-regression spot-check, not a
    full Layer 12 re-run).
    """
    rng = np.random.default_rng(int(seed))
    n = 1500
    stable_x = rng.standard_normal(n)
    flaky = rng.standard_normal(n)
    noise = {f"noise_{k}": rng.standard_normal(n) for k in range(5)}
    cols = {"stable_x": stable_x, "flaky": flaky, **noise}
    X = pd.DataFrame(cols)
    # y has strong stable signal + weak flaky signal.
    logit = 1.2 * stable_x + 0.3 * flaky + 0.5 * rng.standard_normal(n)
    y = pd.Series((logit > 0).astype(np.int64), name="y")
    return X, y


class TestNoRegressionLayer12Stability:

    def test_C5_stable_outranks_flaky_with_layer41_additions(self):
        """C5: on the simplified drift fixture, the strong ``stable_x`` signal must
        outrank the weak ``flaky`` signal in the single-fit MRMR selection -- the
        Layer 12 discrimination contract. The Layer 41 summary additions must not
        change this ordering.

        The contract is on the SIGNAL, not the literal raw column: FE may keep the
        stable signal as the raw ``stable_x`` OR re-express it via stable-derived
        engineered children (``stable_x__relu_*``, ``sub(neg(stable_x),log(flaky))``).
        On this fixture (y = 1.2*stable_x + 0.3*flaky + noise) the engineered support
        is measurably BETTER at the deployed logistic objective than the raw alone
        (CV-AUC ~0.949 vs raw stable_x ~0.932; the engineered ``stable_x__relu_gt``
        has |corr| ~0.87 with the logit, ``flaky`` only ~0.22), so pinning the literal
        raw is over-strict. The discrimination contract is that every selected feature
        CARRIES stable's signal and that flaky never displaces it.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _layer12_drift_frame(seed=0)
        m = MRMR(dcd_enable=True, verbose=0, random_seed=0).fit(X, y)
        sup = list(m.get_feature_names_out())

        def _carries_stable(name: str) -> bool:
            return "stable_x" in str(name)

        # The stable signal must be present, as the raw OR a stable-derived child.
        assert any(_carries_stable(c) for c in sup), (
            f"stable signal must survive single-fit MRMR (raw or engineered); got support={sup}"
        )

        # flaky must never DISPLACE stable. The displacement contract is on the GREEDY SELECTION RANK
        # (``support_rank`` in the provenance frame: 0 = first MRMR pick), NOT on the positional order of
        # ``get_feature_names_out()`` -- which lists ALL emitted columns (greedy picks PLUS engineered
        # children / retained raw operands with ``support_rank == -1``) in provenance-row order, so a raw
        # operand that was kept around (e.g. ``flaky`` as an operand of the selected ``sub(prewarp(stable_x),
        # prewarp(flaky))`` compound, rank -1) can appear at list position 0 without being a discriminative
        # pick at all. Rank by ``support_rank`` so the test pins selection precedence, not emission order.
        prov = m.fe_provenance_
        ranked = prov[prov["support_rank"] >= 0].sort_values("support_rank", kind="stable")
        picked = [str(nm) for nm in ranked["feature_name"].tolist()]
        assert picked, f"single-fit MRMR must greedily select at least one feature; provenance={prov.to_string()}"
        # Every greedily-PICKED feature must carry stable before any flaky-only pick displaces it.
        flaky_only_picks = [i for i, c in enumerate(picked) if ("flaky" in c and not _carries_stable(c))]
        if flaky_only_picks:
            first_stable = min((i for i, c in enumerate(picked) if _carries_stable(c)), default=len(picked))
            assert first_stable < min(flaky_only_picks), (
                f"stable signal must be greedily selected before a flaky-only feature; got pick order={picked}"
            )
        else:
            # No flaky-only pick at all -> the strongest contract (stable's signal dominates) holds outright.
            assert any(_carries_stable(c) for c in picked), (
                f"the greedy MRMR picks must carry stable's signal; got pick order={picked}"
            )

    def test_C5b_pickle_round_trip_preserves_cluster_members(self):
        """C5 bis: ``cluster_members_`` survives pickle/unpickle so the
        new accessor is a real fitted-attribute citizen (compatible with
        joblib persistence used by production training pipelines).
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _strong_cluster_frame()
        m = MRMR(dcd_enable=True, dcd_tau_cluster=0.5, verbose=0,
                  random_seed=0).fit(X, y)
        # Round-trip.
        m_re = pickle.loads(pickle.dumps(m))
        assert m_re.cluster_members_ == m.cluster_members_, (
            "cluster_members_ did not survive pickle round-trip; the "
            "attribute is not a real first-class fitted citizen."
        )
        assert m_re.dcd_["cluster_anchors_names"] == m.dcd_["cluster_anchors_names"]
        assert m_re.dcd_["cluster_diagnostics"] == m.dcd_["cluster_diagnostics"]
        assert m_re.dcd_["tau_cluster"] == m.dcd_["tau_cluster"]


# ---------------------------------------------------------------------------
# C6: NO-REGRESSION vs Layer 27 stress (hybrid orth pair adversarial).
# ---------------------------------------------------------------------------


class TestNoRegressionLayer27Adversarial:

    def test_C6_layer41_additions_do_not_perturb_layer27_no_noise_cross(self):
        """C6: the Layer 27 adversarial contract is that hybrid-orth-pair
        does not select a noise-noise cross. Layer 41 only adds summary
        keys -- it must not move the support_ on this fixture. Smaller
        n than the production Layer 27 test to keep this fast.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        rng = np.random.default_rng(42)
        n = 1000
        x_real = rng.standard_normal(n)
        noises = {f"noise_{k}": rng.standard_normal(n) for k in range(4)}
        X = pd.DataFrame({"x_real": x_real, **noises})
        y = pd.Series((x_real ** 2 - 1.0 + 0.2 * rng.standard_normal(n) > 0).astype(int))
        m = MRMR(
            verbose=0, random_seed=0,
            fe_hybrid_orth_pair_enable=True,
            fe_hybrid_orth_enable=True,
            fe_hybrid_orth_degrees=(2,),
            fe_hybrid_orth_pair_max_degree=2,
            interactions_max_order=1, fe_max_steps=0,
        ).fit(X, y)
        sup = list(m.get_feature_names_out())
        # No noise-noise cross of the form noise_i__N__M__noise_j or
        # noise_i_X_noise_j (the Layer 27 forbidden pattern).
        for name in sup:
            if name == "x_real" or "x_real" in name:
                continue
            # Be permissive about exact naming -- the prohibition is "no
            # cross-term combining TWO noise columns".
            noise_hits = sum(1 for k in range(4) if f"noise_{k}" in name)
            assert noise_hits < 2, (
                f"Layer 27 regression: noise-noise cross {name!r} appeared "
                f"in support after Layer 41 additions."
            )


# ---------------------------------------------------------------------------
# C7: NO-REGRESSION vs Layer 35 kitchen-sink.
# ---------------------------------------------------------------------------


class TestNoRegressionLayer35KitchenSink:

    def test_C7_layer35_fit_completes_with_layer41_additions(self):
        """C7: minimal Layer 35 spot-check. The kitchen-sink config from
        Layer 35 enables multiple FE mechanisms simultaneously; Layer 41
        only adds summary keys, so the fit must complete and expose
        ``cluster_members_`` as either ``None`` (DCD turned off in this
        config) or a dict. No support_ assertion here -- Layer 35 owns
        that.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        rng = np.random.default_rng(0)
        n = 800
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        X = pd.DataFrame({
            "x1": x1,
            "x2": x2,
            "noise_0": rng.standard_normal(n),
            "noise_1": rng.standard_normal(n),
        })
        y = pd.Series((x1 ** 2 + x2 - 0.5 + 0.1 * rng.standard_normal(n) > 0).astype(int))
        m = MRMR(
            verbose=0, random_seed=0,
            interactions_max_order=1, fe_max_steps=0,
            dcd_enable=True,
            fe_hybrid_orth_enable=True,
            fe_hybrid_orth_degrees=(2,),
            fe_hybrid_orth_top_k=3,
        ).fit(X, y)
        # Layer 41 attribute must be present.
        assert hasattr(m, "cluster_members_")
        # It's either None (no clusters) or a valid dict.
        cm = m.cluster_members_
        assert cm is None or isinstance(cm, dict)
        # If it's a dict, the diagnostic sub-keys must be present
        # (lets future schema migrations notice missing fields).
        if isinstance(cm, dict) and cm:
            diag = m.dcd_.get("cluster_diagnostics", {})
            for anchor_name in cm:
                assert anchor_name in diag, (
                    f"Anchor {anchor_name!r} appears in cluster_members_ "
                    f"but not in cluster_diagnostics; the two views drifted."
                )


# ---------------------------------------------------------------------------
# Bonus: cross-tau monotonicity sanity-check for the new diagnostic.
# ---------------------------------------------------------------------------


class TestClusterDiagnosticsMonotonicity:

    def test_higher_tau_yields_higher_min_pair_su_when_clusters_form(self):
        """A natural consequence of the membership rule: raising
        ``tau_cluster`` admits only tighter clusters, so ``min_pair_su``
        across all reported clusters is monotone non-decreasing in tau.

        This is a SANITY test on the new diagnostic, not a property of
        the MRMR pipeline; if a future change broke the diagnostic by
        e.g. using SU(member, member) instead of SU(member, anchor)
        without intent, this would fire.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _strong_cluster_frame(n=2000, n_members=5, seed=0)
        mins = {}
        for tau in (0.4, 0.6, 0.75):
            m = MRMR(dcd_enable=True, dcd_tau_cluster=tau, verbose=0,
                      random_seed=0).fit(X, y)
            diag = m.dcd_.get("cluster_diagnostics", {})
            cluster_mins = [info["min_pair_su"] for info in diag.values()
                             if info["min_pair_su"] is not None]
            if cluster_mins:
                mins[tau] = min(cluster_mins)
        # Only assert when we have observations at 2+ taus.
        observed_taus = sorted(mins.keys())
        if len(observed_taus) >= 2:
            for i in range(1, len(observed_taus)):
                t_lo, t_hi = observed_taus[i - 1], observed_taus[i]
                # Allow tiny slack for FP noise.
                assert mins[t_hi] >= mins[t_lo] - 1e-6, (
                    f"min_pair_su not monotone in tau: tau={t_lo}->"
                    f"min={mins[t_lo]:.4f}, tau={t_hi}->min={mins[t_hi]:.4f}"
                )


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
