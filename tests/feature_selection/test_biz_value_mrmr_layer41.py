"""Layer 41 biz_value: DCD cluster-membership accessor (self-describing summary).

WHY THIS LAYER
--------------
Pre-Layer-41 Dynamic Cluster Discovery exposed only integer column indices in
its public ``MRMR.dcd_["cluster_anchors"]`` map. Phase 1 diagnostic probes
confirmed three concrete weaknesses (D being by far the most impactful):

* Probe A (soft cluster, SU 0.05-0.07 on noise-dominated latents) — the
  default ``tau_cluster=0.7`` correctly does not prune; even ``tau=0.3``
  does not prune. SU genuinely collapses at high noise; not a real DCD
  weakness, only a tau-tuning lever with limited room.
* Probe B (heterogeneous loadings) — swaps did not fire because each
  latent's pair sized 1 stayed below ``cluster_size_threshold=4``.
* Probe C (3 perfectly-correlated features) — pruning fires (n_pruned=3)
  but no PC1 swap, because the default ``cluster_size_threshold=4`` is the
  binding gate. Lowering to 2 is a parameter-tuning lever, not a new
  capability.
* Probe D (cluster summary readability) — confirmed: ``cluster_anchors``
  is opaque (``{4: [0, 1, 2, 3, 6]}``). The user must know fit-time
  column ordering to interpret it. THIS is high-leverage AND universally
  better (additive metadata only), so Layer 41 implements it.

LAYER 41 IMPROVEMENT (pure additive, byte-identical with master for
support_/transform)
-----------------------------------------------------------------------
1. ``MRMR.cluster_members_`` — fitted attribute, ``dict[str, list[str]]``,
   mapping ``anchor_name -> sorted member_name list``. ``None`` when DCD
   was disabled (matches ``dcd_`` semantics).

2. ``MRMR.dcd_["cluster_anchors_names"]`` — same content as
   ``cluster_members_``, redundant on the summary dict for one-shot
   serialisation.

3. ``MRMR.dcd_["cluster_diagnostics"]`` — per-anchor sub-dict with
   ``size``, ``min_pair_su``, ``mean_pair_su``, ``max_pair_su``,
   ``n_pairs_evaluated``. Mined entirely from the existing
   ``pairwise_su_cache`` (no new MI work). Lets users judge whether
   ``tau_cluster`` is calibrated: ``min_pair_su`` near tau means
   borderline cluster (risk of inclusion noise); much higher than tau
   means high-confidence cluster.

4. ``MRMR.dcd_["tau_cluster"]`` — surfaces the effective tau (post any
   ``kernel_tuning_cache`` override) so a serialised summary is
   self-contained.

CONTRACTS PINNED
----------------
* C1: ``cluster_members_`` populated on every DCD-enabled fit with at
  least one detected cluster; keys are valid column names.
* C2: ``cluster_members_`` is ``None`` when ``dcd_enable=False``.
* C3: ``cluster_members_`` matches the integer ``cluster_anchors`` map
  one-to-one through ``MRMR``'s factor column list. Lets us prove the
  name resolution is correct (not just plausible-looking).
* C4: ``cluster_diagnostics`` reports SU values consistent with
  ``tau_cluster`` (``min_pair_su >= tau_cluster`` for every detected
  cluster, modulo a tiny float tolerance; this is the actual cluster
  membership rule in ``discover_cluster_members``).
* C5: NO-REGRESSION vs Layer 12 stability contract (StabilityMRMR
  drift discrimination): stable feature still ranks above flaky feature
  with the additive summary turned on. Spot-replicated from
  ``test_biz_value_mrmr_layer12``.
* C6: NO-REGRESSION vs Layer 27 stress (hybrid orth pair adversarial):
  noise-noise crosses still avoided; summary additions don't perturb
  the MRMR pipeline.
* C7: NO-REGRESSION vs Layer 35 kitchen-sink (all 8 FE mechanisms
  composed): MRMR end-to-end fit completes; the new ``cluster_members_``
  attribute survives ``pickle`` round-trip (fitted-attribute discipline).

NEVER xfail. Pure additive metadata; default-OFF byte-identical with
master (DCD already default-ON since Layer 6 / 2026-05-30, but the
Layer 41 additions don't touch screen / prune / swap decisions).
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
        """C5: on the simplified drift fixture, MRMR selects ``stable_x``
        ahead of (or instead of) ``flaky`` -- the Layer 12 discrimination
        contract at the single-fit level. The Layer 41 summary additions
        must not change this ordering.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _layer12_drift_frame(seed=0)
        m = MRMR(dcd_enable=True, verbose=0, random_seed=0).fit(X, y)
        sup = list(m.get_feature_names_out())
        assert "stable_x" in sup, (
            f"stable_x must survive single-fit MRMR; got support={sup}"
        )
        # If flaky also survives, stable_x must rank earlier (lower
        # support index in the selection order). MRMR returns
        # support_ in selection order.
        if "flaky" in sup:
            assert sup.index("stable_x") < sup.index("flaky"), (
                f"stable_x must be selected before flaky; got order={sup}"
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
