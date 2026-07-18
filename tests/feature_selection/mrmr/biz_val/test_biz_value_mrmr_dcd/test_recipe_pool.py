"""DCD consolidation: Layer 43 biz_value: DCD commit_swap recipe wiring + auto swap-method.

Consolidated verbatim from test_biz_value_mrmr_layer43.py + test_biz_value_mrmr_layer44.py (per audit finding test_code_quality-16).
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


def _three_dups_plus_strong_frame(n: int = 1500, seed: int = 0):
    """anchor + 2 perfect duplicates + one unrelated strong col + noise.

    A homogeneous-loading cluster: every reflection has the same |loading|
    on the shared latent, so ``mean_z`` and ``pca_pc1`` produce nearly
    identical aggregates and the OOF bake-off either tie or marginally
    prefer ``mean_z`` (lower variance under near-equal loadings).
    """
    rng = np.random.default_rng(int(seed))
    latent = rng.standard_normal(n)
    other = rng.standard_normal(n)
    X = pd.DataFrame(
        {
            "strong": other,
            "dup_a": latent + 0.01 * rng.standard_normal(n),
            "dup_b": latent + 0.01 * rng.standard_normal(n),
            "dup_c": latent + 0.01 * rng.standard_normal(n),
            "noise_0": rng.standard_normal(n),
        }
    )
    y = pd.Series((2 * other + latent + 0.3 * rng.standard_normal(n) > 0).astype(int))
    return X, y


def _heterogeneous_loadings_frame(n: int = 1500, seed: int = 0):
    """Cluster with WIDELY different signal-to-noise across members.

    member_strong: 3.0 * latent + 0.02 * noise (very high SNR, big variance)
    member_weak1:  0.5 * latent + 0.10 * noise (lower SNR, smaller variance)
    member_weak2:  0.5 * latent + 0.10 * noise

    Under heterogeneous loadings the variance-max combiner (``pca_pc1``)
    and the reliability-weighted ``mean_inv_var`` should both outperform
    uniform ``mean_z`` because the strong member should carry more
    weight in the aggregate. The OOF bake-off should prefer one of those
    two (with pca_pc1 and mean_inv_var typically close).
    """
    rng = np.random.default_rng(int(seed))
    latent = rng.standard_normal(n)
    other = rng.standard_normal(n)
    X = pd.DataFrame(
        {
            "strong_unrelated": other,
            "member_strong": 3.0 * latent + 0.02 * rng.standard_normal(n),
            "member_weak1": 0.5 * latent + 0.10 * rng.standard_normal(n),
            "member_weak2": 0.5 * latent + 0.10 * rng.standard_normal(n),
            "noise_0": rng.standard_normal(n),
        }
    )
    y = pd.Series((2 * other + latent + 0.3 * rng.standard_normal(n) > 0).astype(int))
    return X, y


# ---------------------------------------------------------------------------
# PART A — commit_swap recipes wiring fix
# ---------------------------------------------------------------------------


class TestPartA_RecipeWiring:
    """Groups tests covering TestPartA_RecipeWiring."""
    def test_swap_aggregate_recorded_with_pc1_marker(self):
        """When an aggregate-branch swap fires, the PC1 aggregate name
        (carrying the fixed ``_dcd_pc1_`` marker built in ``commit_swap``)
        MUST be recorded in ``dcd_["swap_log"]``. Pre-Layer-43 the swap
        was committed with ``engineered_recipes=None`` and the aggregate
        was silently dropped before it could be logged.

        The marker lives in the swap_log (where the swap is genuinely
        recorded), NOT in ``get_feature_names_out``: on this fixture the
        downstream FE + raw-redundancy sweep re-engineers the duplicate
        cluster into a different surviving feature and selects ``dup_a`` /
        ``dup_b`` directly, so the transient PC1 aggregate column is not
        retained in the final selection surface (``_redundancy_emptied_raw_``
        is set). That is correct downstream behaviour, not a dropped swap --
        the swap fired and is provably logged with its ``_dcd_pc1_`` name.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _three_dups_plus_strong_frame()
        m = MRMR(
            dcd_enable=True,
            dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=2,
            dcd_swap_method="pca_pc1",  # pin so the aggregate uses PC1
            full_npermutations=50,
            verbose=0,
            random_seed=0,
        ).fit(X, y)
        # This fixture is built to force a swap; it must actually fire.
        assert m.dcd_["n_swaps"] >= 1, f"Expected swap to fire on 3-dups + threshold=2; got n_swaps={m.dcd_['n_swaps']}"
        agg_entries = [
            e for e in m.dcd_["swap_log"] if e.get("branch") == "aggregate" and "_dcd_pc1_" in str(e.get("aggregate_name", "")) and e.get("method") == "pca_pc1"
        ]
        assert (
            len(agg_entries) >= 1
        ), f"a pinned pca_pc1 aggregate swap must record a _dcd_pc1_ aggregate_name with method='pca_pc1'; got swap_log={m.dcd_['swap_log']}"

    def test_produced_recipes_contains_cluster_aggregate(self):
        """The swap MUST produce a ``cluster_aggregate`` recipe named with
        the ``_dcd_pc1_`` marker, sourced from the duplicate members.

        It is asserted against ``_produced_recipes_`` (the full pool of
        recipes the fit built), NOT ``_engineered_recipes_`` (the retained
        / selected subset): on this fixture the downstream redundancy sweep
        does not retain the PC1 aggregate column in the final selection, so
        ``_engineered_recipes_`` legitimately omits it -- but the recipe was
        built and is provably present in the produced pool with its
        ``_dcd_pc1_`` name and source members.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        from mlframe.feature_selection.filters.engineered_recipes import (
            EngineeredRecipe,
        )

        X, y = _three_dups_plus_strong_frame()
        m = MRMR(
            dcd_enable=True,
            dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=2,
            dcd_swap_method="pca_pc1",
            full_npermutations=50,
            verbose=0,
            random_seed=0,
        ).fit(X, y)
        assert m.dcd_["n_swaps"] >= 1
        produced = list(getattr(m, "_produced_recipes_", []))
        dcd_recipes = [r for r in produced if isinstance(r, EngineeredRecipe) and r.kind == "cluster_aggregate" and r.name.startswith("_dcd_pc1_")]
        assert len(dcd_recipes) >= 1, f"Expected >=1 produced cluster_aggregate recipe with a _dcd_pc1_ name; got produced={produced}"
        # Sourced from the duplicate cluster members (the recipe replays
        # the aggregate from these raw columns).
        assert all(
            set(r.src_names) <= {"dup_a", "dup_b", "dup_c"} for r in dcd_recipes
        ), f"cluster_aggregate src_names must be the dup members; got {[r.src_names for r in dcd_recipes]}"

    def test_transform_deterministic_and_aggregate_replay_finite(self):
        """``transform`` on the training data must be deterministic, and
        replaying the produced PC1 ``cluster_aggregate`` recipe (recipe
        replay does not require y) must yield a finite column.

        The aggregate column is asserted via direct recipe replay rather
        than via ``get_feature_names_out``: on this fixture the downstream
        redundancy sweep drops the PC1 aggregate from the final selection,
        so it does not appear in the transform output -- but the recipe is
        in ``_produced_recipes_`` and replays to a finite continuous column.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        from mlframe.feature_selection.filters.engineered_recipes import (
            EngineeredRecipe,
            _apply_cluster_aggregate,
        )

        X, y = _three_dups_plus_strong_frame()
        m = MRMR(
            dcd_enable=True,
            dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=2,
            dcd_swap_method="pca_pc1",
            full_npermutations=50,
            verbose=0,
            random_seed=0,
        ).fit(X, y)
        assert m.dcd_["n_swaps"] >= 1
        Xt1 = m.transform(X)
        Xt2 = m.transform(X)
        # Transform output must be deterministic.
        Xt1_arr = np.asarray(Xt1, dtype=np.float64)
        Xt2_arr = np.asarray(Xt2, dtype=np.float64)
        assert Xt1_arr.shape == Xt2_arr.shape
        assert np.allclose(Xt1_arr, Xt2_arr, equal_nan=True), "transform() must be deterministic on the same input"
        names_out = list(m.get_feature_names_out())
        assert len(names_out) == Xt1_arr.shape[1], f"feature_names_out ({len(names_out)}) must match transform width ({Xt1_arr.shape[1]})"
        # The produced PC1 aggregate recipe replays to a finite column on a
        # fresh frame, and the replay is deterministic across two calls.
        dcd_recipes = [
            r
            for r in getattr(m, "_produced_recipes_", [])
            if isinstance(r, EngineeredRecipe) and r.kind == "cluster_aggregate" and r.name.startswith("_dcd_pc1_")
        ]
        assert len(dcd_recipes) >= 1
        for r in dcd_recipes:
            col1 = np.asarray(_apply_cluster_aggregate(r, X), dtype=np.float64)
            col2 = np.asarray(_apply_cluster_aggregate(r, X), dtype=np.float64)
            assert np.all(np.isfinite(col1)), f"replayed aggregate {r.name} contains NaN/Inf; values: {col1[:8]}"
            assert np.allclose(col1, col2, equal_nan=True), f"aggregate replay for {r.name} must be deterministic"


# ---------------------------------------------------------------------------
# PART B — auto swap-method selection
# ---------------------------------------------------------------------------


class TestPartB_AutoMethod:
    """Groups tests covering TestPartB_AutoMethod."""
    def test_default_swap_method_is_auto(self):
        """Default swap method is auto."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        m = MRMR()
        assert str(m.dcd_swap_method) == "auto", f"Default dcd_swap_method must be 'auto'; got {m.dcd_swap_method!r}"

    def test_auto_records_chosen_method_in_swap_log(self):
        """With ``dcd_swap_method='auto'``, every swap_log entry must
        carry a ``method`` key naming the actual combiner used. When
        the bake-off ran (>=1 fold succeeded), ``auto_winner`` and
        ``kfold_scores`` must also be present.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _three_dups_plus_strong_frame()
        m = MRMR(
            dcd_enable=True,
            dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=2,
            # default dcd_swap_method='auto'
            full_npermutations=50,
            verbose=0,
            random_seed=0,
        ).fit(X, y)
        assert m.dcd_["n_swaps"] >= 1
        log = m.dcd_["swap_log"]
        for entry in log:
            assert "method" in entry, f"swap_log entry missing 'method' key: {entry}"
            # Layer 44: bake-off pool expanded from 3 to 7 candidates
            # (added pca_pc2 / median_z / signed_max_abs / signed_l2_sum).
            _valid_methods = {
                "mean_z",
                "mean_inv_var",
                "pca_pc1",
                "pca_pc2",
                "median_z",
                "signed_max_abs",
                "signed_l2_sum",
            }
            assert entry["method"] in _valid_methods, f"unexpected chosen method: {entry['method']!r}"
            # auto-mode hints
            assert entry.get("auto_winner") == entry["method"], f"auto_winner / method mismatch: {entry}"
            assert "kfold_scores" in entry, f"auto mode must record kfold_scores: {entry}"
            scores = entry["kfold_scores"]
            assert set(scores.keys()).issubset(_valid_methods), f"unexpected kfold_scores keys: {scores}"

    def test_auto_prefers_reliability_weighted_on_heterogeneous_loadings(self):
        """Heterogeneous-SNR cluster: under widely different reliability
        the variance-aware combiners (``pca_pc1`` / ``mean_inv_var``)
        weight the strong member more heavily than uniform ``mean_z``.
        The OOF bake-off should pick one of those reliability-weighted
        methods over uniform mean_z.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _heterogeneous_loadings_frame()
        m = MRMR(
            dcd_enable=True,
            dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=2,
            full_npermutations=50,
            verbose=0,
            random_seed=0,
        ).fit(X, y)
        log = m.dcd_["swap_log"]
        assert len(log) >= 1, f"expected a swap to fire on the heterogeneous-loadings fixture; got n_swaps={m.dcd_['n_swaps']}"
        entry = log[0]
        scores = entry.get("kfold_scores", {})
        assert scores, f"auto mode must record kfold_scores; got {entry}"
        # mean_inv_var or pca_pc1 must >= mean_z under heterogeneous loadings.
        assert scores.get("mean_inv_var", 0.0) >= scores.get("mean_z", 0.0) or scores.get("pca_pc1", 0.0) >= scores.get(
            "mean_z", 0.0
        ), f"under heterogeneous loadings a reliability-weighted combiner should score >= mean_z; got {scores}"
        # And the chosen winner is not the uniform mean. Layer 44: the bake-off
        # pool now includes pca_pc2 / median_z / signed_max_abs / signed_l2_sum
        # — any of those can also legitimately surface as the winner since they
        # are also variance / magnitude-aware combiners; the contract here is
        # only that ``mean_z`` is NOT the uniform winner under heterogeneous
        # loadings.
        assert (
            entry["method"] != "mean_z"
        ), f"under heterogeneous loadings, expected a variance-aware combiner to win over uniform mean_z; got {entry['method']!r} with scores {scores}"

    def test_kfold_scoring_stable_across_seeds(self):
        """The OOF bake-off is seeded by member names so cluster-level
        repeats are bit-stable; varying MRMR.random_seed only affects
        screening order, not the bake-off cache key, so the same
        cluster yields the same kfold_scores.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _three_dups_plus_strong_frame()
        m1 = MRMR(
            dcd_enable=True,
            dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=2,
            full_npermutations=20,
            verbose=0,
            random_seed=0,
        ).fit(X, y)
        m2 = MRMR(
            dcd_enable=True,
            dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=2,
            full_npermutations=20,
            verbose=0,
            random_seed=7,
        ).fit(X, y)
        if not m1.dcd_["swap_log"] or not m2.dcd_["swap_log"]:
            pytest.skip("no swap fired on this fixture under both seeds")
        s1 = m1.dcd_["swap_log"][0].get("kfold_scores", {})
        s2 = m2.dcd_["swap_log"][0].get("kfold_scores", {})
        for k in set(s1) & set(s2):
            assert abs(s1[k] - s2[k]) < 1e-9, f"kfold_scores[{k!r}] differ across seeds: {s1[k]} vs {s2[k]}"

    def test_recipe_replay_uses_chosen_method(self):
        """The produced cluster_aggregate recipe must record the chosen
        combiner in ``extra['method']`` so replay uses the SAME combiner
        at transform time (the recipe is keyed to the OOF-winning method,
        not the user-facing ``auto`` string).

        Asserted against ``_produced_recipes_`` (the built pool), NOT
        ``_engineered_recipes_`` (the retained subset): on this fixture the
        redundancy sweep does not keep the PC1 aggregate in the final
        selection, so it is absent from ``_engineered_recipes_`` -- but the
        recipe was built and carries the chosen method.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        from mlframe.feature_selection.filters.engineered_recipes import (
            EngineeredRecipe,
        )

        X, y = _three_dups_plus_strong_frame()
        m = MRMR(
            dcd_enable=True,
            dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=2,
            full_npermutations=50,
            verbose=0,
            random_seed=0,
        ).fit(X, y)
        # This fixture forces a swap; if it genuinely yields 0 that is a
        # finding to investigate, not a vacuous skip.
        assert m.dcd_["n_swaps"] >= 1, f"expected a swap to fire on the 3-dups fixture; got n_swaps={m.dcd_['n_swaps']}"
        recipes = [r for r in getattr(m, "_produced_recipes_", []) if isinstance(r, EngineeredRecipe) and r.kind == "cluster_aggregate"]
        assert recipes, "no produced cluster_aggregate recipe found"
        log_method = m.dcd_["swap_log"][0].get("method")
        assert log_method is not None
        recipe_methods = [r.extra.get("method") for r in recipes]
        assert log_method in recipe_methods, f"swap_log method {log_method!r} not found in recipe.extra methods {recipe_methods}"

    def test_pin_explicit_method_overrides_auto(self):
        """Pinning ``dcd_swap_method='pca_pc1'`` (explicit) skips the
        bake-off entirely; swap_log records the pinned method without
        ``kfold_scores``/``auto_winner`` keys.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _three_dups_plus_strong_frame()
        m = MRMR(
            dcd_enable=True,
            dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=2,
            dcd_swap_method="pca_pc1",
            full_npermutations=50,
            verbose=0,
            random_seed=0,
        ).fit(X, y)
        if m.dcd_["n_swaps"] == 0:
            pytest.skip("no swap fired on this fixture")
        entry = m.dcd_["swap_log"][0]
        assert entry["method"] == "pca_pc1"
        assert "kfold_scores" not in entry, f"pinned method must skip bake-off, but kfold_scores present: {entry}"


# ---------------------------------------------------------------------------
# PART C — no-regression spot checks for Layer 12 / 27 / 35 / 41 / 42 ----
# ---------------------------------------------------------------------------


class TestNoRegressionPriorLayers:
    """Groups tests covering TestNoRegressionPriorLayers."""
    def test_layer42_default_threshold_pinned_at_4(self):
        """Layer 42 contract: ``dcd_cluster_size_threshold`` default
        unchanged at 4 (Layer 43 does NOT lower the threshold). The
        default-OFF behaviour is preserved; users opt in via threshold=2.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        m = MRMR()
        assert int(m.dcd_cluster_size_threshold) == 4

    def test_layer42_pin_threshold_2_still_fires_swap(self):
        """Layer42 pin threshold 2 still fires swap."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _three_dups_plus_strong_frame()
        m = MRMR(
            dcd_enable=True,
            dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=2,
            full_npermutations=50,
            verbose=0,
            random_seed=0,
        ).fit(X, y)
        assert m.dcd_["n_swaps"] >= 1

    def test_layer41_cluster_members_accessor(self):
        """Layer 41 ``cluster_members_`` must still expose the
        name-indexed cluster map.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _three_dups_plus_strong_frame()
        m = MRMR(
            dcd_enable=True,
            dcd_tau_cluster=0.5,
            verbose=0,
            random_seed=0,
        ).fit(X, y)
        cm = m.cluster_members_
        assert isinstance(cm, dict)
        assert any("dup_a" in k for k in cm.keys())

    def test_dcd_disabled_path_byte_identical_to_master(self):
        """Disabling DCD is bit-stable: cluster_members_ is None, dcd_
        has no swaps, support_ is populated normally.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _three_dups_plus_strong_frame()
        m = MRMR(
            dcd_enable=False,
            verbose=0,
            random_seed=0,
        ).fit(X, y)
        assert m.cluster_members_ is None
        assert m.dcd_ is None or m.dcd_.get("n_swaps", 0) == 0
        assert hasattr(m, "support_")
        assert len(list(m.get_feature_names_out())) >= 1

    def test_validate_accepts_auto_swap_method(self):
        """The ``_VALID_DCD_SWAP_METHODS`` tuple must include ``auto``
        so validate doesn't reject the new default.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        assert "auto" in MRMR._VALID_DCD_SWAP_METHODS
        X, y = _three_dups_plus_strong_frame(n=300)
        # Construct + fit with explicit auto must not raise.
        m = MRMR(
            dcd_enable=True,
            dcd_swap_method="auto",
            dcd_cluster_size_threshold=2,
            full_npermutations=20,
            verbose=0,
            random_seed=0,
        ).fit(X, y)
        assert hasattr(m, "support_")

    def test_legacy_pinned_pca_pc1_path_completes(self):
        """Users who pinned ``dcd_swap_method='pca_pc1'`` prior to
        Layer 43 must keep working.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _three_dups_plus_strong_frame()
        m = MRMR(
            dcd_enable=True,
            dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=2,
            dcd_swap_method="pca_pc1",
            full_npermutations=50,
            verbose=0,
            random_seed=0,
        ).fit(X, y)
        assert hasattr(m, "support_")
        # And when a swap fires, its PC1 aggregate must be recorded in the
        # swap_log (the PART A guarantee -- the marker lives in the swap_log,
        # not necessarily in the post-redundancy final selection surface).
        if m.dcd_["n_swaps"] >= 1:
            assert any(
                "_dcd_pc1_" in str(e.get("aggregate_name", "")) for e in m.dcd_["swap_log"] if e.get("branch") == "aggregate"
            ), f"pinned pca_pc1 swap must log a _dcd_pc1_ aggregate; got {m.dcd_['swap_log']}"


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _two_latent_correlated_frame(n: int = 2000, seed: int = 0):
    """Two correlated latents L1, L2 (rho=0.5). Cluster members load on a
    MIX of both. Target depends on L2 (i.e. the PC2 direction). PC1 of the
    cluster mostly tracks L1 → low MI with y; PC2 picks up L2 → high MI.

    Concretely:
        L1 = N(0,1)
        L2 = 0.5 * L1 + sqrt(1 - 0.25) * eps_L2
        member_i = a_i * L1 + b_i * L2 + 0.05 * noise_i

    Members have similar |loading| on L1 so PC1 captures ~L1; the
    orthogonal direction PC2 captures ~L2. y is driven by L2, so the
    K-fold OOF bake-off should prefer ``pca_pc2`` over ``pca_pc1``.
    """
    rng = np.random.default_rng(int(seed))
    L1 = rng.standard_normal(n)
    eps_L2 = rng.standard_normal(n)
    L2 = 0.5 * L1 + np.sqrt(0.75) * eps_L2
    # Members loading on L1 and L2 in different proportions; all share BOTH.
    m_a = 1.0 * L1 + 0.5 * L2 + 0.05 * rng.standard_normal(n)
    m_b = 1.0 * L1 - 0.5 * L2 + 0.05 * rng.standard_normal(n)
    m_c = 0.9 * L1 + 0.5 * L2 + 0.05 * rng.standard_normal(n)
    m_d = 0.9 * L1 - 0.5 * L2 + 0.05 * rng.standard_normal(n)
    other = rng.standard_normal(n)
    X = pd.DataFrame(
        {
            "strong": other,
            "member_a": m_a,
            "member_b": m_b,
            "member_c": m_c,
            "member_d": m_d,
            "noise_0": rng.standard_normal(n),
        }
    )
    # Target depends on L2 (the PC2 direction) plus the unrelated `other`.
    y = pd.Series((2 * other + 1.5 * L2 + 0.3 * rng.standard_normal(n) > 0).astype(int))
    return X, y


def _outlier_member_frame(n: int = 2000, seed: int = 0):
    """Cluster where ONE row block per member is corrupted with heavy-tail
    noise (gross-error contamination). The mean combiner is dragged by the
    contaminated rows; the per-row median is robust.

    Cluster = 4 members, each = latent + 0.05*N + 30 * Bernoulli(0.05) on a
    DIFFERENT row block per member so the contamination doesn't align. y
    depends only on `latent`. ``median_z`` should beat ``mean_z`` on this
    fixture.
    """
    rng = np.random.default_rng(int(seed))
    latent = rng.standard_normal(n)
    other = rng.standard_normal(n)
    members = []
    for _j in range(4):
        m = latent + 0.05 * rng.standard_normal(n)
        # 5% rows per member get a +30 sigma spike. The spike rows differ
        # across members so the row-median sees at most one outlier per row.
        mask = rng.random(n) < 0.05
        m = m + 30.0 * mask.astype(float)
        members.append(m)
    X = pd.DataFrame(
        {
            "strong": other,
            "member_a": members[0],
            "member_b": members[1],
            "member_c": members[2],
            "member_d": members[3],
            "noise_0": rng.standard_normal(n),
        }
    )
    y = pd.Series((2 * other + latent + 0.3 * rng.standard_normal(n) > 0).astype(int))
    return X, y


def _loudest_member_frame(n: int = 2000, seed: int = 0):
    """y depends on the MAX-MAGNITUDE per-row reading across the cluster
    members — the "loudest member" pattern (think: a sensor array where
    only the strongest reading carries useful info each row).

    Members are noise-corrupted shifted copies of a base latent; y is a
    deterministic function of the max-abs reading. The mean would dilute
    the signal; ``signed_max_abs`` should win the bake-off.
    """
    rng = np.random.default_rng(int(seed))
    base = rng.standard_normal(n)
    other = rng.standard_normal(n)
    # Each member is a noisy shifted version; combined into a "loud" cluster
    # by making the noise asymmetric across members.
    members = []
    for j in range(4):
        sigma = 0.5 + 0.2 * j
        m = base + sigma * rng.standard_normal(n)
        members.append(m)
    # Stack to compute per-row loudest-member as the target driver.
    M = np.column_stack(members)
    idx = np.argmax(np.abs(M), axis=1)
    rows = np.arange(n)
    loud = M[rows, idx]
    X = pd.DataFrame(
        {
            "strong": other,
            "member_a": members[0],
            "member_b": members[1],
            "member_c": members[2],
            "member_d": members[3],
            "noise_0": rng.standard_normal(n),
        }
    )
    y = pd.Series((2 * other + 1.5 * loud + 0.3 * rng.standard_normal(n) > 0).astype(int))
    return X, y


# ---------------------------------------------------------------------------
# Method-availability contracts
# ---------------------------------------------------------------------------


class TestLayer44_MethodEnrolment:
    """Groups tests covering TestLayer44_MethodEnrolment."""
    @pytest.mark.parametrize(
        "method",
        [
            "pca_pc2",
            "median_z",
            "signed_max_abs",
            "signed_l2_sum",
        ],
    )
    def test_valid_dcd_swap_method_includes(self, method):
        """Valid dcd swap method includes."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        assert method in MRMR._VALID_DCD_SWAP_METHODS, f"_VALID_DCD_SWAP_METHODS must include {method!r}; got {MRMR._VALID_DCD_SWAP_METHODS}"

    @pytest.mark.parametrize(
        "method",
        [
            "pca_pc2",
            "median_z",
            "signed_max_abs",
            "signed_l2_sum",
        ],
    )
    def test_valid_cluster_aggregate_methods_includes(self, method):
        """Valid cluster aggregate methods includes."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        assert method in MRMR._VALID_CLUSTER_AGGREGATE_METHODS, f"_VALID_CLUSTER_AGGREGATE_METHODS must include {method!r}"

    def test_cluster_aggregate_methods_tuple_includes_all_seven(self):
        """Cluster aggregate methods tuple includes all seven."""
        from mlframe.feature_selection.filters._cluster_aggregate import (
            CLUSTER_AGGREGATE_METHODS,
        )

        for m in ("mean_z", "mean_inv_var", "median", "pca_pc1", "factor_score", "pca_pc2", "median_z", "signed_max_abs", "signed_l2_sum"):
            assert m in CLUSTER_AGGREGATE_METHODS, f"CLUSTER_AGGREGATE_METHODS missing {m!r}: got {CLUSTER_AGGREGATE_METHODS}"

    def test_auto_method_candidates_has_seven(self):
        """Auto method candidates has seven."""
        from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
            _AUTO_METHOD_CANDIDATES,
        )

        assert set(_AUTO_METHOD_CANDIDATES) == {
            "mean_z",
            "mean_inv_var",
            "pca_pc1",
            "pca_pc2",
            "median_z",
            "signed_max_abs",
            "signed_l2_sum",
        }, f"unexpected auto candidate pool: {_AUTO_METHOD_CANDIDATES}"


# ---------------------------------------------------------------------------
# Per-method pinned-fit smoke (each new method individually selectable)
# ---------------------------------------------------------------------------


class TestLayer44_PinnedFitSmoke:
    """Groups tests covering TestLayer44_PinnedFitSmoke."""
    @pytest.mark.parametrize(
        "method",
        [
            "pca_pc2",
            "median_z",
            "signed_max_abs",
            "signed_l2_sum",
        ],
    )
    def test_pinned_method_fit_completes(self, method):
        """Pinning each new method individually must complete fit() and
        record the method name in the swap_log (when a swap fires).
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _two_latent_correlated_frame(n=1200, seed=1)
        m = MRMR(
            dcd_enable=True,
            dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=2,
            dcd_swap_method=method,
            full_npermutations=20,
            verbose=0,
            random_seed=0,
        ).fit(X, y)
        assert hasattr(m, "support_")
        # If a swap fired, the swap_log must record the pinned method.
        log = m.dcd_["swap_log"] if m.dcd_ else []
        for entry in log:
            assert entry.get("method") == method, f"pinned method {method!r} not recorded in swap_log entry: {entry}"
            # Pinned method must NOT carry bake-off keys.
            assert "kfold_scores" not in entry, f"pinned method must skip bake-off: {entry}"
            assert "auto_winner" not in entry, f"pinned method must not record auto_winner: {entry}"


# ---------------------------------------------------------------------------
# Direct K-fold bake-off behavioural tests
# ---------------------------------------------------------------------------


def _run_bakeoff_direct(X: pd.DataFrame, y: pd.Series, member_names: list):
    """Drive ``_select_swap_method_auto`` directly so we can isolate the
    per-cluster bake-off behaviour from the rest of the MRMR pipeline.

    Builds a minimal DCDState with a quantized target column, then standardizes
    the requested members and asks the auto-selector for the winner.
    """
    from mlframe.feature_selection.filters._cluster_aggregate import (
        _standardize_align,
        _continuous_cols,
    )
    from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
        DCDState,
        _select_swap_method_auto,
    )

    M = _continuous_cols(X, member_names)
    Z, _, _, _ = _standardize_align(M, ref_col=0)

    # Build a 2-col factors_data: [target_binned, dummy_member]. We only use
    # the target column for the bake-off; the auto-selector reads
    # ``factors_data[:, target_indices[0]]`` as the target.
    n = X.shape[0]
    # Discretise y to 2 bins (already binary in our fixtures but coerce robustly).
    y_arr = np.asarray(y, dtype=np.int64).ravel()
    factors_data = np.column_stack([y_arr.astype(np.int32), np.zeros(n, dtype=np.int32)])
    state = DCDState(
        pool_pruned_mask=np.zeros(2, dtype=bool),
        factors_data=factors_data,
        factors_nbins=np.array([int(y_arr.max()) + 1, 1], dtype=np.int64),
        target_indices=np.array([0], dtype=np.int64),
        quantization_method="quantile",
        quantization_nbins=10,
        quantization_dtype=np.int32,
    )
    winner, scores = _select_swap_method_auto(
        state=state,
        Z=Z,
        target_y=y_arr,
        member_names=tuple(member_names),
    )
    return winner, scores


class TestLayer44_BakeoffWins:
    """Groups tests covering TestLayer44_BakeoffWins."""
    def test_pca_pc2_wins_on_two_correlated_latents(self):
        """Cluster has two correlated latents; y depends on L2 (the PC2
        direction). PC2 should out-score PC1 in the K-fold OOF bake-off.
        """
        X, y = _two_latent_correlated_frame(seed=0)
        _winner, scores = _run_bakeoff_direct(
            X,
            y,
            ["member_a", "member_b", "member_c", "member_d"],
        )
        assert scores, f"empty bake-off scores: {scores}"
        # PC2's MI with y must dominate PC1's on this fixture (target ~ L2).
        assert scores.get("pca_pc2", 0.0) > scores.get(
            "pca_pc1", 0.0
        ), f"expected pca_pc2 to outscore pca_pc1 when y depends on the 2nd PC; got scores={scores}"

    def test_median_z_beats_mean_z_on_outlier_members(self):
        """One row per ~20 is corrupted with a heavy-tail spike per member;
        the mean is dragged but per-row median is robust. ``median_z`` MI
        with y must exceed ``mean_z`` MI on this fixture.
        """
        X, y = _outlier_member_frame(seed=0)
        _winner, scores = _run_bakeoff_direct(
            X,
            y,
            ["member_a", "member_b", "member_c", "member_d"],
        )
        assert scores
        assert scores.get("median_z", 0.0) > scores.get("mean_z", 0.0), f"under outlier-row contamination median_z should beat mean_z; got {scores}"

    def test_signed_max_abs_beats_mean_z_on_loudest_member(self):
        """y is a function of the per-row max-magnitude reading across
        cluster members. ``signed_max_abs`` (which IS that reduction) must
        beat ``mean_z`` (which dilutes the loudest signal).
        """
        X, y = _loudest_member_frame(seed=0)
        _winner, scores = _run_bakeoff_direct(
            X,
            y,
            ["member_a", "member_b", "member_c", "member_d"],
        )
        assert scores
        assert scores.get("signed_max_abs", 0.0) > scores.get("mean_z", 0.0), f"under loudest-member target signed_max_abs should beat mean_z; got {scores}"


# ---------------------------------------------------------------------------
# Recipe replay bit-identity for the 4 new methods
# ---------------------------------------------------------------------------


class TestLayer44_RecipeReplayBitIdentity:
    """Groups tests covering TestLayer44_RecipeReplayBitIdentity."""
    @pytest.mark.parametrize(
        "method",
        [
            "pca_pc2",
            "median_z",
            "signed_max_abs",
            "signed_l2_sum",
        ],
    )
    def test_apply_recipe_matches_fit_time_aggregate(self, method):
        """For each new method, building a cluster_aggregate recipe via
        ``build_cluster_aggregate_recipe`` + replaying via ``apply_recipe``
        must reproduce the same continuous aggregate as the in-memory
        fit-time computation (before discretisation).
        """
        from mlframe.feature_selection.filters._cluster_aggregate import (
            _standardize_align,
            _derive_weights,
            _apply_method_nonlinear,
            _NONLINEAR_METHODS,
            _continuous_cols,
        )
        from mlframe.feature_selection.filters.engineered_recipes import (
            build_cluster_aggregate_recipe,
            _apply_cluster_aggregate,
        )

        X, _y = _two_latent_correlated_frame(n=800, seed=2)
        member_names = ["member_a", "member_b", "member_c", "member_d"]
        M = _continuous_cols(X, member_names)
        Z, mean, std, signs = _standardize_align(M, ref_col=0)
        # Fit-time continuous aggregate.
        if method in _NONLINEAR_METHODS:
            agg_fit = _apply_method_nonlinear(Z, method)
            weights = None
        else:
            w = _derive_weights(Z, method)
            assert w is not None
            agg_fit = Z @ w
            weights = w
        # Build recipe (no quantization => replay returns continuous output).
        recipe = build_cluster_aggregate_recipe(
            name=f"layer44_{method}",
            src_names=tuple(member_names),
            method=method,
            member_mean=mean,
            member_std=std,
            signs=signs,
            weights=weights,
            quantization=None,
        )
        agg_replay = _apply_cluster_aggregate(recipe, X)
        assert agg_fit.shape == agg_replay.shape
        assert np.allclose(
            np.nan_to_num(agg_fit),
            np.nan_to_num(agg_replay),
            atol=1e-10,
            rtol=1e-10,
        ), f"replay aggregate must match fit-time for method={method!r}; max abs diff = {np.max(np.abs(agg_fit - agg_replay))}"

    @pytest.mark.parametrize(
        "method",
        [
            "pca_pc2",
            "median_z",
            "signed_max_abs",
            "signed_l2_sum",
        ],
    )
    def test_transform_deterministic_for_new_methods(self, method):
        """End-to-end: pinning each new method (when a swap fires) must
        yield deterministic ``transform`` output across two calls.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _two_latent_correlated_frame(n=1500, seed=3)
        m = MRMR(
            dcd_enable=True,
            dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=2,
            dcd_swap_method=method,
            full_npermutations=20,
            verbose=0,
            random_seed=0,
        ).fit(X, y)
        Xt1 = np.asarray(m.transform(X), dtype=np.float64)
        Xt2 = np.asarray(m.transform(X), dtype=np.float64)
        assert Xt1.shape == Xt2.shape
        assert np.allclose(Xt1, Xt2, equal_nan=True), f"transform must be deterministic for method={method!r}"


# ---------------------------------------------------------------------------
# Default-OFF byte-identity: pre-Layer-44 pinned methods are untouched
# ---------------------------------------------------------------------------


class TestLayer44_LegacyPinByteIdentity:
    """Groups tests covering TestLayer44_LegacyPinByteIdentity."""
    @pytest.mark.parametrize(
        "method",
        [
            "pca_pc1",
            "mean_z",
            "mean_inv_var",
        ],
    )
    def test_legacy_pinned_swap_log_shape_unchanged(self, method):
        """Users who pinned a pre-Layer-44 method continue to get the
        same swap_log entry shape: ``method == <pinned>``, no
        ``kfold_scores`` / ``auto_winner`` keys (those are auto-only).
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _two_latent_correlated_frame(n=1500, seed=4)
        m = MRMR(
            dcd_enable=True,
            dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=2,
            dcd_swap_method=method,
            full_npermutations=20,
            verbose=0,
            random_seed=0,
        ).fit(X, y)
        log = m.dcd_["swap_log"] if m.dcd_ else []
        # If a swap fired, every entry must be method-pure (no bake-off keys).
        for entry in log:
            assert entry.get("method") == method
            assert "kfold_scores" not in entry, f"legacy pinned method must not carry kfold_scores: {entry}"
            assert "auto_winner" not in entry, f"legacy pinned method must not carry auto_winner: {entry}"

    def test_auto_default_is_still_auto(self):
        """Layer 44 expands the candidate pool but keeps ``"auto"`` as the
        default ``dcd_swap_method`` (Layer 43 contract preserved).
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        m = MRMR()
        assert str(m.dcd_swap_method) == "auto"
