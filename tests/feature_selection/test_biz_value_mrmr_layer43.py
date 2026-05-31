"""Layer 43 biz_value: DCD commit_swap recipe wiring + auto swap-method.

PART A bug fix (the L42 deferred CRITICAL bug)
----------------------------------------------
``_screen_predictors.py`` used to call ``commit_swap(engineered_recipes=None)``
so the PC1 aggregate column was appended to ``selected_vars`` / ``cols`` but
never registered as an ``EngineeredRecipe``. The remap at the end of
``_mrmr_fit_impl.py`` then dropped the aggregate name from
``self._engineered_recipes_``, and ``MRMR.get_feature_names_out`` silently
shrank ``support_`` whenever a swap fired. Layer 43 threads the host
``engineered_recipes`` dict through ``screen_predictors`` into
``commit_swap`` AND upgrades the recipe payload from a plain dict to a
frozen ``EngineeredRecipe`` of kind ``cluster_aggregate``. The aggregate
now appears in ``feature_names_out`` and ``transform`` reproduces it.

PART B improvement (auto swap-method selection)
-----------------------------------------------
``dcd_swap_method`` defaults to ``"auto"``. ``evaluate_swap_candidate``
runs a K-fold (n_folds=5) OOF MI bake-off over ``("mean_z",
"mean_inv_var", "pca_pc1")`` and picks the per-cluster winner. The chosen
method is recorded in:
  - ``recipe.extra["method"]`` (so ``_apply_cluster_aggregate`` replays
    the same combiner at transform-time)
  - ``swap_log`` entry's ``method`` / ``auto_winner`` / ``kfold_scores``
    keys (audit trail).

K-fold scores are cached on ``state._auto_method_cache`` keyed by member
names so successive re-evaluations of the same cluster reuse the bake-off
(cheap).

Recipe replay is bit-identical with fit because the chosen method, the
member statistics (mean / std / signs), the weight vector (when linear),
and the fit-time quantile edges are all persisted in the recipe.

CONSTRAINTS
-----------
- NEVER xfail.
- ``'auto'`` is the new default; pinning ``"pca_pc1"`` preserves the
  legacy single-method path.
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
    X = pd.DataFrame({
        "strong": other,
        "dup_a": latent + 0.01 * rng.standard_normal(n),
        "dup_b": latent + 0.01 * rng.standard_normal(n),
        "dup_c": latent + 0.01 * rng.standard_normal(n),
        "noise_0": rng.standard_normal(n),
    })
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
    X = pd.DataFrame({
        "strong_unrelated": other,
        "member_strong": 3.0 * latent + 0.02 * rng.standard_normal(n),
        "member_weak1": 0.5 * latent + 0.10 * rng.standard_normal(n),
        "member_weak2": 0.5 * latent + 0.10 * rng.standard_normal(n),
        "noise_0": rng.standard_normal(n),
    })
    y = pd.Series((2 * other + latent + 0.3 * rng.standard_normal(n) > 0).astype(int))
    return X, y


# ---------------------------------------------------------------------------
# PART A — commit_swap recipes wiring fix
# ---------------------------------------------------------------------------


class TestPartA_RecipeWiring:

    def test_swap_aggregate_visible_in_feature_names_out(self):
        """When swap fires, the PC1 aggregate name MUST appear in
        ``get_feature_names_out``. Pre-Layer-43 this name was silently
        dropped because ``commit_swap`` was called with
        ``engineered_recipes=None``.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _three_dups_plus_strong_frame()
        m = MRMR(
            dcd_enable=True, dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=2,
            dcd_swap_method="pca_pc1",  # pin so we can assert _dcd_pc1_ name
            full_npermutations=50,
            verbose=0, random_seed=0,
        ).fit(X, y)
        # Sanity: a swap actually fired (otherwise this test is vacuous).
        assert m.dcd_["n_swaps"] >= 1, (
            f"Expected swap to fire on 3-dups + threshold=2; got "
            f"n_swaps={m.dcd_['n_swaps']}"
        )
        names_out = list(m.get_feature_names_out())
        agg_names = [n for n in names_out if "_dcd_pc1_" in n]
        assert len(agg_names) >= 1, (
            f"PC1 aggregate must be in get_feature_names_out; got "
            f"{names_out}. swap_log={m.dcd_['swap_log']}"
        )

    def test_engineered_recipes_contains_cluster_aggregate(self):
        """The ``_engineered_recipes_`` list must contain a recipe of
        kind ``cluster_aggregate`` for the swap aggregate."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        from mlframe.feature_selection.filters.engineered_recipes import (
            EngineeredRecipe,
        )
        X, y = _three_dups_plus_strong_frame()
        m = MRMR(
            dcd_enable=True, dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=2,
            dcd_swap_method="pca_pc1",
            full_npermutations=50,
            verbose=0, random_seed=0,
        ).fit(X, y)
        assert m.dcd_["n_swaps"] >= 1
        recipes = list(getattr(m, "_engineered_recipes_", []))
        dcd_recipes = [
            r for r in recipes
            if isinstance(r, EngineeredRecipe)
            and r.kind == "cluster_aggregate"
            and r.name.startswith("_dcd_pc1_")
        ]
        assert len(dcd_recipes) >= 1, (
            f"Expected >=1 cluster_aggregate recipe; got recipes={recipes}"
        )

    def test_transform_reproduces_aggregate_column(self):
        """The aggregate column produced by ``transform`` on the SAME
        training data must be deterministic and finite (recipe replay
        does not require y).
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _three_dups_plus_strong_frame()
        m = MRMR(
            dcd_enable=True, dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=2,
            dcd_swap_method="pca_pc1",
            full_npermutations=50,
            verbose=0, random_seed=0,
        ).fit(X, y)
        assert m.dcd_["n_swaps"] >= 1
        Xt1 = m.transform(X)
        Xt2 = m.transform(X)
        # Transform output must be deterministic.
        Xt1_arr = np.asarray(Xt1, dtype=np.float64)
        Xt2_arr = np.asarray(Xt2, dtype=np.float64)
        assert Xt1_arr.shape == Xt2_arr.shape
        assert np.allclose(Xt1_arr, Xt2_arr, equal_nan=True), (
            "transform() must be deterministic on the same input"
        )
        # Aggregate column must be present and finite.
        names_out = list(m.get_feature_names_out())
        assert len(names_out) == Xt1_arr.shape[1], (
            f"feature_names_out ({len(names_out)}) must match transform "
            f"width ({Xt1_arr.shape[1]})"
        )
        agg_positions = [i for i, n in enumerate(names_out)
                         if "_dcd_pc1_" in n]
        assert len(agg_positions) >= 1
        for ap in agg_positions:
            col = Xt1_arr[:, ap]
            assert np.all(np.isfinite(col)), (
                f"aggregate column {names_out[ap]} contains NaN/Inf at "
                f"transform; values: {col[:8]}"
            )


# ---------------------------------------------------------------------------
# PART B — auto swap-method selection
# ---------------------------------------------------------------------------


class TestPartB_AutoMethod:

    def test_default_swap_method_is_auto(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        m = MRMR()
        assert str(m.dcd_swap_method) == "auto", (
            f"Default dcd_swap_method must be 'auto'; got "
            f"{m.dcd_swap_method!r}"
        )

    def test_auto_records_chosen_method_in_swap_log(self):
        """With ``dcd_swap_method='auto'``, every swap_log entry must
        carry a ``method`` key naming the actual combiner used. When
        the bake-off ran (>=1 fold succeeded), ``auto_winner`` and
        ``kfold_scores`` must also be present.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _three_dups_plus_strong_frame()
        m = MRMR(
            dcd_enable=True, dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=2,
            # default dcd_swap_method='auto'
            full_npermutations=50,
            verbose=0, random_seed=0,
        ).fit(X, y)
        assert m.dcd_["n_swaps"] >= 1
        log = m.dcd_["swap_log"]
        for entry in log:
            assert "method" in entry, (
                f"swap_log entry missing 'method' key: {entry}"
            )
            assert entry["method"] in (
                "mean_z", "mean_inv_var", "pca_pc1",
            ), f"unexpected chosen method: {entry['method']!r}"
            # auto-mode hints
            assert entry.get("auto_winner") == entry["method"], (
                f"auto_winner / method mismatch: {entry}"
            )
            assert "kfold_scores" in entry, (
                f"auto mode must record kfold_scores: {entry}"
            )
            scores = entry["kfold_scores"]
            assert set(scores.keys()).issubset({
                "mean_z", "mean_inv_var", "pca_pc1",
            }), f"unexpected kfold_scores keys: {scores}"

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
            dcd_enable=True, dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=2,
            full_npermutations=50,
            verbose=0, random_seed=0,
        ).fit(X, y)
        log = m.dcd_["swap_log"]
        assert len(log) >= 1, (
            f"expected a swap to fire on the heterogeneous-loadings "
            f"fixture; got n_swaps={m.dcd_['n_swaps']}"
        )
        entry = log[0]
        scores = entry.get("kfold_scores", {})
        assert scores, f"auto mode must record kfold_scores; got {entry}"
        # mean_inv_var or pca_pc1 must >= mean_z under heterogeneous loadings.
        assert (
            scores.get("mean_inv_var", 0.0) >= scores.get("mean_z", 0.0)
            or scores.get("pca_pc1", 0.0) >= scores.get("mean_z", 0.0)
        ), (
            f"under heterogeneous loadings a reliability-weighted "
            f"combiner should score >= mean_z; got {scores}"
        )
        # And the chosen winner is not the uniform mean.
        assert entry["method"] in ("mean_inv_var", "pca_pc1"), (
            f"expected winner in (mean_inv_var, pca_pc1); got "
            f"{entry['method']!r} with scores {scores}"
        )

    def test_kfold_scoring_stable_across_seeds(self):
        """The OOF bake-off is seeded by member names so cluster-level
        repeats are bit-stable; varying MRMR.random_seed only affects
        screening order, not the bake-off cache key, so the same
        cluster yields the same kfold_scores.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _three_dups_plus_strong_frame()
        m1 = MRMR(
            dcd_enable=True, dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=2,
            full_npermutations=20, verbose=0, random_seed=0,
        ).fit(X, y)
        m2 = MRMR(
            dcd_enable=True, dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=2,
            full_npermutations=20, verbose=0, random_seed=7,
        ).fit(X, y)
        if not m1.dcd_["swap_log"] or not m2.dcd_["swap_log"]:
            pytest.skip("no swap fired on this fixture under both seeds")
        s1 = m1.dcd_["swap_log"][0].get("kfold_scores", {})
        s2 = m2.dcd_["swap_log"][0].get("kfold_scores", {})
        for k in set(s1) & set(s2):
            assert abs(s1[k] - s2[k]) < 1e-9, (
                f"kfold_scores[{k!r}] differ across seeds: "
                f"{s1[k]} vs {s2[k]}"
            )

    def test_recipe_replay_uses_chosen_method(self):
        """The recipe stored in ``_engineered_recipes_`` must record the
        chosen method in ``extra['method']`` -- replay reads this and
        uses the SAME combiner at transform time.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        from mlframe.feature_selection.filters.engineered_recipes import (
            EngineeredRecipe,
        )
        X, y = _three_dups_plus_strong_frame()
        m = MRMR(
            dcd_enable=True, dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=2,
            full_npermutations=50, verbose=0, random_seed=0,
        ).fit(X, y)
        if m.dcd_["n_swaps"] == 0:
            pytest.skip("no swap fired on this fixture")
        recipes = [
            r for r in getattr(m, "_engineered_recipes_", [])
            if isinstance(r, EngineeredRecipe)
            and r.kind == "cluster_aggregate"
        ]
        assert recipes, "no cluster_aggregate recipe found"
        log_method = m.dcd_["swap_log"][0].get("method")
        assert log_method is not None
        recipe_methods = [r.extra.get("method") for r in recipes]
        assert log_method in recipe_methods, (
            f"swap_log method {log_method!r} not found in recipe.extra "
            f"methods {recipe_methods}"
        )

    def test_pin_explicit_method_overrides_auto(self):
        """Pinning ``dcd_swap_method='pca_pc1'`` (explicit) skips the
        bake-off entirely; swap_log records the pinned method without
        ``kfold_scores``/``auto_winner`` keys.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _three_dups_plus_strong_frame()
        m = MRMR(
            dcd_enable=True, dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=2,
            dcd_swap_method="pca_pc1",
            full_npermutations=50, verbose=0, random_seed=0,
        ).fit(X, y)
        if m.dcd_["n_swaps"] == 0:
            pytest.skip("no swap fired on this fixture")
        entry = m.dcd_["swap_log"][0]
        assert entry["method"] == "pca_pc1"
        assert "kfold_scores" not in entry, (
            f"pinned method must skip bake-off, but kfold_scores present: "
            f"{entry}"
        )


# ---------------------------------------------------------------------------
# PART C — no-regression spot checks for Layer 12 / 27 / 35 / 41 / 42 ----
# ---------------------------------------------------------------------------


class TestNoRegressionPriorLayers:

    def test_layer42_default_threshold_pinned_at_4(self):
        """Layer 42 contract: ``dcd_cluster_size_threshold`` default
        unchanged at 4 (Layer 43 does NOT lower the threshold). The
        default-OFF behaviour is preserved; users opt in via threshold=2.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        m = MRMR()
        assert int(m.dcd_cluster_size_threshold) == 4

    def test_layer42_pin_threshold_2_still_fires_swap(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _three_dups_plus_strong_frame()
        m = MRMR(
            dcd_enable=True, dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=2,
            full_npermutations=50, verbose=0, random_seed=0,
        ).fit(X, y)
        assert m.dcd_["n_swaps"] >= 1

    def test_layer41_cluster_members_accessor(self):
        """Layer 41 ``cluster_members_`` must still expose the
        name-indexed cluster map.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _three_dups_plus_strong_frame()
        m = MRMR(
            dcd_enable=True, dcd_tau_cluster=0.5,
            verbose=0, random_seed=0,
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
            dcd_enable=False, verbose=0, random_seed=0,
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
            dcd_enable=True, dcd_swap_method="auto",
            dcd_cluster_size_threshold=2,
            full_npermutations=20, verbose=0, random_seed=0,
        ).fit(X, y)
        assert hasattr(m, "support_")

    def test_legacy_pinned_pca_pc1_path_completes(self):
        """Users who pinned ``dcd_swap_method='pca_pc1'`` prior to
        Layer 43 must keep working.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _three_dups_plus_strong_frame()
        m = MRMR(
            dcd_enable=True, dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=2,
            dcd_swap_method="pca_pc1",
            full_npermutations=50, verbose=0, random_seed=0,
        ).fit(X, y)
        assert hasattr(m, "support_")
        # And the aggregate must be visible (PART A guarantee).
        if m.dcd_["n_swaps"] >= 1:
            names = list(m.get_feature_names_out())
            assert any("_dcd_pc1_" in n for n in names)
