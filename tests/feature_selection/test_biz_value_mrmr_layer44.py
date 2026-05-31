"""Layer 44 biz_value: enrich the DCD auto bake-off pool with 4 more aggregators.

Layer 43 introduced the K-fold OOF MI bake-off across 3 candidate combiners
(``mean_z``, ``mean_inv_var``, ``pca_pc1``). Layer 44 grows the pool to 7
by adding:

- ``pca_pc2``       — 2nd principal component; surfaces the secondary
                       axis of shared variation in clusters with TWO
                       correlated latents that PC1 alone leaves on the table.
- ``median_z``      — per-row median of standardized members; robust to
                       outlier rows / outlier-member contamination.
- ``signed_max_abs`` — per-row ``sign(z_j*) * max_j|z_j|``; surfaces the
                       loudest single member's signal (best when the
                       target depends on max-magnitude rather than mean).
- ``signed_l2_sum`` — per-row signed quadratic combiner (each member's
                       z**2 with its sign preserved).

The richer menu lets the per-cluster K-fold OOF bake-off pick a combiner
that matches the cluster's structure rather than a one-size-fits-all linear
combiner. Linear aggregators stay on the ``Z @ weights`` fast path; the
non-linear / row-reduction methods route through
``_apply_method_nonlinear`` so fit/transform parity holds without storing
a weight vector.

CONTRACTS
---------
- All 4 new methods are pinnable via ``dcd_swap_method=<name>`` and via
  ``cluster_aggregate_methods`` for the standalone cluster-aggregate step.
- The auto bake-off pool is the strict superset
  ``{mean_z, mean_inv_var, pca_pc1, pca_pc2, median_z, signed_max_abs,
  signed_l2_sum}``.
- Recipe replay reproduces the fit-time aggregate bit-identically (same
  standardized matrix, same row reducer, same fit-time quantile edges).
- Default-OFF byte-identity: pinning a pre-existing method (``pca_pc1``,
  ``mean_z``, ``mean_inv_var``, ``median``) yields the SAME swap-log entry
  shape as on Layer 43 master — the 4 new methods only widen the auto
  candidate pool; they do NOT alter the path of any explicitly pinned
  legacy method.

NEVER xfail.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")


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
    X = pd.DataFrame({
        "strong": other,
        "member_a": m_a,
        "member_b": m_b,
        "member_c": m_c,
        "member_d": m_d,
        "noise_0": rng.standard_normal(n),
    })
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
    for j in range(4):
        m = latent + 0.05 * rng.standard_normal(n)
        # 5% rows per member get a +30 sigma spike. The spike rows differ
        # across members so the row-median sees at most one outlier per row.
        mask = rng.random(n) < 0.05
        m = m + 30.0 * mask.astype(float)
        members.append(m)
    X = pd.DataFrame({
        "strong": other,
        "member_a": members[0],
        "member_b": members[1],
        "member_c": members[2],
        "member_d": members[3],
        "noise_0": rng.standard_normal(n),
    })
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
    X = pd.DataFrame({
        "strong": other,
        "member_a": members[0],
        "member_b": members[1],
        "member_c": members[2],
        "member_d": members[3],
        "noise_0": rng.standard_normal(n),
    })
    y = pd.Series((2 * other + 1.5 * loud + 0.3 * rng.standard_normal(n) > 0).astype(int))
    return X, y


# ---------------------------------------------------------------------------
# Method-availability contracts
# ---------------------------------------------------------------------------


class TestLayer44_MethodEnrolment:

    @pytest.mark.parametrize("method", [
        "pca_pc2", "median_z", "signed_max_abs", "signed_l2_sum",
    ])
    def test_valid_dcd_swap_method_includes(self, method):
        from mlframe.feature_selection.filters.mrmr import MRMR
        assert method in MRMR._VALID_DCD_SWAP_METHODS, (
            f"_VALID_DCD_SWAP_METHODS must include {method!r}; got "
            f"{MRMR._VALID_DCD_SWAP_METHODS}"
        )

    @pytest.mark.parametrize("method", [
        "pca_pc2", "median_z", "signed_max_abs", "signed_l2_sum",
    ])
    def test_valid_cluster_aggregate_methods_includes(self, method):
        from mlframe.feature_selection.filters.mrmr import MRMR
        assert method in MRMR._VALID_CLUSTER_AGGREGATE_METHODS, (
            f"_VALID_CLUSTER_AGGREGATE_METHODS must include {method!r}"
        )

    def test_cluster_aggregate_methods_tuple_includes_all_seven(self):
        from mlframe.feature_selection.filters._cluster_aggregate import (
            CLUSTER_AGGREGATE_METHODS,
        )
        for m in ("mean_z", "mean_inv_var", "median", "pca_pc1",
                  "factor_score", "pca_pc2", "median_z",
                  "signed_max_abs", "signed_l2_sum"):
            assert m in CLUSTER_AGGREGATE_METHODS, (
                f"CLUSTER_AGGREGATE_METHODS missing {m!r}: got "
                f"{CLUSTER_AGGREGATE_METHODS}"
            )

    def test_auto_method_candidates_has_seven(self):
        from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
            _AUTO_METHOD_CANDIDATES,
        )
        assert set(_AUTO_METHOD_CANDIDATES) == {
            "mean_z", "mean_inv_var", "pca_pc1",
            "pca_pc2", "median_z", "signed_max_abs", "signed_l2_sum",
        }, f"unexpected auto candidate pool: {_AUTO_METHOD_CANDIDATES}"


# ---------------------------------------------------------------------------
# Per-method pinned-fit smoke (each new method individually selectable)
# ---------------------------------------------------------------------------


class TestLayer44_PinnedFitSmoke:

    @pytest.mark.parametrize("method", [
        "pca_pc2", "median_z", "signed_max_abs", "signed_l2_sum",
    ])
    def test_pinned_method_fit_completes(self, method):
        """Pinning each new method individually must complete fit() and
        record the method name in the swap_log (when a swap fires).
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _two_latent_correlated_frame(n=1200, seed=1)
        m = MRMR(
            dcd_enable=True, dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=2,
            dcd_swap_method=method,
            full_npermutations=20,
            verbose=0, random_seed=0,
        ).fit(X, y)
        assert hasattr(m, "support_")
        # If a swap fired, the swap_log must record the pinned method.
        log = m.dcd_["swap_log"] if m.dcd_ else []
        for entry in log:
            assert entry.get("method") == method, (
                f"pinned method {method!r} not recorded in swap_log "
                f"entry: {entry}"
            )
            # Pinned method must NOT carry bake-off keys.
            assert "kfold_scores" not in entry, (
                f"pinned method must skip bake-off: {entry}"
            )
            assert "auto_winner" not in entry, (
                f"pinned method must not record auto_winner: {entry}"
            )


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
        _standardize_align, _continuous_cols,
    )
    from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
        DCDState, _select_swap_method_auto,
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
        state=state, Z=Z, target_y=y_arr,
        member_names=tuple(member_names),
    )
    return winner, scores


class TestLayer44_BakeoffWins:

    def test_pca_pc2_wins_on_two_correlated_latents(self):
        """Cluster has two correlated latents; y depends on L2 (the PC2
        direction). PC2 should out-score PC1 in the K-fold OOF bake-off.
        """
        X, y = _two_latent_correlated_frame(seed=0)
        winner, scores = _run_bakeoff_direct(
            X, y, ["member_a", "member_b", "member_c", "member_d"],
        )
        assert scores, f"empty bake-off scores: {scores}"
        # PC2's MI with y must dominate PC1's on this fixture (target ~ L2).
        assert scores.get("pca_pc2", 0.0) > scores.get("pca_pc1", 0.0), (
            f"expected pca_pc2 to outscore pca_pc1 when y depends on the "
            f"2nd PC; got scores={scores}"
        )

    def test_median_z_beats_mean_z_on_outlier_members(self):
        """One row per ~20 is corrupted with a heavy-tail spike per member;
        the mean is dragged but per-row median is robust. ``median_z`` MI
        with y must exceed ``mean_z`` MI on this fixture.
        """
        X, y = _outlier_member_frame(seed=0)
        winner, scores = _run_bakeoff_direct(
            X, y, ["member_a", "member_b", "member_c", "member_d"],
        )
        assert scores
        assert scores.get("median_z", 0.0) > scores.get("mean_z", 0.0), (
            f"under outlier-row contamination median_z should beat "
            f"mean_z; got {scores}"
        )

    def test_signed_max_abs_beats_mean_z_on_loudest_member(self):
        """y is a function of the per-row max-magnitude reading across
        cluster members. ``signed_max_abs`` (which IS that reduction) must
        beat ``mean_z`` (which dilutes the loudest signal).
        """
        X, y = _loudest_member_frame(seed=0)
        winner, scores = _run_bakeoff_direct(
            X, y, ["member_a", "member_b", "member_c", "member_d"],
        )
        assert scores
        assert scores.get("signed_max_abs", 0.0) > scores.get("mean_z", 0.0), (
            f"under loudest-member target signed_max_abs should beat "
            f"mean_z; got {scores}"
        )


# ---------------------------------------------------------------------------
# Recipe replay bit-identity for the 4 new methods
# ---------------------------------------------------------------------------


class TestLayer44_RecipeReplayBitIdentity:

    @pytest.mark.parametrize("method", [
        "pca_pc2", "median_z", "signed_max_abs", "signed_l2_sum",
    ])
    def test_apply_recipe_matches_fit_time_aggregate(self, method):
        """For each new method, building a cluster_aggregate recipe via
        ``build_cluster_aggregate_recipe`` + replaying via ``apply_recipe``
        must reproduce the same continuous aggregate as the in-memory
        fit-time computation (before discretisation).
        """
        from mlframe.feature_selection.filters._cluster_aggregate import (
            _standardize_align, _derive_weights, _apply_method_nonlinear,
            _NONLINEAR_METHODS, _continuous_cols,
        )
        from mlframe.feature_selection.filters.engineered_recipes import (
            build_cluster_aggregate_recipe, _apply_cluster_aggregate,
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
            member_mean=mean, member_std=std, signs=signs,
            weights=weights, quantization=None,
        )
        agg_replay = _apply_cluster_aggregate(recipe, X)
        assert agg_fit.shape == agg_replay.shape
        assert np.allclose(
            np.nan_to_num(agg_fit), np.nan_to_num(agg_replay),
            atol=1e-10, rtol=1e-10,
        ), (
            f"replay aggregate must match fit-time for method={method!r}; "
            f"max abs diff = {np.max(np.abs(agg_fit - agg_replay))}"
        )

    @pytest.mark.parametrize("method", [
        "pca_pc2", "median_z", "signed_max_abs", "signed_l2_sum",
    ])
    def test_transform_deterministic_for_new_methods(self, method):
        """End-to-end: pinning each new method (when a swap fires) must
        yield deterministic ``transform`` output across two calls.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _two_latent_correlated_frame(n=1500, seed=3)
        m = MRMR(
            dcd_enable=True, dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=2,
            dcd_swap_method=method,
            full_npermutations=20,
            verbose=0, random_seed=0,
        ).fit(X, y)
        Xt1 = np.asarray(m.transform(X), dtype=np.float64)
        Xt2 = np.asarray(m.transform(X), dtype=np.float64)
        assert Xt1.shape == Xt2.shape
        assert np.allclose(Xt1, Xt2, equal_nan=True), (
            f"transform must be deterministic for method={method!r}"
        )


# ---------------------------------------------------------------------------
# Default-OFF byte-identity: pre-Layer-44 pinned methods are untouched
# ---------------------------------------------------------------------------


class TestLayer44_LegacyPinByteIdentity:

    @pytest.mark.parametrize("method", [
        "pca_pc1", "mean_z", "mean_inv_var",
    ])
    def test_legacy_pinned_swap_log_shape_unchanged(self, method):
        """Users who pinned a pre-Layer-44 method continue to get the
        same swap_log entry shape: ``method == <pinned>``, no
        ``kfold_scores`` / ``auto_winner`` keys (those are auto-only).
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _two_latent_correlated_frame(n=1500, seed=4)
        m = MRMR(
            dcd_enable=True, dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=2,
            dcd_swap_method=method,
            full_npermutations=20,
            verbose=0, random_seed=0,
        ).fit(X, y)
        log = m.dcd_["swap_log"] if m.dcd_ else []
        # If a swap fired, every entry must be method-pure (no bake-off keys).
        for entry in log:
            assert entry.get("method") == method
            assert "kfold_scores" not in entry, (
                f"legacy pinned method must not carry kfold_scores: {entry}"
            )
            assert "auto_winner" not in entry, (
                f"legacy pinned method must not carry auto_winner: {entry}"
            )

    def test_auto_default_is_still_auto(self):
        """Layer 44 expands the candidate pool but keeps ``"auto"`` as the
        default ``dcd_swap_method`` (Layer 43 contract preserved).
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        m = MRMR()
        assert str(m.dcd_swap_method) == "auto"
