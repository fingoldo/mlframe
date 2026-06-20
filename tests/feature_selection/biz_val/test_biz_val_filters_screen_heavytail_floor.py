"""biz_value: target-over-split maxT noise floor on NARROW pools.

The Westfall-Young maxT permutation-null gain floor (``pooled_permutation_null_gain_floor``) rejects pure-noise columns whose plug-in MI is finite-sample-inflated. It
historically fired only on WIDE pools (``len(pool) >= screen_fdr_min_features``, default 30) where best-of-p selection bias dominates -- the embedding / TF-IDF regime. That
gate MISSES a distinct finite-sample-bias regime on NARROW tabular pools:

* A heavy-tailed (log-normal) regression target that the supervised MDLP binner OVER-SPLITS into ~30 bins while the features bin to ~5. The plug-in MI bias
  ``(nbins_x-1)*(nbins_y-1)/(2n)`` then lifts pure-noise columns past the abs/rel gain floors AFTER the genuine signals are selected, so a noise column leaks into a ~9-column
  pool that never reached the wide-pool gate.

A blunt pool-size drop cannot separate that from a DENSE weak-signal regression pool (sklearn diabetes: 10 genuine-but-weak features) where the SAME narrow pool must keep ALL
features -- lowering the size gate there over-prunes 10 -> 2 and regresses R^2 0.36 -> 0.25. The discriminator is the TARGET, captured by two cheap predicates over the
already-computed bin counts in ``target_oversplit_floor_applies``:

* over-split: ``nbins_y >= oversplit_ratio * median(nbins_x)`` -- the MDLP binner split the target into many more bins than the features carry (the plug-in bias source).
* reliable:  ``n / (nbins_y * median(nbins_x)) >= min_rows_per_joint_cell`` -- the (X,y) joint table is dense enough that the maxT floor itself is not finite-sample garbage.

Both must hold for the floor to bite on a narrow pool. This file pins BOTH sides:

* The WIN: a log-normal-noise pool trips the gate; the floor fires and the noise columns are rejected.
* The NO-REGRESSION: a dense weak-signal regression pool (diabetes-shaped) keeps the gate OFF; the floor is a no-op and all genuine weak features survive.

A future "just always apply the floor on narrow pools" would over-prune the diabetes side; the no-regression half catches that. A future "raise the wide-pool gate back so the
narrow gate is dead code" would re-leak the lognormal noise; the win half catches that.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._permutation_null import (
    pooled_permutation_null_gain_floor,
    target_oversplit_floor_applies,
)
from mlframe.feature_selection.filters.discretization import categorize_dataset


N_TOTAL = 2_500
N_NOISE = 6
SEEDS = (1, 7, 42)


# ---------------------------------------------------------------------------
# Data builders (mirror layer15 lognormal + a diabetes-shaped weak-signal pool)
# ---------------------------------------------------------------------------


def _build_lognormal(seed: int):
    """Heavy-tailed y = exp(1.5*x1 + 0.8*x2 + 0.5*x3 + noise) + 6 noise cols.

    MDLP over-splits the heavy-tailed target into ~30 bins while the features bin to ~5 -- the over-split regime the narrow-pool floor must catch.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(N_TOTAL)
    x2 = rng.standard_normal(N_TOTAL)
    x3 = rng.standard_normal(N_TOTAL)
    latent = 1.5 * x1 + 0.8 * x2 + 0.5 * x3 + 0.3 * rng.standard_normal(N_TOTAL)
    cols = {"x_signal_1": x1, "x_signal_2": x2, "x_signal_3": x3}
    for k in range(N_NOISE):
        cols[f"noise_{k}"] = rng.standard_normal(N_TOTAL)
    return pd.DataFrame(cols), pd.Series(np.exp(latent), name="y")


def _build_dense_weak_regression(seed: int, n: int = 330):
    """A diabetes-shaped pool: small n, every feature weak-but-genuine, an over-split target.

    y is a dense additive combination of 10 standardised features with comparable small coefficients plus moderate noise -- no feature dominates, all carry real signal. At
    small n the MDLP-binned continuous target splits into many bins (nbins_y >> feature nbins), so the over-split predicate trips; but ``n / (nbins_y * feat)`` is tiny
    (~1-2 rows per joint cell), so the reliability predicate must keep the floor OFF.
    """
    rng = np.random.default_rng(seed)
    p = 10
    Xa = rng.standard_normal((n, p))
    coefs = np.linspace(0.6, 1.5, p)
    y = Xa @ coefs + 2.5 * rng.standard_normal(n)
    cols = {f"f{i:02d}": Xa[:, i] for i in range(p)}
    return pd.DataFrame(cols), pd.Series(y, name="y")


def _categorize(X, y):
    df = X.copy()
    df["targ_y"] = y.values
    data, cols, nbins = categorize_dataset(
        df=df,
        method="quantile",
        n_bins=10,
        dtype=np.int32,
        missing_strategy="separate_bin",
        nbins_strategy="mdlp",
        nbins_strategy_kwargs=None,
        y_for_strategy=np.asarray(y),
    )
    y_idx = cols.index("targ_y")
    cand = [i for i in range(len(cols)) if i != y_idx]
    return data, cols, nbins, y_idx, cand


def _corrected_marginal_mi(data, nbins, ci, y_idx):
    n = data.shape[0]
    inv_n = 1.0 / n
    nbx = int(nbins[ci])
    nby = int(nbins[y_idx])
    xc = np.ascontiguousarray(data[:, ci]).astype(np.int64)
    yc = np.ascontiguousarray(data[:, y_idx]).astype(np.int64)
    xcounts = np.bincount(xc, minlength=nbx).astype(np.float64)
    px = xcounts[xcounts > 0] * inv_n
    h_x = -(px * np.log(px)).sum()
    ycounts = np.bincount(yc, minlength=nby).astype(np.float64)
    py = ycounts[ycounts > 0] * inv_n
    h_y = -(py * np.log(py)).sum()
    jc = np.bincount(xc * nby + yc, minlength=nbx * nby).astype(np.float64)
    pj = jc[jc > 0] * inv_n
    h_xy = -(pj * np.log(pj)).sum()
    mi = h_x + h_y - h_xy
    return mi - (nbx - 1) * (nby - 1) / (2.0 * n)


# ---------------------------------------------------------------------------
# WIN side: lognormal over-split pool -> gate fires -> floor rejects noise
# ---------------------------------------------------------------------------


class TestLognormalOversplitFloorFires:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_gate_fires_on_lognormal_narrow_pool(self, seed):
        X, y = _build_lognormal(seed)
        data, cols, nbins, y_idx, cand = _categorize(X, y)
        # Narrow pool: 9 features, below the wide-pool gate of 30.
        assert len(cand) < 30
        nby = int(nbins[y_idx])
        feat = [int(nbins[i]) for i in cand]
        # The over-split signature: target binned into many more levels than features.
        assert nby >= 3 * np.median(feat), (
            f"lognormal target not over-split: nbins_y={nby}, median feat nbins={np.median(feat)}; seed={seed}"
        )
        assert target_oversplit_floor_applies(
            nbins, cand, y_idx, data.shape[0],
        ), f"narrow-pool over-split gate must fire on lognormal; seed={seed}"

    @pytest.mark.parametrize("seed", SEEDS)
    def test_floor_rejects_all_noise_keeps_all_signal(self, seed):
        X, y = _build_lognormal(seed)
        data, cols, nbins, y_idx, cand = _categorize(X, y)
        floor = pooled_permutation_null_gain_floor(
            data, nbins, cand, y_idx,
            n_permutations=25, quantile=0.95,
            cardinality_bias_correction=True, random_seed=seed,
        )
        assert floor > 0.0, f"floor must be positive on lognormal pool; seed={seed}"
        for i in cand:
            name = cols[i]
            mi_corr = _corrected_marginal_mi(data, nbins, i, y_idx)
            if name.startswith("noise_"):
                assert mi_corr < floor, (
                    f"noise column {name} corrected MI {mi_corr:.5f} must be below floor {floor:.5f}; seed={seed}"
                )
            else:
                assert mi_corr >= floor, (
                    f"signal {name} corrected MI {mi_corr:.5f} must clear floor {floor:.5f}; seed={seed}"
                )


class TestLognormalEndToEndNoNoiseLeaks:
    """Full MRMR.fit on the lognormal pool must not leak any noise column -- the production contract the layer15 lognormal noise tests pin."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_mrmr_fit_excludes_noise_on_lognormal(self, seed):
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _build_lognormal(seed)
        sel = MRMR(verbose=0, interactions_max_order=1, fe_max_steps=0, random_seed=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel.fit(X, y)
        names = list(sel.get_feature_names_out())
        leaked = [n for n in names if n.startswith("noise_")]
        assert not leaked, f"noise leaked into log-normal support_: {leaked}; seed={seed}, support={names}"


# ---------------------------------------------------------------------------
# NO-REGRESSION side: dense weak-signal pool -> gate stays OFF -> floor no-op
# ---------------------------------------------------------------------------


class TestDenseWeakSignalFloorStaysOff:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_gate_stays_off_on_dense_weak_signal_pool(self, seed):
        X, y = _build_dense_weak_regression(seed)
        data, cols, nbins, y_idx, cand = _categorize(X, y)
        nby = int(nbins[y_idx])
        feat = [int(nbins[i]) for i in cand]
        # The target IS over-split at small n (precondition trips), but the (X,y)
        # joint occupancy is too sparse for the floor to be reliable, so the
        # reliability predicate must keep the gate OFF -- preserving the legacy
        # narrow-pool behaviour (no floor) and all 10 weak features.
        rows_per_joint = data.shape[0] / (nby * float(np.median(feat)))
        assert rows_per_joint < 8.0, (
            f"test bug: dense-weak pool unexpectedly dense (rows/joint={rows_per_joint:.1f}); the no-regression case must be the sparse-occupancy regime; seed={seed}"
        )
        assert not target_oversplit_floor_applies(
            nbins, cand, y_idx, data.shape[0],
        ), (
            f"narrow-pool floor gate must stay OFF on the dense weak-signal pool (would over-prune genuine weak features); seed={seed}"
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_mrmr_keeps_majority_of_weak_features(self, seed):
        """With the floor correctly OFF, MRMR must retain a healthy fraction of the 10 genuine weak features (the diabetes over-prune regression was 10 -> 2)."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _build_dense_weak_regression(seed)
        sel = MRMR(
            verbose=0, interactions_max_order=1, fe_max_steps=0,
            dcd_enable=False, cluster_aggregate_enable=False,
            build_friend_graph=False, cat_fe_config=None,
            random_seed=seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel.fit(X, y)
        n_kept = sel.n_features_
        assert n_kept >= 5, (
            f"floor must not over-prune the dense weak-signal pool: kept only {n_kept}/10 features (the diabetes 10->2 regression signature); seed={seed}, "
            f"support={list(sel.get_feature_names_out())}"
        )


# ---------------------------------------------------------------------------
# Gate threshold behaviour (unit-level, no fit)
# ---------------------------------------------------------------------------


class TestGateThresholdSemantics:

    def test_low_cardinality_target_never_over_split(self):
        """A binary / 3-class classification target (nbins_y in {2,3}) is never over-split relative to multi-bin features, so the gate stays OFF regardless of n."""
        nbins = np.array([5, 5, 5, 3], dtype=np.int64)  # 3 features + 3-class target
        cand = [0, 1, 2]
        assert not target_oversplit_floor_applies(nbins, cand, 3, 5000)

    def test_oversplit_but_sparse_stays_off(self):
        """Over-split target but sub-threshold joint occupancy -> OFF (the diabetes regime)."""
        nbins = np.array([5, 5, 5, 50], dtype=np.int64)
        cand = [0, 1, 2]
        # n=300 -> 300 / (50*5) = 1.2 rows per joint cell < 8 -> unreliable -> OFF.
        assert not target_oversplit_floor_applies(nbins, cand, 3, 300)

    def test_oversplit_and_dense_fires(self):
        """Over-split target AND dense joint occupancy -> ON (the lognormal regime)."""
        nbins = np.array([5, 5, 5, 30], dtype=np.int64)
        cand = [0, 1, 2]
        # n=2500 -> 2500 / (30*5) = 16.7 rows per joint cell >= 8 -> reliable -> ON.
        assert target_oversplit_floor_applies(nbins, cand, 3, 2500)

    def test_degenerate_pool_returns_false(self):
        nbins = np.array([1, 30], dtype=np.int64)  # only a constant feature + target
        assert not target_oversplit_floor_applies(nbins, [0], 1, 2500)
        # Single-class target.
        nbins2 = np.array([5, 5, 1], dtype=np.int64)
        assert not target_oversplit_floor_applies(nbins2, [0, 1], 2, 2500)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--no-cov", "-p", "no:randomly"])
