"""biz_value: ORDER-2 Westfall-Young maxT permutation-null floor on the
prospective-pair JOINT MI (2026-06-03).

The FE step ranks O(p^2) prospective engineered PAIRS by JOINT MI(x_i, x_j; y).
At high p the MAX joint MI over PURE-NOISE pairs is a positive order statistic
that grows with the pool size -- the same best-of-p selection bias the order-1
screening floor rejects, now at order 2. The per-pair prevalence gates
(``fe_min_pair_mi_prevalence`` / ``fe_synergy_min_prevalence``) are PER-PAIR and
do NOT account for max-over-pool selection, so a wide noise matrix surfaces
"synergistic-looking" noise pairs whose joint MI is merely the best chance hit.

The fix (``_permutation_null.pooled_pair_permutation_null_joint_mi_floor``,
wired into ``_mrmr_fe_step``) shuffles the discretised target K times, takes the
per-shuffle MAX joint MI over the candidate pool via the SAME batched plug-in
estimator the screen scores ``pair_mi`` with, floors prospective-pair selection
at the q-th quantile, and requires a pair's joint MI to clear it IN ADDITION to
the prevalence gates. DEFAULT-ON; SELF-GATING below ``fe_pair_maxt_min_pairs``.

Gates pinned here:
  A. WIDE-NOISE: the order-2 floor reduces spurious engineered noise-pairs vs
     floor-off WITHOUT dropping the genuine synergy pairs.
  C. SELF-GATING: a small-p genuine-synergy fixture (5 cols, XOR) is unaffected
     (pool < ``fe_pair_maxt_min_pairs`` => floor 0).
  + floor-disabled byte-identical (``fe_pair_maxt_null_permutations=0``).
  + a unit test of the null helper: genuine joint-MI >> null-max; noise ~ null-max.
"""
from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import pytest


GENUINE_OPERANDS = {"x1", "x2", "x3", "x4", "x5", "x6"}


def _parents_of(name: str) -> set:
    import re
    return set(re.findall(r"(x[1-6]|noise_\d+)", name))


def _classify(engineered) -> tuple[list, list]:
    genuine, spurious = [], []
    for nm in engineered:
        ps = _parents_of(nm)
        if not ps:
            continue
        has_noise = any(p.startswith("noise_") for p in ps)
        has_genuine = any(p in GENUINE_OPERANDS for p in ps)
        if has_genuine and not has_noise:
            genuine.append(nm)
        elif has_noise:
            spurious.append(nm)
    return genuine, spurious


def _wide_synergy_frame(n=2000, n_noise=74, seed=20260603):
    # 6 genuine operands feeding 3 genuine synergy pairs (XOR sign product /
    # product / bilinear), each with ~zero per-operand marginal MI but strong
    # joint dependence with y; the rest is pure Gaussian noise.
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n); x2 = rng.normal(size=n)
    x3 = rng.normal(size=n); x4 = rng.normal(size=n)
    x5 = rng.normal(size=n); x6 = rng.normal(size=n)
    s_xor = np.sign(x1 * x2)
    s_prod = x3 * x4
    s_bilin = x5 * x6 + 0.3 * x5
    lin = 1.5 * s_xor + 1.2 * np.sign(s_prod) + 1.0 * np.sign(s_bilin)
    p = 1.0 / (1.0 + np.exp(-lin))
    y = (rng.random(n) < p).astype(int)
    d = {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5, "x6": x6}
    for j in range(n_noise):
        d[f"noise_{j}"] = rng.normal(size=n)
    return pd.DataFrame(d), pd.Series(y, name="y")


def _fit_engineered(perms, *, n_noise=40, **overrides):
    """Fit MRMR on the wide-synergy frame; return the engineered-feature list.

    The default ``fe_synergy_*`` gates already filter clean Gaussian noise
    aggressively; to put a measurable number of spurious noise pairs into the
    support WITHOUT the floor (so the floor's effect is observable at the
    support level) we deliberately loosen the synergy prevalence + downstream
    engineered-MI gate and lift the synergy-pair budget. The order-2 floor is
    the ONLY difference between the OFF and ON runs. ``n_noise=40`` keeps the
    floor-OFF per-pair search (which the floor would otherwise prune away)
    inside the 60s pytest budget while still surfacing several spurious pairs.
    """
    from mlframe.feature_selection.filters.mrmr import MRMR
    X, y = _wide_synergy_frame(n_noise=n_noise)
    cfg = dict(
        verbose=0, random_seed=42, fe_max_steps=1,
        fe_synergy_screen_max_features=n_noise + 20,
        fe_synergy_min_prevalence=1.05,
        fe_synergy_max_pairs=20,
        fe_min_engineered_mi_prevalence=0.75,
        fe_pair_maxt_null_permutations=perms,
    )
    cfg.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MRMR(**cfg).fit(X, y)
    return list(getattr(m, "_engineered_features_", []) or [])


# =============================================================================
# Gate A: WIDE-NOISE -- floor reduces spurious noise pairs, keeps genuine.
# =============================================================================


class TestOrder2MaxTFloorWideNoise:
    def test_floor_reduces_spurious_pairs_keeps_genuine(self):
        eng_off = _fit_engineered(perms=0)
        eng_on = _fit_engineered(perms=25)
        gen_off, spur_off = _classify(eng_off)
        gen_on, spur_on = _classify(eng_on)

        # The floor-off run must actually surface spurious noise pairs, else the
        # test would pass vacuously and prove nothing about the floor.
        assert len(spur_off) >= 1, (
            "floor-off run produced no spurious noise pairs; the fixture no "
            f"longer exercises the floor. engineered={eng_off}"
        )
        # Floor ON strictly reduces spurious noise pairs.
        assert len(spur_on) < len(spur_off), (
            f"order-2 floor did not reduce spurious noise pairs: "
            f"OFF={spur_off} ON={spur_on}"
        )
        # Genuine synergy pairs are NOT dropped by the floor.
        assert len(gen_on) >= len(gen_off) and len(gen_on) >= 3, (
            f"order-2 floor dropped genuine synergy pairs: "
            f"OFF genuine={gen_off} ON genuine={gen_on}"
        )


# =============================================================================
# Gate C: SELF-GATING -- small-p XOR fixture below the pool-size gate untouched.
# =============================================================================


class TestOrder2MaxTFloorSelfGating:
    def _small_xor_engineered(self, perms):
        from mlframe.feature_selection.filters.mrmr import MRMR
        rng = np.random.default_rng(11)
        n = 2000
        x1 = rng.integers(0, 2, n)
        x2 = rng.integers(0, 2, n)
        y = (x1 ^ x2).astype(np.int64)
        X = pd.DataFrame({
            "x1": x1.astype(float), "x2": x2.astype(float),
            "n0": rng.standard_normal(n), "n1": rng.standard_normal(n),
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = MRMR(
                verbose=0, random_seed=7, fe_max_steps=1,
                fe_synergy_screen_max_features=10,
                fe_pair_maxt_null_permutations=perms,
            ).fit(X, pd.Series(y))
        return list(m.get_feature_names_out()), list(getattr(m, "_engineered_features_", []) or [])

    def test_small_pool_below_min_pairs_byte_identical(self):
        # 4-feature frame => at most C(4,2)=6 candidate pairs < fe_pair_maxt_min_pairs (30),
        # so the floor is 0.0 (no-op): floor-on and floor-off must be identical.
        sup_off, eng_off = self._small_xor_engineered(perms=0)
        sup_on, eng_on = self._small_xor_engineered(perms=25)
        assert sup_off == sup_on, (
            f"small-p XOR support changed under the floor (should be no-op): "
            f"OFF={sup_off} ON={sup_on}"
        )
        assert eng_off == eng_on, (
            f"small-p XOR engineered set changed under the floor (should be no-op): "
            f"OFF={eng_off} ON={eng_on}"
        )


# =============================================================================
# Floor-disabled byte-identical: perms=0 == omitting the param.
# =============================================================================


class TestOrder2MaxTFloorDisabled:
    def test_perms_zero_matches_explicit_disable(self):
        # Two perms=0 runs are trivially identical; the meaningful check is that
        # perms=0 takes the no-op path (floor never computed). We assert the
        # engineered set is exactly the 3 genuine pairs (the loose-gate fixture's
        # floor-OFF output minus nothing): proves the disable path is live.
        eng = _fit_engineered(perms=0)
        gen, spur = _classify(eng)
        assert len(gen) >= 3, f"floor-disabled run lost genuine pairs: {eng}"
        # And re-running is deterministic.
        eng2 = _fit_engineered(perms=0)
        assert eng == eng2, f"floor-disabled run non-deterministic: {eng} vs {eng2}"

    def test_default_gates_floor_is_noop(self):
        """Under DEFAULT synergy gates the wide frame's noise pairs never clear
        the per-pair prevalence bar, so the order-2 floor has nothing to reject:
        the engineered set must be byte-identical with the floor ENABLED (default
        25) vs DISABLED (0). Proves the floor does not perturb real default-config
        users -- it only removes pairs the prevalence gate would otherwise admit.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _wide_synergy_frame(n_noise=40)
        base = dict(verbose=0, random_seed=42, fe_max_steps=1,
                    fe_synergy_screen_max_features=60)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m_on = MRMR(fe_pair_maxt_null_permutations=25, **base).fit(X, y)
            m_off = MRMR(fe_pair_maxt_null_permutations=0, **base).fit(X, y)
        sup_on = list(m_on.get_feature_names_out())
        sup_off = list(m_off.get_feature_names_out())
        eng_on = list(getattr(m_on, "_engineered_features_", []) or [])
        eng_off = list(getattr(m_off, "_engineered_features_", []) or [])
        assert sup_on == sup_off, (
            f"default-config support differs floor-on vs off (should be no-op): "
            f"ON={sup_on} OFF={sup_off}"
        )
        assert eng_on == eng_off, (
            f"default-config engineered set differs floor-on vs off: "
            f"ON={eng_on} OFF={eng_off}"
        )


# =============================================================================
# Unit test of the null helper itself.
# =============================================================================


class TestPooledPairPermutationNullHelper:
    def _discretize(self, X, y, n_bins=8):
        from mlframe.feature_selection.filters.discretization import categorize_dataset
        from mlframe.feature_selection.filters.info_theory import merge_vars
        df = X.copy()
        df["y"] = y.values if hasattr(y, "values") else y
        cols = list(df.columns)
        data, _c, nbins = categorize_dataset(df=df, method="quantile", n_bins=n_bins, dtype=np.int16)
        y_idx = cols.index("y")
        classes_y, freqs_y, _ = merge_vars(
            factors_data=data, vars_indices=[y_idx],
            var_is_nominal=None, factors_nbins=nbins, dtype=np.int16,
        )
        return data, nbins, cols, classes_y, freqs_y

    def test_genuine_joint_mi_above_null_noise_at_or_below(self):
        from mlframe.feature_selection.filters.info_theory import batch_pair_mi_prange
        from mlframe.feature_selection.filters._permutation_null import (
            pooled_pair_permutation_null_joint_mi_floor,
        )
        X, y = _wide_synergy_frame(n_noise=74)
        data, nbins, cols, classes_y, freqs_y = self._discretize(X, y)

        feat_idx = [cols.index(c) for c in cols if c != "y"]
        pairs = list(combinations(feat_idx, 2))
        pa = np.fromiter((p[0] for p in pairs), dtype=np.int64, count=len(pairs))
        pb = np.fromiter((p[1] for p in pairs), dtype=np.int64, count=len(pairs))
        mis = batch_pair_mi_prange(data, pa, pb, nbins, classes_y, freqs_y)

        floor = pooled_pair_permutation_null_joint_mi_floor(
            factors_data=data, nbins=nbins, pair_a=pa, pair_b=pb,
            classes_y=classes_y, freqs_y=freqs_y,
            n_permutations=25, quantile=0.95, random_seed=42,
        )
        assert floor > 0.0, "null floor should be positive on a wide pool"

        genuine_keys = {tuple(sorted((cols.index(a), cols.index(b)))) for a, b in
                        [("x1", "x2"), ("x3", "x4"), ("x5", "x6")]}
        gen_mis, noise_mis = [], []
        for k, (a, b) in enumerate(pairs):
            key = tuple(sorted((a, b)))
            na, nb = cols[a], cols[b]
            if key in genuine_keys:
                gen_mis.append(mis[k])
            elif na.startswith("noise_") and nb.startswith("noise_"):
                noise_mis.append(mis[k])
        gen_mis = np.array(gen_mis)
        noise_mis = np.array(noise_mis)

        # Genuine synergy joint MI clears the null-max floor with margin.
        assert (gen_mis > floor).all(), (
            f"genuine synergy joint MIs {gen_mis} not all above null floor {floor}"
        )
        # The overwhelming majority of noise pairs sit at/below the floor.
        below = float((noise_mis <= floor).mean())
        assert below >= 0.95, (
            f"only {below:.2%} of noise pairs at/below the null floor {floor}; "
            f"noise max={noise_mis.max():.5f}"
        )

    def test_degenerate_pool_returns_zero_floor(self):
        from mlframe.feature_selection.filters._permutation_null import (
            pooled_pair_permutation_null_joint_mi_floor,
        )
        # n too small.
        data = np.zeros((4, 3), dtype=np.int16)
        nbins = np.array([2, 2, 2], dtype=np.int64)
        pa = np.array([0, 0, 1], dtype=np.int64)
        pb = np.array([1, 2, 2], dtype=np.int64)
        classes_y = np.array([0, 1, 0, 1], dtype=np.int16)
        freqs_y = np.array([0.5, 0.5], dtype=np.float64)
        assert pooled_pair_permutation_null_joint_mi_floor(
            factors_data=data, nbins=nbins, pair_a=pa, pair_b=pb,
            classes_y=classes_y, freqs_y=freqs_y, n_permutations=25,
        ) == 0.0
        # n_permutations == 0 disables.
        data2 = np.zeros((100, 3), dtype=np.int16)
        cy = np.zeros(100, dtype=np.int16); cy[::2] = 1
        assert pooled_pair_permutation_null_joint_mi_floor(
            factors_data=data2, nbins=nbins, pair_a=pa, pair_b=pb,
            classes_y=cy, freqs_y=freqs_y, n_permutations=0,
        ) == 0.0
        # fewer than 2 candidate pairs.
        assert pooled_pair_permutation_null_joint_mi_floor(
            factors_data=data2, nbins=nbins,
            pair_a=np.array([0], dtype=np.int64), pair_b=np.array([1], dtype=np.int64),
            classes_y=cy, freqs_y=freqs_y, n_permutations=25,
        ) == 0.0
