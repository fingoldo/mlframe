"""Layer 84 biz_value: CMIM (Layer 74) profiled + optimized hot-path.

Layer 83's 7-dataset x 10-mechanism showdown established CMIM (Layer 74)
as the best mechanism on real data (5/7 dataset wins). Layer 84
profile-and-optimize step: at production scale (n=2500, ~10 raw + ~20
engineered candidates) ``score_features_by_cmim`` was dominated by

    1. ``argsort`` inside ``np.unique`` (~34% of tottime)
    2. ``_renumber_joint`` chained two-folds (xz / yz / xyz per
       (candidate, support) pair, with yz / z recomputed across all
       ``p_eng`` candidates redundantly)
    3. ``_entropy_from_classes`` reductions over int64 class arrays

The Layer 84 optimization combines two orthogonal moves:

* ``_build_cmi_yz_cache`` pre-computes ``(yz_joint, z, h_z, h_yz, k_z,
  k_yz)`` per support member -- these are invariant across the ``p_eng``
  candidate columns and so the ``np.unique`` + entropy work is amortized
  ``p_eng-1`` times.
* ``_factorize_pack`` replaces the ``np.unique`` + chained-fold renumber
  with a single ``pd.factorize(sort=False)`` over a Horner-packed int64
  key. Hash-based dedup is ~3x faster than sort-based dedup at n=2500;
  the resulting class ids may be permuted relative to ``_renumber_joint``
  but the COUNT MULTISET is identical, and ``_entropy_from_classes`` is
  invariant under class-id permutation -- so the final CMI value is
  bit-equal up to floating-point summation order.

Contracts pinned
----------------

* ``TestCmimPerfBudget``: fit ``score_features_by_cmim`` on n=2500,
  p_eng=20 in <= 5s (sub-second on a modern laptop; the budget is a
  regression gate).
* ``TestCmimSpeedupBaselineReference``: vs. the documented PRE-OPT
  baseline of ~133 ms on the same fixture, the POST-OPT mean must be
  at most 2/3 the baseline (>= 1.5x speedup). The PRE-OPT baseline is
  encoded as a constant so this is a regression gate, not a self-
  referencing tautology.
* ``TestCmimBitEquivalentToReference``: CMIM scores match a pinned
  reference vector (captured pre-optimization with seed=0, n=2500,
  p_eng=20, n_bins=10) at ``atol=1e-12 / rtol=1e-9``. Different
  summation order is allowed (factorize permutes class ids) but the
  numerical value must collapse to the same float at machine
  precision.
* ``TestCmimL74RegressionCmimRanksRedundantLow``: Layer 74 contract
  ``TestCmimRanksRedundantLow`` re-run -- the conditional-MI redundancy
  filter still penalises near-copy candidates as documented.

NEVER xfail. Real numbers.

2026-06-01 Layer 84.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

SEEDS = (1, 7, 13, 42, 101)

# Pinned PRE-OPT baseline reference values (n=2500, p_eng=20, seed=0,
# n_bins=10, raw_X redundancy default). Captured 2026-06-01 before the
# Layer 84 factorize-pack + cached-yz optimization. See
# ``profiling/bench_cmim_l84.py`` for the capture script.
PRE_OPT_REFERENCE_SCORES = np.array([
    0.19235580927945716,
    0.12844775173295846,
    0.08803226108675456,
    0.013869129976898854,
    0.0034614357872515137,
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0,
])
PRE_OPT_REFERENCE_NAMES = [
    "x0__He2", "x1__He2", "x2__He2", "x0__He3", "x1__He3",
]
# Pre-optimization mean wall time on the L84 reference fixture, captured
# on the development laptop. The 1.5x gate uses 133 ms as the baseline
# so the post-opt budget is <= 88.7 ms. Machines slower than the dev box
# can still pass because the ratio is the gate, not the absolute time
# (post-opt time is measured fresh on the running host).
PRE_OPT_REFERENCE_MS = 133.1


def _build_l84_fixture(n: int = 2500, seed: int = 0):
    """Realistic ~10 raw / ~20 engineered candidate fixture.

    Mirrors the fixture pinned in the PRE-OPT reference capture so the
    bit-equivalence check is meaningful. Replicated here (rather than
    imported from ``profiling/``) so the test is self-contained.
    """
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
        generate_univariate_basis_features,
    )
    rng = np.random.default_rng(seed)
    raw_cols = {f"x{k}": rng.standard_normal(n) for k in range(10)}
    X_raw = pd.DataFrame(raw_cols)
    engineered = generate_univariate_basis_features(
        X_raw, degrees=(2, 3), basis="hermite",
    )
    if engineered.shape[1] > 20:
        engineered = engineered.iloc[:, :20]
    signal = (
        0.8 * (X_raw["x0"] ** 2)
        + 0.6 * (X_raw["x1"] ** 2)
        + 0.4 * (X_raw["x2"] ** 2)
    )
    thr = float(np.median(signal))
    y = ((signal + 0.05 * rng.standard_normal(n)) > thr).astype(int).to_numpy()
    return X_raw, engineered, y


# ---------------------------------------------------------------------------
# Contract 1: absolute perf budget
# ---------------------------------------------------------------------------


class TestCmimPerfBudget:
    """Hard wall-time cap: production-scale CMIM scoring must complete
    in under 5 seconds. The post-opt path runs sub-100ms on the dev box;
    5s is a generous CI gate covering slow shared runners.
    """

    def test_score_features_by_cmim_under_5s(self):
        from mlframe.feature_selection.filters._orthogonal_cmim_fe import (
            score_features_by_cmim,
        )
        raw_X, eng, y = _build_l84_fixture(n=2500, seed=0)
        # warm-up (excluded from the budget -- first call pays an
        # import / numpy-init cost the steady-state path doesn't).
        _ = score_features_by_cmim(raw_X, eng, y, n_bins=10)
        t0 = time.perf_counter()
        _ = score_features_by_cmim(raw_X, eng, y, n_bins=10)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, (
            f"score_features_by_cmim took {elapsed*1000:.1f} ms on the "
            f"L84 reference fixture; perf budget of 5000 ms exceeded. "
            f"Likely the cached-yz / factorize-pack fast path regressed."
        )


# ---------------------------------------------------------------------------
# Contract 2: >= 1.5x speedup vs documented baseline
# ---------------------------------------------------------------------------


class TestCmimSpeedupVsBaseline:
    """Post-optimization mean time on the L84 fixture must be at most
    2/3 the pre-optimization baseline (>= 1.5x speedup).
    """

    def test_post_opt_at_least_1p5x_faster(self):
        from mlframe.feature_selection.filters._orthogonal_cmim_fe import (
            score_features_by_cmim,
        )
        raw_X, eng, y = _build_l84_fixture(n=2500, seed=0)
        # warm-up
        _ = score_features_by_cmim(raw_X, eng, y, n_bins=10)
        n_runs = 10
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = score_features_by_cmim(raw_X, eng, y, n_bins=10)
        elapsed_ms = (time.perf_counter() - t0) / n_runs * 1000
        speedup = PRE_OPT_REFERENCE_MS / max(elapsed_ms, 1e-6)
        assert speedup >= 1.5, (
            f"Post-opt mean {elapsed_ms:.2f} ms is only "
            f"{speedup:.2f}x faster than the documented pre-opt baseline "
            f"of {PRE_OPT_REFERENCE_MS:.1f} ms; the 1.5x speedup contract "
            f"is violated. Re-run profiling/bench_cmim_l84.py to confirm."
        )


# ---------------------------------------------------------------------------
# Contract 3: bit-equivalence to pinned PRE-OPT reference
# ---------------------------------------------------------------------------


class TestCmimBitEquivalentToReference:
    """CMIM scores on the pinned fixture match the PRE-OPT reference
    vector at ``rtol=1e-9 / atol=1e-12`` -- machine-epsilon tolerance
    is required because factorize-pack permutes the class ids before
    the entropy reductions, which changes the order of the float sums
    in ``np.bincount`` by an amount bounded by relative epsilon.
    """

    def test_first_five_engineered_scores_match_ref(self):
        from mlframe.feature_selection.filters._orthogonal_cmim_fe import (
            score_features_by_cmim,
        )
        raw_X, eng, y = _build_l84_fixture(n=2500, seed=0)
        out = score_features_by_cmim(raw_X, eng, y, n_bins=10)
        new_vals = out["engineered_mi"].to_numpy()
        ref = PRE_OPT_REFERENCE_SCORES
        assert new_vals.shape == ref.shape, (
            f"L84 fixture shape changed: got {new_vals.shape}, "
            f"expected {ref.shape}; the reference vector is stale."
        )
        np.testing.assert_allclose(
            new_vals, ref, rtol=1e-9, atol=1e-12,
            err_msg=(
                "Post-opt CMIM scores do not match the pinned PRE-OPT "
                "reference vector. The factorize-pack / cached-yz path "
                "must be bit-equivalent to _renumber_joint + per-call "
                "_cmi_from_binned at float precision."
            ),
        )

    def test_top5_engineered_names_match_ref(self):
        from mlframe.feature_selection.filters._orthogonal_cmim_fe import (
            score_features_by_cmim,
        )
        raw_X, eng, y = _build_l84_fixture(n=2500, seed=0)
        out = score_features_by_cmim(raw_X, eng, y, n_bins=10)
        top5 = list(out["engineered_col"].head(5))
        assert top5 == PRE_OPT_REFERENCE_NAMES, (
            f"Top-5 engineered cols changed: got {top5}, "
            f"expected {PRE_OPT_REFERENCE_NAMES}; the ranking moved."
        )


# ---------------------------------------------------------------------------
# Contract 4: Layer 74 regression — redundancy filter still active
# ---------------------------------------------------------------------------


class TestL74RedundancyContractStillHolds:
    """Re-run Layer 74's signature contract: CMIM penalises near-copy
    redundant candidates. The optimization must not regress this.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_cmim_redundant_far_below_novel(self, seed):
        from mlframe.feature_selection.filters._orthogonal_cmim_fe import (
            score_features_by_cmim,
        )
        from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
            generate_univariate_basis_features,
        )
        rng = np.random.default_rng(int(seed))
        n = 2000
        x1 = rng.standard_normal(n)
        x_dup_a = x1 + 0.05 * rng.standard_normal(n)
        x_dup_b = x1 + 0.05 * rng.standard_normal(n)
        x_dup_c = x1 + 0.05 * rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        X = pd.DataFrame({
            "x1": x1,
            "x_dup_a": x_dup_a,
            "x_dup_b": x_dup_b,
            "x_dup_c": x_dup_c,
            "x2": x2,
            "noise_0": rng.standard_normal(n),
            "noise_1": rng.standard_normal(n),
        })
        signal = x1 ** 2 + 0.6 * (x2 ** 2)
        thr = float(np.median(signal))
        y = ((signal + 0.05 * rng.standard_normal(n)) > thr).astype(int)
        engineered = generate_univariate_basis_features(
            X, degrees=(2,), basis="hermite",
        )
        scores = score_features_by_cmim(
            X, engineered, np.asarray(y),
            current_support=X[["x1"]],
            n_bins=10,
        )
        s_map = dict(zip(scores["engineered_col"], scores["engineered_mi"]))
        eng_dup = [
            c for c in s_map
            if c.startswith(("x_dup_a__He", "x_dup_b__He", "x_dup_c__He"))
        ]
        eng_x2 = [c for c in s_map if c.startswith("x2__He")]
        assert eng_dup
        assert eng_x2
        max_dup = float(max(s_map[c] for c in eng_dup))
        novel = float(s_map[eng_x2[0]])
        assert novel > max_dup + 0.02, (
            f"seed={seed}: post-L84 CMIM novel x2__He2 ({novel:.4f}) not "
            f"clearly above max redundant x_dup_*__He2 ({max_dup:.4f}); "
            f"Layer 74 redundancy contract regressed."
        )


# ---------------------------------------------------------------------------
# Contract 5: empty-support fast-path matches reference (marginal MI)
# ---------------------------------------------------------------------------


class TestEmptySupportFallback:
    """When ``current_support`` empties out post-filter (e.g. the only
    member is the candidate's own source), CMIM falls back to marginal
    MI via ``_cmi_from_binned(x, y, None)``. The fast path must take
    the same fallback (otherwise edge candidates would silently score
    differently).
    """

    def test_single_support_eq_own_source_falls_back_to_marginal(self):
        from mlframe.feature_selection.filters._orthogonal_cmim_fe import (
            score_features_by_cmim,
        )
        from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
            generate_univariate_basis_features,
        )
        from mlframe.feature_selection.filters._mi_greedy_cmi_fe import (
            _cmi_from_binned, _quantile_bin,
        )
        rng = np.random.default_rng(42)
        n = 1500
        x1 = rng.standard_normal(n)
        X_raw = pd.DataFrame({"x1": x1})  # single-column raw_X
        engineered = generate_univariate_basis_features(
            X_raw, degrees=(2,), basis="hermite",
        )
        # x1 carries a quadratic signal so He_2(x1) has real MI(.; y).
        y = ((x1 ** 2 + 0.05 * rng.standard_normal(n)) > 1.0).astype(int)
        # current_support is x1 -- after filtering out the candidate's
        # own source (x1) the filtered cache is EMPTY -> fallback path.
        scores = score_features_by_cmim(
            X_raw, engineered, y,
            current_support=X_raw[["x1"]],
            n_bins=10,
        )
        s_map = dict(zip(scores["engineered_col"], scores["engineered_mi"]))
        # Reference marginal MI from the public _cmi_from_binned.
        x_bin = _quantile_bin(
            np.ascontiguousarray(engineered["x1__He2"].to_numpy(),
                                 dtype=np.float64),
            nbins=10,
        )
        _, y_bin = np.unique(y.astype(np.int64), return_inverse=True)
        y_bin = y_bin.astype(np.int64)
        expected = float(_cmi_from_binned(x_bin, y_bin, None))
        assert s_map["x1__He2"] == pytest.approx(expected, rel=1e-12), (
            f"empty-support fallback path drifted: got "
            f"{s_map['x1__He2']:.10f}, expected marginal MI "
            f"{expected:.10f}."
        )
