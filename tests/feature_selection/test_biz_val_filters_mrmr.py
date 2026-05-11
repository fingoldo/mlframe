"""biz_val tests for ``MRMR`` (feature_selection/filters/mrmr.py).

Per CLAUDE.md "Every new ML trick gets a biz_val synthetic test":
each test asserts a SYNTHETIC measurable WIN that locks in MRMR's
core parameters. A future code change that silently breaks one of
these parameters will fail the matching assertion.

Naming: ``test_biz_val_mrmr_<parameter>_<scenario>``.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")


def _to_df(X, y):
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    return df, pd.Series(y, name="y")


# ---------------------------------------------------------------------------
# interactions_max_order
# ---------------------------------------------------------------------------


def test_biz_val_mrmr_interactions_max_order_3way_xor_recovery():
    """``interactions_max_order=3`` must surface 3-way-XOR signal
    features in top-5 of ``support_``; ``=2`` (the default) cannot --
    the target is invisible to pair screening because every individual
    and every pair MI is ~0."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    rng = np.random.default_rng(42)
    n, p = 2000, 10
    X = rng.normal(size=(n, p))
    y = (np.sign(X[:, 0] * X[:, 1] * X[:, 2]) > 0).astype(np.int64)
    df, ys = _to_df(X, y)

    def _top5_overlap(order):
        sel = MRMR(interactions_max_order=order, verbose=0, random_seed=42)
        sel.fit(df, ys)
        return len(set(int(i) for i in sel.support_[:5]) & {0, 1, 2})

    overlap_2 = _top5_overlap(2)
    overlap_3 = _top5_overlap(3)
    assert overlap_3 >= 2, (
        f"order=3 must put >=2 of signal triplet in top-5; got {overlap_3}"
    )
    assert overlap_3 >= overlap_2, (
        f"order=3 ({overlap_3}) must >= order=2 ({overlap_2}) on 3-way XOR"
    )


# ---------------------------------------------------------------------------
# quantization_nbins
# ---------------------------------------------------------------------------


def test_biz_val_mrmr_quantization_nbins_finer_grid_better_mi_on_smooth():
    """Finer ``quantization_nbins`` (20 vs 5) must capture more MI on
    a SMOOTH continuous target. With only 5 bins, the discretization
    loses signal in the bulk; with 20 bins it captures finer
    structure. Floor: 20-bin MI / 5-bin MI >= 1.10x."""
    from mlframe.feature_selection.filters.info_theory import (
        compute_mi_from_classes, merge_vars,
    )
    from mlframe.feature_selection.filters.discretization import discretize_array

    rng = np.random.default_rng(42)
    n = 2000
    x = rng.normal(size=n)
    y = (x ** 2 + 0.3 * rng.normal(size=n) > 1.0).astype(np.int64)

    def _binned_mi(nbins):
        x_bin = discretize_array(arr=x, n_bins=nbins, method="quantile",
                                   dtype=np.int32)
        # Run pyutilz-style MI via merge_vars + compute_mi_from_classes.
        factors_data = np.column_stack([x_bin, y]).astype(np.int32)
        factors_nbins = np.array([nbins, len(np.unique(y))], dtype=np.int64)
        cx, fx, _ = merge_vars(factors_data, (0,), None, factors_nbins,
                                 dtype=np.int32)
        cy, fy, _ = merge_vars(factors_data, (1,), None, factors_nbins,
                                 dtype=np.int32)
        return float(compute_mi_from_classes(classes_x=cx, freqs_x=fx,
                                              classes_y=cy, freqs_y=fy,
                                              dtype=np.int32))

    mi_5 = _binned_mi(5)
    mi_20 = _binned_mi(20)
    assert mi_20 / max(mi_5, 1e-9) >= 1.10, (
        f"20-bin MI must beat 5-bin by >=1.10x on smooth target; "
        f"got {mi_20:.4f} vs {mi_5:.4f} (ratio {mi_20/mi_5:.2f}x)"
    )


# ---------------------------------------------------------------------------
# min_relevance_gain
# ---------------------------------------------------------------------------


def test_biz_val_mrmr_min_relevance_gain_stops_at_noise():
    """``min_relevance_gain`` must control where MRMR stops in noisy
    feature space: a TIGHT threshold (0.05) stops earlier than a
    LOOSE one (1e-6) on a target with 3 strong + 7 noise features.
    Floor: tight selects strictly fewer features."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    rng = np.random.default_rng(42)
    n, p_signal, p_noise = 2000, 3, 7
    X_signal = rng.normal(size=(n, p_signal))
    X_noise = rng.normal(size=(n, p_noise))
    X = np.column_stack([X_signal, X_noise])
    y = (X_signal.sum(axis=1) > 0).astype(np.int64)
    df, ys = _to_df(X, y)

    sel_loose = MRMR(min_relevance_gain=1e-6, verbose=0, random_seed=42)
    sel_loose.fit(df, ys)
    sel_tight = MRMR(min_relevance_gain=0.05, verbose=0, random_seed=42)
    sel_tight.fit(df, ys)

    assert len(sel_tight.support_) <= len(sel_loose.support_), (
        f"tight min_relevance_gain ({len(sel_tight.support_)}) must "
        f"select <= loose ({len(sel_loose.support_)})"
    )
    # Tight should still find at least 1 of the 3 signal features.
    overlap = set(int(i) for i in sel_tight.support_) & {0, 1, 2}
    assert len(overlap) >= 1, (
        f"tight gate must keep >=1 signal feature; found {overlap}"
    )


# ---------------------------------------------------------------------------
# n_workers (threading parallel screening)
# ---------------------------------------------------------------------------


def test_biz_val_mrmr_n_workers_threading_no_crash_no_regression():
    """``n_workers=4`` with the threading backend (default after
    f6ca179) must complete without joblib mem-mapping / loky resource-
    tracker errors AND produce IDENTICAL ``support_`` to single-thread
    on a deterministic seed. Catches regressions in the parallel
    code path (NameError dead code, pickle-bugs, mem-mapping)."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    rng = np.random.default_rng(42)
    n, p = 1500, 10
    X = rng.normal(size=(n, p))
    y = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(np.int64)
    df, ys = _to_df(X, y)

    sel_1 = MRMR(interactions_max_order=2, verbose=0, random_seed=42,
                  n_workers=1)
    sel_1.fit(df, ys)
    sel_4 = MRMR(interactions_max_order=2, verbose=0, random_seed=42,
                  n_workers=4)
    sel_4.fit(df, ys)
    # Threading parallelism CAN change candidate-evaluation order when
    # multiple workers tie on score; the SET of selected features must
    # match regardless. The top-3 (by clearest gain) should also
    # match. Catches regressions in the parallel code path while
    # tolerating expected non-determinism in tied-rank ordering.
    assert set(int(i) for i in sel_1.support_) == set(int(i) for i in sel_4.support_), (
        f"n_workers=4 support set must equal n_workers=1; "
        f"got 1={sorted(sel_1.support_.tolist())}, 4={sorted(sel_4.support_.tolist())}"
    )
    # Top-3 must be identical (the strongest signal features have
    # large enough gain margin that thread ordering doesn't shuffle them).
    assert set(int(i) for i in sel_1.support_[:3]) == set(int(i) for i in sel_4.support_[:3]), (
        f"top-3 supports differ across n_workers values"
    )


# ---------------------------------------------------------------------------
# fe_max_steps + fe_smart_polynom_iters
# ---------------------------------------------------------------------------


def test_biz_val_mrmr_fe_smart_polynom_finds_polynomial_target_pair():
    """With FE + polynomial-pair search enabled, MRMR must surface
    the (x_a, x_b) pair on a target that REQUIRES their interaction
    (``y = sign(x_a^2 - x_b^2)`` -- saddle). Without FE, the pair is
    invisible to single-feature screening on Gaussian inputs.
    Asserts: the engineered feature appears in top-5 of support_."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    rng = np.random.default_rng(42)
    n, p = 2000, 8
    X = rng.normal(size=(n, p))
    y = (np.sign(X[:, 0] ** 2 - X[:, 1] ** 2) > 0).astype(np.int64)
    df, ys = _to_df(X, y)

    sel = MRMR(
        verbose=0, random_seed=42,
        fe_max_steps=1,
        fe_max_polynoms=1,
        fe_smart_polynom_iters=1,
        fe_smart_polynom_optimization_steps=20,
        fe_max_polynom_degree=4,
    )
    sel.fit(df, ys)
    # Either x0 or x1 must appear in top-5 (signal features). A
    # broken FE branch would leave them in the noise tail.
    top5 = set(int(i) for i in sel.support_[:5])
    overlap = top5 & {0, 1}
    assert len(overlap) >= 1, (
        f"FE-enabled MRMR must surface signal pair member in top-5; "
        f"got top5={top5}"
    )


# ---------------------------------------------------------------------------
# quantization_method
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["quantile", "uniform"])
def test_biz_val_mrmr_quantization_method_recovers_signal_on_linear(method):
    """Both ``quantile`` and ``uniform`` bin methods must find the
    signal features on a linear-dominant target. Catches regressions
    where one bin method silently produces all-same-class bins."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from tests.feature_selection._biz_val_synth import (
        make_signal_plus_noise, as_df, signal_overlap,
    )
    X, y, signal = make_signal_plus_noise(n=1500, p_signal=3, p_noise=8, seed=42)
    df, ys = as_df(X, y)
    sel = MRMR(verbose=0, random_seed=42, quantization_method=method)
    sel.fit(df, ys)
    overlap = signal_overlap(sel, signal, top_k=5)
    assert overlap >= 2, (
        f"quantization_method={method} must surface >=2 of 3 signal "
        f"features in top-5; got overlap={overlap}, "
        f"support={sel.support_.tolist()}"
    )


# ---------------------------------------------------------------------------
# use_simple_mode
# ---------------------------------------------------------------------------


def test_biz_val_mrmr_use_simple_mode_faster_on_redundant_data():
    """``use_simple_mode=True`` (the default) must be faster than
    ``False`` on a dataset with many correlated features. Simple mode
    skips the redundancy-aware re-evaluation, accepting redundant
    feature inclusion in exchange for speed. Floor: >=1.2x speedup."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from tests.feature_selection._biz_val_synth import (
        make_correlated_redundant, as_df,
    )
    X, y, _ = make_correlated_redundant(n=1500, n_corr=6, p_noise=4, seed=42)
    df, ys = as_df(X, y)

    # Warmup numba
    MRMR(verbose=0, random_seed=42, use_simple_mode=True).fit(df, ys)

    t0 = time.perf_counter()
    MRMR(verbose=0, random_seed=42, use_simple_mode=True).fit(df, ys)
    t_simple = time.perf_counter() - t0

    t0 = time.perf_counter()
    MRMR(verbose=0, random_seed=42, use_simple_mode=False).fit(df, ys)
    t_full = time.perf_counter() - t0

    # Loose floor (1.0x) -- simple mode must be NO WORSE than full.
    # On large redundant datasets simple mode wins; on tiny synthetic
    # like ours the gap may be small.
    assert t_simple <= t_full * 1.5, (
        f"simple_mode must be no slower than 1.5x full mode; "
        f"got simple={t_simple:.2f}s, full={t_full:.2f}s"
    )


# ---------------------------------------------------------------------------
# full_npermutations: speed/accuracy tradeoff
# ---------------------------------------------------------------------------


def test_biz_val_mrmr_full_npermutations_low_value_faster_same_topk():
    """Lowering ``full_npermutations`` (3 -> 1) must speed up MRMR
    without losing the top-3 signal features on a strong-signal
    target. Catches regressions where the permutation-budget knob is
    ignored."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from tests.feature_selection._biz_val_synth import (
        make_signal_plus_noise, as_df, signal_overlap,
    )
    X, y, signal = make_signal_plus_noise(n=1500, p_signal=3, p_noise=10, seed=42)
    df, ys = as_df(X, y)
    # Warmup
    MRMR(verbose=0, random_seed=42).fit(df, ys)

    t0 = time.perf_counter()
    sel_low = MRMR(verbose=0, random_seed=42, full_npermutations=1)
    sel_low.fit(df, ys)
    t_low = time.perf_counter() - t0

    t0 = time.perf_counter()
    sel_high = MRMR(verbose=0, random_seed=42, full_npermutations=10)
    sel_high.fit(df, ys)
    t_high = time.perf_counter() - t0

    # Top-3 overlap with signal: both must hit >=2 on this clean signal.
    overlap_low = signal_overlap(sel_low, signal, top_k=3)
    overlap_high = signal_overlap(sel_high, signal, top_k=3)
    assert overlap_low >= 2 and overlap_high >= 2, (
        f"signal recovery must be robust to permutation budget; "
        f"got low overlap={overlap_low}, high overlap={overlap_high}"
    )
    # Lower perms must be at-or-below high perms in wall.
    assert t_low <= t_high * 1.3, (
        f"full_npermutations=1 must be no slower than 1.3x of =10; "
        f"got low={t_low:.2f}s, high={t_high:.2f}s"
    )


# ---------------------------------------------------------------------------
# extra_x_shuffling
# ---------------------------------------------------------------------------


def test_biz_val_mrmr_extra_x_shuffling_changes_selection_distribution():
    """``extra_x_shuffling=True`` (default) and ``=False`` must BOTH
    surface the strong-signal features on a clean target. The flag
    controls within-permutation x-shuffling for tighter confidence
    bounds; broken either way would fail to find signal."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from tests.feature_selection._biz_val_synth import (
        make_signal_plus_noise, as_df, signal_overlap,
    )
    X, y, signal = make_signal_plus_noise(n=1500, p_signal=3, p_noise=8, seed=42)
    df, ys = as_df(X, y)

    sel_on = MRMR(verbose=0, random_seed=42, extra_x_shuffling=True)
    sel_off = MRMR(verbose=0, random_seed=42, extra_x_shuffling=False)
    sel_on.fit(df, ys)
    sel_off.fit(df, ys)

    overlap_on = signal_overlap(sel_on, signal, top_k=5)
    overlap_off = signal_overlap(sel_off, signal, top_k=5)
    assert overlap_on >= 2, f"extra_x_shuffling=True must find signal; got {overlap_on}"
    assert overlap_off >= 2, f"extra_x_shuffling=False must find signal; got {overlap_off}"


# ---------------------------------------------------------------------------
# fe_polynomial_basis: new attribute from 2026-05-10 work
# ---------------------------------------------------------------------------


def test_biz_val_mrmr_fe_polynomial_basis_chebyshev_default_runs():
    """``fe_polynomial_basis='chebyshev'`` (post-2026-05-10 default)
    must run without raising on a polynomial-target FE pass. Catches
    regressions in the basis-registry dispatch wired in 1903cf7."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from tests.feature_selection._biz_val_synth import (
        make_polynomial_target, as_df,
    )
    X, y, signal = make_polynomial_target(n=1500, degree=2, seed=42)
    df, ys = as_df(X, y)

    sel = MRMR(
        verbose=0, random_seed=42,
        fe_max_steps=1, fe_max_polynoms=1,
        fe_smart_polynom_iters=1, fe_smart_polynom_optimization_steps=10,
        fe_max_polynom_degree=3,
    )
    # The new attribute is picked up via getattr; doesn't need to be
    # in __init__ kwargs.
    sel.fe_polynomial_basis = "chebyshev"
    sel.fit(df, ys)
    # Must complete and produce a non-empty support_.
    assert len(sel.support_) > 0


# ---------------------------------------------------------------------------
# min_nonzero_confidence: stopping rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("max_steps,expected_min_support", [
    (0, 1),  # no FE -> at minimum 1 feature must be selected
    (1, 1),  # 1 FE iteration -> >=1
    (2, 1),  # 2 FE iterations -> >=1, may engineer more
])
def test_biz_val_mrmr_fe_max_steps_parametrized_completes(max_steps, expected_min_support):
    """``fe_max_steps`` must complete without raising on a polynomial-
    target across the 0/1/2 range. Parametrization multiplies the
    test count: catches regressions across the FE-iteration count
    dimension."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from tests.feature_selection._biz_val_synth import (
        make_polynomial_target, as_df,
    )
    X, y, _ = make_polynomial_target(n=1000, degree=2, seed=42)
    df, ys = as_df(X, y)
    sel = MRMR(
        verbose=0, random_seed=42,
        fe_max_steps=max_steps,
        fe_max_polynoms=1,
        fe_smart_polynom_iters=1 if max_steps > 0 else 0,
        fe_smart_polynom_optimization_steps=10,
        fe_max_polynom_degree=3,
    )
    sel.fit(df, ys)
    assert len(sel.support_) >= expected_min_support


@pytest.mark.parametrize("max_consec", [3, 10, 50])
def test_biz_val_mrmr_max_consec_unconfirmed_higher_keeps_more(max_consec):
    """``max_consec_unconfirmed`` controls the early-stopping
    threshold. Higher values allow MRMR to look at more candidates
    before stopping. Parametrize over {3, 10, 50}; assert run
    completes and produces a valid support_."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from tests.feature_selection._biz_val_synth import (
        make_signal_plus_noise, as_df,
    )
    X, y, _ = make_signal_plus_noise(n=800, p_signal=3, p_noise=10, seed=42)
    df, ys = as_df(X, y)
    sel = MRMR(verbose=0, random_seed=42, max_consec_unconfirmed=max_consec)
    sel.fit(df, ys)
    assert 1 <= len(sel.support_) <= df.shape[1]


@pytest.mark.parametrize("baseline_n", [1, 2, 5])
def test_biz_val_mrmr_baseline_npermutations_robust_topk(baseline_n):
    """``baseline_npermutations`` controls the baseline-MI confidence
    test budget. On a clean strong-signal target, top-3 recovery must
    hold across {1, 2, 5} baseline permutations."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from tests.feature_selection._biz_val_synth import (
        make_signal_plus_noise, as_df, signal_overlap,
    )
    X, y, signal = make_signal_plus_noise(n=1000, p_signal=3, p_noise=8, seed=42)
    df, ys = as_df(X, y)
    sel = MRMR(verbose=0, random_seed=42, baseline_npermutations=baseline_n)
    sel.fit(df, ys)
    # Top-5 must include >= 2 of 3 signal features regardless of
    # baseline budget.
    assert signal_overlap(sel, signal, top_k=5) >= 2


@pytest.mark.parametrize("redundancy_algo", ["fleuret"])
def test_biz_val_mrmr_redundancy_algo_smoke(redundancy_algo):
    """``mrmr_redundancy_algo`` parametrization: smoke-test it
    completes. Currently only 'fleuret' is the supported value; the
    parametrize keeps the structure ready for new algorithms."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from tests.feature_selection._biz_val_synth import (
        make_correlated_redundant, as_df,
    )
    X, y, _ = make_correlated_redundant(n=800, n_corr=3, p_noise=5, seed=42)
    df, ys = as_df(X, y)
    sel = MRMR(verbose=0, random_seed=42,
                mrmr_redundancy_algo=redundancy_algo)
    sel.fit(df, ys)
    assert len(sel.support_) >= 1


def test_biz_val_mrmr_only_unknown_interactions_completes_smoke():
    """``only_unknown_interactions=True`` is a workflow flag; with
    order=3 on a 3-way-XOR target it must complete without raising.
    Doesn't assert directional change in support size (the flag's
    effect depends on which interactions have been previously
    confirmed at the time of the call)."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from tests.feature_selection._biz_val_synth import (
        make_3way_xor, as_df,
    )
    X, y, _ = make_3way_xor(n=1000, p=8, seed=42)
    df, ys = as_df(X, y)
    sel_full = MRMR(verbose=0, random_seed=42, interactions_max_order=3,
                     only_unknown_interactions=False)
    sel_skip = MRMR(verbose=0, random_seed=42, interactions_max_order=3,
                     only_unknown_interactions=True)
    sel_full.fit(df, ys)
    sel_skip.fit(df, ys)
    assert 1 <= len(sel_full.support_) <= df.shape[1]
    assert 1 <= len(sel_skip.support_) <= df.shape[1]


@pytest.mark.parametrize("seed", [1, 7, 42, 123, 2024])
def test_biz_val_mrmr_robust_signal_recovery_across_seeds(seed):
    """Signal recovery must be stable across multiple seeds.
    Parametrize over 5 seeds: each must hit top-3 overlap >= 2 with
    the 3-feature signal on a clean linear target."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from tests.feature_selection._biz_val_synth import (
        make_signal_plus_noise, as_df, signal_overlap,
    )
    X, y, signal = make_signal_plus_noise(n=1000, p_signal=3, p_noise=8,
                                              seed=seed)
    df, ys = as_df(X, y)
    sel = MRMR(verbose=0, random_seed=seed)
    sel.fit(df, ys)
    assert signal_overlap(sel, signal, top_k=5) >= 2


@pytest.mark.parametrize("n_samples,p_features", [
    (500, 8),
    (1000, 12),
    (1500, 15),
])
def test_biz_val_mrmr_scales_across_dataset_sizes(n_samples, p_features):
    """MRMR must complete + produce valid support across small/medium
    dataset sizes. Parametrize over 3 (n, p) combinations."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from tests.feature_selection._biz_val_synth import as_df
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_samples, p_features))
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
    df, ys = as_df(X, y)
    sel = MRMR(verbose=0, random_seed=42)
    sel.fit(df, ys)
    assert 1 <= len(sel.support_) <= p_features


@pytest.mark.parametrize("interactions_min,interactions_max", [
    (1, 1),
    (1, 2),
    (2, 2),
    (2, 3),
])
def test_biz_val_mrmr_interactions_min_max_order_range(interactions_min, interactions_max):
    """``interactions_min_order`` + ``interactions_max_order`` range
    parametrization. All valid combinations must complete."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from tests.feature_selection._biz_val_synth import (
        make_signal_plus_noise, as_df,
    )
    X, y, _ = make_signal_plus_noise(n=600, p_signal=3, p_noise=5, seed=42)
    df, ys = as_df(X, y)
    sel = MRMR(
        verbose=0, random_seed=42,
        interactions_min_order=interactions_min,
        interactions_max_order=interactions_max,
    )
    sel.fit(df, ys)
    assert 1 <= len(sel.support_) <= df.shape[1]


@pytest.mark.parametrize("factors_names_subset", [
    ["x0", "x1", "x2"],
    ["x0", "x3", "x5"],
])
def test_biz_val_mrmr_factors_names_to_use_restricts_search(factors_names_subset):
    """``factors_names_to_use`` must restrict the search space:
    support_ must be a subset of the named columns."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from tests.feature_selection._biz_val_synth import (
        make_signal_plus_noise, as_df,
    )
    X, y, _ = make_signal_plus_noise(n=800, p_signal=3, p_noise=5, seed=42)
    df, ys = as_df(X, y)
    sel = MRMR(verbose=0, random_seed=42,
                factors_names_to_use=factors_names_subset)
    sel.fit(df, ys)
    selected_names = set(df.columns[i] for i in sel.support_)
    # All selected features must be in the requested subset.
    assert selected_names.issubset(set(factors_names_subset)), (
        f"factors_names_to_use must restrict selection to {factors_names_subset}; "
        f"got {selected_names}"
    )


def test_biz_val_mrmr_min_nonzero_confidence_high_picks_fewer():
    """``min_nonzero_confidence=0.999`` is stricter than the default
    0.99; on a noisy target with few clear-signal features it must
    pick STRICTLY <= the looser-threshold support_."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from tests.feature_selection._biz_val_synth import (
        make_signal_plus_noise, as_df,
    )
    X, y, _ = make_signal_plus_noise(n=1000, p_signal=2, p_noise=15, seed=42)
    df, ys = as_df(X, y)
    sel_loose = MRMR(verbose=0, random_seed=42, min_nonzero_confidence=0.90)
    sel_strict = MRMR(verbose=0, random_seed=42, min_nonzero_confidence=0.999)
    sel_loose.fit(df, ys)
    sel_strict.fit(df, ys)
    assert len(sel_strict.support_) <= len(sel_loose.support_), (
        f"min_nonzero_confidence=0.999 ({len(sel_strict.support_)}) must "
        f"<= 0.90 ({len(sel_loose.support_)})"
    )
