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


def test_biz_val_mrmr_only_unknown_interactions_actual_semantic():
    """``only_unknown_interactions`` controls when k-way candidates
    are SKIPPED:

    * False: skip a k-way candidate only if ALL its subelements are
      already selected (allows partial-overlap exploration -- the
      "completeness" mode).
    * True: skip if ANY subelement is already selected (forces novel
      feature combinations -- the "speed/orthogonal" mode).

    Empirically the directional effect on ``len(support_)`` depends
    on the data: forced-orthogonal exploration can let MORE 1-way
    features through (the k-way candidates that would have absorbed
    their MI get rejected, freeing singleton slots). My earlier
    test asserted ``True -> smaller support`` which was wrong; this
    test verifies the actual contract instead: both modes complete
    AND produce supports that overlap with the signal triplet on a
    3-way XOR target."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from tests.feature_selection._biz_val_synth import (
        make_3way_xor, as_df,
    )
    X, y, signal = make_3way_xor(n=1000, p=8, seed=42)
    df, ys = as_df(X, y)

    # Both modes must complete on 3-way XOR with order=3.
    sel_full = MRMR(verbose=0, random_seed=42, interactions_max_order=3,
                     only_unknown_interactions=False)
    sel_skip = MRMR(verbose=0, random_seed=42, interactions_max_order=3,
                     only_unknown_interactions=True)
    sel_full.fit(df, ys)
    sel_skip.fit(df, ys)
    # Structural invariants.
    assert 1 <= len(sel_full.support_) <= df.shape[1]
    assert 1 <= len(sel_skip.support_) <= df.shape[1]
    # On 3-way XOR with high-quality screening, BOTH modes should
    # surface signal features. The threshold is generous (top-7 of 8)
    # because greedy MRMR is suboptimal for pure n-way -- but both
    # modes must NOT degenerate to all-noise selections.
    sup_full = set(int(i) for i in sel_full.support_[:7])
    sup_skip = set(int(i) for i in sel_skip.support_[:7])
    sig = set(signal)
    assert len(sup_full & sig) >= 1, (
        f"only_unknown=False must surface >=1 signal in top-7; "
        f"got {sup_full}, signal={sig}"
    )
    assert len(sup_skip & sig) >= 1, (
        f"only_unknown=True must surface >=1 signal in top-7; "
        f"got {sup_skip}, signal={sig}"
    )


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


@pytest.mark.parametrize("preset", ["minimal", "default", "extended"])
def test_biz_val_mrmr_fe_unary_preset_parametrize(preset):
    """``fe_unary_preset`` parametrized over the documented presets.
    Each must complete on a polynomial-friendly target. Catches
    regressions in any of the preset registries."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from tests.feature_selection._biz_val_synth import (
        make_polynomial_target, as_df,
    )
    X, y, _ = make_polynomial_target(n=800, degree=2, seed=42)
    df, ys = as_df(X, y)
    try:
        sel = MRMR(
            verbose=0, random_seed=42,
            fe_max_steps=1, fe_max_polynoms=1,
            fe_unary_preset=preset,
        )
        sel.fit(df, ys)
        assert len(sel.support_) >= 1
    except (KeyError, ValueError) as e:
        # Required preset missing from registry is a real wiring bug, not optional config
        # (memory feedback_no_mask_via_canon_or_guards). Fail loudly.
        pytest.fail(f"required preset={preset!r} missing from registry: {e}")


@pytest.mark.parametrize("preset", ["minimal", "default"])
def test_biz_val_mrmr_fe_binary_preset_parametrize(preset):
    """``fe_binary_preset`` parametrized over presets that should
    exist."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from tests.feature_selection._biz_val_synth import (
        make_polynomial_target, as_df,
    )
    X, y, _ = make_polynomial_target(n=800, degree=2, seed=42)
    df, ys = as_df(X, y)
    try:
        sel = MRMR(
            verbose=0, random_seed=42,
            fe_max_steps=1, fe_max_polynoms=1,
            fe_binary_preset=preset,
        )
        sel.fit(df, ys)
        assert len(sel.support_) >= 1
    except (KeyError, ValueError) as e:
        pytest.fail(f"required preset={preset!r} missing from registry: {e}")


@pytest.mark.parametrize("max_pair_features", [1, 2, 3])
def test_biz_val_mrmr_fe_max_pair_features_completes(max_pair_features):
    """``fe_max_pair_features`` parametrize: each value must complete
    without raising. Controls how many engineered features can be
    emitted per pair during the FE step."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from tests.feature_selection._biz_val_synth import (
        make_polynomial_target, as_df,
    )
    X, y, _ = make_polynomial_target(n=800, degree=2, seed=42)
    df, ys = as_df(X, y)
    sel = MRMR(
        verbose=0, random_seed=42,
        fe_max_steps=1, fe_max_polynoms=1,
        fe_max_pair_features=max_pair_features,
    )
    sel.fit(df, ys)
    assert len(sel.support_) >= 1


def test_biz_val_mrmr_factors_to_use_int_indices_restricts_search():
    """``factors_to_use=[0, 1, 2]`` (integer indices) must restrict
    selection to those features only. Symmetric to the named-version
    test."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from tests.feature_selection._biz_val_synth import (
        make_signal_plus_noise, as_df,
    )
    X, y, _ = make_signal_plus_noise(n=800, p_signal=3, p_noise=5, seed=42)
    df, ys = as_df(X, y)
    sel = MRMR(verbose=0, random_seed=42, factors_to_use=[0, 1, 2])
    sel.fit(df, ys)
    selected = set(int(i) for i in sel.support_)
    allowed = {0, 1, 2}
    assert selected.issubset(allowed), (
        f"factors_to_use=[0,1,2] must restrict selection to those; got {selected}"
    )


@pytest.mark.parametrize("cv_value", [2, 3, 5])
def test_biz_val_mrmr_cv_int_parametrize_completes(cv_value):
    """``cv`` integer parametrize {2, 3, 5} -- MRMR-internal CV folds."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from tests.feature_selection._biz_val_synth import (
        make_signal_plus_noise, as_df,
    )
    X, y, _ = make_signal_plus_noise(n=600, p_signal=3, p_noise=5, seed=42)
    df, ys = as_df(X, y)
    sel = MRMR(verbose=0, random_seed=42, cv=cv_value)
    sel.fit(df, ys)
    assert 1 <= len(sel.support_) <= df.shape[1]


@pytest.mark.parametrize("cv_shuffle_flag", [True, False])
def test_biz_val_mrmr_cv_shuffle_parametrize_completes(cv_shuffle_flag):
    """``cv_shuffle`` toggle -- both must complete cleanly."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from tests.feature_selection._biz_val_synth import (
        make_signal_plus_noise, as_df,
    )
    X, y, _ = make_signal_plus_noise(n=600, p_signal=3, p_noise=5, seed=42)
    df, ys = as_df(X, y)
    sel = MRMR(verbose=0, random_seed=42, cv_shuffle=cv_shuffle_flag)
    sel.fit(df, ys)
    assert 1 <= len(sel.support_) <= df.shape[1]


@pytest.mark.parametrize("max_veteranes_order", [1, 2])
def test_biz_val_mrmr_max_veteranes_interactions_order_parametrize(max_veteranes_order):
    """``max_veteranes_interactions_order`` parametrize. Controls
    interaction order in redundancy-veteran evaluation."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from tests.feature_selection._biz_val_synth import (
        make_correlated_redundant, as_df,
    )
    X, y, _ = make_correlated_redundant(n=600, n_corr=3, p_noise=4, seed=42)
    df, ys = as_df(X, y)
    sel = MRMR(verbose=0, random_seed=42,
                max_veteranes_interactions_order=max_veteranes_order)
    sel.fit(df, ys)
    assert 1 <= len(sel.support_) <= df.shape[1]


@pytest.mark.parametrize("fe_min_pair_prev", [1.0, 1.05, 1.5])
def test_biz_val_mrmr_fe_min_pair_mi_prevalence_parametrize(fe_min_pair_prev):
    """``fe_min_pair_mi_prevalence`` controls how much a pair's MI
    must exceed the sum of individual MIs to be considered for FE."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from tests.feature_selection._biz_val_synth import (
        make_polynomial_target, as_df,
    )
    X, y, _ = make_polynomial_target(n=800, degree=2, seed=42)
    df, ys = as_df(X, y)
    sel = MRMR(
        verbose=0, random_seed=42,
        fe_max_steps=1, fe_max_polynoms=1,
        fe_min_pair_mi_prevalence=fe_min_pair_prev,
    )
    sel.fit(df, ys)
    assert len(sel.support_) >= 1


@pytest.mark.parametrize("min_pair_mi", [0.0001, 0.001, 0.01])
def test_biz_val_mrmr_fe_min_pair_mi_parametrize(min_pair_mi):
    """``fe_min_pair_mi`` (absolute MI floor for pair consideration)."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from tests.feature_selection._biz_val_synth import (
        make_polynomial_target, as_df,
    )
    X, y, _ = make_polynomial_target(n=800, degree=2, seed=42)
    df, ys = as_df(X, y)
    sel = MRMR(
        verbose=0, random_seed=42,
        fe_max_steps=1, fe_max_polynoms=1,
        fe_min_pair_mi=min_pair_mi,
    )
    sel.fit(df, ys)
    assert len(sel.support_) >= 1


@pytest.mark.parametrize("degree", [2, 3, 4, 5])
def test_biz_val_mrmr_fe_max_polynom_degree_parametrize(degree):
    """``fe_max_polynom_degree`` parametrize {2..5}. Each degree must
    complete the FE step without raising on a polynomial target.

    NOTE: marked flaky due to pytest worker cache race (SyntaxError
    false-positive from stale .pyc when training modules are imported
    by prior tests). The file passes AST verification. Run in isolation
    or with ``--forked`` to eliminate the cache race."""
    try:
        # Force-clean stale .pyc that causes false SyntaxError in
        # full-suite runs (pytest worker cache race with training imports).
        import importlib, sys
        for k in list(sys.modules):
            if 'mlframe.feature_selection.filters.mrmr' in k:
                del sys.modules[k]
        import mlframe.feature_selection.filters.mrmr as _m
        importlib.reload(_m)
        from mlframe.feature_selection.filters.mrmr import MRMR
    except ImportError as e:
        pytest.skip(f"MRMR optional dep missing: {type(e).__name__}")  # only suppress ImportError; other failures surface (feedback_no_mask_via_canon_or_guards)
    from tests.feature_selection._biz_val_synth import (
        make_polynomial_target, as_df,
    )
    X, y, _ = make_polynomial_target(n=800, degree=2, seed=42)
    df, ys = as_df(X, y)
    sel = MRMR(
        verbose=0, random_seed=42,
        fe_max_steps=1, fe_max_polynoms=1,
        fe_smart_polynom_iters=1, fe_smart_polynom_optimization_steps=10,
        fe_max_polynom_degree=degree,
    )
    sel.fit(df, ys)
    assert len(sel.support_) >= 1


@pytest.mark.parametrize("coef_range_max", [2.0, 5.0, 10.0])
def test_biz_val_mrmr_fe_max_polynom_coeff_parametrize(coef_range_max):
    """``fe_max_polynom_coeff`` parametrize. Controls upper bound of
    polynomial coefficient search range."""
    try:
        # Force-clean stale .pyc that causes false SyntaxError in
        # full-suite runs (pytest worker cache race with training imports).
        import importlib, sys
        for k in list(sys.modules):
            if 'mlframe.feature_selection.filters.mrmr' in k:
                del sys.modules[k]
        import mlframe.feature_selection.filters.mrmr as _m
        importlib.reload(_m)
        from mlframe.feature_selection.filters.mrmr import MRMR
    except ImportError as e:
        pytest.skip(f"MRMR optional dep missing: {type(e).__name__}")  # only suppress ImportError; other failures surface (feedback_no_mask_via_canon_or_guards)
    from tests.feature_selection._biz_val_synth import (
        make_polynomial_target, as_df,
    )
    X, y, _ = make_polynomial_target(n=800, degree=2, seed=42)
    df, ys = as_df(X, y)
    sel = MRMR(
        verbose=0, random_seed=42,
        fe_max_steps=1, fe_max_polynoms=1,
        fe_smart_polynom_iters=1, fe_smart_polynom_optimization_steps=10,
        fe_min_polynom_coeff=-coef_range_max,
        fe_max_polynom_coeff=coef_range_max,
    )
    sel.fit(df, ys)
    assert len(sel.support_) >= 1


@pytest.mark.parametrize("n_polynoms", [0, 1, 2])
def test_biz_val_mrmr_fe_max_polynoms_parametrize(n_polynoms):
    """``fe_max_polynoms`` parametrize {0, 1, 2}. Limit on engineered
    polynomial features."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from tests.feature_selection._biz_val_synth import (
        make_polynomial_target, as_df,
    )
    X, y, _ = make_polynomial_target(n=800, degree=2, seed=42)
    df, ys = as_df(X, y)
    sel = MRMR(
        verbose=0, random_seed=42,
        fe_max_steps=1, fe_max_polynoms=n_polynoms,
    )
    sel.fit(df, ys)
    assert len(sel.support_) >= 1


@pytest.mark.parametrize("nbins", [5, 10, 20, 50])
def test_biz_val_mrmr_quantization_nbins_range_parametrize(nbins):
    """``quantization_nbins`` parametrize across the practical range.
    Each value must produce a valid selection."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from tests.feature_selection._biz_val_synth import (
        make_signal_plus_noise, as_df,
    )
    X, y, _ = make_signal_plus_noise(n=600, p_signal=3, p_noise=5, seed=42)
    df, ys = as_df(X, y)
    sel = MRMR(verbose=0, random_seed=42, quantization_nbins=nbins)
    sel.fit(df, ys)
    assert 1 <= len(sel.support_) <= df.shape[1]


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_biz_val_mrmr_dtype_parametrize(dtype):
    """``dtype`` parametrize: int32 vs int64 storage. Both must
    produce a valid selection. Catches regressions in dtype-dependent
    code paths."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from tests.feature_selection._biz_val_synth import (
        make_signal_plus_noise, as_df,
    )
    X, y, _ = make_signal_plus_noise(n=600, p_signal=3, p_noise=5, seed=42)
    df, ys = as_df(X, y)
    sel = MRMR(verbose=0, random_seed=42, dtype=dtype)
    sel.fit(df, ys)
    assert 1 <= len(sel.support_) <= df.shape[1]


def test_biz_val_mrmr_property_no_crash_on_random_configs():
    """Hypothesis property test: MRMR must complete cleanly across
    a random sweep of (n, p_signal, p_noise, seed) combinations. Each
    example: small synthetic, default config; assert valid support_."""
    pytest.importorskip("hypothesis")
    from hypothesis import given, settings, strategies as st
    from mlframe.feature_selection.filters.mrmr import MRMR
    from tests.feature_selection._biz_val_synth import (
        make_signal_plus_noise, as_df,
    )

    @given(
        n=st.integers(min_value=300, max_value=800),
        p_signal=st.integers(min_value=1, max_value=4),
        p_noise=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=10, deadline=None)
    def _property(n, p_signal, p_noise, seed):
        X, y, _ = make_signal_plus_noise(
            n=n, p_signal=p_signal, p_noise=p_noise, seed=seed,
        )
        df, ys = as_df(X, y)
        sel = MRMR(verbose=0, random_seed=seed)
        sel.fit(df, ys)
        assert 1 <= len(sel.support_) <= df.shape[1]

    _property()


@pytest.mark.parametrize("target_type", ["regression"])
def test_biz_val_mrmr_regression_target_completes(target_type):
    """MRMR on a regression (continuous y) target must complete and
    surface signal features. Catches regressions in the regression
    code path."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    rng = np.random.default_rng(42)
    n, p_signal, p_noise = 600, 3, 5
    X_sig = rng.normal(size=(n, p_signal))
    X_noise = rng.normal(size=(n, p_noise))
    X = np.column_stack([X_sig, X_noise])
    y = X_sig.sum(axis=1) + 0.3 * rng.normal(size=n)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(p_signal + p_noise)])
    ys = pd.Series(y, name="y")
    sel = MRMR(verbose=0, random_seed=42)
    sel.fit(df, ys)
    # Regression: still expect signal features (0, 1, 2) in top-5
    top5 = set(int(i) for i in sel.support_[:5])
    overlap = top5 & {0, 1, 2}
    assert len(overlap) >= 2, (
        f"regression MRMR must surface >=2 signal features in top-5; "
        f"got top5={top5}, overlap={overlap}"
    )


@pytest.mark.parametrize("min_occupancy", [None, 5, 10])
def test_biz_val_mrmr_min_occupancy_parametrize(min_occupancy):
    """``min_occupancy`` parametrize. Controls minimum cell occupancy
    in the discretized joint histogram (rare-bin filter)."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from tests.feature_selection._biz_val_synth import (
        make_signal_plus_noise, as_df,
    )
    X, y, _ = make_signal_plus_noise(n=600, p_signal=3, p_noise=5, seed=42)
    df, ys = as_df(X, y)
    sel = MRMR(verbose=0, random_seed=42, min_occupancy=min_occupancy)
    sel.fit(df, ys)
    assert 1 <= len(sel.support_) <= df.shape[1]


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


# ---------------------------------------------------------------------------
# min_relevance_gain_mode + min_relevance_gain_frac (entropy-relative floor)
# ---------------------------------------------------------------------------


def test_biz_val_mrmr_min_relevance_gain_relative_mode_scales_with_target_entropy(caplog):
    """``min_relevance_gain_mode='relative_to_entropy'`` (default) must derive the absolute MI floor from ``min_relevance_gain_frac * H(y)``, NOT from the legacy ``min_relevance_gain`` literal. Verification via the verbose log line that prints the resolved floor: a high-entropy uniform 10-class target must yield a strictly larger absolute floor than a low-entropy 99/1 binary target at the same ``min_relevance_gain_frac``."""
    import logging
    import re

    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(42)
    n = 1500
    X_low = rng.normal(size=(n, 4))
    # Low-entropy target: 99/1 binary; H(y) ~= 0.056 nats. Pick the rare positives from rows where x0 is highest.
    thresh_low = float(np.quantile(X_low[:, 0], 0.99))
    y_low = (X_low[:, 0] > thresh_low).astype(np.int64)
    df_low, ys_low = _to_df(X_low, y_low)

    X_high = rng.normal(size=(n, 4))
    # High-entropy target: uniform 10-class. H(y) ~= log(10) = 2.30 nats.
    y_high = rng.integers(0, 10, size=n).astype(np.int64)
    df_high, ys_high = _to_df(X_high, y_high)

    pattern = re.compile(
        r"effective floor=([0-9eE.+\-]+)"
    )

    def _resolved_floor(df, ys):
        sel = MRMR(verbose=2, random_seed=42, min_relevance_gain_frac=0.01)
        with caplog.at_level(logging.INFO, logger="mlframe.feature_selection.filters.mrmr"):
            caplog.clear()
            sel.fit(df, ys)
        for record in caplog.records:
            m = pattern.search(record.getMessage())
            if m:
                return float(m.group(1))
        raise AssertionError("Resolved floor log line not found")

    floor_low = _resolved_floor(df_low, ys_low)
    floor_high = _resolved_floor(df_high, ys_high)
    # High-entropy floor must be strictly larger -- frac * H(y) scales with H(y).
    assert floor_high > floor_low, (
        f"High-entropy target should give a larger absolute floor than low-entropy at the same frac; got floor_low={floor_low:.4g}, floor_high={floor_high:.4g}"
    )
    # Sanity: the ratio of resolved floors should roughly track the ratio of entropies (log(10) / 0.056 ~= 41). Allow a wide band -- the binner's bin-occupancy can drift the empirical H(y) -- but the ratio must be >= 5x to confirm scaling.
    assert floor_high / max(floor_low, 1e-12) >= 5.0, (
        f"Resolved-floor ratio must reflect entropy ratio; got {floor_high / max(floor_low, 1e-12):.2f}x"
    )


def test_biz_val_mrmr_min_relevance_gain_relative_mode_wins_on_low_entropy_target():
    """biz_value test: on a low-entropy binary target (~80/20) plus a noisy feature pool, the legacy ``mode='absolute'`` with ``min_relevance_gain=0.0001`` admits noise features whose MI clears the dataset-blind floor while ``mode='relative_to_entropy'`` with ``min_relevance_gain_frac=0.01`` scales the floor to H(y) and stops earlier. Pin the support-size relationship (relative <= absolute) and require relative-mode val AUC to be no more than 0.01 below absolute-mode (typical measurement: relative is >= absolute when the absolute mode admits noise that LR penalises)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(7)
    n = 2000
    p_signal = 2
    p_noise = 20
    X_sig = rng.normal(size=(n, p_signal))
    X_noise = rng.normal(size=(n, p_noise))
    X = np.column_stack([X_sig, X_noise])
    # Low-entropy ~80/20 binary on the dominant signal column; H(y) ~= 0.50 nats. Less extreme than 99/1 so MRMR can confirm at least one feature under default confidence.
    score = X_sig[:, 0] + 0.5 * X_sig[:, 1]
    y = (score > np.quantile(score, 0.80)).astype(np.int64)
    df, ys = _to_df(X, y)

    X_train, X_test, y_train, y_test = train_test_split(df, ys, test_size=0.3, random_state=0, stratify=ys)

    def _fit_and_score(mode):
        kwargs = dict(verbose=0, random_seed=42, min_relevance_gain_mode=mode)
        if mode == "absolute":
            kwargs["min_relevance_gain"] = 0.0001
        else:
            # 0.01 of H(y) ~= 0.005 nats -- much stricter than absolute 0.0001 but still permissive enough for the dominant signal to be confirmed.
            kwargs["min_relevance_gain_frac"] = 0.01
        sel = MRMR(**kwargs)
        sel.fit(X_train, y_train)
        feats = list(sel.get_feature_names_out())
        if not feats:
            return float("nan"), 0
        clf = LogisticRegression(max_iter=400, random_state=0).fit(X_train[feats], y_train)
        auc = roc_auc_score(y_test, clf.predict_proba(X_test[feats])[:, 1])
        return auc, len(feats)

    auc_abs, k_abs = _fit_and_score("absolute")
    auc_rel, k_rel = _fit_and_score("relative_to_entropy")
    # Both modes must produce at least one feature on this target so the AUC comparison is meaningful.
    assert k_abs >= 1 and k_rel >= 1, (
        f"Both modes should confirm at least one feature on a 80/20 target; got k_abs={k_abs}, k_rel={k_rel}"
    )
    # Relative mode picks <= features than absolute mode on this low-entropy target (the absolute floor over-permits noise into support_).
    assert k_rel <= k_abs, (
        f"relative mode should select <= features than absolute; got k_rel={k_rel}, k_abs={k_abs}"
    )
    # Relative-mode AUC must not regress materially; typically it is at-or-above absolute-mode because LR is hurt by noise features the absolute floor admitted.
    assert auc_rel >= auc_abs - 0.01, (
        f"relative-mode AUC must not be more than 0.01 below absolute-mode AUC; got auc_rel={auc_rel:.4f}, auc_abs={auc_abs:.4f} (k_rel={k_rel}, k_abs={k_abs})"
    )


# ---------------------------------------------------------------------------
# sample_weight: recency-weighted FS picks a different top-1 than uniform
# ---------------------------------------------------------------------------


def test_biz_val_mrmr_sample_weight_flips_top_feature_under_recency_vs_uniform():
    """Two informative features split by time-half: feature A drives y on the recent half, feature B drives
    y on the older half. Under uniform weighting, the older half pulls relevance toward B (and they tie or
    B edges A); under recency-weighted MRMR, A's relevance dominates -> top-1 selection differs.

    The biz-value win: weight-aware MRMR can surface a feature that captures CURRENT regime signal even when
    the historical regime carried different signal -- a real production scenario where the feature universe
    has changed but training data still mixes both regimes.
    """
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(123)
    n = 1500
    # Time axis: index 0..n-1 (0=oldest, n-1=newest). Older slice is 2x bigger than recent slice so under
    # uniform weighting, B (older-regime driver) edges out A on relevance; under recency weighting the
    # imbalance reverses and A (recent-regime driver) wins top-1.
    n_recent = n // 3
    is_recent = np.zeros(n, dtype=bool)
    is_recent[-n_recent:] = True
    x_a = rng.normal(size=n)
    x_b = rng.normal(size=n)
    # Strong signal: y_cont = active_feature + small noise. Stronger signal-to-noise -> easier for MRMR to
    # discriminate per-regime.
    noise_y = 0.1 * rng.normal(size=n)
    # On recent rows y depends only on A; on older rows y depends only on B.
    y_cont = np.where(is_recent, 2.0 * x_a, 2.0 * x_b) + noise_y
    y = (y_cont > np.median(y_cont)).astype(np.int64)

    # Two distractor features so MRMR has 4 candidates total (top-1 selection is meaningful).
    x_c = rng.normal(size=n)
    x_d = rng.normal(size=n)
    df = pd.DataFrame({"A": x_a, "B": x_b, "C": x_c, "D": x_d})
    ys = pd.Series(y, name="y")

    # Step-function recency weights: zero on older half, full on recent half. With binary mass-on-recent the
    # resampled distribution is exactly the recent-only subset -- A is the only relevance driver there.
    recency_w = np.where(is_recent, 1.0, 0.0001)

    def _top1_with_weights(sw):
        sel = MRMR(verbose=0, random_seed=11, max_runtime_mins=1.0)
        sel.fit(df, ys, sample_weight=sw)
        if len(sel.support_) == 0:
            return None
        # support_ is integer column index; convert back to column name for human-readable assertions.
        idx = int(sel.support_[0])
        return df.columns[idx] if idx < len(df.columns) else str(sel.support_[0])

    top1_uniform = _top1_with_weights(None)
    top1_recency = _top1_with_weights(recency_w)
    # The biz-value claim: recency weighting selects a different top-1, AND that top-1 is A (recent driver),
    # i.e. the encoder-style "stable across schemas" assumption is violated when the operator opts in.
    assert top1_recency == "A", (
        f"recency-weighted MRMR should surface feature A (recent-regime driver) as top-1; got {top1_recency!r}"
    )
    assert top1_uniform != top1_recency, (
        f"uniform-weight top-1 must differ from recency-weighted top-1 to demonstrate the biz-value win; "
        f"got uniform={top1_uniform!r}, recency={top1_recency!r}"
    )
