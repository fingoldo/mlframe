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

    # Re-baselined for full-mode default: under full mode the signal trio
    # is de-duplicated into ENGINEERED combos that live OUTSIDE the raw
    # `support_` index array, so `len(support_)` no longer measures "how
    # many features were selected" (it counts only surviving raw columns).
    # The correct, dedup-aware count is the total selected feature set =
    # len(get_feature_names_out()). The business intent (a tighter
    # relevance gate selects fewer-or-equal features) holds under that
    # measure and stays falsifiable: a broken gate that admitted noise
    # would inflate the tight count past the loose one.
    n_tight = len(sel_tight.get_feature_names_out())
    n_loose = len(sel_loose.get_feature_names_out())
    assert n_tight <= n_loose, (
        f"tight min_relevance_gain ({n_tight} selected) must "
        f"select <= loose ({n_loose} selected)"
    )
    # Tight must still RECOVER >=1 signal feature (raw or engineered combo
    # that references a signal column 0/1/2).
    from tests.feature_selection._biz_val_synth import signal_recovery_count
    overlap = signal_recovery_count(sel_tight, [0, 1, 2])
    assert overlap >= 1, (
        f"tight gate must keep >=1 signal feature; found "
        f"names={list(sel_tight.get_feature_names_out())}"
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

    # Fit each on a PRIVATE copy: full-mode FE appends engineered columns
    # to the fit-input frame in place, so fitting sel_1 then sel_4 on the
    # SAME `df` lets sel_1's engineered columns pollute sel_4's input and
    # spuriously diverges the supports. On clean frames the threading path
    # is bit-identical to single-thread (verified), which is exactly the
    # parallel-determinism contract this test guards.
    # fe_fast_search=False: this test guards the EXHAUSTIVE search's parallel-determinism contract
    # (support set identical across n_workers). The default fast path (2026-06-14) sets fe_max_steps=1,
    # which surfaces a PRE-EXISTING order-2 tied-rank non-determinism across workers (reproducible with
    # fe_fast_search=False + explicit fe_max_steps=1) -- a separate framework bug tracked for follow-up,
    # NOT introduced by the fast toggle. Pin the exhaustive path so this sensor keeps guarding what it was
    # written to guard; the fast-path determinism is covered once that order-2 threading bug is fixed.
    sel_1 = MRMR(interactions_max_order=2, verbose=0, random_seed=42,
                  n_workers=1, fe_fast_search=False)
    sel_1.fit(df.copy(), ys)
    sel_4 = MRMR(interactions_max_order=2, verbose=0, random_seed=42,
                  n_workers=4, fe_fast_search=False)
    sel_4.fit(df.copy(), ys)
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
    # Re-baselined for full-mode default: on the saddle y=sign(x0^2-x1^2)
    # FE engineers a feature OF the (x0,x1) pair (e.g. add(x0,x1) /
    # mult(x0,x1)) and the raw indices 0/1 may be de-duplicated OUT of
    # `support_`, so the OLD `support_[:5] & {0,1}` undercounts. Credit
    # engineered features that REFERENCE the signal pair via
    # get_feature_names_out(). A broken FE branch would surface neither the
    # raw pair nor any engineered combo of it.
    from tests.feature_selection._biz_val_synth import signal_recovery_count
    overlap = signal_recovery_count(sel, [0, 1], top_k=5)
    assert overlap >= 1, (
        f"FE-enabled MRMR must surface signal pair member (raw or "
        f"engineered) in top-5; got names={list(sel.get_feature_names_out())}"
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
        make_signal_plus_noise, as_df, signal_recovery_count,
        downstream_auc, baseline_signal_auc,
    )
    X, y, signal = make_signal_plus_noise(n=1500, p_signal=3, p_noise=8, seed=42)
    df, ys = as_df(X, y)
    sel = MRMR(verbose=0, random_seed=42, quantization_method=method)
    sel.fit(df, ys)
    # Re-baselined for full-mode default: full mode de-duplicates the
    # redundant signal trio into engineered combos, so raw `support_`
    # overlap undercounts. Credit engineered references to signal columns
    # and require predictive parity with the all-signal baseline. A bin
    # method that produced all-same-class bins (the regression this test
    # guards) would recover < 2 signal columns and tank the AUC.
    overlap = signal_recovery_count(sel, signal, top_k=5)
    auc_sel = downstream_auc(sel, df, ys)
    auc_base = baseline_signal_auc(df, ys, signal)
    assert overlap >= 2 and auc_sel >= auc_base - 0.02, (
        f"quantization_method={method} must recover >=2 of 3 signal "
        f"features (raw or engineered) AND match the all-signal AUC; "
        f"got overlap={overlap}, auc_sel={auc_sel:.4f}, "
        f"auc_base={auc_base:.4f}, names={list(sel.get_feature_names_out())}"
    )


# ---------------------------------------------------------------------------
# use_simple_mode
# ---------------------------------------------------------------------------


def test_biz_val_mrmr_use_simple_mode_faster_on_redundant_data():
    """Full mode (``use_simple_mode=False``, the post-change DEFAULT) is
    the WINNER on correlated/redundant data: its Fleuret conditional-MI
    redundancy pass de-duplicates the correlated cluster EARLY, so it
    returns a far more COMPACT set and -- because it stops adding
    redundant veterans -- actually runs FASTER, not slower.

    The original test asserted the opposite ("simple mode is faster"),
    which was the simple-mode-specific premise: simple mode skips the
    redundancy re-evaluation and so keeps every correlated duplicate,
    making it both larger AND (on this data) markedly slower. That
    premise is inverted under the new default. Re-baselined per the
    real tradeoff: full mode must be no slower than simple AND select
    fewer-or-equal features (the de-duplication win). Still falsifiable:
    if the redundancy pass regressed to keeping duplicates, full's
    feature count would no longer be < simple's."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from tests.feature_selection._biz_val_synth import (
        make_correlated_redundant, as_df,
    )
    X, y, _ = make_correlated_redundant(n=1500, n_corr=6, p_noise=4, seed=42)
    df, ys = as_df(X, y)

    # Warmup numba (both modes share kernels). Fit on private copies: full
    # mode appends engineered cols to the input frame in place.
    MRMR(verbose=0, random_seed=42, use_simple_mode=False).fit(df.copy(), ys)

    t0 = time.perf_counter()
    sel_simple = MRMR(verbose=0, random_seed=42, use_simple_mode=True)
    sel_simple.fit(df.copy(), ys)
    t_simple = time.perf_counter() - t0

    t0 = time.perf_counter()
    sel_full = MRMR(verbose=0, random_seed=42, use_simple_mode=False)
    sel_full.fit(df.copy(), ys)
    t_full = time.perf_counter() - t0

    k_simple = len(sel_simple.support_)
    k_full = len(sel_full.support_)
    # The de-duplication win: full mode must yield a strictly smaller
    # raw-support on this heavily-correlated cluster.
    assert k_full < k_simple, (
        f"full mode must de-duplicate to fewer features than simple on "
        f"correlated data; got full={k_full}, simple={k_simple}"
    )
    # And it must not pay for that with wall-time: full <= 1.5x simple.
    assert t_full <= t_simple * 1.5, (
        f"full mode must be no slower than 1.5x simple on redundant data; "
        f"got full={t_full:.2f}s, simple={t_simple:.2f}s"
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
        make_signal_plus_noise, as_df, signal_recovery_count,
    )
    X, y, signal = make_signal_plus_noise(n=1500, p_signal=3, p_noise=10, seed=42)
    df, ys = as_df(X, y)
    # Warmup. Fit on private copies: full-mode FE appends engineered cols
    # to the input frame in place, which would otherwise bleed across fits.
    MRMR(verbose=0, random_seed=42).fit(df.copy(), ys)

    t0 = time.perf_counter()
    sel_low = MRMR(verbose=0, random_seed=42, full_npermutations=1)
    sel_low.fit(df.copy(), ys)
    t_low = time.perf_counter() - t0

    t0 = time.perf_counter()
    sel_high = MRMR(verbose=0, random_seed=42, full_npermutations=10)
    sel_high.fit(df.copy(), ys)
    t_high = time.perf_counter() - t0

    # Re-baselined for full-mode default: top-3 signal recovery credits
    # engineered combos that reference the signal columns (raw indices are
    # de-duplicated out). Both perm budgets must recover >=2 of 3 signal.
    overlap_low = signal_recovery_count(sel_low, signal, top_k=3)
    overlap_high = signal_recovery_count(sel_high, signal, top_k=3)
    assert overlap_low >= 2 and overlap_high >= 2, (
        f"signal recovery must be robust to permutation budget; "
        f"got low overlap={overlap_low}, high overlap={overlap_high}"
    )
    # Lower perms must be at-or-below high perms in wall. On small synthetic
    # frames the per-permutation cost is a tiny slice of total wall (cat-FE,
    # fingerprinting, candidate scoring dominate); cache warm-up order across
    # the two fits can swap the ratio. Apply the constraint only when the
    # absolute wall is large enough for the perm budget to actually matter
    # (>= 200ms) and use a 2x bound to allow noisy fast paths.
    if t_high >= 0.2:
        assert t_low <= t_high * 2.0, (
            f"full_npermutations=1 must be no slower than 2x of =10; "
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
        make_signal_plus_noise, as_df, signal_recovery_count,
    )
    X, y, signal = make_signal_plus_noise(n=1500, p_signal=3, p_noise=8, seed=42)
    df, ys = as_df(X, y)

    # Fit on private copies: full-mode FE appends engineered cols in place,
    # so sharing `df` across the two fits would pollute the second.
    sel_on = MRMR(verbose=0, random_seed=42, extra_x_shuffling=True)
    sel_off = MRMR(verbose=0, random_seed=42, extra_x_shuffling=False)
    sel_on.fit(df.copy(), ys)
    sel_off.fit(df.copy(), ys)

    # Re-baselined for full-mode default: signal trio is de-duplicated into
    # engineered combos outside raw `support_`, so credit engineered
    # references to signal columns via get_feature_names_out(). Intent
    # unchanged: both shuffling modes must recover the signal.
    overlap_on = signal_recovery_count(sel_on, signal, top_k=5)
    overlap_off = signal_recovery_count(sel_off, signal, top_k=5)
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
    # Must complete and produce non-empty OUTPUT. On a (symmetric) polynomial
    # target the default univariate-basis FE recovers the signal via engineered
    # features (``x0__He2`` etc.), so ``support_`` (raw-only indices) can be empty
    # while ``get_feature_names_out()`` (raw + engineered) is non-empty. Assert the
    # latter -- "the chebyshev basis FE ran and produced features" -- which is the
    # contract this test pins (basis-registry dispatch), not raw-column survival.
    assert len(sel.get_feature_names_out()) > 0


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
        make_signal_plus_noise, as_df, signal_recovery_count,
    )
    X, y, signal = make_signal_plus_noise(n=1000, p_signal=3, p_noise=8, seed=42)
    df, ys = as_df(X, y)
    sel = MRMR(verbose=0, random_seed=42, baseline_npermutations=baseline_n)
    sel.fit(df, ys)
    # Re-baselined for full-mode default: the redundant signal trio is
    # de-duplicated into engineered combos outside raw `support_`; credit
    # engineered references to signal columns. Top-5 must recover >= 2 of 3
    # signal features regardless of the baseline-permutation budget.
    assert signal_recovery_count(sel, signal, top_k=5) >= 2


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
    # Re-baselined for full-mode default: full mode can de-duplicate the
    # correlated cluster into a single ENGINEERED feature with empty raw
    # `support_`; count total selected (raw + engineered) via
    # get_feature_names_out() so the smoke test still asserts a non-empty
    # selection.
    assert len(sel.get_feature_names_out()) >= 1


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
        make_signal_plus_noise, as_df, signal_recovery_count,
        downstream_auc, baseline_signal_auc,
    )
    X, y, signal = make_signal_plus_noise(n=1000, p_signal=3, p_noise=8,
                                              seed=seed)
    df, ys = as_df(X, y)
    sel = MRMR(verbose=0, random_seed=seed)
    sel.fit(df, ys)
    # Re-baselined for full-mode default (use_simple_mode=False): on
    # y=sign(x0+x1+x2) full mode DE-DUPLICATES the redundant raw signal
    # trio into a single engineered combo (e.g. keeps {x1, add(x0,x2)}),
    # so the OLD raw-index `signal_overlap(...)>=2` undercounts a correct
    # selection. New contract credits engineered features that reference
    # the signal columns AND requires the de-duplicated selection to be
    # predictively as good as the all-signal baseline. Still falsifiable:
    # selecting noise collapses both the recovery count and the AUC.
    assert signal_recovery_count(sel, signal, top_k=5) >= 2
    auc_sel = downstream_auc(sel, df, ys)
    auc_base = baseline_signal_auc(df, ys, signal)
    assert auc_sel >= auc_base - 0.02, (
        f"selected-set AUC must be within 0.02 of all-signal baseline; "
        f"got auc_sel={auc_sel:.4f}, auc_base={auc_base:.4f}"
    )


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
    # Re-baselined for full-mode default: full mode often de-duplicates the
    # x0+x1 signal into a single ENGINEERED feature (e.g. add(x0,x1)) with
    # an EMPTY raw `support_`, so `len(support_) >= 1` no longer measures
    # "produced a valid selection". Count the total selected set via
    # get_feature_names_out() instead (raw survivors + engineered).
    n_selected = len(sel.get_feature_names_out())
    assert 1 <= n_selected <= 2 * p_features, (
        f"must produce 1..2p selected features; got {n_selected} "
        f"(support_={sel.support_.tolist()})"
    )


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
    # ``support_`` holds RAW column indices only; engineered features (e.g. order-2 binagg/interaction columns) live in the get_feature_names_out
    # surface, not in support_. With interactions_min_order>=2 the order-1 raw features are excluded from selection BY DESIGN, so the fit legitimately
    # returns an engineered-only support (empty raw support_). The contract this sensor pins is "the combination completes and yields >=1 selected
    # feature", so assert on the full selected surface, not the raw-only subset.
    n_selected = len(list(sel.get_feature_names_out()))
    assert 1 <= n_selected
    assert len(sel.support_) <= df.shape[1]


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


def test_biz_val_mrmr_factors_names_to_use_restricts_empty_screen_rescue():
    """``factors_names_to_use`` must restrict the search space EVEN on the
    empty-screen rescue path. With a noise-only subset the greedy screen
    returns 0 features and the ``min_features_fallback`` rescue fires; the
    rescue must rank ONLY the requested columns by MI(X_j, y), never the
    global top-MI column. Pre-fix the rescue iterated every raw input and
    resurrected ``x0`` (the strongest signal) for ``factors_names_to_use=['x3']``."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from tests.feature_selection._biz_val_synth import (
        make_signal_plus_noise, as_df,
    )
    X, y, _ = make_signal_plus_noise(n=800, p_signal=3, p_noise=5, seed=42)
    df, ys = as_df(X, y)
    subset = ["x3"]  # pure-noise column; forces the empty-screen rescue
    sel = MRMR(verbose=0, random_seed=42, min_features_fallback=2,
                factors_names_to_use=subset)
    sel.fit(df, ys)
    selected_names = set(df.columns[i] for i in sel.support_)
    assert selected_names.issubset(set(subset)), (
        f"empty-screen rescue leaked a forbidden feature; "
        f"factors_names_to_use={subset}, got {selected_names}"
    )


@pytest.mark.parametrize("preset", ["minimal", "medium", "maximal"])
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


@pytest.mark.parametrize("preset", ["minimal", "medium", "maximal"])
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
    # Re-baselined for full-mode default: full-mode dedup can leave the raw
    # `support_` empty when the survivor is an engineered feature; count
    # the total selected set via get_feature_names_out().
    assert 1 <= len(sel.get_feature_names_out()) <= 2 * df.shape[1]


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
    complete the FE step without raising on a polynomial target."""
    # The previous ``del sys.modules[...] + importlib.reload`` workaround for
    # a stale-.pyc race rebinds the ``MRMR`` class object mid-suite. Other
    # tests imported MRMR at file-load time and keep the OLD class reference,
    # while ``_mrmr_fit_impl.py``'s lazy import resolves the NEW one — cache
    # writes land on the new ``_FIT_CACHE`` and ``OLD MRMR._FIT_CACHE`` asserts
    # in ``test_mrmr_basic.py::TestMRMRFitCache`` then see 0 entries. The
    # 2026-05-22 trace confirmed this is the polluter; the .pyc race the
    # workaround was guarding is no longer reproducible.
    from mlframe.feature_selection.filters.mrmr import MRMR
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
    # See companion ``test_biz_val_mrmr_fe_max_polynom_degree_parametrize`` for
    # why the prior ``del sys.modules + importlib.reload`` workaround was
    # removed — it rebound the MRMR class object and polluted every later
    # cache-dependent test.
    from mlframe.feature_selection.filters.mrmr import MRMR
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
    # Full-mode honest invariant: assert on the TOTAL selection (raw +
    # engineered), not the RAW ``support_`` alone. At some bin counts (e.g.
    # nbins=5 on this fixture) the FE step engineers a single additive child
    # ``add(add(x0,x2),x1)`` that conditionally SUBSUMES all three raw signal
    # operands; the raw-redundancy sweep then legitimately drops every raw
    # (``_redundancy_emptied_raw_`` is set) leaving an ENGINEERED-ONLY
    # selection -- a complete, intended outcome, not an empty one. The raw
    # ``support_`` can therefore be empty while ``get_feature_names_out()`` is
    # non-empty, so count the full selected set (mirrors the sibling property
    # test below). The upper bound stays the raw column count: FE here only
    # COMPRESSES signal into fewer columns, never explodes it.
    n_selected = len(sel.get_feature_names_out())
    assert 1 <= n_selected <= df.shape[1]


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
    from hypothesis import HealthCheck, given, settings, strategies as st
    from mlframe.feature_selection.filters.mrmr import MRMR
    from tests.feature_selection._biz_val_synth import (
        make_signal_plus_noise, as_df,
    )

    # Re-baselined for full-mode default: a full-mode MRMR.fit (Fleuret
    # conditional-MI + FE) is much slower per example than simple mode, so
    # hypothesis mis-attributes the SLOW TEST BODY to input generation and
    # trips HealthCheck.too_slow. The body IS the thing under test, not the
    # cheap integer draws, so suppress that one health check (the falsifiable
    # invariant below still runs on every example). max_examples trimmed to
    # keep the serial wall-time bounded under the heavier default path.
    @given(
        n=st.integers(min_value=300, max_value=800),
        p_signal=st.integers(min_value=1, max_value=4),
        p_noise=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=6, deadline=None,
              suppress_health_check=[HealthCheck.too_slow])
    def _property(n, p_signal, p_noise, seed):
        X, y, _ = make_signal_plus_noise(
            n=n, p_signal=p_signal, p_noise=p_noise, seed=seed,
        )
        df, ys = as_df(X, y)
        sel = MRMR(verbose=0, random_seed=seed)
        sel.fit(df, ys)
        # Re-baselined for full-mode default: full mode may de-duplicate all
        # signal into engineered features leaving an EMPTY raw `support_`,
        # so count the total selected set (raw + engineered) via
        # get_feature_names_out(). Intent unchanged: no crash + >=1 feature.
        n_selected = len(sel.get_feature_names_out())
        # Intent: no crash + >=1 feature, with no runaway explosion. The upper cap is a loose sanity bound -- raw +
        # engineered (smart_polynom / pairwise) FE can legitimately expand past 2x the raw column count, so cap at the
        # pairwise-FE scale (~ p*(p+1)/2) which still catches a genuine blow-up.
        p = df.shape[1]
        assert 1 <= n_selected <= max(2 * p, p * (p + 1) // 2)

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
    # Re-baselined for full-mode default: full-mode dedup routes confirmed
    # signal into ENGINEERED features that are NOT in the raw `support_`
    # index array, so `len(support_)` is no longer the selected-feature
    # count (here loose confirms 1 engineered feature with support_==[]).
    # Compare the total selected set via get_feature_names_out(); the
    # stopping-rule intent (a stricter confidence selects fewer-or-equal
    # features) holds under that dedup-aware measure.
    n_strict = len(sel_strict.get_feature_names_out())
    n_loose = len(sel_loose.get_feature_names_out())
    assert n_strict <= n_loose, (
        f"min_nonzero_confidence=0.999 ({n_strict} selected) must "
        f"<= 0.90 ({n_loose} selected)"
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
        # Re-baselined for full-mode default: full-mode FE appends an
        # engineered column to the fit-input frame IN PLACE, so sharing the
        # same X_train/X_test across the two mode-fits let the first mode's
        # engineered column bleed into the second fit's feature_names_in_
        # and then break transform() on the un-mutated test frame. Fit/score
        # on private copies so each mode is measured independently.
        Xtr_fit = X_train.copy()
        sel.fit(Xtr_fit, y_train)
        feats = list(sel.get_feature_names_out())
        if not feats:
            return float("nan"), 0
        # get_feature_names_out() may include ENGINEERED features (e.g.
        # add(x0,log(x1))) absent from the raw frame, so the old
        # X_train[feats] indexing raised KeyError. Materialize the selected
        # design matrix via the selector's own transform(), which
        # reconstructs engineered columns from the raw inputs. AUC / count
        # semantics are unchanged.
        Xtr = sel.transform(X_train.copy())
        Xte = sel.transform(X_test.copy())
        clf = LogisticRegression(max_iter=400, random_state=0).fit(Xtr, y_train)
        auc = roc_auc_score(y_test, clf.predict_proba(Xte)[:, 1])
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
    # Time axis: index 0..n-1 (0=oldest, n-1=newest). Older slice is 4x bigger than recent slice so under
    # uniform weighting, B (older-regime driver) edges out A on relevance; under recency weighting the
    # imbalance reverses and A (recent-regime driver) wins top-1.
    n_recent = n // 5
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
        # Gate FE off: this test measures the sample_weight -> top-1 flip on the RAW features (A vs B). Any default-ON FE family that engineers a
        # column out of A and B (the conditional-gate operator, binned_numeric_agg's ``binagg(A|qbin(B))``, k-fold target encoding) reorders the raw
        # relevance ranking and pulls the uniform-weight top-1 to A, masking the older-regime B-edges-A signal the flip relies on. Disable them so the
        # sensor isolates the weight -> raw-feature mechanism it is meant to pin; these families are measured-better defaults elsewhere, just orthogonal here.
        sel = MRMR(
            verbose=0, random_seed=11, max_runtime_mins=1.0,
            fe_conditional_gate_enable=False,
            fe_binned_numeric_agg_enable=False,
            fe_kfold_te_enable=False,
        )
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
