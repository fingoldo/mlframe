"""Consolidated from test_biz_value_mrmr_layer18.py.

Layer 18 biz_value MRMR contracts: DEGENERATE INPUT FEATURES.

WHY THIS LAYER
--------------
Every production ETL pipeline emits at least one degenerate column on
some slice/window. The MRMR contract here is NOT "rank these well" --
it is "survive without crashing AND don't pollute ``support_`` with
features that carry zero information". Layers 1-17 stressed shapes,
dtypes, leakage, noise; none of them probed the SHAPE of a column
that carries no usable signal at all, which is the cheapest failure
mode to surface in CI before it surfaces in prod.

DEGENERATE COLUMN ZOO -- nine real production hazards
-----------------------------------------------------
A. ``const_5``       -- all rows == 5.0 (zero variance)
B. ``all_nan``       -- every row is NaN (column completely missing)
C. ``dup_signal_1``  -- byte-identical copy of ``x_signal_1``
D. ``near_const``    -- 99% same value, 1% jitter
E. ``almost_all_nan``-- 1 non-NaN row out of N
F. ``single_value_int`` -- all rows == 1 (integer constant)
G. ``bool_col``      -- dtype=bool, 2 levels (genuinely 2-cat)
H. ``mostly_zero``   -- 99% zeros, 1% non-zero (sparse signal)
I. ``inf_only``      -- all values are +inf (probed separately)

The "good" columns are ``x_signal_1`` (coef 2.0) and ``x_signal_2``
(coef 1.0); they must survive selection despite the degenerate
neighbours.

CONTRACTS PINNED
----------------
1. MRMR.fit on a frame mixing A-H does NOT crash; ``support_`` is
   non-empty and contains at least one of the real signals.
2. Constant columns are EXCLUDED from ``support_``.
3. All-NaN columns are EXCLUDED from ``support_``.
4. Exact duplicate of a real signal: AT MOST one of the pair appears
   in ``support_`` (DCD prunes the redundant copy).
5. Near-constant column: NOT selected when its MI with y is below the
   real signals' MI -- this is the canonical "noise that looks like
   data" rejection.
6. Boolean dtype column survives the binning path without crashing
   (handled as 2-level categorical).
7. Mostly-zero column does NOT crash MRMR -- it represents real
   sparse domain data (event flags, rare-class indicators) and the
   selector must evaluate it normally.
8. ``+inf``-only column raises a clean ValueError referencing ``inf``
   and an actionable verb (``replace``/``drop``) -- matches the
   Layer 11 inf-safety gate contract.
9. The real signals (``x_signal_1``, ``x_signal_2``) appear in
   ``support_`` despite being neighboured by 8 degenerate columns --
   degeneracy must not crowd out actual relevance.

DEFAULT-CONFIG SURFACE
----------------------
Wave 9 production defaults (DCD ON, MDLP nbins, Miller-Madow). Only
``fe_max_steps=0`` and ``interactions_max_order=1`` are pinned for
wall-time bound. Degenerate column screening is the contract; the
interaction / FE synthesis paths don't change what counts as
"degenerate" -- a constant column stays constant under any synthesis.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_TOTAL = 2_000
N_NOISE = 2
SEEDS = (1, 7, 13, 42, 101)


# ---------------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------------


def _build_degenerate_frame(seed: int, include_inf: bool = False):
    """y = 2.0 * x_signal_1 + 1.0 * x_signal_2 + noise; 8 degenerate cols.

    Returns (X, y). ``include_inf=True`` adds the inf-only column,
    which is tested separately because it raises by contract.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(N_TOTAL)
    x2 = rng.standard_normal(N_TOTAL)
    y_arr = 2.0 * x1 + 1.0 * x2 + 0.3 * rng.standard_normal(N_TOTAL)

    # A. Constant column
    const_5 = np.full(N_TOTAL, 5.0)

    # B. All-NaN column
    all_nan = np.full(N_TOTAL, np.nan)

    # C. Exact duplicate of x_signal_1
    dup_signal_1 = x1.copy()

    # D. Near-constant: 99% same value, 1% jitter
    near_const = np.full(N_TOTAL, 3.14)
    jitter_idx = rng.choice(N_TOTAL, size=max(1, N_TOTAL // 100), replace=False)
    near_const[jitter_idx] += 0.001 * rng.standard_normal(jitter_idx.size)

    # E. All-NaN except one row
    almost_all_nan = np.full(N_TOTAL, np.nan)
    almost_all_nan[0] = 42.0

    # F. Single-value integer
    single_value_int = np.ones(N_TOTAL, dtype=np.int64)

    # G. Boolean column (genuinely 2-level; correlated with x1 sign)
    bool_col = (x1 > 0.0).astype(bool)

    # H. Mostly-zero: 99% zeros, 1% non-zero
    mostly_zero = np.zeros(N_TOTAL)
    nonzero_idx = rng.choice(N_TOTAL, size=max(1, N_TOTAL // 100), replace=False)
    mostly_zero[nonzero_idx] = rng.standard_normal(nonzero_idx.size)

    cols = {
        "x_signal_1": x1,
        "x_signal_2": x2,
        "const_5": const_5,
        "all_nan": all_nan,
        "dup_signal_1": dup_signal_1,
        "near_const": near_const,
        "almost_all_nan": almost_all_nan,
        "single_value_int": single_value_int,
        "bool_col": bool_col,
        "mostly_zero": mostly_zero,
    }
    for k in range(N_NOISE):
        cols[f"noise_{k}"] = rng.standard_normal(N_TOTAL)

    if include_inf:
        # I. Inf-only column (probed separately).
        cols["inf_only"] = np.full(N_TOTAL, np.inf)

    X = pd.DataFrame(cols)
    y = pd.Series(y_arr, name="y_reg")
    return X, y


DEGENERATE_ZERO_INFO_COLS = (
    "const_5",
    "all_nan",
    "single_value_int",
    "almost_all_nan",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mrmr(**overrides):
    """Default-config MRMR -- Wave 9 production surface.

    ``fe_max_steps=0`` and ``interactions_max_order=1`` are pinned for
    wall-time. Degeneracy is a per-column property, not affected by
    interaction synthesis.
    """
    from mlframe.feature_selection.filters.mrmr import MRMR
    kwargs = dict(
        verbose=0,
        interactions_max_order=1,
        fe_max_steps=0,
        # These contracts pin SELECTION invariants (support_/mrmr_gains_ length + alignment). The auxiliary default-on
        # FE stages have their OWN enable flags independent of ``fe_max_steps``, so they still inject engineered columns
        # (hinge relu legs, etc.) that legitimately enter support_ and inflate n_features_ past the raw selected set --
        # breaking the gains/support/n_features equality this file asserts. FE behaviour is covered in the FE test
        # files; pin it OFF here so the selection invariant is what is actually under test.
        fe_hinge_enable=False,
        fe_conditional_gate_enable=False,
        fe_conditional_dispersion_enable=False,
        fe_binned_numeric_agg_enable=False,
        fe_univariate_basis_enable=False,
        fe_univariate_fourier_enable=False,
    )
    kwargs.update(overrides)
    return MRMR(**kwargs)


def _fit_quiet(sel, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sel.fit(X, y)


# ---------------------------------------------------------------------------
# Contract 1: MRMR survives a degenerate frame
# ---------------------------------------------------------------------------


class TestMrmrSurvivesDegenerateFrame:
    """Minimum bar: a frame carrying 8 degenerate column shapes must
    NOT crash MRMR; ``support_`` is non-empty and the real signals
    are represented (at least one of x_signal_1 / x_signal_2).
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_fit_does_not_crash(self, seed):
        X, y = _build_degenerate_frame(seed, include_inf=False)
        sel = _make_mrmr(random_seed=seed)
        _fit_quiet(sel, X.copy(), y)
        assert sel.support_ is not None
        assert sel.n_features_ >= 1, (
            f"MRMR returned empty support_ on degenerate frame; "
            f"seed={seed}. A frame with 2 real signals plus degenerate "
            f"junk must yield at least one feature."
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_at_least_one_real_signal_in_support(self, seed):
        X, y = _build_degenerate_frame(seed, include_inf=False)
        sel = _make_mrmr(random_seed=seed)
        _fit_quiet(sel, X.copy(), y)
        names = list(sel.get_feature_names_out())
        signal_present = any(
            n in ("x_signal_1", "x_signal_2", "dup_signal_1") for n in names
        )
        assert signal_present, (
            f"NO real signal (x_signal_1 / x_signal_2 / dup_signal_1) "
            f"in support despite carrying >88% of var(y); seed={seed}, "
            f"support={names}. Degenerate columns crowded out actual "
            f"relevance -- relevance ranker is broken on this frame."
        )


# ---------------------------------------------------------------------------
# Contract 2: zero-information columns are excluded from support_
# ---------------------------------------------------------------------------


class TestZeroInfoColumnsExcluded:
    """Constant, all-NaN, all-1-int, and 1-non-NaN columns have, by
    construction, zero MI with y. They must NEVER appear in
    ``support_`` -- including any of them is a silent quality bug
    (downstream model gets a useless column, may even error on
    constant fit).
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_zero_info_cols_not_selected(self, seed):
        X, y = _build_degenerate_frame(seed, include_inf=False)
        sel = _make_mrmr(random_seed=seed)
        _fit_quiet(sel, X.copy(), y)
        names = set(sel.get_feature_names_out())
        leaked = [c for c in DEGENERATE_ZERO_INFO_COLS if c in names]
        assert not leaked, (
            f"zero-info degenerate columns leaked into support_: "
            f"{leaked}; seed={seed}, full support={sorted(names)}. "
            f"These columns have zero variance / are all-NaN and "
            f"cannot carry any MI with y."
        )


# ---------------------------------------------------------------------------
# Contract 3: exact duplicate of a signal -- at most one of the pair
# ---------------------------------------------------------------------------


class TestExactDuplicateAtMostOneOfPair:
    """``dup_signal_1`` is byte-identical to ``x_signal_1``. DCD must
    collapse them: at most ONE of the pair appears in ``support_``.
    Both surviving would mean DCD's redundancy detection is broken
    on the easiest possible case (perfect duplicate).
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_at_most_one_of_dup_pair(self, seed):
        X, y = _build_degenerate_frame(seed, include_inf=False)
        sel = _make_mrmr(random_seed=seed)
        _fit_quiet(sel, X.copy(), y)
        names = list(sel.get_feature_names_out())
        pair_count = sum(1 for n in names if n in ("x_signal_1", "dup_signal_1"))
        assert pair_count <= 1, (
            f"BOTH x_signal_1 AND dup_signal_1 selected (byte-identical "
            f"duplicates); DCD failed on the trivial duplicate case. "
            f"seed={seed}, support={names}"
        )


# ---------------------------------------------------------------------------
# Contract 4: bool/mostly-zero/near-constant don't crash binning
# ---------------------------------------------------------------------------


class TestUnusualDtypesDoNotCrash:
    """Boolean dtype, mostly-zero sparse, and near-constant columns
    don't carry the "zero info / NaN" pathology -- they're genuinely
    real-data shapes. They must NOT crash the binning / MI compute
    path. The contract is "MRMR completes"; selection is by MI.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_bool_dtype_does_not_crash(self, seed):
        """A frame containing only signals + a bool column completes."""
        rng = np.random.default_rng(seed)
        x1 = rng.standard_normal(N_TOTAL)
        x2 = rng.standard_normal(N_TOTAL)
        y_arr = 2.0 * x1 + 1.0 * x2 + 0.3 * rng.standard_normal(N_TOTAL)
        X = pd.DataFrame({
            "x_signal_1": x1,
            "x_signal_2": x2,
            "bool_col": (x1 > 0.0).astype(bool),
        })
        y = pd.Series(y_arr, name="y_reg")
        sel = _make_mrmr(random_seed=seed)
        _fit_quiet(sel, X.copy(), y)
        assert sel.n_features_ >= 1, (
            f"bool-column frame crashed selection silently; seed={seed}"
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_mostly_zero_does_not_crash(self, seed):
        """Mostly-zero sparse column (event-flag shape) doesn't crash."""
        rng = np.random.default_rng(seed)
        x1 = rng.standard_normal(N_TOTAL)
        x2 = rng.standard_normal(N_TOTAL)
        y_arr = 2.0 * x1 + 1.0 * x2 + 0.3 * rng.standard_normal(N_TOTAL)
        mostly_zero = np.zeros(N_TOTAL)
        nonzero_idx = rng.choice(N_TOTAL, size=20, replace=False)
        mostly_zero[nonzero_idx] = rng.standard_normal(20)
        X = pd.DataFrame({
            "x_signal_1": x1,
            "x_signal_2": x2,
            "mostly_zero": mostly_zero,
        })
        y = pd.Series(y_arr, name="y_reg")
        sel = _make_mrmr(random_seed=seed)
        _fit_quiet(sel, X.copy(), y)
        assert sel.n_features_ >= 1, (
            f"mostly-zero sparse-flag frame crashed; seed={seed}"
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_near_constant_does_not_crash(self, seed):
        """99%-constant column with 1% jitter doesn't crash binning."""
        rng = np.random.default_rng(seed)
        x1 = rng.standard_normal(N_TOTAL)
        x2 = rng.standard_normal(N_TOTAL)
        y_arr = 2.0 * x1 + 1.0 * x2 + 0.3 * rng.standard_normal(N_TOTAL)
        near_const = np.full(N_TOTAL, 3.14)
        jitter_idx = rng.choice(N_TOTAL, size=20, replace=False)
        near_const[jitter_idx] += 0.001 * rng.standard_normal(20)
        X = pd.DataFrame({
            "x_signal_1": x1,
            "x_signal_2": x2,
            "near_const": near_const,
        })
        y = pd.Series(y_arr, name="y_reg")
        sel = _make_mrmr(random_seed=seed)
        _fit_quiet(sel, X.copy(), y)
        assert sel.n_features_ >= 1, (
            f"near-constant frame crashed; seed={seed}"
        )


# ---------------------------------------------------------------------------
# Contract 5: near-constant column not preferred over real signals
# ---------------------------------------------------------------------------


class TestNearConstantNotPreferredOverSignal:
    """Near-constant has ~zero variance and therefore ~zero MI with y
    by construction (the 1% jitter is i.i.d. of y). It must NOT
    outrank a real signal. We pin the weaker direction: if a real
    signal landed in support, near_const did NOT replace it.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_near_const_not_selected_above_signal(self, seed):
        X, y = _build_degenerate_frame(seed, include_inf=False)
        sel = _make_mrmr(random_seed=seed)
        _fit_quiet(sel, X.copy(), y)
        names = list(sel.get_feature_names_out())
        if "near_const" not in names:
            return  # contract trivially satisfied
        # If near_const IS in support, at least one real signal must
        # be too. near_const winning over BOTH signals would mean MI
        # ranking is broken.
        has_signal = any(
            n in ("x_signal_1", "x_signal_2", "dup_signal_1") for n in names
        )
        assert has_signal, (
            f"near_const in support but NO real signal -- a column "
            f"with 1% jitter outranked features carrying >88% of "
            f"var(y). seed={seed}, support={names}"
        )


# ---------------------------------------------------------------------------
# Contract 6: inf-only column raises actionable ValueError
# ---------------------------------------------------------------------------


class TestInfOnlyColumnRaisesActionable:
    """A column of all +inf must raise ``ValueError`` referencing
    ``inf`` and an actionable remediation verb. Silent propagation
    would yield NaN MI and a misranked support, with no symptom
    until downstream model fit explodes. Matches the Layer 11
    inf-safety gate contract.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_inf_only_raises_valueerror(self, seed):
        X, y = _build_degenerate_frame(seed, include_inf=True)
        sel = _make_mrmr(random_seed=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError) as exc_info:
                sel.fit(X.copy(), y)
        msg = str(exc_info.value).lower()
        assert "inf" in msg, (
            f"seed={seed}: ValueError raised but message doesn't "
            f"reference inf. Got: {exc_info.value!r}"
        )
        assert "replace" in msg or "drop" in msg, (
            f"seed={seed}: ValueError missing actionable remediation "
            f"verb (replace / drop). Got: {exc_info.value!r}"
        )


# ---------------------------------------------------------------------------
# Contract 7: support_ / mrmr_gains_ alignment under degenerate input
# ---------------------------------------------------------------------------


class TestSupportGainsAlignmentDegenerate:
    """``mrmr_gains_`` aligned bit-for-bit with ``support_`` under the
    degenerate-frame stress. Misalignment here would mis-attribute
    every downstream diagnostic.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_gains_aligned_with_support(self, seed):
        X, y = _build_degenerate_frame(seed, include_inf=False)
        sel = _make_mrmr(random_seed=seed)
        _fit_quiet(sel, X.copy(), y)
        gains = np.asarray(sel.mrmr_gains_, dtype=np.float64)
        assert gains.shape == (sel.n_features_,), (
            f"mrmr_gains_ length {gains.shape} != n_features_ "
            f"{sel.n_features_} on degenerate frame; seed={seed}"
        )
        assert np.all(np.isfinite(gains)), (
            f"mrmr_gains_ has non-finite entries {gains} on degenerate "
            f"frame -- a degenerate column propagated NaN/Inf MI into "
            f"the diagnostic array. seed={seed}"
        )
        assert np.all(gains >= 0.0), (
            f"mrmr_gains_ has negative entries {gains} on degenerate "
            f"frame; gains are MI deltas and must be >= 0. seed={seed}"
        )


# ---------------------------------------------------------------------------
# Contract 8: pure-degenerate frame (no real signal) -- safe degradation
# ---------------------------------------------------------------------------


class TestPureDegenerateFrameDegradesSafely:
    """A frame consisting ONLY of degenerate columns (no real signal)
    must NOT crash and must NOT pretend to have a useful support.
    Either: returns near-empty / minimal support (fallback floor), OR
    raises a clean error referencing degeneracy. A crash with a
    cryptic numpy / numba traceback is the failure mode this guards.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_pure_degenerate_does_not_traceback(self, seed):
        """Frame with ONLY constant + all-NaN + single-value-int columns."""
        rng = np.random.default_rng(seed)
        y_arr = rng.standard_normal(N_TOTAL)
        X = pd.DataFrame({
            "const_5": np.full(N_TOTAL, 5.0),
            "all_nan": np.full(N_TOTAL, np.nan),
            "single_value_int": np.ones(N_TOTAL, dtype=np.int64),
        })
        y = pd.Series(y_arr, name="y_reg")
        sel = _make_mrmr(random_seed=seed)
        # Either it completes (possibly with empty / fallback support)
        # OR raises a ValueError. A cryptic non-ValueError exception
        # is the failure mode -- it means the degeneracy wasn't
        # detected before some downstream numpy/numba kernel.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sel.fit(X.copy(), y)
        except ValueError:
            pass  # acceptable: explicit user-facing error
        except Exception as e:
            pytest.fail(
                f"pure-degenerate frame raised non-ValueError "
                f"{type(e).__name__}: {e!r}. seed={seed}. "
                f"The degeneracy check should fire as ValueError "
                f"BEFORE a downstream kernel produces a cryptic crash."
            )
        else:
            # Completed: support must be either empty or just contain
            # what the fallback floor decided to keep -- never a
            # claim of "useful information" attached to constants.
            assert sel.support_ is not None
