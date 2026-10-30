"""``PermutedTargetBinner`` returns the codes, columns and bin counts of ``categorize_dataset`` for ``optimal_joint``, permutation after permutation, and the honest-null hook uses it."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import _adaptive_nbins as A
from mlframe.feature_selection.filters.discretization import categorize_dataset
from mlframe.feature_selection.filters.discretization._null_binner import PermutedTargetBinner


def _frame(n=2500, seed=0):
    """Columns of different shape: Gaussian, tied, skewed, with NaN, low-cardinality."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "normal": rng.standard_normal(n),
            "ties": np.round(rng.standard_normal(n), 1),
            "skew": rng.lognormal(size=n),
            "withnan": np.where(rng.random(n) < 0.08, np.nan, rng.standard_normal(n)),
            "lowcard": rng.integers(0, 4, n).astype(np.float64),
            "f32": rng.standard_normal(n).astype(np.float32),
        }
    )


@pytest.mark.parametrize("missing", ["separate_bin", "propagate", "fillna_zero"])
def test_codes_equal_categorize_dataset_for_every_permutation(missing):
    """Same codes (as int64), same columns and same bin counts as the whole-frame categorisation, with a plan cache in the kwargs."""
    frame = _frame()
    rng = np.random.default_rng(1)
    y = (rng.random(len(frame)) < 0.4).astype(np.int64)
    kwargs = {"optimal_joint_plan_cache": A.OptimalJointPlanCache()}
    binner = PermutedTargetBinner(frame, missing, "optimal_joint", kwargs, np.int16)
    for s in range(6):
        yy = y if s == 0 else y[np.random.default_rng(s).permutation(y.size)]
        got = binner.codes(yy)
        assert got is not None
        want, cols, nb = categorize_dataset(
            df=frame, method="quantile", n_bins=10, dtype=np.int16, missing_strategy=missing, nbins_strategy="optimal_joint", nbins_strategy_kwargs={}, y_for_strategy=yy, cache_dir=None
        )
        assert got[1] == list(cols)
        assert np.array_equal(got[0], want.astype(np.int64)), s
        assert np.array_equal(got[2], nb), s


def test_fixed_columns_fill_their_place_in_the_requested_order():
    """With ``out_cols`` and ``fixed`` the matrix has the requested column order, the fixed codes unchanged, and equals the categorisation of the re-binned columns elsewhere."""
    frame = _frame()
    y = (np.random.default_rng(1).random(len(frame)) < 0.4).astype(np.int64)
    dep = frame[["normal", "skew", "withnan"]]
    fixed_codes = np.arange(len(frame)) % 5
    out_cols = ["skew", "fx", "normal", "withnan"]
    binner = PermutedTargetBinner(dep, "separate_bin", "optimal_joint", {}, np.int16, out_cols=out_cols, fixed={"fx": (fixed_codes, 5)})
    assert binner.ok
    got = binner.codes(y)
    assert got is not None and got[1] == out_cols
    want, cols, nb = categorize_dataset(
        df=dep, method="quantile", n_bins=10, dtype=np.int16, missing_strategy="separate_bin", nbins_strategy="optimal_joint", nbins_strategy_kwargs={}, y_for_strategy=y, cache_dir=None
    )
    pos = {c: j for j, c in enumerate(cols)}
    for i, c in enumerate(out_cols):
        if c == "fx":
            assert np.array_equal(got[0][:, i], fixed_codes) and got[2][i] == 5
        else:
            assert np.array_equal(got[0][:, i], want[:, pos[c]]) and got[2][i] == nb[pos[c]]
    assert not PermutedTargetBinner(dep, "separate_bin", "optimal_joint", {}, np.int16, out_cols=["normal", "missing"]).ok


def test_the_binner_declines_what_it_does_not_cover():
    """A frame with a categorical column, or an empty one, is not supported (the caller uses ``categorize_dataset``)."""
    frame = _frame(100)
    assert PermutedTargetBinner.supported(frame)
    assert not PermutedTargetBinner.supported(frame.assign(c=pd.Categorical(["a", "b"] * 50)))
    assert not PermutedTargetBinner.supported(frame.iloc[:0])


def test_the_honest_null_hook_returns_the_same_codes_with_and_without_the_binner(monkeypatch):
    """A fit with ``nbins_strategy='optimal_joint'``: the hook's codes for fixed permutations equal those with ``MLFRAME_OJ_NULL_BINNER=0``."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(7)
    n = 1500
    X = pd.DataFrame(rng.standard_normal((n, 6)), columns=[f"f{i}" for i in range(6)])
    X.loc[rng.random(n) < 0.1, "f2"] = np.nan
    y = ((X["f0"].fillna(0) + X["f1"] + 0.5 * rng.standard_normal(n)) > 0).astype(np.int64).to_numpy()

    def fit_and_codes(disable):
        """Fit under the switch; the hook's codes for three fixed permutations."""
        if disable:
            monkeypatch.setenv("MLFRAME_OJ_NULL_BINNER", "0")
        else:
            monkeypatch.delenv("MLFRAME_OJ_NULL_BINNER", raising=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = MRMR(verbose=0, random_seed=0, n_jobs=1, nbins_strategy="optimal_joint", fe_max_steps=0).fit(X, y)
        fn = model._honest_null_codes_fn
        return [fn(np.random.default_rng(s).permutation(n)) for s in range(3)]

    fast, plain = fit_and_codes(False), fit_and_codes(True)
    for (c1, nb1), (c0, nb0) in zip(fast, plain):
        assert np.array_equal(c1, c0) and np.array_equal(nb1, nb0)


def test_corrected_mi_best_equals_the_per_column_loop():
    """The parallel kernel gives the maximum of the per-column Miller-Madow corrected plug-in MI, reading columns past the re-binned block from the fixed codes."""
    from mlframe.feature_selection.filters._numba_utils import _mi_codes_njit, corrected_mi_best

    rng = np.random.default_rng(3)
    n = 3000
    codes = rng.integers(0, 6, (n, 4)).astype(np.int64)
    fixed = rng.integers(0, 5, (n, 7)).astype(np.int8)
    nb, fnb = np.array([6, 6, 6, 6]), np.array([0, 0, 0, 0, 5, 5, 5])
    yc = np.where(rng.random(n) < 0.3, codes[:, 1] % 2, rng.integers(0, 2, n)).astype(np.int64)
    cols = np.array([0, 1, 2, 3, 4, 6])
    best = 0.0
    for c in cols:
        x, kx = (codes[:, c], nb[c]) if c < 4 else (fixed[:, c].astype(np.int64), fnb[c])
        mi = _mi_codes_njit(np.ascontiguousarray(x), int(kx), yc, 2)
        best = max(best, mi - (np.unique(x).size - 1.0) * (np.unique(yc).size - 1.0) / (2.0 * n))
    for layout in (np.ascontiguousarray, np.asfortranarray):
        got = corrected_mi_best(layout(codes), nb, fixed, fnb, cols, yc, 2, float(np.unique(yc).size), n, np.arange(n, dtype=np.int64))
        assert got == pytest.approx(best, abs=1e-12)


def test_the_candidate_scoring_runs_once_per_permutation_in_one_kernel(monkeypatch):
    """After the first permutation built the plans, the winners of all columns come from the batched kernel and the per-column search is not run."""
    from mlframe.feature_selection.filters import _optimal_joint_edges as O

    frame = _frame()
    y = (np.random.default_rng(1).random(len(frame)) < 0.4).astype(np.int64)
    kwargs = {"optimal_joint_plan_cache": A.OptimalJointPlanCache()}
    binner = PermutedTargetBinner(frame[["normal", "skew", "withnan", "f32"]], "separate_bin", "optimal_joint", kwargs, np.int16)
    assert binner.codes(y) is not None  # builds the plans
    calls = []
    real = O._optimal_joint_best_m
    monkeypatch.setattr(O, "_optimal_joint_best_m", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    for s in range(1, 4):
        assert binner.codes(y[np.random.default_rng(s).permutation(y.size)]) is not None
    assert not calls and binner._packed is not None


def test_a_single_column_equals_its_column_of_the_matrix():
    """``column`` returns the codes and bin count the full matrix holds for that column, for re-binned and fixed columns, permutation after permutation."""
    frame = _frame()
    y = (np.random.default_rng(1).random(len(frame)) < 0.4).astype(np.int64)
    dep = frame[["normal", "skew", "withnan"]]
    fixed_codes = np.arange(len(frame)) % 5
    out_cols = ["skew", "fx", "normal", "withnan"]
    kwargs = {"optimal_joint_plan_cache": A.OptimalJointPlanCache()}
    binner = PermutedTargetBinner(dep, "separate_bin", "optimal_joint", kwargs, np.int16, out_cols=out_cols, fixed={"fx": (fixed_codes, 5)})
    for s in range(4):
        yy = y[np.random.default_rng(s).permutation(y.size)]
        full = binner.codes(yy)
        assert full is not None
        for i in range(len(out_cols)):
            one = binner.column(yy, i)
            assert one is not None and np.array_equal(one[0], full[0][:, i]) and one[1] == full[2][i], (s, i)


def test_corrected_mi_best_at_a_row_subset_equals_the_kernel_on_the_subset():
    """Reading the codes at ``rows`` gives the value of the kernel run on the row-subset copy of the codes."""
    from mlframe.feature_selection.filters._numba_utils import corrected_mi_best

    rng = np.random.default_rng(4)
    n, m = 2000, 700
    codes = rng.integers(0, 5, (n, 3)).astype(np.int16)
    fixed = rng.integers(0, 4, (n, 5)).astype(np.int8)
    nb, fnb = np.array([5, 5, 5]), np.array([0, 0, 0, 4, 4])
    rows = np.sort(rng.choice(n, m, replace=False)).astype(np.int64)
    yc = (rng.random(n) < 0.4).astype(np.int64)
    cols = np.array([0, 1, 2, 3, 4])
    got = corrected_mi_best(codes, nb, fixed, fnb, cols, yc[rows], 2, 2.0, m, rows)
    want = corrected_mi_best(np.ascontiguousarray(codes[rows]), nb, np.ascontiguousarray(fixed[rows]), fnb, cols, yc[rows], 2, 2.0, m, np.arange(m, dtype=np.int64))
    assert got == want


def test_target_dependent_columns_equal_the_full_distinct_count():
    """The head-sample shortcut decides ``nunique > cap`` exactly as the full count: continuous, low-cardinality, constant and late-diverse columns."""
    from mlframe.feature_selection.filters._honest_spec import _LOW_CARD_CAP
    from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_stages._binning import _target_dependent_columns

    rng = np.random.default_rng(0)
    n = 20000
    late = np.zeros(n)
    late[-(_LOW_CARD_CAP + 5) :] = np.arange(_LOW_CARD_CAP + 5)  # many distinct values, none in the head
    frame = pd.DataFrame(
        {
            "cont": rng.standard_normal(n),
            "few": rng.integers(0, _LOW_CARD_CAP, n).astype(np.float64),
            "edge": rng.integers(0, _LOW_CARD_CAP + 1, n).astype(np.float64),
            "const": np.ones(n),
            "late": late,
            "nan_cont": np.where(rng.random(n) < 0.3, np.nan, rng.standard_normal(n)),
        }
    )
    got = _target_dependent_columns(frame, list(frame.columns))
    want = {c for c in frame.columns if frame[c].nunique(dropna=True) > _LOW_CARD_CAP}
    assert got == want and {"cont", "late"} <= got and "const" not in got
