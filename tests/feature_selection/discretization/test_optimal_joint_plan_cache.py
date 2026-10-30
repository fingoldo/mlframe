"""The target-independent plan of ``edges_optimal_joint`` (folds, train-fold edges, binned validation values) is built once per column for a null that permutes the target.

The cache must change the cost and nothing else: edges equal those of the plain cross-validated search, ``categorize_dataset`` returns the same codes with and without it, and the plan of a column is
built once however many permutations are binned. ``_handle_missing`` fills a NaN column with its median exactly as before while taking medians only of the columns that hold a NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import _adaptive_nbins as A
from mlframe.feature_selection.filters.discretization import _handle_missing, categorize_dataset


def _reference_edges(x, y, candidates=(4, 8, 16, 32), n_splits=3, base="quantile", random_state=0, max_y_classes=64):
    """The cross-validated bin-count search written out directly: per fold and candidate, train-fold edges, validation bins, plug-in MI against the binned validation target."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y).ravel()
    mask = np.isfinite(x)
    if y.dtype.kind == "f":
        mask &= np.isfinite(y)
    x, y = x[mask], y[mask]
    n = x.size
    if n < n_splits * 4:
        return A.edges_freedman_diaconis(x, base=base)
    fold_idx = np.random.default_rng(random_state).permutation(n) % n_splits
    scores: dict[int, list] = {m: [] for m in candidates if m >= 2}
    for k in range(n_splits):
        train, val = fold_idx != k, fold_idx == k
        if val.sum() < 4:
            continue
        val_y_b, k_y = A._bin_y_for_mi(y[val], max_y_classes=max_y_classes)
        for m in candidates:
            if m < 2 or train.sum() < m:
                continue
            edges = A._edges_from_quantiles(x[train], m) if base == "quantile" else A._edges_from_uniform(x[train], m)
            if edges.size == 0:
                continue
            x_b = np.searchsorted(edges, x[val], side="right").astype(np.int64)
            k_x = int(x_b.max()) + 1
            scores[m].append(float(A._plug_in_mi_njit(x_b, val_y_b, k_x, k_y, False)))
    best, best_m = -np.inf, None
    for m in candidates:
        if m >= 2 and scores.get(m) and float(np.mean(scores[m])) > best:
            best, best_m = float(np.mean(scores[m])), m
    if best_m is None:
        return A.edges_freedman_diaconis(x, base=base)
    return A._edges_from_quantiles(x, best_m) if base == "quantile" else A._edges_from_uniform(x, best_m)


def _columns(n=3000, seed=0):
    """Columns of different shape: Gaussian, tied, skewed, one with NaN."""
    rng = np.random.default_rng(seed)
    return {
        "normal": rng.standard_normal(n),
        "ties": np.round(rng.standard_normal(n), 1),
        "skew": rng.lognormal(size=n),
        "withnan": np.where(rng.random(n) < 0.08, np.nan, rng.standard_normal(n)),
    }


def _targets(n=3000, seed=1):
    """A binary, a multiclass and a continuous target."""
    rng = np.random.default_rng(seed)
    return {"binary": (rng.random(n) < 0.4).astype(np.int64), "multi": rng.integers(0, 5, n), "float": rng.standard_normal(n)}


@pytest.mark.parametrize("base", ["quantile", "uniform"])
def test_edges_with_the_plan_cache_equal_the_plain_search(base):
    """For every column, target and permutation of the target the edges are those of the search written out directly, with and without the cache."""
    cache = A.OptimalJointPlanCache()
    rng = np.random.default_rng(2)
    for name, x in _columns().items():
        for tname, y in _targets().items():
            for rep in range(3):
                yy = y if rep == 0 else y[rng.permutation(y.size)]
                want = _reference_edges(x, yy, base=base)
                assert np.array_equal(A.edges_optimal_joint(x, yy, base=base), want), (name, tname, rep, "no cache")
                assert np.array_equal(A.edges_optimal_joint(x, yy, base=base, plan_cache=cache), want), (name, tname, rep, "cache")


def test_the_plan_of_a_column_is_built_once_for_all_permutations(monkeypatch):
    """Binning one column against many permutations of the target builds one plan."""
    from mlframe.feature_selection.filters import _optimal_joint_edges as O

    built = []
    real = O._optimal_joint_plan
    monkeypatch.setattr(O, "_optimal_joint_plan", lambda *a, **k: (built.append(1), real(*a, **k))[1])
    x = _columns()["normal"]
    y = _targets()["binary"]
    cache = A.OptimalJointPlanCache()
    rng = np.random.default_rng(3)
    for _ in range(8):
        A.edges_optimal_joint(x, y[rng.permutation(y.size)], plan_cache=cache)
    assert len(built) == 1


def test_a_plan_that_exceeds_the_budget_is_not_kept_and_the_result_is_unchanged():
    """With no room for a plan every call computes its own, and the edges are still those of the plain search."""
    x, y = _columns()["skew"], _targets()["multi"]
    cache = A.OptimalJointPlanCache(max_bytes=1)
    assert np.array_equal(A.edges_optimal_joint(x, y, plan_cache=cache), _reference_edges(x, y))
    assert not cache._plans


def test_categorize_dataset_returns_the_same_codes_with_and_without_the_plan_cache():
    """The whole binning of a frame (low-cardinality, tied, skewed and NaN columns) is unchanged by the cache, permutation after permutation."""
    rng = np.random.default_rng(4)
    n = 3000
    frame = pd.DataFrame({**_columns(n), "lowcard": rng.integers(0, 4, n).astype(np.float64), "n2": rng.standard_normal(n)})
    y = _targets(n)["binary"]
    kw = dict(method="quantile", n_bins=10, dtype=np.int16, missing_strategy="separate_bin", nbins_strategy="optimal_joint", cache_dir=None)
    cache = A.OptimalJointPlanCache()
    for s in range(4):
        yy = y[np.random.default_rng(s).permutation(n)]
        d0, c0, nb0 = categorize_dataset(df=frame, y_for_strategy=yy, nbins_strategy_kwargs={}, **kw)
        d1, c1, nb1 = categorize_dataset(df=frame, y_for_strategy=yy, nbins_strategy_kwargs={"optimal_joint_plan_cache": cache}, **kw)
        assert np.array_equal(d0, d1) and list(c0) == list(c1) and list(nb0) == list(nb1), s


def _handle_missing_reference(arr, strategy):
    """``_handle_missing`` as it was: the median taken over every column."""
    mask = np.isnan(arr)
    if not mask.any():
        return arr
    if strategy == "fillna_zero":
        return np.where(mask, 0.0, arr)
    med = np.where(np.isnan(np.nanmedian(arr, axis=0)), 0.0, np.nanmedian(arr, axis=0))
    if arr.flags.writeable:
        arr[mask] = np.broadcast_to(med, arr.shape)[mask]
        return arr
    return np.where(mask, med, arr)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("read_only", [False, True])
@pytest.mark.parametrize("strategy", ["separate_bin", "propagate", "fillna_zero"])
def test_handle_missing_fills_as_before_taking_medians_of_the_nan_columns_only(dtype, read_only, strategy):
    """Same filled values and dtype as the median over every column, for clean, partly NaN and all-NaN columns."""
    rng = np.random.default_rng(5)
    a = rng.standard_normal((200, 7)).astype(dtype)
    a[rng.random(200) < 0.2, 1] = np.nan
    a[:, 4] = np.nan
    a[::7, 6] = np.nan
    x1, x2 = a.copy(), a.copy()
    if read_only:
        x1.flags.writeable = False
        x2.flags.writeable = False
    got = _handle_missing(x2, strategy=strategy)
    want = _handle_missing_reference(x1, strategy)
    assert got.dtype == want.dtype and np.array_equal(got, want, equal_nan=True)


def test_handle_missing_takes_the_median_only_of_columns_with_a_nan(monkeypatch):
    """The number of columns whose median is computed equals the number of columns that hold a NaN."""
    import mlframe.feature_selection.filters.discretization as D

    seen = []
    real = np.nanmedian
    monkeypatch.setattr(D.np, "nanmedian", lambda a, axis=None, **k: (seen.append(a.shape), real(a, axis=axis, **k))[1])
    a = np.random.default_rng(6).standard_normal((100, 50))
    a[3, 10] = np.nan
    a[5, 20] = np.nan
    D._handle_missing(a, strategy="separate_bin")
    assert seen == [(100, 2)]


def test_the_honest_null_hook_uses_the_cache_and_returns_the_codes_of_the_uncached_search(monkeypatch):
    """A fit with ``nbins_strategy='optimal_joint'`` binds the plan cache into its honest null; the re-binned codes of a permutation equal those with ``MLFRAME_OJ_PLAN_CACHE=0``."""
    import warnings

    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(7)
    n = 1500
    X = pd.DataFrame(rng.standard_normal((n, 6)), columns=[f"f{i}" for i in range(6)])
    X.loc[rng.random(n) < 0.1, "f2"] = np.nan
    y = ((X["f0"].fillna(0) + X["f1"] + 0.5 * rng.standard_normal(n)) > 0).astype(np.int64).to_numpy()

    def fit_and_codes(disable):
        """Fit under the given switch and return the hook's codes for three fixed permutations, plus the plans the cache holds."""
        if disable:
            monkeypatch.setenv("MLFRAME_OJ_PLAN_CACHE", "0")
        else:
            monkeypatch.delenv("MLFRAME_OJ_PLAN_CACHE", raising=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = MRMR(verbose=0, random_seed=0, n_jobs=1, nbins_strategy="optimal_joint", fe_max_steps=0).fit(X, y)
        fn = model._honest_null_codes_fn
        perms = [np.random.default_rng(s).permutation(n) for s in range(3)]
        return [fn(p) for p in perms], list(model.get_feature_names_out())

    cached, names_cached = fit_and_codes(False)
    plain, names_plain = fit_and_codes(True)
    assert names_cached == names_plain
    for (c1, nb1), (c0, nb0) in zip(cached, plain):
        assert np.array_equal(c1, c0) and np.array_equal(nb1, nb0)
