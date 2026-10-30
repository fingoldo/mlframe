"""The validated-MDLP honest null runs in the numba chain for a continuous target and returns the floor of the categorise-per-permutation path."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_stages._binning import _validated_target_classes


def test_target_classes_follow_the_production_rebucketing():
    """Up to 64 distinct values the target is untouched; above that it becomes its quantile class, the same rule as ``mdlp_bin_edges_oos_validated``."""
    rng = np.random.default_rng(0)
    few = rng.integers(0, 5, 500)
    assert np.array_equal(_validated_target_classes(few), few)
    cont = rng.normal(size=2000)
    got = _validated_target_classes(cont)
    edges = np.unique(np.quantile(cont, np.linspace(0.0, 1.0, 65)[1:-1]))
    assert np.array_equal(got, np.searchsorted(edges, cont, side="right")) and np.unique(got).size == 64
    assert _validated_target_classes(np.append(cont, np.nan)) is None


def test_numba_null_matches_the_categorise_per_permutation_floor():
    """On a small frame with a continuous target both paths give the same median null maximum up to the sampling error of 300 permutations
    (the held-out split of the cut search is drawn differently in the two chains, so the values agree in distribution, not draw by draw)."""
    from mlframe.feature_selection.filters import _honest_mdlp_null as H
    from mlframe.feature_selection.filters._honest_spec import build_honest_spec
    from mlframe.feature_selection.filters.discretization import categorize_dataset

    rng = np.random.default_rng(3)
    n = 700
    X = pd.DataFrame({f"c{j}": rng.normal(size=n) for j in range(4)})
    y = rng.normal(size=n)
    kw = dict(method="quantile", n_bins=10, dtype=np.int16, missing_strategy="median", nbins_strategy="mdlp_validated", nbins_strategy_kwargs={}, cache_dir=None)
    df = X.assign(y=y)
    data, cols, nbins = categorize_dataset(df=df, y_for_strategy=y, **kw)

    def slow(row_perm):
        """Re-bin every column on the permuted raw target."""
        d, c, nb = categorize_dataset(df=df, y_for_strategy=y[np.asarray(row_perm)], **kw)
        pos = {str(k): j for j, k in enumerate(c)}
        return np.stack([d[:, pos[str(k)]] for k in cols], axis=1).astype(np.int64), np.array([nb[pos[str(k)]] for k in cols], dtype=np.int64)

    y_idx = list(cols).index("y")
    pool = [i for i in range(len(cols)) if i != y_idx]
    spec_fn = type("F", (), {})()
    spec_fn.spec = build_honest_spec(X=df, cols=list(cols), y_raw=_validated_target_classes(y), max_depth=8, method="oos")
    fast = H.honest_fdr_floor(spec_fn, data, nbins, y_idx, pool, 300, 0.5, 0)
    ref = H.honest_fdr_floor(slow, data, nbins, y_idx, pool, 300, 0.5, 0)
    assert abs(fast - ref) < 0.2 * ref


def test_numba_null_serves_columns_with_missing_values_and_matches_the_categorise_floor():
    """Columns with missing values stay in the numba chain for the validated strategy (the spec used to be refused, which sent the whole null to the categorise-per-permutation path) and give
    the same median null maximum as that path, up to the sampling error of 300 permutations.

    The missing rows are binned by the separate-bin rule: the cut search reads the column with its missing values filled by the median, and the missing rows then take one more code."""
    from mlframe.feature_selection.filters import _honest_mdlp_null as H
    from mlframe.feature_selection.filters._honest_spec import build_honest_spec
    from mlframe.feature_selection.filters.discretization import categorize_dataset

    rng = np.random.default_rng(5)
    n = 700
    X = pd.DataFrame({f"c{j}": rng.normal(size=n) for j in range(4)})
    for c in ("c0", "c2"):
        X.loc[rng.random(n) < 0.12, c] = np.nan
    y = rng.normal(size=n)
    kw = dict(method="quantile", n_bins=10, dtype=np.int16, missing_strategy="separate_bin", nbins_strategy="mdlp_validated", nbins_strategy_kwargs={}, cache_dir=None)
    df = X.assign(y=y)
    data, cols, nbins = categorize_dataset(df=df, y_for_strategy=y, **kw)

    def slow(row_perm):
        """Re-bin every column on the permuted raw target."""
        d, c, nb = categorize_dataset(df=df, y_for_strategy=y[np.asarray(row_perm)], **kw)
        pos = {str(k): j for j, k in enumerate(c)}
        return np.stack([d[:, pos[str(k)]] for k in cols], axis=1).astype(np.int64), np.array([nb[pos[str(k)]] for k in cols], dtype=np.int64)

    y_idx = list(cols).index("y")
    pool = [i for i in range(len(cols)) if i != y_idx]
    spec_fn = type("F", (), {})()
    spec_fn.spec = build_honest_spec(X=df, cols=list(cols), y_raw=_validated_target_classes(y), max_depth=8, method="oos")
    assert spec_fn.spec is not None and spec_fn.spec["has_nan"], "a frame with missing values must still get the numba spec"
    fast = H.honest_fdr_floor(spec_fn, data, nbins, y_idx, pool, 300, 0.5, 0)
    ref = H.honest_fdr_floor(slow, data, nbins, y_idx, pool, 300, 0.5, 0)
    assert abs(fast - ref) < 0.2 * ref, (fast, ref)
