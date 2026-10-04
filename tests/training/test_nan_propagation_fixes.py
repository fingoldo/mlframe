"""Wave-21 sensors: NaN-propagation through percentile / argmax / quantile.

Five P0 sites where a NaN-bearing input column silently collapsed downstream
to a useless constant / wrong-winner pick / silent gate-rejection:

1. ``boruta_shap.py:661`` -- ``np.percentile(shadow_importance, p)`` returned
   NaN when ANY shadow importance was NaN, then ``X_importance > NaN`` was
   all-False, silently rejecting EVERY feature from the Boruta gate.

2. ``discretization.py:edges()`` and inline at ``discretize_array``
   ``np.percentile(arr, quantiles)`` on a NaN-bearing column made every
   bin edge NaN; ``np.digitize`` / ``np.searchsorted`` then bucketed
   every row to bin 0 -- the entire discretised feature collapsed to a
   constant. The FS pipeline calls this 6000+ times per fit (per the
   module docstring), so the blast radius is the entire screening
   stage.

3. ``get_binning_edges()`` (njit twin of #2) same shape inside a
   ``@njit`` body. Numba doesn't expose ``np.nanpercentile``; fix is
   an inlined ``arr[~np.isnan(arr)]`` filter that numba supports.

4. ``_rfecv.py`` two ``np.argmax(cv_mean_perf)`` winner-picker sites
   would pick a NaN slot when every fold of one candidate N was
   degenerate (``_helpers.py:337`` produces NaN only when all folds
   are NaN). Caller then returned a never-evaluated N as ``support_``.

5. ``fe_baselines.py:266`` ``np.argmax(mi_arr)`` for the "best baseline"
   pick: if the batch MI kernel emitted NaN for a degenerate feature,
   argmax picks the NaN's index -> downstream MI gate sees a bogus
   score and the feature engineering pipeline emits the wrong "best".

6. ``apriori_itemsets.py:44`` ``np.quantile(X_ref[:, j], ...)`` on
   NaN-bearing columns silently emits all-NaN edges + all-zero bin
   columns (entire feature collapsed to "always-bin-0").

All six fixes use ``np.nanpercentile`` / ``np.nanquantile`` / a finite
mask + ``np.argmax(arr[finite_mask])`` and raise / fall back loudly
when EVERY value is NaN (so the operator sees the degenerate case
explicitly).
"""

from __future__ import annotations


import numpy as np
import pytest

# ---- Site 1: boruta_shap shadow threshold -------------------------------


def _shadow_state(shadows, real):
    """Minimal BorutaShap-like state for ``calculate_hits``: one real column per entry of ``real``."""
    from types import SimpleNamespace

    cols = [f"c{i}" for i in range(len(real))]
    return SimpleNamespace(
        Shadow_feature_import_=np.asarray(shadows, dtype=float),
        X_feature_import_=np.asarray(real, dtype=float),
        percentile=100,
        ncols_=len(cols),
        columns_=cols,
        order_={c: i for i, c in enumerate(cols)},
    )


def test_boruta_shadow_threshold_uses_nanpercentile(caplog):
    """A NaN among the shadow importances must not poison the threshold: real columns above the finite shadow maximum still score a hit."""
    from mlframe.feature_selection.boruta_shap._shadow_stats import calculate_hits

    hits = calculate_hits(_shadow_state([0.1, 0.2, np.nan, 0.3], [0.5, 0.25, 0.9]))
    assert hits.tolist() == [1.0, 0.0, 1.0]

    with caplog.at_level("WARNING"):
        degenerate = calculate_hits(_shadow_state([np.nan, np.nan], [0.5, 0.9]))
    assert degenerate.tolist() == [0.0, 0.0]
    assert [r for r in caplog.records if "shadow_threshold is non-finite" in r.getMessage()], "the all-NaN case must be reported, not silent"


# ---- Site 2 + 3: discretization edges + njit twin -----------------------


def test_discretization_edges_nan_input_finite_output():
    """The ``edges()`` helper must produce finite bin_edges even when the
    input array contains NaN. Pre-fix the entire feature collapsed to bin 0."""
    from mlframe.feature_selection.filters.discretization import edges

    arr = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0])
    quantiles = np.linspace(0, 100, 5)
    result = edges(arr, quantiles)
    assert np.all(np.isfinite(result)), (
        f"edges() produced non-finite output on NaN-bearing input: {result}. "
        f"Wave 21 P0 regression: bin_edges collapse silently bucketed every "
        f"row to bin 0 downstream."
    )


def test_discretize_array_quantile_path_handles_nan():
    """The inlined quantile path inside ``discretize_array`` must not
    collapse a NaN-bearing column to bin 0 across the board."""
    from mlframe.feature_selection.filters.discretization import discretize_array

    arr = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0])
    out = discretize_array(arr, n_bins=4, method="quantile")
    # Post-fix: finite values get sensibly bucketed across the n_bins;
    # NaN row still bucketed (to bin 0 or wherever NaN searchsort returns)
    # but the FINITE rows must span multiple bins.
    finite_bins = out[~np.isnan(arr)]
    n_distinct = len(set(int(b) for b in finite_bins.tolist()))
    assert n_distinct >= 2, (
        f"discretize_array collapsed finite values to {n_distinct} bin(s); "
        f"pre-fix the all-NaN edges silently bucketed every row to bin 0. "
        f"Full output: {out.tolist()}"
    )


def test_get_binning_edges_njit_nan_filter():
    """The @njit twin ``get_binning_edges`` must filter NaN inline."""
    from mlframe.feature_selection.filters.discretization import get_binning_edges

    arr = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0])
    bin_edges = get_binning_edges(arr, n_bins=4, method="quantile")
    assert np.all(np.isfinite(bin_edges)), f"get_binning_edges (@njit twin) produced non-finite output: {bin_edges}. Wave 21 P0 regression."


def test_get_binning_edges_all_nan_column_returns_degenerate_not_raise():
    """mrmr_audit_2026-07-20 B-11 (P0): an all-NaN column has zero finite values, so ``np.percentile`` on the
    empty post-mask array used to raise ValueError. In isolation that propagated correctly, but inside
    ``_discretize_2d_array_njit``'s ``prange`` loop the exception was silently swallowed (a documented numba
    limitation), leaving the output column at whatever garbage ``np.empty_like`` happened to contain. The fix
    makes an all-NaN column deterministically map to bin 0 (correct: zero finite values carries zero
    information) in BOTH the isolated call and the batch prange path, with no exception and no garbage."""
    from mlframe.feature_selection.filters.discretization import get_binning_edges, _discretize_2d_array_njit

    all_nan = np.full(200, np.nan, dtype=np.float64)
    edges = get_binning_edges(all_nan, n_bins=5, method="quantile")
    assert np.all(np.isfinite(edges)), f"get_binning_edges on an all-NaN column must return finite degenerate edges, got {edges}"

    rng = np.random.default_rng(0)
    arr2d = np.column_stack([rng.standard_normal(2000), np.full(2000, np.nan, dtype=np.float64), rng.standard_normal(2000)])
    out = _discretize_2d_array_njit(arr2d, n_bins=5, method="quantile", min_ncats=50, min_values=None, max_values=None, dtype=np.int8)
    nan_col_codes = np.unique(out[:, 1])
    assert nan_col_codes.tolist() == [0], (
        f"the all-NaN middle column must deterministically bin to 0 for every row inside the batch prange "
        f"path, got distinct codes {nan_col_codes.tolist()} -- pre-fix this was garbage int8 values from an "
        f"uninitialized output buffer (the ValueError never propagated out of the prange loop)."
    )
    # Sibling real-valued columns must be unaffected by the degenerate neighbor.
    assert len(np.unique(out[:, 0])) > 1, "column 0 (real signal) must still be discretized normally"
    assert len(np.unique(out[:, 2])) > 1, "column 2 (real signal) must still be discretized normally"


# ---- Site 4: RFECV winner-picker (2 sites) ------------------------------


@pytest.mark.parametrize("rule", ["one_se_max", "one_se_min"])
def test_rfecv_winner_picker_skips_nan_candidates(rule):
    """Both RFECV winner pickers ignore an all-NaN-folds candidate. ``np.argmax`` returns the NaN slot
    (N=1 here), which made the NaN band pick that never-evaluated N; the finite best is N=2."""
    from types import SimpleNamespace

    from sklearn.linear_model import LogisticRegression

    from mlframe.feature_selection.wrappers.rfecv import RFECV
    from mlframe.feature_selection.wrappers.rfecv._diagnostics import n_features_one_se_

    nfeatures, means, stds = [0, 1, 2, 3], [0.5, np.nan, 0.8, 0.7], [0.0, 0.01, 0.01, 0.2]

    rfecv = RFECV(estimator=LogisticRegression(), n_features_selection_rule=rule)
    rfecv.feature_names_in_ = ["a", "b", "c"]
    rfecv.n_features_in_ = 3
    rfecv.selected_features_ = {1: [0], 2: [0, 1], 3: [0, 1, 2]}
    rfecv.feature_importances_ = {}
    rfecv.select_optimal_nfeatures_(checked_nfeatures=nfeatures, cv_mean_perf=means, cv_std_perf=stds, smooth_perf=0)
    assert rfecv.n_features_ == 2

    fitted = SimpleNamespace(cv_results_={"nfeatures": nfeatures, "cv_mean_perf": means, "cv_std_perf": stds}, n_features_=99)
    assert n_features_one_se_(fitted, rule.rsplit("_", 1)[1]) == 2


# ---- Site 5: fe_baselines best-baseline picker --------------------------


def test_fe_baselines_handles_all_nan_mi_arr(monkeypatch):
    """A NaN MI for some trivial features never wins the best-baseline pick; when every MI is NaN no baseline is returned."""
    from mlframe.feature_selection.filters.fe_baselines import best_trivial_pair, trivial_pair_features
    from mlframe.feature_selection.filters.hermite_fe import shared

    rng = np.random.default_rng(0)
    x_a, x_b = rng.uniform(1.0, 2.0, 100), rng.uniform(1.0, 2.0, 100)
    y = rng.integers(0, 2, 100)
    n_feats = len(trivial_pair_features(x_a, x_b))
    assert n_feats > 3

    def _mi_with_one_finite(X, *_a, **_kw):
        """NaN everywhere except the last column."""
        out = np.full(X.shape[1], np.nan)
        out[-1] = 0.4
        return out

    monkeypatch.setattr(shared, "plugin_mi_classif_batch_njit", _mi_with_one_finite)
    name, arr, mi = best_trivial_pair(x_a, x_b, y)
    assert name is not None and arr is not None
    assert mi == 0.4

    monkeypatch.setattr(shared, "plugin_mi_classif_batch_njit", lambda X, *_a, **_kw: np.full(X.shape[1], np.nan))
    name, arr, mi = best_trivial_pair(x_a, x_b, y)
    assert name is None and arr is None
    assert np.isnan(mi)


# ---- Site 6: apriori_itemsets discretiser -------------------------------


def test_apriori_itemsets_handles_nan_column():
    """NaN in either the train or the query matrix is rejected loudly (never silently binned), and a clean run yields finite top_k + 2 features."""
    pytest.importorskip("mlxtend")
    from mlframe.feature_engineering.transformer.apriori_itemsets import compute_apriori_itemsets_features

    rng = np.random.default_rng(0)
    n = 300
    X = rng.uniform(size=(n, 3)).astype(np.float32)
    y = (X[:, 0] > 0.5).astype(np.float32)
    kwargs = dict(seed=1, task="binary", top_k=4, min_support=0.1, max_len=2)

    out = compute_apriori_itemsets_features(X, y, X[:50], **kwargs)
    assert out.shape == (50, 6)
    assert np.isfinite(out.to_numpy()).all()

    X_nan = X.copy()
    X_nan[::7, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        compute_apriori_itemsets_features(X_nan, y, X[:50], **kwargs)
    with pytest.raises(ValueError, match="non-finite"):
        compute_apriori_itemsets_features(X, y, X_nan[:50], **kwargs)
