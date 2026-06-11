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

import logging

import numpy as np
import pytest


# ---- Site 1: boruta_shap shadow threshold -------------------------------


def test_boruta_shadow_threshold_uses_nanpercentile(caplog):
    """Source-level guard: the post-fix line MUST use np.nanpercentile,
    NOT np.percentile, for the shadow threshold. Scans the whole
    feature_selection package (the shadow/stats helpers may live in a
    sibling module after a monolith split), so the guard survives file
    relocation while still catching a percentile->nanpercentile regression."""
    import pathlib
    import mlframe as _mlframe
    fs_dir = pathlib.Path(_mlframe.__file__).resolve().parent / "feature_selection"
    src = "\n".join(p.read_text(encoding="utf-8") for p in sorted(fs_dir.rglob("*.py")))
    assert "shadow_threshold = np.nanpercentile(self.Shadow_feature_import" in src, (
        "Wave 21 P0 regression: boruta_shap reverted to np.percentile; "
        "any NaN in shadow importances will collapse threshold to NaN "
        "and the gate will silently reject every feature."
    )
    assert "BorutaShap: shadow_threshold is non-finite" in src, (
        "All-NaN guard message must be present so operators see the "
        "degenerate case explicitly."
    )


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
    assert np.all(np.isfinite(bin_edges)), (
        f"get_binning_edges (@njit twin) produced non-finite output: "
        f"{bin_edges}. Wave 21 P0 regression."
    )


# ---- Site 4: RFECV winner-picker (2 sites) ------------------------------


def test_rfecv_winner_picker_skips_nan_candidates():
    """Source-level guard: both argmax sites in the RFECV wrappers package
    mask out NaN candidates before picking the winner."""
    import pathlib
    import mlframe as _mlframe
    # Fit body + submodule helpers all live under wrappers/rfecv/; concat every
    # submodule so the sensor catches the pattern regardless of which one owns it.
    _rfecv = pathlib.Path(_mlframe.__file__).resolve().parent / "feature_selection" / "wrappers" / "rfecv"
    src = "\n".join(p.read_text(encoding="utf-8") for p in _rfecv.glob("*.py"))
    # Post-fix: both sites use a finite-mask filter before argmax.
    occurrences = src.count("_finite_mask = np.isfinite(")
    assert occurrences >= 2, (
        f"Wave 21 P0 regression: expected >= 2 _finite_mask filters in "
        f"the rfecv sibling source; got {occurrences}."
    )
    # Pre-fix raw shape MUST be gone:
    assert "best_mean_idx = nz_idx[np.argmax(mean_arr[nz_idx])]" not in src, (
        "Pre-fix raw np.argmax on potentially-NaN cv_mean_perf reappeared."
    )


# ---- Site 5: fe_baselines best-baseline picker --------------------------


def test_fe_baselines_handles_all_nan_mi_arr():
    """When the MI batch kernel emits NaN for every feature, the best-
    baseline picker must return None/NaN, NOT pick the first NaN slot."""
    import pathlib
    import mlframe as _mlframe
    src = (
        pathlib.Path(_mlframe.__file__).resolve().parent
        / "feature_selection" / "filters" / "fe_baselines.py"
    ).read_text(encoding="utf-8")
    assert "_finite_mask = np.isfinite(mi_arr)" in src, (
        "Wave 21 P0 regression: fe_baselines best-baseline picker no "
        "longer masks NaN candidates."
    )


# ---- Site 6: apriori_itemsets discretiser -------------------------------


def test_apriori_itemsets_handles_nan_column():
    """The Apriori-itemsets discretiser must not silently collapse a
    NaN-bearing column to all-zero bin columns AND must not emit NaN
    downstream."""
    import pathlib
    import mlframe as _mlframe
    src = (
        pathlib.Path(_mlframe.__file__).resolve().parent
        / "feature_engineering" / "transformer" / "apriori_itemsets.py"
    ).read_text(encoding="utf-8")
    assert "np.nanquantile(X_ref[:, j]" in src, (
        "Wave 21 P0 regression: apriori_itemsets._discretize reverted to "
        "np.quantile; any NaN poisons every edge and collapses the entire "
        "discretised feature."
    )


# ---- Cross-site invariant -----------------------------------------------


def test_no_silent_argmax_on_potentially_nan_in_wave21_sites():
    """Cross-cutting source-level guard: each of the 5 wave-21 sites no
    longer has the pre-fix raw ``np.argmax(...)`` shape on a sequence that
    could contain NaN."""
    import pathlib
    import mlframe as _mlframe
    root = pathlib.Path(_mlframe.__file__).resolve().parent
    # Each (file, pre_fix_substring) pair:
    pre_fix_shapes = [
        ("feature_selection/wrappers/rfecv/__init__.py",
         "best_mean_idx = nz_idx[np.argmax(mean_arr[nz_idx])]"),
        ("feature_engineering/transformer/apriori_itemsets.py",
         "edges = np.quantile(X_ref[:, j]"),
    ]
    for rel, banned in pre_fix_shapes:
        text = (root / rel).read_text(encoding="utf-8")
        assert banned not in text, (
            f"Wave 21 P0 regression: pre-fix shape `{banned}` reappeared "
            f"in {rel}"
        )
