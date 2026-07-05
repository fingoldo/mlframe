"""MRMR fit-time artifact retention + export for cross-selector reuse.

Carved out of ``_mrmr_fit_impl.py`` per the >1k-LOC sibling-module rule. The
in-fit screen builds three reusable intermediates that subsequent selectors
typically recompute from scratch:

* per-column bin assignments (the integer-binned matrix produced by
  ``categorize_dataset``);
* per-column bin counts;
* per-feature direct MI(X_j, y), populated as the screen evaluates candidates
  and surfaced as Symmetric Uncertainty via the marginal entropies.

A common pipeline pattern is ``MRMR(n_features_to_keep=K).fit(X, y)`` ->
``ShapProxiedFS(precomputed=...).fit(X_narrowed, y)``: the second selector
benefits from MRMR's relevance ranking (skip / replace its own univariate
F-statistic pre-screen) and could in principle reuse the bins for any MI-based
filter that doesn't natively re-quantise. The retention is opt-in via
``MRMR(retain_artifacts=True)`` so the legacy memory-footprint path is
byte-identical when the flag is False.

The export schema is documented in ``_ARTIFACT_SCHEMA``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _assert_nonneg_codes(codes: np.ndarray, what: str) -> None:
    """Fail loudly at the source if a binned-code column carries a negative value before ``np.bincount``.

    The upstream NaN-shift in ``categorize_dataset`` guarantees non-negative codes; a negative one means that shift regressed.
    Without this guard ``np.bincount`` raises an opaque ``ValueError: 'list' argument must have no negative elements`` /
    ``negative dimensions are not allowed`` that gives no hint of the real cause.
    """
    if codes.size and codes.min() < 0:
        raise ValueError(
            f"compute_mrmr_artifacts: {what} contains negative bin code(s) (min={int(codes.min())}); "
            "the NaN +1 shift in categorize_dataset must keep all codes >= 0 before bincount. This indicates an upstream binning regression."
        )


# Schema: the dict returned by ``MRMR.export_artifacts`` and consumed by
# ``ShapProxiedFS(precomputed=...)``. Keys are stable string identifiers;
# consumers MUST tolerate missing optional keys (forward compat).
_ARTIFACT_SCHEMA = {
    # REQUIRED keys
    "feature_names": "list[str], length=n_features_in_, ordered to match X_train.columns at MRMR.fit() time",
    "su_to_target": "np.ndarray, shape=(n_features_in_,), dtype=float64. Symmetric Uncertainty SU(X_j, y) in [0, 1]. NaN where SU could not be computed (e.g. constant column).",
    "mi_to_target": "np.ndarray, shape=(n_features_in_,), dtype=float64. Direct MI(X_j, y) in nats. NaN where MI was not cached during the screen (e.g. cardinality-bias-rejected features).",
    "mrmr_kept_indices": "list[int], positional indices into feature_names of the MRMR-selected subset (== self.support_.tolist() for the raw columns; engineered features are not included).",
    # OPTIONAL keys
    "bins": "dict[str, np.ndarray] | None, mapping feature_name -> per-row binned values (dtype quantization_dtype). Present when retain_bins=True (default), absent or None otherwise.",
    "nbins_per_feature": "dict[str, int] | None, mapping feature_name -> bin count. Present together with 'bins'.",
    "n_samples_at_fit": "int, the row count at which the artifacts were computed (consumers should warn / discard on shape mismatch).",
    "schema_version": "int, currently 1. Bumped if a future change breaks back-compat for consumers.",
}

# Current schema revision. Consumers in different mlframe versions may inspect
# this to fall back gracefully if the producer is older.
ARTIFACT_SCHEMA_VERSION = 1


def compute_mrmr_artifacts(
    *,
    data: np.ndarray,
    cols: list[str],
    nbins: np.ndarray,
    target_indices: np.ndarray,
    cached_MIs: dict,
    feature_names_in: list[str],
    support_original: np.ndarray,
    retain_bins: bool,
    dtype=np.int32,
) -> dict[str, Any]:
    """Build the artifact dict from in-fit state, called inside MRMR.fit() when
    ``retain_artifacts=True``.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_cols_in_data_space)
        The integer-binned matrix produced by ``categorize_dataset``. Includes
        the target column at ``target_indices[0]`` plus all candidate features
        (raw + engineered). NOT yet remapped to original-frame indices.
    cols : list[str]
        Column names matching ``data`` axis-1 (cols-space). Engineered features
        carry synthetic names; the consumer can filter these via
        ``set(cols) & set(feature_names_in)``.
    nbins : np.ndarray, shape (n_cols_in_data_space,), dtype int
        Per-column bin count.
    target_indices : np.ndarray, shape (1 or k,), dtype int
        Index of the target column inside ``data``.
    cached_MIs : dict[tuple[int, ...], float]
        Direct-MI cache populated by the screen; key=(col_idx_in_cols_space,),
        value=MI(X_col, y) in nats. Pairs / higher-order tuples are ignored.
    feature_names_in : list[str]
        Original-frame column names; the artifact axes mirror this order.
    support_original : np.ndarray, shape (n_selected,), dtype int
        ``self.support_`` -- positional indices into ``feature_names_in`` of the
        MRMR-kept raw features.
    retain_bins : bool
        When True, copy the binned columns into the export dict (memory cost:
        roughly ``n_samples * n_features_in * dtype-bytes``). When False, only
        the SU / MI vectors are returned.
    dtype : numpy dtype
        Match the screen's working dtype so cached MI numbers are comparable
        across fits.

    Returns
    -------
    dict
        Conforms to ``_ARTIFACT_SCHEMA`` above.
    """
    n_features_in = len(feature_names_in)
    n_samples = int(data.shape[0])

    # Map ORIGINAL-frame column name -> position inside ``cols`` (data-space).
    # categorize_dataset reorders columns when categoricals exist, so this
    # cannot be assumed to be identity.
    name_to_data_col = {c: i for i, c in enumerate(cols)}

    # Target marginal entropy: H(y). Computed once and reused below.
    y_idx = int(target_indices[0])
    y_bins = data[:, y_idx]
    y_nbins = int(nbins[y_idx])
    _assert_nonneg_codes(y_bins, "target column y_bins")
    y_counts = np.bincount(y_bins, minlength=y_nbins).astype(np.float64)
    y_total = float(y_counts.sum())
    if y_total > 0:
        y_p = y_counts / y_total
        y_p_nz = y_p[y_p > 0]
        h_y = float(-np.sum(y_p_nz * np.log(y_p_nz)))
    else:
        h_y = 0.0

    mi_to_target = np.full(n_features_in, np.nan, dtype=np.float64)
    su_to_target = np.full(n_features_in, np.nan, dtype=np.float64)
    bins_dict: dict[str, np.ndarray] | None = {} if retain_bins else None
    nbins_dict: dict[str, int] | None = {} if retain_bins else None

    for orig_idx, name in enumerate(feature_names_in):
        data_col = name_to_data_col.get(name)
        if data_col is None:
            # Column dropped or renamed by categorize_dataset (rare; the
            # original-frame name should be preserved for raw columns).
            continue

        # Direct MI from the cached screen output: keyed by (col_idx,) tuple.
        mi = cached_MIs.get((data_col,))
        if mi is None:
            # The cardinality-bias pre-screen at _screen_predictors.py:375 can
            # reject a column before computing MI; we leave NaN so consumers
            # know the value is missing rather than zero.
            mi_val = np.nan
        else:
            mi_val = float(mi)
        mi_to_target[orig_idx] = mi_val

        # Marginal H(X_j) from the binned column.
        x_bins = data[:, data_col]
        x_nb = int(nbins[data_col])
        _assert_nonneg_codes(x_bins, f"feature column {name!r} x_bins")
        x_counts = np.bincount(x_bins, minlength=x_nb).astype(np.float64)
        x_total = float(x_counts.sum())
        if x_total > 0:
            x_p = x_counts / x_total
            x_p_nz = x_p[x_p > 0]
            h_x = float(-np.sum(x_p_nz * np.log(x_p_nz)))
        else:
            h_x = 0.0

        # SU(X, y) = 2 * I(X, y) / (H(X) + H(y)). Use the cached MI when
        # available; degrade to NaN when either MI is missing or both entropies
        # are zero (constant column AND constant target).
        denom = h_x + h_y
        if (not np.isnan(mi_val)) and denom > 1e-12:
            su_to_target[orig_idx] = max(0.0, min(1.0, 2.0 * mi_val / denom))
        # else: leave NaN

        if retain_bins:
            # np.ascontiguousarray + astype copies the column out of the shared
            # ``data`` matrix so a later in-place edit (e.g. DCD aggregate
            # append) cannot corrupt the export. Cost: n_samples * dtype-bytes
            # per kept column.
            bins_dict[name] = np.ascontiguousarray(x_bins, dtype=dtype).copy()
            nbins_dict[name] = x_nb

    artifacts: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "feature_names": list(feature_names_in),
        "su_to_target": su_to_target,
        "mi_to_target": mi_to_target,
        "mrmr_kept_indices": [int(i) for i in np.asarray(support_original).tolist()],
        "n_samples_at_fit": n_samples,
    }
    if retain_bins:
        artifacts["bins"] = bins_dict
        artifacts["nbins_per_feature"] = nbins_dict
    return artifacts


def validate_artifact_dict(artifacts: dict | None) -> bool:
    """Cheap sanity check on a precomputed artifact dict received from a
    consumer (ShapProxiedFS / future selectors). Returns True if the dict is
    structurally valid AND carries at minimum a usable SU vector. Logs a
    warning and returns False on any failure -- callers should then fall back
    to recomputing from scratch.
    """
    if artifacts is None or not isinstance(artifacts, dict):
        return False
    su = artifacts.get("su_to_target")
    names = artifacts.get("feature_names")
    if su is None or names is None:
        logger.warning("Precomputed artifact dict missing required keys (su_to_target / feature_names); " "ignoring and recomputing from scratch.")
        return False
    try:
        su_arr = np.asarray(su)
    except Exception:  # pragma: no cover -- defensive
        return False
    if su_arr.ndim != 1 or su_arr.shape[0] != len(names):
        logger.warning(
            "Precomputed artifact su_to_target shape %s does not match feature_names len %d; " "ignoring and recomputing from scratch.",
            su_arr.shape,
            len(names),
        )
        return False
    return True
