"""Precomputed-artifact consumer for ShapProxiedFS.

Carved sibling-style off ``shap_proxied_fs.py`` (which is already >1k LOC) per
the module-size policy. Handles validation + axis-alignment of an
``MRMR.export_artifacts()`` dict (see ``_mrmr_artifacts.py``) against the
``X.columns`` the caller hands to ShapProxiedFS.fit().

The canonical pipeline is::

    mrmr = MRMR(n_features_to_keep=K, retain_artifacts=True).fit(X, y)
    art = mrmr.export_artifacts()
    X_narrow = X.iloc[:, mrmr.support_]
    # Caller restricts the artifact dict to the narrowed feature set:
    art_narrow = restrict_artifacts(art, mrmr.support_)
    sps = ShapProxiedFS(precomputed=art_narrow).fit(X_narrow, y)

ShapProxiedFS then skips its own univariate F-statistic pre-screen and ranks
candidates by ``SU(X_j, y)`` from the MRMR screen. The selected subset is
unchanged for SU-vs-F-ranking-equivalent regimes; on mixed-cardinality data
the SU ranking is more honest (Witten-Frank-Hall cardinality normalisation).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def restrict_artifacts(artifacts: dict, kept_indices) -> dict:
    """Slice an MRMR artifact dict to the subset of features actually kept.

    Useful in the canonical pipeline where MRMR narrowed ``X`` from N to K
    features and the downstream ShapProxiedFS sees the K-column ``X_narrow``;
    the artifact arrays (axis-aligned to the original N) must be re-axis-aligned
    to the K-column survivor set so feature_names == X_narrow.columns.

    Parameters
    ----------
    artifacts : dict
        Output of ``MRMR.export_artifacts()``. Schema in
        ``filters/_mrmr_artifacts.py``.
    kept_indices : array-like[int]
        Positional indices into ``artifacts['feature_names']`` to keep,
        typically ``mrmr.support_``.

    Returns
    -------
    dict
        New dict with su_to_target / mi_to_target / feature_names sliced to
        ``kept_indices``; bins (if present) restricted to the kept feature
        names; schema_version + n_samples_at_fit forwarded unchanged.
    """
    if not isinstance(artifacts, dict):
        raise TypeError(
            f"restrict_artifacts: expected dict from MRMR.export_artifacts(), got {type(artifacts).__name__}"
        )
    idx = np.asarray(kept_indices, dtype=np.intp)
    names_full = list(artifacts.get("feature_names", []))
    if names_full and idx.size > 0 and int(idx.max()) >= len(names_full):
        raise ValueError(
            f"restrict_artifacts: kept_indices max={int(idx.max())} >= feature_names len={len(names_full)}"
        )

    out: dict[str, Any] = {}
    if "schema_version" in artifacts:
        out["schema_version"] = artifacts["schema_version"]
    if "n_samples_at_fit" in artifacts:
        out["n_samples_at_fit"] = artifacts["n_samples_at_fit"]

    out["feature_names"] = [names_full[i] for i in idx.tolist()] if names_full else []

    su = artifacts.get("su_to_target")
    if su is not None:
        su_arr = np.asarray(su)
        out["su_to_target"] = su_arr[idx] if su_arr.size else su_arr

    mi = artifacts.get("mi_to_target")
    if mi is not None:
        mi_arr = np.asarray(mi)
        out["mi_to_target"] = mi_arr[idx] if mi_arr.size else mi_arr

    bins = artifacts.get("bins")
    if isinstance(bins, dict):
        kept_names = out["feature_names"]
        out["bins"] = {n: bins[n] for n in kept_names if n in bins}
        nbins_pf = artifacts.get("nbins_per_feature")
        if isinstance(nbins_pf, dict):
            out["nbins_per_feature"] = {n: nbins_pf[n] for n in kept_names if n in nbins_pf}

    # Drop ``mrmr_kept_indices`` from the restricted view -- it referred to
    # the original axis; consumers operating on the restricted dict don't
    # need it (the survivor set IS the restricted axis).
    return out


def align_precomputed_to_X(
    precomputed: dict | None,
    X: pd.DataFrame | np.ndarray,
) -> tuple[dict | None, dict]:
    """Validate a precomputed dict and align it to ``X.columns``.

    Returns ``(aligned_dict_or_None, report_block)``. Report block carries a
    small diagnostic dict the caller surfaces under
    ``shap_proxy_report_['precomputed_used']`` so the pipeline result records
    whether the artifacts were honoured.

    Failure modes (warn + return None):
    * precomputed is not a dict or missing required keys
    * feature_names doesn't match X.columns (after permitting subset / superset
      with explicit re-indexing -- see logic below)
    * SU / MI arrays have wrong shape

    Success modes:
    * Exact match (names == X.columns in order) -> arrays consumed verbatim
    * Permutation match (same set, different order) -> reindex arrays to
      X.columns order
    * X.columns is a SUBSET of feature_names -> slice arrays to X.columns
    """
    report: dict[str, Any] = {
        "honoured": False,
        "reason": "no_precomputed_provided",
    }
    if precomputed is None:
        return None, report

    if not isinstance(precomputed, dict):
        report["reason"] = f"not_a_dict_got_{type(precomputed).__name__}"
        logger.warning(
            "ShapProxiedFS(precomputed=...): expected dict, got %s; ignoring.",
            type(precomputed).__name__,
        )
        return None, report

    su = precomputed.get("su_to_target")
    names = precomputed.get("feature_names")
    if su is None or names is None:
        report["reason"] = "missing_required_keys"
        logger.warning(
            "ShapProxiedFS(precomputed=...): missing 'su_to_target' or 'feature_names'; ignoring."
        )
        return None, report

    su_arr = np.asarray(su, dtype=np.float64)
    names = list(names)
    if su_arr.ndim != 1 or su_arr.shape[0] != len(names):
        report["reason"] = (
            f"shape_mismatch_su={su_arr.shape}_names={len(names)}"
        )
        logger.warning(
            "ShapProxiedFS(precomputed=...): su_to_target shape %s vs feature_names len %d; ignoring.",
            su_arr.shape, len(names),
        )
        return None, report

    if hasattr(X, "columns"):
        X_cols = list(X.columns)
    else:
        # numpy ndarray input -- ShapProxiedFS internally wraps to DataFrame
        # but we can still align if the precomputed dict's feature_names are
        # synthesizable positional names; otherwise reject.
        X_cols = [f"f{i}" for i in range(X.shape[1])]

    if names == X_cols:
        aligned = dict(precomputed)
        report["honoured"] = True
        report["reason"] = "exact_match"
        report["n_features"] = len(X_cols)
        report["su_available"] = True
        report["mi_available"] = precomputed.get("mi_to_target") is not None
        report["bins_available"] = isinstance(precomputed.get("bins"), dict)
        return aligned, report

    name_to_idx = {n: i for i, n in enumerate(names)}

    if set(X_cols).issubset(name_to_idx):
        # Re-index by X_cols. Handles both permutation and the
        # post-MRMR-narrowing case where ``X_narrow.columns`` is a subset of
        # the original artifact axis (caller forgot to call
        # ``restrict_artifacts``; we recover transparently).
        order = np.array([name_to_idx[c] for c in X_cols], dtype=np.intp)
        aligned: dict[str, Any] = {
            "feature_names": list(X_cols),
            "su_to_target": su_arr[order],
        }
        if "schema_version" in precomputed:
            aligned["schema_version"] = precomputed["schema_version"]
        if "n_samples_at_fit" in precomputed:
            aligned["n_samples_at_fit"] = precomputed["n_samples_at_fit"]
        mi = precomputed.get("mi_to_target")
        if mi is not None:
            mi_arr = np.asarray(mi, dtype=np.float64)
            if mi_arr.shape == su_arr.shape:
                aligned["mi_to_target"] = mi_arr[order]
        bins = precomputed.get("bins")
        if isinstance(bins, dict):
            aligned["bins"] = {n: bins[n] for n in X_cols if n in bins}
        nbins_pf = precomputed.get("nbins_per_feature")
        if isinstance(nbins_pf, dict):
            aligned["nbins_per_feature"] = {n: nbins_pf[n] for n in X_cols if n in nbins_pf}

        report["honoured"] = True
        report["reason"] = (
            "permutation_match" if len(X_cols) == len(names) else "subset_match"
        )
        report["n_features"] = len(X_cols)
        report["su_available"] = True
        report["mi_available"] = "mi_to_target" in aligned
        report["bins_available"] = "bins" in aligned
        return aligned, report

    # Names don't form a subset; consumer / producer disagree -> reject.
    missing = set(X_cols) - set(names)
    report["reason"] = "feature_name_mismatch"
    report["n_missing"] = len(missing)
    report["sample_missing"] = list(sorted(missing))[:8]
    logger.warning(
        "ShapProxiedFS(precomputed=...): %d X.columns are absent from precomputed feature_names "
        "(sample: %s); ignoring artifacts and recomputing from scratch.",
        len(missing), list(sorted(missing))[:5],
    )
    return None, report


def su_to_prefilter_keep(
    aligned: dict,
    *,
    keep_top: int,
) -> np.ndarray:
    """Rank features by SU(X_j, y) descending and return the top-keep positional indices.

    Used in place of the ANOVA F-statistic pre-screen when the precomputed
    dict carries an SU vector. NaN SU values are treated as -inf (sorted last)
    so cardinality-bias-rejected features land at the bottom of the ranking.

    The ranking is BIT-IDENTICAL to ``np.argsort(-su)`` for finite values,
    with NaN handling that mirrors the F-statistic prefilter's behaviour
    (which returns 0.0 for constant columns and lets them sink).
    """
    su = np.asarray(aligned["su_to_target"], dtype=np.float64).copy()
    su[~np.isfinite(su)] = -np.inf
    keep_top = int(min(keep_top, su.shape[0]))
    if keep_top <= 0:
        return np.empty(0, dtype=np.intp)
    # ``argsort`` is stable on numpy 1.15+; the (-su, idx) tuple sort is
    # equivalent in semantics but slower. argsort + reverse-take is fine here.
    order = np.argsort(-su, kind="stable")
    return order[:keep_top].astype(np.intp)
