"""``gaussian_power_transform_search``: pick the power transform closest to Gaussian, unsupervised (no target).

``mlframe.preprocessing.auto_transform_select.select_column_transforms`` already searches a transform zoo, but
its selection criterion is a downstream SUPERVISED probe score (cross-validated AUC/RMSE against a target) --
unusable before a target exists, or when the goal is purely distribution shape (e.g. feeding a linear/distance-
based model that assumes near-Gaussian inputs, independent of any specific label). This module searches a small
set of power/exponential transforms (identity, sqrt, log1p, Box-Cox, Yeo-Johnson) and keeps whichever minimizes
absolute skewness -- a fast, unsupervised normality proxy (full normality tests like Shapiro-Wilk are O(n log n)
per candidate and add little over skewness for this ranking purpose at typical feature-engineering scale).

Skew minimization is purely a shape criterion: a transform can flatten a distribution while weakening the very
linear relationship a downstream model relies on -- e.g. a lognormal feature ``x`` fed into ``y = a*x + noise``
has a near-perfect Pearson correlation with ``y`` on the raw scale, but ``log(x)`` (the skew-minimizing pick) can
correlate much more weakly, since the true relationship is linear in ``x``, not in ``log(x)``. Spearman rank
correlation can't catch this -- every candidate transform here is monotonic, so rank correlation with any target
is invariant under all of them by construction. Passing ``y`` with ``require_target_correlation_retention`` opts
into a Pearson-based guard: candidates that drop the linear relationship below the given fraction of the raw
column's correlation with ``y`` are excluded from the skew-minimizing pick, falling back toward less aggressive
transforms (``identity`` in the worst case). Omitting ``y``/the retention threshold reproduces the exact prior
unsupervised behavior.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, skew

_CANDIDATE_TRANSFORMS = ("identity", "sqrt_signed", "log1p_signed", "boxcox", "yeo_johnson")


def _apply_transform(x: np.ndarray, transform_name: str) -> Optional[np.ndarray]:
    if transform_name == "identity":
        return x
    if transform_name == "sqrt_signed":
        return np.asarray(np.sign(x) * np.sqrt(np.abs(x)), dtype=np.float64)
    if transform_name == "log1p_signed":
        return np.asarray(np.sign(x) * np.log1p(np.abs(x)), dtype=np.float64)
    if transform_name == "boxcox":
        if np.any(x <= 0):
            return None  # Box-Cox requires strictly positive input.
        from scipy.stats import boxcox

        try:
            transformed, _ = boxcox(x)
        except ValueError:
            return None
        return np.asarray(transformed, dtype=np.float64)
    if transform_name == "yeo_johnson":
        from sklearn.preprocessing import PowerTransformer

        try:
            transformed = PowerTransformer(method="yeo-johnson").fit_transform(x.reshape(-1, 1)).ravel()
        except ValueError:
            return None
        return np.asarray(transformed, dtype=np.float64)
    raise ValueError(f"_apply_transform: unknown transform_name {transform_name!r}")


def _safe_abs_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Absolute Pearson correlation, 0.0 for degenerate (zero-variance or too-short) inputs."""
    if a.size < 3 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return 0.0
    r, _ = pearsonr(a, b)
    return float(abs(r)) if np.isfinite(r) else 0.0


def gaussian_power_transform_search(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    candidate_transforms: Optional[Sequence[str]] = None,
    y: Optional[Union[np.ndarray, pd.Series]] = None,
    require_target_correlation_retention: Optional[float] = None,
) -> Dict[str, dict]:
    """Pick the per-column power transform that minimizes absolute skewness (closest to Gaussian), unsupervised.

    Parameters
    ----------
    df
        Feature frame.
    columns
        Columns to search; defaults to all numeric columns.
    candidate_transforms
        Transform names to try; defaults to ``identity``, ``sqrt_signed``, ``log1p_signed``, ``boxcox``
        (skipped when the column has non-positive values), ``yeo_johnson``.
    y
        Optional target array, same length as ``df``, used only when ``require_target_correlation_retention``
        is set. Rows non-finite in either the column or ``y`` are dropped pairwise for the correlation check.
    require_target_correlation_retention
        Opt-in guard, ``None`` by default (exact prior unsupervised behavior). When set (e.g. ``0.9``), a
        candidate transform is only eligible for the skew-minimizing pick if its absolute Pearson correlation
        with ``y`` is at least this fraction of the RAW column's absolute Pearson correlation with ``y`` --
        guards against a transform that flattens skew while destroying a linear target relationship. Requires
        ``y``. If the raw column has ~zero correlation with ``y`` (nothing to preserve), the guard is skipped.
        If every candidate fails the guard, ``identity`` is force-kept as the fallback (it always satisfies the
        guard by construction, since its retention ratio is 1.0).

    Returns
    -------
    dict
        ``{column_name: {"best_transform": str, "best_abs_skew": float, "all_abs_skew": {transform: float},
        ...target-guard keys present only when require_target_correlation_retention is set: "raw_target_corr",
        "all_target_corr", "target_correlation_rejected"}}``.
    """
    if columns is None:
        columns = [c for c in df.select_dtypes(include=[np.number]).columns]
    columns = list(columns)
    if candidate_transforms is None:
        candidate_transforms = _CANDIDATE_TRANSFORMS

    if require_target_correlation_retention is not None and y is None:
        raise ValueError("gaussian_power_transform_search: require_target_correlation_retention needs y")
    y_arr: Optional[np.ndarray] = None if y is None else np.asarray(y, dtype=np.float64)

    results: Dict[str, dict] = {}
    for col in columns:
        raw = df[col].to_numpy(dtype=np.float64)
        finite = raw[np.isfinite(raw)]
        if finite.size < 3:
            continue
        finite_fill = raw.copy()
        finite_fill[~np.isfinite(finite_fill)] = np.median(finite)

        abs_skews: Dict[str, float] = {}
        transformed_by_name: Dict[str, np.ndarray] = {}
        for transform_name in candidate_transforms:
            transformed = _apply_transform(finite_fill, transform_name)
            if transformed is None or not np.all(np.isfinite(transformed)) or np.std(transformed) == 0.0:
                continue
            abs_skews[transform_name] = float(abs(skew(transformed)))
            transformed_by_name[transform_name] = transformed

        if not abs_skews:
            continue

        info: Dict[str, object] = {}
        eligible = abs_skews
        if require_target_correlation_retention is not None and y_arr is not None:
            pair_mask = np.isfinite(finite_fill) & np.isfinite(y_arr)
            raw_target_corr = _safe_abs_pearson(finite_fill[pair_mask], y_arr[pair_mask])
            min_required = raw_target_corr * require_target_correlation_retention
            all_target_corr: Dict[str, float] = {}
            rejected = []
            eligible = {}
            for transform_name, abs_skew_val in abs_skews.items():
                transformed = transformed_by_name[transform_name]
                target_corr = _safe_abs_pearson(transformed[pair_mask], y_arr[pair_mask])
                all_target_corr[transform_name] = target_corr
                if transform_name == "identity" or raw_target_corr == 0.0 or target_corr >= min_required:
                    eligible[transform_name] = abs_skew_val
                else:
                    rejected.append(transform_name)
            info["raw_target_corr"] = raw_target_corr
            info["all_target_corr"] = all_target_corr
            info["target_correlation_rejected"] = rejected

        best_transform = min(eligible, key=eligible.get)  # type: ignore[arg-type]
        info = {"best_transform": best_transform, "best_abs_skew": abs_skews[best_transform], "all_abs_skew": abs_skews, **info}
        results[col] = info

    return results


def apply_gaussian_power_transform(df: pd.DataFrame, search_result: Dict[str, dict]) -> pd.DataFrame:
    """Apply the winning transform from :func:`gaussian_power_transform_search`'s result to each column.

    Returns ``df`` (shallow copy) with each searched column replaced by its best-scoring transform in place.
    """
    out = df.copy(deep=False)
    for col, info in search_result.items():
        raw = out[col].to_numpy(dtype=np.float64)
        finite = raw[np.isfinite(raw)]
        finite_fill = raw.copy()
        if finite.size:
            finite_fill[~np.isfinite(finite_fill)] = np.median(finite)
        transformed = _apply_transform(finite_fill, info["best_transform"])
        if transformed is not None:
            out[col] = transformed
    return out


__all__ = ["gaussian_power_transform_search", "apply_gaussian_power_transform"]
