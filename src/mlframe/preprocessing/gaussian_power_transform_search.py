"""``gaussian_power_transform_search``: pick the power transform closest to Gaussian, unsupervised (no target).

``mlframe.preprocessing.auto_transform_select.select_column_transforms`` already searches a transform zoo, but
its selection criterion is a downstream SUPERVISED probe score (cross-validated AUC/RMSE against a target) --
unusable before a target exists, or when the goal is purely distribution shape (e.g. feeding a linear/distance-
based model that assumes near-Gaussian inputs, independent of any specific label). This module searches a small
set of power/exponential transforms (identity, sqrt, log1p, Box-Cox, Yeo-Johnson) and keeps whichever minimizes
absolute skewness -- a fast, unsupervised normality proxy (full normality tests like Shapiro-Wilk are O(n log n)
per candidate and add little over skewness for this ranking purpose at typical feature-engineering scale).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import skew

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


def gaussian_power_transform_search(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    candidate_transforms: Optional[Sequence[str]] = None,
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

    Returns
    -------
    dict
        ``{column_name: {"best_transform": str, "best_abs_skew": float, "all_abs_skew": {transform: float}}}``.
    """
    if columns is None:
        columns = [c for c in df.select_dtypes(include=[np.number]).columns]
    columns = list(columns)
    if candidate_transforms is None:
        candidate_transforms = _CANDIDATE_TRANSFORMS

    results: Dict[str, dict] = {}
    for col in columns:
        raw = df[col].to_numpy(dtype=np.float64)
        finite = raw[np.isfinite(raw)]
        if finite.size < 3:
            continue
        finite_fill = raw.copy()
        finite_fill[~np.isfinite(finite_fill)] = np.median(finite)

        abs_skews: Dict[str, float] = {}
        for transform_name in candidate_transforms:
            transformed = _apply_transform(finite_fill, transform_name)
            if transformed is None or not np.all(np.isfinite(transformed)) or np.std(transformed) == 0.0:
                continue
            abs_skews[transform_name] = float(abs(skew(transformed)))

        if not abs_skews:
            continue
        best_transform = min(abs_skews, key=abs_skews.get)  # type: ignore[arg-type]
        results[col] = {"best_transform": best_transform, "best_abs_skew": abs_skews[best_transform], "all_abs_skew": abs_skews}

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
