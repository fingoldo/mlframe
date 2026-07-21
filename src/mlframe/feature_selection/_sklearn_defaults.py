"""Shared task-detection heuristic for the cluster's ad-hoc "task-appropriate RandomForest default" helpers.

``ace._default_estimator`` and ``functional_adapters._default_tree_estimator`` were byte-for-byte
duplicates of the same low-cardinality classification-vs-regression heuristic (the latter's own docstring
says "mirrors ace._default_estimator verbatim"). Hoisted here so the float-binary-target ambiguity fix only
needs to exist once.
"""
from __future__ import annotations

import warnings

import numpy as np


def is_classification_target(y: np.ndarray, n: int | None = None) -> bool:
    """Low-cardinality heuristic: integer/label/bool/string y with few unique values is classification.

    A float-typed y that LOOKS like a low-cardinality classification target (e.g. ``{0.0, 1.0}`` labels,
    all whole-number values) still routes to regression (the conservative default -- a genuinely
    continuous-but-coarse score is a real, more common case than an accidentally-float-cast label), but
    now WARNS about the ambiguity instead of silently defaulting -- unlike the int/label branch, this
    exact ambiguity previously had no signal at all, unlike ``HybridSelector``'s explicit validate/warn on
    the same dtype-vs-value-sniff mismatch.
    """
    y_arr = np.asarray(y)
    n_eff = int(n) if n is not None else int(y_arr.shape[0])
    if y_arr.dtype.kind in ("i", "u", "b", "O", "U", "S"):
        return bool(np.unique(y_arr).size <= max(20, int(np.sqrt(max(n_eff, 1)))))
    if y_arr.dtype.kind == "f":
        finite_y = y_arr[np.isfinite(y_arr)]
        if finite_y.size and np.all(np.mod(finite_y, 1.0) == 0.0) and np.unique(finite_y).size <= max(20, int(np.sqrt(max(n_eff, 1)))):
            warnings.warn(
                f"is_classification_target: y is float-dtype but looks like a low-cardinality integer-valued "
                f"target ({np.unique(finite_y).size} distinct values, all whole numbers) -- defaulting to "
                "regression. If this is really a classification target, cast y to int first.",
                stacklevel=3,
            )
    return False


def default_tree_estimator(y: np.ndarray, n: int | None = None, random_state: int = 0, n_estimators: int = 120):
    """Task-appropriate RandomForest default: classifier for low-cardinality integer/label y, else regressor."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    if is_classification_target(y, n):
        return RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=random_state, max_features="sqrt")
    return RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=random_state, max_features="sqrt")


__all__ = ["is_classification_target", "default_tree_estimator"]
