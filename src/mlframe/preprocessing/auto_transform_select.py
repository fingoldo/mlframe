"""Automatic per-column transform selection via a cheap univariate probe model.

Rather than hand-picking a preprocessing transform per numeric column, fit a cheap 1-feature probe model
against the target for each candidate transform (identity, log1p, RankGauss, and the sklearn scaler zoo from
:func:`mlframe.preprocessing.scalers.make_all_scalers`) and keep whichever transform gives the best
cross-validated univariate score -- an automated alternative to guessing which scaling a given column needs.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

import numpy as np
import pandas as pd


class _Probe(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> Any: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...

from mlframe.feature_selection.filters import generate_rankgauss_features
from mlframe.preprocessing.scalers import make_all_scalers


def _candidate_transforms() -> List[str]:
    return ["identity", "log1p_signed", "rankgauss"] + [name for name, _ in make_all_scalers()]


def _apply_transform(x: np.ndarray, transform_name: str) -> Optional[np.ndarray]:
    if transform_name == "identity":
        return x
    if transform_name == "log1p_signed":
        return np.asarray(np.sign(x) * np.log1p(np.abs(x)), dtype=np.float64)
    if transform_name == "rankgauss":
        enc_df, _ = generate_rankgauss_features(pd.DataFrame({"c": x}), ["c"])
        return np.asarray(enc_df.iloc[:, 0].to_numpy(), dtype=np.float64)
    for name, scaler in make_all_scalers():
        if name == transform_name:
            try:
                return np.asarray(scaler.fit_transform(x.reshape(-1, 1)).ravel(), dtype=np.float64)
            except ValueError:
                return None
    raise ValueError(f"_apply_transform: unknown transform_name {transform_name!r}")


def select_column_transforms(
    df: pd.DataFrame,
    y: np.ndarray,
    columns: Optional[Sequence[str]] = None,
    probe_model_fn: Optional[Callable[[], _Probe]] = None,
    n_splits: int = 3,
    candidate_transforms: Optional[Sequence[str]] = None,
    task: str = "classification",
    random_state: int = 0,
) -> Dict[str, dict]:
    """Pick the best-scoring transform per numeric column via a cheap cross-validated univariate probe.

    Parameters
    ----------
    df
        Feature frame.
    y
        Target aligned to ``df``.
    columns
        Columns to audit; defaults to all numeric columns.
    probe_model_fn
        Zero-arg factory returning a fresh sklearn-compatible estimator (must expose
        ``predict_proba`` for ``task="classification"`` or ``predict`` for ``task="regression"``); defaults
        to a small ``LogisticRegression``/``Ridge``.
    n_splits
        CV folds for the probe score.
    candidate_transforms
        Transform names to try; defaults to identity, log1p (sign-preserving), RankGauss, and the sklearn
        scaler zoo (`mlframe.preprocessing.scalers.make_all_scalers`).
    task
        ``"classification"`` (scored by ROC AUC) or ``"regression"`` (scored by negative RMSE, higher-better).

    Returns
    -------
    dict
        ``{column_name: {"best_transform": str, "best_score": float, "all_scores": {transform: score}}}``.
    """
    if columns is None:
        columns = [c for c in df.select_dtypes(include=[np.number]).columns]
    columns = list(columns)
    if candidate_transforms is None:
        candidate_transforms = _candidate_transforms()

    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import KFold, StratifiedKFold

    from mlframe.metrics.core import fast_roc_auc

    if probe_model_fn is None:
        if task == "classification":
            probe_model_fn = lambda: LogisticRegression(max_iter=200)  # noqa: E731
        elif task == "regression":
            probe_model_fn = lambda: Ridge()  # noqa: E731
        else:
            raise ValueError(f"select_column_transforms: task must be 'classification' or 'regression'; got {task!r}")

    y = np.asarray(y)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state) if task == "classification" else KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_indices = list(splitter.split(np.zeros(len(y)), y))

    results: Dict[str, dict] = {}
    for col in columns:
        raw = df[col].to_numpy(dtype=np.float64)
        finite_fill = raw.copy()
        finite_fill[~np.isfinite(finite_fill)] = np.nanmedian(finite_fill[np.isfinite(finite_fill)]) if np.isfinite(finite_fill).any() else 0.0

        scores: Dict[str, float] = {}
        for transform_name in candidate_transforms:
            transformed = _apply_transform(finite_fill, transform_name)
            if transformed is None or not np.all(np.isfinite(transformed)):
                continue
            fold_scores = []
            for train_idx, test_idx in fold_indices:
                model = probe_model_fn()
                model.fit(transformed[train_idx].reshape(-1, 1), y[train_idx])
                if task == "classification":
                    proba = model.predict_proba(transformed[test_idx].reshape(-1, 1))[:, 1]
                    fold_scores.append(fast_roc_auc(y[test_idx], proba))
                else:
                    pred = model.predict(transformed[test_idx].reshape(-1, 1))
                    fold_scores.append(-float(np.sqrt(np.mean((y[test_idx] - pred) ** 2))))
            scores[transform_name] = float(np.mean(fold_scores))

        if not scores:
            continue
        best_transform = max(scores, key=scores.get)  # type: ignore[arg-type]
        results[col] = {"best_transform": best_transform, "best_score": scores[best_transform], "all_scores": scores}

    return results


__all__ = ["select_column_transforms"]
