"""Automatic per-column transform selection via a cheap univariate probe model.

Rather than hand-picking a preprocessing transform per numeric column, fit a cheap 1-feature probe model
against the target for each candidate transform (identity, log1p, RankGauss, and the sklearn scaler zoo from
:func:`mlframe.preprocessing.scalers.make_all_scalers`) and keep whichever transform gives the best
cross-validated univariate score -- an automated alternative to guessing which scaling a given column needs.

An opt-in ``multivariate_probe`` mode extends the same candidate-transform loop with a small nonlinear probe
fit on the transformed column plus a few correlated context columns (and their pairwise product with the
transformed column). This catches columns whose signal is purely an INTERACTION with another feature --
invisible to any univariate probe, which sees only chance-level AUC/RMSE regardless of transform choice.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

import numpy as np
import pandas as pd


class _Probe(Protocol):
    """Structural type for the small sklearn-like estimators used as univariate/multivariate probes."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Fit the probe model on features ``X`` and target ``y``."""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the probe model's predictions for ``X``."""
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return the probe model's predicted class probabilities for ``X``."""
        ...

from mlframe.feature_selection.filters import generate_rankgauss_features
from mlframe.preprocessing.scalers import make_all_scalers


def _candidate_transforms() -> List[str]:
    """Return the names of all transforms considered by the per-column selection loop."""
    return ["identity", "log1p_signed", "rankgauss"] + [name for name, _ in make_all_scalers()]


def _apply_transform(x: np.ndarray, transform_name: str) -> Optional[np.ndarray]:
    """Apply the named transform to ``x`` in one shot (fit and transform on the SAME data).

    Only safe for a transform's final replay over the whole column (e.g. after the CV loop has
    already picked the best transform); never for scoring, where fitting on the same rows the
    score is read from leaks that fold's own statistics into its "held-out" evaluation. Use
    :func:`_fit_transform_fold` inside any CV loop instead.
    """
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


def _fit_transform_fold(x: np.ndarray, transform_name: str, train_idx: np.ndarray, test_idx: np.ndarray) -> Optional["tuple[np.ndarray, np.ndarray]"]:
    """Fit ``transform_name`` on ``x[train_idx]`` ONLY, then apply it to both ``x[train_idx]`` and
    ``x[test_idx]``. Returns ``None`` if the transform fails to fit on the train slice.

    Every candidate transform except identity/log1p_signed carries fit statistics (a scaler's
    mean/std/min-max, RankGauss's sorted fit values) derived from whatever data it is fit on;
    fitting on the FULL column (train+test) before a CV split leaks the test fold's own
    statistics into the "held-out" score, optimistically biasing it. Fitting per-fold on the
    train slice only, then replaying onto the test slice, matches standard CV preprocessing
    practice (fit on train, transform train and test).
    """
    x_train, x_test = x[train_idx], x[test_idx]
    if transform_name == "identity":
        return x_train, x_test
    if transform_name == "log1p_signed":
        # Stateless per-element transform, no fit statistics -- safe to apply independently.
        f = lambda a: np.asarray(np.sign(a) * np.log1p(np.abs(a)), dtype=np.float64)  # noqa: E731
        return f(x_train), f(x_test)
    if transform_name == "rankgauss":
        from mlframe.feature_selection.filters._extra_fe_families import apply_rankgauss, engineered_name_rankgauss

        enc_df, recipes = generate_rankgauss_features(pd.DataFrame({"c": x_train}), ["c"])
        train_out = np.asarray(enc_df.iloc[:, 0].to_numpy(), dtype=np.float64)
        recipe = recipes[engineered_name_rankgauss("c")]
        test_out = np.asarray(apply_rankgauss(pd.DataFrame({"c": x_test}), recipe), dtype=np.float64).ravel()
        return train_out, test_out
    for name, scaler in make_all_scalers():
        if name == transform_name:
            try:
                scaler.fit(x_train.reshape(-1, 1))
                train_out = np.asarray(scaler.transform(x_train.reshape(-1, 1)).ravel(), dtype=np.float64)
                test_out = np.asarray(scaler.transform(x_test.reshape(-1, 1)).ravel(), dtype=np.float64)
                return train_out, test_out
            except ValueError:
                return None
    raise ValueError(f"_fit_transform_fold: unknown transform_name {transform_name!r}")


def _top_correlated_context_columns(df: pd.DataFrame, col: str, pool: List[str], n_context_features: int) -> List[str]:
    """Rank ``pool`` columns by |Pearson r| against ``col`` and keep the top ``n_context_features``."""
    target = df[col].to_numpy(dtype=np.float64)
    target_fill = target.copy()
    target_fill[~np.isfinite(target_fill)] = np.nanmedian(target_fill[np.isfinite(target_fill)]) if np.isfinite(target_fill).any() else 0.0

    scored: List[tuple] = []
    for cand in pool:
        if cand == col:
            continue
        other = df[cand].to_numpy(dtype=np.float64)
        other_fill = other.copy()
        other_fill[~np.isfinite(other_fill)] = np.nanmedian(other_fill[np.isfinite(other_fill)]) if np.isfinite(other_fill).any() else 0.0
        if np.std(target_fill) == 0.0 or np.std(other_fill) == 0.0:
            continue
        corr = np.corrcoef(target_fill, other_fill)[0, 1]
        if np.isfinite(corr):
            scored.append((abs(corr), cand))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [name for _, name in scored[:n_context_features]]


def select_column_transforms(
    df: pd.DataFrame,
    y: np.ndarray,
    columns: Optional[Sequence[str]] = None,
    probe_model_fn: Optional[Callable[[], _Probe]] = None,
    n_splits: int = 3,
    candidate_transforms: Optional[Sequence[str]] = None,
    task: str = "classification",
    random_state: int = 0,
    multivariate_probe: bool = False,
    context_columns: Optional[Sequence[str]] = None,
    n_context_features: int = 2,
    multivariate_probe_model_fn: Optional[Callable[[], _Probe]] = None,
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
    multivariate_probe
        Opt-in (default ``False``, univariate-only, output bit-identical to omitting these params). When
        ``True``, each candidate transform is scored by fitting ``multivariate_probe_model_fn`` on the
        transformed column plus its top-``n_context_features`` correlated context columns plus their
        pairwise product with the transformed column, instead of the plain 1-feature univariate probe. This
        catches transforms whose value is an interaction effect with another column, invisible to any
        univariate probe (which would score every transform at chance level).
    context_columns
        Pool of columns eligible as multivariate context; defaults to all other numeric columns in ``df``.
        Ignored when ``multivariate_probe`` is ``False``.
    n_context_features
        How many top-|correlation| columns from ``context_columns`` to use as context per audited column.
        Ignored when ``multivariate_probe`` is ``False``.
    multivariate_probe_model_fn
        Zero-arg factory for the multivariate probe; defaults to a small
        ``HistGradientBoostingClassifier``/``HistGradientBoostingRegressor``. Ignored when
        ``multivariate_probe`` is ``False``.

    Returns
    -------
    dict
        ``{column_name: {"best_transform": str, "best_score": float, "all_scores": {transform: score}}}``.
        The ``"probe_mode"`` (``"multivariate"``) and ``"context_columns"`` keys are added ONLY when
        ``multivariate_probe=True`` -- the default (univariate) path omits them entirely, keeping its output
        dict bit-identical to before this option existed. A caller that wants a uniform schema regardless of
        mode should read them via ``.get("probe_mode", "univariate")`` / ``.get("context_columns", [])``.
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

    if multivariate_probe and multivariate_probe_model_fn is None:
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        if task == "classification":
            multivariate_probe_model_fn = lambda: HistGradientBoostingClassifier(max_iter=50, max_depth=3, random_state=random_state)  # noqa: E731
        else:
            multivariate_probe_model_fn = lambda: HistGradientBoostingRegressor(max_iter=50, max_depth=3, random_state=random_state)  # noqa: E731

    if multivariate_probe:
        context_pool = list(context_columns) if context_columns is not None else [c for c in df.select_dtypes(include=[np.number]).columns]

    y = np.asarray(y)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state) if task == "classification" else KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_indices = list(splitter.split(np.zeros(len(y)), y))

    results: Dict[str, dict] = {}
    for col in columns:
        raw = df[col].to_numpy(dtype=np.float64)
        finite_fill = raw.copy()
        finite_fill[~np.isfinite(finite_fill)] = np.nanmedian(finite_fill[np.isfinite(finite_fill)]) if np.isfinite(finite_fill).any() else 0.0

        col_context_columns: List[str] = []
        context_matrix: Optional[np.ndarray] = None
        if multivariate_probe:
            col_context_columns = _top_correlated_context_columns(df, col, context_pool, n_context_features)
            context_arrays = []
            for ctx_col in col_context_columns:
                ctx_raw = df[ctx_col].to_numpy(dtype=np.float64)
                ctx_fill = ctx_raw.copy()
                ctx_fill[~np.isfinite(ctx_fill)] = np.nanmedian(ctx_fill[np.isfinite(ctx_fill)]) if np.isfinite(ctx_fill).any() else 0.0
                context_arrays.append(ctx_fill)
            context_matrix = np.column_stack(context_arrays) if context_arrays else np.zeros((len(finite_fill), 0))

        scores: Dict[str, float] = {}
        for transform_name in candidate_transforms:
            fold_scores = []
            transform_failed = False
            for train_idx, test_idx in fold_indices:
                # Fit the transform on THIS FOLD'S train rows only, then apply it to both train
                # and test -- fitting on the full column (train+test) before splitting would leak
                # the test fold's own statistics into its "held-out" score.
                fitted = _fit_transform_fold(finite_fill, transform_name, train_idx, test_idx)
                if fitted is None:
                    transform_failed = True
                    break
                transformed_train, transformed_test = fitted
                if not (np.all(np.isfinite(transformed_train)) and np.all(np.isfinite(transformed_test))):
                    transform_failed = True
                    break
                if multivariate_probe:
                    assert context_matrix is not None and multivariate_probe_model_fn is not None
                    ctx_train, ctx_test = context_matrix[train_idx], context_matrix[test_idx]
                    inter_train = [transformed_train.reshape(-1, 1) * ctx_train[:, i : i + 1] for i in range(ctx_train.shape[1])]
                    inter_test = [transformed_test.reshape(-1, 1) * ctx_test[:, i : i + 1] for i in range(ctx_test.shape[1])]
                    feature_train = np.column_stack([transformed_train.reshape(-1, 1), ctx_train, *inter_train])
                    feature_test = np.column_stack([transformed_test.reshape(-1, 1), ctx_test, *inter_test])
                    model = multivariate_probe_model_fn()
                    model.fit(feature_train, y[train_idx])
                    if task == "classification":
                        proba = model.predict_proba(feature_test)[:, 1]
                        fold_scores.append(fast_roc_auc(y[test_idx], proba))
                    else:
                        pred = model.predict(feature_test)
                        fold_scores.append(-float(np.sqrt(np.mean((y[test_idx] - pred) ** 2))))
                else:
                    model = probe_model_fn()
                    model.fit(transformed_train.reshape(-1, 1), y[train_idx])
                    if task == "classification":
                        proba = model.predict_proba(transformed_test.reshape(-1, 1))[:, 1]
                        fold_scores.append(fast_roc_auc(y[test_idx], proba))
                    else:
                        pred = model.predict(transformed_test.reshape(-1, 1))
                        fold_scores.append(-float(np.sqrt(np.mean((y[test_idx] - pred) ** 2))))
            if transform_failed or not fold_scores:
                continue
            scores[transform_name] = float(np.mean(fold_scores))

        if not scores:
            continue
        best_transform = max(scores, key=scores.get)  # type: ignore[arg-type]
        col_result = {"best_transform": best_transform, "best_score": scores[best_transform], "all_scores": scores}
        if multivariate_probe:
            col_result["probe_mode"] = "multivariate"
            col_result["context_columns"] = col_context_columns
        results[col] = col_result

    return results


__all__ = ["select_column_transforms"]
