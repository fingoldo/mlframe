"""Quick-model factory + scoring methods for BaselineDiagnostics.

Carved out of ``baseline_diagnostics`` via method-rebinding (W10E pattern). The
functions here are bound onto :class:`BaselineDiagnostics` at the parent module's
bottom; ``self`` stays the first arg so identity and behaviour are preserved.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def _make_quick_model(self, target_type: str,
                      init_score: np.ndarray | None = None,
                      n_jobs: int = -1):
    """Build a fresh quick LightGBM model. Lazy import keeps the
    diagnostic optional - if LightGBM is unavailable the whole
    component will skip with a clear error."""
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "BaselineDiagnostics requires lightgbm; install it or set "
            "BaselineDiagnosticsConfig.enabled=False."
        ) from exc

    kwargs = dict(
        n_estimators=self.config.quick_model_n_estimators,
        num_leaves=self.config.quick_model_num_leaves,
        learning_rate=self.config.quick_model_learning_rate,
        random_state=self.config.random_state,
        n_jobs=n_jobs,
        verbose=-1,
        force_col_wise=True,  # quiet the cold-cache "auto-choose" warning
    )
    if target_type == "regression":
        return lgb.LGBMRegressor(**kwargs)
    if target_type == "binary_classification":
        return lgb.LGBMClassifier(**kwargs)
    raise ValueError(f"Unsupported target_type for quick model: {target_type}")


def _fit_quick_and_score(
    self,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_cols: Sequence[str],
    cat_features: Sequence[str],
    target_type: str,
    metric_name: str,
    init_score: np.ndarray | None = None,
    inner_n_jobs: int = -1,
) -> tuple[float, np.ndarray]:
    """Fit quick LightGBM on a holdout split, return ``(metric, fi_array)``.

    FI is taken from the LightGBM model.feature_importances_ aligned
    to ``feature_cols`` order. Holdout = simple 80/20 random split
    seeded by ``config.random_state``. The diagnostic is meant to
    be cheap, not robust: a single fold is sufficient to surface
    dominant features at percentage-point resolution.
    """
    from sklearn.model_selection import train_test_split
    from .baseline_diagnostics import _compute_metric

    # Single train_test_split that includes init_score when present;
    # this guarantees the (X_tr, y_tr, init_score_tr) triple stays
    # row-aligned even with stratify=y on binary, where two
    # separate split calls would produce different shuffles.
    n = len(X)
    if n < 50:
        X_tr, y_tr = X, y
        X_va, y_va = X, y
        init_score_tr = init_score
        init_score_va_local = init_score
    else:
        stratify_arg = (
            y if target_type == "binary_classification"
            and len(np.unique(y)) > 1 else None
        )
        if init_score is not None:
            (X_tr, X_va, y_tr, y_va,
             init_score_tr, init_score_va_local) = train_test_split(
                X, y, init_score,
                test_size=0.2,
                random_state=self.config.random_state,
                stratify=stratify_arg,
            )
        else:
            X_tr, X_va, y_tr, y_va = train_test_split(
                X, y,
                test_size=0.2,
                random_state=self.config.random_state,
                stratify=stratify_arg,
            )
            init_score_tr = None
            init_score_va_local = None

    model = self._make_quick_model(target_type, n_jobs=inner_n_jobs)
    fit_kwargs = {}
    if cat_features:
        usable = [c for c in cat_features if c in X.columns]
        if usable:
            fit_kwargs["categorical_feature"] = usable
    if init_score_tr is not None:
        # LightGBM accepts init_score for both regression and binary via fit();
        # it's the per-row additive baseline (raw scale for regression, logit
        # scale for binary).
        fit_kwargs["init_score"] = init_score_tr

    model.fit(X_tr, y_tr, **fit_kwargs)

    if target_type == "binary_classification":
        if init_score_va_local is not None:
            # Booster trained on residual logit; predict_proba alone returns
            # sigmoid(tree_output) and MISSES the init_score offset. Use
            # raw_score=True to get the tree margin, add init_score, then sigmoid manually.
            try:
                tree_logit = np.asarray(
                    model.predict(X_va, raw_score=True),
                ).reshape(-1)
            except TypeError:
                tree_logit = np.asarray(
                    model.booster_.predict(X_va, raw_score=True),
                ).reshape(-1)
            full_logit = tree_logit + np.asarray(init_score_va_local).reshape(-1)
            # Numerically safe sigmoid.
            y_pred = np.where(
                full_logit >= 0,
                1.0 / (1.0 + np.exp(-full_logit)),
                np.exp(full_logit) / (1.0 + np.exp(full_logit)),
            )
        else:
            y_pred = model.predict_proba(X_va)[:, 1]
    else:
        y_pred = model.predict(X_va)
        if init_score_va_local is not None:
            # LightGBM regression predict adds init_score back in recent
            # versions. We add it explicitly to stay version-agnostic; if
            # double-added the test would catch a 2x deviation from y_va.
            y_pred = y_pred + init_score_va_local

    metric_val = _compute_metric(np.asarray(y_va), np.asarray(y_pred), metric_name)

    fi = np.asarray(getattr(model, "feature_importances_", []), dtype=np.float64)
    if fi.size != len(feature_cols):
        # Fallback: zero-importances if model didn't expose them.
        fi = np.zeros(len(feature_cols), dtype=np.float64)
    return metric_val, fi
