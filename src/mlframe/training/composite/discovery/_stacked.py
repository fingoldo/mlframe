"""Two-pass stacked variants of :meth:`CompositeTargetDiscovery.fit`.

Carved out of ``composite_discovery`` via method-rebinding to keep the parent
facade under the LOC budget. Methods are bound onto the class at the parent
module's bottom; ``self`` stays the first arg so identity and behaviour are
preserved.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from .screening import _extract_column_array

logger = logging.getLogger(__name__)


def fit_stacked(
    self,
    df: Any,
    target_col: str,
    feature_cols: Sequence[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray | None = None,
    test_idx: np.ndarray | None = None,
    *,
    n_oof_folds: int = 3,
    max_pass1_specs_to_stack: int = 3,
):
    """2-pass stacked composite discovery (Pack #3).

    Pass 1 runs the standard :meth:`fit`. For each of the top
    ``max_pass1_specs_to_stack`` specs (ranked by tiny CV-RMSE),
    compute OOF predictions on the train rows via
    :func:`composite_oof_predictions` and append them as new feature
    columns to ``df``. Pass 2 calls :meth:`fit` again on the
    augmented feature set; it may find composites where the
    residual-of-residual structure becomes the new dominant base
    (e.g. ``y = f(x_a) + g(x_b)``: pass 1 absorbs ``f(x_a)`` via
    ``linres-x_a``, pass 2 then finds ``linres-x_b`` on the
    leftover residual).

    Resulting ``self.specs_`` = pass1_specs UNION pass2_specs
    (collisions dropped, pass-1 wins).
    """
    self.fit(df, target_col, feature_cols, train_idx, val_idx, test_idx)
    pass1_specs = list(self.specs_) if self.specs_ else []
    if not pass1_specs:
        logger.info("[CompositeTargetDiscovery.stacked] pass1 yielded 0 specs; skipping pass 2.")
        return self

    rank_by_tiny = getattr(self, "tiny_rerank_scores_", None) or {}
    ranked = sorted(
        pass1_specs,
        key=lambda s: rank_by_tiny.get(s.name, float("inf")),
    )
    top_specs = ranked[: int(max_pass1_specs_to_stack)]

    from ..ensemble.feature_stacking import composite_oof_predictions
    from ..estimator import CompositeTargetEstimator

    # Lightweight Ridge inner so pass 2 cost stays small.
    from sklearn.linear_model import Ridge
    _train_idx_arr = np.asarray(train_idx)
    y_full = _extract_column_array(df, target_col)
    y_train = y_full[_train_idx_arr]
    oof_cols: dict[str, np.ndarray] = {}
    for spec in top_specs:
        def _factory(_s=spec):  # bind spec
            return CompositeTargetEstimator(
                base_estimator=Ridge(alpha=1e-3),
                transform_name=_s.transform_name,
                base_column=_s.base_column,
            )
        if hasattr(df, "iloc"):
            X_train = df.iloc[_train_idx_arr].reset_index(drop=True)
        else:
            try:
                import polars as _pl  # type: ignore
                if isinstance(df, _pl.DataFrame):
                    mask = np.zeros(df.height, dtype=bool)
                    mask[_train_idx_arr] = True
                    X_train = df.filter(_pl.Series(mask))
                else:
                    raise TypeError(type(df).__name__)
            except Exception:
                logger.warning(
                    "[CompositeTargetDiscovery.stacked] cannot slice df type=%s; skipping pass 2.",
                    type(df).__name__,
                )
                return self
        try:
            preds = composite_oof_predictions(
                _factory, X_train, y_train,
                n_splits=int(n_oof_folds),
                random_state=int(self.config.random_state),
            )
            oof_cols[f"_oof_{spec.name}"] = preds
        except Exception as _oof_err:
            logger.warning(
                "[CompositeTargetDiscovery.stacked] OOF for spec=%s failed: %s",
                spec.name, _oof_err,
            )

    if not oof_cols:
        return self

    # Augment df: write OOF cols on TRAIN rows only; NaN on val/test rows
    # (pass 2 only inspects train_idx so val/test fill is irrelevant).
    df_aug = df
    new_feature_cols = list(feature_cols)
    try:
        if hasattr(df_aug, "assign"):
            _full_cols: dict[str, np.ndarray] = {}
            _n_total = len(df_aug)
            for col_name, train_vec in oof_cols.items():
                _v = np.full(_n_total, np.nan, dtype=np.float64)
                _v[_train_idx_arr] = train_vec
                _full_cols[col_name] = _v
                new_feature_cols.append(col_name)
            df_aug = df_aug.assign(**_full_cols)
        else:
            import polars as _pl
            if isinstance(df_aug, _pl.DataFrame):
                _n_total = df_aug.height
                for col_name, train_vec in oof_cols.items():
                    _v = np.full(_n_total, np.nan, dtype=np.float64)
                    _v[_train_idx_arr] = train_vec
                    df_aug = df_aug.with_columns(_pl.Series(col_name, _v))
                    new_feature_cols.append(col_name)
    except Exception as _aug_err:
        logger.warning(
            "[CompositeTargetDiscovery.stacked] could not augment df: %s. Returning pass-1 only.",
            _aug_err,
        )
        return self

    try:
        self.fit(df_aug, target_col, new_feature_cols, train_idx, val_idx, test_idx)
    except Exception as _p2_err:
        logger.warning(
            "[CompositeTargetDiscovery.stacked] pass 2 failed: %s. Returning pass-1 specs.",
            _p2_err,
        )
        self.specs_ = pass1_specs
        return self
    pass2_specs = list(self.specs_) if self.specs_ else []
    existing_names = {s.name for s in pass1_specs}
    merged = pass1_specs + [s for s in pass2_specs if s.name not in existing_names]
    self.specs_ = merged
    logger.info(
        "[CompositeTargetDiscovery.stacked] pass1=%d specs, pass2=%d new specs, total=%d",
        len(pass1_specs),
        len([s for s in pass2_specs if s.name not in existing_names]),
        len(merged),
    )
    return self


def fit_stacked_on_residual(
    self,
    df: Any,
    target_col: str,
    feature_cols: Sequence[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray | None = None,
    test_idx: np.ndarray | None = None,
    *,
    n_oof_folds: int = 3,
    residual_aggregation: str = "mean",
):
    """Pack #4: residual-target stacked discovery (ALTERNATIVE to ``fit_stacked``).

    Where ``fit_stacked`` adds the OOF predictions of pass-1 specs as new FEATURES (treating them as bases for pass-2 transforms), this variant operates on a residual TARGET: pass-1 specs collectively predict ``pass1_pred``, then ``y_residual = y - pass1_pred`` becomes the new target for pass-2 discovery. Mathematically the more direct approach for residual-of-residual structure when feature stacking blocks at the discovery gate.

    Aggregation strategies:
    - ``"mean"``: ``pass1_pred = mean(oof_preds_per_spec)``. Robust to a single overfit spec.
    - ``"first"``: ``pass1_pred = oof_preds_of_best_spec`` (best by tiny CV-RMSE).

    Returns: ``self.specs_`` = pass1_specs UNION pass2_specs, with ``discovered_on_residual=True`` annotation in spec metadata for pass-2 entries. Suite-side training integration (fit pass-2 specs on the actual residual not raw y) is the follow-up step -- the current scaffolding returns the specs for inspection / experimentation.
    """
    from sklearn.linear_model import Ridge

    self.fit(df, target_col, feature_cols, train_idx, val_idx, test_idx)
    pass1_specs = list(self.specs_) if self.specs_ else []
    if not pass1_specs:
        logger.info(
            "[CompositeTargetDiscovery.stacked_on_residual] pass1 yielded 0 specs; "
            "skipping residual-target pass 2."
        )
        return self

    rank_by_tiny = getattr(self, "tiny_rerank_scores_", None) or {}
    ranked = sorted(
        pass1_specs,
        key=lambda s: rank_by_tiny.get(s.name, float("inf")),
    )

    from ..estimator import CompositeTargetEstimator
    from ..ensemble.feature_stacking import composite_oof_predictions

    _train_idx_arr = np.asarray(train_idx)
    y_full = _extract_column_array(df, target_col)
    y_train = y_full[_train_idx_arr]
    n_train = _train_idx_arr.size

    if hasattr(df, "iloc"):
        X_train = df.iloc[_train_idx_arr].reset_index(drop=True)
    else:
        try:
            import polars as _pl
            if isinstance(df, _pl.DataFrame):
                _mask = np.zeros(df.height, dtype=bool)
                _mask[_train_idx_arr] = True
                X_train = df.filter(_pl.Series(_mask))
            else:
                raise TypeError(type(df).__name__)
        except Exception:
            logger.warning(
                "[CompositeTargetDiscovery.stacked_on_residual] cannot slice df type=%s; "
                "returning pass-1 only.",
                type(df).__name__,
            )
            return self

    oof_preds_per_spec: list[np.ndarray] = []
    for spec in ranked[: max(1, len(ranked))]:
        def _factory(_s=spec):
            return CompositeTargetEstimator(
                base_estimator=Ridge(alpha=1e-3),
                transform_name=_s.transform_name,
                base_column=_s.base_column,
            )
        try:
            _oof = composite_oof_predictions(
                _factory, X_train, y_train,
                n_splits=int(n_oof_folds),
                random_state=int(self.config.random_state),
            )
            if np.all(np.isfinite(_oof)):
                oof_preds_per_spec.append(_oof)
        except Exception as _oof_err:
            logger.warning(
                "[CompositeTargetDiscovery.stacked_on_residual] OOF for "
                "spec=%s failed: %s",
                spec.name, _oof_err,
            )

    if not oof_preds_per_spec:
        logger.info(
            "[CompositeTargetDiscovery.stacked_on_residual] no usable OOF "
            "predictions; returning pass-1 specs only."
        )
        return self

    if residual_aggregation == "first":
        pass1_pred = oof_preds_per_spec[0]
    else:  # "mean" (default)
        pass1_pred = np.mean(np.stack(oof_preds_per_spec, axis=0), axis=0)

    y_residual_train = y_train.astype(np.float64) - pass1_pred.astype(np.float64)
    _residual_col = f"__y_residual__{target_col}"
    if hasattr(df, "assign"):
        _y_full_residual = np.full(len(df), np.nan, dtype=np.float64)
        _y_full_residual[_train_idx_arr] = y_residual_train
        df_with_resid = df.assign(**{_residual_col: _y_full_residual})
    else:
        try:
            import polars as _pl
            _y_full_residual = np.full(df.height, np.nan, dtype=np.float64)
            _y_full_residual[_train_idx_arr] = y_residual_train
            df_with_resid = df.with_columns(
                _pl.Series(_residual_col, _y_full_residual)
            )
        except Exception as _aug_err:
            logger.warning(
                "[CompositeTargetDiscovery.stacked_on_residual] could not "
                "build residual-target df: %s. Returning pass-1 only.",
                _aug_err,
            )
            return self

    try:
        self.fit(
            df_with_resid, _residual_col, feature_cols,
            train_idx, val_idx, test_idx,
        )
    except Exception as _p2_err:
        logger.warning(
            "[CompositeTargetDiscovery.stacked_on_residual] pass-2 fit on "
            "residual target failed: %s. Returning pass-1 only.",
            _p2_err,
        )
        self.specs_ = pass1_specs
        return self

    pass2_specs = list(self.specs_) if self.specs_ else []
    # Annotate pass-2 specs so suite-side training can route them to the
    # residual-aware fit path (future work; today the specs are stored
    # but not auto-trained on residual).
    for spec in pass2_specs:
        try:
            object.__setattr__(spec, "discovered_on_residual", True)
        except Exception:
            pass

    existing_names = {s.name for s in pass1_specs}
    merged = pass1_specs + [s for s in pass2_specs if s.name not in existing_names]
    self.specs_ = merged
    logger.info(
        "[CompositeTargetDiscovery.stacked_on_residual] pass1=%d specs, "
        "pass2=%d new residual-target specs, total=%d",
        len(pass1_specs),
        len([s for s in pass2_specs if s.name not in existing_names]),
        len(merged),
    )
    return self
