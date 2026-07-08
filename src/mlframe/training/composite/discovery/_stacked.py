"""Two-pass stacked variants of :meth:`CompositeTargetDiscovery.fit`.

Bound onto the class via method-rebinding at the parent module's bottom;
``self`` stays the first arg so identity and behaviour are preserved.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from .screening import _extract_column_array

logger = logging.getLogger(__name__)


# Prefix marking the ephemeral OOF feature columns appended in pass 1. Pass-2
# specs that adopt one of these as their ``base_column`` are NOT reconstructable
# by the suite at integration time (the column lives only in the local
# ``df_aug`` and the suite's ``_build_full_column_from_splits`` silently returns
# an all-NaN column for any name absent from the per-split frames -> a composite
# trained on garbage).
_OOF_FEATURE_PREFIX = "_oof_"


def _spec_base_columns(spec: Any) -> tuple[str, ...] | None:
    """Full base-column tuple for a spec, or ``None`` for a single-base spec.

    A multi-base spec's OOF estimator must reconstruct the SAME multi-column base matrix the
    spec represents; dropping ``extra_base_columns`` silently substitutes a different
    single-base transform, so its OOF column no longer matches the spec it feeds.
    """
    extra = tuple(getattr(spec, "extra_base_columns", ()) or ())
    if not extra:
        return None
    return (spec.base_column, *extra)


def _warn_unrebuildable_oof_specs(pass2_specs, existing_names):
    """Warn for any *new* pass-2 spec whose base is an ephemeral ``_oof_*`` column.

    Such a spec references a feature that exists only in the in-memory augmented
    frame; the suite cannot rebuild it from the persisted split frames, so the
    composite would train on an all-NaN base. We surface this loudly rather than
    let it degrade silently (the full fix is to persist the OOF recipe and
    rebuild it before training -- architectural, suite-side).

    Returns the list of unrebuildable spec names (for callers/tests).
    """
    bad: list[str] = []
    for _s in pass2_specs:
        if _s.name in existing_names:
            continue
        _bases = (getattr(_s, "base_column", "") or "",) + tuple(getattr(_s, "extra_base_columns", ()) or ())
        if any(str(_b).startswith(_OOF_FEATURE_PREFIX) for _b in _bases):
            bad.append(_s.name)
    if bad:
        logger.warning(
            "[CompositeTargetDiscovery.stacked] %d pass-2 spec(s) reference "
            "ephemeral OOF feature column(s) (prefix %r) that exist only in the "
            "augmented training frame: %s. The suite cannot rebuild these bases "
            "from the persisted split frames (they become all-NaN -> the "
            "composite would train on garbage). These specs are kept in specs_ "
            "for inspection but should NOT be trained as-is; persist the OOF "
            "recipe and rebuild the column before training to use them (A6).",
            len(bad), _OOF_FEATURE_PREFIX, bad,
        )
    return bad


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
    time_aware: bool = False,
    cv_splitter: Any = None,
):
    """2-pass stacked composite discovery.

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

    ``time_aware`` / ``cv_splitter``: control the fold scheme of the
    OOF-prediction step. By default the OOF uses a shuffled K-fold, which
    leaks future->past on temporal data (a pass-2 base built from
    future-contaminated OOF predictions). Pass ``time_aware=True`` (or an
    explicit ``cv_splitter``) on temporally-ordered data so the OOF folds
    respect time order. The suite forwards ``time_aware=True`` automatically
    when a ``time_column`` is configured. Defaults preserve the historical
    shuffled-K-fold numerics for non-temporal callers.
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
    # Slice the train frame ONCE -- X_train is invariant across specs, so the
    # prior per-spec rebuild materialised the same train slice N times (the
    # sibling fit_stacked_on_residual already hoists this).
    if hasattr(df, "iloc"):
        X_train = df.iloc[_train_idx_arr].reset_index(drop=True)
    else:
        try:
            import polars as _pl
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
    oof_cols: dict[str, np.ndarray] = {}
    for spec in top_specs:
        def _factory(_s=spec):  # bind spec
            return CompositeTargetEstimator(
                base_estimator=Ridge(alpha=1e-3),
                transform_name=_s.transform_name,
                base_column=_s.base_column,
                base_columns=_spec_base_columns(_s),
            )
        try:
            preds = composite_oof_predictions(
                _factory, X_train, y_train,
                n_splits=int(n_oof_folds),
                random_state=int(self.config.random_state),
                time_aware=bool(time_aware),
                cv_splitter=cv_splitter,
            )
            oof_cols[f"{_OOF_FEATURE_PREFIX}{spec.name}"] = preds
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
    # Pass-2 specs that adopted an ephemeral ``_oof_*`` column as their base
    # cannot be rebuilt by the suite -> warn loudly so they aren't trained on an
    # all-NaN base silently.
    _warn_unrebuildable_oof_specs(pass2_specs, existing_names)
    _new_specs = [s for s in pass2_specs if s.name not in existing_names]
    merged = pass1_specs + _new_specs
    self.specs_ = merged
    logger.info(
        "[CompositeTargetDiscovery.stacked] pass1=%d specs, pass2=%d new specs, total=%d",
        len(pass1_specs),
        len(_new_specs),
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
    max_pass1_specs_to_aggregate: int = 3,
    time_aware: bool = False,
    cv_splitter: Any = None,
):
    """Residual-target stacked discovery (ALTERNATIVE to ``fit_stacked``).

    Where ``fit_stacked`` adds the OOF predictions of pass-1 specs as new FEATURES (treating them as bases for pass-2 transforms), this variant operates on a residual TARGET: pass-1 specs collectively predict ``pass1_pred``, then ``y_residual = y - pass1_pred`` becomes the new target for pass-2 discovery. Mathematically the more direct approach for residual-of-residual structure when feature stacking blocks at the discovery gate.

    Aggregation strategies:
    - ``"mean"``: ``pass1_pred = mean(oof_preds_per_spec)``. Robust to a single overfit spec.
    - ``"first"``: ``pass1_pred = oof_preds_of_best_spec`` (best by tiny CV-RMSE).

    ``max_pass1_specs_to_aggregate``: cap on how many top-ranked pass-1
    specs (best tiny CV-RMSE first) contribute their OOF predictions to
    ``pass1_pred``. Previously this path aggregated EVERY pass-1 spec
    (``ranked[:max(1,len(ranked))]`` is a no-op slice), so weak/overfit tail
    specs polluted the ``mean`` aggregate and the leftover residual. The cap
    now matches the feature-stack sibling's behaviour (default 3). For
    ``residual_aggregation="first"`` only the single best spec is used, so the
    cap is immaterial there. Set to ``0`` or a negative value to disable the
    cap (aggregate all pass-1 specs, the historical behaviour).

    ``time_aware`` / ``cv_splitter``: control the OOF fold scheme exactly
    as in :func:`fit_stacked` -- default shuffled K-fold (historical numerics);
    pass ``time_aware=True`` on temporal data to avoid future->past leakage in
    the OOF predictions that define the residual target. The suite forwards
    ``time_aware=True`` automatically when a ``time_column`` is configured.

    Returns: ``self.specs_`` = pass1_specs UNION pass2_specs, with ``discovered_on_residual=True`` annotation in spec metadata for pass-2 entries. Suite-side training integration (fit pass-2 specs on the actual residual not raw y) is the follow-up step -- the current scaffolding returns the specs for inspection / experimentation. Because the suite does NOT yet route these specs through a residual-aware training path, a WARNING is emitted listing them: their ``fitted_params`` were fit against the residual but training would apply them against raw ``y``, so they must be treated as inspection-only until the residual-aware path lands.
    """
    from sklearn.linear_model import Ridge

    self.fit(df, target_col, feature_cols, train_idx, val_idx, test_idx)
    pass1_specs = list(self.specs_) if self.specs_ else []
    if not pass1_specs:
        logger.info("[CompositeTargetDiscovery.stacked_on_residual] pass1 yielded 0 specs; " "skipping residual-target pass 2.")
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
                "[CompositeTargetDiscovery.stacked_on_residual] cannot slice df type=%s; " "returning pass-1 only.",
                type(df).__name__,
            )
            return self

    # Cap the number of pass-1 specs whose OOF predictions feed the aggregate.
    # The old ``ranked[:max(1,len(ranked))]`` was a no-op slice (selected ALL
    # specs), letting weak tail specs pollute the ``mean`` aggregate. ``<=0``
    # disables the cap (historical aggregate-all behaviour).
    _cap = int(max_pass1_specs_to_aggregate)
    if _cap <= 0:
        _ranked_capped = list(ranked)
    else:
        _ranked_capped = ranked[: max(1, _cap)]

    oof_preds_per_spec: list[np.ndarray] = []
    for spec in _ranked_capped:
        def _factory(_s=spec):
            return CompositeTargetEstimator(
                base_estimator=Ridge(alpha=1e-3),
                transform_name=_s.transform_name,
                base_column=_s.base_column,
                base_columns=_spec_base_columns(_s),
            )
        try:
            _oof = composite_oof_predictions(
                _factory, X_train, y_train,
                n_splits=int(n_oof_folds),
                random_state=int(self.config.random_state),
                time_aware=bool(time_aware),
                cv_splitter=cv_splitter,
            )
            if np.all(np.isfinite(_oof)):
                oof_preds_per_spec.append(_oof)
        except Exception as _oof_err:
            logger.warning(
                "[CompositeTargetDiscovery.stacked_on_residual] OOF for " "spec=%s failed: %s",
                spec.name,
                _oof_err,
            )

    if not oof_preds_per_spec:
        logger.info("[CompositeTargetDiscovery.stacked_on_residual] no usable OOF " "predictions; returning pass-1 specs only.")
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
            df_with_resid = df.with_columns(_pl.Series(_residual_col, _y_full_residual))
        except Exception as _aug_err:
            logger.warning(
                "[CompositeTargetDiscovery.stacked_on_residual] could not " "build residual-target df: %s. Returning pass-1 only.",
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
            "[CompositeTargetDiscovery.stacked_on_residual] pass-2 fit on " "residual target failed: %s. Returning pass-1 only.",
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
        except Exception as e:
            logger.debug("swallowed exception in _stacked.py: %s", e)
            pass

    existing_names = {s.name for s in pass1_specs}
    _new_residual_specs = [s for s in pass2_specs if s.name not in existing_names]
    # These pass-2 specs carry ``fitted_params`` fit against the RESIDUAL
    # target, but the suite has no residual-aware training route yet -- it would
    # apply them against raw ``y``, mis-using the residual-fitted params. The
    # ``discovered_on_residual`` flag is currently read only by tests, not by
    # suite routing, so surface the hazard explicitly rather than let it merge
    # silently into ``specs_``. (Full fix: a residual-aware training path or not
    # merging these into ``specs_`` at all -- cross-file, suite-side.)
    if _new_residual_specs:
        logger.warning(
            "[CompositeTargetDiscovery.stacked_on_residual] %d pass-2 spec(s) "
            "were discovered on the RESIDUAL target and carry residual-fitted "
            "params: %s. They are merged into specs_ and flagged "
            "discovered_on_residual=True for inspection, but the suite does NOT "
            "yet route them through a residual-aware training path -- training "
            "would apply their params against raw y. Treat as inspection-only "
            "until the residual-aware training path lands (A7).",
            len(_new_residual_specs), [s.name for s in _new_residual_specs],
        )
    merged = pass1_specs + _new_residual_specs
    self.specs_ = merged
    logger.info(
        "[CompositeTargetDiscovery.stacked_on_residual] pass1=%d specs, " "pass2=%d new residual-target specs, total=%d",
        len(pass1_specs),
        len(_new_residual_specs),
        len(merged),
    )
    return self
