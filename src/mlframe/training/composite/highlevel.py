"""One-call convenience over the composite-target public API.

``discover_and_wrap(df, target_col, feature_cols, train_idx=None, ...)`` is a
thin orchestration helper for the common "I just want a fitted composite-target
predictor" path. It chains the existing public surface end-to-end:

1. **Config.** When no ``config`` is supplied it calls
   :func:`suggest_discovery_config` to derive a data-driven
   ``CompositeTargetDiscoveryConfig`` (cheap, sample-bounded -- never copies the
   frame). A caller-supplied ``config`` (object or ``dict``) is used verbatim.
2. **Discover.** Runs :meth:`CompositeTargetDiscovery.fit` on the train rows.
3. **Pick.** Selects the best discovered spec (``specs_[0]`` -- discovery returns
   them already ranked best-first by mi_gain / tiny-CV RMSE).
4. **Wrap + fit.** Builds a :class:`CompositeTargetEstimator` for that spec and
   fits it on the train rows (inner regressor trained on the T-scale composite
   target, inverse-side state stashed for y-scale predict).
5. **Calibrate (optional).** When ``calibrate_conformal=True`` and a held-out
   slice is available it calibrates a split-conformal radius on rows the inner
   never trained on (conformal validity rests on exchangeability -- in-sample
   rows would silently mis-cover).
6. **Report.** Renders a stakeholder-ready Markdown report via
   :func:`report_to_markdown`.

Returns a small :class:`DiscoverAndWrapResult` carrying
``{estimator, spec, config, report_markdown}`` (plus the live discovery object
and the calibration slice for callers that want to introspect).

Design notes / contract
-----------------------
- **Pure orchestration.** No new numeric path, no transform, no frame copy. The
  only frame reads are the targeted narrow column pulls the underlying
  estimator / discovery already do; ``df`` is never materialised whole or
  down-converted (polars stays polars). On a 100+ GB frame this helper adds
  zero copies of its own.
- **Train-only fit.** Every fitted parameter (transform alpha/beta, inner model,
  conformal radius) comes from the train rows or a disjoint held-out slice --
  never the test rows. ``train_idx`` defaults to "all rows" only for the small
  in-memory case; callers with a real holdout MUST pass ``train_idx`` so the
  honest-estimate rows stay untouched.
- **No-spec graceful path.** When discovery finds nothing (degenerate data,
  every candidate below ``eps_mi_gain``), ``estimator`` / ``spec`` are ``None``
  and ``report_markdown`` still renders the (empty) discovery report so the
  caller gets an explanation instead of a crash.

This module is intentionally side-effect-free at import (no model construction,
no env reads) so importing it is free.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .autoconfig import suggest_discovery_config
from .discovery import CompositeTargetDiscovery
from .estimator import CompositeTargetEstimator
from .provenance import report_to_markdown
from .spec import CompositeSpec

logger = logging.getLogger(__name__)


@dataclass
class DiscoverAndWrapResult:
    """Result of :func:`discover_and_wrap`.

    Attributes
    ----------
    estimator
        The fitted :class:`CompositeTargetEstimator` for the best discovered
        spec, ready for ``predict`` (and ``predict_interval`` when
        ``calibrate_conformal`` was requested). ``None`` when discovery found no
        spec.
    spec
        The chosen :class:`CompositeSpec`. ``None`` when discovery found no spec.
    config
        The ``CompositeTargetDiscoveryConfig`` actually used (suggested or
        caller-supplied), so the run is fully reproducible from the result.
    report_markdown
        Stakeholder-ready Markdown discovery report (always non-empty -- it
        renders the summary + tables even when no spec survived).
    discovery
        The live :class:`CompositeTargetDiscovery` instance (all specs, the
        rejection report, tiny-rerank scores) for callers that want to
        introspect beyond the single best spec.
    config_rationale
        ``{field: reason}`` map from the auto-config step; empty when the caller
        supplied an explicit ``config``.
    conformal_alpha
        The conformal level calibrated, or ``None`` when calibration was not
        requested / not possible (no held-out slice).
    """

    estimator: Optional[CompositeTargetEstimator]
    spec: Optional[CompositeSpec]
    config: Any
    report_markdown: str
    discovery: CompositeTargetDiscovery
    config_rationale: dict[str, str] = field(default_factory=dict)
    conformal_alpha: Optional[float] = None


def _default_inner_estimator() -> Any:
    """Construct a small, fast default inner regressor.

    Prefers LightGBM (already a project dep and the suite's default inner); falls
    back to sklearn's ``HistGradientBoostingRegressor`` so the helper still works
    on a stripped install. The prototype is unfitted -- the wrapper clones it at
    ``fit`` time, so the same prototype is reusable across calls.
    """
    try:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=200, num_leaves=31, learning_rate=0.05,
            min_child_samples=20, verbose=-1, n_jobs=1, random_state=0,
        )
    except Exception:  # pragma: no cover - lightgbm is a project dep
        from sklearn.ensemble import HistGradientBoostingRegressor

        # random_state pinned: HistGB early_stopping="auto" draws a random train/val split at n>10000.
        return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=0)


def _resolve_train_idx(df: Any, train_idx: Any) -> np.ndarray:
    """Normalise ``train_idx`` to an integer position array.

    ``None`` means "all rows" -- only sensible for the small in-memory case; a
    real holdout caller passes explicit positions. A boolean mask is converted
    to positions; a non-integer dtype is rejected loudly (the same contract
    discovery's ``fit`` enforces, surfaced one level earlier for a clearer
    error).
    """
    n = int(len(df))
    if train_idx is None:
        return np.arange(n, dtype=np.int64)
    arr = np.asarray(train_idx)
    if arr.dtype == bool:
        return np.flatnonzero(arr)
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError("train_idx must be integer positions or a boolean mask, got dtype " f"{arr.dtype!r}")
    return arr


def _select_rows(df: Any, idx: np.ndarray) -> Any:
    """Row-subset view over ``df`` (polars gather / pandas iloc).

    Never copies untouched columns and never materialises the full frame. The
    selected view feeds the wrapper's ``fit`` / ``predict`` (which themselves
    only pull the narrow base + feature columns they need).
    """
    try:
        import polars as pl

        if isinstance(df, pl.DataFrame):
            return df[idx]
    except ImportError:  # pragma: no cover - polars optional
        pass
    import pandas as pd

    if isinstance(df, pd.DataFrame):
        return df.iloc[idx]
    raise TypeError(f"discover_and_wrap: unsupported df type {type(df).__name__}")


def _extract_target(df: Any, target_col: str, idx: np.ndarray) -> np.ndarray:
    """1-D float target array for the given rows (no whole-frame pull)."""
    from .discovery.screening import _extract_column_array

    y = _extract_column_array(df, target_col, rows=idx)
    return np.asarray(y, dtype=np.float64).reshape(-1)


def _feature_cols_for_spec(
    feature_cols: Sequence[str], spec: CompositeSpec,
) -> list[str]:
    """Feature columns the wrapper's X must carry: the model features PLUS the
    spec's base column(s).

    The wrapper extracts the base from X by name then drops it before the inner
    estimator sees it, so the base column must be present in X even when it is
    not in the user's ``feature_cols`` list. Group columns (grouped transforms)
    are handled the same way but discovery does not currently emit grouped specs
    from the default path, so only base columns are added here.
    """
    cols = list(feature_cols)
    bases = [spec.base_column, *(getattr(spec, "extra_base_columns", ()) or ())]
    for b in bases:
        if b and b not in cols:
            cols.append(b)
    return cols


def discover_and_wrap(
    df: Any,
    target_col: str,
    feature_cols: Sequence[str],
    train_idx: Any = None,
    *,
    config: Any = None,
    base_estimator: Any = None,
    calibrate_conformal: bool = False,
    conformal_alpha: float = 0.1,
    holdout_idx: Any = None,
    time_ordering: Any = None,
    sample_weight: Any = None,
    config_overrides: Optional[dict[str, Any]] = None,
    **fit_kwargs: Any,
) -> DiscoverAndWrapResult:
    """Discover the best composite target for ``target_col`` and return a fitted
    estimator + provenance report in one call.

    Parameters
    ----------
    df
        Pandas or polars frame containing ``target_col`` + ``feature_cols``.
        Read-only; never copied or down-converted.
    target_col
        Regression target column.
    feature_cols
        Candidate feature columns. The base-column pool is drawn from this set
        (when ``config.base_candidates="auto"``); the chosen base is added back
        to the wrapper's X automatically.
    train_idx
        Row positions (or boolean mask) to fit on. ``None`` = all rows -- only
        for the small in-memory case; pass explicit positions when a holdout
        exists so the honest-estimate rows stay untouched.
    config
        A ``CompositeTargetDiscoveryConfig`` (or a ``dict`` of its fields). When
        ``None`` it is auto-suggested from ``df`` via
        :func:`suggest_discovery_config`.
    base_estimator
        Unfitted inner regressor prototype. ``None`` -> a small LightGBM
        (fallback ``HistGradientBoostingRegressor``).
    calibrate_conformal
        When True, calibrate a split-conformal radius at ``conformal_alpha`` on a
        held-out slice (``holdout_idx`` if given, else the discovery val/test
        rows when disjoint from train). Skipped with a log line when no disjoint
        held-out rows are available.
    conformal_alpha
        Conformal mis-coverage level (band has marginal coverage >= 1 - alpha).
    holdout_idx
        Explicit calibration rows (positions / mask). Must be disjoint from
        ``train_idx``. Falls back to the discovery val/test rows when omitted.
    time_ordering
        Optional per-row sortable key forwarded to discovery so the MI screen
        uses a forward-walk CV on temporal data.
    sample_weight
        Optional per-train-row weights forwarded to the wrapper's ``fit``.
    config_overrides
        Extra fields forwarded to :func:`suggest_discovery_config` (caller wins
        over the suggested values). Ignored when ``config`` is supplied.
    **fit_kwargs
        Forwarded to the inner estimator's ``fit``.

    Returns
    -------
    DiscoverAndWrapResult
        ``{estimator, spec, config, report_markdown, ...}``.
    """
    feature_cols = list(feature_cols)
    train_pos = _resolve_train_idx(df, train_idx)

    # ---- 1. config ---------------------------------------------------------
    rationale: dict[str, str] = {}
    if config is None:
        config, rationale = suggest_discovery_config(
            df, target_col, feature_cols, **(config_overrides or {}),
        )
    # else: caller-supplied config (object or dict) is honoured verbatim;
    # CompositeTargetDiscovery.__init__ coerces a dict into the config object.

    # ---- 2. discover -------------------------------------------------------
    discovery = CompositeTargetDiscovery(config)
    holdout_pos = None if holdout_idx is None else _resolve_train_idx(df, holdout_idx)
    # Pass the held-out rows as test_idx so discovery's leakage guard asserts
    # train/holdout disjointness for us (raises early on overlap).
    discovery.fit(
        df, target_col=target_col, feature_cols=feature_cols,
        train_idx=train_pos, test_idx=holdout_pos, time_ordering=time_ordering,
    )

    specs: list[CompositeSpec] = list(getattr(discovery, "specs_", []) or [])
    failures = list(getattr(discovery, "report_", []) or [])
    cfg_seed = getattr(config, "random_state", None)

    # ---- 6 (early): report renders even on the no-spec path ----------------
    report_markdown = report_to_markdown(
        target_col=target_col, specs=specs, failures=failures,
        random_state=cfg_seed,
        spec_metrics=_spec_metrics_from_discovery(discovery),
    )

    if not specs:
        logger.warning(
            "[discover_and_wrap] discovery found no spec for target=%r; " "returning estimator=None (see report_markdown for the rejection " "trail).",
            target_col,
        )
        return DiscoverAndWrapResult(
            estimator=None, spec=None, config=config,
            report_markdown=report_markdown, discovery=discovery,
            config_rationale=rationale, conformal_alpha=None,
        )

    # ---- 3. pick best ------------------------------------------------------
    spec = specs[0]

    # ---- 4. build + fit wrapper on train rows ------------------------------
    inner = base_estimator if base_estimator is not None else _default_inner_estimator()
    base_columns = tuple(c for c in (spec.base_column, *(getattr(spec, "extra_base_columns", ()) or ())) if c)
    estimator = CompositeTargetEstimator(
        base_estimator=inner,
        transform_name=spec.transform_name,
        base_column=spec.base_column,
        base_columns=base_columns or None,
    )

    wrapper_cols = _feature_cols_for_spec(feature_cols, spec)
    X_train = _select_cols(_select_rows(df, train_pos), wrapper_cols)
    y_train = _extract_target(df, target_col, train_pos)
    fit_extra: dict[str, Any] = dict(fit_kwargs)
    if sample_weight is not None:
        fit_extra["sample_weight"] = np.asarray(sample_weight).reshape(-1)
    estimator.fit(X_train, y_train, **fit_extra)

    # ---- 5. optional conformal calibration on held-out rows ----------------
    cal_alpha: Optional[float] = None
    if calibrate_conformal:
        cal_pos = _resolve_calibration_rows(
            holdout_pos, discovery, train_pos,
        )
        if cal_pos is not None and cal_pos.size:
            X_cal = _select_cols(_select_rows(df, cal_pos), wrapper_cols)
            y_cal = _extract_target(df, target_col, cal_pos)
            try:
                estimator.calibrate_conformal(X_cal, y_cal, alpha=conformal_alpha)
                cal_alpha = float(conformal_alpha)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "[discover_and_wrap] conformal calibration failed: %s; " "estimator returned without an interval.",
                    exc,
                )
        else:
            logger.info("[discover_and_wrap] calibrate_conformal requested but no " "disjoint held-out rows available; skipping calibration.")

    return DiscoverAndWrapResult(
        estimator=estimator, spec=spec, config=config,
        report_markdown=report_markdown, discovery=discovery,
        config_rationale=rationale, conformal_alpha=cal_alpha,
    )


def _select_cols(frame: Any, cols: Sequence[str]) -> Any:
    """Column-subset view (polars ``.select`` / pandas ``[cols]``).

    Operates on an already row-subset frame; never copies the untouched columns
    of the original full frame.
    """
    cols = list(cols)
    try:
        import polars as pl

        if isinstance(frame, pl.DataFrame):
            return frame.select(cols)
    except ImportError:  # pragma: no cover - polars optional
        pass
    return frame[cols]


def _resolve_calibration_rows(
    holdout_pos: Optional[np.ndarray],
    discovery: CompositeTargetDiscovery,
    train_pos: np.ndarray,
) -> Optional[np.ndarray]:
    """Pick the calibration rows for conformal: explicit ``holdout_pos`` if
    given, else the discovery val/test rows, filtered to be disjoint from train.

    Conformal validity requires the calibration rows to be exchangeable with
    test rows and NOT seen by the inner at fit -- so any train overlap is dropped
    (with the disjoint remainder kept rather than failing outright).
    """
    if holdout_pos is not None:
        cand = holdout_pos
    else:
        val = getattr(discovery, "val_idx_", None)
        test = getattr(discovery, "test_idx_", None)
        parts = [p for p in (val, test) if p is not None and len(p)]
        if not parts:
            return None
        cand = np.unique(np.concatenate([np.asarray(p) for p in parts]))
    train_set = np.asarray(train_pos)
    mask = ~np.isin(cand, train_set)
    return cand[mask]


def _spec_metrics_from_discovery(
    discovery: CompositeTargetDiscovery,
) -> dict[str, dict[str, Any]]:
    """Per-spec tiny-CV-RMSE side channel for the Markdown report.

    Pulls the y-scale tiny-CV RMSE (when the rerank ran) so the report's metrics
    matrix shows the actual prediction-objective number alongside mi_gain. Empty
    when ``screening="mi"`` (no rerank) -- the report degrades to dashes.
    """
    scores = getattr(discovery, "tiny_rerank_scores_", {}) or {}
    return {name: {"tiny_cv_rmse": float(v)} for name, v in scores.items()}
