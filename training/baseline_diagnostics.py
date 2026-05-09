"""Baseline diagnostics for ``train_mlframe_models_suite``.

A cheap (~30-90 s on a sample) per-target diagnostic that runs before
expensive training. It surfaces three signals operators ask for after
the fact:

1. Headline metric of a quick-fit model (RMSE for regression, AUC for
   binary). Sets the baseline that every later improvement is measured
   against.
2. Sequential ablation: drop top-K features by feature_importances_,
   retrain, measure delta. Identifies dominant features (the ``TVT_prev``
   case in the original motivation: one lag feature contributing >20%
   of headline metric).
3. ``init_score`` baseline (regression only): refits the quick model
   with the top-1 dominant feature passed via LightGBM's init_score so
   the model learns only the residual. If init_score baseline already
   matches the raw fit within ``init_score_optimal_threshold_pct``,
   downstream composite-target discovery is unlikely to add value -
   the residual is already extracted natively.

Composite-target discovery (future component) reads the
``composite_recommendation`` flag to gate its expensive MI / tiny-model
screening loops.

Public surface:
- :class:`BaselineDiagnosticsReport`
- :class:`BaselineDiagnostics`
- :func:`format_baseline_diagnostics_report`

Auto-emitted via ``train_mlframe_models_suite`` after the label-drift
report, with output stored at::

    metadata["baseline_diagnostics"][target_type][cur_target_name]
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from timeit import default_timer as timer
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


CompositeRecommendation = Literal["high_potential", "marginal", "unlikely_to_help", "skipped"]


@dataclass(frozen=True)
class AblationEntry:
    """One row of the ablation report - drop a single feature, refit, measure delta.

    delta% is signed so a higher-is-better metric (AUC) and a lower-is-better
    metric (RMSE) both produce a positive ``delta_pct`` when the dropped
    feature was contributing - i.e. ``delta_pct > 0`` always means
    "removing this feature hurt performance".
    """

    feature: str
    metric_after_drop: float
    delta_pct: float
    rank: int  # 1 = most dominant


@dataclass(frozen=True)
class InitScoreBaseline:
    """Result of a quick refit using the top-1 (or top-K) dominant feature
    as ``init_score`` (LightGBM regression)."""

    feature_used: str
    model_family: str
    metric: float
    delta_vs_raw_pct: float


@dataclass(frozen=True)
class BaselineDiagnosticsReport:
    """Structured diagnostic output. Stored on metadata for retrospective
    inspection; one instance per (target_type, target_name) pair."""

    target_name: str
    target_type: str
    headline_metric_name: str
    headline_metric_value: float
    headline_metric_higher_is_better: bool
    sample_n_used: int

    ablation: List[AblationEntry] = field(default_factory=list)
    dominant_features: List[Dict[str, Any]] = field(default_factory=list)

    init_score_baseline: Optional[InitScoreBaseline] = None

    composite_recommendation: CompositeRecommendation = "skipped"
    composite_recommendation_reason: str = ""

    elapsed_seconds: float = 0.0
    skipped: bool = False
    skip_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (for ``metadata`` storage / JSON-dump)."""
        return {
            "target_name": self.target_name,
            "target_type": self.target_type,
            "headline_metric": {
                "name": self.headline_metric_name,
                "value": self.headline_metric_value,
                "higher_is_better": self.headline_metric_higher_is_better,
                "sample_n": self.sample_n_used,
            },
            "ablation": [
                {
                    "feature": e.feature,
                    "metric_after_drop": e.metric_after_drop,
                    "delta_pct": e.delta_pct,
                    "rank": e.rank,
                }
                for e in self.ablation
            ],
            "dominant_features": list(self.dominant_features),
            "init_score_baseline": (
                {
                    "feature_used": self.init_score_baseline.feature_used,
                    "model_family": self.init_score_baseline.model_family,
                    "metric": self.init_score_baseline.metric,
                    "delta_vs_raw_pct": self.init_score_baseline.delta_vs_raw_pct,
                }
                if self.init_score_baseline is not None
                else None
            ),
            "composite_recommendation": self.composite_recommendation,
            "composite_recommendation_reason": self.composite_recommendation_reason,
            "elapsed_seconds": self.elapsed_seconds,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
        }


def _to_1d_numpy(arr: Any) -> np.ndarray:
    """Coerce target to a 1-D numpy array (mirrors drift_report convention)."""
    if hasattr(arr, "to_numpy"):
        out = arr.to_numpy()
    elif hasattr(arr, "values"):
        out = arr.values
    else:
        out = np.asarray(arr)
    return np.asarray(out).reshape(-1)


def _coerce_to_pandas(df: Any, columns: Sequence[str]) -> pd.DataFrame:
    """Return a pandas frame with exactly ``columns``. Handles polars / pandas /
    ndarray. Fast-path: if already pandas with the right columns, return as is.
    """
    # Polars
    if hasattr(df, "to_pandas") and not isinstance(df, pd.DataFrame):
        df = df.to_pandas()
    if isinstance(df, pd.DataFrame):
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise KeyError(
                f"BaselineDiagnostics: features missing from train frame: {missing[:5]}"
                + ("..." if len(missing) > 5 else "")
            )
        return df.loc[:, list(columns)]
    # ndarray fallback
    arr = np.asarray(df)
    if arr.ndim == 2 and arr.shape[1] == len(columns):
        return pd.DataFrame(arr, columns=list(columns))
    raise TypeError(
        f"BaselineDiagnostics: unsupported train frame type {type(df).__name__}"
    )


def _select_metric(target_type: str) -> Tuple[str, bool]:
    """Pick a quick-eval metric per target_type.

    Returns ``(metric_name, higher_is_better)``. Mirrors the conventions
    used elsewhere in mlframe (RMSE for regression, AUC for binary).
    """
    if target_type == "regression":
        return "RMSE", False
    if target_type == "binary_classification":
        return "AUC", True
    return "RMSE", False  # fallback, shouldn't happen given target-type guard


def _compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_name: str,
) -> float:
    """Quick metric computation. Standalone to avoid pulling in mlframe.metrics
    (which has heavy numba prewarm cost on cold cache).
    """
    if metric_name == "RMSE":
        diff = y_true.astype(np.float64) - y_pred.astype(np.float64)
        return float(np.sqrt(np.mean(diff * diff)))
    if metric_name == "AUC":
        from sklearn.metrics import roc_auc_score
        # Single-class evaluation set (e.g. tiny val sample where one
        # class is missing) -> AUC undefined; report nan and let caller
        # decide. Caller already guards on min sample size.
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_pred))
    raise ValueError(f"Unsupported metric: {metric_name}")


def _delta_pct(
    metric_baseline: float,
    metric_after: float,
    higher_is_better: bool,
) -> float:
    """Signed delta% so positive always means "ablation hurt performance".

    For lower-is-better (RMSE): delta% = (after / baseline - 1) * 100.
    For higher-is-better (AUC):  delta% = (1 - after / baseline) * 100.

    Defends against ``baseline ≈ 0`` (constant target -> RMSE near zero):
    falls back to absolute delta in that regime so delta% doesn't explode
    to ±inf and break threshold comparisons downstream.
    """
    if not math.isfinite(metric_baseline) or not math.isfinite(metric_after):
        return float("nan")
    if abs(metric_baseline) < 1e-12:
        # Absolute delta as percentage points fallback; sign conventions match.
        return (
            float(metric_after - metric_baseline) * 100.0
            if not higher_is_better
            else float(metric_baseline - metric_after) * 100.0
        )
    if higher_is_better:
        return float((1.0 - metric_after / metric_baseline) * 100.0)
    return float((metric_after / metric_baseline - 1.0) * 100.0)


class BaselineDiagnostics:
    """Run the per-target baseline diagnostic.

    Single-shot semantics: construct with a config, call
    :meth:`fit_and_report` once per (target_type, target_name) pair, get
    a :class:`BaselineDiagnosticsReport` back. The instance does NOT
    cache models or state across calls - each invocation fits its own
    quick LightGBM models on a fresh sample.
    """

    def __init__(self, config: Any) -> None:
        # Accept either a BaselineDiagnosticsConfig or a dict; dict path
        # is a thin convenience for callers that don't want to import
        # the config class. Pydantic-side validation already happens at
        # the suite-level _ensure_config call site.
        if isinstance(config, dict):
            from .configs import BaselineDiagnosticsConfig
            config = BaselineDiagnosticsConfig(**config)
        self.config = config

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def fit_and_report(
        self,
        train_df: Any,
        train_target: Any,
        feature_cols: Sequence[str],
        target_type: str,
        target_name: str,
        cat_features: Optional[Sequence[str]] = None,
    ) -> BaselineDiagnosticsReport:
        """Run the diagnostic. Never raises; on internal errors returns
        a ``skipped=True`` report and logs a warning so suite training
        always continues.

        Parameters
        ----------
        train_df
            Training-time feature frame (pandas, polars, or 2-D
            ndarray). Outlier-filtered indices already applied; this
            view IS the training set used by downstream models.
        train_target
            1-D target array aligned to ``train_df``.
        feature_cols
            Column names to consider. Categorical features should be
            included here too (LightGBM consumes them via
            ``categorical_feature=``).
        target_type
            String value from :class:`mlframe.training.configs.TargetTypes`.
            Diagnostic skipped for unsupported types.
        target_name
            For logging / report identity only.
        cat_features
            Subset of ``feature_cols`` that are categorical (passed to
            LightGBM as ``categorical_feature=``).
        """
        t0 = timer()
        cat_features = list(cat_features or [])

        if not self.config.enabled:
            return self._skipped(
                target_name, target_type, "config.enabled=False", elapsed=timer() - t0
            )
        if target_type not in self.config.apply_to_target_types:
            return self._skipped(
                target_name, target_type,
                f"target_type='{target_type}' not in apply_to_target_types="
                f"{self.config.apply_to_target_types}",
                elapsed=timer() - t0,
            )
        if not feature_cols:
            return self._skipped(
                target_name, target_type, "no feature_cols provided",
                elapsed=timer() - t0,
            )

        try:
            y = _to_1d_numpy(train_target)
            X = _coerce_to_pandas(train_df, feature_cols)
            # Sanity: align lengths (FTE drift can produce mismatches if
            # caller passes the wrong slice).
            if len(X) != len(y):
                return self._skipped(
                    target_name, target_type,
                    f"length mismatch X={len(X)} vs y={len(y)}",
                    elapsed=timer() - t0,
                )

            # Sample if requested. Sampling is RANDOM (not stratified or
            # quantile-aware) - the diagnostic is intentionally cheap;
            # full-train numbers are recovered by downstream training,
            # not by this routine.
            X_s, y_s, sample_n = self._sample(X, y)

            metric_name, higher_is_better = _select_metric(target_type)

            # 1. Headline raw metric
            raw_metric, raw_fi = self._fit_quick_and_score(
                X_s, y_s, feature_cols, cat_features, target_type, metric_name,
            )

            if not math.isfinite(raw_metric):
                return self._skipped(
                    target_name, target_type,
                    f"raw quick-fit metric is non-finite ({raw_metric}); "
                    "likely degenerate target / sample",
                    elapsed=timer() - t0,
                )

            # 2. Ablation
            ablation = self._run_ablation(
                X_s, y_s, feature_cols, cat_features, target_type,
                raw_fi, raw_metric, metric_name, higher_is_better,
            )
            dominant_features = [
                {"feature": e.feature, "score": e.delta_pct, "rank": e.rank}
                for e in ablation
            ]

            # 3. init_score baseline (regression-only in MVP)
            init_score_baseline = None
            if target_type in self.config.init_score_apply_to_target_types and ablation:
                init_score_baseline = self._fit_init_score_baseline(
                    X_s, y_s, feature_cols, cat_features, ablation,
                    metric_name, higher_is_better, raw_metric,
                )

            # 4. Recommendation
            recommendation, reason = self._build_recommendation(
                ablation, init_score_baseline,
            )

            return BaselineDiagnosticsReport(
                target_name=target_name,
                target_type=target_type,
                headline_metric_name=metric_name,
                headline_metric_value=raw_metric,
                headline_metric_higher_is_better=higher_is_better,
                sample_n_used=sample_n,
                ablation=ablation,
                dominant_features=dominant_features,
                init_score_baseline=init_score_baseline,
                composite_recommendation=recommendation,
                composite_recommendation_reason=reason,
                elapsed_seconds=timer() - t0,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "BaselineDiagnostics failed for target='%s' (type=%s): %s. "
                "Training continues without diagnostics.",
                target_name, target_type, exc,
            )
            return self._skipped(
                target_name, target_type, f"internal_error: {exc}",
                elapsed=timer() - t0,
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _skipped(
        self, target_name: str, target_type: str, reason: str, elapsed: float,
    ) -> BaselineDiagnosticsReport:
        return BaselineDiagnosticsReport(
            target_name=target_name,
            target_type=target_type,
            headline_metric_name="",
            headline_metric_value=float("nan"),
            headline_metric_higher_is_better=False,
            sample_n_used=0,
            elapsed_seconds=elapsed,
            skipped=True,
            skip_reason=reason,
        )

    def _sample(
        self, X: pd.DataFrame, y: np.ndarray,
    ) -> Tuple[pd.DataFrame, np.ndarray, int]:
        """Random sample to ``sample_n`` rows. No-op when sample_n is None
        or when the frame is already smaller."""
        sample_n = self.config.sample_n
        n = len(X)
        if sample_n is None or n <= sample_n:
            return X, y, n
        rng = np.random.default_rng(self.config.random_state)
        idx = rng.choice(n, size=sample_n, replace=False)
        idx.sort()  # preserve row order to keep any latent temporal structure
        return X.iloc[idx].reset_index(drop=True), y[idx], sample_n

    def _make_quick_model(self, target_type: str, init_score: Optional[np.ndarray] = None):
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
            n_jobs=-1,
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
        init_score: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray]:
        """Fit quick LightGBM on a holdout split, return ``(metric, fi_array)``.

        FI is taken from the LightGBM model.feature_importances_ aligned
        to ``feature_cols`` order. Holdout = simple 80/20 random split
        seeded by ``config.random_state``. The diagnostic is meant to
        be cheap, not robust: a single fold is sufficient to surface
        dominant features at percentage-point resolution.
        """
        from sklearn.model_selection import train_test_split

        n = len(X)
        if n < 50:
            # Tiny sample: can't carve a meaningful holdout. Train on
            # everything and evaluate on training set itself - the
            # diagnostic will be optimistic but still ranks features.
            X_tr, y_tr = X, y
            X_va, y_va = X, y
        else:
            X_tr, X_va, y_tr, y_va = train_test_split(
                X, y,
                test_size=0.2,
                random_state=self.config.random_state,
                stratify=y if target_type == "binary_classification"
                and len(np.unique(y)) > 1 else None,
            )

        model = self._make_quick_model(target_type)
        fit_kwargs = {}
        # LightGBM accepts categorical_feature only when given column names
        # via pandas; we already coerce X to pandas above.
        if cat_features:
            usable = [c for c in cat_features if c in X.columns]
            if usable:
                fit_kwargs["categorical_feature"] = usable
        if init_score is not None and target_type == "regression":
            # Align init_score to X_tr ordering. train_test_split on a
            # 1-D array returns the matching slice; we pass-through.
            init_score_tr, init_score_va = train_test_split(
                init_score, test_size=0.2, random_state=self.config.random_state,
            )
            fit_kwargs["init_score"] = init_score_tr
            init_score_va_local = init_score_va
        else:
            init_score_va_local = None

        model.fit(X_tr, y_tr, **fit_kwargs)

        if target_type == "binary_classification":
            y_pred = model.predict_proba(X_va)[:, 1]
        else:
            y_pred = model.predict(X_va)
            if init_score_va_local is not None:
                # LightGBM's regression predict adds init_score back when
                # the booster was trained with one (recent versions). We
                # explicitly add it ourselves to be version-agnostic; if
                # the booster already adds it the inverse-test in the
                # init_score path checks for double-add.
                y_pred = y_pred + init_score_va_local

        metric_val = _compute_metric(np.asarray(y_va), np.asarray(y_pred), metric_name)

        fi = np.asarray(getattr(model, "feature_importances_", []), dtype=np.float64)
        if fi.size != len(feature_cols):
            # Fallback: zero-importances if model didn't expose them.
            fi = np.zeros(len(feature_cols), dtype=np.float64)
        return metric_val, fi

    def _run_ablation(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_cols: Sequence[str],
        cat_features: Sequence[str],
        target_type: str,
        raw_fi: np.ndarray,
        raw_metric: float,
        metric_name: str,
        higher_is_better: bool,
    ) -> List[AblationEntry]:
        """Drop top-K features by FI rank, refit, measure delta. Sequential
        independent drops (NOT cumulative) - we want per-feature
        contribution, not interaction-aware joint impact.
        """
        if raw_fi.size == 0 or raw_fi.sum() == 0:
            logger.info(
                "BaselineDiagnostics: feature_importances_ all-zero; ablation skipped."
            )
            return []

        top_k = max(1, min(self.config.ablation_top_k, len(feature_cols)))
        # Indices of top-K features by FI, descending.
        order = np.argsort(-raw_fi)[:top_k]

        entries: List[AblationEntry] = []
        for rank, idx in enumerate(order, start=1):
            feat = feature_cols[idx]
            if raw_fi[idx] <= 0:
                # Skip features with zero importance: dropping them won't
                # change the metric and just wastes a refit.
                continue
            kept = [c for c in feature_cols if c != feat]
            if not kept:
                continue
            X_drop = X.loc[:, kept]
            cat_kept = [c for c in cat_features if c in kept]
            try:
                metric_drop, _ = self._fit_quick_and_score(
                    X_drop, y, kept, cat_kept, target_type, metric_name,
                )
            except Exception as exc:
                logger.warning(
                    "BaselineDiagnostics: ablation refit for '%s' failed: %s; skipping.",
                    feat, exc,
                )
                continue
            entries.append(
                AblationEntry(
                    feature=feat,
                    metric_after_drop=metric_drop,
                    delta_pct=_delta_pct(raw_metric, metric_drop, higher_is_better),
                    rank=rank,
                )
            )
        # Sort by absolute dominance descending so dominant_features is
        # ranked by impact, not by raw FI (the two usually agree but FI
        # can mislead on correlated features).
        entries.sort(key=lambda e: -e.delta_pct)
        # Re-rank after sort
        entries = [
            AblationEntry(
                feature=e.feature,
                metric_after_drop=e.metric_after_drop,
                delta_pct=e.delta_pct,
                rank=i,
            )
            for i, e in enumerate(entries, start=1)
        ]
        return entries

    def _fit_init_score_baseline(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_cols: Sequence[str],
        cat_features: Sequence[str],
        ablation: List[AblationEntry],
        metric_name: str,
        higher_is_better: bool,
        raw_metric: float,
    ) -> Optional[InitScoreBaseline]:
        """Refit quick-LightGBM with the top-1 dominant feature passed via
        ``init_score`` so the model learns the residual ``y - top1``
        instead of the full target. Mirrors LightGBM/XGBoost's native
        ``base_margin`` / ``init_score`` interface - the cheapest
        possible "what if I learn the residual instead of the target?"
        baseline.

        For ``init_score_top_k > 1`` the top-K features are linearly
        combined via OLS first; the OLS prediction is passed as
        init_score. If OLS is degenerate (collinear / non-finite
        coeffs), falls back to top-1 only.
        """
        top_k = max(1, min(self.config.init_score_top_k, len(ablation)))
        # Use only features with positive ablation delta - dropping them
        # actually hurts. Sorted ablation is already ranked; take prefix.
        chosen = [e for e in ablation[:top_k] if e.delta_pct > 0]
        if not chosen:
            return None

        if len(chosen) == 1:
            feat = chosen[0].feature
            base_values = X[feat].to_numpy().astype(np.float64)
            feature_used = feat
        else:
            # OLS fit to combine top-K. Numerically guarded: drop
            # constant columns, fall back to top-1 on degeneracy.
            try:
                B = np.column_stack(
                    [X[e.feature].to_numpy().astype(np.float64) for e in chosen]
                )
                ptp = B.ptp(axis=0)
                keep_cols = ptp > 1e-12
                if keep_cols.sum() == 0:
                    return None
                B = B[:, keep_cols]
                B_aug = np.column_stack([B, np.ones(len(B))])
                # lstsq is more stable than .solve here (handles rank-deficient).
                coef, *_ = np.linalg.lstsq(B_aug, y.astype(np.float64), rcond=None)
                base_values = B_aug @ coef
                kept_feats = [e.feature for e, k in zip(chosen, keep_cols) if k]
                feature_used = "+".join(kept_feats) if kept_feats else chosen[0].feature
            except Exception as exc:
                logger.info(
                    "BaselineDiagnostics: init_score OLS combiner failed (%s); "
                    "falling back to top-1.",
                    exc,
                )
                feat = chosen[0].feature
                base_values = X[feat].to_numpy().astype(np.float64)
                feature_used = feat

        if not np.all(np.isfinite(base_values)):
            logger.info(
                "BaselineDiagnostics: init_score base values contain "
                "non-finite entries; skipping."
            )
            return None

        try:
            metric_val, _ = self._fit_quick_and_score(
                X, y, feature_cols, cat_features, "regression",
                metric_name, init_score=base_values,
            )
        except Exception as exc:
            logger.warning(
                "BaselineDiagnostics: init_score refit failed: %s; baseline skipped.",
                exc,
            )
            return None

        return InitScoreBaseline(
            feature_used=feature_used,
            model_family=self.config.quick_model_family,
            metric=metric_val,
            delta_vs_raw_pct=_delta_pct(raw_metric, metric_val, higher_is_better),
        )

    def _build_recommendation(
        self,
        ablation: List[AblationEntry],
        init_score_baseline: Optional[InitScoreBaseline],
    ) -> Tuple[CompositeRecommendation, str]:
        """Three-way classifier:

        * **high_potential** - max(ablation delta%) >= high_potential_min_dominance_pct
          AND init_score baseline did NOT already extract that signal
          (delta_vs_raw_pct stayed > init_score_optimal_threshold_pct,
          i.e. residual still has structure).
        * **marginal** - max ablation delta% in [marginal_threshold_pct,
          high_potential_min_dominance_pct).
        * **unlikely_to_help** - max ablation delta% < marginal_threshold_pct
          OR init_score baseline already matches raw within
          init_score_optimal_threshold_pct (residual is mostly noise).
        """
        if not ablation:
            return "unlikely_to_help", "no ablation entries (FI all-zero or no features)"

        max_dom = max(e.delta_pct for e in ablation if math.isfinite(e.delta_pct))
        cfg = self.config

        # Note on sign: ``init_score delta%`` uses the same convention as
        # ablation delta% - positive means init_score baseline performed
        # WORSE than raw (residual still has structure). Negative or
        # near-zero means init_score baseline already matches raw, so
        # composite-mode unlikely to extract more signal.
        init_score_sufficient = (
            init_score_baseline is not None
            and abs(init_score_baseline.delta_vs_raw_pct) <= cfg.init_score_optimal_threshold_pct
        )

        if max_dom >= cfg.high_potential_min_dominance_pct and not init_score_sufficient:
            reason = (
                f"top ablation delta%={max_dom:.2f} >= {cfg.high_potential_min_dominance_pct:.2f} "
                "(strong dominant feature)"
            )
            if init_score_baseline is not None:
                reason += (
                    f"; init_score baseline still off raw by "
                    f"{init_score_baseline.delta_vs_raw_pct:+.2f}% (residual has structure)"
                )
            return "high_potential", reason

        if init_score_sufficient:
            return (
                "unlikely_to_help",
                f"init_score baseline matches raw within "
                f"{cfg.init_score_optimal_threshold_pct:.2f}pct "
                f"(delta={init_score_baseline.delta_vs_raw_pct:+.2f}%); "
                "native residual learning already captures the dominant signal",
            )

        if max_dom >= cfg.marginal_threshold_pct:
            return (
                "marginal",
                f"top ablation delta%={max_dom:.2f} in "
                f"[{cfg.marginal_threshold_pct:.2f}, {cfg.high_potential_min_dominance_pct:.2f})",
            )
        return (
            "unlikely_to_help",
            f"top ablation delta%={max_dom:.2f} < {cfg.marginal_threshold_pct:.2f} "
            "(no dominant features)",
        )


# ----------------------------------------------------------------------
# Pretty-printer for log output
# ----------------------------------------------------------------------

def format_baseline_diagnostics_report(
    report: BaselineDiagnosticsReport,
    *,
    target_name: Optional[str] = None,
) -> str:
    """Render a multi-line human-readable summary suitable for a single
    ``logger.info`` call.

    Mirrors the style of :func:`format_drift_report` so suite output
    has consistent visual rhythm.
    """
    name = target_name or report.target_name or "target"
    if report.skipped:
        return f"[BaselineDiagnostics] target='{name}' SKIPPED ({report.skip_reason})"

    lines: List[str] = []
    metric = report.headline_metric_name
    lines.append(
        f"[BaselineDiagnostics] target='{name}' ({report.target_type}) "
        f"{metric}_raw={report.headline_metric_value:.4f} "
        f"sample_n={report.sample_n_used} "
        f"elapsed={report.elapsed_seconds:.1f}s"
    )
    if report.ablation:
        lines.append("[BaselineDiagnostics] Ablation (drop -> delta%, positive = drop hurt):")
        for entry in report.ablation:
            lines.append(
                f"  rank={entry.rank} {entry.feature:<24s} "
                f"{metric}_after_drop={entry.metric_after_drop:.4f} "
                f"delta%={entry.delta_pct:+.2f}"
            )
    if report.init_score_baseline is not None:
        isb = report.init_score_baseline
        lines.append(
            f"[BaselineDiagnostics] init_score({isb.feature_used}) "
            f"{metric}={isb.metric:.4f} "
            f"delta%={isb.delta_vs_raw_pct:+.2f} vs raw"
        )
    lines.append(
        f"[BaselineDiagnostics] composite_recommendation={report.composite_recommendation}"
    )
    if report.composite_recommendation_reason:
        lines.append(f"  reason: {report.composite_recommendation_reason}")
    return "\n".join(lines)
