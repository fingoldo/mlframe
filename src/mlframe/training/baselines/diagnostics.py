"""Baseline diagnostics for ``train_mlframe_models_suite``.

A cheap (~30-90 s on a sample) per-target diagnostic that runs before
expensive training. It surfaces three signals operators ask for after
the fact:

1. Headline metric of a quick-fit model (RMSE for regression, AUC for
   binary). Sets the baseline that every later improvement is measured
   against.
2. Sequential ablation: drop top-K features by feature_importances_,
   retrain, measure delta. Identifies dominant features (the lag-feature
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
from typing import Any, Literal, Sequence

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

    ablation: list[AblationEntry] = field(default_factory=list)
    dominant_features: list[dict[str, Any]] = field(default_factory=list)

    init_score_baseline: InitScoreBaseline | None = None

    composite_recommendation: CompositeRecommendation = "skipped"
    composite_recommendation_reason: str = ""

    elapsed_seconds: float = 0.0
    skipped: bool = False
    skip_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
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


from ..utils import coerce_to_1d_numpy as _to_1d_numpy  # noqa: E402,F401
# Local name preserved for downstream importers (``dummy`` imports ``_to_1d_numpy`` from here).


def _coerce_to_pandas(df: Any, columns: Sequence[str]) -> pd.DataFrame:
    """Return a pandas frame with exactly ``columns``. Handles polars / pandas /
    ndarray. Fast-path: if already pandas with the right columns, return as is.
    """
    # Polars: route through the Arrow-backed bridge (~32x faster than the
    # consolidating default .to_pandas() on multi-million-row frames).
    if hasattr(df, "to_pandas") and not isinstance(df, pd.DataFrame):
        from ..utils import get_pandas_view_of_polars_df as _get_pandas_view
        df = _get_pandas_view(df)
    if isinstance(df, pd.DataFrame):
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise KeyError(f"BaselineDiagnostics: features missing from train frame: {missing[:5]}" + ("..." if len(missing) > 5 else ""))
        return df.loc[:, list(columns)]
    # ndarray fallback
    arr = np.asarray(df)
    if arr.ndim == 2 and arr.shape[1] == len(columns):
        return pd.DataFrame(arr, columns=list(columns))
    raise TypeError(f"BaselineDiagnostics: unsupported train frame type {type(df).__name__}")


def _select_metric(target_type: str) -> tuple[str, bool]:
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
    """Quick metric computation. Standalone to avoid pulling in mlframe.metrics.core
    (which has heavy numba prewarm cost on cold cache).
    """
    if metric_name == "RMSE":
        diff = y_true.astype(np.float64) - y_pred.astype(np.float64)
        return float(np.sqrt(np.mean(diff * diff)))
    if metric_name == "AUC":
        from mlframe.metrics.core import fast_roc_auc
        # Single-class evaluation set (e.g. tiny val sample where one
        # class is missing) -> AUC undefined; report nan and let caller
        # decide. Caller already guards on min sample size.
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(fast_roc_auc(y_true, y_pred))
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
        return float(metric_after - metric_baseline) * 100.0 if not higher_is_better else float(metric_baseline - metric_after) * 100.0
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

    # Bound onto this class after definition (see bottom of module) from sibling modules so
    # each stays under the LOC budget; declared here so mypy resolves the attribute reads/calls above.
    _make_quick_model: Any
    _fit_quick_and_score: Any
    _run_ablation: Any
    _fit_init_score_baseline: Any
    _build_recommendation: Any

    def __init__(self, config: Any) -> None:
        # Accept either a BaselineDiagnosticsConfig or a dict; dict path
        # is a thin convenience for callers that don't want to import
        # the config class. Pydantic-side validation already happens at
        # the suite-level _ensure_config call site.
        if isinstance(config, dict):
            from ..configs import BaselineDiagnosticsConfig
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
        cat_features: Sequence[str] | None = None,
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
            return self._skipped(target_name, target_type, "config.enabled=False", elapsed=timer() - t0)
        if target_type not in self.config.apply_to_target_types:
            return self._skipped(
                target_name,
                target_type,
                f"target_type='{target_type}' not in apply_to_target_types=" f"{self.config.apply_to_target_types}",
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
            # The quick model is LightGBM, which rejects object/string feature columns ("pandas dtypes must be int,
            # float or bool") even when they are named in categorical_feature -- it needs pandas 'category' dtype.
            # Cast the declared categoricals that arrive as object/string ONCE on the full frame (via assign, so the
            # untouched numeric blocks are reused -- no whole-frame copy on a possibly-huge frame) so the sample /
            # split / ablation all inherit consistent category codes. Without this the ENTIRE diagnostic was silently
            # skipped (broad-except-swallowed) for any frame carrying string categoricals -- e.g. a pandas frame with
            # string cat columns, or a multi_target_regression combo downgraded to regression. Polars Categorical /
            # Enum columns already arrive as 'category' via the Arrow bridge and are left untouched.
            # Cast EVERY non-numeric feature column (not just the declared cat_features) to 'category' -- an object/
            # string column the caller did NOT declare categorical (e.g. a text feature, or an undeclared string col)
            # would otherwise reach the LightGBM quick model and raise the same "pandas dtypes must be int/float/bool".
            # A column whose object cells are non-scalar (embedding List / ndarray) can't be a category; skip it (best-
            # effort -- the quick model would ignore it or the outer except still guards).
            _needs_cat = []
            for c in X.columns:
                _dt = X[c].dtype
                if _dt.kind in "iufb" or isinstance(_dt, pd.CategoricalDtype):
                    continue
                _first = X[c].iloc[0] if len(X) else None
                if hasattr(_first, "__len__") and not isinstance(_first, (str, bytes)):
                    continue  # non-scalar cells (embeddings) -- not castable to a category
                _needs_cat.append(c)
            if _needs_cat:
                X = X.assign(**{c: X[c].astype("category") for c in _needs_cat})
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
                    target_name,
                    target_type,
                    f"raw quick-fit metric is non-finite ({raw_metric}); " "likely degenerate target / sample",
                    elapsed=timer() - t0,
                )

            # 2. Ablation
            ablation = self._run_ablation(
                X_s, y_s, feature_cols, cat_features, target_type,
                raw_fi, raw_metric, metric_name, higher_is_better,
            )
            dominant_features = [{"feature": e.feature, "score": e.delta_pct, "rank": e.rank} for e in ablation]

            # 3. init_score baseline (regression + binary classification).
            init_score_baseline = None
            if target_type in self.config.init_score_apply_to_target_types and ablation:
                init_score_baseline = self._fit_init_score_baseline(
                    X_s, y_s, feature_cols, cat_features, ablation,
                    metric_name, higher_is_better, raw_metric,
                    target_type=target_type,
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
        except (ValueError, TypeError, RuntimeError, ImportError, KeyError, AttributeError, IndexError) as exc:  # pragma: no cover - defensive
            # Narrow catch covers degenerate-input ValueError, dtype-mismatch
            # TypeError, LightGBM RuntimeError, missing optional dep ImportError,
            # config KeyError. KeyboardInterrupt / MemoryError / generic Exception
            # still propagate so a programming bug is not swallowed as
            # "internal_error".
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
        """Build the placeholder report for a target that diagnostics did not run on (e.g. disabled, unsupported type), with NaN headline metrics and ``skipped=True`` so callers can distinguish it from a genuine zero-value result."""
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
    ) -> tuple[pd.DataFrame, np.ndarray, int]:
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

    # Methods _make_quick_model, _fit_quick_and_score, _run_ablation,
    # _fit_init_score_baseline, _build_recommendation are bound onto this class
    # from sibling modules at the bottom of this file. Kept here as a forward
    # reference so the class block is otherwise self-contained.


# ----------------------------------------------------------------------
# Pretty-printer for log output
# ----------------------------------------------------------------------

def format_baseline_diagnostics_report(
    report: BaselineDiagnosticsReport,
    *,
    target_name: str | None = None,
) -> str:
    """Render a multi-line human-readable summary suitable for a single
    ``logger.info`` call.

    Mirrors the style of :func:`format_drift_report` so suite output
    has consistent visual rhythm.
    """
    name = target_name or report.target_name or "target"
    if report.skipped:
        return f"[BaselineDiagnostics] target='{name}' SKIPPED ({report.skip_reason})"

    lines: list[str] = []
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
            lines.append(f"  rank={entry.rank} {entry.feature:<24s} " f"{metric}_after_drop={entry.metric_after_drop:.4f} " f"delta%={entry.delta_pct:+.2f}")
    if report.init_score_baseline is not None:
        isb = report.init_score_baseline
        lines.append(f"[BaselineDiagnostics] init_score({isb.feature_used}) " f"{metric}={isb.metric:.4f} " f"delta%={isb.delta_vs_raw_pct:+.2f} vs raw")
    lines.append(f"[BaselineDiagnostics] composite_recommendation={report.composite_recommendation}")
    if report.composite_recommendation_reason:
        lines.append(f"  reason: {report.composite_recommendation_reason}")
    return "\n".join(lines)


from ._baseline_diagnostics_quick_model import (
    _fit_quick_and_score as _fit_quick_and_score,
    _make_quick_model as _make_quick_model,
)
from ._baseline_diagnostics_ablation import _run_ablation as _run_ablation
from ._baseline_diagnostics_init_score import (
    _fit_init_score_baseline as _fit_init_score_baseline,
)
from ._baseline_diagnostics_recommend import (
    _build_recommendation as _build_recommendation,
)

BaselineDiagnostics._make_quick_model = _make_quick_model
BaselineDiagnostics._fit_quick_and_score = _fit_quick_and_score
BaselineDiagnostics._run_ablation = _run_ablation
BaselineDiagnostics._fit_init_score_baseline = _fit_init_score_baseline
BaselineDiagnostics._build_recommendation = _build_recommendation
