"""Pre-training Dummy-baseline report.

Runs once per (target_type, target_name) AFTER ``BaselineDiagnostics`` and
BEFORE the per-model training loop. Computes a tabular comparison of
trivial / dummy baselines on val + test, picks the strongest by a
target-type-specific primary metric, and emits one overlay plot for the
strongest baseline only.

Sit-alongside relationship with ``baseline_diagnostics.BaselineDiagnostics``:
that module answers "is the target predictable from these features at
all?" via LightGBM quick-fit + feature ablation; this module answers "is
the task even hard?" via comparison to trivial reference predictors.

Design contract (per plan v3 -- 21 defenses D1-D21):
- Default INFO output: <= 2 lines per target (verdict + plot path).
- Full per-baseline table at DEBUG level.
- Suite-end summary with cross-target verdict + 4 canonical UPPERCASE
  WARN tokens (``BEST_MODEL_BELOW_DUMMY``, ``ALL_BASELINES_BELOW_RANDOM``,
  ``TS_BEATS_TREES``, ``PARTIAL_FAILURE``).
- All per-cell metrics in their own try/except -> NaN row on failure
  (D1).
- Strongest-pick gated on non-degeneracy + paired-bootstrap robustness
  (D2).
- LTR group sanity gate (D3); per-output multi-output regression (D4);
  log_loss as headline classification metric, not AUC (D5); suite-end
  summary integration via ``compute_suite_end_summary`` (D6).
- Promoted IMPORTANT defenses D7-D17 (per-target phase qualifier,
  object-dtype gate, n_finite header, n<10 sample-noise gate, slugified
  paths, plot suppression on <2 finite, deterministic per-target seed,
  schema_version, NaN->None serialization, bootstrap CI when n<2000,
  statsmodels deferred import).
- Promoted v3 inline doc/safety items D18-D21 (plot uniqueness trace,
  alpha=0.5<->median note, sklearn>=1.0 assert, hash recipe).

Pure w.r.t. ``(target_type, train_y, val_y, test_y, timestamps,
group_ids, cat_features_chosen)`` -- see ``_baseline_inputs_hash`` for
sweep-orchestrator memoization.

Quantile-regression scope note
------------------------------
For ``target_type="quantile_regression"`` the dispatcher routes through
the regression path: it emits the constant-prediction baselines (mean,
median, quantile_p25, quantile_p75, per_group_mean, plus TS naive when
timestamps are monotonic) measured by RMSE/MAE. That gives the operator
a constant-prediction *floor* against which the multi-alpha quantile
regressor's pinball loss is informative even without explicit per-alpha
empirical-quantile rows. Per-alpha empirical-quantile baselines paired with
mean-pinball-loss-across-alpha as primary metric (plan v3 catalog) require
multi-output prediction plumbing -- predictions become (N, K) instead of
(N,) -- and are deferred until that plumbing is added. The alpha=0.5 row
in such a future expansion would be identical-by-construction to the
``median`` baseline already emitted (D19 self-consistency note).
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# Hard sklearn requirement check. mlframe ships sklearn >= 1.0; the
# check is defensive against constrained environments / future
# loosened pins.
import sklearn
if sklearn.__version__ < "1.0":
    raise RuntimeError(
        f"mlframe.training.baselines.dummy requires sklearn >= 1.0 for "
        f"mean_pinball_loss; got {sklearn.__version__}"
    )

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (
    log_loss,
    mean_absolute_error,
    mean_pinball_loss,
    mean_squared_error,
    roc_auc_score,
)

from .diagnostics import _to_1d_numpy
from ..evaluation import _canonical_multilabel_y
from ._dummy_baseline_compute import (
    _compute_regression_baselines, _compute_classification_baselines,
    _compute_quantile_baselines,
    _safe_metric, _per_group_predict, _pick_per_group_categorical,
    _per_target_seed,
)

# Numba @njit kernels live in ``_dummy_numba_kernels.py``; re-imported below
# so historical in-module call sites (the bootstrap path, the within-group
# rank helper, the multilabel macro log-loss) keep resolving identity-equal.
# The ``_NUMBA_AVAILABLE`` flag comes from the sibling -- it reflects whether
# ``import numba`` succeeded there (optional dep).
from ._dummy_numba_kernels import _NUMBA_AVAILABLE  # noqa: E402,F401

if _NUMBA_AVAILABLE:
    from ._dummy_numba_kernels import (  # noqa: E402,F401
        _numba_macro_log_loss,
        _numba_micro_log_loss,
        _numba_within_group_descending_rank,
        _numba_paired_bootstrap_rmse,
        _numba_paired_bootstrap_mae,
        _numba_bootstrap_rmse_samples,
        _numba_bootstrap_mae_samples,
        _numba_paired_bootstrap_logloss_binary,
        _numba_bootstrap_logloss_binary_samples,
    )

logger = logging.getLogger(__name__)


def _warmup_numba_kernels(verbose: bool = False) -> None:
    """Trigger numba JIT compilation of all dummy_baselines kernels.

    Pre-warms ``_numba_macro_log_loss``, ``_numba_micro_log_loss``,
    ``_numba_within_group_descending_rank``, and the four bootstrap
    kernels (RMSE/MAE x paired/CI) plus the two log-loss kernels by
    invoking each on a 100-row synthetic input. Subsequent first real
    calls skip the 6-10s JIT cold-start that would otherwise dominate
    the multi_output_regression first-target wall time.

    Idempotent: numba caches compilations process-wide, so calling
    this multiple times is essentially free after the first.

    Re-entrancy guard: this function calls
    ``metrics.core.prewarm_numba_cache`` (forward), and that function
    calls back into us (reverse). Without the ``_in_progress`` sentinel
    the pair mutually recurses past the stack limit before either
    try/except sees the failure (observed 2026-05-20 on S: full-suite
    run). The flag is set on the function itself so it's process-local
    and visible from both sides.

    Cost: ~2-5 seconds on the first call (one-time JIT compilation per
    kernel). Subsequent calls are <10ms. Returns silently on numba
    unavailable; logs at DEBUG by default (set ``verbose=True`` for
    INFO).
    """
    if not _NUMBA_AVAILABLE:
        return
    if getattr(_warmup_numba_kernels, "_in_progress", False):
        return
    _warmup_numba_kernels._in_progress = True
    try:
        _warmup_numba_kernels_body(verbose)
    finally:
        _warmup_numba_kernels._in_progress = False


def _warmup_numba_kernels_body(verbose: bool = False) -> None:
    import time as _time
    log = logger.info if verbose else logger.debug
    t0 = _time.time()
    try:
        rng = np.random.default_rng(0)
        # Multilabel kernels
        y_ml = np.zeros((100, 3), dtype=np.int64)
        y_ml[:50, 0] = 1
        y_ml[:30, 1] = 1
        y_ml[:70, 2] = 1
        p_ml = rng.uniform(0.1, 0.9, (100, 3)).astype(np.float64)
        _numba_macro_log_loss(y_ml, p_ml, 100, 3)
        _numba_micro_log_loss(y_ml, p_ml, 100, 3)
        # Within-group rank
        gids = np.array([0, 0, 1, 1, 1, 2] * 17, dtype=np.int64)[:100]
        _numba_within_group_descending_rank(gids)
        # Bootstrap kernels
        y = rng.normal(size=100).astype(np.float64)
        p1 = (y + rng.normal(0, 0.5, 100)).astype(np.float64)
        p2 = (y + rng.normal(0, 0.7, 100)).astype(np.float64)
        _numba_paired_bootstrap_rmse(y, p1, p2, 50, 7)
        _numba_paired_bootstrap_mae(y, p1, p2, 50, 7)
        _numba_bootstrap_rmse_samples(y, p1, 50, 7)
        _numba_bootstrap_mae_samples(y, p1, 50, 7)
        # Binary log-loss kernels
        y_b = (y > 0).astype(np.int64)
        prob1 = np.clip((y + 0.5) * 0.3, 0.05, 0.95).astype(np.float64)
        prob2 = np.clip((y + 0.3) * 0.4, 0.05, 0.95).astype(np.float64)
        _numba_paired_bootstrap_logloss_binary(y_b, prob1, prob2, 50, 7)
        _numba_bootstrap_logloss_binary_samples(y_b, prob1, 50, 7)
        log("[dummy-baselines] numba kernels pre-warmed in %.2fs", _time.time() - t0)
    except Exception as e:
        logger.debug(
            "[dummy-baselines] numba pre-warmup failed (%s: %s); "
            "first real call will JIT-compile lazily",
            type(e).__name__, e,
        )

    # Also warm the full metric / calibration / classification-report
    # kernel set. Pre-fix only the bootstrap subset above was warmed;
    # ``fast_calibration_report`` and its 10+ inner kernels JIT-compiled
    # lazily inside ``report_probabilistic_model_perf`` on the first
    # binary_classification fit, charging 10-16s of compile cost to the
    # training-phase profile attribution. Surfaced by the 500k binary_class
    # x lgb fuzz profile 2026-05-19: 27.2s cold-cache vs 10.1s warm-cache
    # for the same combo + seed, the 17.1s delta sitting almost entirely
    # in numba.dispatcher._compile_for_args under fast_calibration_report.
    # The kernels all have cache=True via NUMBA_NJIT_PARAMS so the warmup
    # populates the on-disk ``__pycache__/*.nbc`` files for ALL subsequent
    # processes too. Best-effort; failures fall back to lazy compile.
    try:
        from ...metrics.core import prewarm_numba_cache as _prewarm_metric_kernels
        _t1 = _time.time()
        _prewarm_metric_kernels()
        log(
            "[dummy-baselines] metric kernel cache pre-warmed in %.2fs",
            _time.time() - _t1,
        )
    except Exception as _exc:
        logger.debug(
            "[dummy-baselines] metric kernel pre-warmup failed (%s: %s); "
            "calibration report will JIT-compile lazily on first call",
            type(_exc).__name__, _exc,
        )

# ``BaselineReport`` NamedTuple + ``SCHEMA_VERSION`` moved to
# ``_dummy_report_type.py``; re-imported below so historical
# ``from .dummy import BaselineReport`` resolves.
from ._dummy_report_type import BaselineReport, SCHEMA_VERSION  # noqa: E402,F401
def _baseline_inputs_hash(
    target_type: str,
    train_y: Any,
    val_y: Any,
    test_y: Any,
    timestamps: Any = None,
    group_ids: Any = None,
) -> str:
    """SHA256 of the inputs that determine baseline output.

    Use from a hyperparam-sweep orchestrator to memoize across runs:
    ``compute_dummy_baselines`` is pure w.r.t. these inputs (the fitted
    cat_features pick is determined inside, deterministic per-input).

    Documented in the module docstring; not called by
    ``compute_dummy_baselines`` itself (the contract is caller-side).
    """
    h = hashlib.sha256()
    h.update(str(target_type).encode())
    for arr in (train_y, val_y, test_y, timestamps, group_ids):
        if arr is None:
            h.update(b"None")
        else:
            try:
                a = np.ascontiguousarray(_to_1d_numpy(arr) if hasattr(arr, "__len__") else np.array([arr]))
                h.update(a.tobytes())
            except Exception:
                h.update(repr(arr).encode())
    return h.hexdigest()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


_SLUG_RE = re.compile(r"[^a-zA-Z0-9_]+")


def _slugify(s: str) -> str:
    """Cheap slugifier for safe path components (D11)."""
    if not s:
        return "unnamed"
    return _SLUG_RE.sub("_", str(s)).strip("_") or "unnamed"


def _is_finite_mask(y: np.ndarray) -> np.ndarray:
    """Boolean mask of rows with finite numeric / non-null values."""
    if y.dtype.kind in "fc":
        return np.isfinite(y)
    if y.dtype == object:
        return np.array([v is not None and (not isinstance(v, float) or np.isfinite(v)) for v in y])
    # int / bool / etc. -- always finite for our purposes
    return np.ones(len(y), dtype=bool)


def _has_signal(target_type: str, y_ref: np.ndarray, n_min: int = 10) -> tuple[bool, str]:
    """Per-target non-degeneracy gate for strongest-pick reference split (D10, D2).

    Returns ``(has_signal, reason_if_not)``.
    """
    if y_ref is None or len(y_ref) < n_min:
        return False, f"n_ref={0 if y_ref is None else len(y_ref)} < {n_min}"
    if target_type in ("regression", "quantile_regression"):
        if y_ref.ndim > 1:
            # multi-output: need signal in at least one column
            has_var = any(np.std(y_ref[:, k]) > 1e-9 for k in range(y_ref.shape[1]))
        else:
            has_var = np.std(y_ref) > 1e-9
        if not has_var:
            return False, "y_ref has zero variance"
    elif target_type in ("binary_classification", "multiclass_classification"):
        n_classes = len(np.unique(y_ref))
        if n_classes < 2:
            return False, f"y_ref has only {n_classes} class"
    elif target_type == "multilabel_classification":
        # multilabel: at least one label needs both 0 and 1 in val
        if y_ref.ndim != 2:
            return False, f"multilabel y_ref ndim={y_ref.ndim} != 2"
        any_label_has_signal = any(
            len(np.unique(y_ref[:, k])) >= 2 for k in range(y_ref.shape[1])
        )
        if not any_label_has_signal:
            return False, "no multilabel column has both classes"
    return True, ""


# ---------------------------------------------------------------------
# Time-series detection (statsmodels deferred import)
# ---------------------------------------------------------------------


# Timestamp / period / monotonicity helpers moved to
# ``_dummy_timeseries.py``; re-exported below so the orchestrator continues
# to call them via the same names. See sibling for SSOT.
from ._dummy_timeseries import (  # noqa: E402,F401
    _normalize_timestamps,
    _is_temporally_monotonic,
    _infer_ts_step_periods,
    _detect_acf_periods,
    _resolve_ts_periods,
)



def compute_dummy_baselines(
    target_type: str,
    target_name: str,
    *,
    train_X: Any,
    val_X: Any,
    test_X: Any,
    train_y: Any,
    val_y: Any,
    test_y: Any,
    timestamps_train: Any = None,
    timestamps_val: Any = None,
    timestamps_test: Any = None,
    group_ids_train: Any = None,
    group_ids_val: Any = None,
    group_ids_test: Any = None,
    doc_ids_train: Any = None,
    doc_ids_val: Any = None,
    doc_ids_test: Any = None,
    cat_features: Sequence[str] | None = None,
    target_label_encoder: Any = None,
    quantile_alphas: Sequence[float] | None = None,
    config: Any = None,
    plot_file_prefix: str = "",
) -> BaselineReport:
    """Compute dummy baselines for one (target_type, target_name).

    Public entry point. Routes to per-target dispatcher, computes
    per-cell metrics in isolated try/except, picks strongest with
    non-degeneracy + paired-bootstrap robustness gates, optionally
    saves overlay plot for strongest baseline.
    """
    import time as _time
    t0 = _time.time()

    if config is None:
        from ..configs import DummyBaselinesConfig
        config = DummyBaselinesConfig()

    # Coerce y to 1D / 2D numpy as appropriate (object-dtype gate).
    train_y_arr = _coerce_y(train_y, target_type, target_name)
    val_y_arr = _coerce_y(val_y, target_type, target_name) if val_y is not None else None
    test_y_arr = _coerce_y(test_y, target_type, target_name) if test_y is not None else None

    if train_y_arr is None:
        return _empty_report(target_type, target_name, t0, reason="object-dtype-target")

    n_train = len(train_y_arr)
    n_val = 0 if val_y_arr is None else len(val_y_arr)
    n_test = 0 if test_y_arr is None else len(test_y_arr)
    n_train_finite = int(_is_finite_mask(train_y_arr).sum()) if train_y_arr.ndim == 1 else n_train
    n_val_finite = int(_is_finite_mask(val_y_arr).sum()) if val_y_arr is not None and val_y_arr.ndim == 1 else n_val
    n_test_finite = int(_is_finite_mask(test_y_arr).sum()) if test_y_arr is not None and test_y_arr.ndim == 1 else n_test

    # Skip block if both val and test are uninformative
    if n_val_finite < 2 and n_test_finite < 2:
        logger.warning(
            "[DUMMY_BASELINES] FAILED target='%s' - both val (%d/%d finite) and "
            "test (%d/%d finite) targets have <2 finite values",
            target_name, n_val_finite, n_val, n_test_finite, n_test,
        )
        return _empty_report(target_type, target_name, t0, reason="both-splits-uninformative")

    # Multi-output regression. For 2D y in regression / quantile_regression,
    # run the dispatcher per output and aggregate per-output strongest +
    # cross-output normalized strongest. Headline emission stays one verdict
    # block per target (not K verdicts).
    if (
        target_type in ("regression", "quantile_regression")
        and train_y_arr.ndim == 2
        and train_y_arr.shape[1] > 1
    ):
        return _compute_multi_output_regression(
            target_type=target_type,
            target_name=target_name,
            train_X=train_X, val_X=val_X, test_X=test_X,
            train_y_arr=train_y_arr, val_y_arr=val_y_arr, test_y_arr=test_y_arr,
            timestamps_train=timestamps_train, timestamps_val=timestamps_val,
            timestamps_test=timestamps_test,
            cat_features=cat_features,
            config=config,
            plot_file_prefix=plot_file_prefix,
            t0=t0,
            n_train=n_train, n_val=n_val, n_test=n_test,
            n_train_finite=n_train_finite, n_val_finite=n_val_finite,
            n_test_finite=n_test_finite,
        )

    # Normalize timestamps once (mixed-tz handling).
    ts_train = _normalize_timestamps(timestamps_train)
    ts_val = _normalize_timestamps(timestamps_val)
    ts_test = _normalize_timestamps(timestamps_test)

    # Dispatch by target_type
    val_preds: dict[str, np.ndarray] = {}
    test_preds: dict[str, np.ndarray] = {}
    extras: dict[str, Any] = {}

    if target_type == "quantile_regression" and quantile_alphas is not None:
        # Per-alpha empirical-quantile baselines + pinball-loss metric.
        # Falls back to regression path when quantile_alphas not provided.
        val_preds, test_preds, extras = _compute_quantile_baselines(
            target_name, train_y_arr, val_y_arr, test_y_arr,
            list(quantile_alphas), config,
        )
        extras["quantile_alphas"] = list(quantile_alphas)
    elif target_type in ("regression", "quantile_regression"):
        val_preds, test_preds, extras = _compute_regression_baselines(
            target_name, train_X, val_X, test_X,
            train_y_arr, val_y_arr, test_y_arr,
            ts_train, ts_val, ts_test,
            cat_features, config, target_type=target_type,
        )
    elif target_type in ("binary_classification", "multiclass_classification"):
        # Determine n_classes from train + val + test union
        all_y = np.concatenate([
            train_y_arr,
            val_y_arr if val_y_arr is not None else np.array([], dtype=train_y_arr.dtype),
            test_y_arr if test_y_arr is not None else np.array([], dtype=train_y_arr.dtype),
        ])
        unique_classes = np.unique(all_y[~pd.isna(all_y)] if all_y.dtype.kind in "fc" else all_y)
        n_classes = max(2, len(unique_classes))
        # Label-encode to positions 0..K-1 against the sorted class union. The classification
        # baselines assume positional labels: ``np.bincount(train_y, minlength=K)`` (returns
        # max(label)+1 wide for non-0-indexed labels -> phantom class-0 column -> wrong-width
        # (N, K) prob matrices) and ``log_loss(y, p, labels=np.arange(K))`` in the metrics table
        # (raises when a raw label like 3 is not in {0,1,2} -> every classification metric NaN).
        # Integer multiclass targets are NOT label-encoded upstream (only string/object are), so
        # {1,2,3} / {10,20,30} reach here raw. searchsorted is identity for already-0..K-1 labels,
        # so the common path is bit-identical; only non-contiguous / non-0-based labels are remapped.
        _cls_sorted = unique_classes
        if len(_cls_sorted) and not np.array_equal(_cls_sorted, np.arange(len(_cls_sorted))):
            def _enc(_y):
                if _y is None:
                    return None
                return np.searchsorted(_cls_sorted, np.asarray(_y)).astype(np.int64)
            train_y_arr = _enc(train_y_arr)
            val_y_arr = _enc(val_y_arr)
            test_y_arr = _enc(test_y_arr)
        val_preds, test_preds, extras = _compute_classification_baselines(
            target_name, train_X, val_X, test_X,
            train_y_arr, val_y_arr, test_y_arr,
            ts_train, cat_features, config,
            target_type=target_type, n_classes=n_classes,
        )
        extras["n_classes"] = n_classes
        extras["class_labels"] = list(_cls_sorted)
    elif target_type == "multilabel_classification":
        val_preds, test_preds, extras = _compute_multilabel_baselines(
            target_name, train_y_arr, val_y_arr, test_y_arr, config,
        )
    elif target_type == "learning_to_rank":
        val_preds, test_preds, extras = _compute_ltr_baselines(
            target_name,
            train_y_arr, val_y_arr, test_y_arr,
            group_ids_train, group_ids_val, group_ids_test,
            ts_train, ts_val, ts_test,
            config,
            doc_ids_train=doc_ids_train,
            doc_ids_val=doc_ids_val,
            doc_ids_test=doc_ids_test,
        )
    else:
        return _empty_report(
            target_type, target_name, t0,
            reason=f"unsupported target_type={target_type}",
        )

    # Compute metrics table
    table, primary_metric = _compute_metrics_table(
        target_type, val_preds, test_preds, val_y_arr, test_y_arr,
        group_ids_val=group_ids_val, group_ids_test=group_ids_test,
        extras=extras,
    )

    # Strongest-pick: non-degeneracy gate + paired-bootstrap
    strongest, ts_period_used = _pick_strongest(
        target_type, table, val_y_arr, test_y_arr, primary_metric, extras, config,
    )

    # Paired-bootstrap robustness: compute delta vs runner-up + 95% CI +
    # P(strongest beats runner-up). Below `strongest_min_beat_runner_up_prob`
    # the strongest is annotated as TIE and the overlay plot is skipped.
    # Gated on the same n-threshold as bootstrap CI -- at large n the
    # point-estimate signal-to-noise is high enough that paired bootstrap
    # is just expensive ceremony (~3-4s on n=10^5).
    n_ref_for_paired = min(
        n_val_finite if n_val_finite > 0 else 10_000_000,
        n_test_finite if n_test_finite > 0 else 10_000_000,
    )
    if (
        strongest is not None
        and primary_metric is not None
        and n_ref_for_paired < config.bootstrap_ci_threshold
    ):
        try:
            paired = _paired_bootstrap_vs_runner_up(
                target_type, strongest, primary_metric, table,
                val_preds, test_preds, val_y_arr, test_y_arr,
                n_resamples=config.paired_bootstrap_n_resamples,
                seed=_per_target_seed(config.random_state, target_name) + 1,
            )
            if paired is not None:
                extras["paired_bootstrap"] = paired
                if paired.get("p_strongest_beats") is not None and (
                    paired["p_strongest_beats"] < config.strongest_min_beat_runner_up_prob
                ):
                    extras["tie"] = True
        except Exception as e:
            logger.debug(
                "[dummy-baselines] target='%s' paired-bootstrap failed (%s); skipping",
                target_name, e,
            )

    # Bootstrap CI for strongest baseline when min(n_val, n_test) < 2000.
    # Below that threshold the noise floor on RMSE / log_loss / NDCG is non-
    # trivial (>1%), so a CI grounds the verdict line. Above 2000, point
    # estimate is accurate to <1% and CI is suppressed to keep output compact.
    n_ref_for_ci = min(
        n_val_finite if n_val_finite > 0 else 10_000_000,
        n_test_finite if n_test_finite > 0 else 10_000_000,
    )
    if (
        strongest is not None
        and primary_metric is not None
        and n_ref_for_ci < config.bootstrap_ci_threshold
        and n_ref_for_ci >= 10
    ):
        try:
            ci = _bootstrap_ci_for_strongest(
                target_type, strongest, primary_metric,
                val_preds, test_preds, val_y_arr, test_y_arr,
                n_resamples=config.bootstrap_ci_n_resamples,
                seed=_per_target_seed(config.random_state, target_name),
            )
            if ci is not None:
                extras["bootstrap_ci"] = ci
        except Exception as e:
            logger.debug(
                "[dummy-baselines] target='%s' bootstrap CI failed (%s); skipping",
                target_name, e,
            )

    # The standard report_model_perf pipeline (gated by config.plot_strongest) already
    # produces per-model scatter + residual + calibration charts, so the dedicated overlay
    # PNG is off by default; it renders only when the operator opts into config.overlay_plot.
    plot_path = None

    # Expose strongest-baseline val/test predictions via
    # ``extras`` so a downstream consumer (core.py, between
    # dummy-baselines computation and the per-target model-training
    # loop) can render the "best-baseline-overlay" pre-training chart
    # the user repeatedly asked for. We keep the prediction arrays
    # OUT of ``BaselineReport``'s top-level fields (they'd bloat
    # JSON serialization of metadata.pkl) and store them in extras
    # under explicit keys that the renderer reads by name. Memory
    # cost: 2 x n_split float arrays per target, freed once the
    # renderer consumes them.
    if strongest is not None:
        sv = val_preds.get(strongest)
        st = test_preds.get(strongest)
        if sv is not None:
            extras["strongest_val_preds"] = np.asarray(sv)
        if st is not None:
            extras["strongest_test_preds"] = np.asarray(st)

    elapsed_s = _time.time() - t0
    report = BaselineReport(
        target_type=target_type,
        target_name=target_name,
        table=table,
        strongest=strongest,
        primary_metric=primary_metric,
        ts_period_used=ts_period_used,
        plot_path=plot_path,
        elapsed_s=elapsed_s,
        n_train=n_train, n_val=n_val, n_test=n_test,
        n_train_finite=n_train_finite, n_val_finite=n_val_finite, n_test_finite=n_test_finite,
        extras=extras,
    )

    # Optional dedicated pre-training overlay (off by default; standard reports cover the floor).
    # BaselineReport is an immutable NamedTuple, so rebuild it with the saved path via _replace.
    if getattr(config, "overlay_plot", False) and strongest is not None:
        _ov_save = (plot_file_prefix + "dummy_overlay.png") if plot_file_prefix else None
        try:
            _fig = plot_best_dummy_baseline_overlay(
                report, val_y=val_y_arr, test_y=test_y_arr,
                save_path=_ov_save, show=False,
            )
            if _fig is not None and _ov_save:
                report = report._replace(plot_path=_ov_save)
        except Exception as _ov_err:
            logger.debug("[dummy-baselines] target='%s' overlay plot failed (%s); skipping", target_name, _ov_err)

    return report


# ---------------------------------------------------------------------
# Multilabel + LTR dispatchers + metrics + plot + helpers
# ---------------------------------------------------------------------


# Multilabel / LTR / target-coercion helpers moved to
# ``_dummy_compute_helpers.py``; re-exported below so the orchestrator
# continues to call them via the same names. See sibling for SSOT.
from ._dummy_compute_helpers import (  # noqa: E402,F401
    _compute_multilabel_baselines,
    _compute_ltr_baselines,
    _within_group_descending_index,
    _coerce_y,
    _empty_report,
)
# Metrics-table compute + strongest-pick + overlay-plot moved to
# ``_dummy_metrics_pick_plot.py``; re-exported below so the orchestrator
# continues to call them via the same names. See sibling for SSOT.
from ._dummy_metrics_pick_plot import (  # noqa: E402,F401
    _compute_metrics_table,
    _pick_strongest,
    plot_best_dummy_baseline_overlay,
    _safe_metric_for_title,
)

# Bootstrap CIs + paired-bootstrap robustness moved to ``_dummy_bootstrap.py``;
# re-exported below so the orchestrator continues to call them via the same
# names. See sibling for SSOT.
from ._dummy_bootstrap import (  # noqa: E402,F401
    _paired_bootstrap_vs_runner_up,
    _vectorized_bootstrap_logloss_samples,
    _bootstrap_ci_for_strongest,
)

def _compute_multi_output_regression(
    *,
    target_type: str,
    target_name: str,
    train_X: Any, val_X: Any, test_X: Any,
    train_y_arr: np.ndarray,
    val_y_arr: np.ndarray | None,
    test_y_arr: np.ndarray | None,
    timestamps_train: Any, timestamps_val: Any, timestamps_test: Any,
    cat_features: Sequence[str] | None,
    config: Any,
    plot_file_prefix: str,
    t0: float,
    n_train: int, n_val: int, n_test: int,
    n_train_finite: int, n_val_finite: int, n_test_finite: int,
) -> BaselineReport:
    """Multi-output regression dispatcher.

    Runs ``compute_dummy_baselines`` per output (K independent calls), then
    aggregates a per-output strongest-pick block + cross-output normalized
    strongest-pick (mean of ``RMSE / std(y_train_per_target)``). Output 0
    is used for the overlay plot; outputs 1..K-1 plots suppressed (one-
    plot-per-target invariant).
    """
    import time as _time
    K = train_y_arr.shape[1]
    per_output_strongest: list[dict[str, Any]] = []
    sub_reports: list[BaselineReport] = []

    for k in range(K):
        sub_train = train_y_arr[:, k]
        sub_val = val_y_arr[:, k] if val_y_arr is not None and val_y_arr.ndim == 2 else None
        sub_test = test_y_arr[:, k] if test_y_arr is not None and test_y_arr.ndim == 2 else None
        # Pass plot_file_prefix only for k=0 (one plot per target invariant)
        sub_plot_prefix = plot_file_prefix if k == 0 else ""
        sub_rep = compute_dummy_baselines(
            target_type=target_type,
            target_name=f"{target_name}[{k}]",
            train_X=train_X, val_X=val_X, test_X=test_X,
            train_y=sub_train, val_y=sub_val, test_y=sub_test,
            timestamps_train=timestamps_train,
            timestamps_val=timestamps_val,
            timestamps_test=timestamps_test,
            cat_features=cat_features,
            config=config,
            plot_file_prefix=sub_plot_prefix,
        )
        sub_reports.append(sub_rep)
        if sub_rep.strongest is not None and sub_rep.primary_metric is not None:
            try:
                primary_value = float(
                    sub_rep.table.loc[sub_rep.strongest, sub_rep.primary_metric]
                )
            except (KeyError, ValueError, TypeError) as _lookup_err:
                # Surface the silent NaN substitution: pre-fix this swallowed bare
                # Exception, NaN flowed into per_output_strongest -> cross-output
                # "mean normalized RMSE" aggregator silently averaged NaN with real
                # values. Operator reads the cross-output number without knowing one
                # of the K outputs was faked.
                logger.warning(
                    "dummy_baselines multi_output: lookup failed for output=%d "
                    "strongest=%r metric=%r (%s: %s); NaN substituted in per-output "
                    "aggregation -- cross-output summary will average NaN with real values.",
                    k, sub_rep.strongest, sub_rep.primary_metric,
                    type(_lookup_err).__name__, _lookup_err,
                )
                primary_value = float("nan")
            # Normalized aggregation: RMSE / std(y_train_per_target)
            std_k = float(np.std(sub_train)) if sub_train.size > 1 else 1.0
            normalized = primary_value / std_k if std_k > 1e-12 else float("nan")
            per_output_strongest.append({
                "output": k,
                "name": sub_rep.strongest,
                "primary_metric": sub_rep.primary_metric,
                "primary_value": primary_value,
                "normalized": normalized,
            })

    # Cross-output normalized strongest-pick: pick the baseline whose mean
    # normalized RMSE across outputs is lowest.
    cross_output: dict[str, Any] | None = None
    if per_output_strongest:
        # Aggregate per-baseline mean normalized RMSE across all sub-reports.
        baseline_norm_means: dict[str, list[float]] = {}
        for k, sub_rep in enumerate(sub_reports):
            sub_train_k = train_y_arr[:, k]
            std_k = float(np.std(sub_train_k)) if sub_train_k.size > 1 else 1.0
            if std_k <= 1e-12:
                continue
            for baseline_name in sub_rep.table.index:
                v = sub_rep.table.loc[baseline_name].get(sub_rep.primary_metric)
                if v is None or not np.isfinite(v):
                    continue
                baseline_norm_means.setdefault(str(baseline_name), []).append(float(v) / std_k)
        if baseline_norm_means:
            mean_norms = {
                name: float(np.mean(vals)) for name, vals in baseline_norm_means.items()
                if len(vals) >= max(1, K // 2)  # require coverage on >= half of outputs
            }
            if mean_norms:
                best_name = min(mean_norms, key=mean_norms.get)
                cross_output = {
                    "name": best_name,
                    "mean_normalized": mean_norms[best_name],
                }

    # Build a primary table from output-0 (representative). Strongest /
    # primary_metric / plot_path inherit from sub_reports[0].
    rep0 = sub_reports[0] if sub_reports else None
    extras = {
        "n_outputs": K,
        "per_output_strongest": per_output_strongest,
    }
    if cross_output is not None:
        extras["cross_output_strongest"] = cross_output

    return BaselineReport(
        target_type=target_type,
        target_name=target_name,
        table=rep0.table if rep0 is not None else pd.DataFrame(),
        strongest=cross_output["name"] if cross_output else (rep0.strongest if rep0 else None),
        primary_metric=rep0.primary_metric if rep0 else None,
        ts_period_used=rep0.ts_period_used if rep0 else None,
        plot_path=rep0.plot_path if rep0 else None,
        elapsed_s=_time.time() - t0,
        n_train=n_train, n_val=n_val, n_test=n_test,
        n_train_finite=n_train_finite, n_val_finite=n_val_finite, n_test_finite=n_test_finite,
        extras=extras,
    )


# Suite-end summary + verdict-table formatting moved to
# ``_dummy_summary_format.py``; re-exported below so the orchestrator and
# caller-side imports continue to resolve. See sibling for SSOT.
from ._dummy_summary_format import (  # noqa: E402,F401
    format_suite_end_summary,
    format_unified_target_verdict_table,
)


__all__ = [
    "compute_dummy_baselines",
    "format_suite_end_summary",
    "format_unified_target_verdict_table",
    "BaselineReport",
    "SCHEMA_VERSION",
    "_baseline_inputs_hash",
]
