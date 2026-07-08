"""Pluggable metrics registry for target-type-aware reporting.

Replaces hardcoded metric calls in ``report_probabilistic_model_perf`` with
a registry indexed by ``TargetTypes``. Built-in registrations for
multilabel (``hamming_loss``, ``subset_accuracy``, ``jaccard_score_multilabel``)
land at import time.

Extensibility
-------------
External callers can register domain-specific metrics without touching
``evaluation.py``:

    from mlframe.training.metrics_registry import register_metric
    from mlframe.training.configs import TargetTypes

    def my_custom_multilabel_metric(y_true, probs_NK, preds_NK):
        return some_score

    register_metric(
        TargetTypes.MULTILABEL_CLASSIFICATION,
        "my_custom_metric",
        my_custom_multilabel_metric,
    )

``register_metric`` idempotent — re-registering a name overwrites the
previous impl (useful for A/B comparison or test stubs).

The metric callable receives ``(y_true, probs_NK, preds_NK)`` where:
- y_true   : 1-D labels (binary/multiclass) OR 2-D indicator (multilabel)
- probs_NK : canonicalised (N, K) probability matrix
- preds_NK : decision-rule output (1-D argmax or 2-D binary threshold)

Returns any value with a ``__format__`` method (typically float).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Any

from .configs import TargetTypes


@dataclass(frozen=True)
class MetricSpec:
    """Structured metadata for a registered metric.

    Fields:
    - fn: the metric callable ``(y_true, probs_NK, preds_NK) -> value``.
    - higher_is_better: True when larger values mean better performance
      (e.g. accuracy, AUC), False for losses (e.g. log_loss, hamming).
    - description: optional human-readable blurb surfaced in introspection.
    """
    fn: Callable
    higher_is_better: bool = True
    description: str = ""


_REGISTRY: dict[TargetTypes, dict[str, MetricSpec]] = {}


def register_metric(
    target_type: TargetTypes,
    name: str,
    fn: Callable,
    *,
    higher_is_better: bool = True,
    description: str = "",
) -> None:
    """Register a metric function for a target type.

    The optional ``higher_is_better`` flag lets downstream callers (lift-pct,
    strongest-pick, leaderboards) interpret the metric direction without
    hard-coding string lookups against the metric name. The ``description``
    surfaces in :func:`list_registered_specs` for help-text rendering.

    Idempotent: re-registering the same name overwrites.
    """
    _REGISTRY.setdefault(target_type, {})[name] = MetricSpec(
        fn=fn, higher_is_better=bool(higher_is_better), description=description,
    )


def unregister_metric(target_type: TargetTypes, name: str) -> None:
    """Remove a registered metric. No-op if not registered."""
    if target_type in _REGISTRY:
        _REGISTRY[target_type].pop(name, None)


def iter_extra_metrics(target_type: TargetTypes, y_true, probs_NK, preds_NK) -> Iterator[tuple[str, Any]]:
    """Yield (name, value) for every registered metric on this target type.

    Narrow exception catch: only the documented failure modes for sklearn
    metric callables propagate as recoverable (ValueError on degenerate
    inputs, ZeroDivisionError on empty groups, TypeError on shape
    mismatches). Anything else (KeyboardInterrupt, MemoryError, programming
    bugs in caller-supplied metrics) bubbles up so a real bug is not
    masquerading as "metric not applicable".
    """
    import logging
    logger = logging.getLogger(__name__)

    for name, spec in _REGISTRY.get(target_type, {}).items():
        try:
            value = spec.fn(y_true, probs_NK, preds_NK)
            yield name, value
        except (ValueError, ZeroDivisionError, TypeError, FloatingPointError) as e:
            # WARNING not DEBUG -- silently omitting a metric from the report is a
            # substantive event the operator needs to see. Pre-fix the operator saw
            # the report missing the metric row entirely and concluded the metric
            # was never configured (or the model was fine), not "the metric crashed
            # on degenerate data". Common upstream causes: roc_auc on a single-class
            # slice (val/test became class-degenerate due to outlier_detection +
            # tight aging window), pinball on shape mismatch (quantile preds vs
            # scalar y_true).
            try:
                _n = len(y_true) if y_true is not None else 0
            except TypeError:
                _n = -1  # un-sized iterator etc.
            logger.warning(
                "metric %r failed on target_type=%s n=%d: %s: %s; omitted from report",
                name, target_type, _n, type(e).__name__, e,
            )


def list_registered(target_type: TargetTypes) -> list:
    """Introspection: list registered metric names for a target type."""
    return list(_REGISTRY.get(target_type, {}).keys())


def list_registered_specs(target_type: TargetTypes) -> dict[str, MetricSpec]:
    """Introspection: full {name: MetricSpec} map (direction + description)."""
    return dict(_REGISTRY.get(target_type, {}))


def get_metric_direction(
    target_type: TargetTypes, name: str,
) -> Optional[bool]:
    """Return ``higher_is_better`` for a registered metric, or None if absent."""
    spec = _REGISTRY.get(target_type, {}).get(name)
    return None if spec is None else spec.higher_is_better


# Built-in known directions for common metric NAMES (independent of target
# type). Wave 20 fix: 6 production sites previously rolled their own
# ad-hoc substring / whitelist tables that DISAGREED with each other
# (e.g. dummy_baselines.py:606 included AUC; dummy_baselines.py:1807
# excluded it). This single table replaces all of them.
#
# The lookup is case-insensitive and strips common prefixes (val_, test_,
# oof_, train_) and `@k` rank-cutoff suffixes (NDCG@10 -> NDCG).
#
# Higher-is-better metrics: rank quality, classification quality, R^2.
# Lower-is-better metrics: regression losses, calibration losses,
# probabilistic losses.
#
# Names should be stored canonicalised (lowercase, prefix-stripped, @k-stripped)
# in the table; lookup canonicalises the query the same way.
_KNOWN_METRIC_DIRECTIONS_HIGHER: frozenset[str] = frozenset({
    # Ranking-quality metrics
    "ndcg", "map", "mrr", "ap", "average_precision", "precision_at_k",
    "recall_at_k", "hit_rate", "hit_at_k",
    "dcg", "dcg_at_k", "err", "expected_reciprocal_rank",
    # Classification-quality metrics
    "auc", "roc_auc", "pr_auc", "auprc", "auc_mu",
    "accuracy", "accuracy_score",
    "f1", "f1_score", "f1_macro", "f1_micro", "f1_weighted",
    "f_beta", "f0_5", "f2",
    "precision", "precision_macro", "precision_micro", "precision_weighted",
    "recall", "recall_macro", "recall_micro", "recall_weighted",
    "sensitivity", "specificity", "tpr", "tnr", "npv",
    "balanced_accuracy", "matthews_corrcoef", "mcc", "mcc_multiclass",
    "cohen_kappa", "kappa", "weighted_kappa", "quadratic_weighted_kappa", "qwk",
    "subset_accuracy", "jaccard_score_multilabel", "jaccard", "jaccard_macro",
    "gini",
    # Binary higher-is-better extras from 2026-05-28 audit batch.
    "g_mean", "ks", "ks_statistic", "bss", "brier_skill_score",
    "lift", "lift_at_k",
    # Top-k accuracy for multiclass
    "top_k_accuracy", "top1", "top3", "top5",
    # Regression-quality metrics where higher means better
    "r2", "r2_score", "explained_variance", "explainedvariance",
    "nse", "nash_sutcliffe", "pearson", "spearman", "kendall_tau",
    "concordance_index", "c_index",
    # Tier 2 (2026-05-28): Accuracy Ratio = 2*AUC-1
    "accuracy_ratio", "ar", "cap_ar",
})

_KNOWN_METRIC_DIRECTIONS_LOWER: frozenset[str] = frozenset({
    # Regression losses
    "rmse", "mae", "mse", "mape", "mape_mean", "smape", "mdape", "wmape",
    "huber_loss", "median_absolute_error", "max_error",
    "rmsle", "mase", "cv_rmse", "rmspe", "logcosh", "logcosh_loss",
    # Tier 2 GLM deviances (2026-05-28) - all lower-is-better losses
    "poisson_deviance", "gamma_deviance", "tweedie_deviance",
    # Tier 2 calibration / probabilistic forecasting
    "hosmer_lemeshow", "hosmer_lemeshow_chi2", "hl_chi2",
    "crps", "crps_from_quantiles",
    # Signed bias / drift losses where 0 is best (use |.| -> lower)
    "mbe", "mean_bias_error",
    # Probabilistic / calibration losses
    "log_loss", "logloss", "brier", "brier_score", "cross_entropy",
    # Exponential proper scoring rule (per-object minimizer is the true probability).
    "exploss",
    # Multi-class / multi-label aggregation variants of the probabilistic
    # losses. _dummy_metrics_pick_plot.py emits ``log_loss_macro`` /
    # ``log_loss_micro`` per split; without these the canonical lookup
    # returns None and _pick_strongest warns then silently defaults to
    # minimize. Direction is the same as the un-aggregated parent (mean
    # of per-class losses is still a loss).
    "log_loss_macro", "log_loss_micro", "log_loss_weighted",
    "logloss_macro", "logloss_micro", "logloss_weighted",
    "brier_macro", "brier_micro", "brier_weighted",
    "brier_score_macro", "brier_score_micro", "brier_score_weighted",
    "cross_entropy_macro", "cross_entropy_micro", "cross_entropy_weighted",
    "kl_divergence", "kl", "js_divergence", "js", "wasserstein",
    "perplexity",
    "ice", "integral_error", "integral_calibration_error",
    "ece", "expected_calibration_error",
    "rps", "ranked_probability_score",
    # Drift / distributional - higher value = more drift = worse
    "psi", "population_stability_index", "ks_distribution_distance",
    # Calibration test statistics: |Z|=0 means well-calibrated; treat the
    # raw Z as a distance from 0 -> closer-to-0 better, i.e. lower is
    # better in absolute value. Callers should report |Z| if they
    # interpret it as "miscalibration magnitude".
    "spiegelhalter_z",
    # Multilabel losses
    "hamming_loss", "hamming", "coverage_error", "ranking_loss",
    "label_ranking_loss", "one_error",
    # Multilabel ranking-quality also: lrap is higher_is_better
    # (added separately below).
    # Confusion-derived rates where higher is worse
    "fpr", "fnr",
    # Quantile losses
    "pinball", "pinball_loss", "quantile_loss",
    # Native booster eval-metric aliases (LightGBM / XGBoost / CatBoost). All are lower-is-better losses; they
    # arrive verbatim from ``evals_log`` / ``env.evaluation_result_list`` so the monotonic-decline + worsening
    # detectors must resolve their direction or they self-disable on an otherwise-known metric.
    "l2", "l1", "binary_logloss", "multi_logloss", "binary_error", "multi_error", "error",
    "fair", "tweedie", "gamma", "poisson",
})

# Carry-out: LRAP is higher-is-better; ensure it lands in HIGHER bucket
# even though it's a multilabel-ranking metric (not in the
# classification-quality cluster above).
_KNOWN_METRIC_DIRECTIONS_HIGHER = frozenset(_KNOWN_METRIC_DIRECTIONS_HIGHER | {"lrap", "label_ranking_average_precision"})


def _canonicalise_metric_name(name: str) -> str:
    """Strip common prefixes (val_/test_/oof_/train_) and @k rank-cutoff
    suffixes; lowercase. Used by ``metric_name_higher_is_better``."""
    if not isinstance(name, str):
        return ""
    s = name.strip().lower()
    for prefix in ("val_", "test_", "oof_", "train_", "holdout_"):
        if s.startswith(prefix):
            s = s[len(prefix) :]
            break
    if "@" in s:
        s = s.split("@", 1)[0]
    return s


def metric_name_higher_is_better(name: str) -> Optional[bool]:
    """Target-type-agnostic direction lookup for a metric name.

    Returns:
        True if the metric is in the known higher-is-better set.
        False if the metric is in the known lower-is-better set.
        None if the name is genuinely unknown (caller must decide whether
        to raise / default / warn). Returning None rather than a default
        prevents the ``endswith("e") -> 'min'`` anti-pattern that wave 20
        identified as the root cause of training the WORST iteration on
        custom metric names.

    Wave 20 fix: replaces 6 ad-hoc substring/whitelist tables in
    dummy_baselines.py / _callbacks.py / _phase_composite_post.py that
    disagreed with each other and missed common classification metrics
    (F1, accuracy, R2, precision, recall, AP) AND common regression
    losses (MAPE, MSE, ICE, brier, KL, perplexity).

    Example:
        >>> metric_name_higher_is_better("val_AUC")
        True
        >>> metric_name_higher_is_better("test_RMSE")
        False
        >>> metric_name_higher_is_better("val_NDCG@10")
        True
        >>> metric_name_higher_is_better("custom_unregistered_metric")  # None
    """
    s = _canonicalise_metric_name(name)
    if not s:
        return None
    if s in _KNOWN_METRIC_DIRECTIONS_HIGHER:
        return True
    if s in _KNOWN_METRIC_DIRECTIONS_LOWER:
        return False
    # Fallback: scan the per-target registry for ANY match. A metric
    # registered for one target type usually has the same direction for
    # all (a single metric callable doesn't change meaning across target
    # types). Return None if no match anywhere.
    for _tt_specs in _REGISTRY.values():
        spec = _tt_specs.get(name)
        if spec is not None:
            return spec.higher_is_better
        # Also try canonicalised registry key in case the registry stored
        # the un-prefixed name.
        spec = _tt_specs.get(s)
        if spec is not None:
            return spec.higher_is_better
    return None


# ----------------------------------------------------------------------------
# Built-in registrations — land at import time
# ----------------------------------------------------------------------------


def _register_builtin_multilabel():
    """Register the core multilabel metrics (hamming/subset/jaccard) at import time."""
    from mlframe.metrics.core import (
        hamming_loss, subset_accuracy, jaccard_score_multilabel,
    )

    def _ham(y_true, probs_NK, preds_NK):
        """Hamming loss: fraction of individual (sample, label) entries mispredicted, averaged over the whole indicator matrix."""
        return hamming_loss(y_true, preds_NK)

    def _sub(y_true, probs_NK, preds_NK):
        """Subset accuracy: fraction of samples whose FULL predicted label set exactly matches the true set (all-or-nothing per row)."""
        return subset_accuracy(y_true, preds_NK)

    def _jac(y_true, probs_NK, preds_NK):
        """Sample-averaged Jaccard index: mean over rows of |true ∩ pred| / |true ∪ pred| between the true and predicted label sets."""
        return jaccard_score_multilabel(y_true, preds_NK)

    register_metric(
        TargetTypes.MULTILABEL_CLASSIFICATION, "hamming_loss", _ham,
        higher_is_better=False,
        description="Fraction of labels predicted incorrectly per sample (lower is better).",
    )
    register_metric(
        TargetTypes.MULTILABEL_CLASSIFICATION, "subset_accuracy", _sub,
        higher_is_better=True,
        description="Exact-match accuracy: 1 only when every label is correct.",
    )
    register_metric(
        TargetTypes.MULTILABEL_CLASSIFICATION, "jaccard_samples", _jac,
        higher_is_better=True,
        description="Per-sample Jaccard (intersection over union) averaged across rows.",
    )


_register_builtin_multilabel()


def _register_builtin_classification():
    """Scalar ordinal-agreement + proper-scoring metrics for single-label targets.

    QWK / weighted_kappa need INTEGER predicted labels. The reporter passes the hard predictions
    (preds_NK, equivalent to an argmax of the probabilities for these target types) alongside the
    probability matrix, so both y_true and preds are joint-encoded to contiguous ``0..C-1`` ordinal
    codes here (searchsorted over the sorted label union preserves numeric label order). exploss is
    a binary proper scoring rule reading the positive-class probability column, so it is registered
    for BINARY only; there is no canonical single-vector multiclass form.
    """
    import numpy as _np
    from mlframe.metrics.classification import (
        exploss as _exploss_fn,
        quadratic_weighted_kappa as _qwk_fn,
        weighted_kappa as _wk_fn,
    )

    def _encode_ordinal(y_true, preds):
        """Joint-encode true and predicted labels to contiguous ``0..C-1`` ordinal codes.

        Both arrays are ravelled, then the union of their unique values is sorted and used as the
        code table via ``searchsorted`` so that both y_true and preds land on the SAME code scale
        (required for kappa's weight matrix, which is indexed by absolute code distance). Numeric
        label order is preserved since the union is sorted before assigning codes.
        """
        yt = _np.asarray(y_true).ravel()
        yp = _np.asarray(preds).ravel()
        uniq = _np.unique(_np.concatenate([yt, yp]))
        yt_code = _np.searchsorted(uniq, yt).astype(_np.int64)
        yp_code = _np.searchsorted(uniq, yp).astype(_np.int64)
        return yt_code, yp_code, int(uniq.shape[0])

    def _qwk(y_true, probs_NK, preds_NK):
        """Quadratic weighted kappa: chance-corrected ordinal agreement penalising large-distance misclassifications quadratically more than adjacent-class ones. Operates on jointly-encoded integer codes, not raw labels."""
        yt, yp, nc = _encode_ordinal(y_true, preds_NK)
        return _qwk_fn(yt, yp, n_classes=nc)

    def _wk(y_true, probs_NK, preds_NK):
        """Linear weighted kappa: chance-corrected ordinal agreement penalising misclassifications proportionally to class distance (milder than quadratic weighting). Operates on jointly-encoded integer codes."""
        yt, yp, nc = _encode_ordinal(y_true, preds_NK)
        return _wk_fn(yt, yp, weights="linear", n_classes=nc)

    def _exp(y_true, probs_NK, preds_NK):
        """Exponential proper scoring loss on the positive-class probability column (sklearn column-1 convention); y_true is binarised against the second sorted label before scoring."""
        probs = _np.asarray(probs_NK, dtype=_np.float64)
        pos = probs[:, 1] if probs.ndim == 2 and probs.shape[1] >= 2 else probs.ravel()
        yt = _np.asarray(y_true).ravel()
        uniq = _np.unique(yt)
        # Positive class = sklearn column-1 convention (second sorted label); probs[:, 1] aligns with it.
        pos_label = uniq[1] if uniq.shape[0] > 1 else uniq[-1]
        y_bin = (yt == pos_label).astype(_np.int64)
        return _exploss_fn(y_bin, pos)

    for _tt in (TargetTypes.BINARY_CLASSIFICATION, TargetTypes.MULTICLASS_CLASSIFICATION):
        register_metric(
            _tt, "quadratic_weighted_kappa", _qwk, higher_is_better=True,
            description="Quadratic-weighted Cohen kappa on ordinal integer labels (hard preds); higher is better.",
        )
        register_metric(
            _tt, "weighted_kappa", _wk, higher_is_better=True,
            description="Linear-weighted Cohen kappa on ordinal integer labels (hard preds); higher is better.",
        )
    register_metric(
        TargetTypes.BINARY_CLASSIFICATION, "exploss", _exp, higher_is_better=False,
        description="Exponential proper scoring loss on the positive-class probability; lower is better.",
    )


_register_builtin_classification()


def _register_builtin_multilabel_extras():
    """Wire the 2026-05-28 multilabel additions through the registry."""
    from mlframe.metrics.core import (
        label_ranking_average_precision,
        coverage_error,
        label_ranking_loss,
        one_error,
        multilabel_f1_macro,
        multilabel_f1_micro,
        multilabel_f1_weighted,
    )

    def _lrap(y_true, probs_NK, preds_NK):
        """Label Ranking Average Precision: for each true label, the fraction of higher-or-equal-ranked labels that are also true, averaged over samples and labels."""
        return label_ranking_average_precision(y_true, probs_NK)

    def _cov(y_true, probs_NK, preds_NK):
        """Coverage error: average number of top-ranked-by-score labels one must scan down through to include every true label of a sample."""
        return coverage_error(y_true, probs_NK)

    def _rloss(y_true, probs_NK, preds_NK):
        """Ranking loss: average fraction of (relevant, irrelevant) label pairs per sample that are ordered incorrectly by predicted score."""
        return label_ranking_loss(y_true, probs_NK)

    def _oneerr(y_true, probs_NK, preds_NK):
        """One-error: fraction of samples whose single highest-scored label is not among the true labels."""
        return one_error(y_true, probs_NK)

    def _f1ma(y_true, probs_NK, preds_NK):
        """Macro-averaged multilabel F1: per-label F1 computed independently then averaged with equal weight per label, regardless of label frequency."""
        return multilabel_f1_macro(y_true, preds_NK)

    def _f1mi(y_true, probs_NK, preds_NK):
        """Micro-averaged multilabel F1: true/false positives and negatives pooled across all labels before computing a single F1, so frequent labels dominate."""
        return multilabel_f1_micro(y_true, preds_NK)

    def _f1w(y_true, probs_NK, preds_NK):
        """Support-weighted multilabel F1: per-label F1 averaged with weight proportional to each label's positive count."""
        return multilabel_f1_weighted(y_true, preds_NK)

    register_metric(
        TargetTypes.MULTILABEL_CLASSIFICATION, "lrap", _lrap,
        higher_is_better=True,
        description="Label Ranking Average Precision: precision at each true label's rank, averaged.",
    )
    register_metric(
        TargetTypes.MULTILABEL_CLASSIFICATION, "coverage_error", _cov,
        higher_is_better=False,
        description="Avg rank the model must scan down to cover every true label.",
    )
    register_metric(
        TargetTypes.MULTILABEL_CLASSIFICATION, "ranking_loss", _rloss,
        higher_is_better=False,
        description="Avg fraction of incorrectly-ordered (true, false) label pairs per sample.",
    )
    register_metric(
        TargetTypes.MULTILABEL_CLASSIFICATION, "one_error", _oneerr,
        higher_is_better=False,
        description="Fraction of samples whose argmax-scored label is not in the true label set.",
    )
    register_metric(
        TargetTypes.MULTILABEL_CLASSIFICATION, "f1_macro", _f1ma,
        higher_is_better=True,
        description="Mean per-label F1 (equal weight per label).",
    )
    register_metric(
        TargetTypes.MULTILABEL_CLASSIFICATION, "f1_micro", _f1mi,
        higher_is_better=True,
        description="Pooled-counts F1 across labels.",
    )
    register_metric(
        TargetTypes.MULTILABEL_CLASSIFICATION, "f1_weighted", _f1w,
        higher_is_better=True,
        description="Per-label F1 weighted by positive-support per label.",
    )

    # 2026-05-28 follow-up: per-label AUC aggregations.
    from mlframe.metrics.core import (
        multilabel_auc_macro,
        multilabel_auc_weighted,
    )

    def _auc_ma(y_true, probs_NK, preds_NK):
        """Macro-averaged per-label ROC AUC: AUC computed independently per label column, then averaged with equal weight per label."""
        return multilabel_auc_macro(y_true, probs_NK)

    def _auc_wt(y_true, probs_NK, preds_NK):
        """Support-weighted per-label ROC AUC: per-label AUC averaged with weight proportional to each label's positive support."""
        return multilabel_auc_weighted(y_true, probs_NK)

    register_metric(
        TargetTypes.MULTILABEL_CLASSIFICATION, "auc_macro", _auc_ma,
        higher_is_better=True,
        description="Mean per-label ROC AUC (sklearn average='macro').",
    )
    register_metric(
        TargetTypes.MULTILABEL_CLASSIFICATION, "auc_weighted", _auc_wt,
        higher_is_better=True,
        description="Per-label ROC AUC weighted by positive support.",
    )


_register_builtin_multilabel_extras()


def _register_builtin_multi_target_regression():
    """F-34 (2026-05-31): per-target + aggregated metrics for
    MULTI_TARGET_REGRESSION (y_true and preds both shape (N, K)).

    macro = mean across K target columns (equal weight per target).
    micro = pooled across (N * K) samples (uniform across elements;
            scale-sensitive when target columns have different scales).
    max / min = worst-case per-target value (caller wants to know the
                weakest target rather than the average).
    """
    import numpy as _np
    from mlframe.metrics.core import (
        fast_mean_absolute_error as _mae_fn,
        fast_mean_squared_error as _mse_fn,
        fast_r2_score as _r2_fn,
    )

    def _coerce_nk(y_true, preds):
        """Coerce both args to 2-D (N, K). Tolerant of (N,) -> (N, 1), and of a
        flattened (N*K,) preds vector -> (N, K) when its element count matches a
        2-D ``y_true``. Multi-target predictions sometimes reach the reporter
        C-order-raveled to (N*K,); pre-fix this hit the ``pr.reshape(-1, 1)``
        fallback -> (N*K, 1), so sklearn saw inconsistent samples [N, N*K] and
        every MTR metric (rmse_macro / mae_macro / r2_macro / ...) was omitted from
        the report. reshape(yt.shape) is the exact inverse of the C-order ravel, so
        the per-row K-vectors are recovered (verified bit-exact vs the (N, K) path)."""
        yt = _np.asarray(y_true)
        pr = _np.asarray(preds)
        if yt.ndim == 1:
            yt = yt.reshape(-1, 1)
        if pr.ndim == 1 and yt.ndim == 2 and yt.shape[1] > 1 and pr.size == yt.size:
            pr = pr.reshape(yt.shape)
        elif pr.ndim == 1:
            pr = pr.reshape(-1, 1)
        return yt, pr

    def _rmse_macro(y_true, probs_NK, preds_NK):
        """Mean of per-target-column RMSE (equal weight per target, regardless of each column's scale)."""
        yt, pr = _coerce_nk(y_true, preds_NK)
        per_col = _np.sqrt(_mse_fn(yt, pr, multioutput="raw_values"))
        return float(per_col.mean())

    def _rmse_micro(y_true, probs_NK, preds_NK):
        """RMSE pooled across all (N*K) elements as one flat vector; scale-sensitive since columns with larger magnitude dominate the pooled squared error."""
        yt, pr = _coerce_nk(y_true, preds_NK)
        return float(_np.sqrt(_mse_fn(yt.ravel(), pr.ravel())))

    def _rmse_max(y_true, probs_NK, preds_NK):
        """Worst-case per-target-column RMSE; surfaces the single weakest target rather than averaging it away."""
        yt, pr = _coerce_nk(y_true, preds_NK)
        per_col = _np.sqrt(_mse_fn(yt, pr, multioutput="raw_values"))
        return float(per_col.max())

    def _mae_macro(y_true, probs_NK, preds_NK):
        """Mean of per-target-column MAE (equal weight per target)."""
        yt, pr = _coerce_nk(y_true, preds_NK)
        per_col = _mae_fn(yt, pr, multioutput="raw_values")
        return float(per_col.mean())

    def _mae_max(y_true, probs_NK, preds_NK):
        """Worst-case per-target-column MAE; surfaces the single weakest target rather than averaging it away."""
        yt, pr = _coerce_nk(y_true, preds_NK)
        per_col = _mae_fn(yt, pr, multioutput="raw_values")
        return float(per_col.max())

    def _r2_macro(y_true, probs_NK, preds_NK):
        """Mean of per-target-column R^2 (sklearn ``multioutput='uniform_average'``): equal weight per target regardless of its variance."""
        yt, pr = _coerce_nk(y_true, preds_NK)
        return float(_r2_fn(yt, pr, multioutput="uniform_average"))

    def _r2_min(y_true, probs_NK, preds_NK):
        """Worst-case per-target-column R^2; exposes the laggard target that a macro average would mask."""
        yt, pr = _coerce_nk(y_true, preds_NK)
        per_col = _r2_fn(yt, pr, multioutput="raw_values")
        return float(per_col.min())

    register_metric(
        TargetTypes.MULTI_TARGET_REGRESSION, "rmse_macro", _rmse_macro,
        higher_is_better=False,
        description="Mean per-target RMSE (equal weight per target column).",
    )
    register_metric(
        TargetTypes.MULTI_TARGET_REGRESSION, "rmse_micro", _rmse_micro,
        higher_is_better=False,
        description="Pooled RMSE across all (N*K) samples; scale-sensitive.",
    )
    register_metric(
        TargetTypes.MULTI_TARGET_REGRESSION, "rmse_max", _rmse_max,
        higher_is_better=False,
        description="Worst-case per-target RMSE -- catches degenerate targets.",
    )
    register_metric(
        TargetTypes.MULTI_TARGET_REGRESSION, "mae_macro", _mae_macro,
        higher_is_better=False,
        description="Mean per-target MAE.",
    )
    register_metric(
        TargetTypes.MULTI_TARGET_REGRESSION, "mae_max", _mae_max,
        higher_is_better=False,
        description="Worst-case per-target MAE.",
    )
    register_metric(
        TargetTypes.MULTI_TARGET_REGRESSION, "r2_macro", _r2_macro,
        higher_is_better=True,
        description="Mean per-target R^2 (sklearn multioutput='uniform_average').",
    )
    register_metric(
        TargetTypes.MULTI_TARGET_REGRESSION, "r2_min", _r2_min,
        higher_is_better=True,
        description="Worst-case per-target R^2 -- exposes the laggard target.",
    )


_register_builtin_multi_target_regression()
