"""Leaf helper for the ensemble winner-selection rule.

Holds ``_ENSEMBLE_RANK_METRIC_CANDIDATES`` plus ``_read_ensemble_metric`` and
``_choose_ensemble_flavour``. Lives in its own module so callers in
``_phase_train_one_target`` and ``_phase_train_one_target_ensembling`` can
both top-level import without forming an import cycle through the parent
``_phase_train_one_target`` (which historically housed these helpers and
required a lazy in-function import from the ensembling sibling).

Parent ``_phase_train_one_target`` re-exports the names at its bottom so
historical ``from ._phase_train_one_target import _choose_ensemble_flavour``
imports keep resolving.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("mlframe.training.core._phase_train_one_target")


# Per-(split, metric, direction) candidates probed in order to rank ensemble flavours. ``oof.*`` is the only honest
# selection surface (cross_val_predict held-out signal, never used for ES); ``val.*`` is the back-compat fallback for
# single-fold suites that did not stamp OOF; ``test.*`` is the last resort (selecting on it biases downstream test
# metrics optimistic -- emits a one-time WARN).
#
# The list is TASK-AWARE because the report path stamps DIFFERENT metric keys per task: classification ensembles carry
# ``ice`` / ``roc_auc`` / ``pr_auc`` / ``brier_loss`` (nested under class 1 for binary), regression carries ``RMSE`` /
# ``MAE`` / ``R2`` -- and NEITHER stamps the legacy ``integral_error`` key. Probing only ``integral_error`` / ``rmse``
# silently matched nothing on a classification run and fell through to the deterministic first-flavour ('arithm')
# fallback every time, ignoring the genuinely best-calibrated flavour. Within each split the calibration metric is
# tried first (most decision-relevant), then ranking AUC, then the regression losses; the first family any candidate
# exposes wins, so a single list serves both tasks without a task flag. ``integral_error`` / lowercase ``rmse`` are
# retained for back-compat with callers / fixtures that stamp those exact keys (``_read_ensemble_metric`` matches keys
# case-insensitively so production ``RMSE`` resolves the ``rmse`` candidate).
_CLASSIFICATION_METRICS = (
    ("ice", "lower"),
    ("integral_error", "lower"),
    ("brier_loss", "lower"),
    ("roc_auc", "higher"),
    ("pr_auc", "higher"),
    ("log_loss", "lower"),
)
_REGRESSION_METRICS = (
    ("rmse", "lower"),
    ("mae", "lower"),
    ("r2", "higher"),
)


def _build_rank_candidates():
    """Expand (split x metric-family) into the flat probe order: oof first, then val, then test."""
    _metrics = _CLASSIFICATION_METRICS + _REGRESSION_METRICS
    return tuple(
        (split, metric, direction)
        for split in ("oof", "val", "test")
        for metric, direction in _metrics
    )


_ENSEMBLE_RANK_METRIC_CANDIDATES = _build_rank_candidates()


def _read_ensemble_metric(ens_result, split: str, metric: str):
    """Read ``ens_result.metrics[split][metric]`` (or nested int-keyed dict 1) returning float or None.

    The metric layout produced by ``train_and_evaluate_model`` is ``model.metrics[split]`` where the value is either a
    flat dict (regression) or a ``{1: {...}}`` class-indexed nested dict (binary / multiclass classification). For
    classifier metrics nested under class 1 the read drills one level; otherwise the flat lookup wins. The lookup is
    case-insensitive so the production regression key ``RMSE`` resolves the lowercase ``rmse`` candidate. Any access /
    type error returns ``None`` so the chooser silently skips the flavour.
    """
    try:
        _m = getattr(ens_result, "metrics", None)
        if not isinstance(_m, dict):
            return None
        _split = _m.get(split)
        if not isinstance(_split, dict):
            return None
        _val = _lookup_metric_ci(_split, metric)
        if _val is None and 1 in _split and isinstance(_split[1], dict):
            _val = _lookup_metric_ci(_split[1], metric)
        if _val is None:
            return None
        _f = float(_val)
        if not np.isfinite(_f):
            return None
        return _f
    except Exception:
        return None


def _lookup_metric_ci(_d: dict, metric: str):
    """Return ``_d[metric]`` with an exact-key fast path then a case-insensitive fallback (production ``RMSE`` vs candidate ``rmse``)."""
    _v = _d.get(metric)
    if _v is not None:
        return _v
    _lower = metric.lower()
    for _k, _kv in _d.items():
        if isinstance(_k, str) and _k.lower() == _lower:
            return _kv
    return None


def _choose_ensemble_flavour(ensembles_dict: dict) -> str | None:
    """Pick the winning ensemble flavour key from ``score_ensemble``'s return dict.

    ``score_ensemble`` returns ``{flavour_name: ens_result}`` for every candidate it evaluated; the
    suite has no native "winner" concept so we rank by the first metric family any candidate exposes,
    in the task-aware ``_ENSEMBLE_RANK_METRIC_CANDIDATES`` probe order (classification calibration /
    AUC keys then regression losses, each across oof -> val -> test), respecting per-metric direction
    (AUC / R2 higher-is-better, the rest lower-is-better). ``" conf"``-suffixed entries (confident-
    subset variants of each flavour) are skipped here -- they reuse the parent flavour's preds on a
    different subset and aren't independent candidates. ``_diversity`` is a side-channel report stamped
    by ``score_ensemble`` rather than an ensemble; skip it too.

    Return values:
      - ``None`` only when ``ensembles_dict`` is empty / not-a-dict / contains zero non-skip candidates.
      - First-emitted flavour name (deterministic via ``ensembling_methods`` insertion order) when at
        least one candidate exists but NONE expose any of the canonical ranking metrics. A WARN log
        line is emitted in this fallback path so an operator grepping the suite log for
        ``no candidate exposed`` can distinguish a fallback win from a metric-driven win.
      - Otherwise the flavour name whose ranking metric scored best per
        ``_ENSEMBLE_RANK_METRIC_CANDIDATES``.
    """
    if not isinstance(ensembles_dict, dict) or not ensembles_dict:
        return None
    _candidates = {
        k: v for k, v in ensembles_dict.items()
        if isinstance(k, str) and not k.endswith(" conf") and not k.startswith("_")
    }
    if not _candidates:
        return None
    for _split, _metric, _direction in _ENSEMBLE_RANK_METRIC_CANDIDATES:
        _scored = [
            (k, _read_ensemble_metric(v, _split, _metric))
            for k, v in _candidates.items()
        ]
        _scored = [(k, s) for k, s in _scored if s is not None]
        if not _scored:
            continue
        if _direction == "lower":
            _scored.sort(key=lambda kv: (kv[1], kv[0]))
        else:
            _scored.sort(key=lambda kv: (-kv[1], kv[0]))
        # The module-top comment promises a "one-time WARN at first use" of the test.* fallback because using the honest test split for model selection biases every subsequent test-set metric optimistic. Surface that WARN so production runs reaching this branch are visible in the suite log.
        if _split == "test":
            logger.warning(
                "[_choose_ensemble_flavour] resolved winner %r via test.%s (oof.* and val.* "
                "metrics absent). Using test for selection converts it into a model-selection "
                "surface and biases downstream test-set metrics; stamp OOF on every candidate "
                "in production callers.",
                _scored[0][0], _metric,
            )
        return _scored[0][0]
    _fallback = next(iter(_candidates.keys()))
    _probed_metrics = sorted({m for _, m, _ in _ENSEMBLE_RANK_METRIC_CANDIDATES})
    logger.warning(
        "[_choose_ensemble_flavour] no candidate exposed any canonical ranking metric (probed %s across "
        "oof/val/test); falling back to first-emitted flavour %r (deterministic via dict-insertion / "
        "ensembling_methods order).",
        _probed_metrics, _fallback,
    )
    return _fallback
