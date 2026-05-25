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


# Candidate metric paths probed (in order) to rank ensemble flavours. ``oof.*`` is the only honest
# selection surface (cross_val_predict held-out signal, never used for ES); val.* is the back-compat
# fallback for single-fold suites that did not stamp OOF. ``integral_error`` is the canonical
# calibration metric for classifiers; ``rmse`` is the regression fallback.
#
# ``("test", ...)`` candidates DELIBERATELY come last and emit a one-time WARN at first use --
# selecting on the honest test split converts it into a model-selection surface that biases every
# subsequent test-set metric optimistic. Tests are kept ONLY as a last-resort fallback for unit
# fixtures where oof / val are both absent; production callers should always stamp OOF.
_ENSEMBLE_RANK_METRIC_CANDIDATES = (
    ("oof", "integral_error", "lower"),
    ("oof", "rmse", "lower"),
    ("val", "integral_error", "lower"),
    ("val", "rmse", "lower"),
    ("test", "integral_error", "lower"),
    ("test", "rmse", "lower"),
)


def _read_ensemble_metric(ens_result, split: str, metric: str):
    """Read ``ens_result.metrics[split][metric]`` (or nested int-keyed dict 1) returning float or None.

    The metric layout produced by ``train_and_evaluate_model`` is ``model.metrics[split]`` where the
    value is either a flat dict or a ``{1: {...}}`` class-indexed nested dict (binary / multiclass).
    For classifier metrics nested under class 1 the read drills one level; otherwise the flat lookup
    wins. Any access / type error returns ``None`` so the chooser silently skips the flavour.
    """
    try:
        _m = getattr(ens_result, "metrics", None)
        if not isinstance(_m, dict):
            return None
        _split = _m.get(split)
        if not isinstance(_split, dict):
            return None
        _val = _split.get(metric)
        if _val is None and 1 in _split and isinstance(_split[1], dict):
            _val = _split[1].get(metric)
        if _val is None:
            return None
        _f = float(_val)
        if not np.isfinite(_f):
            return None
        return _f
    except Exception:
        return None


def _choose_ensemble_flavour(ensembles_dict: dict) -> str | None:
    """Pick the winning ensemble flavour key from ``score_ensemble``'s return dict.

    ``score_ensemble`` returns ``{flavour_name: ens_result}`` for every candidate it evaluated; the
    suite has no native "winner" concept so we apply the same selection rule as ``compare_ensembles``
    (oof.integral_error / rmse ascending). ``" conf"``-suffixed entries (confident-subset variants of
    each flavour) are skipped here -- they reuse the parent flavour's preds on a different subset and
    aren't independent candidates. ``_diversity`` is a side-channel report stamped by
    ``score_ensemble`` rather than an ensemble; skip it too.

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
    logger.warning(
        "[_choose_ensemble_flavour] no candidate exposed any of the canonical ranking metrics %s; "
        "falling back to first-emitted flavour %r (deterministic via dict-insertion / "
        "ensembling_methods order).",
        [(s, m) for s, m, _ in _ENSEMBLE_RANK_METRIC_CANDIDATES], _fallback,
    )
    return _fallback
