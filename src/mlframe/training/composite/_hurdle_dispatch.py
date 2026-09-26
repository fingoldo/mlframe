"""Train a :class:`~mlframe.training.composite.hurdle.HurdleRegressor` alongside the suite's models on zero-inflated targets.

A zero-inflated amount -- most rows exactly 0 because the event did not happen, the rest a skewed positive magnitude -- is
the target shape the default regressors handle worst. A single model fit on ``log1p(y)`` puts most of its mass on one
point and the inverse squeezes the rest back into a narrow band: on the synthetic hurdle case the plain log1p model
predicted with 1.8% of the target's spread, matching a production run's collapsed ``target_total_charge`` models, while the
hurdle kept 12.1% (``tests/training/composite/test_biz_val_hurdle_regressor.py``). Against a plain LightGBM on raw y, which
does not collapse, the gain is smaller but consistent: better rank agreement on every seed tried, no negative amounts
(the plain model predicted down to -177), equal RMSE (``test_biz_val_hurdle_suite.py``). The point-mass gate in composite
discovery already refuses the curved y-transforms on such a target; this supplies the model that fits it instead.

Detection reads the target itself, not the analyzer's pathology strings: the atom must be the modal value, hold at least
``ZERO_INFLATION_FRACTION_THRESHOLD`` of the train rows, and sit at the target's minimum (a hurdle's "no event" is the
floor of the amount, never an interior value). One estimator is injected per distinct atom value and restricted to the
targets that carry it, so it never trains on a target it was not chosen for.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from mlframe.training.composite.discovery.shared import MIN_ROWS_FOR_POINT_MASS_CHECK, POINT_MASS_FRACTION_THRESHOLD

logger = logging.getLogger(__name__)

ZERO_INFLATION_FRACTION_THRESHOLD: float = POINT_MASS_FRACTION_THRESHOLD
"""Share of train rows on the atom above which a hurdle model is added. Same bar as the point-mass gate that refuses the
curved y-transforms, so the target that loses those transforms is exactly the target that gains the hurdle. Must stay at
or above 0.5: ``zero_inflated_atom`` relies on that to skip the modal-value search."""
if ZERO_INFLATION_FRACTION_THRESHOLD < 0.5:  # pragma: no cover - guards a future edit of the shared constant
    raise ValueError("ZERO_INFLATION_FRACTION_THRESHOLD must be >= 0.5 for zero_inflated_atom's minimum-only test")


def zero_inflated_atom(y: Any) -> Optional[float]:
    """The "no event" value of a zero-inflated target, or None when ``y`` is not one.

    ``y`` qualifies when its most frequent exact value covers at least ``ZERO_INFLATION_FRACTION_THRESHOLD`` of the finite
    rows, is also its minimum, and is not the only value (a constant target has no magnitude to model).
    """
    arr = np.asarray(y, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size < MIN_ROWS_FOR_POINT_MASS_CHECK:
        return None
    # The atom must be the minimum, and with a threshold of at least one half a value holding that share is necessarily
    # the modal one, so "the minimum's share clears the bar" is the whole test: one O(n) pass, no sort, no sampling.
    floor = float(arr.min())
    frac = float(np.count_nonzero(arr == floor)) / arr.size
    if frac < ZERO_INFLATION_FRACTION_THRESHOLD or frac >= 1.0:
        return None
    return floor


def _default_hurdle_halves() -> tuple[Any, Any]:
    """``(classifier, regressor)`` prototypes: LightGBM when installed (pandas categoricals, speed), else sklearn HGB."""
    try:
        from lightgbm import LGBMClassifier

        from ._estimator_dispatch import _default_base_estimator

        # n_jobs=-1: the LGBM default resolves physical cores through a Windows subprocess probe (see _default_base_estimator).
        return LGBMClassifier(n_estimators=200, num_leaves=31, verbose=-1, random_state=0, n_jobs=-1), _default_base_estimator()
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        return HistGradientBoostingClassifier(max_iter=200, random_state=0), HistGradientBoostingRegressor(max_iter=200, random_state=0)


def zero_inflated_regression_targets(target_by_type: Any, train_idx: Any) -> dict[float, list[str]]:
    """``{atom: [target names]}`` for every regression target that is zero-inflated on its train rows."""
    from .._configs_base import TargetTypes

    by_atom: dict[float, list[str]] = {}
    for name, y in ((target_by_type or {}).get(TargetTypes.REGRESSION) or {}).items():
        y_full = np.asarray(y.to_numpy() if hasattr(y, "to_numpy") else y).reshape(-1)
        try:
            y_train = y_full[np.asarray(train_idx)] if train_idx is not None else y_full
        except (IndexError, TypeError):
            y_train = y_full
        atom = zero_inflated_atom(y_train)
        if atom is not None:
            by_atom.setdefault(atom, []).append(str(name))
    return by_atom


def maybe_inject_hurdle_for_zero_inflated(
    ctx: Any, metadata: dict, mlframe_models: list, target_by_type: Any, train_idx: Any, behavior_config: Any,
) -> list:
    """Append one ``("hurdle", HurdleRegressor)`` entry per zero-inflation atom, restricted to the targets that carry it.

    Gated by ``behavior_config.hurdle_for_zero_inflated`` (default ON). Returns ``mlframe_models`` unchanged when the flag
    is off or no regression target is zero-inflated.
    """
    if not getattr(behavior_config, "hurdle_for_zero_inflated", True):
        return mlframe_models
    by_atom = zero_inflated_regression_targets(target_by_type, train_idx)
    if not by_atom:
        return mlframe_models

    from ._estimator_dispatch import register_injected_models
    from .hurdle import HurdleRegressor

    new_models = list(mlframe_models)
    for atom, names in by_atom.items():
        clf, reg = _default_hurdle_halves()
        est = HurdleRegressor(classifier=clf, regressor=reg, zero_value=atom, random_state=0)
        est._mlframe_only_targets = frozenset(names)
        label = "hurdle" if len(by_atom) == 1 else f"hurdle_at_{atom:g}"
        new_models.append((label, est))
        metadata.setdefault("hurdle_for_zero_inflated", {})[label] = {"zero_value": atom, "targets": sorted(names)}
        logger.info(
            "[hurdle] %d regression target(s) sit on a point mass at %g for at least %.0f%% of train rows (%s); training a "
            "HurdleRegressor (classifier for the event, regressor for its magnitude) alongside the requested models for them.",
            len(names), atom, 100 * ZERO_INFLATION_FRACTION_THRESHOLD, ", ".join(sorted(names)[:8]),
        )
    register_injected_models(ctx, new_models)
    return new_models
