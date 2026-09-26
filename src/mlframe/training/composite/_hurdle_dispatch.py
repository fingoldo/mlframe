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
or above 0.5: ``zero_inflation_verdict`` relies on that to take the median as the only candidate atom."""
if ZERO_INFLATION_FRACTION_THRESHOLD < 0.5:  # pragma: no cover - guards a future edit of the shared constant
    raise ValueError("ZERO_INFLATION_FRACTION_THRESHOLD must be >= 0.5 for zero_inflation_verdict's median-as-atom test")


BELOW_ATOM_TOLERANCE: float = 0.001
"""Largest share of rows below the atom that still counts as zero-inflated. A production ``total_charge`` was 74% zeros
with a few refunds at -2.17: requiring the atom to be the exact minimum declined the hurdle over a handful of rows, and
said nothing. Those rows are treated as "no event" (``HurdleRegressor(below_zero="no_event")``)."""


def zero_inflation_verdict(y: Any) -> "tuple[Optional[float], int, Optional[str]]":
    """``(atom, n_below, reason)``: the "no event" value of a zero-inflated target and how many rows lie below it, or
    ``(None, n_below, reason)`` with why a target with a point mass does not qualify (``reason`` None: no point mass).

    A target qualifies when one exact value covers at least ``ZERO_INFLATION_FRACTION_THRESHOLD`` of the finite rows,
    at most ``BELOW_ATOM_TOLERANCE`` of them lie below it, and it is not the only value. With a threshold of at least
    one half, a value holding that share is necessarily the median, so one pass finds it: no sort, no sampling.
    """
    arr = np.asarray(y, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size < MIN_ROWS_FOR_POINT_MASS_CHECK:
        return None, 0, None
    atom = float(np.median(arr))
    frac = float(np.count_nonzero(arr == atom)) / arr.size
    if frac < ZERO_INFLATION_FRACTION_THRESHOLD or frac >= 1.0:
        return None, 0, None
    n_below = int(np.count_nonzero(arr < atom))
    n_above = int(np.count_nonzero(arr > atom))
    if n_above == 0:
        return None, n_below, (
            f"{frac:.0%} of rows sit at {atom:g}, which is the target's MAXIMUM: a hurdle models an event above a floor, so "
            f"this is not one. A constant filled in for \"no event\" looks exactly like this; model the event and the "
            f"value given the event separately instead."
        )
    if n_below > BELOW_ATOM_TOLERANCE * arr.size:
        return None, n_below, (
            f"{frac:.0%} of rows sit at {atom:g}, but {n_below:_} row(s) ({n_below / arr.size:.2%}, minimum {float(arr.min()):g}) "
            f"lie below it, more than the {BELOW_ATOM_TOLERANCE:.1%} a hurdle's \"no event\" floor tolerates."
        )
    return atom, n_below, None


def zero_inflated_atom(y: Any) -> Optional[float]:
    """The "no event" value of a zero-inflated target, or None when ``y`` is not one (see :func:`zero_inflation_verdict`)."""
    return zero_inflation_verdict(y)[0]


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
    return _scan_zero_inflation(target_by_type, train_idx)[0]


def _scan_zero_inflation(target_by_type: Any, train_idx: Any) -> "tuple[dict[float, list[str]], dict[float, dict[str, int]]]":
    """``({atom: [target names]}, {atom: {target: rows below the atom}})``; a point-mass target that does not qualify is
    logged with the reason."""
    from .._configs_base import TargetTypes

    by_atom: dict[float, list[str]] = {}
    below: dict[float, dict[str, int]] = {}
    for name, y in ((target_by_type or {}).get(TargetTypes.REGRESSION) or {}).items():
        y_full = np.asarray(y.to_numpy() if hasattr(y, "to_numpy") else y).reshape(-1)
        try:
            y_train = y_full[np.asarray(train_idx)] if train_idx is not None else y_full
        except (IndexError, TypeError):
            y_train = y_full
        atom, n_below, reason = zero_inflation_verdict(y_train)
        if atom is not None:
            by_atom.setdefault(atom, []).append(str(name))
            if n_below:
                below.setdefault(atom, {})[str(name)] = n_below
        elif reason is not None:
            logger.warning("[hurdle] no HurdleRegressor for %s: %s", name, reason)
    return by_atom, below


def maybe_inject_hurdle_for_zero_inflated(
    ctx: Any, metadata: dict, mlframe_models: list, target_by_type: Any, train_idx: Any, behavior_config: Any,
) -> list:
    """Append one ``("hurdle", HurdleRegressor)`` entry per zero-inflation atom, restricted to the targets that carry it.

    Gated by ``behavior_config.hurdle_for_zero_inflated`` (default ON). Returns ``mlframe_models`` unchanged when the flag
    is off or no regression target is zero-inflated.
    """
    if not getattr(behavior_config, "hurdle_for_zero_inflated", True):
        return mlframe_models
    by_atom, below = _scan_zero_inflation(target_by_type, train_idx)
    if not by_atom:
        return mlframe_models

    from ._estimator_dispatch import register_injected_models
    from .hurdle import HurdleRegressor

    new_models = list(mlframe_models)
    for atom, names in by_atom.items():
        clf, reg = _default_hurdle_halves()
        below_rows = below.get(atom, {})
        est = HurdleRegressor(classifier=clf, regressor=reg, zero_value=atom, random_state=0, below_zero="no_event" if below_rows else "event")
        if below_rows:
            logger.warning(
                "[hurdle] %s: %s row(s) lie below the point mass at %g; the hurdle counts them as \"no event\". Check the source "
                "if they should not exist (negative amounts are usually refunds or corrections).",
                ", ".join(sorted(below_rows)), ", ".join(f"{v:_}" for _, v in sorted(below_rows.items())), atom,
            )
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
