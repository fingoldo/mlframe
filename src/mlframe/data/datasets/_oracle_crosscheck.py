"""An outside opinion on the oracle's exact mutual information, from a library nobody here wrote.

`assert_estimators_disjoint` keeps the oracle's estimator families apart from the arms' backends, which
stops the rank-correlation metric from measuring self-agreement. It does nothing at all about the other
failure: the oracle's own closed form being wrong. Both the value and the test of that value are written
here, by the same hand, against the same understanding of the same formula -- and a shared misunderstanding
produces a test that passes.

So the exact path is checked against `dit`, a third-party information-theory package with its own joint
distribution object and its own implementation of `I(X;Y)`. The comparison is made on the exact joint LAW
rather than on a sample: for a discrete column the generator knows `P(x, y)` exactly -- it is the average
of `true_prob` within each level -- so both sides compute the same quantity with no estimation error
between them, and a disagreement is a disagreement about the mathematics rather than about binning.

`dit` is optional. It is imported inside the function, and its absence raises with the install line rather
than degrading to a silently skipped check: a cross-check that quietly does not run is worse than none,
because the manifest would still say the oracle was cross-checked.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["CrossCheck", "exact_joint", "mutual_information_via_dit", "crosscheck_exact_mi", "assert_oracle_agrees_with_dit", "crosscheck_summary", "reference_crosscheck", "DEFAULT_TOLERANCE"]

#: How far the two implementations may differ before the cross-check fails. Both compute the same exact
#: quantity, so the only honest tolerance is floating-point accumulation over the joint's cells.
DEFAULT_TOLERANCE = 1e-9

#: The equal-width grid the oracle evaluates its exact identity on. Mirrored here because it is what makes
#: one level per bin possible, and therefore what sets the level cap this check can run under.
_ORACLE_BINS = 64


@dataclass(frozen=True)
class CrossCheck:
    """One comparison of the oracle's exact MI against a third party's, on one column's exact joint law."""

    column: str
    ours: float
    theirs: float
    tolerance: float
    n_levels: int

    @property
    def difference(self) -> float:
        """The absolute gap between the two implementations, in nats."""
        return float(abs(self.ours - self.theirs))

    @property
    def agrees(self) -> bool:
        """Whether the gap is inside the tolerance."""
        return bool(self.difference <= self.tolerance)


def exact_joint(codes: np.ndarray, true_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return the exact joint `P(level, y)` of one discrete column with a binary target, and its levels.

    Args:
        codes: The column's level per row. Any hashable-by-value dtype numpy can take `np.unique` of.
        true_prob: `P(y = 1 | row)` under the law the labels were drawn from, one per row.

    Returns:
        A `(n_levels, 2)` joint summing to one, and the levels in the order its rows are in. The joint is
        conditional on the REALISED column, which is the comparator the benchmark scores against -- the
        population version would be a different number and is not what any arm ever sees.

    Raises:
        ValueError: When the two arrays disagree in length, or a probability sits outside `[0, 1]`.
    """
    codes = np.asarray(codes)
    probabilities = np.asarray(true_prob, dtype=np.float64)
    if codes.shape[0] != probabilities.shape[0]:
        raise ValueError(f"codes and true_prob must describe the same rows; got {codes.shape[0]} and {probabilities.shape[0]}")
    if probabilities.size and (float(probabilities.min()) < 0.0 or float(probabilities.max()) > 1.0):
        raise ValueError("true_prob must be a probability in [0, 1]; a link that escaped its squashing function is the usual cause")

    levels = np.unique(codes)
    n = float(codes.shape[0])
    joint = np.zeros((levels.size, 2), dtype=np.float64)
    for index, level in enumerate(levels):
        mask = codes == level
        joint[index, 1] = float(probabilities[mask].sum()) / n
        joint[index, 0] = float(mask.sum()) / n - joint[index, 1]
    return joint, levels


def _mutual_information(joint: np.ndarray) -> float:
    """Return `I(X;Y)` in nats from an exact joint, with zero-probability cells contributing nothing.

    Used only to check the two-argument form of the comparison itself; the quantity actually cross-checked
    is the oracle's own `_exact_mi_independent`, which is what the benchmark publishes.
    """
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    outer = px * py
    mask = (joint > 0.0) & (outer > 0.0)
    return float((joint[mask] * np.log(joint[mask] / outer[mask])).sum())


def mutual_information_via_dit(joint: np.ndarray) -> float:
    """Return `I(X;Y)` in nats for the same joint, computed by `dit` rather than by this package.

    Args:
        joint: A `(n_levels, 2)` exact joint, as `exact_joint` returns.

    Returns:
        The mutual information in nats. `dit` works in bits by default, so the distribution is built with
        base `e` explicitly rather than its result being rescaled afterwards -- a rescale is one more line
        of ours between the two implementations, which is exactly what this check is trying to avoid.

    Raises:
        ImportError: When `dit` is not installed, naming the install. A cross-check that skips itself would
            leave the manifest claiming a verification that never ran.
    """
    try:
        from dit import Distribution
        from dit.shannon import mutual_information
    except ImportError as exc:  # pragma: no cover - exercised by the environment, not by a test
        raise ImportError("the oracle cross-check needs the third-party `dit` package: pip install 'mlframe[stats]'") from exc

    # An outcome is a PAIR, one symbol per random variable, because `mutual_information` indexes variables
    # by position -- a flat label would make the joint one variable and the answer zero.
    outcomes: List[Tuple[str, str]] = []
    probabilities: List[float] = []
    for level in range(joint.shape[0]):
        for label in range(joint.shape[1]):
            mass = float(joint[level, label])
            if mass > 0.0:
                outcomes.append((f"x{level}", f"y{label}"))
                probabilities.append(mass)
    distribution = Distribution(outcomes, probabilities)
    distribution.set_base("e")
    return float(mutual_information(distribution, [0], [1]))


def crosscheck_exact_mi(
    columns: Dict[str, np.ndarray],
    true_prob: np.ndarray,
    tolerance: float = DEFAULT_TOLERANCE,
    max_levels: int = _ORACLE_BINS + 1,
) -> List[CrossCheck]:
    """Compare this package's exact MI with `dit`'s, column by column, on the exact joint law.

    Args:
        columns: Columns by name, each carrying consecutive integer level codes. The oracle bins on an
            equal-width grid of `_ORACLE_BINS` cells, which separates consecutive integers one per bin
            exactly while the level count stays at or below `max_levels` -- above it two levels would
            share a bin, the two sides would be computing different joints, and the check would report a
            disagreement that is about resolution rather than about the formula.
        true_prob: `P(y = 1 | row)` under the law the labels came from.
        tolerance: Permitted gap in nats.
        max_levels: The level cap above which a column is skipped.

    Returns:
        One result per column that was checked, in the order the columns were given. The `ours` side is the
        oracle's own exact path, not a second implementation written for this check.
    """
    from ._oracle import _exact_mi_independent

    out: List[CrossCheck] = []
    for name, values in columns.items():
        joint, levels = exact_joint(values, true_prob)
        if levels.size > max_levels or levels.size < 2:
            logger.debug("skipping %s in the oracle cross-check: %d levels", name, levels.size)
            continue
        ours = _exact_mi_independent(np.asarray(values, dtype=np.float64), np.asarray(true_prob, dtype=np.float64))
        if ours is None:
            logger.debug("skipping %s in the oracle cross-check: the oracle declined to compute an exact value", name)
            continue
        out.append(
            CrossCheck(
                column=name,
                ours=float(ours),
                theirs=mutual_information_via_dit(joint),
                tolerance=tolerance,
                n_levels=int(levels.size),
            )
        )
    return out


def assert_oracle_agrees_with_dit(columns: Dict[str, np.ndarray], true_prob: np.ndarray, tolerance: float = DEFAULT_TOLERANCE) -> List[CrossCheck]:
    """Run the cross-check and raise on any disagreement, returning the results when they all agree.

    Args:
        columns: Discrete columns by name.
        true_prob: The law the labels came from.
        tolerance: Permitted gap in nats.

    Returns:
        Every comparison that ran, so a caller can record how much was actually checked in the manifest.

    Raises:
        ValueError: On the first column where the two implementations disagree beyond `tolerance`.
    """
    checks = crosscheck_exact_mi(columns, true_prob, tolerance=tolerance)
    bad = [c for c in checks if not c.agrees]
    if bad:
        worst = max(bad, key=lambda c: c.difference)
        raise ValueError(
            f"the oracle's exact mutual information disagrees with dit on {len(bad)} of {len(checks)} columns; "
            f"worst is {worst.column!r} at {worst.ours:.9f} against {worst.theirs:.9f} nats"
        )
    return checks


def crosscheck_summary(checks: List[CrossCheck]) -> Optional[Dict[str, Any]]:
    """Return the manifest record of one cross-check run, or `None` when nothing was checkable.

    `None` rather than an empty record on purpose: a manifest carrying "cross-checked 0 columns" reads as a
    verification, and the absence of one should look like an absence.
    """
    if not checks:
        return None
    return {
        "estimator": "dit.shannon.mutual_information",
        "columns_checked": len(checks),
        "worst_difference_nats": float(max(c.difference for c in checks)),
        "tolerance_nats": float(checks[0].tolerance),
        "all_agree": all(c.agrees for c in checks),
    }


def reference_crosscheck(n_levels: Tuple[int, ...] = (2, 5, 17, 64), n_samples: int = 4000, seed: int = 20260923) -> Dict[str, Any]:
    """Run the cross-check on a fixed discrete probe and return the record a run's manifest carries.

    The probe is built here rather than taken from the scenario library on purpose: it exists to exercise
    the exact identity at several level counts, which no registered bed is shaped for, and pinning it to a
    fixed seed makes the manifest's number comparable between runs.

    Returns:
        A record naming the third-party estimator, how many columns agreed and the worst gap; or one
        stating plainly that the check did NOT run, with the reason, when `dit` is not installed. It never
        returns silence -- a manifest that simply omits the field reads as a run that had no oracle.
    """
    rng = np.random.default_rng(seed)
    columns: Dict[str, np.ndarray] = {}
    probabilities: Optional[np.ndarray] = None
    for levels in n_levels:
        codes = rng.integers(0, levels, size=n_samples)
        columns[f"levels_{levels}"] = codes
        centred = (codes - (levels - 1) / 2.0) / max(levels - 1, 1)
        contribution = 3.6 * centred
        probabilities = contribution if probabilities is None else probabilities + contribution
    if probabilities is None:
        return {"ran": False, "reason": "no level counts were requested"}
    law = 1.0 / (1.0 + np.exp(-(probabilities / len(n_levels) + rng.normal(0.0, 0.3, n_samples))))

    try:
        checks = crosscheck_exact_mi(columns, law)
    except ImportError as exc:
        return {"ran": False, "reason": str(exc)}
    summary = crosscheck_summary(checks)
    if summary is None:
        return {"ran": False, "reason": "no column in the probe was separable at the oracle's resolution"}
    return {"ran": True, **summary}
