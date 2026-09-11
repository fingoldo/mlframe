"""The oracle: what is achievable on a generated dataset, and how much each column knows about the target.

Two quantities, and the difference between how they are computed is the whole design.

**The ceiling is exact, not estimated.** Every corruption in this package is required to state how it moves
``true_prob``, so the probability attached to a generated dataset IS the law its labels came from. The best
achievable Brier score is then ``mean p(1-p)``, the best log-loss ``mean H(p)``, the best accuracy
``mean max(p, 1-p)``, and the best AUC the probability that a random positive outranks a random negative
under that law -- all expectations conditional on the realised feature matrix, with no sampling step
anywhere. The plan's original rule (compute analytically, refuse where no closed form exists) would have
deleted the corrupted families, which are the ones where a ceiling matters most; requiring the update
instead keeps both the families and the exactness.

**Reference MI is an estimate and is labelled as one.** Binning a large sample is biased and variable, so
this is ``mi_reference``, never ``true_mi``. Three things follow. An exact value is reported wherever a
closed form exists -- with independent features and a known ``true_prob``, ``I(X_j; Y) = H(Ybar) -
E_x[H(E[p|x])]``, a one-dimensional integral. Several estimators are reported alongside it, and their
spread is the error bar. And when that spread exceeds the effect anyone would measure with it, the bundle
says ``unreliable`` rather than handing back a number nobody can defend.

The estimator families here are deliberately disjoint from the MI backends the benchmark's arms use.
Scoring an arm's ranking against an oracle built from the arm's own estimator measures agreement with
itself; :func:`assert_estimators_disjoint` makes that a loud failure rather than a quiet one.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from mlframe.data.datasets.ground_truth import Ceiling, MIBundle, MIEstimate

logger = logging.getLogger(__name__)

__all__ = [
    "ORACLE_ESTIMATORS",
    "assert_estimators_disjoint",
    "exact_ceiling",
    "reference_mi",
    "binary_entropy",
]

# Named families this oracle uses. Kept explicit so the disjointness assertion has something to compare
# against, and so a reader can see at a glance that none of them is the arms' binned-MI backend.
ORACLE_ESTIMATORS: Tuple[str, ...] = ("exact_conditional_entropy", "equal_width_plugin", "miller_madow", "chao_shen")

# Above this, the estimators disagree by more than any effect the benchmark measures, so the bundle refuses
# to be used for a rank correlation rather than reporting a number that depends on which estimator ran.
_UNRELIABLE_SPREAD = 0.02


def assert_estimators_disjoint(arm_backends: Sequence[str]) -> None:
    """Raise when an arm's MI backend is also an oracle estimator.

    Args:
        arm_backends: Backend names the scored arms use.

    Raises:
        ValueError: If any backend is shared. A shared estimator turns "the arm ranks columns the way the
            oracle does" into "the arm agrees with itself", which is not a finding.
    """
    shared = sorted(set(arm_backends) & set(ORACLE_ESTIMATORS))
    if shared:
        raise ValueError(
            f"oracle estimators {shared} are also arm backends; the oracle must not share an estimator family "
            "with the arms it scores, or the rank-correlation metric measures self-agreement"
        )


def binary_entropy(probability: np.ndarray) -> np.ndarray:
    """Return the binary entropy in nats, with the ``0 log 0 = 0`` limit taken rather than warned about."""
    p = np.clip(np.asarray(probability, dtype=np.float64), 0.0, 1.0)
    out = np.zeros_like(p)
    inside = (p > 0.0) & (p < 1.0)
    out[inside] = -(p[inside] * np.log(p[inside]) + (1.0 - p[inside]) * np.log(1.0 - p[inside]))
    return out


def exact_ceiling(true_prob: np.ndarray, metric: str = "auc") -> Ceiling:
    """Return the best achievable score under the law ``true_prob``, conditional on the realised rows.

    Args:
        true_prob: Per-row probability of the positive class, as the generator recorded it AFTER every
            declared corruption.
        metric: ``"auc"``, ``"brier"``, ``"logloss"`` or ``"accuracy"``.

    Returns:
        A :class:`Ceiling` with ``method="closed_form"`` and zero standard error: these are expectations
        under a known law, not estimates from a sample, so the only uncertainty left is the label draw --
        which is a property of the evaluation, not of the ceiling.

    Raises:
        ValueError: On an unsupported metric, or on probabilities outside [0, 1] -- both mean the caller is
            holding something other than a probability, and quietly clipping would hide it.
    """
    p = np.asarray(true_prob, dtype=np.float64).ravel()
    if p.size == 0:
        raise ValueError("cannot compute a ceiling from an empty probability vector")
    if np.any(p < 0.0) or np.any(p > 1.0) or not np.all(np.isfinite(p)):
        raise ValueError("true_prob must be finite and inside [0, 1]; got values outside it")

    caveats: Tuple[str, ...] = ()
    if metric == "brier":
        value = float(np.mean(p * (1.0 - p)))
    elif metric == "accuracy":
        value = float(np.mean(np.maximum(p, 1.0 - p)))
    elif metric == "logloss":
        value = float(np.mean(binary_entropy(p)))
        if np.any((p == 0.0) | (p == 1.0)):
            caveats += (
                "the law is deterministic on some rows, so the log-loss ceiling is zero there and is attainable "
                "only in the limit; a model that never predicts exactly 0 or 1 cannot reach it",
            )
    elif metric == "auc":
        from mlframe.data.datasets._target import bayes_auc

        value = float(bayes_auc(p))
        if not np.isfinite(value):
            caveats += ("every row shares one probability, so no ranking exists and the AUC ceiling is undefined",)
    else:
        raise ValueError(f"unsupported ceiling metric {metric!r}; expected one of auc, brier, logloss, accuracy")

    prevalence = float(np.mean(p))
    minority = min(prevalence, 1.0 - prevalence) * p.size
    if minority < 5000:
        # Not a caveat about THIS number, which is exact, but about what a run measured against it can
        # resolve: a metric computed on the minority class is unstable well before the ceiling is.
        caveats += (f"only about {minority:.0f} minority rows: a metric measured against this ceiling is noisy long before the ceiling is",)

    return Ceiling(value=value, se=0.0, method="closed_form", conditional_on="realized_X", n_oracle=int(p.size), metric=metric, caveats=caveats)


def _plugin_mi(column: np.ndarray, labels: np.ndarray, n_bins: int) -> float:
    """Equal-width plug-in mutual information in nats, from a joint histogram."""
    values = np.asarray(column, dtype=np.float64)
    finite = np.isfinite(values)
    values, target = values[finite], np.asarray(labels)[finite]
    if values.size == 0:
        return 0.0
    edges = np.linspace(values.min(), values.max(), n_bins + 1)
    codes = np.clip(np.digitize(values, edges[1:-1], right=False), 0, n_bins - 1)
    classes = np.unique(target)
    joint = np.zeros((n_bins, classes.size), dtype=np.float64)
    for index, label in enumerate(classes):
        joint[:, index] = np.bincount(codes[target == label], minlength=n_bins)
    total = joint.sum()
    if total <= 0:
        return 0.0
    joint /= total
    row, col = joint.sum(axis=1, keepdims=True), joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = joint * (np.log(joint) - np.log(row) - np.log(col))
    return float(np.nansum(np.where(joint > 0, terms, 0.0)))


def _miller_madow(column: np.ndarray, labels: np.ndarray, n_bins: int) -> float:
    """Plug-in MI with the Miller-Madow bias correction, which the plug-in estimator needs at small n."""
    plugin = _plugin_mi(column, labels, n_bins)
    n = int(np.isfinite(np.asarray(column, dtype=np.float64)).sum())
    if n <= 1:
        return plugin
    classes = int(np.unique(labels).size)
    # (cells - rows - cols + 1) / (2n): the standard first-order correction for the entropy plug-in's
    # downward bias in mutual information.
    correction = (n_bins * classes - n_bins - classes + 1) / (2.0 * n)
    return float(plugin + correction)


def _chao_shen(column: np.ndarray, labels: np.ndarray, n_bins: int) -> float:
    """Coverage-adjusted MI: the plug-in scaled by the Chao-Shen sample-coverage estimate.

    A third family rather than a third tuning of the same one -- the point of the bundle is disagreement
    between families, and two variants of one estimator would agree by construction.
    """
    values = np.asarray(column, dtype=np.float64)
    finite = np.isfinite(values)
    n = int(finite.sum())
    if n == 0:
        return 0.0
    edges = np.linspace(values[finite].min(), values[finite].max(), n_bins + 1)
    codes = np.clip(np.digitize(values[finite], edges[1:-1], right=False), 0, n_bins - 1)
    counts = np.bincount(codes, minlength=n_bins)
    singletons = int(np.sum(counts == 1))
    coverage = 1.0 - singletons / n if n > 0 else 1.0
    coverage = min(max(coverage, 1e-6), 1.0)
    return float(_plugin_mi(column, labels, n_bins) / coverage)


def _exact_mi_independent(column: np.ndarray, true_prob: np.ndarray, n_bins: int = 64) -> Optional[float]:
    """Return ``I(X_j; Y)`` computed from the law itself, when the probability varies with this column.

    With ``true_prob`` in hand the identity is ``H(mean p) - E_x[H(E[p | x])]``, and the only approximation
    is the resolution at which ``E[p | x]`` is evaluated -- not a sample estimate of a density. Returns
    ``None`` when the column carries no variation to condition on.
    """
    values = np.asarray(column, dtype=np.float64)
    p = np.asarray(true_prob, dtype=np.float64)
    finite = np.isfinite(values)
    if finite.sum() < 2 or np.ptp(values[finite]) == 0.0:
        return None
    edges = np.linspace(values[finite].min(), values[finite].max(), n_bins + 1)
    codes = np.clip(np.digitize(values[finite], edges[1:-1], right=False), 0, n_bins - 1)
    p_finite = p[finite]

    weights = np.bincount(codes, minlength=n_bins).astype(np.float64)
    sums = np.bincount(codes, weights=p_finite, minlength=n_bins)
    occupied = weights > 0
    conditional = np.zeros(n_bins, dtype=np.float64)
    conditional[occupied] = sums[occupied] / weights[occupied]
    share = weights / weights.sum()
    marginal = float(np.mean(p_finite))
    return float(binary_entropy(np.array([marginal]))[0] - np.sum(share[occupied] * binary_entropy(conditional[occupied])))


def reference_mi(
    columns: Dict[str, np.ndarray],
    labels: np.ndarray,
    true_prob: Optional[np.ndarray] = None,
    bin_counts: Sequence[int] = (5, 10, 20, 50),
) -> Dict[str, MIBundle]:
    """Return one :class:`MIBundle` per column: the exact value where available, plus disagreeing estimates.

    Args:
        columns: Realised feature columns.
        labels: Realised labels.
        true_prob: The law the labels came from; when present, each bundle carries an exact value against
            which the estimators can be read.
        bin_counts: Bin counts the plug-in family is evaluated at. Several, because a single bin count is a
            hidden tuning parameter that decides the answer on a point-mass or heavy-tailed column.

    Returns:
        Column name to bundle. A bundle whose estimator spread exceeds what the benchmark measures is
        flagged ``unreliable``, which suppresses the rank-correlation metric rather than reporting a number
        that depends on which estimator happened to run.
    """
    out: Dict[str, MIBundle] = {}
    for name, values in columns.items():
        estimates = [
            MIEstimate(value=_plugin_mi(values, labels, n_bins), estimator="equal_width_plugin", n_bins=n_bins, n_samples=int(values.size))
            for n_bins in bin_counts
        ]
        estimates.append(MIEstimate(value=_miller_madow(values, labels, 10), estimator="miller_madow", n_bins=10, n_samples=int(values.size)))
        estimates.append(MIEstimate(value=_chao_shen(values, labels, 10), estimator="chao_shen", n_bins=10, n_samples=int(values.size)))

        exact: Optional[MIEstimate] = None
        if true_prob is not None:
            value = _exact_mi_independent(values, true_prob)
            if value is not None:
                exact = MIEstimate(value=value, estimator="exact_conditional_entropy", n_samples=int(values.size))

        bundle = MIBundle(estimates=tuple(estimates), exact=exact)
        spread = bundle.spread()
        caveats: Tuple[str, ...] = ()
        if spread > _UNRELIABLE_SPREAD:
            caveats += (f"estimators disagree by {spread:.4f} nats, more than the effects this benchmark measures",)
        out[name] = MIBundle(estimates=bundle.estimates, exact=bundle.exact, unreliable=spread > _UNRELIABLE_SPREAD, caveats=caveats)
    return out
