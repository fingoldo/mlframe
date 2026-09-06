"""Marginal column draws: the families, the outlier injection, and the standardisation that follows them.

Three decisions here are load bearing and none of them is cosmetic.

**Standardisation is on by default.** In an additively generated structural model the marginal variance
grows with depth in the topological order, so sorting columns by raw variance partially recovers the causal
order (Reisach, Seiler & Drton, NeurIPS 2021). A benchmark built on such data rewards a "sort by variance"
control arm that never looks at the target, and the reward is an artefact of the generator rather than a
property of feature selection. Every column is therefore scaled to unit variance and the original scale is
RECORDED rather than discarded, so a scenario that wants to study varsortability still can.

**Outliers are injected before the link, never after.** Contaminating a feature leaves the conditional law
of the target given the features untouched, so ``true_prob`` needs no update and the Bayes ceiling stays
exact. Contaminating the score after the link would change that law with no update available, which is the
one thing the generator refuses to do.

**Categorical columns are ``pl.Enum`` with explicit levels, never ``pl.Categorical``.** polars re-derives a
``Categorical`` dictionary per slice, so a train/test split of one generated frame yields two frames whose
integer codes mean different things -- silently, with no error. The levels are known here by construction.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

from mlframe.data.datasets._rng import stream_for
from mlframe.data.datasets.spec import FeatureSpec, resolve_knob

logger = logging.getLogger(__name__)

__all__ = [
    "SUPPORTED_FAMILIES",
    "draw_family",
    "inject_outliers",
    "standardize",
    "draw_feature",
    "draw_features",
]

# Families a scenario may ask for. Heavy tails, skew, point masses and discreteness are all represented,
# because each breaks a different selector: equal-mass binning degenerates on point masses, correlation
# ranks collapse under heavy tails, and impurity importances inflate with cardinality.
SUPPORTED_FAMILIES: Tuple[str, ...] = (
    "normal",
    "lognormal",
    "student_t",
    "uniform",
    "exponential",
    "gamma",
    "beta",
    "bernoulli",
    "poisson",
    "zipf",
    "zero_inflated_normal",
    "categorical",
)


def _param(params: Dict[str, float], name: str, default: float) -> float:
    """Return a distribution parameter, falling back to a documented default.

    ``params.get(name) or default`` would be wrong here and is the bug this helper exists to avoid: a
    legitimate zero (a zero mean, a zero inflation rate) is falsy and would be silently replaced.
    """
    value = params.get(name)
    return float(default if value is None else value)


def draw_family(family: str, params: Dict[str, float], size: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``size`` values from one marginal family.

    Args:
        family: One of :data:`SUPPORTED_FAMILIES`.
        params: Family parameters; missing ones take their documented default.
        size: Number of rows.
        rng: The stream to draw from.

    Returns:
        A float64 array of length ``size``. Discrete families still return floats: the dtype a column ends
        up with is the spec's decision, applied once at frame assembly, not a side effect of its family.

    Raises:
        ValueError: If ``family`` is not supported, so a typo fails at generation instead of silently
            producing standard normals under a misleading name.
    """
    if family == "normal":
        return rng.normal(_param(params, "loc", 0.0), _param(params, "scale", 1.0), size)
    if family == "lognormal":
        return rng.lognormal(_param(params, "mean", 0.0), _param(params, "sigma", 1.0), size)
    if family == "student_t":
        return rng.standard_t(_param(params, "df", 4.0), size)
    if family == "uniform":
        return rng.uniform(_param(params, "low", 0.0), _param(params, "high", 1.0), size)
    if family == "exponential":
        return rng.exponential(_param(params, "scale", 1.0), size)
    if family == "gamma":
        return rng.gamma(_param(params, "shape", 2.0), _param(params, "scale", 1.0), size)
    if family == "beta":
        return rng.beta(_param(params, "a", 2.0), _param(params, "b", 5.0), size)
    if family == "bernoulli":
        return (rng.random(size) < _param(params, "p", 0.5)).astype(np.float64)
    if family == "poisson":
        return rng.poisson(_param(params, "lam", 3.0), size).astype(np.float64)
    if family == "zipf":
        # Capped, because a raw Zipf draw is unbounded and one astronomically large code would dominate any
        # downstream scaling; the cap is part of the declared family, not a silent clip.
        raw = rng.zipf(_param(params, "a", 2.0), size).astype(np.float64)
        return np.asarray(np.minimum(raw, _param(params, "max_value", 1000.0)), dtype=np.float64)
    if family == "zero_inflated_normal":
        values = rng.normal(_param(params, "loc", 0.0), _param(params, "scale", 1.0), size)
        zeros = rng.random(size) < _param(params, "inflation", 0.3)
        values[zeros] = 0.0
        return values
    if family == "categorical":
        n_levels = int(_param(params, "n_levels", 3.0))
        return rng.integers(0, max(n_levels, 1), size).astype(np.float64)
    raise ValueError(f"unsupported marginal family {family!r}; supported: {SUPPORTED_FAMILIES}")


def inject_outliers(values: np.ndarray, fraction: float, rng: np.random.Generator, magnitude: float = 8.0) -> np.ndarray:
    """Replace a fraction of the rows with far-out values, returning a new array.

    Applied to a FEATURE before the link, so the conditional law of the target given the features is
    unchanged and the Bayes ceiling needs no update. Injecting into the score after the link is forbidden
    elsewhere in the generator for exactly the opposite reason.

    Args:
        values: The clean column.
        fraction: Share of rows to contaminate; zero returns the input unchanged.
        rng: The stream that picks the rows and the signs.
        magnitude: How many robust scales out the contaminated values land.

    Returns:
        A contaminated copy, or the original array when ``fraction`` is zero.
    """
    if fraction <= 0.0:
        return values
    n = values.size
    count = max(1, round(fraction * n))
    idx = rng.choice(n, size=min(count, n), replace=False)
    # Median absolute deviation rather than the standard deviation: the scale must not be set by the
    # contamination it is being used to size.
    mad = float(np.median(np.abs(values - np.median(values))))
    scale = mad * 1.4826 if mad > 0 else float(np.std(values)) or 1.0
    out: np.ndarray = values.copy()
    out[idx] = np.median(values) + magnitude * scale * rng.choice([-1.0, 1.0], size=idx.size)
    return out


def standardize(values: np.ndarray) -> Tuple[np.ndarray, float]:
    """Centre and scale a column to unit variance, returning the scaled column and its original scale.

    The scale is returned rather than dropped so a scenario studying varsortability can put it back, and so
    the truth record can state what was removed.
    """
    sd = float(np.std(values))
    if sd <= 0.0:
        # A constant column carries no information and scaling it would divide by zero. It is centred and
        # reported with a zero scale, which is the honest statement about what it is.
        logger.warning("constant column encountered; centring only, scale recorded as 0.0")
        return values - float(np.mean(values)), 0.0
    return (values - float(np.mean(values))) / sd, sd


def draw_feature(feature: FeatureSpec, n: int, root_seed: int, spec_name: str, knob_rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, float]:
    """Draw one column end to end: family, outliers, standardisation.

    The stream is addressed by NAME (``spec_name``/``feature.name``), never by position, so inserting a
    column into the middle of a spec leaves every other column bit-identical. Positional spawning would
    shift all downstream streams and silently change data that the scenario did not touch.

    Args:
        feature: The column's declaration.
        n: Number of rows.
        root_seed: The dataset's root seed.
        spec_name: The dataset name, which namespaces the stream.
        knob_rng: Stream for resolving priors on knobs; defaults to the column's own stream so a spec with
            a prior-valued knob stays reproducible without a caller having to thread one through.

    Returns:
        ``(column, pre_standardization_scale)``. The scale is ``1.0`` when the column was left unscaled, so
        a consumer can always multiply by it to recover the raw column.
    """
    rng = stream_for(root_seed, spec_name, "feature", feature.name)
    fraction = resolve_knob(feature.outlier_fraction, knob_rng if knob_rng is not None else rng)

    values = draw_family(feature.family, dict(feature.params), n, rng)
    values = inject_outliers(values, float(fraction), rng)
    if feature.dtype == "category" or not feature.standardize:
        # A categorical column's codes are labels; scaling them would turn a nominal level into a magnitude.
        return values, 1.0
    return standardize(values)


def draw_features(
    features: Tuple[FeatureSpec, ...],
    n: int,
    root_seed: int,
    spec_name: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Draw every independently declared column, returning the columns and their pre-standardisation scales.

    Columns that are reflections of a latent, or otherwise built from other columns, are NOT produced here:
    this layer draws exogenous marginals only, and the latent and link layers overwrite what they own.
    """
    columns: Dict[str, np.ndarray] = {}
    scales: Dict[str, float] = {}
    for feature in features:
        values, scale = draw_feature(feature, n, root_seed, spec_name)
        columns[feature.name] = values
        scales[feature.name] = scale
    return columns, scales
