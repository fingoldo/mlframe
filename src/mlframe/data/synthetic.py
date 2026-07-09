"""Synthetic data generation helpers: random-variable sampling, probability-to-class-label assignment, and full modelling dataset construction for tests/benchmarks."""

from __future__ import annotations

import time
from typing import Optional

from scipy import stats
import numpy as np, pandas as pd
from itertools import combinations
from sklearn.utils import check_random_state

from pyutilz.string import rpad
from pyutilz.system import tqdmu

import logging

logger = logging.getLogger(__name__)

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

    def njit(*args, **kwargs):  # pragma: no cover
        """No-op stand-in for ``numba.njit`` when numba isn't installed: returns the function unchanged, supporting both bare-decorator and decorator-with-args call forms."""
        def wrap(fn):
            """Identity wrapper returning ``fn`` unchanged."""
            return fn

        if args and callable(args[0]):
            return args[0]
        return wrap


@njit(cache=True)
def _assign_classes_from_probability_kernel(predictors: np.ndarray, draw: np.ndarray, n_classes: int, out: np.ndarray) -> np.ndarray:
    """For each row, walk its class-probability vector cumulatively and assign the class whose cumulative-sum boundary first exceeds the row's uniform ``draw``; float32 accumulation is intentional (see below) to bit-match the legacy Python loop."""
    # ``total`` is seeded as float32 to match the legacy Python loop bit-for-bit: there ``total`` started as a Python float
    # but ``py_float += np_float32`` follows numpy promotion and stays float32, so the cumulative walk rounded in float32.
    # A float64 accumulator here would flip the chosen class on rare draws landing within ~1e-8 of a cumulative boundary.
    n_samples = predictors.shape[0]
    for i in range(n_samples):
        total = np.float32(0.0)
        out[i] = n_classes - 1
        for j in range(n_classes):
            total = total + predictors[i, j]
            if draw[i] < total:
                out[i] = j
                break
    return out


def sample_random_variable(
    kind: str = "cat",
    size: int = 1000,
    shift: float = 0,
    scale: float = 1.0,
    include: Optional[set] = None,
    exclude: Optional[set] = None,
    max_time_per10k: float = 1.0,
    randomize_params: bool = True,
    random_state=None,
):
    """
    Samples from a specified categorical or continuous distribution.
    """
    if exclude is None:
        exclude = set(["ksone", "gausshyper", "kstwo", "cosine", "frechet_l", "frechet_r"])
    if include is None:
        include = set()
    # All random draws go through this generator so callers can reproduce results.
    generator = check_random_state(random_state)

    cats = [dst for dst in stats._distr_params.distdiscrete if (dst[0] in include or len(include) == 0) and (dst[0] not in exclude)]
    conts = [dst for dst in stats._distr_params.distcont if (dst[0] in include or len(include) == 0) and (dst[0] not in exclude)]

    if kind == "cont":
        source = conts
    elif kind == "cat":
        source = cats
    elif kind == "mixed":
        source = cats + conts

    dist_name, params = source[generator.randint(0, len(source))]

    # Append recommended dist params.
    # Note: `params` came out of random.choice as a tuple, but we convert to list
    # to allow appending. The membership check below must compare tuples since
    # `conts` entries are tuples.
    params = list(params)
    if (dist_name, tuple(params)) in conts:
        if randomize_params:
            params = [*params, shift * generator.rand(), scale * generator.rand()]
        else:
            params = [*params, shift, scale]

    # Create instance of a random variable
    dist = getattr(stats, dist_name)

    # Create frozen random variable using parameters and add it to the list to be used to draw the probability density functions
    start = time.time()
    rv = dist(*params)
    data = rv.rvs(size=size, random_state=generator)
    end = time.time()

    # Report if taking too long
    if size > 0:
        if (end - start) * 10_000 / size > max_time_per10k:
            logger.warning("Sampling %s from %s took %s sec.", size, dist_name, f"{end-start:,.0f}")

    return dist_name.replace("_", "-"), data


def assign_classes_from_probability(predictors: np.ndarray, draw: np.ndarray, n_classes: int, out: np.ndarray | None = None) -> np.ndarray:
    """Pick a class per row by walking the cumulative per-row predictor probabilities against a uniform draw.

    ``predictors`` are float32 per-row normalized to sum ~1.0, but float32 rounding (and degenerate all-zero rows
    where the row sum was guarded to 0) can leave the cumulative total just under the draw, so the default class is
    the last one — without it ``out`` (an ``np.empty`` buffer) would retain uninitialized garbage for those rows.
    """
    n_samples = predictors.shape[0]
    if out is None:
        out = np.empty(n_samples, dtype=np.int32)
    return np.asarray(_assign_classes_from_probability_kernel(predictors, draw, n_classes, out))


def generate_modelling_data(
    n_samples: int = 100_000,
    *,
    n_singly_correlated: int = 3,  # dependent on a single (randomly chosen) predictor. cont.
    n_mutually_correlated: int = 1,  # dependent on a random subset of predictors (>1). cont.
    n_directly_correlated: int = 0,  # directly dependent on the generated outcome itself, instead of predictors.
    # cat (easiest, just pick from the classes with probs such that one of the real class is highest, rest equal) or cont.
    n_unrelated_single: int = 3,  # features not dependent on any true predictor. cat or cont.
    n_unrelated_intercorrelated: int = 1,  # features made up from existing single unrelated+noise.
    n_repeated: int = 0,  # some of existing features may go duplicated randomly!
    n_informative: int = 3,  # what's directly influencing the target
    generation: str = "probability",  # if "formula", target is generated by exact formula (+noise). else - as a sampling from the probability var.
    n_classes: int = 3,  # 0 for regression
    weights=None,  # only used by generation="formula": per-predictor weight for the linear combination forming the target (None -> equal weights)
    flip_y: float = 0.01,  # for some rows, final class is just flipped randomly to create more disorder.
    shift: float = 0.0,
    scale: float = 1.0,
    shuffle: bool = True,
    random_state: Optional[int] = None,
    feature_noise: float = 0.05,  # gaussian jitter (fraction of each correlated feature's std) added to singly/mutually correlated features
    target_noise: float = 0.05,  # only used by generation="formula": gaussian jitter (fraction of the formula's std) added before binning into n_classes
    timeseries: bool = False,  # when True, correlated features' dependence on their source predictor(s) drifts linearly (0.5x->1.5x) over the sample axis
    # how to create cat features?
    min_cardinality: int = 2,  # used only when max_cardinality is set: unrelated_single features are then qcut-discretized to a random cardinality in [min_cardinality, max_cardinality]
    max_cardinality: Optional[int] = None,
    include_distributions: Optional[set] = None,
    return_dataframe: bool = True,
):
    """
    Creates a synthetic dataset with random inputs (sampled from randomly chosen distributions with random parameters).
    Target is nclasses classification. It can be defined:
        1) by calculating some formula over a set of cont and cat predictors
        final range of that virtual variables is split into nclasses parts. Some target_noise% can be added.

        2) by using some predictors (up to nclasses) DIRECTLY as generative probs.


    In 2), it must be a challenge for a ML model to pick exactly and only generative probs variables. Or, at least, fare with them in metrics.

    Features can be correlated in a linear or non-linear way (way is reflected in the var's name)+ additive noise (some % of varaible's range).

    Will Brier score of a ML model fare the score or bare probs?

    Attempt to model a situation in sports, where bookies odds are said to be the only and the best estimates of the winning probability.
    """
    if include_distributions is None:
        include_distributions = set([])
    if n_classes > 0:
        if n_informative > n_classes:
            raise ValueError(f"n_informative ({n_informative}) must be <= n_classes ({n_classes})")

    if n_unrelated_single < n_unrelated_intercorrelated:
        raise ValueError(f"n_unrelated_single ({n_unrelated_single}) must be >= n_unrelated_intercorrelated ({n_unrelated_intercorrelated})")

    generator = check_random_state(random_state)
    # random.seed(random_state)

    n_features = (
        n_informative + n_singly_correlated + n_mutually_correlated + n_directly_correlated + n_unrelated_single + n_unrelated_intercorrelated + n_repeated
    )

    # Initialize X and y

    fnames = []
    X = np.empty((n_samples, n_features), dtype=np.float32)

    # ----------------------------------------------------------------------------------------------------------------------------
    # Initially draw informative features
    # ----------------------------------------------------------------------------------------------------------------------------

    if n_classes > 0:
        # classification
        y = np.empty(n_samples, dtype=np.int32)

        if generation == "probability":

            predictors = np.empty((n_samples, n_classes), dtype=np.float32)
            for j in tqdmu(range(n_classes), desc=rpad("predictors")):
                dist_name, predictors[:, j] = sample_random_variable(kind="cont", size=n_samples, shift=shift, scale=scale, include=include_distributions, random_state=generator)
                pred_min, pred_max = predictors[:, j].min(), predictors[:, j].max()
                # Guarded normalization: avoid division by zero when the sampled
                # column is constant (all values equal).
                span = pred_max - pred_min
                predictors[:, j] = np.where(span == 0, 0.0, (predictors[:, j] - pred_min) / (span if span != 0 else 1.0))

                fnames.append(f"prob_{dist_name}_{j}")
            pred_sums = predictors.sum(axis=1)

            for j in range(n_classes):
                # Guard divide-by-zero when the row of predictors sums to zero.
                predictors[:, j] = np.where(pred_sums == 0, 0.0, predictors[:, j] / np.where(pred_sums == 0, 1.0, pred_sums))

            # Let's draw from 0 to 1 and pick the class
            draw = generator.rand(n_samples)
            assign_classes_from_probability(predictors, draw, n_classes, out=y)

            X[:, :n_informative] = predictors[:, :n_informative]
        else:
            # "formula": draw n_informative continuous predictors, combine them into a single
            # scalar via a (possibly weighted) linear combination, jitter it by target_noise
            # (same gaussian-jitter-scaled-by-std shape as feature_noise below), then split its
            # range into n_classes bins via quantile cuts -- mirrors the qcut discretization
            # already used for unrelated_single/max_cardinality further down this function.
            predictors = np.empty((n_samples, n_informative), dtype=np.float32)
            for j in tqdmu(range(n_informative), desc=rpad("predictors")):
                dist_name, predictors[:, j] = sample_random_variable(kind="cont", size=n_samples, shift=shift, scale=scale, include=include_distributions, random_state=generator)
                fnames.append(f"formula_{dist_name}_{j}")

            # Min-max normalize each column to [0, 1] before combining -- sample_random_variable
            # draws from randomly chosen distribution families whose raw scales can differ by
            # orders of magnitude (e.g. a heavy-tailed pick vs. a bounded one); without this,
            # `weights` would be meaningless whenever one column's raw scale dominates the sum
            # regardless of its weight. Mirrors the normalization already applied to the
            # "probability" branch's predictors above.
            normalized = np.empty_like(predictors)
            for j in range(n_informative):
                col_min, col_max = predictors[:, j].min(), predictors[:, j].max()
                span = col_max - col_min
                normalized[:, j] = np.where(span == 0, 0.0, (predictors[:, j] - col_min) / (span if span != 0 else 1.0))

            formula = np.average(normalized, axis=1, weights=weights).astype(np.float64)
            if target_noise:
                formula = formula + generator.normal(0.0, target_noise * (np.std(formula) or 1.0), size=n_samples)

            y[:] = pd.qcut(formula, q=n_classes, labels=False, duplicates="drop").astype(np.int32)
            # qcut can drop empty/duplicate-boundary bins (e.g. a heavy-tailed sampled
            # distribution), yielding fewer than n_classes distinct labels; that is an accepted
            # property of the underlying data, not an error, exactly like max_cardinality's own
            # qcut call elsewhere in this function.

            X[:, :n_informative] = predictors

    else:
        # regression
        y = np.empty(n_samples, dtype=np.float32)
        raise NotImplementedError

    idx = n_informative

    # ----------------------------------------------------------------------------------------------------------------------------
    # Create correlated features
    # ----------------------------------------------------------------------------------------------------------------------------

    # Coefficient-drift-over-time: when timeseries=True, treat the sample axis as a time axis and
    # scale each correlated feature's dependence on its source predictor(s) by a factor drifting
    # linearly from 0.5x (start) to 1.5x (end), so the correlation strength is not stationary.
    time_drift = 1.0 + (np.linspace(0.0, 1.0, n_samples, dtype=np.float32) - 0.5) if timeseries else None

    if n_singly_correlated > 0:
        # dependent on a single (randomly chosen) predictor. cont.

        for j in tqdmu(range(n_singly_correlated), desc=rpad("singly_correlated")):
            dist_name, X[:, idx + j] = sample_random_variable(kind="cont", size=n_samples, shift=shift, scale=scale, include=include_distributions, random_state=generator)

            k = generator.choice(range(n_informative))
            X[:, idx + j] *= X[:, k]
            if time_drift is not None:
                X[:, idx + j] *= time_drift
            if feature_noise:
                X[:, idx + j] += generator.normal(0.0, feature_noise * (np.std(X[:, idx + j]) or 1.0), size=n_samples).astype(np.float32)

            fnames.append(f"sc_{dist_name}_{k}")
    idx += n_singly_correlated

    if n_mutually_correlated > 0:
        # dependent on a random subset of predictors (>1). cont.

        for j in tqdmu(range(n_mutually_correlated), desc=rpad("mutually_correlated")):
            combs = list(combinations(range(n_informative), generator.choice(range(2, n_informative + 1))))
            current_combination = combs[generator.choice(len(combs))]

            dist_name, X[:, idx + j] = sample_random_variable(kind="cont", size=n_samples, shift=shift, scale=scale, include=include_distributions, random_state=generator)

            for k in current_combination:
                X[:, idx + j] *= X[:, k]
            if time_drift is not None:
                X[:, idx + j] *= time_drift
            if feature_noise:
                X[:, idx + j] += generator.normal(0.0, feature_noise * (np.std(X[:, idx + j]) or 1.0), size=n_samples).astype(np.float32)

            fnames.append(f"mc_{dist_name}_{'-'.join(map(str,current_combination))}")
    idx += n_mutually_correlated

    if n_directly_correlated > 0:
        # directly dependent on the generated outcome itself, instead of predictors.
        # cat (easiest, just pick from the classes with probs such that one of the real class is highest, rest equal).
        raise NotImplementedError
    idx += n_directly_correlated

    # ----------------------------------------------------------------------------------------------------------------------------
    # Create unrelated features
    # ----------------------------------------------------------------------------------------------------------------------------

    if n_unrelated_single > 0:
        # features not dependent on any true predictor. cat or cont.
        for j in tqdmu(range(n_unrelated_single), desc=rpad("unrelated_single")):
            dist_name, X[:, idx + j] = sample_random_variable(kind="mixed", size=n_samples, shift=shift, scale=scale, include=include_distributions, random_state=generator)
            if max_cardinality is not None:
                # Discretize into a random cardinality within [min_cardinality, max_cardinality] so callers
                # exercising cat-feature-cardinality-sensitive code paths (e.g. target/frequency encoders)
                # get controllable categorical arity instead of the raw continuous/discrete draw's natural cardinality.
                n_bins = generator.randint(min_cardinality, max_cardinality + 1)
                X[:, idx + j] = pd.qcut(X[:, idx + j], q=n_bins, labels=False, duplicates="drop").astype(np.float32)
            fnames.append(f"unr_{dist_name}")

    idx += n_unrelated_single

    if n_unrelated_intercorrelated > 0:
        # features not dependent on any true predictor, but interdependent on themselves. cat or cont.
        for j in tqdmu(range(n_unrelated_intercorrelated), desc=rpad("unrelated_intercorrelated")):
            combs = list(combinations(range(n_unrelated_single), generator.choice(range(2, n_unrelated_single + 1))))
            current_combination = combs[generator.choice(len(combs))]

            dist_name, X[:, idx + j] = sample_random_variable(kind="mixed", size=n_samples, shift=shift, scale=scale, include=include_distributions, random_state=generator)

            for k in current_combination:
                # `idx - n_unrelated_single + k` reaches back into the block of
                # unrelated-single features created in the previous section so
                # the new intercorrelated feature depends on a combination of them.
                X[:, idx + j] *= X[:, idx - n_unrelated_single + k]

            fnames.append(f"unrintrc_{dist_name}_{'-'.join(map(str,current_combination))}")

    idx += n_unrelated_intercorrelated

    # ----------------------------------------------------------------------------------------------------------------------------
    # Repeat some features
    # ----------------------------------------------------------------------------------------------------------------------------

    if n_repeated > 0:
        n = idx
        # Uniformly pick source columns from [0, n). The previous formula
        # `((n-1)*rand + 0.5).astype(intp)` under-sampled column 0 and
        # column n-1 (each had half the weight of interior columns).
        indices = generator.randint(0, n, size=n_repeated)
        X[:, n : n + n_repeated] = X[:, indices]

    # ----------------------------------------------------------------------------------------------------------------------------
    # Randomly replace labels
    # ----------------------------------------------------------------------------------------------------------------------------

    if flip_y >= 0.0:
        flip_mask = generator.rand(n_samples) < flip_y
        y[flip_mask] = generator.randint(n_classes, size=flip_mask.sum())

    # ----------------------------------------------------------------------------------------------------------------------------
    # Randomly permute features
    # ----------------------------------------------------------------------------------------------------------------------------

    if shuffle:
        indices = np.arange(n_features)
        generator.shuffle(indices)
        X[:, :] = X[:, indices]
        fnames = np.array(fnames)[indices].tolist()

    if return_dataframe:
        X = pd.DataFrame(data=X, columns=fnames)
    return X, y, fnames
