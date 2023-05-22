import time

import scipy
import random
from scipy import stats
import numpy as np, pandas as pd
from itertools import combinations
from sklearn.utils import check_array, check_random_state

from pyutilz.string import rpad
from pyutilz.system import tqdmu

from pyutilz.logging import init_logging
import logging


def sample_random_variable(
    kind: str = "cat",
    size: int = 1000,
    shift: float = 0,
    scale: float = 1.0,
    include: set = set(),
    exclude: set = set(["ksone", "gausshyper", "kstwo", "cosine", "frechet_l", "frechet_r"]),
    max_time_per10k: float = 1.0,
    randomize_params: bool = True,
):
    """
        Samples from a specified categorical or continuous distribution.
    """
    cats = [dst for dst in stats._distr_params.distdiscrete if (dst[0] in include or len(include) == 0) and (dst[0] not in exclude)]
    conts = [dst for dst in stats._distr_params.distcont if (dst[0] in include or len(include) == 0) and (dst[0] not in exclude)]

    if kind == "cont":
        source = conts
    elif kind == "cat":
        source = cats
    elif kind == "mixed":
        source = cats + conts

    dist_name, params = random.choice(source)

    # Append recommended dist params
    params = list(params)
    if (dist_name, params) in conts:
        if randomize_params:
            params = params + [shift * random.random(), scale * random.random()]
        else:
            params = params + [shift, scale]

    # Create instance of a random variable
    dist = getattr(stats, dist_name)

    # Create frozen random variable using parameters and add it to the list to be used to draw the probability density functions
    start = time.time()
    rv = dist(*params)
    data = rv.rvs(size=size)
    end = time.time()

    # Report if taking too long
    if size > 0:
        if (end - start) * 10_000 / size > max_time_per10k:
            logging.warning(f"Sampling {size} from {dist_name} took {end-start:,.0f} sec.")

    return dist_name.replace("_", "-"), data
    
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
    weights=None,
    flip_y: float = 0.01,  # for some rows, final class is just flipped randomly to create more disorder.
    shift: float = 0.0,
    scale: float = 1.0,
    shuffle: bool = True,
    random_state: int = None,
    feature_noise: float = 0.05,  # used additively when generating correlated features
    target_noise: float = 0.05,  # used additively when generating target in mode 1
    timeseries: bool = False,  # dependencies (coefficients, at least) are changing over time?
    # how to create cat features?
    min_cardinality: int = 2,
    max_cardinality: int = None,
    include_distributions: set = set([]),
    return_dataframe: bool = True,
):
    """
        Creates a synthetic dataset with random inputs (sampled from randomly chosen distributions with random parameters).
        Target is nclasses classification. It can be defined:
            1) by calculating some formula over a set of cont and cat predictors
            final range of that virual variables is split into nclasses parts. Some target_noise% can be added.
            
            2) by using some predictors (up to nclasses) DIRECTLY as generative probs.
            
        
        In 2), it must be a challenge for a ML model to pick exactly and only generative probs variables. Or, at least, fare with them in metrics.
        
        Features can be correlated in a linear or non-linear way (way is reflected in the var's name)+ additive noise (some % of varaible's range).
        
        Will Brier score of a ML model fare the score or bare probs?
        
        Attempt to model a situation in sports, where bookies odds are said to be the only and the best estimates of the winning probability.
    """
    if n_classes > 0:
        assert n_informative <= n_classes

    assert n_unrelated_single >= n_unrelated_intercorrelated

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
                dist_name, predictors[:, j] = sample_random_variable(kind="cont", size=n_samples, shift=shift, scale=scale, include=include_distributions)
                pred_min, pred_max = predictors[:, j].min(), predictors[:, j].max()
                predictors[:, j] = (predictors[:, j] - pred_min) / (pred_max - pred_min)

                fnames.append(f"prob_{dist_name}_{j}")
            pred_sums = predictors.sum(axis=1)

            for j in range(n_classes):
                predictors[:, j] = predictors[:, j] / pred_sums

            # Let's draw from 0 to 1 and pick the class
            draw = np.random.rand(n_samples)
            for i in range(n_samples):
                total = 0.0
                for j in range(n_classes):
                    total += predictors[i, j]
                    if draw[i] < total:
                        y[i] = j
                        break

            X[:, :n_informative] = predictors[:, :n_informative]
        else:
            raise NotImplementedError

    else:
        # regression
        y = np.empty(n_samples, dtype=np.float32)
        raise NotImplementedError

    idx = n_informative

    # ----------------------------------------------------------------------------------------------------------------------------
    # Create correlated features
    # ----------------------------------------------------------------------------------------------------------------------------

    if n_singly_correlated > 0:
        # dependent on a single (randomly chosen) predictor. cont.

        for j in tqdmu(range(n_singly_correlated), desc=rpad("singly_correlated")):
            dist_name, X[:, idx + j] = sample_random_variable(kind="cont", size=n_samples, shift=shift, scale=scale, include=include_distributions)

            k = np.random.choice(range(n_informative))
            X[:, idx + j] *= X[:, k]

            fnames.append(f"sc_{dist_name}_{k}")
    idx += n_singly_correlated

    if n_mutually_correlated > 0:
        # dependent on a random subset of predictors (>1). cont.

        for j in tqdmu(range(n_mutually_correlated), desc=rpad("mutually_correlated")):
            combs = list(combinations(range(n_informative), np.random.choice(range(2, n_informative + 1))))
            current_combination = combs[np.random.choice(len(combs))]

            dist_name, X[:, idx + j] = sample_random_variable(kind="cont", size=n_samples, shift=shift, scale=scale, include=include_distributions)

            for k in current_combination:
                X[:, idx + j] *= X[:, k]

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
            dist_name, X[:, idx + j] = sample_random_variable(kind="mixed", size=n_samples, shift=shift, scale=scale, include=include_distributions)
            fnames.append(f"unr_{dist_name}")

    idx += n_unrelated_single

    if n_unrelated_intercorrelated > 0:
        # features not dependent on any true predictor, but interdependent on themselves. cat or cont.
        for j in tqdmu(range(n_unrelated_intercorrelated), desc=rpad("unrelated_intercorrelated")):
            combs = list(combinations(range(n_unrelated_single), np.random.choice(range(2, n_unrelated_single + 1))))
            current_combination = combs[np.random.choice(len(combs))]

            dist_name, X[:, idx + j] = sample_random_variable(kind="mixed", size=n_samples, shift=shift, scale=scale, include=include_distributions)

            for k in current_combination:
                X[:, idx + j] *= X[:, idx - n_unrelated_single + k]

            fnames.append(f"unrintrc_{dist_name}_{'-'.join(map(str,current_combination))}")

    idx += n_unrelated_intercorrelated

    # ----------------------------------------------------------------------------------------------------------------------------
    # Repeat some features
    # ----------------------------------------------------------------------------------------------------------------------------

    if n_repeated > 0:
        n = idx
        indices = ((n - 1) * generator.rand(n_repeated) + 0.5).astype(np.intp)
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