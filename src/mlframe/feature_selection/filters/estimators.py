"""Alternative MI estimators for the mRMR pipeline.

The default discretized plug-in estimator (``compute_mi_from_classes`` in ``info_theory.py``) is fast but biased on small samples and
high-cardinality conditioning sets. This module exposes alternatives that trade some speed for accuracy:

* ``ksg_mi`` -- k-Nearest-Neighbor estimator (Kraskov 2004), wraps ``sklearn.feature_selection.mutual_info_classif/regression``. Operates
  on continuous data without discretisation, asymptotically unbiased. ~2-5x slower than discretized plug-in on n=10k.
* ``miller_madow_mi`` -- discretized plug-in with Miller-Madow bias correction applied to all entropy terms in the MI decomposition.
  Negligible speed cost.
* ``nsb_mi`` (placeholder) -- Bayesian (Nemenman-Shafee-Bialek) estimator. Best for small N. Implemented via optional dependency on
  ``ndd`` package; raises ``ImportError`` if not installed.

USABILITY_A-8 / c8_usability_wrappers.md (/ 2026-07-22): ``MRMR`` has no ``estimator=``
constructor kwarg -- ``ksg_mi_with_target``/``ksg_mi_pair``/``ksg_mi_with_significance``/``nsb_mi`` are
confirmed dead from ``MRMR.fit``'s production path; call them directly from this module if needed.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


def ksg_mi_with_target(
    X: np.ndarray,
    y: np.ndarray,
    feature_indices: Sequence[int],
    *,
    n_neighbors: int = 3,
    discrete_target: bool = True,
    random_state: int = 42,
) -> np.ndarray:
    """Kraskov-Stoegbauer-Grassberger MI estimate of each feature with target.

    Uses ``sklearn.feature_selection.mutual_info_classif/regression``, which implements the KSG estimator on continuous numeric data
    (k-NN-based, no discretisation). Returns shape ``(len(feature_indices),)`` -- MI of feature_i with target in nats (sklearn convention).

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Continuous (un-discretised) feature matrix.
    y : array, shape (n_samples,)
        Target. ``discrete_target=True`` for classification, False for regression.
    feature_indices : sequence of int
        Columns of X to evaluate. Avoids calling sklearn on full X when only a few columns are needed.

    Returns
    -------
    mi : ndarray, shape (len(feature_indices),)
        MI estimates in nats.

    Notes
    -----
    On n=10000 with 100 features: ~1-2s for all 100. ~10x slower than discretized plug-in but asymptotically unbiased. Recommended when
    support is correctness-critical or discretisation artefacts dominate the error budget.
    """
    if discrete_target:
        from sklearn.feature_selection import mutual_info_classif as _mi_func
    else:
        from sklearn.feature_selection import mutual_info_regression as _mi_func

    X_sub = X[:, list(feature_indices)] if X.ndim == 2 else X.reshape(-1, 1)
    return np.asarray(_mi_func(
        X_sub, y,
        n_neighbors=n_neighbors,
        random_state=random_state,
    ))


def ksg_mi_pair(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_neighbors: int = 3,
    discrete_target: bool = True,
    random_state: int = 42,
) -> float:
    """Single-pair convenience wrapper around ``ksg_mi_with_target``."""
    res = ksg_mi_with_target(
        X=x.reshape(-1, 1),
        y=y,
        feature_indices=[0],
        n_neighbors=n_neighbors,
        discrete_target=discrete_target,
        random_state=random_state,
    )
    return float(res[0])


def ksg_mi_with_significance(
    X: np.ndarray,
    y: np.ndarray,
    feature_indices: Sequence[int],
    *,
    n_neighbors: int = 3,
    discrete_target: bool = True,
    n_permutations: int = 50,
    alpha: float = 0.05,
    random_state: int = 42,
    n_jobs: int = -1,
) -> tuple:
    """KSG MI ranking + permutation-test significance filter.

    Plain KSG top-K (``ksg_mi_with_target`` then sort) overfits on noisy data: noise features can score high MI by chance. This wrapper
    adds a per-feature permutation test:

    1. Compute KSG MI of every requested column with target.
    2. For each column, run ``n_permutations`` shuffles of ``y`` and compute KSG MI on the shuffle.
       ``p_value = (1 + #(perm_mi >= observed_mi)) / (1 + n_permutations)`` -- the standard conservative permutation-test p-value.
    3. Reject features with ``p_value > alpha``.

    Returns
    -------
    (mi_arr, p_arr, support) : tuple
        ``mi_arr[i]`` -- KSG MI of feature_i with target.
        ``p_arr[i]`` -- permutation-test p-value.
        ``support`` -- ndarray of feature indices that passed ``p_value <= alpha``, sorted by MI descending.

    Notes
    -----
    Speed: ``n_permutations`` extra KSG calls per feature. For n=10000, p=100, n_permutations=50 ~ 1-2 minutes serial. Use ``n_jobs=-1``
    for joblib parallelism over features.

    The shuffle is on ``y`` (not on ``X[:, j]``); this preserves the marginal of ``X`` (so KSG's k-NN structure is unchanged) while
    breaking the X-y dependency. Standard recipe in ``sklearn.model_selection.permutation_test_score``.
    """
    from joblib import Parallel, delayed

    rng = np.random.default_rng(random_state)
    Xn = X[:, list(feature_indices)] if X.ndim == 2 else X.reshape(-1, 1)
    n_features = Xn.shape[1]

    observed = ksg_mi_with_target(
        X=Xn, y=y, feature_indices=list(range(n_features)),
        n_neighbors=n_neighbors, discrete_target=discrete_target,
        random_state=random_state,
    )

    # Permutation test: shuffle y per iteration; recompute MI for all features.
    shuffle_seeds = rng.integers(0, 2**31 - 1, size=n_permutations)

    def _one_perm(seed: int) -> np.ndarray:
        """Recompute per-feature KSG MI against one seeded shuffle of ``y`` (one permutation-null draw)."""
        local_rng = np.random.default_rng(seed)
        y_shuf = y.copy()
        local_rng.shuffle(y_shuf)
        return ksg_mi_with_target(
            X=Xn, y=y_shuf, feature_indices=list(range(n_features)),
            n_neighbors=n_neighbors, discrete_target=discrete_target,
            random_state=int(seed),
        )

    if n_jobs == 1:
        perm_mis = [_one_perm(s) for s in shuffle_seeds]
    else:
        # backend="threading": mirrors MRMR.__init__ default flip
        # (commit 0da27e0). The per-permutation worker reads ``X`` /
        # ``y`` from the enclosing closure -- with loky each worker
        # process would deep-copy the entire frame, repeating the
        # iter-371 OOM cascade. Threading shares the arrays zero-copy
        # and the inner mi kernel releases the GIL.
        perm_mis = Parallel(n_jobs=n_jobs, max_nbytes=int(1e7), backend="threading")(delayed(_one_perm)(s) for s in shuffle_seeds)
    perm_mi_arr = np.stack(perm_mis, axis=0)  # (n_permutations, n_features)

    # Conservative p-value: (1 + #failures) / (1 + n_perms).
    n_failures = (perm_mi_arr >= observed[None, :]).sum(axis=0)
    p_values = (1 + n_failures) / (1 + n_permutations)

    significant = np.where(p_values <= alpha)[0]
    # Sort the surviving indices by their MI descending.
    # Wave 58 (2026-05-20): lexsort with significant-index tiebreaker for
    # deterministic order across runs when MIs tie.
    _obs_sig = observed[significant]
    order = significant[np.lexsort((significant, -_obs_sig))]
    support = np.asarray([feature_indices[i] for i in order], dtype=np.int64)

    return observed, p_values, support


def nsb_mi(
    classes_x: np.ndarray,
    classes_y: np.ndarray,
    nbins_x: int | None = None,
    nbins_y: int | None = None,
) -> float:
    """Nemenman-Shafee-Bialek Bayesian MI estimate (optional dep ``ndd``).

    Best for small sample sizes (n < 1000) where plug-in estimators are heavily biased. Slower than plug-in by ~50x.

    Raises
    ------
    ImportError
        If the ``ndd`` package is not installed. Install via ``pip install ndd``.
    """
    try:
        import ndd
    except ImportError as e:
        raise ImportError("nsb_mi requires the optional `ndd` package. " "Install via `pip install ndd`.") from e

    if nbins_x is None:
        nbins_x = int(classes_x.max()) + 1
    if nbins_y is None:
        nbins_y = int(classes_y.max()) + 1

    # Joint histogram.
    joint = np.zeros((nbins_x, nbins_y), dtype=np.int64)
    for cx, cy in zip(classes_x, classes_y):
        joint[cx, cy] += 1
    return float(ndd.mutual_information(joint))
