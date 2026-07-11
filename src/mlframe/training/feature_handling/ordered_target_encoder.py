"""CatBoost-style ordered (row-level expanding) target encoding.

The existing ``LeakageSafeEncoder`` with ``time_aware=True`` orders its FOLDS chronologically (shuffle=False
KFold) but each row within a fold is still encoded from that fold's aggregate train statistics -- coarser
than CatBoost's actual "ordered target statistics", which compute each row's encoding from a running
EXPANDING statistic over only the rows strictly BEFORE it in a specified order. This module implements that
true row-level expanding encoder: ``encoding_i = (running_target_sum_c + prior * smoothing) / (running_count_c
+ smoothing)``, computed causally as of row ``i`` -- a leak-free option natural for streaming/online data
where K-fold isn't the right mental model at all.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd


def ordered_target_encode(
    categories: np.ndarray,
    y: np.ndarray,
    order: Optional[np.ndarray] = None,
    smoothing: float = 1.0,
    prior: Optional[float] = None,
    noise_std: float = 0.0,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> np.ndarray:
    """Encode each row using only the target statistic accumulated from PRIOR rows of the same category.

    Parameters
    ----------
    categories
        ``(n,)`` categorical values to encode.
    y
        ``(n,)`` target aligned to ``categories``.
    order
        Optional ``(n,)`` ordering key (row index or timestamp); rows are processed in ascending order of
        this array. Defaults to the input row order (``np.arange(n)``) when ``None``.
    smoothing
        CatBoost's "a" parameter: pulls a category's encoding toward ``prior`` when its running count is
        small; ``encoding = (running_sum + smoothing * prior) / (running_count + smoothing)``.
    prior
        The global prior added for unseen/zero-count rows; defaults to the overall target mean.
    noise_std
        Standard deviation of additive Gaussian noise applied to each TRAINING-row encoding (multiplied by
        the encoding's own scale, i.e. relative noise: ``encoded * (1 + N(0, noise_std))``). Default ``0.0``
        (no noise). The classic regularization companion to expanding-mean encoding: even with the leak-free
        causal ordering, a model can still learn to exploit fine-grained encoding VALUE differences as a
        near-identity proxy for the category itself on high-cardinality columns; injecting noise blurs that
        without touching the underlying causal-ordering leakage guarantee.
    random_state
        Seed or ``np.random.Generator`` for the noise draw. Ignored when ``noise_std == 0.0``.

    Returns
    -------
    np.ndarray
        ``(n,)`` encoded values, same order as the INPUT arrays (not the sort order) -- a category's first-
        ever occurrence (in ``order``) gets exactly ``prior`` (zero running count), matching CatBoost's
        convention.
    """
    categories = np.asarray(categories)
    y = np.asarray(y, dtype=np.float64)
    n = categories.shape[0]
    if order is None:
        sort_idx = np.arange(n)
    else:
        sort_idx = np.argsort(np.asarray(order), kind="mergesort")

    global_prior = float(np.mean(y)) if prior is None else float(prior)

    sorted_cats = categories[sort_idx]
    sorted_y = y[sort_idx]

    df = pd.DataFrame({"cat": sorted_cats, "y": sorted_y})
    grouped = df.groupby("cat", sort=False)["y"]
    # cumsum/cumcount computed over the SORTED (causal-order) rows, then shifted by one row within each
    # category so row i sees only strictly-prior rows (cumsum up to and including i, minus y_i itself).
    running_sum = grouped.cumsum() - sorted_y
    running_count = grouped.cumcount()

    encoded_sorted = (running_sum + smoothing * global_prior) / (running_count + smoothing)

    encoded = np.empty(n, dtype=np.float64)
    encoded[sort_idx] = encoded_sorted.to_numpy()

    if noise_std > 0.0:
        rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
        encoded = encoded * (1.0 + rng.normal(loc=0.0, scale=noise_std, size=n))

    return encoded


__all__ = ["ordered_target_encode"]
