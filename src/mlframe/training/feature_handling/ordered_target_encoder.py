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

from typing import Dict, Mapping, Optional, Union

import numpy as np
import pandas as pd


def ordered_target_encode(
    categories: np.ndarray,
    y: np.ndarray,
    order: Optional[np.ndarray] = None,
    smoothing: float = 1.0,
    prior: Optional[float] = None,
    noise_std: float = 0.0,
    noise_count_halflife: Optional[float] = None,
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
    noise_count_halflife
        Optional opt-in schedule: when set (``> 0``), the EFFECTIVE noise std applied to a given row is
        decayed by that row's own causal running observation count for its category, ``effective_std =
        noise_std * 2 ** (-running_count / noise_count_halflife)`` -- a category's very first few occurrences
        (running-count expanding-mean statistic still a near-random point estimate) get close to the full
        ``noise_std``, while rows of a category that has already accumulated many prior observations (the
        expanding mean is already a stable, low-variance estimate) get progressively less noise instead of
        the same constant blur regardless of sample size. ``None`` (default) applies the constant ``noise_std``
        uniformly, exactly as before -- bit-identical to omitting this parameter.
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
        if noise_count_halflife is not None and noise_count_halflife > 0.0:
            running_count_full = np.empty(n, dtype=np.float64)
            running_count_full[sort_idx] = running_count.to_numpy()
            effective_std = noise_std * np.power(2.0, -running_count_full / noise_count_halflife)
            encoded = encoded * (1.0 + rng.normal(loc=0.0, scale=1.0, size=n) * effective_std)
        else:
            encoded = encoded * (1.0 + rng.normal(loc=0.0, scale=noise_std, size=n))

    return encoded


def ordered_target_encode_batch(
    categories_by_column: Mapping[str, np.ndarray],
    y: np.ndarray,
    order: Optional[np.ndarray] = None,
    smoothing: float = 1.0,
    prior: Optional[float] = None,
    noise_std: float = 0.0,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> Dict[str, np.ndarray]:
    """Encode several category columns that share the same ``y``/``order`` in one shared sort pass.

    When ``noise_std == 0.0`` this is bit-identical, column-by-column, to calling :func:`ordered_target_encode`
    separately for each column with the same ``y``, ``order``, ``smoothing`` and ``prior`` -- but the causal
    ``argsort`` of ``order`` and the global prior are computed exactly ONCE and reused across all columns,
    instead of once per column. Real-world caller: ``categorical_powerset_concat``'s ``prune_against_target``
    scores every generated composite column against the same ``(y, order)`` pair. When ``noise_std > 0.0`` each
    column still gets independent, reproducible noise (via ``SeedSequence`` spawning), but the exact draws
    differ from ``ordered_target_encode`` since a single ``random_state`` must fan out to N independent columns
    rather than one.

    Parameters
    ----------
    categories_by_column
        Mapping of column name -> ``(n,)`` categorical values, all aligned to the same ``y``/``order``.
    y, order, smoothing, prior, noise_std, random_state
        Same contract as :func:`ordered_target_encode`, shared across every column. ``random_state`` (when an
        int seed) is used to derive one independent ``np.random.Generator`` PER COLUMN via ``np.random.SeedSequence``
        spawning, so results don't depend on dict iteration order and don't collide across columns.

    Returns
    -------
    Dict[str, np.ndarray]
        Column name -> ``(n,)`` encoded values, same order as the INPUT arrays.
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.shape[0]
    if order is None:
        sort_idx = np.arange(n)
    else:
        sort_idx = np.argsort(np.asarray(order), kind="mergesort")

    global_prior = float(np.mean(y)) if prior is None else float(prior)
    sorted_y = y[sort_idx]

    if noise_std > 0.0:
        base_seed = random_state if not isinstance(random_state, np.random.Generator) else None
        seed_sequences = np.random.SeedSequence(base_seed).spawn(len(categories_by_column))
    else:
        seed_sequences = []

    result: Dict[str, np.ndarray] = {}
    for col_idx, (name, categories) in enumerate(categories_by_column.items()):
        categories = np.asarray(categories)
        sorted_cats = categories[sort_idx]

        df = pd.DataFrame({"cat": sorted_cats, "y": sorted_y})
        grouped = df.groupby("cat", sort=False)["y"]
        running_sum = grouped.cumsum() - sorted_y
        running_count = grouped.cumcount()

        encoded_sorted = (running_sum + smoothing * global_prior) / (running_count + smoothing)

        encoded = np.empty(n, dtype=np.float64)
        encoded[sort_idx] = encoded_sorted.to_numpy()

        if noise_std > 0.0:
            rng = np.random.default_rng(seed_sequences[col_idx])
            encoded = encoded * (1.0 + rng.normal(loc=0.0, scale=noise_std, size=n))

        result[name] = encoded

    return result


__all__ = ["ordered_target_encode", "ordered_target_encode_batch"]
