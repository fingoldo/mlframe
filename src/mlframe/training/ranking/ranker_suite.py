"""Top-level Learning-to-Rank training suite.

Mirrors the shape of ``train_mlframe_models_suite`` but with LTR-specific
dispatch:

- Accepts the same ``df`` + ``FeaturesAndTargetsExtractor`` inputs.
- ``mlframe_models`` is filtered to ``{cb, xgb, lgb}`` (HGB / Linear have
  no native ranker; warns and drops).
- ``group_ids`` from the FTE's ``group_field`` is REQUIRED. Raises a
  helpful error otherwise.
- Splits via ``make_train_test_split(groups=...)`` so query integrity is
  preserved end-to-end.
- For each model: ``fit_ranker`` + ``predict_ranker_scores`` + ranking
  metrics (NDCG@k, MAP@k, MRR).
- When ``use_mlframe_ensembles=True``, runs RRF / Borda ensembling over
  the model scores.
- Saves models via joblib for sklearn-wrapper consistency. Metadata
  (target_type, ranking_config, per-model NDCG@k) persisted alongside.

This is invoked by ``train_mlframe_models_suite`` when the target type
resolves to ``LEARNING_TO_RANK``; users can also call it directly.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _filter_models_for_ranking(mlframe_models: list[str] | None) -> list[str]:
    """Drop models without native rankers (HGB, Linear, etc.) with WARN.

    Supported: CatBoost / XGBoost / LightGBM ship native rankers; MLP gets
    a custom RankNet/ListNet pairwise/listwise loss
    (mlframe.training.neural.ranker.MLPRanker).
    """
    if not mlframe_models:
        return ["cb", "xgb", "lgb"]
    requested = [m.lower() for m in mlframe_models]
    supported = {"cb", "xgb", "lgb", "mlp"}
    kept = [m for m in requested if m in supported]
    dropped = [m for m in requested if m not in supported]
    if dropped:
        logger.warning(
            "LTR target_type: dropping %d model(s) without native ranker: %s. "
            "Supported rankers: cb / xgb / lgb (native) + mlp (RankNet/ListNet); "
            "HGB / Linear / sklearn would need a regression-then-rerank "
            "workaround (not implemented). Surviving models: %s",
            len(dropped), dropped, kept,
        )
    if not kept:
        raise NotImplementedError(f"LTR target_type: every requested model {requested} lacks a " "native ranker. Pick at least one of cb / xgb / lgb / mlp.")
    return kept


def _strategy_for_model(model_name: str):
    """Return the strategy instance for a given model short-tag."""
    from mlframe.training.strategies import (
        CatBoostStrategy, XGBoostStrategy, TreeModelStrategy, NeuralNetStrategy,
    )
    name = model_name.lower()
    if name == "cb":
        return CatBoostStrategy()
    if name == "xgb":
        return XGBoostStrategy()
    if name == "lgb":
        return TreeModelStrategy()
    if name == "mlp":
        return NeuralNetStrategy()
    raise ValueError(f"unknown ranker model {model_name!r}")


def rrf_fuse(ranks_per_member: list[np.ndarray], k: int = 60) -> np.ndarray:
    """Reciprocal Rank Fusion of per-member 1-based ranks.

    Each input ``ranks_per_member[m]`` is a 1-D length-N array of 1-based
    ranks (rank=1 best). The aggregated per-item score is
    ``sum_m 1 / (k + rank_m(item))``. Higher aggregated score = more relevant.

    Scale-invariant: the rank-position transform discards the raw score
    distribution, so heterogeneous score scales (CB sigmoid vs XGB logit vs
    LGB lambdarank) blend safely without external calibration. TREC-standard
    ``k=60``.

    Parameters
    ----------
    ranks_per_member : list of (N,) integer arrays
        Each array is the per-item rank assigned by one base ranker. Within
        each query group the caller is responsible for producing dense
        1-based ranks; this helper does NOT recompute ranks from scores.
        Use ``ensemble_ranker_scores`` from ``mlframe.training.ranking`` when
        you have raw scores instead of ranks.
    k : int
        RRF damping constant. ``k=60`` is the TREC default; smaller
        emphasises the top of each ranking, larger flattens.

    Returns
    -------
    np.ndarray, shape (N,)
        Per-item aggregated RRF score, higher = better.
    """
    if not ranks_per_member:
        raise ValueError("rrf_fuse: ranks_per_member is empty")
    if k <= 0:
        raise ValueError(f"rrf_fuse: k must be > 0, got {k!r}")
    base = np.asarray(ranks_per_member[0])
    if base.ndim != 1:
        raise ValueError(f"rrf_fuse: ranks must be 1-D, member 0 has ndim={base.ndim}")
    aggregated = np.zeros(base.shape[0], dtype=np.float64)
    for i, r in enumerate(ranks_per_member):
        arr = np.asarray(r, dtype=np.float64)
        if arr.shape != base.shape:
            raise ValueError(f"rrf_fuse: member {i} shape {arr.shape!r} != member 0 shape {base.shape!r}")
        if np.any(arr <= 0):
            raise ValueError(f"rrf_fuse: member {i} contains non-positive ranks; expected 1-based dense ranks (>= 1)")
        aggregated += 1.0 / (k + arr)
    return aggregated


def borda_fuse(
    ranks_per_member: list[np.ndarray],
    group_sizes: np.ndarray | None = None,
) -> np.ndarray:
    """Borda count fusion of per-member ranks.

    Per item ``score = sum_m (n_items_in_group - rank_m(item))``. Higher score
    = more relevant. Like RRF, Borda is scale-invariant -- it operates purely
    on rank positions -- but it underweights the bottom of long lists less
    aggressively than RRF (linear vs reciprocal), so it can be a better fit
    when the full ranking matters (e.g. recall-oriented LTR) rather than
    just the head.

    Parameters
    ----------
    ranks_per_member : list of (N,) integer arrays
        1-based dense ranks within each query group. Same contract as
        ``rrf_fuse``.
    group_sizes : (N,) array, optional
        Per-row size of the query group the row belongs to (so each row
        knows the ``n_items_in_group`` to subtract its rank from). When
        omitted, the helper assumes one global group of size N (single-query
        ranking).

    Returns
    -------
    np.ndarray, shape (N,)
        Per-item aggregated Borda score, higher = better.
    """
    if not ranks_per_member:
        raise ValueError("borda_fuse: ranks_per_member is empty")
    base = np.asarray(ranks_per_member[0])
    if base.ndim != 1:
        raise ValueError(f"borda_fuse: ranks must be 1-D, member 0 has ndim={base.ndim}")
    n = base.shape[0]
    if group_sizes is None:
        sizes = np.full(n, n, dtype=np.float64)
    else:
        sizes = np.asarray(group_sizes, dtype=np.float64)
        if sizes.shape != base.shape:
            raise ValueError(f"borda_fuse: group_sizes shape {sizes.shape!r} != ranks shape {base.shape!r}")

    aggregated = np.zeros(n, dtype=np.float64)
    for i, r in enumerate(ranks_per_member):
        arr = np.asarray(r, dtype=np.float64)
        if arr.shape != base.shape:
            raise ValueError(f"borda_fuse: member {i} shape {arr.shape!r} != member 0 shape {base.shape!r}")
        if np.any(arr <= 0):
            raise ValueError(f"borda_fuse: member {i} contains non-positive ranks; expected 1-based dense ranks (>= 1)")
        # C-Low-2: a rank exceeding its group size (sizes - arr < 0) means the caller passed a stale
        # rank under a reduced group_sizes vector, producing a negative Borda contribution that flips
        # the score sign and confuses operators. Refuse loudly rather than aggregating silently.
        if np.any(arr > sizes):
            _bad = int(np.argmax(arr > sizes))
            raise ValueError(
                f"borda_fuse: member {i} rank {float(arr[_bad])!r} at row {_bad} exceeds its group "
                f"size {float(sizes[_bad])!r}. Either rerank within each group or update group_sizes "
                "to match the rank vector."
            )
        aggregated += sizes - arr
    return aggregated


__all__ = ["train_mlframe_ranker_suite", "rrf_fuse", "borda_fuse"]


# ----------------------------------------------------------------------
# Sibling-module re-export. The 906-LOC ``train_mlframe_ranker_suite``
# body lives in ``_ranker_suite_train.py`` so this file stays below the
# 1k-LOC monolith threshold.
# ----------------------------------------------------------------------
from ._ranker_suite_train import train_mlframe_ranker_suite  # noqa: E402,F401
