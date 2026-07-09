"""Out-of-fold loop for Mode A row-attention - mirrors ``bruteforce._kfold_target_encode``.

For each fold ``f`` of the user-supplied splitter:
    1. Re-fit the standardiser on ``X_train[train_idx_f]``  (per-fold to prevent leakage).
    2. Apply the FIXED projections (same across folds - they're transform-parameters, not folded artefacts).
    3. Build per-head hnswlib indices on the projected train-fold subset.
    4. Project the val-fold subset and query top-k.
    5. Run stage-4 (fused or njit) to produce per-head aggregates.
    6. Write outputs into the val-fold rows of the master output array.

Projections are fixed across folds because they parameterise the transform itself (analogous to "we're using the same scaler config across folds, only the
fit parameters change"). Re-randomising per fold would mix two sources of variance (the splitter's and the projection RNG's) and make the OOF features
non-comparable across folds.

The standardiser RE-fits per fold because it has fitted state and "fit per fold" matches the standard cross-validation pattern (sklearn KFold + Pipeline with
StandardScaler does the same internally via ``cross_val_predict``).

The ``splitter`` is REQUIRED by the public API (no default). For time-series data the caller passes ``TimeSeriesSplit``; for iid the caller passes ``KFold``. We
do not default to KFold because the majority of real tabular ML data has temporal structure and a silent KFold default leaks the past into the future.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

from ._aggregation import compute_extra_aggregates, dedupe_by_correlation
from ._projection import (
    apply_projection,
    build_importance_weighted_projection,
    build_nca_projection,
    build_random_projections,
    build_shap_weighted_projection,
    build_supervised_projections_pls,
)

# Aggregate names handled by the fused stage-4 kernel directly.
_BASE_AGGS = ("y_mean", "y_std", "x_mean")
# Aggregate names computed by the numpy-level post-processor (compute_extra_aggregates).
_EXTRA_AGGS = ("y_iqr", "y_skew", "x_centroid_dist")

logger = logging.getLogger(__name__)


def kfold_attention_loop(
    X_train: np.ndarray,
    y_train: np.ndarray,
    splitter: Any,
    *,
    seed: int,
    n_heads: int,
    head_dim: int,
    k: int,
    softmax_temp: float,
    aggregate: tuple[str, ...],
    standardize: bool,
    ann_M: int,
    ann_ef_construction: int,
    ann_ef_search: int,
    num_threads: int | None,
    stage4_callable: Callable[..., None],
    dtype: type = np.float32,
    projection: str = "random",
) -> dict[str, np.ndarray]:
    """Run the per-fold OOF loop and return the assembled output dict keyed by ``{aggregate}_h{head}``.

    Output shapes per key:
        ``y_mean_h{h}``  - (n_train,)
        ``y_std_h{h}``   - (n_train,)
        ``x_mean_h{h}``  - (n_train, head_dim)

    Each row is filled exactly once (in the fold where it appears in ``val_idx``). The splitter must cover every row at least once in ``val_idx`` (KFold and
    TimeSeriesSplit both satisfy this); shuffled splitters with overlap would over-write earlier folds' outputs harmlessly (last fold wins) but indicate caller
    confusion.

    ``stage4_callable`` is injected so the same loop body works on GPU (cupy fused RawKernel) or CPU (njit fallback) without branching here. The signature
    matches ``row_attention_stage4_njit`` / ``row_attention_stage4_cupy``: ``(q_proj, k_proj, y_train, topk_ids, softmax_temp, y_mean_out, y_std_out, x_mean_out)``.
    """
    from ._row_attention_ann import build_hnsw_index, query_topk

    n_train, d_input = X_train.shape

    # Output arrays sized to full train, filled per-fold.
    outputs: dict[str, np.ndarray] = {}
    for h in range(n_heads):
        if "y_mean" in aggregate:
            outputs[f"y_mean_h{h}"] = np.zeros(n_train, dtype=dtype)
        if "y_std" in aggregate:
            outputs[f"y_std_h{h}"] = np.zeros(n_train, dtype=dtype)
        if "x_mean" in aggregate:
            outputs[f"x_mean_h{h}"] = np.zeros((n_train, head_dim), dtype=dtype)
        for ex in _EXTRA_AGGS:
            if ex in aggregate:
                outputs[f"{ex}_h{h}"] = np.zeros(n_train, dtype=dtype)

    # Random projections are constructed ONCE, used unchanged across folds. PLS projections MUST be refit per fold (they use y_train, so a single global PLS fit
    # would leak the validation fold's targets through the projection matrix).
    if projection == "random":
        projections_global = build_random_projections(d_input=d_input, n_heads=n_heads, head_dim=head_dim, seed=seed, dtype=dtype)
    elif projection in ("pls", "importance", "shap", "nca"):
        projections_global = None  # rebuilt per fold (importance/SHAP fit aux LGB per fold; PLS/NCA fit per fold)
    else:
        raise ValueError(f"projection must be 'random', 'pls', 'importance', 'shap', or 'nca'; got {projection!r}.")

    splits = list(splitter.split(X_train))
    n_folds = len(splits)
    logger.info("kfold_attention_loop: %d folds, n_train=%d, n_heads=%d, head_dim=%d, k=%d, projection=%s", n_folds, n_train, n_heads, head_dim, k, projection)

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        t_fold0 = _now()
        X_tr = X_train[train_idx]
        X_va = X_train[val_idx]
        y_tr = y_train[train_idx]

        # Per-fold standardiser refit.
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(X_tr)
            X_tr_s = scaler.transform(X_tr).astype(dtype, copy=False)
            X_va_s = scaler.transform(X_va).astype(dtype, copy=False)
        else:
            X_tr_s = X_tr.astype(dtype, copy=False)
            X_va_s = X_va.astype(dtype, copy=False)

        # Build projections (random reuses global; PLS / importance / SHAP refit on this fold's train subset only).
        if projection == "pls":
            projections = build_supervised_projections_pls(
                X=X_tr_s, y=y_tr, n_heads=n_heads, head_dim=head_dim,
                seed=int(seed) + fold_idx,
                dtype=dtype,
            )
        elif projection == "importance":
            projections = build_importance_weighted_projection(
                X=X_tr_s, y=y_tr, n_heads=n_heads, head_dim=head_dim,
                seed=int(seed) + fold_idx,
                dtype=dtype,
            )
        elif projection == "shap":
            projections = build_shap_weighted_projection(
                X=X_tr_s, y=y_tr, n_heads=n_heads, head_dim=head_dim,
                seed=int(seed) + fold_idx,
                dtype=dtype,
            )
        elif projection == "nca":
            projections = build_nca_projection(
                X=X_tr_s, y=y_tr, n_heads=n_heads, head_dim=head_dim,
                seed=int(seed) + fold_idx,
                dtype=dtype,
            )
        else:
            assert projections_global is not None  # projection == "random" is the only path reaching this else, and that branch built projections_global above
            projections = projections_global

        # Project both fold subsets with the projection matrices.
        k_proj_tr = apply_projection(X_tr_s, projections, l2_normalize=True)  # (n_heads, |train_idx|, head_dim)
        q_proj_va = apply_projection(X_va_s, projections, l2_normalize=True)  # (n_heads, |val_idx|,   head_dim)

        for h in range(n_heads):
            # Build per-head ANN index on this fold's train subset.
            index = build_hnsw_index(
                k_proj_tr[h], M=ann_M, ef_construction=ann_ef_construction, num_threads=num_threads,
            )
            topk_ids, _dists = query_topk(index, q_proj_va[h], k=k, ef_search=ann_ef_search, num_threads=num_threads)

            # Stage 4: fused gather + softmax + aggregates.
            y_mean_v = np.empty(val_idx.shape[0], dtype=np.float32)
            y_std_v = np.empty(val_idx.shape[0], dtype=np.float32)
            x_mean_v = np.empty((val_idx.shape[0], head_dim), dtype=np.float32)
            stage4_callable(
                q_proj_va[h], k_proj_tr[h], y_tr.astype(np.float32, copy=False),
                topk_ids, softmax_temp,
                y_mean_v, y_std_v, x_mean_v,
            )

            # Scatter into master outputs at val_idx.
            if "y_mean" in aggregate:
                outputs[f"y_mean_h{h}"][val_idx] = y_mean_v.astype(dtype, copy=False)
            if "y_std" in aggregate:
                outputs[f"y_std_h{h}"][val_idx] = y_std_v.astype(dtype, copy=False)
            if "x_mean" in aggregate:
                outputs[f"x_mean_h{h}"][val_idx] = x_mean_v.astype(dtype, copy=False)

            # Extra aggregates (y_iqr, y_skew, x_centroid_dist): post-processed in numpy after the fused kernel.
            extra_aggs = tuple(a for a in aggregate if a in _EXTRA_AGGS)
            if extra_aggs:
                extras = compute_extra_aggregates(
                    q_proj=q_proj_va[h], k_proj=k_proj_tr[h], y_train=y_tr.astype(np.float32, copy=False),
                    topk_ids=topk_ids, softmax_temp=softmax_temp, aggregates=extra_aggs,
                )
                for agg_name, arr in extras.items():
                    outputs[f"{agg_name}_h{h}"][val_idx] = arr.astype(dtype, copy=False)
        logger.info("kfold_attention_loop: fold %d/%d completed in %.1fs", fold_idx + 1, n_folds, _now() - t_fold0)

    return outputs


def stack_outputs_to_array(
    outputs: dict[str, np.ndarray],
    aggregate: tuple[str, ...],
    n_heads: int,
    head_dim: int,
) -> tuple[np.ndarray, list[str]]:
    """Concatenate per-head, per-aggregate outputs into a single (N, F) array with deterministic column names.

    Column ordering: heads outermost, then aggregate type, then per-aggregate dim. Example with n_heads=2, head_dim=4, aggregate=(y_mean, y_std, x_mean):

        h0_y_mean, h0_y_std, h0_x_mean_d0, h0_x_mean_d1, h0_x_mean_d2, h0_x_mean_d3,
        h1_y_mean, h1_y_std, h1_x_mean_d0, h1_x_mean_d1, h1_x_mean_d2, h1_x_mean_d3

    Predictable order matters: downstream model feature importance / SHAP plots cite column names, so callers (and us in tests) need a stable contract.
    """
    cols: list[np.ndarray] = []
    names: list[str] = []
    for h in range(n_heads):
        for agg in aggregate:
            key = f"{agg}_h{h}"
            arr = outputs[key]
            if arr.ndim == 1:
                cols.append(arr.reshape(-1, 1))
                names.append(f"attn_h{h}_{agg}")
            else:
                cols.append(arr)
                names.extend(f"attn_h{h}_{agg}_d{d}" for d in range(arr.shape[1]))
    matrix = np.concatenate(cols, axis=1)
    return matrix, names


def apply_dedupe(
    matrix: np.ndarray,
    names: list[str],
    dedupe_threshold: float | None,
) -> tuple[np.ndarray, list[str]]:
    """Drop near-duplicate columns via ``dedupe_by_correlation`` if threshold is set.

    Returns the filtered ``(matrix, names)`` tuple. Logs the count of dropped columns at INFO so callers see what was removed without enabling DEBUG.
    """
    if dedupe_threshold is None:
        return matrix, names
    keep = dedupe_by_correlation(matrix, threshold=dedupe_threshold)
    n_dropped = int((~keep).sum())
    if n_dropped > 0:
        logger.info("apply_dedupe: dropped %d / %d columns at |corr| > %.3f", n_dropped, matrix.shape[1], dedupe_threshold)
    return matrix[:, keep], [n for n, keep_i in zip(names, keep) if keep_i]


def _now() -> float:
    """Monotonic wall-clock reading used for stage-duration logging in ``kfold_attention_loop``; local import keeps ``time`` off the module's hot import path."""
    import time
    return time.perf_counter()
