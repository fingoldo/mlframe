"""Public API: ``compute_row_attention`` + low-level ``build_key_bank`` / ``attend`` pair.

Mode A (OOF on train):  ``compute_row_attention(X_train, y_train, X_query=None, splitter=KFold(5), ...)``
Mode B (inference):     ``compute_row_attention(X_train, y_train, X_query=X_val, splitter=KFold(5), ...)``  (splitter ignored; key-bank built once)

The low-level pair is for inference-pipeline reuse: build the key-bank once via ``build_key_bank``, then call ``attend(bank, X_query, ...)`` repeatedly for every
incoming batch without rebuilding the 10M-row hnswlib index each time.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np
import polars as pl

from ._key_bank import KeyBank, _key_bank_fingerprint, save_key_bank, try_load_key_bank
from ._kernels_njit import row_attention_stage4_njit
from ._oof import apply_dedupe, kfold_attention_loop, stack_outputs_to_array
from ._projection import (
    apply_projection,
    build_importance_weighted_projection,
    build_nca_projection,
    build_random_projections,
    build_shap_weighted_projection,
    build_supervised_projections_pls,
    validate_projection_dims,
)
from ._row_attention_ann import build_hnsw_index, query_topk
from ._utils import free_gpu_memory_pool, is_gpu_available, require_seed, validate_numeric_input

logger = logging.getLogger(__name__)

# Default ANN hyperparameters - documented in compute_row_attention; exposed as constants here so tests / benches can reference the same values.
DEFAULT_ANN_M = 16
DEFAULT_ANN_EF_CONSTRUCTION = 200
DEFAULT_ANN_EF_SEARCH = 64


def compute_row_attention(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Any,
    *,
    seed: int,
    n_heads: int = 4,
    head_dim: int = 8,
    k: int = 32,
    metric: Literal["cosine"] = "cosine",
    softmax_temp: float = 1.0,
    aggregate: tuple[str, ...] = ("y_mean", "y_std"),
    standardize: bool = True,
    projection: Literal["random", "pls", "importance", "shap", "nca"] = "random",
    k_scales: tuple[int, ...] = (),
    gpu_stage4: Union[bool, Literal["auto"]] = "auto",
    keep_key_bank_on_gpu: bool = False,
    ann_M: int = DEFAULT_ANN_M,
    ann_ef_construction: int = DEFAULT_ANN_EF_CONSTRUCTION,
    ann_ef_search: int = DEFAULT_ANN_EF_SEARCH,
    num_threads: int | None = None,
    dtype: type = np.float32,
    cache_dir: Path | None = None,
    release_memory_after: bool = True,
    dedupe_threshold: float | None = 0.99,
    allow_overcomplete: bool = False,
    column_prefix: str = "attn",
) -> pl.DataFrame:
    """Multi-head softmax-weighted kNN-target-encoding over random-subspace projections.

    Pipeline (4 stages; backend per stage is fixed):
        1. CPU: validate, standardise (RobustScaler).
        2. CPU: project X via per-head random Gaussian matrices to (n_heads, N, head_dim).
        3. CPU: per-head hnswlib build + top-k query.
        4. GPU (or CPU fallback): fused gather + softmax + weighted-sum-V producing per-head y_mean / y_std / x_mean.

    Mode A vs Mode B:
        - ``X_query is None``  -> OOF on X_train via ``splitter``.   Each train row attends only to non-self training rows in its fold's complement.
        - ``X_query is not None`` -> single-pass: key-bank built once from full X_train; query rows attend to all training rows.

    Returns a polars DataFrame with deterministic column naming ``{column_prefix}_h{head}_{agg}`` (scalar) or ``{column_prefix}_h{head}_{agg}_d{dim}`` (vector
    aggregates like ``x_mean``). Row order matches the relevant input (X_train rows for Mode A, X_query rows for Mode B).

    Leakage rules (see test_leakage_row_attention.py):
        - seed must be a literal int (raises if derived from data).
        - Projections are fixed across folds; standardiser refits per fold.
        - In Mode A, val rows of fold f attend only to train rows of fold f's complement.

    Hardware notes:
        - ``gpu_stage4='auto'`` dispatches stage 4 to GPU if cupy is available; the other three stages are CPU.
        - ``keep_key_bank_on_gpu=True`` (Mode B only) keeps the projected K-bank in GPU memory across multiple ``attend`` calls; ~10x speedup for production
          inference with many query batches.
        - At N=10M, d=20k: Mode A takes ~1-2 hours (5x hnswlib build dominates); Mode B takes ~30-45 min one-time, then ~12 s per 1M-query batch.
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=True)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=True)
        if X_query.shape[1] != X_train.shape[1]:
            raise ValueError(f"X_query has d={X_query.shape[1]} but X_train has d={X_train.shape[1]}.")
    if y_train.ndim != 1:
        raise ValueError(f"y_train must be 1-D for v1 (multi-output deferred to v1.1); got shape {y_train.shape}.")
    if y_train.shape[0] != X_train.shape[0]:
        raise ValueError(f"y_train length {y_train.shape[0]} != X_train rows {X_train.shape[0]}.")
    if k < 1 or k > X_train.shape[0]:
        raise ValueError(f"k must be in [1, n_train]; got {k} with n_train={X_train.shape[0]}.")
    if metric != "cosine":
        raise ValueError(f"metric='{metric}' not in v1; only 'cosine' is supported (multi-metric cycling dropped per ML #9 critique).")
    if softmax_temp <= 0:
        raise ValueError(f"softmax_temp must be positive; got {softmax_temp}.")
    valid_aggs = ("y_mean", "y_std", "x_mean", "y_iqr", "y_skew", "x_centroid_dist")
    for agg in aggregate:
        if agg not in valid_aggs:
            raise ValueError(f"aggregate item '{agg}' not in {valid_aggs}.")

    validate_projection_dims(X_train.shape[1], head_dim, allow_overcomplete=allow_overcomplete)

    stage4_callable = _select_stage4_backend(gpu_stage4=gpu_stage4, n_queries_hint=(X_query.shape[0] if X_query is not None else X_train.shape[0] // max(1, _splitter_n_splits(splitter))))

    # k_scales support: when set, run the whole pipeline for each k in (k,) + k_scales, then concat the outputs side-by-side. This captures local + medium +
    # global similarity simultaneously. When k_scales is empty (default), use just the single k.
    ks_to_run = (k,) + tuple(k_scales) if k_scales else (k,)
    all_outputs: dict[str, np.ndarray] = {}
    all_names: list[str] = []

    for k_value in ks_to_run:
        if X_query is None:
            outputs = kfold_attention_loop(
                X_train=X_train, y_train=y_train, splitter=splitter,
                seed=seed, n_heads=n_heads, head_dim=head_dim, k=k_value, softmax_temp=softmax_temp,
                aggregate=aggregate, standardize=standardize,
                ann_M=ann_M, ann_ef_construction=ann_ef_construction, ann_ef_search=ann_ef_search,
                num_threads=num_threads, stage4_callable=stage4_callable, dtype=dtype,
                projection=projection,
            )
        else:
            bank = build_key_bank(
                X_train=X_train, y_train=y_train, seed=seed, n_heads=n_heads, head_dim=head_dim,
                standardize=standardize, ann_M=ann_M, ann_ef_construction=ann_ef_construction,
                num_threads=num_threads, dtype=dtype, cache_dir=cache_dir, projection=projection,
            )
            if keep_key_bank_on_gpu and is_gpu_available():
                bank.to_device()
            try:
                outputs = attend(
                    bank=bank, X_query=X_query, k=k_value, softmax_temp=softmax_temp, aggregate=aggregate,
                    ann_ef_search=ann_ef_search, num_threads=num_threads, stage4_callable=stage4_callable, dtype=dtype,
                )
            finally:
                # Wave 52 (2026-05-20): wrap free_device in try/except so CUDA cleanup
                # errors on a broken context (e.g. after attend() OOM) don't mask the
                # original failure the operator needs to see.
                if keep_key_bank_on_gpu:
                    try:
                        bank.free_device()
                    except Exception as _free_err:
                        import logging as _lg
                        _lg.getLogger(__name__).warning(
                            "row_attention: bank.free_device() failed (likely after upstream CUDA error): %s",
                            _free_err,
                        )
        # Tag each k-scale's output keys with the k value to keep column names unique across scales. The tag must be a PREFIX (k32_y_mean_h0) so the
        # ``stack_outputs_to_array`` lookup pattern ``f"{agg}_h{h}"`` continues to find the right key (k32_y_mean is the aggregate, h0 is appended by the stack).
        if len(ks_to_run) > 1:
            outputs = {f"k{k_value}_{key}": arr for key, arr in outputs.items()}
            agg_for_stack = tuple(f"k{k_value}_{a}" for a in aggregate)
        else:
            agg_for_stack = aggregate
        matrix_k, names_k = stack_outputs_to_array(outputs, aggregate=agg_for_stack, n_heads=n_heads, head_dim=head_dim)
        all_outputs[f"_matrix_k{k_value}"] = matrix_k
        all_names.extend(names_k)

    # Concatenate all scales horizontally.
    matrix = np.concatenate([all_outputs[f"_matrix_k{k}"] for k in ks_to_run], axis=1)
    names = all_names
    matrix, names = apply_dedupe(matrix, names, dedupe_threshold=dedupe_threshold)
    # Rebrand column prefix per caller.
    names = [n.replace("attn_", f"{column_prefix}_") if column_prefix != "attn" else n for n in names]

    if release_memory_after:
        free_gpu_memory_pool()

    return pl.DataFrame({n: matrix[:, i] for i, n in enumerate(names)})


def build_key_bank(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    n_heads: int = 4,
    head_dim: int = 8,
    standardize: bool = True,
    projection: Literal["random", "pls", "importance", "shap", "nca"] = "random",
    ann_M: int = DEFAULT_ANN_M,
    ann_ef_construction: int = DEFAULT_ANN_EF_CONSTRUCTION,
    num_threads: int | None = None,
    dtype: type = np.float32,
    cache_dir: Path | None = None,
) -> KeyBank:
    """Build (and optionally cache to disk) the projected K-bank + per-head hnswlib indices for the FULL X_train.

    Used by Mode B and by external callers who want to drive multiple ``attend`` calls without re-paying the build cost. For Mode A, the OOF loop builds a fresh
    bank per fold internally (it cannot share this single full-train bank, by definition).

    Disk-cache:
        ``cache_dir/<fingerprint>/`` holds ``projections.npy + k_proj.npy + y_train.npy + ann_h{h}.bin + metadata.pkl``. Fingerprint = sha256 of X_train bytes +
        all build parameters. A cache hit on a 10M-row build saves 10-30 minutes; a cache miss writes atomically (tmp dir + rename).
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=True)
    validate_projection_dims(X_train.shape[1], head_dim)

    if cache_dir is not None:
        fp = _key_bank_fingerprint(
            X_train=X_train, seed=seed, n_heads=n_heads, head_dim=head_dim,
            metric="cosine", standardize=standardize, ann_M=ann_M, ann_ef_construction=ann_ef_construction,
        )
        cached = try_load_key_bank(cache_dir, fp)
        if cached is not None:
            logger.info("build_key_bank: cache hit at %s/%s", cache_dir, fp)
            return cached

    n_train, d_input = X_train.shape

    if standardize:
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler().fit(X_train)
        X_std = scaler.transform(X_train).astype(dtype, copy=False)
    else:
        scaler = None
        X_std = X_train.astype(dtype, copy=False)

    if projection == "pls":
        projections = build_supervised_projections_pls(
            X=X_std, y=y_train, n_heads=n_heads, head_dim=head_dim, seed=seed, dtype=dtype,
        )
    elif projection == "importance":
        projections = build_importance_weighted_projection(
            X=X_std, y=y_train, n_heads=n_heads, head_dim=head_dim, seed=seed, dtype=dtype,
        )
    elif projection == "shap":
        projections = build_shap_weighted_projection(
            X=X_std, y=y_train, n_heads=n_heads, head_dim=head_dim, seed=seed, dtype=dtype,
        )
    elif projection == "nca":
        projections = build_nca_projection(
            X=X_std, y=y_train, n_heads=n_heads, head_dim=head_dim, seed=seed, dtype=dtype,
        )
    elif projection == "random":
        projections = build_random_projections(d_input=d_input, n_heads=n_heads, head_dim=head_dim, seed=seed, dtype=dtype)
    else:
        raise ValueError(f"projection must be 'random', 'pls', 'importance', 'shap', or 'nca'; got {projection!r}.")
    k_proj = apply_projection(X_std, projections, l2_normalize=True)  # (n_heads, n_train, head_dim)

    ann_indices: list[Any] = []
    for h in range(n_heads):
        logger.info("build_key_bank: building hnswlib index for head %d/%d (n_train=%d, head_dim=%d)", h + 1, n_heads, n_train, head_dim)
        index = build_hnsw_index(k_proj[h], M=ann_M, ef_construction=ann_ef_construction, num_threads=num_threads)
        ann_indices.append(index)

    bank = KeyBank(
        projections=projections, k_proj=k_proj, y_train=y_train.astype(np.float32, copy=False),
        ann_indices=ann_indices, standardiser=scaler, seed=seed,
    )

    if cache_dir is not None:
        save_key_bank(bank, cache_dir, fp, ann_space="cosine")

    return bank


def attend(
    bank: KeyBank,
    X_query: np.ndarray,
    *,
    k: int = 32,
    softmax_temp: float = 1.0,
    aggregate: tuple[str, ...] = ("y_mean", "y_std"),
    ann_ef_search: int = DEFAULT_ANN_EF_SEARCH,
    num_threads: int | None = None,
    stage4_callable: Any = None,
    dtype: type = np.float32,
) -> dict[str, np.ndarray]:
    """Score a batch of query rows against the pre-built bank. Returns the same output-dict shape as ``kfold_attention_loop``.

    ``stage4_callable`` may be None - in which case we re-select per the GPU policy (auto). For repeated inference calls it's cheaper to pass the same callable
    explicitly so the auto-detection doesn't re-run per call.

    If ``bank.k_proj_device`` is set (caller previously did ``bank.to_device()``), the cupy stage 4 dispatch uses the device-resident bank arrays, skipping H2D.
    """
    validate_numeric_input(X_query, name="X_query", allow_fp16=True)
    if X_query.shape[1] != bank.d_input:
        raise ValueError(f"X_query has d={X_query.shape[1]} but bank.d_input={bank.d_input}.")
    if stage4_callable is None:
        stage4_callable = _select_stage4_backend(gpu_stage4="auto", n_queries_hint=X_query.shape[0])

    if bank.standardiser is not None:
        X_q_std = bank.standardiser.transform(X_query).astype(dtype, copy=False)
    else:
        X_q_std = X_query.astype(dtype, copy=False)

    q_proj = apply_projection(X_q_std, bank.projections, l2_normalize=True)  # (n_heads, n_query, head_dim)
    n_query = X_query.shape[0]

    outputs: dict[str, np.ndarray] = {}
    for h in range(bank.n_heads):
        topk_ids, _dists = query_topk(bank.ann_indices[h], q_proj[h], k=k, ef_search=ann_ef_search, num_threads=num_threads)
        y_mean_v = np.empty(n_query, dtype=np.float32)
        y_std_v = np.empty(n_query, dtype=np.float32)
        x_mean_v = np.empty((n_query, bank.head_dim), dtype=np.float32)

        # If the bank is GPU-resident AND we're using the cupy stage 4, pass device arrays through to avoid re-upload.
        if bank.k_proj_device is not None and stage4_callable.__name__ == "row_attention_stage4_cupy":
            stage4_callable(
                q_proj[h], bank.k_proj[h], bank.y_train, topk_ids, softmax_temp,
                y_mean_v, y_std_v, x_mean_v,
                k_proj_device=bank.k_proj_device[h], y_train_device=bank.y_train_device,
            )
        else:
            stage4_callable(
                q_proj[h], bank.k_proj[h], bank.y_train, topk_ids, softmax_temp,
                y_mean_v, y_std_v, x_mean_v,
            )

        if "y_mean" in aggregate:
            outputs[f"y_mean_h{h}"] = y_mean_v.astype(dtype, copy=False)
        if "y_std" in aggregate:
            outputs[f"y_std_h{h}"] = y_std_v.astype(dtype, copy=False)
        if "x_mean" in aggregate:
            outputs[f"x_mean_h{h}"] = x_mean_v.astype(dtype, copy=False)
        # Extra aggregates: post-process in numpy.
        extra_aggs = tuple(a for a in aggregate if a in ("y_iqr", "y_skew", "x_centroid_dist"))
        if extra_aggs:
            from ._aggregation import compute_extra_aggregates
            extras = compute_extra_aggregates(
                q_proj=q_proj[h], k_proj=bank.k_proj[h], y_train=bank.y_train,
                topk_ids=topk_ids, softmax_temp=softmax_temp, aggregates=extra_aggs,
            )
            for agg_name, arr in extras.items():
                outputs[f"{agg_name}_h{h}"] = arr.astype(dtype, copy=False)
    return outputs


def _select_stage4_backend(
    *,
    gpu_stage4: Union[bool, Literal["auto"]],
    n_queries_hint: int,
) -> Any:
    """Pick the stage-4 callable based on ``gpu_stage4`` and runtime availability.

    For ``gpu_stage4='auto'``, GPU wins essentially always when available (stage 4 is bandwidth-bound, fused RawKernel beats cupy primitives by 3-9x). The
    ``n_queries_hint`` argument lets us future-proof: at very small n_queries (<256) kernel launch overhead approaches the work, so we may want to fall back to
    CPU in that corner. v1 always picks GPU when available; revisit if profiling shows the threshold matters.
    """
    # Wave 28 P0 fix (2026-05-20): symmetric with the random_features.py
    # use_gpu fix. ``is True``/``is False`` silently rejected
    # ``np.bool_(True/False)`` from config dicts; the bool() coerce
    # accepts both Python bool and numpy bool uniformly.
    if isinstance(gpu_stage4, str):
        if gpu_stage4 != "auto":
            raise ValueError(f"gpu_stage4 must be True, False, or 'auto'; got {gpu_stage4!r}.")
        # Fall through to auto-dispatch below.
    else:
        _flag = bool(gpu_stage4)
        if not _flag:
            return row_attention_stage4_njit
        if not is_gpu_available():
            logger.info("gpu_stage4=True but GPU is not available; using CPU njit stage 4.")
            return row_attention_stage4_njit
        from ._kernels_cupy import row_attention_stage4_cupy
        return row_attention_stage4_cupy
    if not is_gpu_available():
        return row_attention_stage4_njit
    # v1 corner: skip GPU when n_queries is so small that launch overhead dominates.
    if n_queries_hint < 256:
        return row_attention_stage4_njit
    from ._kernels_cupy import row_attention_stage4_cupy
    return row_attention_stage4_cupy


def _splitter_n_splits(splitter: Any) -> int:
    """Best-effort: ask the splitter how many folds it will produce. Falls back to 5 if the splitter doesn't expose ``n_splits``."""
    n = getattr(splitter, "n_splits", None)
    if isinstance(n, int) and n > 0:
        return n
    return 5
