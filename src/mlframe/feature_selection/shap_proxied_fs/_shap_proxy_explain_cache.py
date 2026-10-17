"""Outer ``shap_phi_`` disk-cache helpers for ``compute_shap_matrix`` (carved out of ``_shap_proxy_explain`` to keep it under the size budget)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger("mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain")


def _compute_shap_matri_out_fold_true_splitter(out_of_fold: bool, n_models: int, rng: np.random.Generator, n_splits: int) -> None:
    """Block of compute_shap_matrix starting at ``if not out_of_fold:``."""
    if not out_of_fold:
        for _ in range(n_models):
            rng.integers(0, 2**31 - 1)
    else:
        rng.integers(0, 2**31 - 1)
        for _ in range(n_splits * n_models):
            rng.integers(0, 2**31 - 1)


def _open_shap_disk_cache(
    cache_dir: Optional[Union[str, Path]],
    model_template: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    rng: np.random.Generator,
    *,
    classification: bool,
    out_of_fold: bool,
    n_splits: int,
    n_models: int,
    config_jitter: bool,
    return_variance: bool,
    return_per_fold_phi_mean: bool,
    shap_backend: str,
    n_estimators_cap: Optional[int],
    inner_n_jobs_cap: bool,
) -> tuple[Any, Optional[str], Any]:
    """Open the outer ``shap_phi_`` disk cache and look up the call; returns ``(cache, cache_key, hit)`` (``hit`` is None on a miss or when disabled)."""
    # Content-addressable disk cache: keyed by (X summary, y summary, booster params, fold/model
    # state). The OOF-SHAP stage dominates wall-clock at C3/C4; in hyperparam sweeps and ablations
    # the same (X, y, template) tuple recurs verbatim, so a hit returns phi + base + tail without
    # paying the per-fold xgboost fit + TreeSHAP attribution cost. ``cache_dir=None`` skips entirely
    # (default, zero behaviour change). The RNG bit-stream is snapshotted at entry so the cache key
    # reflects the seed every fold WILL see, not whatever state ``rng`` reaches after the function
    # consumes draws - without that snapshot a hit on a partially-consumed rng would key under
    # different bits than the original miss.
    _cache = None
    _cache_key = None
    if cache_dir is not None:
        try:
            from mlframe.utils.disk_cache import DiskCache, compose_key, hash_array_summary, hash_object

            _cache = DiskCache(cache_dir)
            try:
                _params = model_template.get_params(deep=False)
            # best-effort: only the cache key degrades (repr of the template instead of its params); the computation is unaffected
            except Exception as e:
                logger.debug("get_params(deep=False) failed, using repr() as the cache-key params: %s", e)
                _params = {"_repr": repr(model_template)}
            _rng_state: Any
            try:
                _rng_state = rng.bit_generator.state
            except AttributeError:
                _rng_state = repr(rng)
            _x_key = hash_array_summary(X.values if hasattr(X, "values") else np.asarray(X))
            _y_key = hash_array_summary(np.asarray(y))
            _state_key = hash_object({
                "params": _params,
                "rng": _rng_state,
                "n_splits": int(n_splits),
                "n_models": int(n_models),
                "out_of_fold": bool(out_of_fold),
                "config_jitter": bool(config_jitter),
                "return_variance": bool(return_variance),
                "return_per_fold_phi_mean": bool(return_per_fold_phi_mean),
                "shap_backend": str(shap_backend),
                "n_estimators_cap": n_estimators_cap,
                "inner_n_jobs_cap": bool(inner_n_jobs_cap),
                "classification": bool(classification),
                "columns": list(map(str, getattr(X, "columns", []))),
            })
            _cache_key = "shap_phi_" + compose_key(_x_key, _y_key, _state_key)
            _hit = _cache.get(_cache_key)
            if _hit is not None:
                logger.debug("compute_shap_matrix: cache hit key=%s", _cache_key)
                # Replay the exact ``rng.integers`` draws the miss path would consume so the shared
                # rng's post-state matches byte-for-byte. Without this, downstream stages that share
                # the same rng (selector heuristics, refine, trust-guard) would observe different
                # draws between cache-hit and cache-miss runs - breaking subset bit-identity.
                # Sequence (mirrors the non-cached code path below):
                #   * out_of_fold=False: n_models seeds.
                #   * out_of_fold=True: 1 splitter seed + n_splits * n_models fold seeds.
                _compute_shap_matri_out_fold_true_splitter(out_of_fold, n_models, rng, n_splits)
                return _cache, _cache_key, _hit
        except Exception as exc:
            # Cache failures are non-fatal: we lose the speedup but the compute path stays correct.
            log_throttle(logger, "shap_proxy.cache_disabled", logging.WARNING, "compute_shap_matrix: the disk cache is disabled for this call (%s), the result is recomputed", exc)
            _cache = None
            _cache_key = None
    return _cache, _cache_key, None


def _store_shap_cache(cache: Any, cache_key: Optional[str], result: Any) -> Any:
    """Write ``result`` into the outer ``shap_phi_`` disk cache under the pre-computed key (no-op if caching is disabled or unavailable); returns ``result`` unchanged for chaining at the return site."""
    if cache is not None and cache_key is not None:
        try:
            cache.put(cache_key, result)
        except Exception as exc:
            logger.debug("compute_shap_matrix: cache put failed (%s)", exc)
    return result
