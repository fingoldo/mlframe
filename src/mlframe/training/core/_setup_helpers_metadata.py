"""Metadata builders / finalizers for ``_setup_helpers``.

Carved from ``_setup_helpers.py``. Re-exported from the parent.
Imports the disk-cache helpers from ``_setup_helpers_pipeline_cache``
directly (sibling-to-sibling) to avoid a parent re-import cycle.
"""

from __future__ import annotations

import logging
from os.path import join
from typing import TYPE_CHECKING, Any

import psutil

from pyutilz.strings import slugify

from ._setup_helpers_pipeline_cache import (
    _PIPELINE_JSON_ROUNDTRIP_CACHE,
    _PolarsDsPipelineJsonProxy,
    _load_pipeline_disk_cache_into_memory,
    _persist_pipeline_disk_cache,
    pipeline_json_cache_key,
)

if TYPE_CHECKING:
    from ..configs import (  # noqa: F401
        PreprocessingBackendConfig,
        PreprocessingConfig,
        TrainingBehaviorConfig,
        TrainingSplitConfig,
    )
    from ._training_context import TrainingContext  # noqa: F401

logger = logging.getLogger(__name__)


def _create_initial_metadata(
    model_name: str,
    target_name: str,
    mlframe_models: list[str],
    preprocessing_config: "PreprocessingConfig",
    pipeline_config: "PreprocessingBackendConfig",
    split_config: "TrainingSplitConfig",
) -> dict[str, Any]:
    """Create the initial metadata dictionary for tracking training."""
    def _as_dict(cfg):
        if cfg is None or isinstance(cfg, dict):
            return cfg
        if hasattr(cfg, "model_dump"):
            return cfg.model_dump()
        return cfg

    return {
        "model_name": model_name,
        "target_name": target_name,
        "mlframe_models": mlframe_models,
        "configs": {
            "preprocessing": _as_dict(preprocessing_config),
            "pipeline": _as_dict(pipeline_config),
            "split": _as_dict(split_config),
        },
    }


def _initialize_training_defaults(
    common_params_dict: dict[str, Any] | None,
    rfecv_models: list[str] | None,
    mrmr_kwargs: dict[str, Any] | None,
    *,
    suite_verbose: int | None = None,
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """Initialize default values for training parameters.

    The MRMR default kwargs (n_workers / verbose / fe_max_steps / max_runtime_mins)
    are SHALLOW-MERGED into a caller-supplied dict so passing
    ``mrmr_kwargs={"some_knob": x}`` extends the defaults instead of replacing them
    entirely. Prior code dropped the 5-hour runtime cap silently whenever the
    caller supplied any kwarg. ``suite_verbose`` is the suite-level verbose so the
    MRMR verbose tracks the operator setting (no MRMR chatter in silent CI runs).
    ``psutil.cpu_count(logical=False)`` can return None on container hosts;
    ``or 1`` keeps ``max()`` safe.
    """
    if common_params_dict is None:
        common_params_dict = {}

    if rfecv_models is None:
        rfecv_models = []

    _mrmr_verbose_default = int(suite_verbose) if suite_verbose is not None else 1
    _default_mrmr_kwargs = dict(
        n_workers=max(1, psutil.cpu_count(logical=False) or 1),
        verbose=_mrmr_verbose_default,
        fe_max_steps=1,
        max_runtime_mins=300,
    )
    if mrmr_kwargs is None:
        mrmr_kwargs = _default_mrmr_kwargs
    else:
        mrmr_kwargs = {**_default_mrmr_kwargs, **mrmr_kwargs}

    return (
        common_params_dict,
        rfecv_models,
        mrmr_kwargs,
    )


def _finalize_and_save_metadata(ctx: "TrainingContext", *, verbose: int | None = None) -> None:
    """Finalize ``ctx.metadata`` (set outlier_detector / OD result / trainset stats / slug maps) and atomically save to disk.

    ``verbose=None`` reads ``ctx.verbose``; explicit ``0`` silences the save-log -- used by ``finalize_suite`` to avoid
    the duplicate "Saved metadata to ..." line when the suite saves once after main.py already saved partway.
    """
    metadata = ctx.metadata
    _verbose = ctx.verbose if verbose is None else verbose
    metadata.update(
        {
            "outlier_detector": ctx.outlier_detector,
            "outlier_detection_result": ctx.outlier_detection_result,
            "trainset_features_stats": ctx.trainset_features_stats,
        }
    )

    # Defensive dict() copy so metadata's slug maps are decoupled from ctx (and from the
    # picklable suite metadata that a long-running serving process loads ONCE and uses
    # across many predict calls). Pre-fix the assignment aliased ctx's dict; any
    # predict-side ``setdefault`` for a fallback slug would mutate the loaded metadata
    # in place and leak the entry into the next predict call -- effectively building up
    # phantom slug entries across the serving session.
    if ctx.slug_to_original_target_type:
        metadata["slug_to_original_target_type"] = dict(ctx.slug_to_original_target_type)
    if ctx.slug_to_original_target_name:
        metadata["slug_to_original_target_name"] = dict(ctx.slug_to_original_target_name)

    # Wrap the polars-ds Pipeline with a JSON-serializing proxy BEFORE pickling.
    # Pre-fix: pickle descended through every internal pl.Expr in the Pipeline,
    # and on complex Pipeline state (Yeo-Johnson + dim_reducer + ordinal_encode
    # over many categories) each PyExpr.__setstate__ took 100-200ms in
    # polars' Rust deserializer. 19 such PyExprs = 2.2s load wall. Surfaced
    # by the 100k binary_classification x lgb profile 2026-05-19 (seed
    # 20260522): load 2.27s, of which 2.21s sat in PyExpr.__setstate__.
    # The proxy stores Pipeline.to_json() and reconstructs via from_json()
    # on load -- 10x speedup on the same combo (2.27s -> 0.21s measured).
    # Validate round-trip BEFORE wrapping: polars-ds from_json doesn't
    # support all step types (e.g. some encoder variants raise
    # ComputeError "could not deserialize input into an expression").
    # When from_json roundtrip fails, fall through to standard pickle
    # so correctness is never sacrificed for speed.
    _pipeline_orig = metadata.get("pipeline")
    if _pipeline_orig is not None:
        try:
            from polars_ds.pipeline import Pipeline as _PdsPipeline
            if isinstance(_pipeline_orig, _PdsPipeline):
                try:
                    _js = _pipeline_orig.to_json()
                    _js_hash = pipeline_json_cache_key(_js)
                    # iter275: hydrate from disk cache once per process; gives
                    # cross-process reuse so fuzz combo re-runs / pytest-xdist
                    # workers / CI all skip the 8.5s validation after the first
                    # process verified this JSON. No-op after the first call.
                    _load_pipeline_disk_cache_into_memory()
                    _rt_ok = _PIPELINE_JSON_ROUNDTRIP_CACHE.get(_js_hash)
                    if _rt_ok is None:
                        # Cache miss: pay the 5-8s validation, then memoise
                        # the result so subsequent fits in this process skip
                        # it AND future processes inherit the verdict via the
                        # disk cache. Negative-result caching too: a JSON
                        # that fails parse this time will fail every time
                        # (deterministic).
                        try:
                            _PdsPipeline.from_json(_js)
                            _rt_ok = True
                        except Exception:
                            _rt_ok = False
                        _PIPELINE_JSON_ROUNDTRIP_CACHE[_js_hash] = _rt_ok
                        _persist_pipeline_disk_cache()
                    if _rt_ok:
                        metadata["pipeline"] = _PolarsDsPipelineJsonProxy(_pipeline_orig)
                    else:
                        raise RuntimeError("cached: roundtrip parse failed")
                except Exception as _rt_exc:
                    logger.debug(
                        "polars-ds Pipeline JSON roundtrip failed (%s); "
                        "falling back to standard pickle for the Pipeline. "
                        "Load may be slow on complex pipelines.", _rt_exc,
                    )
        except ImportError:
            pass  # polars-ds unavailable; nothing to wrap

    # Atomic write (serialize -> temp file -> os.replace) avoids metadata.* corruption when two
    # train runs race on the same target. Reader sees the complete old or new file, never partial.
    if ctx.data_dir and ctx.models_dir:
        metadata_dir = join(ctx.data_dir, ctx.models_dir, slugify(ctx.target_name), slugify(ctx.model_name))
        metadata_file = join(metadata_dir, "metadata.pkl.zst")
        from mlframe.training.io import atomic_write_bytes
        import pickle as _pickle
        # Probe zstandard FIRST so the optional-dep choice is decoupled from IO failure handling.
        # Previously a missing zstandard was caught alongside genuine atomic_write_bytes IO errors,
        # which made it confusing whether the fallback fired because of missing dep or disk error.
        try:
            import zstandard as _zstd
            _have_zstd = True
        except ImportError:
            _have_zstd = False

        if _have_zstd:
            # ``threads=-1`` enables zstd's multi-threaded compression API,
            # which splits the input into independent frames and compresses
            # them in parallel. Bench at 28 MB / level=3: 65 ms -> 33 ms
            # (~2x); at 57 MB: 131 ms -> 54 ms (~2.4x). Output is identical
            # to single-threaded -- the frame format decompresses the same.
            # save_mlframe_model already wires the same multi-threaded zstd
            # via ``zstd_kwargs``; this metadata path was the only saver
            # still on the single-threaded default.
            _cctx = _zstd.ZstdCompressor(level=3, threads=-1)
            def _writer(f):
                f.write(_cctx.compress(_pickle.dumps(metadata, protocol=5)))
        else:
            # Fallback: uncompressed pickle. .pkl extension lets the reader's magic-byte sniff route it.
            metadata_file = join(metadata_dir, "metadata.pkl")
            def _writer(f):
                _pickle.dump(metadata, f, protocol=5)

        try:
            atomic_write_bytes(metadata_file, _writer)
            if _verbose:
                logger.info("Saved metadata to %s", metadata_file)
        except OSError as e:
            logger.error(f"Failed to save metadata to {metadata_file}: {e}")
            raise
