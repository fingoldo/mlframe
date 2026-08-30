"""Reads of ``pipeline_config`` that decide whether the polars frames still have to become pandas frames.

Both questions are pure config inspection, and both have to cope with ``PreprocessingBackendConfig`` arriving
either as the dataclass or as a plain dict (callers pass both). Keeping them out of the suite facade keeps that
dual-shape handling in one place instead of inline in an already-long argument list.
"""

from __future__ import annotations

from typing import Any

_STAGE_KEYS = ("categorical_encoding", "scaler_name", "imputer_strategy", "dim_reducer_name")


def _config_get(pipeline_config: Any, name: str) -> Any:
    """One config value, whichever form the config takes."""
    if isinstance(pipeline_config, dict):
        return pipeline_config.get(name)
    return getattr(pipeline_config, name, None)


def any_pipeline_stage_requested(pipeline_config: Any) -> bool:
    """True when the caller asked for at least one transform whose fitted state would have to travel with the frames.

    An unknown config shape returns True: not being able to tell must not silently disable a conversion something
    downstream may depend on. With no encoder, scaler, imputer or reducer requested there is no fitted pipeline
    state in either representation, so an unapplied polars pipeline is not a reason to convert -- which is the
    common CatBoost-only configuration.
    """
    if pipeline_config is None:
        return False
    try:
        return any(_config_get(pipeline_config, name) is not None for name in _STAGE_KEYS)
    except Exception:
        return True


def needs_polars_pre_clone(pipeline_config: Any, *, was_polars_input: bool) -> bool:
    """True when categorical encoding will mutate the polars frames, so the pre-encoding copy is still needed."""
    if not was_polars_input:
        return False
    if _config_get(pipeline_config, "skip_categorical_encoding"):
        return False
    return _config_get(pipeline_config, "categorical_encoding") is not None


__all__ = ["any_pipeline_stage_requested", "needs_polars_pre_clone"]
