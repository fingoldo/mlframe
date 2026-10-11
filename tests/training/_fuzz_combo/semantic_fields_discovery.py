"""Semantic generators for ``CompositeTargetDiscoveryConfig`` (see :mod:`semantic_fields` for the mechanism)."""

from __future__ import annotations

from typing import Any, Dict, List

from .semantic_fields import SemanticContext, _choice, _register, _subset

# ---------------------------------------------------------------------------
# CompositeTargetDiscoveryConfig
# ---------------------------------------------------------------------------


def _group_key_columns(ctx: SemanticContext) -> List[str]:
    """Columns usable as a per-entity group key: repeating categorical codes, or the MRMR-FE group key when the frame has it."""
    from .frame_builder import MRMR_FE_GROUP_COL

    cols = ctx.columns()
    return ctx.cat_columns() + ([MRMR_FE_GROUP_COL] if MRMR_FE_GROUP_COL in cols else [])


@_register(
    "CompositeTargetDiscoveryConfig",
    "group_column",
    "engineer_causal_group_column",
    "engineer_causal_lags",
    "engineer_causal_ops",
    "engineer_causal_trailing_windows",
    "engineer_causal_first_fill",
)
def _gen_causal_recipe(rng, ctx, model) -> Dict[str, Any]:
    """One grouped-causal-base recipe: a real group key, a non-empty op subset, lags only with ``lag``, windows only with ``trailing_mean``.

    Skipped on frames without a group-like column: the engineer is a documented no-op there, so the recipe would test nothing.
    """
    from mlframe.training.composite.discovery._grouped_causal_bases import _VALID_OPS

    keys = _group_key_columns(ctx)
    if not keys or rng.random() < 0.25:
        return {}
    key = _choice(rng, keys)
    out: Dict[str, Any] = {}
    slot = _choice(rng, ("group_column", "engineer_causal_group_column", "both"))
    if slot in ("group_column", "both"):
        out["group_column"] = key
    if slot in ("engineer_causal_group_column", "both"):
        out["engineer_causal_group_column"] = key
    ops = tuple(sorted(_subset(rng, _VALID_OPS)))
    out["engineer_causal_ops"] = ops
    if "lag" in ops:
        out["engineer_causal_lags"] = tuple(sorted(_subset(rng, (1, 2, 3, 5), max_size=3)))
    if "trailing_mean" in ops:
        out["engineer_causal_trailing_windows"] = tuple(sorted(_subset(rng, (2, 3, 5, 10), max_size=3)))
    out["engineer_causal_first_fill"] = _choice(rng, ("nan", "nan", "group_first"))
    if "engineer_causal_bases" not in model.model_fields_set:
        out["engineer_causal_bases"] = True
    return out


@_register("CompositeTargetDiscoveryConfig", "per_group_column")
def _gen_per_group(rng, ctx, model) -> Dict[str, Any]:
    """Per-group discovery keyed on a real group column (switched on with it; groups under ``per_group_min_rows`` fall back to global)."""
    keys = _group_key_columns(ctx)
    if not keys or rng.random() < 0.5:
        return {}
    out: Dict[str, Any] = {"per_group_column": _choice(rng, keys)}
    if "per_group_discovery_enabled" not in model.model_fields_set:
        out["per_group_discovery_enabled"] = True
    return out


@_register("CompositeTargetDiscoveryConfig", "dominant_features_hint")
def _gen_dominant_hint(rng, ctx, model) -> Dict[str, Any]:
    """Real numeric features the auto-base should start from (``num_0`` / ``num_1`` carry the target signal)."""
    return {} if rng.random() < 0.3 else {"dominant_features_hint": _subset(rng, ctx.numeric_feature_columns(), max_size=3)}


@_register("CompositeTargetDiscoveryConfig", "forbidden_base_patterns")
def _gen_forbidden_patterns(rng, ctx, model) -> Dict[str, Any]:
    """No patterns, the defaults plus one forbidding the strongest real base, or a pattern covering two real features."""
    defaults = list(getattr(model, "forbidden_base_patterns"))
    options: List[List[str]] = [[], [*defaults, r"^num_0$"], [r"^num_[01]$"]]
    return {} if rng.random() < 0.3 else {"forbidden_base_patterns": _choice(rng, options)}


@_register("CompositeTargetDiscoveryConfig", "auto_base_null_block_length")
def _gen_null_block_length(rng, ctx, model) -> Dict[str, Any]:
    """``"auto"`` (sqrt n), ``1`` (row-level i.i.d. shuffle), or a fixed block length."""
    return {"auto_base_null_block_length": _choice(rng, ("auto", 1, 5, 20))}


@_register("CompositeTargetDiscoveryConfig", "base_ranking_criterion")
def _gen_base_ranking(rng, ctx, model) -> Dict[str, Any]:
    """The two branches ``_auto_base`` dispatches on."""
    return {"base_ranking_criterion": _choice(rng, ("mi", "mrmr"))}


@_register("CompositeTargetDiscoveryConfig", "cross_target_calibration_method")
def _gen_calibration_method(rng, ctx, model) -> Dict[str, Any]:
    """An ``OutputCalibrator`` method (its own ``_VALID_METHODS``)."""
    from mlframe.training.composite.ensemble._calibration import _VALID_METHODS

    return {"cross_target_calibration_method": _choice(rng, _VALID_METHODS)}


@_register("CompositeTargetDiscoveryConfig", "fail_on_no_gain")
def _gen_fail_mode(rng, ctx, model) -> Dict[str, Any]:
    """``fallback_raw`` / ``warn``. ``raise`` exists to abort a run on purpose (CI mode), so sampling it would log an intended abort as a fuzz failure."""
    return {"fail_on_no_gain": _choice(rng, ("fallback_raw", "warn"))}


@_register("CompositeTargetDiscoveryConfig", "tiny_consensus")
def _gen_tiny_consensus(rng, ctx, model) -> Dict[str, Any]:
    """The two consensus modes the validator accepts."""
    return {"tiny_consensus": _choice(rng, ("union", "borda"))}


@_register("CompositeTargetDiscoveryConfig", "tiny_rerank_backend")
def _gen_rerank_backend(rng, ctx, model) -> Dict[str, Any]:
    """One of ``RERANK_BACKENDS``."""
    from mlframe.training.composite.discovery._tiny_rerank_process import RERANK_BACKENDS

    return {"tiny_rerank_backend": _choice(rng, RERANK_BACKENDS)}
