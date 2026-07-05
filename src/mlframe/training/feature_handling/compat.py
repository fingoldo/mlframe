"""
Model x axis x handler compatibility matrix + early validation.

``_MODEL_AXIS_SUPPORT`` is the central source of truth for "which
handler methods does model X accept on axis Y". Validation runs at
FHC construction (via the ``_validate_handlers`` entry point), paired
with the active ``mlframe_models`` list, and raises ``ValueError``
with ``difflib.get_close_matches()`` suggestions for typo'd method
names.

Public API for downstream consumers:
  * :data:`_MODEL_AXIS_SUPPORT` -- frozen lookup table.
  * :data:`_NATIVE_EMBEDDING_OUTPUT_SUPPORT` -- which models accept
    ``output="as_embedding_feature"``.
  * :func:`register_model_axis_support` -- runtime extension point
    for custom user models.
  * :func:`validate_handler_for_model` -- single-spec validator with
    actionable error messages.
  * :func:`validate_fhc_handlers` -- full FHC pass: walks defaults +
    per-model overrides, collects all errors, raises a combined
    ``ValueError`` once.
"""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

from mlframe.training.feature_handling.axis import Axis

if TYPE_CHECKING:
    from mlframe.training.feature_handling.handlers import (
        CatHandlerSpec,
        TextHandlerSpec,
    )


# =====================================================================
# The matrix
# =====================================================================
#
# Keyed on (model_kind, axis); value is the set of handler ``method``
# strings the model accepts on that axis.

_MODEL_AXIS_SUPPORT: Dict[Tuple[str, Axis], FrozenSet[str]] = {
    # CatBoost: native cat + native text + frozen embeddings (CB has
    # native embedding_features=).
    ("cb", Axis.CAT): frozenset({"native", "ordinal", "onehot", "target_mean", "target_m_estimate", "target_james_stein", "target_loo", "woe", "drop"}),
    ("cb", Axis.TEXT): frozenset({"native", "tfidf", "hashing", "frozen_text_embedding", "drop"}),
    # XGBoost: native cat (enable_categorical=True), no native text.
    ("xgb", Axis.CAT): frozenset({"native", "ordinal", "onehot", "target_mean", "target_m_estimate", "target_james_stein", "target_loo", "woe", "drop"}),
    ("xgb", Axis.TEXT): frozenset({"tfidf", "hashing", "frozen_text_embedding", "drop"}),
    # LightGBM: native cat (`categorical_feature=`), no native text.
    ("lgb", Axis.CAT): frozenset({"native", "ordinal", "onehot", "target_mean", "target_m_estimate", "target_james_stein", "target_loo", "woe", "drop"}),
    ("lgb", Axis.TEXT): frozenset({"tfidf", "hashing", "frozen_text_embedding", "drop"}),
    # HGB / RF / NGB: dense-only, no native cat (sklearn pipeline encodes).
    ("hgb", Axis.CAT): frozenset({"ordinal", "onehot", "target_mean", "target_m_estimate", "target_james_stein", "target_loo", "woe", "drop"}),
    ("hgb", Axis.TEXT): frozenset({"tfidf", "hashing", "frozen_text_embedding", "drop"}),
    ("ngb", Axis.CAT): frozenset({"ordinal", "onehot", "target_mean", "target_m_estimate", "target_james_stein", "target_loo", "woe", "drop"}),
    ("ngb", Axis.TEXT): frozenset({"tfidf", "hashing", "frozen_text_embedding", "drop"}),
    # Linear / Ridge / SGD: same dense-or-sparse, no native cat.
    ("linear", Axis.CAT): frozenset({"ordinal", "onehot", "target_mean", "target_m_estimate", "target_james_stein", "target_loo", "woe", "drop"}),
    ("linear", Axis.TEXT): frozenset({"tfidf", "hashing", "frozen_text_embedding", "drop"}),
    ("ridge", Axis.CAT): frozenset({"ordinal", "onehot", "target_mean", "target_m_estimate", "target_james_stein", "target_loo", "woe", "drop"}),
    ("ridge", Axis.TEXT): frozenset({"tfidf", "hashing", "frozen_text_embedding", "drop"}),
    # TabNet: native cat via cat_idxs/cat_dims; embedding=our TabularInputEncoder.
    ("tabnet", Axis.CAT): frozenset({"native", "embedding", "ordinal", "onehot", "drop"}),
    ("tabnet", Axis.TEXT): frozenset({"tfidf", "hashing", "frozen_text_embedding", "drop"}),
    # MLP / Recurrent: neural; cat="embedding" is the default and
    # learnable_text_embedding is exclusive to neural.
    ("mlp", Axis.CAT): frozenset({"embedding", "ordinal", "onehot", "drop"}),
    ("mlp", Axis.TEXT): frozenset({"frozen_text_embedding", "learnable_text_embedding", "tfidf", "hashing", "drop"}),
    ("recurrent", Axis.CAT): frozenset({"embedding", "ordinal", "onehot", "drop"}),
    ("recurrent", Axis.TEXT): frozenset({"frozen_text_embedding", "learnable_text_embedding", "tfidf", "hashing", "drop"}),
}

# Models that accept ``output="as_embedding_feature"``: the handler
# emits an embedding vector that bypasses concat-to-numeric and goes
# into a model-native embedding-features slot. Currently only CatBoost
# (existing wiring at ``training/core.py:4020-4023`` -- extended in
# phase F).
_NATIVE_EMBEDDING_OUTPUT_SUPPORT: Set[str] = {"cb"}

# Models considered "neural" -- ``learnable_text_embedding`` accepts
# only these. ``embedding`` cat-method also accepts only these
# (TabNet via the TabularInputEncoder branch, not its native cat_idxs).
_NEURAL_MODELS: FrozenSet[str] = frozenset({"mlp", "recurrent", "tabnet"})


def register_model_axis_support(
    model_kind: str,
    axis: Axis,
    methods: Iterable[str],
) -> None:
    """Extension hook for user-defined models.

    The compat matrix is closed by default to make typos surface
    early. Users with a custom model that needs a different combination
    of handler methods register here at import time. Idempotent --
    re-registering the same set is a no-op; re-registering a different
    set raises so stray reloads are caught.
    """
    key = (model_kind, axis)
    new_set = frozenset(methods)
    existing = _MODEL_AXIS_SUPPORT.get(key)
    if existing is not None and existing != new_set:
        raise ValueError(f"compat matrix entry for {key!r} already registered as {sorted(existing)!r}; " f"cannot reassign to {sorted(new_set)!r}")
    _MODEL_AXIS_SUPPORT[key] = new_set


# =====================================================================
# Validators
# =====================================================================


def _suggest_method(typo: str, valid: Iterable[str]) -> str:
    """Return ``" did you mean 'tfidf'?"`` style suggestion, capped to
    3 closest matches."""
    matches = difflib.get_close_matches(typo, list(valid), n=3, cutoff=0.5)
    if not matches:
        return ""
    if len(matches) == 1:
        return f" Did you mean {matches[0]!r}?"
    return f" Did you mean one of: {', '.join(repr(m) for m in matches)}?"


def validate_handler_for_model(
    model_kind: str,
    axis: Axis,
    method: str,
    output: Optional[str] = None,
    is_neural_required: bool = False,
) -> None:
    """Single-spec validator. Raises ``ValueError`` with actionable
    message + difflib suggestion on mismatch.
    """
    # Cross-cutting rule: ``learnable_text_embedding`` is neural-only.
    # Check this FIRST so the user gets the most-specific error message
    # ("requires neural") instead of the matrix-default ("not in valid
    # methods"); cross-cutting policies beat matrix lookup.
    if method == "learnable_text_embedding" and model_kind not in _NEURAL_MODELS:
        raise ValueError(
            f"method='learnable_text_embedding' requires a neural model; "
            f"got {model_kind!r}. Neural models: {sorted(_NEURAL_MODELS)}. "
            f"For non-neural use 'frozen_text_embedding' (frozen feature extractor)."
        )

    key = (model_kind, axis)
    if key not in _MODEL_AXIS_SUPPORT:
        valid_models = sorted({k[0] for k in _MODEL_AXIS_SUPPORT})
        raise ValueError(
            f"unknown model_kind {model_kind!r} for axis {axis.value!r}; "
            f"registered models: {valid_models}.{_suggest_method(model_kind, valid_models)} "
            f"To register a custom model, call register_model_axis_support()."
        )
    valid_methods = _MODEL_AXIS_SUPPORT[key]
    if method not in valid_methods:
        raise ValueError(
            f"model {model_kind!r} does not support method={method!r} on axis={axis.value!r}; "
            f"valid methods: {sorted(valid_methods)}.{_suggest_method(method, valid_methods)}"
        )

    # Cross-axis policy: ``output="as_embedding_feature"`` is allowed
    # only when the model has a native embedding-features slot.
    if output == "as_embedding_feature" and model_kind not in _NATIVE_EMBEDDING_OUTPUT_SUPPORT:
        raise ValueError(
            f"output='as_embedding_feature' is not supported by model {model_kind!r}; "
            f"only models with native embedding-features slots accept it: "
            f"{sorted(_NATIVE_EMBEDDING_OUTPUT_SUPPORT)}. Use output='auto' or "
            f"output='concat_with_numeric' instead."
        )

    if is_neural_required and model_kind not in _NEURAL_MODELS:
        raise ValueError(f"this handler requires a neural model; got {model_kind!r}.")


def validate_fhc_handlers(
    *,
    text_specs_per_model: Dict[str, List[TextHandlerSpec]],
    cat_specs_per_model: Dict[str, List[CatHandlerSpec]],
    active_models: Iterable[str],
) -> None:
    """FHC-level validator. Walks defaults + per_model overrides for
    every active model, accumulates errors, raises one combined
    ``ValueError`` so users see ALL mismatches in one go (rather than
    fix one error, run again, fix the next).
    """
    active = set(active_models)
    errors: List[str] = []

    for model_kind in active:
        for spec in text_specs_per_model.get(model_kind, []):
            try:
                validate_handler_for_model(
                    model_kind=model_kind,
                    axis=Axis.TEXT,
                    method=spec.method,
                    output=spec.output,
                )
            except ValueError as e:
                errors.append(f"  - {model_kind} text: {e}")
        for spec in cat_specs_per_model.get(model_kind, []):
            try:
                validate_handler_for_model(
                    model_kind=model_kind,
                    axis=Axis.CAT,
                    method=spec.method,
                )
            except ValueError as e:
                errors.append(f"  - {model_kind} cat: {e}")

    if errors:
        raise ValueError("FeatureHandlingConfig has incompatible handler/model combinations:\n" + "\n".join(errors))


__all__ = [
    "_MODEL_AXIS_SUPPORT",
    "_NATIVE_EMBEDDING_OUTPUT_SUPPORT",
    "_NEURAL_MODELS",
    "register_model_axis_support",
    "validate_handler_for_model",
    "validate_fhc_handlers",
]
