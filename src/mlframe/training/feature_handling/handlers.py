"""
Per-method TypedDict params + concrete ``HandlerSpec`` subclasses.

Discriminated-union params: each params class carries a
``kind: Literal[...]`` discriminator so pydantic can route on it.
Misspelled fields raise at construction; misspelled methods route to
the wrong params class only if the ``kind`` collides, which is
impossible with disjoint string discriminators.

The handler classes (``TextHandlerSpec``, ``CatHandlerSpec``) register
themselves with the axis registry from
:mod:`mlframe.training.feature_handling.axis` so they can be looked up
by axis without import-cycles.

Naming:
  * ``frozen_text_embedding``
  * ``learnable_text_embedding``
  * ``output="as_embedding_feature"``
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

from mlframe.training.feature_handling.axis import (
    Axis,
    HandlerSpec,
    register_handler_spec,
)

if TYPE_CHECKING:
    from mlframe.training.feature_handling.providers import EmbeddingProvider  # noqa: F401

# Eager import for pydantic v2 -- forward ref resolution at module load
# rather than via ``model_rebuild()`` which can fail if order changes.
from mlframe.training.feature_handling.providers import EmbeddingProvider  # noqa: E402

# =====================================================================
# Per-method TypedDict params
# =====================================================================
#
# Every params class:
#   * ``model_config = ConfigDict(extra="forbid")`` -- typo'd field
#     names raise ``ValidationError`` at construction.
#   * ``kind: Literal["..."]`` -- discriminator; pydantic uses this to
#     route Union[TfidfParams, HashingParams, ...] to the right class.
#
# The ``kind`` value MATCHES the corresponding ``HandlerSpec.method``
# value, so a sanity validator on the spec can compare the two and
# reject mismatches (defending against silent Union mismatch).


class TfidfParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["tfidf"] = "tfidf"
    max_features: int = 5000
    ngram_range: Tuple[int, int] = (1, 2)
    # ``min_df=2`` drops hapax legomena -- one-shot tokens that contribute pure noise to the vocab; on
    # 5M-row text columns this trims vocab by 30-60% before ``max_features`` kicks in. ``min_df=1``
    # (legacy default) materialised every misspelling / proper noun / random-suffixed ID.
    min_df: Union[int, float] = 2
    max_df: Union[int, float] = 1.0
    sublinear_tf: bool = True
    norm: Literal["l1", "l2"] = "l2"


class HashingParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["hashing"] = "hashing"
    n_features: int = 2**18
    ngram_range: Tuple[int, int] = (1, 2)
    norm: Literal["l1", "l2"] = "l2"


class FrozenEmbeddingParams(BaseModel):
    """Params for ``method="frozen_text_embedding"``.

    The ``provider`` field, when set, overrides the FHC-level
    ``default_text_provider``.
    """
    model_config = ConfigDict(extra="forbid")
    kind: Literal["frozen_text_embedding"] = "frozen_text_embedding"
    provider: Optional[EmbeddingProvider] = None
    pool: Literal["cls", "mean", "max"] = "mean"
    max_length: int = 512
    batch_size: Union[int, Literal["auto"]] = "auto"
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    dtype: Literal["fp16", "fp32"] = "fp16"


class LearnableEmbeddingParams(FrozenEmbeddingParams):
    """Params for ``method="learnable_text_embedding"``. Extends frozen
    with an LR multiplier for the HF-block inside the optimiser. Only
    valid for neural models (mlp / recurrent) -- the compat matrix
    rejects it elsewhere.
    """
    model_config = ConfigDict(extra="forbid")
    kind: Literal["learnable_text_embedding"] = "learnable_text_embedding"  # type: ignore[assignment]  # intentional discriminator override for this pydantic subclass
    finetune_lr_mult: float = 0.1


class TargetEncodeParams(BaseModel):
    """Params for ``method in {target_mean, target_m_estimate, ...}``.

    CV-fold-aware fitting is enforced by ``LeakageSafeEncoder`` (phase
    N) -- ``cv`` here is the fold count it uses internally.
    """
    model_config = ConfigDict(extra="forbid")
    kind: Literal["target_mean", "target_m_estimate", "target_james_stein", "target_loo", "woe"]
    # 3.0 (not 10.0) for the MEAN encoders: held-out sweep (bench_target_encoder_smoothing) shows 3.0 wins the majority of cells; 10.0 over-shrinks.
    # WoE is unaffected -- it uses the separate woe_smoothing cushion below (Jeffreys 0.5), passed independently into LeakageSafeEncoder.
    smoothing: float = 3.0
    woe_smoothing: Optional[float] = None
    cv: int = 5
    prior: Literal["mean", "median"] = "mean"
    random_state: Optional[int] = None


class CustomParams(BaseModel):
    """Escape hatch for a user-supplied sklearn pipeline / transformer
    plugged in as a handler. Mirrors existing
    ``FeatureSelectionConfig.custom_pre_pipelines`` precedent.
    ``transformer`` is required; the validator rejects non-callable
    ``fit`` / ``transform``.
    """
    model_config = ConfigDict(extra="allow")  # transformer object is opaque
    kind: Literal["custom"] = "custom"
    transformer: Any
    output_kind: Literal["dense", "sparse", "embedding"] = "dense"


class NoParams(BaseModel):
    """Placeholder for methods that take no params (e.g. ``native``,
    ``ordinal``, ``onehot``, ``drop``)."""
    model_config = ConfigDict(extra="forbid")
    kind: Literal["native", "ordinal", "onehot", "drop", "embedding"]


# =====================================================================
# HandlerSpec subclasses
# =====================================================================

# Discriminated unions for params. pydantic picks the right class
# based on the ``kind`` literal. Misspelled methods or method/params
# mismatches raise ``ValidationError`` immediately.

TextHandlerParams = Union[
    TfidfParams,
    HashingParams,
    FrozenEmbeddingParams,
    LearnableEmbeddingParams,
    CustomParams,
    NoParams,
]

CatHandlerParams = Union[
    TargetEncodeParams,
    CustomParams,
    NoParams,
]


class TextHandlerSpec(BaseModel, HandlerSpec):
    """How a text column is consumed by ONE model.

    Multiple specs per column (chained handler list) get their outputs
    concatenated at fit time -- see plan §5 multi-handler concat. Insertion
    order in the chain determines column-name order.
    """
    model_config = ConfigDict(extra="forbid")

    method: Literal[
        "native",
        "tfidf",
        "hashing",
        "frozen_text_embedding",
        "learnable_text_embedding",
        "custom",
        "drop",
    ]
    params: TextHandlerParams = Field(default_factory=lambda: NoParams(kind="drop"), discriminator="kind")

    output: Literal["auto", "concat_with_numeric", "as_embedding_feature", "sparse_block"] = "auto"
    apply_to_columns: Optional[List[str]] = None
    svd_dim: Optional[int] = None  # per-handler SVD override

    @classmethod
    def axis(cls) -> Axis:
        return Axis.TEXT

    def model_post_init(self, __context: Any) -> None:
        """Validate that method and params.kind agree.

        The discriminated Union prevents wrong params class for a given
        ``kind``, but a typo'd method like ``method="tdfif"`` would
        flunk the literal check first. This post-init catches the
        opposite error class: well-formed params whose ``kind`` doesn't
        match the chosen ``method``.
        """
        # NoParams covers many kinds, so check membership.
        params_kind = getattr(self.params, "kind", None)
        if params_kind is None:
            return
        # NoParams.kind is one of native/ordinal/onehot/drop/embedding;
        # method matching depends on which axis. For text-axis NoParams,
        # acceptable kinds are: native / drop.
        if isinstance(self.params, NoParams):
            allowed_for_text_no_params = {"native", "drop"}
            if self.method in allowed_for_text_no_params:
                return
            # Otherwise the user passed e.g. method="tfidf" with no
            # params, which silently would create TfidfParams() defaults
            # in pydantic-V2 if the discriminator allows -- but our
            # default_factory makes NoParams(drop) the fall-through, so
            # method="tfidf" + default params is an inconsistent state.
            raise ValueError(f"TextHandlerSpec(method={self.method!r}) requires explicit " f"params; got NoParams. Provide params=TfidfParams(...) etc.")
        if params_kind != self.method:
            raise ValueError(
                f"TextHandlerSpec method={self.method!r} does not match "
                f"params.kind={params_kind!r}; the discriminated union routed "
                f"to the wrong params class. Fix by aligning method and params.kind."
            )


class CatHandlerSpec(BaseModel, HandlerSpec):
    """How a categorical column is consumed by ONE model.

    Symmetric to :class:`TextHandlerSpec`. Methods include
    ``target_*`` encoders that are CV-fold-aware via the
    ``LeakageSafeEncoder`` (phase N).
    """
    model_config = ConfigDict(extra="forbid")

    method: Literal[
        "native",
        "ordinal",
        "onehot",
        "embedding",
        "target_mean",
        "target_m_estimate",
        "target_james_stein",
        "target_loo",
        "woe",
        "custom",
        "drop",
    ]
    params: CatHandlerParams = Field(default_factory=lambda: NoParams(kind="drop"), discriminator="kind")

    apply_to_columns: Optional[List[str]] = None
    group_columns: Optional[List[str]] = None  # group-aware encoding

    @classmethod
    def axis(cls) -> Axis:
        return Axis.CAT

    def model_post_init(self, __context: Any) -> None:
        """Mirror TextHandlerSpec validation. NoParams acceptable for
        method in {native, ordinal, onehot, embedding, drop}; target_*
        and woe require TargetEncodeParams; custom requires CustomParams.
        """
        params_kind = getattr(self.params, "kind", None)
        if params_kind is None:
            return
        if isinstance(self.params, NoParams):
            allowed_no_params = {"native", "ordinal", "onehot", "embedding", "drop"}
            if self.method in allowed_no_params:
                return
            raise ValueError(f"CatHandlerSpec(method={self.method!r}) requires explicit " f"params; got NoParams.")
        if params_kind != self.method:
            raise ValueError(f"CatHandlerSpec method={self.method!r} does not match " f"params.kind={params_kind!r}.")


# Register at import time so axis lookups work.
register_handler_spec(TextHandlerSpec)
register_handler_spec(CatHandlerSpec)


__all__ = [
    "TfidfParams",
    "HashingParams",
    "FrozenEmbeddingParams",
    "LearnableEmbeddingParams",
    "TargetEncodeParams",
    "CustomParams",
    "NoParams",
    "TextHandlerSpec",
    "CatHandlerSpec",
    "TextHandlerParams",
    "CatHandlerParams",
]
