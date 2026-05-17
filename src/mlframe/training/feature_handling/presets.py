"""
One-line preset factories for :class:`FeatureHandlingConfig`.

Round-2 UX U-R2-2: the verbose ``FeatureHandlingConfig(per_model={...
nested-4-levels ...})`` is fine for power users, but the 80% case
("just TF-IDF for everyone" / "CatBoost-native everywhere") should
fit on one line.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from mlframe.training.feature_handling.config import (
    CacheConfig,
    FeatureHandlingConfig,
)
from mlframe.training.feature_handling.handlers import (
    CatHandlerSpec,
    FrozenEmbeddingParams,
    NoParams,
    TextHandlerSpec,
    TfidfParams,
)

if TYPE_CHECKING:
    pass


def tfidf_only(
    max_features: int = 5000,
    ngram_range: tuple = (1, 2),
    **fhc_kwargs: Any,
) -> FeatureHandlingConfig:
    """All models receive TF-IDF for text columns + native cat handling.

    Equivalent to::

        FeatureHandlingConfig(
            default_text=[TextHandlerSpec(method="tfidf", params=TfidfParams(max_features=...))],
            default_cat=[CatHandlerSpec(method="ordinal", params=NoParams(kind="ordinal"))],
        )
    """
    return FeatureHandlingConfig(
        default_text=[
            TextHandlerSpec(
                method="tfidf",
                params=TfidfParams(max_features=max_features, ngram_range=ngram_range),
            )
        ],
        default_cat=[CatHandlerSpec(method="ordinal", params=NoParams(kind="ordinal"))],
        **fhc_kwargs,
    )


def cb_native_only(**fhc_kwargs: Any) -> FeatureHandlingConfig:
    """Per-model spec wired so CatBoost gets native text + native cat,
    other models implicitly drop text (no defaults).

    Equivalent to::

        FeatureHandlingConfig(
            default_text=[TextHandlerSpec(method="drop")],
            default_cat=[CatHandlerSpec(method="native")],
            per_model={"cb": ModelHandlingOverride(
                text=[TextHandlerSpec(method="native")],
            )},
        )
    """
    from mlframe.training.feature_handling.config import ModelHandlingOverride

    return FeatureHandlingConfig(
        default_text=[TextHandlerSpec(method="drop", params=NoParams(kind="drop"))],
        default_cat=[CatHandlerSpec(method="ordinal", params=NoParams(kind="ordinal"))],
        per_model={
            "cb": ModelHandlingOverride(
                text=[TextHandlerSpec(method="native", params=NoParams(kind="native"))],
            ),
        },
        **fhc_kwargs,
    )


def embedding_only(
    provider: Optional[Any] = None,  # EmbeddingProvider
    pool: str = "mean",
    **fhc_kwargs: Any,
) -> FeatureHandlingConfig:
    """All models receive frozen-text-embeddings for text + native cat.

    Default provider is ``intfloat/multilingual-e5-small``; pass an
    explicit ``EmbeddingProvider`` to override.
    """
    return FeatureHandlingConfig(
        default_text=[
            TextHandlerSpec(
                method="frozen_text_embedding",
                params=FrozenEmbeddingParams(provider=provider, pool=pool),  # type: ignore[arg-type]
            )
        ],
        default_cat=[CatHandlerSpec(method="ordinal", params=NoParams(kind="ordinal"))],
        **fhc_kwargs,
    )


__all__ = ["tfidf_only", "cb_native_only", "embedding_only"]
