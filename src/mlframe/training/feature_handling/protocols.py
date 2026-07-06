"""
``FrozenFeaturizerProvider`` and ``TrainableFeaturizerProvider`` Protocols.

Round-3 architecture R2-4 fix: the previous ``FeaturizerProvider``
single Protocol leaked HF specifics into all consumers because
``transform()->ndarray`` works for frozen modes but ``trainable``
needs ``as_module()->nn.Module`` for gradient flow. Splitting clarifies
intent: any provider implements ``Frozen``; only some (HF, SBERT) also
implement ``Trainable``.

This module is concrete-impl-light so importing it does not pull in
torch / transformers. Concrete providers (``HuggingFaceProvider``,
``OpenAIEmbeddingProvider`` etc) live in their own modules and import
their backends lazily.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import numpy as np

if TYPE_CHECKING:
    import torch.nn as nn  # noqa: F401


@runtime_checkable
class FrozenFeaturizerProvider(Protocol):
    """Frozen text-embedding provider. Returns dense ndarrays.

    Implementations:
      * Pretrained HuggingFace model in eval mode (the v1 default).
      * Sentence-Transformers wrapper (phase J).
      * OpenAI / Cohere / Jina / Voyage HTTP API (phase J).
      * Local ONNX runtime (phase J).
      * FastText (phase J).

    Lifecycle: ``acquire()`` loads weights / opens HTTP session;
    ``release()`` drops VRAM / closes session. Use as a context
    manager via the registry, not directly -- naked ``acquire()`` is
    banned (round-3 chaos C22) because release-on-error semantics
    are easy to miss otherwise.
    """

    @property
    def signature(self) -> str:
        """Stable string used in cache keys. Phase D-version of the
        signature includes lib_version + revision_sha; phase B uses
        the simpler EmbeddingProvider.signature."""

    @property
    def embedding_dim(self) -> int: ...

    def fit(self, train_texts: Sequence[str]) -> FrozenFeaturizerProvider:
        """No-op for pretrained models; for FastText etc. trains the
        embedding model on the supplied corpus. Returns self for
        chaining."""

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        """Returns ``[len(texts), embedding_dim]`` dense fp32 / fp16
        ndarray. Implementations choose dtype per ``EmbeddingProvider.params``."""

    def acquire(self) -> None:
        """Load weights, open sessions. Idempotent.
        Called via the registry under a per-signature lock."""

    def release(self) -> None:
        """Drop weights, free VRAM, close sessions. Idempotent."""


@runtime_checkable
class TrainableFeaturizerProvider(FrozenFeaturizerProvider, Protocol):
    """Extends Frozen with an :meth:`as_module` accessor that returns
    the underlying ``nn.Module`` so the consumer (the
    ``TabularInputEncoder`` in phase G) can register it as a submodule
    and let autograd flow through.

    Phase A2 stub-types this with ``Any`` to avoid importing torch
    in modules that don't actually train; phase G defines the concrete
    type more strictly.
    """

    def as_module(
        self,
        finetune_lr_mult: float = 0.1,
    ) -> Tuple[Any, Callable[[Sequence[str]], Dict[str, Any]]]:
        """Return ``(nn.Module, tokenize_fn)`` -- the trainable module
        for gradient flow, and a tokeniser callable that the encoder
        invokes during forward pass.

        Phase G consumes this; phase B providers implement only the
        frozen path.
        """


__all__ = ["FrozenFeaturizerProvider", "TrainableFeaturizerProvider"]
