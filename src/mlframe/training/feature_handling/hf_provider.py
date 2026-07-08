"""
HuggingFaceProvider: concrete frozen-text-embedding implementation.

Frozen mode only. ``transform(texts) -> ndarray``. Uses
``transformers.AutoModel`` + ``transformers.AutoTokenizer`` directly
so no extra deps (no ``sentence-transformers``).

Default model behaviour:
  * ``intfloat/multilingual-e5-small`` is the FHC-level default
    (multilingual is the safer default; English-only users opt in
    to ``BAAI/bge-small-en-v1.5`` via explicit provider).
  * E5 family auto-prefix: input texts get ``"passage: "`` prepended
    automatically (e5 was trained with this convention; missing it
    halves retrieval quality). Detected by checking if model name
    contains ``"-e5-"`` or starts with ``e5-``.

Lifecycle:
  * ``acquire()`` loads the model + tokenizer onto the resolved
    device (auto / cuda / cpu / mps), in the configured dtype.
  * ``release()`` drops both, calls ``torch.cuda.empty_cache()`` on
    GPU.
  * Loaded model + tokenizer ARE NOT pickled with the provider
    (signature is enough to re-acquire).

Robustness:
  * ``trust_remote_code=False`` hardcoded (require explicit opt-in;
    HF repos with malicious modeling.py would otherwise execute on
    import).
  * GPU OOM mid-batch -> halve batch and retry. Driver-crash
    (``CudaErrorClass.CONTEXT_LOST``) -> abort with restart-Python
    message.
  * Empty / NaN / non-string input -> coerce to empty string with
    a warning.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import warnings
from typing import TYPE_CHECKING, Any, List, Optional, Sequence

import numpy as np

from mlframe.training.feature_handling.locking import PIDAwareFileLock
from mlframe.training.feature_handling.providers import EmbeddingProvider
from mlframe.training.feature_handling.system import (
    CudaErrorClass,
    classify_cuda_error,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import torch  # noqa: F401


# Models that require an instruction prefix for the embedding to make
# sense. E5 family was trained with "passage: " / "query: " prefixes.
# Missing it doesn't crash but quality drops noticeably.
_E5_FAMILY_MARKERS = ("-e5-", "/e5-", "_e5_")


def _needs_e5_prefix(model_name: str) -> bool:
    name = model_name.lower()
    return any(m in name for m in _E5_FAMILY_MARKERS)


def _hf_cache_lock_path(model_name: str, revision: Optional[str]) -> str:
    """Build a stable cross-process lock path keyed on the HF model
    signature. Two processes that both call ``AutoTokenizer.from_pretrained``
    on a fresh HF cache directory can race on partial-download writes
    (``transformers`` does NOT serialise cache writes across processes);
    serialising them here turns the race into a download-then-reuse.
    The lock lives under the HF cache root so it follows the user's
    ``HF_HOME`` / ``TRANSFORMERS_CACHE`` settings.
    """
    # Resolve the HF cache root. ``HF_HOME`` is the modern var;
    # ``TRANSFORMERS_CACHE`` is legacy; finally fall back to OS tmp so
    # the lock still lives somewhere even if the HF env is unset.
    root = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE") or os.path.join(tempfile.gettempdir(), "mlframe-hf-locks")
    sig = f"{model_name}@{revision or 'main'}"
    sig_hash = hashlib.blake2b(sig.encode("utf-8"), digest_size=12).hexdigest()
    lock_dir = os.path.join(root, ".mlframe-fhc-locks")
    try:
        os.makedirs(lock_dir, exist_ok=True)
    except OSError:
        lock_dir = tempfile.gettempdir()
    return os.path.join(lock_dir, f"hf_{sig_hash}.lock")


def _resolve_device(device_param: str) -> str:
    """Resolve ``"auto"`` to a concrete device string. Round-3 R2-18:
    enumerate cuda > mps > xpu > cpu, not just cuda-or-cpu.
    """
    if device_param != "auto":
        return device_param
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if hasattr(torch, "xpu") and torch.xpu.is_available():  # type: ignore[attr-defined]
            return "xpu"
        return "cpu"
    except ImportError:  # pragma: no cover
        return "cpu"


class HuggingFaceProvider:
    """Frozen-text-embedding provider via ``transformers.AutoModel``.

    Construct via :class:`EmbeddingProvider` (kind="huggingface");
    the FHC consumer (phase E) is what actually instantiates this.

    Implements :class:`FrozenFeaturizerProvider`.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
    ):
        if embedding_provider.kind != "huggingface":
            raise ValueError(f"HuggingFaceProvider expects kind='huggingface', got " f"{embedding_provider.kind!r}")
        self._cfg = embedding_provider
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._device: Optional[str] = None
        self._embedding_dim: Optional[int] = None
        self._auto_prefix: Optional[str] = None  # e.g. "passage: " for e5
        self._is_loaded = False

    # ------------------------------------------------------------------
    # Required Protocol surface
    # ------------------------------------------------------------------

    @property
    def signature(self) -> str:
        return self._cfg.signature

    @property
    def embedding_dim(self) -> int:
        if self._embedding_dim is None:
            raise RuntimeError("embedding_dim not available before acquire() -- " "call .acquire() first or wrap in `with acquire_provider(...)`")
        return self._embedding_dim

    def fit(self, train_texts: Sequence[str]) -> HuggingFaceProvider:
        """No-op for pretrained HF models. Returns self for chaining."""
        return self

    def acquire(self) -> None:
        """Load model + tokenizer onto resolved device."""
        if self._is_loaded:
            return

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:  # pragma: no cover
            raise ImportError("HuggingFaceProvider requires `transformers` and `torch`. " "Install with: pip install transformers torch") from e

        cfg = self._cfg
        model_name = cfg.model
        params = cfg.resolve_secrets().params  # may raise on missing env

        # Round-3 S4: trust_remote_code=False hardcoded; users opt in
        # via params={"trust_remote_code": True} which we explicitly
        # surface as a warning.
        trust_remote = bool(params.get("trust_remote_code", False))
        if trust_remote:
            warnings.warn(
                f"HuggingFaceProvider loading {model_name!r} with "
                f"trust_remote_code=True. The repo's modeling.py will "
                f"execute on load -- only proceed if you trust the source.",
                UserWarning,
                stacklevel=2,
            )

        revision = params.get("revision")  # None resolves to "main"
        device = _resolve_device(params.get("device", "auto"))
        dtype_str = params.get("dtype", "fp16")
        torch_dtype = torch.float16 if dtype_str == "fp16" else torch.float32

        logger.info(
            "[fhc] HuggingFaceProvider acquire: model=%s revision=%s device=%s dtype=%s",
            model_name, revision, device, dtype_str,
        )

        # Serialise the ``from_pretrained`` downloads behind a
        # per-(model, revision) cross-process lock. Without it two
        # processes targeting an empty HF cache both race to fill the
        # shard files; the second sees half-written blobs and either
        # crashes or - worse - silently loads truncated weights.
        # PIDAwareFileLock + StaleLockReclaimed keep the safety net.
        lock_path = _hf_cache_lock_path(model_name, revision)
        with PIDAwareFileLock(lock_path, timeout=600.0):
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    model_name, revision=revision, trust_remote_code=trust_remote,
                )
            except (ValueError, ImportError):
                # transformers defaults to the FAST (Rust) tokenizer; for a model that ships only a slow
                # tokenizer (vocab.txt, no tokenizer.json -- e.g. prajjwal1/bert-tiny) it tries to CONVERT
                # slow->fast, which needs sentencepiece/tiktoken installed and raises otherwise. The slow
                # tokenizer yields identical token ids for embedding, so fall back to it rather than hard-fail.
                self._tokenizer = AutoTokenizer.from_pretrained(
                    model_name, revision=revision, trust_remote_code=trust_remote, use_fast=False,
                )
            # Some tokenizers don't have a pad token (e.g. GPT-2 family);
            # falling back to eos_token is the standard fix for embedding.
            if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            self._model = AutoModel.from_pretrained(
                model_name,
                revision=revision,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote,
            )
        self._model = self._model.to(device).eval()
        self._device = device
        # Stash load params so a CUDA-context-loss mid-inference can rebuild the model on CPU (the GPU
        # context is dead after such a failure; the frozen model produces identical embeddings on CPU).
        self._load_model_name = model_name
        self._load_revision = revision
        self._load_torch_dtype = torch_dtype
        self._load_trust_remote = trust_remote

        # Embedding dim is hidden_size on the config
        self._embedding_dim = int(self._model.config.hidden_size)

        # E5 family auto-prefix detection. User can override via
        # params={"prefix": ...} including ``None`` to disable.
        if "prefix" in params:
            self._auto_prefix = params["prefix"]  # may be None
        elif _needs_e5_prefix(model_name):
            self._auto_prefix = "passage: "
            logger.info("[fhc] auto-applying e5-family prefix 'passage: ' for %s", model_name)
        else:
            self._auto_prefix = None

        self._is_loaded = True

    def release(self) -> None:
        """Drop model + tokenizer; call torch.cuda.empty_cache() on GPU."""
        if not self._is_loaded:
            return
        try:
            import torch
            self._model = None
            self._tokenizer = None
            if self._device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass
        self._is_loaded = False

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        """Compute embeddings; returns ``[N, hidden_size]`` ndarray.

        Uses pooled output: ``mean`` (default) over non-padding token
        states, ``cls`` token, or ``max`` per-dim. Configurable via
        ``EmbeddingProvider.params["pool"]``.
        """
        if not self._is_loaded:
            raise RuntimeError("HuggingFaceProvider not acquired -- call acquire() first")

        # Coerce non-string entries to empty so empty/null input doesn't
        # crash; Unicode round-trip safety relies on the underlying tokenizer.
        clean_texts: List[str] = []
        for t in texts:
            if t is None:
                clean_texts.append("")
            elif not isinstance(t, str):
                clean_texts.append(str(t))
            else:
                clean_texts.append(t)
        if self._auto_prefix:
            clean_texts = [self._auto_prefix + t for t in clean_texts]

        params = self._cfg.params
        max_length = int(params.get("max_length", 512))
        batch_size = self._resolve_batch_size(params.get("batch_size", "auto"))
        pool = params.get("pool", "mean")

        return self._batched_inference(clean_texts, batch_size, max_length, pool)

    def _resolve_batch_size(self, bs_param) -> int:
        """Resolve ``"auto"`` to a sensible per-device default. Round-3
        F11 mandates VRAM-aware tuning; phase B uses a conservative
        constant (256 GPU / 32 CPU) and phase D adds OOM-driven halving.
        """
        if isinstance(bs_param, int):
            return bs_param
        # "auto"
        try:
            import torch
            if torch.cuda.is_available():
                return 256
            return 32
        except ImportError:  # pragma: no cover
            return 32

    def _batched_inference(
        self,
        texts: List[str],
        batch_size: int,
        max_length: int,
        pool: str,
    ) -> np.ndarray:
        """Run inference batches with OOM-halve retry. Driver-crash
        (CONTEXT_LOST) aborts with a clear message.
        """
        import torch

        outputs: List[np.ndarray] = []
        i = 0
        n = len(texts)
        original_batch = max(1, batch_size)
        current_batch = original_batch
        # Number of consecutive successful batches required before
        # attempting to grow back toward the caller-requested batch
        # size. A transient OOM (other process briefly spiked VRAM)
        # shouldn't permanently halve us, so once we've seen
        # ``_recover_after_n_ok`` batches succeed at the reduced size
        # we double the batch (capped at the original). This stops the
        # pathological "halved-once, halved-forever" mode the legacy
        # loop locked into for the remainder of a multi-million-row
        # transform call.
        _recover_after_n_ok = 8
        consecutive_ok = 0
        assert self._tokenizer is not None and self._model is not None, "HFProvider: encode() called before load()"
        while i < n:
            batch = texts[i : i + current_batch]
            try:
                with torch.no_grad():
                    enc = self._tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt",
                    )
                    enc = {k: v.to(self._device) for k, v in enc.items()}
                    model_out = self._model(**enc)
                    pooled = self._pool(model_out, enc["attention_mask"], pool)
                    outputs.append(pooled.cpu().numpy())
                i += current_batch
                consecutive_ok += 1
                if current_batch < original_batch and consecutive_ok >= _recover_after_n_ok:
                    new_batch = min(original_batch, current_batch * 2)
                    if new_batch != current_batch:
                        logger.info(
                            "[fhc] HuggingFaceProvider batch_size recovering "
                            "from %d -> %d after %d ok batches",
                            current_batch, new_batch, consecutive_ok,
                        )
                        current_batch = new_batch
                    consecutive_ok = 0
            except Exception as exc:
                cls = classify_cuda_error(exc)
                if cls == CudaErrorClass.OUT_OF_MEMORY:
                    if current_batch <= 1:
                        raise
                    new_batch = max(1, current_batch // 2)
                    warnings.warn(
                        f"HuggingFaceProvider GPU OOM at batch_size={current_batch}; " f"halving to {new_batch} and retrying",
                        UserWarning,
                        stacklevel=2,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    current_batch = new_batch
                    consecutive_ok = 0
                    continue
                if cls == CudaErrorClass.CONTEXT_LOST:
                    # A lost CUDA context can't be reused, but the frozen model gives identical embeddings
                    # on CPU. This commonly fires when other CUDA libraries (cupy / numba.cuda for GPU MI)
                    # share the device in-process and a cublas handle collides. Rebuild the model fresh on
                    # CPU (the GPU tensors are unusable post-loss) and retry the batch there.
                    if str(self._device) != "cpu" and getattr(self, "_load_model_name", None):
                        warnings.warn(
                            "HuggingFaceProvider: CUDA context lost during inference (likely a multi-library "
                            "GPU cublas collision); falling back to CPU for the rest of this transform.",
                            UserWarning, stacklevel=2,
                        )
                        from transformers import AutoModel
                        self._model = AutoModel.from_pretrained(
                            self._load_model_name, revision=self._load_revision,
                            torch_dtype="float32", trust_remote_code=self._load_trust_remote,
                        ).to("cpu").eval()
                        self._device = "cpu"
                        consecutive_ok = 0
                        continue
                    raise RuntimeError(
                        "CUDA context lost (driver crash / kernel panic) and no CPU fallback available. "
                        "Restart Python -- the provider cannot recover in-process."
                    ) from exc
                raise
        if not outputs:
            return np.zeros((0, self._embedding_dim or 0), dtype=np.float32)
        return np.asarray(np.concatenate(outputs, axis=0).astype(np.float32, copy=False))

    @staticmethod
    def _pool(model_out, attention_mask, pool: str):
        """Standard pooling implementations."""
        last_hidden = model_out.last_hidden_state  # [B, T, H]
        if pool == "cls":
            return last_hidden[:, 0]
        if pool == "max":
            mask = attention_mask.unsqueeze(-1).bool()
            masked = last_hidden.masked_fill(~mask, float("-inf"))
            return masked.max(dim=1).values
        # default: mean over non-padding tokens
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-9)
        return summed / denom

    def __repr__(self) -> str:
        return f"HuggingFaceProvider(model={self._cfg.model!r}, " f"is_loaded={self._is_loaded}, device={self._device!r})"


# ---------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------


def build_provider(embedding_provider: EmbeddingProvider) -> HuggingFaceProvider:
    """Construct the concrete provider instance for an
    :class:`EmbeddingProvider` config. Phase B handles only ``huggingface``;
    phase J adds the rest (sentence-transformers / openai / ...).
    """
    if embedding_provider.kind == "huggingface":
        return HuggingFaceProvider(embedding_provider)
    raise NotImplementedError(
        f"provider kind {embedding_provider.kind!r} not implemented yet "
        f"(phase J: openai/cohere/jina/voyage/onnx/fasttext/tfhub/sentence-transformers/custom). "
        f"Phase B ships 'huggingface' only."
    )


__all__ = ["HuggingFaceProvider", "build_provider"]
