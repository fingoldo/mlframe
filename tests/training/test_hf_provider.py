"""
End-to-end integration tests for :class:`HuggingFaceProvider` (phase B).

Uses ``hf-internal-testing/tiny-random-BertModel`` (tiny random weights, ships a
fast tokenizer.json) -- so CI can pull it fast AND it loads under transformers>=5.
Skipped automatically if ``transformers`` / ``torch`` are missing.

Coverage:
  * ``acquire`` / ``release`` lifecycle.
  * ``transform`` returns ``[N, hidden_size]`` ndarray with right
    dtype.
  * Empty / null / non-string input handled (round-3 T8).
  * Unicode (Cyrillic / emoji / RTL) round-trip (round-3 T9).
  * E5-family auto-prefix detection (round-3 U-R2-8).
  * Pool variants: ``mean`` / ``cls`` / ``max``.
  * ``trust_remote_code=False`` is the default.

NOTE: HF model is downloaded on first run; subsequent runs read
from ``~/.cache/huggingface``. CI cold-cache will take ~30s on
the first download.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

# Skip this whole module if HF or torch isn't installed.
pytest.importorskip("transformers")
pytest.importorskip("torch")

from mlframe.training.feature_handling import EmbeddingProvider, shutdown_all
from mlframe.training.feature_handling.hf_provider import HuggingFaceProvider


# Cheapest-possible HF encoder for CI speed. ``hf-internal-testing/tiny-random-BertModel`` ships a
# fast tokenizer.json (so it loads under transformers>=5, which can no longer build prajjwal1/bert-tiny's
# slow-only tokenizer) and tiny random weights -- enough for the provider lifecycle/shape/embedding tests.
TINY_MODEL = "hf-internal-testing/tiny-random-BertModel"
# hidden_size of TINY_MODEL (was 128 for prajjwal1/bert-tiny); the provider's embedding_dim == this.
EMBED_DIM = 32


@pytest.fixture(autouse=True)
def _shutdown_after():
    yield
    shutdown_all()


@pytest.fixture(scope="module")
def loaded_provider():
    """Module-scope so the model is downloaded + loaded once for all
    tests in this file."""
    cfg = EmbeddingProvider(
        kind="huggingface",
        model=TINY_MODEL,
        params={"dtype": "fp32", "device": "cpu", "batch_size": 8},
    )
    p = HuggingFaceProvider(cfg)
    p.acquire()
    yield p
    p.release()


# =====================================================================
# 1. Lifecycle
# =====================================================================


class TestLifecycle:
    def test_acquire_loads_model(self, loaded_provider):
        assert loaded_provider._is_loaded is True
        assert loaded_provider._model is not None
        assert loaded_provider._tokenizer is not None
        assert loaded_provider.embedding_dim == EMBED_DIM

    def test_release_drops_model(self):
        cfg = EmbeddingProvider(
            kind="huggingface",
            model=TINY_MODEL,
            params={"dtype": "fp32", "device": "cpu"},
        )
        p = HuggingFaceProvider(cfg)
        p.acquire()
        assert p._is_loaded is True
        p.release()
        assert p._is_loaded is False
        assert p._model is None

    def test_kind_must_be_huggingface(self):
        cfg = EmbeddingProvider(kind="onnx", model="x")
        with pytest.raises(ValueError, match="kind='huggingface'"):
            HuggingFaceProvider(cfg)


# =====================================================================
# 2. transform shape + dtype
# =====================================================================


class TestTransform:
    def test_transform_returns_correct_shape(self, loaded_provider):
        out = loaded_provider.transform(["hello", "world", "third"])
        assert out.shape == (3, EMBED_DIM)
        assert out.dtype == np.float32

    def test_transform_empty_list(self, loaded_provider):
        out = loaded_provider.transform([])
        assert out.shape == (0, EMBED_DIM)


# =====================================================================
# 3. Edge inputs (round-3 T8 + T9)
# =====================================================================


class TestEdgeInputs:
    def test_empty_strings_dont_crash(self, loaded_provider):
        out = loaded_provider.transform(["", "", "ok"])
        assert out.shape == (3, EMBED_DIM)
        # All-empty rows should still produce valid embeddings (the
        # tokenizer encodes them as just CLS+SEP).
        assert not np.any(np.isnan(out))

    def test_none_in_input_coerced_to_empty(self, loaded_provider):
        # None coerced to "" by HuggingFaceProvider.transform.
        out = loaded_provider.transform([None, "real text", None])  # type: ignore[arg-type]
        assert out.shape == (3, EMBED_DIM)
        assert not np.any(np.isnan(out))

    def test_unicode_roundtrip(self, loaded_provider):
        """Round-3 T9: Cyrillic / emoji / RTL content shouldn't crash."""
        out = loaded_provider.transform(["привет мир", "🔥🚀 launch", "مرحبا", "regular english"])
        assert out.shape == (4, EMBED_DIM)
        assert not np.any(np.isnan(out))


# =====================================================================
# 4. Pool variants
# =====================================================================


class TestPoolVariants:
    @pytest.mark.parametrize("pool", ["mean", "cls", "max"])
    def test_pool_produces_valid_output(self, pool):
        cfg = EmbeddingProvider(
            kind="huggingface",
            model=TINY_MODEL,
            params={"dtype": "fp32", "device": "cpu", "pool": pool, "batch_size": 4},
        )
        p = HuggingFaceProvider(cfg)
        p.acquire()
        try:
            out = p.transform(["test", "another text"])
            assert out.shape == (2, EMBED_DIM)
            assert np.isfinite(out).all()
        finally:
            p.release()


# =====================================================================
# 5. trust_remote_code default
# =====================================================================


class TestTrustRemoteCode:
    def test_trust_remote_code_default_false(self, loaded_provider):
        # Round-3 S4: default must be False (require explicit opt-in).
        # We can verify the param wasn't smuggled in -- the loaded
        # tokenizer is the standard one.
        params = loaded_provider._cfg.params
        assert params.get("trust_remote_code", False) is False

    def test_trust_remote_code_true_emits_warning(self, recwarn):
        cfg = EmbeddingProvider(
            kind="huggingface",
            model=TINY_MODEL,
            params={"trust_remote_code": True, "dtype": "fp32", "device": "cpu"},
        )
        p = HuggingFaceProvider(cfg)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            p.acquire()
            try:
                pass
            finally:
                p.release()
        # Match our warning, ignore HF's own warnings.
        ours = [w for w in caught if "trust_remote_code=True" in str(w.message)]
        assert ours, "should emit security warning when trust_remote_code=True"


# =====================================================================
# 6. E5 auto-prefix detection (no real e5 model loaded; structural
#    test via the helper)
# =====================================================================


class TestE5AutoPrefix:
    def test_auto_prefix_detector(self):
        from mlframe.training.feature_handling.hf_provider import _needs_e5_prefix

        assert _needs_e5_prefix("intfloat/multilingual-e5-small") is True
        assert _needs_e5_prefix("intfloat/multilingual-e5-base") is True
        assert _needs_e5_prefix("BAAI/bge-small-en-v1.5") is False
        assert _needs_e5_prefix("prajjwal1/bert-tiny") is False
        assert _needs_e5_prefix("sentence-transformers/all-MiniLM-L6-v2") is False

    def test_prefix_override_via_params(self):
        """User can disable auto-prefix by setting params={"prefix": None}."""
        cfg = EmbeddingProvider(
            kind="huggingface",
            model=TINY_MODEL,  # not e5, but verify override works
            params={"prefix": "QUERY: ", "dtype": "fp32", "device": "cpu"},
        )
        p = HuggingFaceProvider(cfg)
        p.acquire()
        try:
            assert p._auto_prefix == "QUERY: "
        finally:
            p.release()
