"""
Tests for the phase-A2 :class:`EmbeddingProvider` structured config.

Coverage:
  * Construction with `kind` literal validation.
  * `from_uri` parsing -- happy path AND malformed URIs (per round-2
    test agent's correction: ``hf://model?device=cuda:0`` is RFC 3986
    valid, NOT malformed).
  * `signature` is stable across runs and does not include resolved
    secret values (round-3 R2-6).
  * `resolve_secrets` swaps `env:VAR` indirections, raises on missing
    env var.
  * `model_dump(scrub_secrets=True)` (default) masks secret-looking
    keys; `__repr__` likewise so debugger / Sentry doesn't leak
    creds (round-3 U-R2-28).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mlframe.training.feature_handling import EmbeddingProvider


# =====================================================================
# 1. Construction + kind validation
# =====================================================================


class TestConstruction:
    """Groups tests covering construction."""
    def test_huggingface_minimal(self):
        """Huggingface minimal."""
        p = EmbeddingProvider(kind="huggingface", model="intfloat/multilingual-e5-small")
        assert p.kind == "huggingface"
        assert p.model == "intfloat/multilingual-e5-small"
        assert p.params == {}

    def test_openai_with_dimensions(self):
        """Openai with dimensions."""
        p = EmbeddingProvider(
            kind="openai",
            model="text-embedding-3-small",
            params={"dimensions": 512},
        )
        assert p.params["dimensions"] == 512

    def test_unknown_kind_rejects(self):
        """Unknown kind rejects."""
        with pytest.raises(ValidationError):
            EmbeddingProvider(kind="lmstudio", model="x")

    def test_extra_field_forbidden(self):
        """Extra field forbidden."""
        with pytest.raises(ValidationError):
            EmbeddingProvider(kind="huggingface", model="x", extra_field="boom")


# =====================================================================
# 2. from_uri parsing
# =====================================================================


class TestFromUri:
    """Groups tests covering from uri."""
    def test_hf_short_alias(self):
        """Hf short alias."""
        p = EmbeddingProvider.from_uri("hf://BAAI/bge-small-en-v1.5")
        assert p.kind == "huggingface"
        assert p.model == "BAAI/bge-small-en-v1.5"
        assert p.params == {}

    def test_hf_with_query_params(self):
        # Note: per round-2 test agent's correction, the ``cuda:0``
        # colon-in-query is RFC 3986 valid, NOT malformed.
        """Hf with query params."""
        p = EmbeddingProvider.from_uri("hf://BAAI/bge-small-en-v1.5?device=cuda:0&dtype=fp16")
        assert p.kind == "huggingface"
        assert p.model == "BAAI/bge-small-en-v1.5"
        assert p.params["device"] == "cuda:0"
        assert p.params["dtype"] == "fp16"

    def test_openai_with_dimensions(self):
        """Openai with dimensions."""
        p = EmbeddingProvider.from_uri("openai://text-embedding-3-small?dimensions=512&api_key_env=OPENAI_API_KEY")
        assert p.kind == "openai"
        assert p.params["dimensions"] == "512"  # raw URL string -- caller casts
        assert p.params["api_key_env"] == "OPENAI_API_KEY"

    def test_sbert_alias(self):
        """Sbert alias."""
        p = EmbeddingProvider.from_uri("sbert://all-MiniLM-L6-v2")
        assert p.kind == "sentence-transformers"
        assert p.model == "all-MiniLM-L6-v2"

    def test_onnx_local_path(self):
        # Note: ONNX URIs to local files use a single leading slash;
        # absolute Windows paths via from_uri are an explicit non-goal
        # for v1 (round-3 R2-4 surface). Document via test:
        """Onnx local path."""
        p = EmbeddingProvider.from_uri("onnx://path/to/model.onnx")
        assert p.kind == "onnx"
        assert p.model == "path/to/model.onnx"

    @pytest.mark.parametrize(
        "bad",
        [
            "hf:noscheme",  # no //
            "://no-kind",  # empty kind
            "hf://?",  # empty model + empty query
            "??",  # nonsense
        ],
    )
    def test_malformed_uris_raise(self, bad):
        """Malformed uris raise."""
        with pytest.raises(ValueError):
            EmbeddingProvider.from_uri(bad)

    def test_unknown_kind_raises(self):
        """Unknown kind raises."""
        with pytest.raises(ValueError, match="unknown provider kind"):
            EmbeddingProvider.from_uri("lmstudio://foo")


# =====================================================================
# 3. Signature stability + secret-scrub
# =====================================================================


class TestSignature:
    """Groups tests covering signature."""
    def test_signature_is_string(self):
        """Signature is string."""
        p = EmbeddingProvider(kind="huggingface", model="intfloat/multilingual-e5-small")
        sig = p.signature
        assert isinstance(sig, str)
        assert sig.startswith("huggingface:intfloat/multilingual-e5-small:")

    def test_signature_stable_across_two_constructions(self):
        """Signature stable across two constructions."""
        p1 = EmbeddingProvider(kind="huggingface", model="x", params={"pool": "mean", "device": "auto"})
        p2 = EmbeddingProvider(
            kind="huggingface",
            model="x",
            params={"device": "auto", "pool": "mean"},
        )
        # Same content, different insertion order -- signatures must match.
        assert p1.signature == p2.signature

    def test_signature_changes_with_model(self):
        """Signature changes with model."""
        p1 = EmbeddingProvider(kind="huggingface", model="A")
        p2 = EmbeddingProvider(kind="huggingface", model="B")
        assert p1.signature != p2.signature

    def test_signature_changes_with_params(self):
        """Signature changes with params."""
        p1 = EmbeddingProvider(kind="huggingface", model="x", params={"pool": "cls"})
        p2 = EmbeddingProvider(kind="huggingface", model="x", params={"pool": "mean"})
        assert p1.signature != p2.signature

    def test_signature_does_NOT_change_with_secret_value(self):
        """Round-3 R2-6: changing api_key value (e.g. env-var swap)
        MUST NOT invalidate cache. Signature scrubs secrets before
        hashing."""
        p1 = EmbeddingProvider(
            kind="openai",
            model="x",
            params={"api_key": "sk-real-1"},
        )
        p2 = EmbeddingProvider(
            kind="openai",
            model="x",
            params={"api_key": "sk-real-2"},
        )
        assert p1.signature == p2.signature

    def test_signature_changes_with_non_secret_param(self):
        """Signature changes with non secret param."""
        p1 = EmbeddingProvider(kind="openai", model="x", params={"dimensions": 256})
        p2 = EmbeddingProvider(kind="openai", model="x", params={"dimensions": 1024})
        assert p1.signature != p2.signature


# =====================================================================
# 4. Secrets — env-var indirection
# =====================================================================


class TestSecrets:
    """Groups tests covering secrets."""
    def test_resolve_secrets_substitutes_env(self, monkeypatch):
        """Resolve secrets substitutes env."""
        monkeypatch.setenv("MLFRAME_TEST_KEY", "sk-test-12345")
        p = EmbeddingProvider(
            kind="openai",
            model="x",
            params={"api_key": "env:MLFRAME_TEST_KEY", "dimensions": 256},
        )
        resolved = p.resolve_secrets()
        assert resolved.params["api_key"] == "sk-test-12345"
        assert resolved.params["dimensions"] == 256
        # Original is unchanged (immutable copy).
        assert p.params["api_key"] == "env:MLFRAME_TEST_KEY"

    def test_resolve_secrets_missing_env_raises(self, monkeypatch):
        """Resolve secrets missing env raises."""
        monkeypatch.delenv("MLFRAME_NONEXISTENT_KEY", raising=False)
        p = EmbeddingProvider(kind="openai", model="x", params={"api_key": "env:MLFRAME_NONEXISTENT_KEY"})
        with pytest.raises(KeyError, match="MLFRAME_NONEXISTENT_KEY"):
            p.resolve_secrets()

    def test_resolve_secrets_no_env_refs_is_noop(self):
        """Resolve secrets no env refs is noop."""
        p = EmbeddingProvider(kind="openai", model="x", params={"api_key": "sk-literal"})
        resolved = p.resolve_secrets()
        assert resolved.params == p.params


# =====================================================================
# 5. Scrubbed serialisation -- secrets out of repr / model_dump
# =====================================================================


class TestScrubSecrets:
    """Groups tests covering scrub secrets."""
    def test_repr_masks_api_key(self):
        """Repr masks api key."""
        p = EmbeddingProvider(
            kind="openai",
            model="x",
            params={"api_key": "sk-leaky"},
        )
        assert "sk-leaky" not in repr(p)
        assert "***" in repr(p)

    def test_model_dump_default_scrubs(self):
        """Model dump default scrubs."""
        p = EmbeddingProvider(
            kind="openai",
            model="x",
            params={"api_key": "sk-leaky", "dimensions": 256},
        )
        d = p.model_dump()
        assert d["params"]["api_key"] == "***"
        assert d["params"]["dimensions"] == 256

    def test_model_dump_unscrubbed_for_runtime_handoff(self):
        """When loading the actual provider we need the real key.
        ``scrub_secrets=False`` exposes it -- only call from inside
        the provider acquire() flow."""
        p = EmbeddingProvider(
            kind="openai",
            model="x",
            params={"api_key": "sk-leaky"},
        )
        d = p.model_dump(scrub_secrets=False)
        assert d["params"]["api_key"] == "sk-leaky"

    @pytest.mark.parametrize(
        "key",
        [
            "api_key",
            "openai_api_key",
            "auth_token",
            "bearer_token",
            "secret_key",
            "credential",
            "password",
            "passwd",
        ],
    )
    def test_repr_masks_various_secret_keywords(self, key):
        """Repr masks various secret keywords."""
        p = EmbeddingProvider(
            kind="custom",
            model="x",
            params={key: "VALUE-LEAKY", "harmless": 5},
        )
        rep = repr(p)
        assert "VALUE-LEAKY" not in rep
        assert str(5) in rep  # harmless preserved

    def test_signature_uses_scrubbed_dump(self):
        """Internal: the signature uses ``_scrub_dict`` so secrets
        don't perturb cache keys."""
        p = EmbeddingProvider(
            kind="openai",
            model="x",
            params={"api_key": "real"},
        )
        # Hash should not contain the literal secret
        assert "real" not in p.signature


# =====================================================================
# 6. Default provider used in FHC
# =====================================================================


class TestProviderInFHC:
    """Groups tests covering provider in f h c."""
    def test_fhc_accepts_provider(self):
        """Fhc accepts provider."""
        from mlframe.training.feature_handling import FeatureHandlingConfig

        provider = EmbeddingProvider(
            kind="huggingface",
            model="intfloat/multilingual-e5-small",
        )
        fhc = FeatureHandlingConfig(default_text_provider=provider)
        assert fhc.default_text_provider.kind == "huggingface"
        assert fhc.default_text_provider.model == "intfloat/multilingual-e5-small"

    def test_fhc_accepts_uri_string_via_provider_construction(self):
        """Fhc accepts uri string via provider construction."""
        from mlframe.training.feature_handling import FeatureHandlingConfig

        # User-friendly idiom: convert URI to structured at construct time.
        provider = EmbeddingProvider.from_uri("hf://intfloat/multilingual-e5-small")
        fhc = FeatureHandlingConfig(default_text_provider=provider)
        assert fhc.default_text_provider.kind == "huggingface"
