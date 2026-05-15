"""
:class:`EmbeddingProvider` structured config + URI parser + secrets
handling. Foundation for the FrozenFeaturizerProvider / TrainableFeaturizerProvider
Protocols (phase B).

Round-2 U-R5: previously the provider was a colon-delimited string
(``"hf:BAAI/bge-small-en-v1.5"``) which broke on:

  * model names with colons (e.g. Windows paths ``onnx:///C:/...``);
  * provider-specific params (OpenAI ``dimensions``, ``api_key``,
    HF ``trust_remote_code``);
  * secrets serialisation (env-var indirection + scrubber).

The structured form solves all three. The string-shorthand
``EmbeddingProvider.from_uri()`` is convenience-only: it parses a
URL-style ``hf://model?param=value`` form into the structured object.

Naming finalised: ``frozen_text_embedding`` / ``learnable_text_embedding``
methods take ``provider: Optional[EmbeddingProvider]`` -- when ``None``
the FHC-level ``default_text_provider`` applies, with auto-locale
detection picking ``intfloat/multilingual-e5-small`` (multilingual
default per round-3 user confirmation).
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional
from urllib.parse import parse_qs, urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    pass


# Conventional secret-keyword set used for repr/dump scrubbing.
# Round-3 S6: previously only "*key*"/"*token*"/"*secret*"; expanded
# to also cover *auth*/*bearer*/*credential*/*password* (false-negative
# coverage class).
_SECRET_KEY_PATTERNS = (
    "key",
    "token",
    "secret",
    "auth",
    "bearer",
    "credential",
    "password",
    "passwd",
)

_ENV_PREFIX = "env:"

_URI_RE = re.compile(r"^(?P<kind>[a-z][a-z0-9-]+)://(?P<rest>.+)$", re.IGNORECASE)

_URI_KIND_ALIAS: Dict[str, str] = {
    "hf": "huggingface",
    "huggingface": "huggingface",
    "sbert": "sentence-transformers",
    "sentence-transformers": "sentence-transformers",
    "openai": "openai",
    "cohere": "cohere",
    "jina": "jina",
    "voyage": "voyage",
    "onnx": "onnx",
    "fasttext": "fasttext",
    "tfhub": "tfhub",
    "custom": "custom",
}


def _is_secret_field(field_name: str) -> bool:
    """Heuristic: does a field name look secret-bearing?"""
    lname = field_name.lower()
    return any(pat in lname for pat in _SECRET_KEY_PATTERNS)


def _scrub_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Mask secret-like values for repr / model_dump."""
    return {
        k: ("***" if _is_secret_field(k) else v)
        for k, v in d.items()
    }


class EmbeddingProvider(BaseModel):
    """Structured embedding-source descriptor. Replaces the earlier
    colon-string. Holds enough information that
    :attr:`signature` (a stable string) is suitable for cache keys
    after phase D wires the proper :class:`ProviderIdentity` dataclass
    that includes library versions etc.

    Provider params support env-var indirection via the ``"env:VAR"``
    sentinel. :meth:`resolve_secrets` substitutes the resolved value
    at runtime; :meth:`model_dump` (default ``scrub_secrets=True``)
    masks any value whose key looks secret-bearing so YAML dumps and
    repr() output do not leak credentials.

    Examples
    --------
    >>> EmbeddingProvider(kind="huggingface", model="intfloat/multilingual-e5-small")
    >>> EmbeddingProvider.from_uri("hf://BAAI/bge-small-en-v1.5?device=cuda:0")
    >>> EmbeddingProvider(
    ...     kind="openai",
    ...     model="text-embedding-3-small",
    ...     params={"api_key": "env:OPENAI_API_KEY", "dimensions": 512},
    ... )
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal[
        "huggingface",
        "sentence-transformers",
        "openai",
        "cohere",
        "jina",
        "voyage",
        "onnx",
        "fasttext",
        "tfhub",
        "custom",
    ]
    model: str
    params: Dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # URI shorthand
    # ------------------------------------------------------------------

    @classmethod
    def from_uri(cls, uri: str) -> EmbeddingProvider:
        """Parse a URL-style URI into the structured form.

        Grammar::

            <kind>://<model>[?<param>=<value>(&<param>=<value>)*]

        Examples
        --------
        >>> p = EmbeddingProvider.from_uri("hf://BAAI/bge-small-en-v1.5?device=cuda:0")
        >>> p.kind, p.model, p.params
        ('huggingface', 'BAAI/bge-small-en-v1.5', {'device': 'cuda:0'})

        Unknown ``kind`` raises ``ValueError``. Query values keep their
        raw URL-decoded string form -- callers cast to int/bool when
        the provider class expects it.
        """
        m = _URI_RE.match(uri.strip())
        if not m:
            raise ValueError(
                f"invalid EmbeddingProvider URI {uri!r}; expected "
                f"'<kind>://<model>[?param=value]'"
            )
        kind_raw = m.group("kind").lower()
        if kind_raw not in _URI_KIND_ALIAS:
            valid = sorted(_URI_KIND_ALIAS)
            raise ValueError(
                f"unknown provider kind {kind_raw!r} in URI {uri!r}; valid: {valid}"
            )
        kind = _URI_KIND_ALIAS[kind_raw]
        rest = m.group("rest")
        # Use urllib.parse to split off the query portion. urlsplit
        # treats the leading bit as the path (since we ate the scheme
        # in the regex) -- but we hand it back the original URI minus
        # the part already-parsed so query handling is robust.
        if "?" in rest:
            model_part, query_part = rest.split("?", 1)
            params_raw = parse_qs(query_part, keep_blank_values=True)
            # parse_qs returns each value as a list; flatten
            params = {k: (v[0] if len(v) == 1 else v) for k, v in params_raw.items()}
        else:
            model_part = rest
            params = {}
        if not model_part:
            raise ValueError(f"empty model name in URI {uri!r}")
        return cls(kind=kind, model=model_part, params=params)

    # ------------------------------------------------------------------
    # Signature
    # ------------------------------------------------------------------

    @property
    def signature(self) -> str:
        """Stable string used in cache keys.

        Phase A2 form: ``{kind}:{model}:{params_hash}`` with secrets
        scrubbed before hashing so a swapped env-var doesn't perturb
        the signature (round-3 R2-6: ``api_key`` env var changes must
        NOT invalidate cache).

        Phase D upgrades to the full :class:`ProviderIdentity` dataclass
        with library versions, revision SHA, dtype, pool, etc.
        """
        scrubbed = _scrub_dict(self.params)
        # Sort keys for determinism
        items = sorted(scrubbed.items(), key=lambda kv: kv[0])
        params_repr = ";".join(f"{k}={v}" for k, v in items)
        h = hashlib.blake2b(params_repr.encode("utf-8"), digest_size=8)
        return f"{self.kind}:{self.model}:{h.hexdigest()}"

    # ------------------------------------------------------------------
    # Secrets resolution
    # ------------------------------------------------------------------

    def resolve_secrets(self) -> EmbeddingProvider:
        """Return a copy with ``"env:VAR"`` references replaced by the
        actual env var value. Raises ``KeyError`` if any referenced
        env var is unset (loud-failure mode -- silent missing-secret
        is the worst-case for debug)."""
        resolved: Dict[str, Any] = {}
        for k, v in self.params.items():
            if isinstance(v, str) and v.startswith(_ENV_PREFIX):
                env_name = v[len(_ENV_PREFIX):]
                env_val = os.environ.get(env_name)
                if env_val is None:
                    raise KeyError(
                        f"EmbeddingProvider param {k!r} references env var "
                        f"{env_name!r} which is unset. Set it or replace "
                        f"with the literal value."
                    )
                resolved[k] = env_val
            else:
                resolved[k] = v
        return self.model_copy(update={"params": resolved})

    # ------------------------------------------------------------------
    # Serialisation -- scrub secrets by default
    # ------------------------------------------------------------------

    def model_dump(self, *, scrub_secrets: bool = True, **kwargs: Any) -> Dict[str, Any]:
        """Override that scrubs secret-like keys in ``params`` by default.

        Round-3 S6 + U-R2-28: stored YAML / repr / debugger output
        must not leak api_key etc. Pass ``scrub_secrets=False`` to
        disable -- only use that for resolved-secrets handoff to the
        actual provider load step.
        """
        out = super().model_dump(**kwargs)
        if scrub_secrets and isinstance(out.get("params"), dict):
            out["params"] = _scrub_dict(out["params"])
        return out

    def __repr__(self) -> str:
        # Round-3 U-R2-28: repr also masks secrets so debugger /
        # traceback / sentry doesn't leak them. pydantic's default
        # repr walks all fields; we override.
        scrubbed_params = _scrub_dict(self.params)
        return (
            f"EmbeddingProvider(kind={self.kind!r}, model={self.model!r}, "
            f"params={scrubbed_params!r})"
        )


__all__ = ["EmbeddingProvider"]
