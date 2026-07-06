"""Backward-compatible re-export of the sha256-sidecar pickle helpers.

The canonical implementation moved to ``pyutilz.core.safe_pickle`` (2026-07-06) so any
project depending on pyutilz gets the same verified-load primitive instead of
re-implementing it. This module now only wraps :func:`verify_sidecar` / :func:`safe_load`
to keep checking mlframe's historical ``MLFRAME_ALLOW_UNVERIFIED_PICKLE`` env var name
(pyutilz's own default is ``PYUTILZ_ALLOW_UNVERIFIED_PICKLE``) -- every internal caller,
the public API, and the env var contract are unchanged.

See ``pyutilz.core.safe_pickle`` for the full docstring (threat model caveat, function
contracts, etc.).
"""
from __future__ import annotations

from typing import Any, Optional

from pyutilz.core.safe_pickle import (
    PickleVerificationError,
    _sha256_of_file,
    safe_dump,
    write_sidecar,
)
from pyutilz.core.safe_pickle import safe_load as _pyutilz_safe_load
from pyutilz.core.safe_pickle import verify_sidecar as _pyutilz_verify_sidecar

__all__ = [
    "PickleVerificationError",
    "verify_sidecar",
    "write_sidecar",
    "safe_load",
    "safe_dump",
]

_ENV_VAR = "MLFRAME_ALLOW_UNVERIFIED_PICKLE"


def verify_sidecar(path: str, *, allow_unverified: Optional[bool] = None) -> bool:
    return _pyutilz_verify_sidecar(path, allow_unverified=allow_unverified, env_var=_ENV_VAR)


def safe_load(path: str, *, allow_unverified: Optional[bool] = None) -> Any:
    return _pyutilz_safe_load(path, allow_unverified=allow_unverified, env_var=_ENV_VAR)
