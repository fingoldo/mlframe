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

import hashlib
import hmac
import logging
import os
import re
from typing import Any, Optional

from mlframe.utils.env_flags import env_flag
from pyutilz.core.safe_pickle import (
    PickleVerificationError,
    _sha256_of_file,  # noqa: F401 -- re-exported for estimators/pipelines.py and inference/predict.py
    safe_dump,
    write_sidecar,
)
from pyutilz.core.safe_pickle import safe_load as _pyutilz_safe_load
from pyutilz.core.safe_pickle import verify_sidecar as _pyutilz_verify_sidecar

__all__ = [
    "PickleVerificationError",
    "verify_sidecar",
    "verify_sidecar_bytes",
    "write_sidecar",
    "safe_load",
    "safe_dump",
    "UNVERIFIED_ENV_VAR",
]

UNVERIFIED_ENV_VAR = "MLFRAME_ALLOW_UNVERIFIED_PICKLE"

_logger = logging.getLogger(__name__)
_SHA256_HEX = re.compile(r"[0-9a-fA-F]{64}")


def verify_sidecar(path: str, *, allow_unverified: Optional[bool] = None) -> bool:
    """Verify ``path``'s sha256 sidecar, honoring mlframe's ``MLFRAME_ALLOW_UNVERIFIED_PICKLE`` env var. See ``pyutilz.core.safe_pickle.verify_sidecar``."""
    return bool(_pyutilz_verify_sidecar(path, allow_unverified=allow_unverified, env_var=UNVERIFIED_ENV_VAR))


def safe_load(path: str, *, allow_unverified: Optional[bool] = None) -> Any:
    """Verify then unpickle ``path``, honoring mlframe's ``MLFRAME_ALLOW_UNVERIFIED_PICKLE`` env var. See ``pyutilz.core.safe_pickle.safe_load``."""
    return _pyutilz_safe_load(path, allow_unverified=allow_unverified, env_var=UNVERIFIED_ENV_VAR)


def verify_sidecar_bytes(path: str, payload: bytes, *, allow_unverified: Optional[bool] = None) -> bool:
    """Verify the sha256 sidecar of ``path`` against ``payload``, the bytes the caller read from it and is about to deserialize.

    Same contract as :func:`verify_sidecar` (sidecar present and matching: True; mismatching, malformed or unreadable: False; sidecar missing: False
    unless ``MLFRAME_ALLOW_UNVERIFIED_PICKLE`` is truthy), but the digest is taken over ``payload`` instead of re-reading ``path``, so what is verified
    is exactly what gets loaded and a file replaced between the check and the open cannot slip through.
    """
    sidecar = path + ".sha256"
    if not os.path.isfile(sidecar):
        if allow_unverified is None:
            allow_unverified = env_flag(UNVERIFIED_ENV_VAR)
        if allow_unverified:
            _logger.warning("verify_sidecar_bytes: no .sha256 sidecar for %s; %s is set so the load proceeds WITHOUT content verification.", path, UNVERIFIED_ENV_VAR)
            return True
        _logger.error("verify_sidecar_bytes: no .sha256 sidecar for %s; refusing to load (default-strict). Set %s=1 for a legacy bundle.", path, UNVERIFIED_ENV_VAR)
        return False
    try:
        with open(sidecar, encoding="utf-8") as f:
            expected = f.read().strip().split()[0].lower()
    except (OSError, UnicodeDecodeError, IndexError) as exc:
        _logger.error("verify_sidecar_bytes: could not read sidecar %s: %s", sidecar, exc)
        return False
    if not _SHA256_HEX.fullmatch(expected):
        _logger.error("verify_sidecar_bytes: sidecar %s does not hold a 64-character hex sha256 digest; refusing to verify.", sidecar)
        return False
    return hmac.compare_digest(expected, hashlib.sha256(payload).hexdigest())
