"""Verified, restricted loading of a suite's ``metadata.*`` file, shared by ``load_mlframe_suite`` and the predict entry points."""

from __future__ import annotations

import io
from typing import Any

from mlframe.training._bounded_zstd import decompress_bounded
from mlframe.training.io import _SafeUnpickler, safe_joblib_load
from mlframe.utils.safe_pickle import UNVERIFIED_ENV_VAR
from mlframe.utils.safe_pickle import verify_sidecar


def load_metadata_file(metadata_file: str, kind: str, caller: str) -> Any:
    """Load ``metadata_file`` (``kind`` is ``pkl.zst``, ``pkl`` or ``joblib``) through the restricted unpickler.

    A pickle metadata file without a matching sha256 sidecar is refused unless ``MLFRAME_ALLOW_UNVERIFIED_PICKLE`` opts in. Legacy ``joblib`` metadata
    predates sidecars and goes through the denylist joblib loader.
    """
    if kind == "joblib":
        return safe_joblib_load(metadata_file)
    if not verify_sidecar(metadata_file):
        raise RuntimeError(
            f"{caller}: sha256 sidecar missing or mismatched on {metadata_file!r}; refusing to load. "
            f"Set {UNVERIFIED_ENV_VAR}=1 to load a legacy bundle without a sidecar."
        )
    with open(metadata_file, "rb") as f:
        raw = f.read()
    payload = decompress_bounded(raw) if kind == "pkl.zst" else raw
    return _SafeUnpickler(io.BytesIO(payload)).load()
