"""JSON load / save for the per-host kernel-tuning cache.

Schema (per file)::

    {
      "schema_version": 1,
      "hw_fingerprint": "cpu_<...>_gpu_<...>_cc<M.m>",
      "timestamp_utc": "2026-05-19T18:14:00Z",
      "kernels": {
        "joint_hist_batched": {
          "axes": ["n_samples", "joint_size"],
          "regions": [
            {
              "n_samples_max": 200000,
              "joint_size_max": 4096,
              "kernel_variant": "shared",
              "block_size": 512,
              "wall_ms": 0.04
            },
            ...
          ]
        },
        "discretize_2d_array": { ... },
        ...
      }
    }

* Per-kernel ``regions`` are listed in priority order; ``dispatch.lookup``
  returns the first region whose bounds match the request.
* Bounds are inclusive UPPER caps (``..._max``); the first matching region
  wins. A region with ``"joint_size_max": null`` matches every joint_size
  -- use as the catch-all fallback at end of the list.

Forward-compat: ``schema_version`` is the only required field at read
time. A reader that doesn't understand the version treats the cache as
missing and triggers a re-sweep; we never partially-apply an unknown
schema.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
from typing import Any, Optional

from .hw_fingerprint import cache_path, hw_fingerprint

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


def load() -> Optional[dict]:
    """Read the live host's cache file. Returns None if absent / unreadable
    / schema-version mismatch."""
    path = cache_path()
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("kernel_tuning_cache: failed to read %s: %s", path, e)
        return None
    if data.get("schema_version") != SCHEMA_VERSION:
        logger.info(
            "kernel_tuning_cache: schema mismatch at %s (got %r, expected %d); "
            "treating as miss",
            path, data.get("schema_version"), SCHEMA_VERSION,
        )
        return None
    if data.get("hw_fingerprint") != hw_fingerprint():
        # Could happen if MLFRAME_KERNEL_CACHE_DIR was shared across hosts.
        logger.info(
            "kernel_tuning_cache: hw_fingerprint mismatch at %s "
            "(got %r, expected %r); treating as miss",
            path, data.get("hw_fingerprint"), hw_fingerprint(),
        )
        return None
    return data


def save(kernels: dict[str, Any]) -> str:
    """Write ``kernels`` to the live host's cache file. Returns the path
    actually written. Atomic: writes to ``<path>.tmp`` then renames.
    """
    path = cache_path()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hw_fingerprint": hw_fingerprint(),
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "kernels": kernels,
    }
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)
    logger.info("kernel_tuning_cache: saved %s", path)
    return path


def update_kernel(kernel_name: str, regions: list[dict]) -> str:
    """Read-modify-write: replace only ``kernels[kernel_name]`` and persist
    the rest. Avoids losing tunings for other kernels when re-running a
    single one's sweep.
    """
    existing = load() or {}
    kernels = existing.get("kernels", {}) if existing else {}
    kernels[kernel_name] = {"axes": ["n_samples", "joint_size"], "regions": regions}
    return save(kernels)


__all__ = ["SCHEMA_VERSION", "load", "save", "update_kernel"]
