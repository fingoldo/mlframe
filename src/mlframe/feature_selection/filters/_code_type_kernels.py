"""Compile cache of the CUDA kernels that take byte codes when every cardinality is at most 127 and 32-bit integer codes otherwise."""

from __future__ import annotations

from typing import Any


def compile_code_kernel(cp: Any, cache: dict, source: str, name: str, narrow: bool) -> Any:
    """The kernel ``name`` of ``source`` for byte codes (``narrow``) or 32-bit integer codes, compiled once per code type into ``cache``."""
    if narrow not in cache:
        cache[narrow] = cp.RawKernel(source.replace("{CT}", "signed char" if narrow else "int"), name)
    return cache[narrow]
