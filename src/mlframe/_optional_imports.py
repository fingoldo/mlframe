"""Lazy access to third-party packages that live in an optional extra, with an ImportError that names the extra to install."""

from __future__ import annotations

import importlib
from types import ModuleType


def import_optional(module: str, extra: str, purpose: str = "") -> ModuleType:
    """Import ``module``; if it is missing raise ``ImportError`` that names the ``mlframe[extra]`` extra providing it."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        suffix = f" ({purpose})" if purpose else ""
        raise ImportError(
            f"{module!r} is required{suffix} but is not installed; install it with `pip install mlframe[{extra}]` or `pip install {module.split('.')[0]}`."
        ) from exc
