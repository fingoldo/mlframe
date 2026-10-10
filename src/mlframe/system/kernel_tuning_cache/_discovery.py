"""Find every registered kernel tuner without running the package's benchmark scripts.

A tuner registers itself when its module is imported, so finding them used to mean importing EVERY module of the package - thousands of them, including benchmark and profiling
scripts that change the process at import: ``CUDA_VISIBLE_DEVICES=''`` (the GPU disappears), ``NUMBA_DISABLE_CUDA=1``, ``MLFRAME_FE_GPU_STRICT=1``, ``MLFRAME_MI_BACKEND=njit``,
``OMP_NUM_THREADS=1``, ``sys.modules["cupy"] = None``. The process that was supposed to measure the GPU then saw none: its hardware fingerprint read "no-gpu", sweeps built no GPU
variants and persisted zero regions, and tunings landed under the wrong fingerprint.

So the files are read, not run. A module is imported only if its syntax tree contains a call to ``kernel_tuner`` - about thirty library modules - and nothing else is touched.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Iterator

from pyutilz.performance.kernel_tuning.registry import get_registry

__all__ = ["discover_specs", "registering_modules"]

logger = logging.getLogger(__name__)

# Modules a registering module's import chain might block and a sweep needs.
_GUARDED = ("cupy",)


def _registers_a_tuner(path: Path) -> bool:
    """Whether the file calls ``kernel_tuner(...)``, also under an import alias (a docstring or comment that merely mentions it does not count)."""
    try:
        text = path.read_text(encoding="utf-8")
        if "kernel_tuner" not in text:
            return False
        tree = ast.parse(text, filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError) as exc:
        logger.warning("tuner discovery: cannot read %s (%s: %s); any tuner it registers is skipped", path, type(exc).__name__, exc)
        return False
    names = {"kernel_tuner"}
    for node in ast.walk(tree):  # `from ...registry import kernel_tuner as _ktuner` registers under another name
        if isinstance(node, ast.ImportFrom):
            names.update(alias.asname for alias in node.names if alias.name == "kernel_tuner" and alias.asname)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
            if name in names:
                return True
    return False


def registering_modules(package: str = "mlframe") -> Iterator[str]:
    """Dotted names of the modules under ``package`` that register a tuner, found by reading their source."""
    spec = importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        return
    for base in spec.submodule_search_locations:
        base_path = Path(base)
        for dirpath, dirnames, filenames in os.walk(base_path):
            dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
            for filename in sorted(filenames):
                if not filename.endswith(".py"):
                    continue
                path = Path(dirpath) / filename
                if not _registers_a_tuner(path):
                    continue
                parts = list(path.relative_to(base_path).with_suffix("").parts)
                if parts and parts[-1] == "__init__":
                    parts.pop()
                yield ".".join([package, *parts])


def _capture(name: str):
    """The module to protect: whatever ``sys.modules`` holds for it (``None`` means the caller blocked it on purpose), else the real module if it can be imported."""
    if name in sys.modules:
        return sys.modules[name]
    try:
        return importlib.import_module(name)
    except Exception as exc:  # not installed / no driver: there is nothing to protect
        logger.debug("tuner discovery: %s is not importable (%s: %s)", name, type(exc).__name__, exc)
        return None


def _import_guarded(modname: str, real: dict) -> None:
    """Import one module; a failure is logged and skipped, and a guarded dependency its import chain blocked is put back either way."""
    try:
        importlib.import_module(modname)
    except (Exception, SystemExit) as exc:  # one broken module must not stop the others
        logger.warning("tuner discovery: could not import %s (%s: %s); its kernels are not tuned", modname, type(exc).__name__, exc)
    finally:
        for name, module in real.items():
            if module is not None and sys.modules.get(name) is None:
                sys.modules[name] = module


def discover_specs(package: str = "mlframe") -> dict:
    """Import the modules that register tuners and return ``{kernel_name: TunerSpec}``.

    A guarded dependency (cupy) that was blocked before the call stays blocked - an explicit CPU-only choice is the caller's - and one that was usable stays usable.
    """
    real = {name: _capture(name) for name in _GUARDED}
    for modname in registering_modules(package):
        _import_guarded(modname, real)
    return get_registry()
