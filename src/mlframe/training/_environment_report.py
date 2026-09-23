"""The interpreter and library versions a run was executed with, logged once at startup.

A production kernel died with an access violation inside LightGBM and the log could not say which LightGBM it was: the
run used an interpreter from another drive than the development checkout, and nothing in the log named either. Version
banners cannot be reconstructed after the fact from a crash dump, so they are recorded before the work starts.

Versions come from the installed distribution metadata rather than by importing each package: reporting must not pay
the import cost of torch or cupy, nor change what a run has loaded. A package that is absent is simply left out; a
package already imported is reported from its own ``__version__`` when the metadata lookup finds nothing.
"""

from __future__ import annotations

import logging
import platform
import sys
from typing import Dict

logger = logging.getLogger(__name__)

REPORTED_PACKAGES: tuple[str, ...] = (
    "mlframe",
    "pyutilz",
    "numpy",
    "scipy",
    "pandas",
    "polars",
    "pyarrow",
    "scikit-learn",
    "lightgbm",
    "xgboost",
    "catboost",
    "numba",
    "llvmlite",
    "joblib",
    "torch",
    "cupy",
    "shap",
)
"""What a crash or a performance regression is usually blamed on: the boosters, the frame libraries, the JIT and the
GPU stacks. Order is the reporting order."""

_IMPORT_NAMES: Dict[str, str] = {"scikit-learn": "sklearn"}
"""Distribution names whose import name differs, for the ``__version__`` fallback."""


def package_versions() -> Dict[str, str]:
    """``{distribution name: version}`` for every reported package that is installed; absent ones are left out."""
    from importlib.metadata import PackageNotFoundError, version

    out: Dict[str, str] = {}
    for name in REPORTED_PACKAGES:
        try:
            out[name] = version(name)
            continue
        except PackageNotFoundError:
            pass
        except Exception as exc:  # a broken distribution must not cost the run its banner
            logger.debug("version lookup failed for %r: %s", name, exc)
        mod = sys.modules.get(_IMPORT_NAMES.get(name, name))
        mod_version = getattr(mod, "__version__", None) if mod is not None else None
        if mod_version:
            out[name] = str(mod_version)
    return out


def environment_summary() -> str:
    """One line naming the interpreter, the platform and every reported package version."""
    versions = ", ".join(f"{name}={ver}" for name, ver in package_versions().items())
    return f"python {sys.version.split()[0]} at {sys.executable}; {platform.platform()}; {versions}"


def log_environment_versions() -> str:
    """Log :func:`environment_summary` at INFO and return it. Never raises."""
    try:
        summary = environment_summary()
    except Exception as exc:  # pragma: no cover - the banner is diagnostics, never a reason to fail a run
        logger.warning("environment version banner unavailable: %s", exc)
        return ""
    logger.info("Environment: %s", summary)
    return summary
