"""Backport of LightGBM 4.7's ``feature_names_in_`` fix to LightGBM 4.5 / 4.6.

On 4.5 / 4.6 ``LGBMModel.feature_names_in_`` always returns names, even for a model fitted on a numpy array (LightGBM's
auto names ``Column_0..Column_{n-1}``). scikit-learn then treats every such model as "fitted with feature names" and
every numpy ``predict`` prints "X does not have valid feature names, but LGBMRegressor was fitted with feature names" --
a false warning that fires from every numpy fit/predict pair (the composite-discovery probes, the achievable-ceiling
precheck, ...). 4.7 raises ``AttributeError`` when the training data had no names; this applies the same rule on
4.5 / 4.6, where the only trace left of "no names given" is the exact auto-name sequence. The genuine warning (fitted on
a DataFrame, predicted on numpy) is untouched.

lightgbm is imported lazily across mlframe, so the patch is applied by a meta-path hook the moment ``lightgbm.sklearn``
is imported (or at once when it already is). The hook costs nothing until then and removes itself after firing.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.metadata
import logging
import re
import sys
from typing import Any

logger = logging.getLogger(__name__)

_TARGET_MODULE = "lightgbm.sklearn"
_VERSION_RE = re.compile(r"(\d+)\.(\d+)")


def needs_backport(version: str) -> bool:
    """True for the LightGBM releases whose ``feature_names_in_`` reports auto names as real ones (4.5 and 4.6)."""
    m = _VERSION_RE.match(str(version))
    return m is not None and (4, 5) <= (int(m[1]), int(m[2])) < (4, 7)


def _is_auto_names(names: Any) -> bool:
    """True when ``names`` is exactly LightGBM's auto sequence ``Column_0..Column_{n-1}`` (the fit had no names)."""
    return len(names) > 0 and all(str(n) == f"Column_{i}" for i, n in enumerate(names))


def patch_lgbm_model_class(cls: type) -> bool:
    """Wrap ``cls.feature_names_in_`` so it raises ``AttributeError`` for a model fitted without feature names.

    Returns True when the class was patched (False when already patched or the property is missing).
    """
    prop = cls.__dict__.get("feature_names_in_")
    if not isinstance(prop, property) or getattr(prop.fget, "_mlframe_backport", False):
        return False
    original_get = prop.fget

    def feature_names_in_(self):
        """scikit-learn compatible ``feature_name_``; absent when the training data had no feature names (as in 4.7)."""
        names = original_get(self)
        if _is_auto_names(names):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute 'feature_names_in_'. The training data did not have "
                "feature names (e.g. was a numpy array rather than a pandas DataFrame)."
            )
        return names

    feature_names_in_._mlframe_backport = True
    cls.feature_names_in_ = property(feature_names_in_, prop.fset, prop.fdel, prop.__doc__)
    return True


def _apply(sklearn_module: Any) -> None:
    """Patch ``lightgbm.sklearn.LGBMModel`` when the installed LightGBM needs it."""
    # Read from the distribution metadata: this runs while ``lightgbm/__init__`` is still executing, before its
    # ``__version__`` is guaranteed to be set.
    if needs_backport(importlib.metadata.version("lightgbm")):
        patch_lgbm_model_class(sklearn_module.LGBMModel)


class _PatchingLoader(importlib.abc.Loader):
    """Runs the real loader, then applies the backport to the freshly executed module."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def create_module(self, spec):
        """Delegate module creation to the wrapped real loader."""
        return self._inner.create_module(spec)

    def exec_module(self, module) -> None:
        """Execute the module with the real loader, then apply the backport (failures are logged, never raised)."""
        self._inner.exec_module(module)
        try:
            _apply(module)
        except Exception as exc:  # -- a compat patch must never break importing lightgbm; worst case the false warning stays
            logger.debug("LightGBM feature_names_in_ backport not applied: %s", exc)


class _LightGBMSklearnFinder(importlib.abc.MetaPathFinder):
    """Intercepts the import of ``lightgbm.sklearn`` once to wrap its loader."""

    def find_spec(self, fullname, path, target=None):
        """Return the real spec for ``lightgbm.sklearn`` with its loader wrapped; the finder uninstalls itself on first hit."""
        if fullname != _TARGET_MODULE:
            return None
        _remove_finder()
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = _PatchingLoader(spec.loader)
        return spec


def _remove_finder() -> None:
    """Drop every installed ``_LightGBMSklearnFinder`` from ``sys.meta_path``."""
    sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, _LightGBMSklearnFinder)]


def install() -> None:
    """Apply the backport now if ``lightgbm.sklearn`` is loaded, else when it is first imported."""
    loaded = sys.modules.get(_TARGET_MODULE)
    if loaded is not None:
        try:
            _apply(loaded)
        except Exception as exc:  # -- a compat patch must never break importing mlframe; worst case the false warning stays
            logger.debug("LightGBM feature_names_in_ backport not applied: %s", exc)
        return
    if not any(isinstance(f, _LightGBMSklearnFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _LightGBMSklearnFinder())


__all__ = ["install", "needs_backport", "patch_lgbm_model_class"]
