"""Ask the INSTALLED booster whether it takes a polars frame, instead of assuming either way.

Two costs come from guessing. Assuming polars works means a dispatch miss deep inside a Cython call, recovered
by a per-call pandas conversion that nothing in the log explains. Assuming it does not means converting a
multi-GB frame on every predict for a library that would have accepted it -- a production run showed exactly
that, a ``[predict fallback] polars->pandas(predict_proba)`` on every single call with no preceding failure.

So the question is answered empirically, once per (library, version), by fitting a two-row model on a polars
frame and predicting from one. The probe is a few milliseconds and its result is cached for the process.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# (library, version) -> whether a polars frame survived fit + predict. Keyed by version so an upgrade inside a
# long-lived process is not answered from a stale probe.
_CACHE: Dict[Tuple[str, str], bool] = {}


def _probe_frame():
    """A tiny polars frame with the dtypes that historically broke the dispatch: float, Enum, and a null."""
    import polars as pl

    return pl.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0],
            "nullable": [1.0, None, 3.0, None],
            "cat": pl.Series(["a", "b", "a", "b"], dtype=pl.Enum(["a", "b"])),
        }
    )


def _probe(library: str) -> bool:
    """Fit and predict a minimal model of ``library`` on a polars frame; True when both succeed."""
    try:
        import polars as pl  # noqa: F401
    except ImportError:
        return False
    try:
        frame = _probe_frame()
        y = [0, 1, 0, 1]
        if library == "catboost":
            from catboost import CatBoostClassifier

            model = CatBoostClassifier(iterations=2, depth=1, verbose=0, allow_writing_files=False)
            model.fit(frame, y, cat_features=["cat"])
        elif library == "lightgbm":
            import lightgbm as lgb

            model = lgb.LGBMClassifier(n_estimators=2, num_leaves=2, verbosity=-1)
            model.fit(frame, y)
        else:
            return False
        model.predict(frame)
        return True
    except Exception as exc:
        logger.debug("%s polars-native probe failed (%s: %s)", library, type(exc).__name__, exc)
        return False


def _version(library: str) -> Optional[str]:
    """Installed version string of ``library``, or None when it is absent."""
    try:
        module = __import__(library)
    except ImportError:
        return None
    return str(getattr(module, "__version__", "unknown"))


def accepts_polars(library: str) -> bool:
    """Whether the installed ``library`` ("catboost" / "lightgbm") consumes a polars frame end to end.

    Cached per (library, version). An absent library answers False, which routes the caller through the pandas
    path it would have needed anyway.
    """
    version = _version(library)
    if version is None:
        return False
    key = (library, version)
    if key not in _CACHE:
        _CACHE[key] = _probe(library)
        logger.debug("%s %s polars-native support: %s", library, version, _CACHE[key])
    return _CACHE[key]


def reset_cache() -> None:
    """Forget the probe results, so a test can re-probe against a patched library."""
    _CACHE.clear()


__all__ = ["accepts_polars", "reset_cache"]
