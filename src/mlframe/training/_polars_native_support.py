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
import subprocess
import sys
import threading
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# (library, version) -> whether a polars frame survived fit + predict. Keyed by version so an upgrade inside a
# long-lived process is not answered from a stale probe.
#
# The lock covers the whole miss-probe-store sequence: the probe FITS A MODEL, so two threads racing the same
# key would each pay that cost, and one could store a result derived from a half-imported library the other
# was still loading. Reads go through it too, so a reader never sees a partially-populated entry.
_CACHE: Dict[Tuple[str, str], bool] = {}
_CACHE_LOCK = threading.Lock()


# The probe body, run as ``python -c``. It must be a STRING executed in a CHILD process, not a function call in
# this one: LightGBM answers a polars frame carrying a Categorical/Enum column with
#     [LightGBM] [Fatal] Unsupported Arrow type: dictionary
#     terminate called without an active exception
# which is a C++ abort, not a Python exception. A try/except around an in-process call cannot catch it -- it
# took the whole pytest worker down on CI (three shards, deterministically) while the same LightGBM build on a
# developer box raised an ordinary exception and looked safe. A child process turns that abort into an exit
# code. Dropping the Enum column instead is not an option: whether the library handles categoricals IS the
# question, and a probe that avoided them would answer True and move the abort onto real data.
_PROBE_SOURCE = """
import sys
import polars as pl

library = sys.argv[1] if len(sys.argv) > 1 else ""
frame = pl.DataFrame(
    {
        "num": [1.0, 2.0, 3.0, 4.0],
        "nullable": [1.0, None, 3.0, None],
        "cat": pl.Series(["a", "b", "a", "b"], dtype=pl.Enum(["a", "b"])),
    }
)
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
    raise SystemExit(2)
model.predict(frame)
"""

# A cold interpreter plus a catboost/lightgbm import is a few seconds; anything beyond this means the probe
# itself hung, and hanging the suite to answer an optimisation question is not a trade worth making.
_PROBE_TIMEOUT_SECONDS = 120


def _probe(library: str) -> bool:
    """True when a CHILD process fits and predicts ``library`` on a polars frame without dying.

    Isolated on purpose -- see ``_PROBE_SOURCE``. Any non-zero exit, signal, timeout or missing interpreter
    answers False, which routes the caller through the pandas path it would otherwise have needed anyway.
    """
    try:
        import polars as pl  # noqa: F401
    except ImportError:
        return False
    if not sys.executable:
        logger.debug("no interpreter to run the %s polars probe in; assuming not native", library)
        return False
    try:
        completed = subprocess.run(  # nosec B603 - fixed source string, no shell, argv built here
            [sys.executable, "-c", _PROBE_SOURCE, library],
            capture_output=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("%s polars-native probe did not complete (%s: %s)", library, type(exc).__name__, exc)
        return False
    if completed.returncode != 0:
        logger.debug(
            "%s polars-native probe exited %s: %s", library, completed.returncode,
            (completed.stderr or b"").decode("utf-8", "replace").strip()[-200:],
        )
    return completed.returncode == 0

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
    with _CACHE_LOCK:
        if key not in _CACHE:
            _CACHE[key] = _probe(library)
            logger.debug("%s %s polars-native support: %s", library, version, _CACHE[key])
        return _CACHE[key]


def reset_cache() -> None:
    """Forget the probe results, so a test can re-probe against a patched library."""
    with _CACHE_LOCK:
        _CACHE.clear()


__all__ = ["accepts_polars", "reset_cache"]
