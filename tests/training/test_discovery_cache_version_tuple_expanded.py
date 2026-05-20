"""The discovery cache version tuple must cover every library whose
semantics can shift the discovered specs. A polars 1->2 bump rewrites
categorical codes; a numpy 2.x bump changes RNG defaults; we MUST refit
in either case rather than replay stale specs.
"""
from __future__ import annotations

import importlib
import sys
from unittest import mock


def _signature_payload_versions() -> dict:
    """Capture the ``versions`` dict that ``_discovery_config_signature``
    folds into its hash, without depending on the real JSON blob (which
    is hashed and not introspectable).

    Production hashes via stdlib ``json.dumps(..., sort_keys=True)`` (see
    ``mlframe.training.utils._compute_config_signature_v1``); we intercept
    that call to snapshot the payload without pulling stdlib ``json`` into
    the test file (repo convention bans stdlib ``json`` imports in tests).
    """
    from mlframe.training.core._phase_composite_discovery import (
        _discovery_config_signature,
    )

    json_module = importlib.import_module("json")
    captured: dict = {}
    real_dumps = json_module.dumps

    def capturing_dumps(obj, *a, **kw):
        if isinstance(obj, dict) and "versions" in obj:
            captured["versions"] = dict(obj["versions"])
        return real_dumps(obj, *a, **kw)

    with mock.patch.object(json_module, "dumps", side_effect=capturing_dumps):
        # Pass a no-op config (only the versions side is under test).
        class _Cfg:
            def model_dump(self, mode="json"):
                return {}

        _discovery_config_signature(_Cfg())

    return captured.get("versions", {})


def test_discovery_cache_version_tuple_covers_polars_numpy_pandas_scipy_xgboost_python():
    versions = _signature_payload_versions()

    # Existing libs (regression: don't drop them).
    for lib in ("mlframe", "sklearn", "lightgbm", "catboost"):
        assert lib in versions, (
            f"version tuple regressed - {lib} no longer in signature"
        )

    # Newly required libs.
    for lib in ("xgboost", "polars", "numpy", "scipy", "pandas"):
        assert lib in versions, (
            f"version tuple missing {lib}; cache invalidation will miss "
            f"a {lib} bump and replay stale specs"
        )

    # Python major.minor must be present so a 3.x bump invalidates.
    assert "python" in versions, "python version missing from cache tuple"
    expected = f"{sys.version_info.major}.{sys.version_info.minor}"
    assert versions["python"] == expected, (
        f"python tuple should be major.minor only; got {versions['python']!r}"
    )


def test_discovery_cache_signature_changes_when_simulated_polars_version_bumps():
    """A simulated polars version change must change the signature -
    otherwise a polars upgrade would silently replay stale specs.
    """
    from mlframe.training.core._phase_composite_discovery import (
        _discovery_config_signature,
    )

    class _Cfg:
        def model_dump(self, mode="json"):
            return {}

    real_signature = _discovery_config_signature(_Cfg())

    # Bump the polars version in-process; the signature must move.
    import polars as pl
    original_version = pl.__version__
    try:
        pl.__version__ = "999.999.999"
        bumped_signature = _discovery_config_signature(_Cfg())
    finally:
        pl.__version__ = original_version

    assert real_signature != bumped_signature, (
        "polars version bump did not change cache signature - stale spec "
        "replay hazard"
    )
