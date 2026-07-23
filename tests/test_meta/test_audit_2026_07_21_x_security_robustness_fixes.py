"""Regression tests for audits/full_audit_2026-07-21/x_security_robustness.md findings SEC-1..SEC-5.

PROP-2 (route the SEC-1 call site through safe_joblib_load as a cheap first step, ahead of the larger
sidecar-writing work) is implemented as part of SEC-1's fix. PROP-3 (defensive __getstate__/__setstate__
on FeatureCache) is implemented as part of SEC-3's fix. PROP-1 is the same test SEC-4 asks for --
covered below. PROP-4 (an offline-report config knob to switch away from CDN plotly.js) is a larger
feature proposal with no reported bug -- deferred; SEC-5's fix only adds the documenting comment.
"""

from __future__ import annotations

import inspect

import pytest

# ---------------------------------------------------------------------------
# SEC-1 / SEC-4: the cached-model reload path now routes through safe_joblib_load's RCE-gadget
# denylist instead of plain joblib.load, and a planted-pickle regression test exists for it.
# ---------------------------------------------------------------------------


def test_sec1_trainer_reload_uses_safe_joblib_load_not_plain_joblib():
    """SEC-1 REGRESSION: the cached-model reload call site must route through safe_joblib_load's
    RCE-gadget denylist, not the unrestricted plain joblib.load."""
    import mlframe.training._trainer_train_and_evaluate as mod

    src = inspect.getsource(mod)
    assert "from mlframe.training.io import safe_joblib_load" in src
    assert "model, *_, pre_pipeline = safe_joblib_load(model_file_name)" in src
    assert "model, *_, pre_pipeline = joblib.load(model_file_name)" not in src


def test_sec1_sec4_safe_joblib_load_blocks_classic_rce_gadget(tmp_path):
    """SEC-4: a planted pickle using the classic os.system RCE gadget at an arbitrary path must be
    refused by safe_joblib_load (the function SEC-1's fix now routes the cached-model reload through),
    instead of silently executing. This is the exact "attacker plants a pickle at model_file_name"
    scenario SEC-1's docstring describes, with zero prior regression coverage per SEC-4."""
    import os
    import pickle

    import joblib

    from mlframe.training.io import safe_joblib_load

    class _EvilPayload:
        """Classic pickle RCE gadget: __reduce__ returns a callable + args pickle executes on load."""

        def __reduce__(self):
            """Return the RCE gadget's (callable, args) pair -- os.system('echo pwned') on unpickle."""
            return (os.system, ("echo pwned",))

    evil_path = tmp_path / "planted_model.dump"
    # joblib.dump so the file has the real joblib/numpy-pickle framing safe_joblib_load expects.
    joblib.dump(_EvilPayload(), evil_path)

    with pytest.raises(pickle.UnpicklingError, match="denylist"):
        safe_joblib_load(str(evil_path))


# ---------------------------------------------------------------------------
# SEC-2: _deserialize's pickle-fallback except no longer swallows genuine I/O errors
# ---------------------------------------------------------------------------


def test_sec2_deserialize_propagates_permission_error_not_pickle_fallback():
    """SEC-2 REGRESSION: a PermissionError from np.load must propagate as itself, not be funnelled
    into the legacy-pickle fallback (which would raise a confusing UnpicklingError instead)."""
    import mlframe.training.feature_handling.cache as mod

    src = inspect.getsource(mod._deserialize)
    assert "except (OSError, zipfile.BadZipFile):" in src
    assert "raise" in src.split("except (OSError, zipfile.BadZipFile):")[1].split("except Exception:")[0]


def test_sec2_deserialize_still_falls_back_for_genuine_legacy_pickle(tmp_path):
    """A genuinely non-npy/npz payload (not an I/O error) must still hit the legacy-pickle fallback,
    confirming SEC-2's narrowing didn't regress the fallback's real purpose."""
    from mlframe.training.feature_handling.cache import CachePickleRefusedError, _deserialize

    garbage_path = tmp_path / "not_a_valid_npy_file.npz"
    garbage_path.write_bytes(b"not a valid npy or pickle payload at all")

    with pytest.raises(CachePickleRefusedError):
        _deserialize(str(garbage_path), allow_pickle=False)


# ---------------------------------------------------------------------------
# SEC-3: FeatureCache now survives a pickle round-trip (threading.Lock stripped/restored)
# ---------------------------------------------------------------------------


def test_sec3_feature_cache_survives_pickle_round_trip():
    """SEC-3 REGRESSION: FeatureCache must be picklable (via __getstate__/__setstate__ dropping and
    re-creating the unpicklable threading.Lock), matching the defensive pattern already applied to
    training/neural/ranker.py's trainer_/CUDA-tensor exclusion for the same underlying bug class."""
    import pickle
    import threading

    from mlframe.training.feature_handling.cache import FeatureCache

    fc = FeatureCache(cache_cfg=None, content_fingerprint=None)
    restored = pickle.loads(pickle.dumps(fc))  # nosec B301 -- round-trip of a locally-created, trusted object
    assert isinstance(restored._lock, type(threading.Lock()))
    assert restored._mem == {}


def test_sec3_feature_cache_getstate_drops_lock():
    """FeatureCache.__getstate__ must not include the live threading.Lock object."""
    from mlframe.training.feature_handling.cache import FeatureCache

    fc = FeatureCache(cache_cfg=None, content_fingerprint=None)
    state = fc.__getstate__()
    assert state["_lock"] is None


# ---------------------------------------------------------------------------
# SEC-5: the CDN plotly.js dependency is now documented at its call site
# ---------------------------------------------------------------------------


def test_sec5_plotly_cdn_dependency_documented():
    """SEC-5: the offline-availability tradeoff of include_plotlyjs='cdn' must be documented inline."""
    import mlframe.reporting.renderers.plotly as mod

    src = inspect.getsource(mod)
    assert "air-gapped" in src or "no outbound internet access" in src
    assert 'include_plotlyjs="cdn"' in src
