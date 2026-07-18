"""Wave-19 P1 sensor: lgb_shim / xgb_shim stamp + verify booster library
version on pickle round-trip.

Pre-fix the shim ``__setstate__`` correctly stripped the C++ pointer
cache (the original P0 the shim was written to solve) but had no
knowledge of which lightgbm / xgboost version produced the rest of
the pickled __dict__. The booster JSON inside ``__dict__`` is
library-version-sensitive across minor versions; loads under a
different installed lightgbm / xgboost could silently mis-restore
booster internals and crash deep in ``predict()``.

Post-fix: ``__getstate__`` stamps ``_saved_lgb_version`` /
``_saved_xgb_version``; ``__setstate__`` compares against the live
``lightgbm.__version__`` / ``xgboost.__version__`` and WARN-logs
drift. Soft contract -- booster libs are typically forward-compatible
for minor versions, so we don't raise; operators just see the skew
before chasing weird predict crashes.
"""

from __future__ import annotations

import logging
import pickle  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data

import pytest


def test_lgb_shim_stamps_lgb_version_on_pickle():
    """Lgb shim stamps lgb version on pickle."""
    pytest.importorskip("lightgbm")
    from mlframe.training.lgb_shim import LGBMClassifierWithDatasetReuse

    inst = LGBMClassifierWithDatasetReuse(n_estimators=2, verbosity=-1)
    state = inst.__getstate__()
    assert (
        "_saved_lgb_version" in state
    ), "Wave 19 P1 regression: lgb_shim.__getstate__ no longer stamps _saved_lgb_version. Booster JSON inside __dict__ is library-version-sensitive."
    assert state["_saved_lgb_version"] not in (None, ""), "stamp must be non-empty"


def test_lgb_shim_warns_on_version_drift(caplog):
    """Lgb shim warns on version drift."""
    pytest.importorskip("lightgbm")
    from mlframe.training.lgb_shim import LGBMClassifierWithDatasetReuse

    inst = LGBMClassifierWithDatasetReuse(n_estimators=2, verbosity=-1)
    payload = pickle.dumps(inst)
    loaded = pickle.loads(payload)  # nosec B301 -- round-trip of a locally-created, trusted object
    # Manually tamper the saved-version to simulate cross-version load.
    loaded._saved_lgb_version = "0.001-FAKE-OLD"
    with caplog.at_level(logging.WARNING, logger="mlframe.training.lgb_shim"):
        # Trigger __setstate__ again with the tampered state to force the
        # WARN. Cleaner approach: deepcopy via pickle round-trip which
        # invokes __setstate__.
        pickle.loads(pickle.dumps(loaded))  # nosec B301 -- round-trip of a locally-created, trusted object
        # Some pickle paths may not re-invoke __setstate__ on a
        # non-mutating round-trip; tolerate either: assert the
        # loaded.tampered version is preserved (sentinel that the field
        # round-tripped), AND verify a WARN fired by manually invoking
        # __setstate__ ourselves -- must stay inside the caplog context
        # or the manual WARN is never captured.
        fresh = LGBMClassifierWithDatasetReuse(n_estimators=2, verbosity=-1)
        fresh.__setstate__({"_saved_lgb_version": "0.001-FAKE-OLD"})
    drift_warns = [r for r in caplog.records if "lightgbm version drift" in r.message]
    assert drift_warns, f"expected lightgbm version drift WARN; got: {[r.message for r in caplog.records]}"


def test_xgb_shim_stamps_xgb_version_on_pickle():
    """Xgb shim stamps xgb version on pickle."""
    pytest.importorskip("xgboost")
    from mlframe.training.xgb_shim import XGBClassifierWithDMatrixReuse

    inst = XGBClassifierWithDMatrixReuse(n_estimators=2)
    state = inst.__getstate__()
    assert "_saved_xgb_version" in state, "Wave 19 P1 regression: xgb_shim.__getstate__ no longer stamps _saved_xgb_version."
    assert state["_saved_xgb_version"] not in (None, ""), "stamp must be non-empty"


def test_xgb_shim_warns_on_version_drift(caplog):
    """Xgb shim warns on version drift."""
    pytest.importorskip("xgboost")
    from mlframe.training.xgb_shim import XGBClassifierWithDMatrixReuse

    fresh = XGBClassifierWithDMatrixReuse(n_estimators=2)
    with caplog.at_level(logging.WARNING, logger="mlframe.training.xgb_shim"):
        fresh.__setstate__({"_saved_xgb_version": "0.001-FAKE-OLD"})
    drift_warns = [r for r in caplog.records if "xgboost version drift" in r.message]
    assert drift_warns, f"expected xgboost version drift WARN; got: {[r.message for r in caplog.records]}"


def test_legacy_pickle_without_stamp_loads_silently(caplog):
    """Pre-fix pickles had no _saved_lgb_version / _saved_xgb_version
    attribute. The new __setstate__ must NOT crash and must NOT warn when
    the stamp is absent (back-compat)."""
    pytest.importorskip("lightgbm")
    from mlframe.training.lgb_shim import LGBMClassifierWithDatasetReuse

    fresh = LGBMClassifierWithDatasetReuse(n_estimators=2, verbosity=-1)
    with caplog.at_level(logging.WARNING, logger="mlframe.training.lgb_shim"):
        # State dict without _saved_lgb_version simulates a legacy pickle.
        fresh.__setstate__({})
    drift_warns = [r for r in caplog.records if "version drift" in r.message]
    assert drift_warns == [], f"legacy pickles (no version stamp) must NOT WARN; got: {[r.message for r in caplog.records]}"
