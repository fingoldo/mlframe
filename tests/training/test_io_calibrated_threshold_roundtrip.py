"""Save/load round-trip fidelity for post-hoc calibrated models + enum-keyed decision thresholds.

Exercises the REAL ``save_mlframe_model`` / ``load_mlframe_model(safe=True)`` path (zstd + ``_SafeUnpickler``):
a fitted binary / multiclass classifier wrapped in a post-hoc isotonic calibrator must predict BIT-IDENTICALLY
after a reload, the ``TargetTypes``-keyed ``decision_thresholds`` entry must survive (StrEnum f-string keys
re-match the slug-map enum at read time), the reloaded enum must keep its identity, and a non-ASCII feature
name + a full-precision float in metadata must survive verbatim. Pre-fix-proof: if any persisted-state element
(calibrator, class order, threshold value, target_type enum, unicode key) were dropped or coerced on the
round-trip, the corresponding assert below fails.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression

from mlframe.training.io import (
    save_mlframe_model,
    load_mlframe_model,
    _load_model_cache_clear,
)
from mlframe.training._calibration_models import (
    _PostHocCalibratedModel,
    _PostHocMultiCalibratedModel,
    _PerClassIsotonicCalibrator,
)
from mlframe.training._configs_base import TargetTypes
from mlframe.training.core._setup_helpers import get_decision_threshold


@pytest.fixture
def _xy():
    rng = np.random.RandomState(0)
    X = rng.randn(200, 5)
    y = (X[:, 0] + 0.5 * rng.randn(200) > 0).astype(int)
    return X, y


def test_binary_calibrated_threshold_roundtrip_bit_identical(_xy, tmp_path):
    X, y = _xy
    base = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    raw = base.predict_proba(X)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw[:, 1], y)
    wrapped = _PostHocCalibratedModel(base, iso)
    pre_proba = wrapped.predict_proba(X)
    pre_pred = wrapped.predict(X)

    tt, tname = TargetTypes.BINARY_CLASSIFICATION, "tgt"
    meta = {
        # f-string over a StrEnum keys on the VALUE, matching the read-side slug-map enum.
        "decision_thresholds": {f"{tt}|{tname}": 0.37},
        "slug_to_original_target_type": {"binary-classification": tt},
        "feature_name_unicode": "признак_é",
        "float_precise": 0.1234567890123456789,
    }
    bundle = SimpleNamespace(model=wrapped, metadata=meta)

    fp = os.path.join(tmp_path, "m.dump")
    assert save_mlframe_model(bundle, fp, verbose=0) is True
    _load_model_cache_clear()
    loaded = load_mlframe_model(fp, safe=True)
    assert loaded is not None, "safe load returned None -- _SafeUnpickler blocked a needed class"

    assert isinstance(loaded.model, _PostHocCalibratedModel)
    assert np.array_equal(pre_proba, loaded.model.predict_proba(X))
    assert np.array_equal(pre_pred, loaded.model.predict(X))

    lm = loaded.metadata
    _tt = lm["slug_to_original_target_type"]["binary-classification"]
    assert _tt is TargetTypes.BINARY_CLASSIFICATION  # enum identity, not a bare string
    thr = get_decision_threshold(lm, f"{_tt}|{tname}", 0.5)
    assert thr == 0.37, "tuned decision threshold lost / key mismatch on round-trip"
    assert lm["feature_name_unicode"] == "признак_é"
    assert lm["float_precise"] == 0.1234567890123456789


def test_multiclass_perclass_calibrator_roundtrip_bit_identical(_xy, tmp_path):
    X, _ = _xy
    y3 = (X[:, 0] > 0).astype(int) + (X[:, 1] > 0).astype(int)  # labels 0,1,2
    base = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y3)
    raw = base.predict_proba(X)
    cal = _PerClassIsotonicCalibrator.fit(raw, y3, TargetTypes.MULTICLASS_CLASSIFICATION, classes=base.classes_)
    wrapped = _PostHocMultiCalibratedModel(base, cal, TargetTypes.MULTICLASS_CLASSIFICATION, classes_=base.classes_)
    pre_proba = wrapped.predict_proba(X)
    pre_pred = wrapped.predict(X)

    fp = os.path.join(tmp_path, "m3.dump")
    assert save_mlframe_model(SimpleNamespace(model=wrapped), fp, verbose=0) is True
    _load_model_cache_clear()
    loaded = load_mlframe_model(fp, safe=True)
    assert loaded is not None

    assert np.array_equal(pre_proba, loaded.model.predict_proba(X))
    assert np.array_equal(pre_pred, loaded.model.predict(X))
    assert loaded.model._target_type is TargetTypes.MULTICLASS_CLASSIFICATION
    assert np.array_equal(loaded.model._classes, base.classes_)
