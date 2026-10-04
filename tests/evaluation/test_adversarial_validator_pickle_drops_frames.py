"""AdversarialValidator must not serialise the caller's fit frames when pickled."""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from mlframe.evaluation.adversarial_validator import AdversarialValidator


def test_pickle_roundtrip_drops_retained_frames_but_keeps_report():
    """Fit keeps references (not copies) for the follow-up methods; the pickle excludes them and the report survives."""
    pytest.importorskip("lightgbm")
    rng = np.random.default_rng(0)
    A = pd.DataFrame(rng.normal(size=(5000, 3)), columns=list("xyz"))
    B = pd.DataFrame(rng.normal(size=(5000, 3)), columns=list("xyz"))
    v = AdversarialValidator(n_splits=2).fit(A, B)
    assert v._X_train is A and v._X_test is B
    blob = pickle.dumps(v)
    with_frames = len(pickle.dumps(dict(v.__dict__)))
    assert with_frames - len(blob) > A.memory_usage().sum() + B.memory_usage().sum() - 1000
    v2 = pickle.loads(blob)  # nosec B301 - the test round-trips a pickle it just produced
    assert not hasattr(v2, "_X_train") and not hasattr(v2, "_X_test")
    pd.testing.assert_frame_equal(v2.report(), v.report())
    with pytest.raises(ValueError, match="pass X_train/X_test"):
        v2.select_validation_fold()
