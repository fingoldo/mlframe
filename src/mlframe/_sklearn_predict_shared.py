"""Shared ``predict`` body for sklearn-style classifiers whose ``predict`` is just an argmax over
``predict_proba``, mapped back through ``classes_``: independently duplicated across estimator
classes, consolidated here so a fix can't silently drift out of sync across copies.
"""
from __future__ import annotations

import numpy as np


def predict_from_proba(classes_: np.ndarray, proba: np.ndarray) -> np.ndarray:
    """Argmax class label over ``proba``, mapped back through ``classes_``."""
    return np.asarray(classes_[np.argmax(proba, axis=1)])


def argmax_predict(self, X) -> np.ndarray:
    """Shared ``.predict`` for classifiers whose prediction is just an argmax over their own
    ``.predict_proba(X)``, mapped back through ``self.classes_``. Bound as a class attribute
    (``predict = argmax_predict``); Python's descriptor protocol resolves it to a normal bound
    method at call time."""
    return predict_from_proba(self.classes_, self.predict_proba(X))
