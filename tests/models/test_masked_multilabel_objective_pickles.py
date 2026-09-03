"""The masked-multilabel objective was a nested closure, so any model carrying it could not be pickled.

`xgb.XGBRegressor(objective=masked_multilabel_logloss_objective())` raised
`PicklingError: Can't pickle local object ...<locals>.objective` on both `pickle.dumps` and `joblib.dump`. The
bundle writer falls back to dill, which serialises bytecode -- so the failure is invisible at save time and the
saved bundle is not guaranteed to load under a different interpreter or xgboost version, which is exactly the
fragility the safe-load allowlist exists to prevent.

It is now a module-level class with `__slots__` and explicit state hooks. The numerics are unchanged, and these
tests check that rather than assume it.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from mlframe.models.masked_multilabel_objective import masked_multilabel_logloss_objective


class _FakeDMatrix:
    """Minimal stand-in exposing the two accessors the objective uses."""

    def __init__(self, labels, weights=None):
        self._labels = labels
        self._weights = weights

    def get_label(self):
        """Sentinel-masked labels, as XGBoost would hand them back."""
        return self._labels

    def get_weight(self):
        """Per-cell weights, or None when the caller supplied none."""
        return self._weights


LABELS = np.array([1.0, 0.0, 2.0, 1.0])  # 2.0 is the default don't-care sentinel
MARGIN = np.array([0.5, -0.5, 3.0, 0.1])


class TestItPickles:
    """The defect, stated directly."""

    def test_the_objective_pickles(self):
        """A local function cannot; a module-level instance can."""
        assert pickle.loads(pickle.dumps(masked_multilabel_logloss_objective())) is not None  # nosec B301 - round-tripping our own object is the assertion

    def test_it_survives_joblib(self):
        """joblib is what the bundle writer actually uses."""
        import io

        import joblib

        buf = io.BytesIO()
        joblib.dump(masked_multilabel_logloss_objective(sentinel=-1.0, use_sample_weight=True), buf)
        buf.seek(0)
        restored = joblib.load(buf)  # nosec B301 - our own object, written two lines up
        assert restored.sentinel == -1.0 and restored.use_sample_weight is True

    def test_the_configuration_survives_the_round_trip(self):
        """`__slots__` leaves no `__dict__`, so the state hooks are load-bearing rather than decorative."""
        obj = masked_multilabel_logloss_objective(sentinel=7.0, use_sample_weight=True)
        back = pickle.loads(pickle.dumps(obj))  # nosec B301 - our own object
        assert (back.sentinel, back.use_sample_weight) == (7.0, True)

    def test_it_is_not_a_local_function(self):
        """The shape that caused it; a repeat would otherwise be silent again."""
        assert "<locals>" not in type(masked_multilabel_logloss_objective()).__qualname__


class TestTheNumericsAreUnchanged:
    """A refactor of the carrier must not move the gradients."""

    def test_a_round_tripped_objective_computes_the_same_values(self):
        """Picklable is not enough; it has to be the same objective afterwards."""
        obj = masked_multilabel_logloss_objective()
        back = pickle.loads(pickle.dumps(obj))  # nosec B301 - our own object
        g1, h1 = obj(MARGIN, _FakeDMatrix(LABELS))
        g2, h2 = back(MARGIN, _FakeDMatrix(LABELS))
        assert np.array_equal(g1, g2) and np.array_equal(h1, h2)

    def test_sentinel_cells_contribute_nothing(self):
        """The objective's core contract: a don't-care cell gets zero grad AND zero hessian."""
        grad, hess = masked_multilabel_logloss_objective()(MARGIN, _FakeDMatrix(LABELS))
        assert grad[2] == 0.0 and hess[2] == 0.0

    def test_weighting_scales_the_unmasked_cells(self):
        """XGBoost does not apply DMatrix weights to a custom objective, so the multiply happens here."""
        plain = masked_multilabel_logloss_objective()
        weighted = masked_multilabel_logloss_objective(use_sample_weight=True)
        g_plain, _ = plain(MARGIN, _FakeDMatrix(LABELS))
        g_weighted, _ = weighted(MARGIN, _FakeDMatrix(LABELS, np.array([1.0, 2.0, 1.0, 3.0])))
        assert g_weighted[3] == pytest.approx(g_plain[3] * 3.0)

    def test_a_missing_weight_vector_is_refused(self):
        """Silently falling back to uniform would defeat the point of asking for weighting."""
        with pytest.raises(ValueError, match="requires dtrain to carry a weight vector"):
            masked_multilabel_logloss_objective(use_sample_weight=True)(MARGIN, _FakeDMatrix(LABELS, None))

    def test_xgboost_can_read_a_name_off_it(self):
        """xgboost reads `__name__` from a custom objective; a class needs it declared."""
        assert masked_multilabel_logloss_objective().__name__ == "masked_multilabel_logloss_objective"
