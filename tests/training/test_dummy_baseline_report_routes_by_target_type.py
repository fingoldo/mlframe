"""A quantile-regression dummy was reported through the classification path.

`_split_preds_probs` decided "2-D means probabilities" from shape alone. Quantile regression predicts K
conditional quantiles and multi-target regression K continuous targets -- both `(N, K)`. So the dummy's
quantile matrix was returned as `probs`, and `report_model_perf`, which infers the task from `probs is not
None` whenever it has no model to ask, routed the whole report into `report_probabilistic_model_perf`. It died
there indexing the matrix by a target value: `IndexError: index 3 is out of bounds for axis 1 with size 3`.

The failure was invisible: the dummy-report block is wrapped in a best-effort `except Exception` that logs a
warning and continues, so every quantile-regression suite run silently lost its pre-training floor report.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training._configs_base import TargetTypes
from mlframe.training.core._misc_helpers import _split_preds_probs

QUANTILES = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [0.5, 1.5, 2.5]])
PROBS = np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1], [0.2, 0.2, 0.6]])


class TestARegressionMatrixIsNotProbabilities:
    """The defect, per target type."""

    @pytest.mark.parametrize("tt", [TargetTypes.QUANTILE_REGRESSION, TargetTypes.MULTI_TARGET_REGRESSION, TargetTypes.REGRESSION, TargetTypes.LEARNING_TO_RANK])
    def test_no_probs_are_reported_for_a_non_classification_target(self, tt):
        """`probs is not None` is what routes the report; for these targets it must stay None."""
        _, probs = _split_preds_probs(QUANTILES, tt)
        assert probs is None, f"{tt} handed a {QUANTILES.shape} matrix downstream as class probabilities"

    def test_the_matrix_survives_as_predictions(self, ):
        """The quantile predictions themselves must reach the regression report intact, not collapsed by argmax."""
        preds, _ = _split_preds_probs(QUANTILES, TargetTypes.QUANTILE_REGRESSION)
        assert np.array_equal(preds, QUANTILES)

    def test_an_argmax_over_quantiles_is_not_taken(self):
        """The specific nonsense the old code produced: a quantile INDEX presented as a class label."""
        preds, _ = _split_preds_probs(QUANTILES, TargetTypes.QUANTILE_REGRESSION)
        assert preds.ndim == 2, "the quantile matrix was collapsed to a per-row argmax"

    def test_the_string_form_of_the_target_type_works_too(self):
        """The dummy phase passes `str(target_type)` around in places."""
        assert _split_preds_probs(QUANTILES, str(TargetTypes.QUANTILE_REGRESSION))[1] is None


class TestClassificationIsUnchanged:
    """The routing fix must not stop classification reports from getting their probabilities."""

    @pytest.mark.parametrize("tt", [TargetTypes.BINARY_CLASSIFICATION, TargetTypes.MULTICLASS_CLASSIFICATION, TargetTypes.MULTILABEL_CLASSIFICATION])
    def test_probs_still_come_through(self, tt):
        """Unchanged contract."""
        preds, probs = _split_preds_probs(PROBS, tt)
        assert probs is PROBS and np.array_equal(preds, [1, 0, 2])

    def test_a_one_dimensional_output_is_never_probabilities(self):
        """Shape still decides for 1-D, in both directions."""
        assert _split_preds_probs(np.array([1.0, 2.0]), TargetTypes.BINARY_CLASSIFICATION)[1] is None

    def test_none_stays_none(self):
        """Unchanged contract."""
        assert _split_preds_probs(None, TargetTypes.REGRESSION) == (None, None)

    def test_an_omitted_target_type_keeps_the_old_shape_only_behaviour(self):
        """Callers that only ever pass classification output must not have to change."""
        assert _split_preds_probs(PROBS)[1] is PROBS

    def test_an_unrecognised_target_type_does_not_raise(self):
        """A best-effort report path must not turn an unknown enum value into a crash."""
        assert _split_preds_probs(PROBS, "some_future_target_type")[1] is PROBS
