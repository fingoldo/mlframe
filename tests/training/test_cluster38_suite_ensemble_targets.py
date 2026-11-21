"""The suite-wide ensemble is not built across different targets (audit 2026-11-20, finding 38-F1)."""

import numpy as np

from mlframe.training.core._predict_main_from_models import _apply_suite_ensemble


def test_no_suite_ensemble_is_written_for_two_targets():
    """Members of two targets predict different quantities; their averaged probabilities must not be published."""
    per_target = {("binary", "a"): np.array([[0.2, 0.8]]), ("binary", "b"): np.array([[0.6, 0.4]])}
    results: dict = {}
    _apply_suite_ensemble([np.array([[0.2, 0.8]]), np.array([[0.6, 0.4]])], None, {}, per_target, results)
    assert "ensemble_probabilities" not in results
