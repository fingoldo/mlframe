"""SU relevance normalisation reuses the caller's target frequencies instead of re-factorising the fit-constant target per candidate."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters import evaluation
from mlframe.feature_selection.filters.info_theory._class_encoding import merge_vars
from mlframe.feature_selection.filters.info_theory._state_and_dispatch import set_su_normalization


def _codes():
    """Binned factors: two candidate columns and the target in column 2."""
    rng = np.random.default_rng(0)
    data = np.column_stack([rng.integers(0, 4, 500), rng.integers(0, 3, 500), rng.integers(0, 2, 500)]).astype(np.int32)
    nbins = np.array([4, 3, 2], dtype=np.int32)
    return data, nbins


def test_supplied_target_freqs_give_the_same_value_with_one_factorisation(monkeypatch):
    """With freqs_y supplied only the candidate side is factorised, and the SU-scaled gain is bit-identical to the recompute."""
    data, nbins = _codes()
    X, y = (0,), (2,)
    _, freqs_y, _ = merge_vars(factors_data=data, vars_indices=np.asarray(y, dtype=np.int64), var_is_nominal=None, factors_nbins=nbins, dtype=np.int32)
    set_su_normalization(True)
    try:
        reference = evaluation._su_normalize_relevance(0.05, X, y, data, nbins, np.int32)
        calls = {"n": 0}
        real = evaluation.merge_vars

        def counting(*a, **kw):
            """Count target/candidate factorisations."""
            calls["n"] += 1
            return real(*a, **kw)

        monkeypatch.setattr(evaluation, "merge_vars", counting)
        reused = evaluation._su_normalize_relevance(0.05, X, y, data, nbins, np.int32, freqs_y=freqs_y)
    finally:
        set_su_normalization(False)
    assert reused == reference
    assert reference != 0.05, "fixture precondition: SU scaling must actually change the gain"
    assert calls["n"] == 1, f"the target was re-factorised although its frequencies were supplied ({calls['n']} merge_vars calls)"
