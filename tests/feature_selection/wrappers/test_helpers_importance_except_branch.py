"""Regression tests for three hot-path hygiene fixes in _helpers_importance.

* ``get_feature_importances`` must return cleanly (no NameError) when importances are non-numeric and
  the NaN-detection ``np.asarray(..., dtype=float)`` raises ``(TypeError, ValueError)``. Pre-fix that
  except branch left ``res_arr`` unassigned -- a latent NameError -- and silently skipped the skip-log.
  The fix initialises ``res_arr = None`` and logs a debug skip.
* ``select_appropriate_feature_importances`` must NOT ``print`` from the freshest-FI path (a stray
  ``print`` in a hot library path); it now logs at debug level.
* ``get_actual_features_ranking`` must return ``[]``, not crash, when ``feature_importances`` is empty
  (no CV fold ever produced FI, e.g. every fold was skipped by ``min_train_size``) -- pre-fix this
  built a zero-weight ``Leaderboard`` and raised ``ValueError``.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.wrappers._helpers_importance import (
    get_actual_features_ranking,
    get_feature_importances,
    select_appropriate_feature_importances,
)
from mlframe.feature_selection.wrappers._enums import VotesAggregation


def test_non_numeric_importances_do_not_raise():
    """Non numeric importances do not raise."""
    features = ["f0", "f1", "f2"]

    def non_numeric_getter(model, data, reference_data, target):
        # Strings cannot be coerced via np.asarray(..., dtype=float) -> hits the except branch.
        """Non numeric getter."""
        return ["a", "b", "c"]

    out = get_feature_importances(
        model=object(),
        current_features=features,
        importance_getter=non_numeric_getter,
        data=None,
    )
    assert set(out.keys()) == set(features)
    assert list(out.values()) == ["a", "b", "c"]


def test_freshest_fi_path_does_not_print(capsys):
    # Hit the use_one_freshest_fi_run branch that pre-fix called print(...).
    """Freshest fi path does not print."""
    feature_importances = {"run_a": np.array([0.5, 0.3, 0.2])}  # len 3 == n_original_features
    select_appropriate_feature_importances(
        feature_importances=feature_importances,
        nfeatures=1,
        n_original_features=3,
        use_all_fi_runs=False,
        use_one_freshest_fi_run=True,
    )
    captured = capsys.readouterr()
    assert captured.out == "", f"freshest-FI path leaked a print to stdout: {captured.out!r}"


def test_empty_feature_importances_returns_empty_ranking_not_crash():
    """No FI runs at all (e.g. every CV fold was skipped) must return [], not raise building a
    zero-weight Leaderboard."""
    ranks = get_actual_features_ranking(
        feature_importances={},
        votes_aggregation_method=VotesAggregation.Borda,
    )
    assert ranks == []
