"""The flavour leaderboard must rank error columns the right way round and must not include the test split.

ENS-07: votenrank ranks every column descending, so on brier_loss / ice / RMSE columns the WORST flavour was rank 1.
ENS-14: test.* columns sat beside oof.*/val.* in a persisted table named a ranking of flavours.
"""

from __future__ import annotations

import types

from mlframe.models.ensembling import _build_votenrank_leaderboard_from_results


def _res(**flavours):
    return {name: types.SimpleNamespace(metrics=m) for name, m in flavours.items()}


def test_the_better_calibrated_flavour_wins_on_an_error_metric():
    res = _res(
        good={"val": {"brier_loss": 0.10, "roc_auc": 0.80}},
        bad={"val": {"brier_loss": 0.30, "roc_auc": 0.80}},
    )
    board = _build_votenrank_leaderboard_from_results(res, is_regression=False)
    assert board is not None
    ranks = board.lb.ranks
    brier_col = next(c for c in ranks.columns if c.endswith("brier_loss"))
    assert ranks.loc["good", brier_col] < ranks.loc["bad", brier_col], "lower Brier must rank first"


def test_test_split_columns_are_excluded():
    res = _res(a={"val": {"roc_auc": 0.7}, "test": {"roc_auc": 0.9}}, b={"val": {"roc_auc": 0.6}, "test": {"roc_auc": 0.95}})
    board = _build_votenrank_leaderboard_from_results(res, is_regression=False)
    assert not any(str(c).startswith("test.") for c in board.table.columns)
