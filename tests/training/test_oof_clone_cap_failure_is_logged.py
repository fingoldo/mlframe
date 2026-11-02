"""A failed early-stopping cap on the OOF clone is logged and recorded, not swallowed (tracker 2026-11-02 wave 9, EXC9-2)."""

import logging

from mlframe.training._oof_fold_policy import cap_early_stopping_clone


class _Refuses:
    """Estimator that has early stopping on and refuses to change it."""

    early_stopping_rounds = 10

    def set_params(self, **kw):
        """Always refuse."""
        raise ValueError("no such parameter")


def test_set_params_failure_warns_and_is_recorded(caplog):
    """The diag dict carries the error and a warning names the estimator."""
    diag = {}
    with caplog.at_level(logging.WARNING):
        cap_early_stopping_clone(_Refuses(), object(), diag)
    assert "no such parameter" in diag["round_budget_error"]
    assert any("_Refuses" in r.getMessage() for r in caplog.records)
