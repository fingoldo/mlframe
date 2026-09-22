"""Choosing the winning ensemble flavour on val must be announced, as choosing it on test already is.

Val is the early-stopping surface of every member, and it is the DEFAULT selection surface because OOF is only stamped
when oof_n_splits >= 2. The run log said nothing, while compare_ensembles raises a UserWarning for the same situation.
"""

from __future__ import annotations

import logging
import types

from mlframe.training.core._ensemble_chooser import _choose_ensemble_flavour


def _flavour(val_auc: float):
    return (types.SimpleNamespace(metrics={"val": {"roc_auc": val_auc}}),)


def test_a_val_based_pick_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="mlframe.training.core._ensemble_chooser"):
        winner = _choose_ensemble_flavour({"arithm": _flavour(0.71), "geo": _flavour(0.74)})
    assert winner == "geo"
    assert "via val." in caplog.text and "oof_n_splits" in caplog.text
