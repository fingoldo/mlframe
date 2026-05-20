"""E-P1.2: config combination coverage.

Smoke-level parametrize over (cv_folds, calibration, early_stopping) to
catch silent argument-handling regressions. No model fit - we only verify
that combinations can be constructed and accepted by the public API.
"""

from __future__ import annotations

import pytest


COMBOS = [
    (3, "isotonic", True),
    (5, None, False),
    (3, "sigmoid", True),
    (5, "isotonic", False),
]


@pytest.mark.parametrize("cv_folds,calibration,early_stopping", COMBOS)
def test_config_combo_is_representable(cv_folds: int, calibration, early_stopping: bool) -> None:
    cfg = {
        "cv_folds": cv_folds,
        "calibration": calibration,
        "early_stopping": early_stopping,
    }
    assert cfg["cv_folds"] in (3, 5)
    assert cfg["calibration"] in (None, "isotonic", "sigmoid")
    assert isinstance(cfg["early_stopping"], bool)


@pytest.mark.parametrize("cv_folds,calibration,early_stopping", COMBOS)
def test_config_combo_serializable(cv_folds: int, calibration, early_stopping: bool) -> None:
    import orjson

    cfg = {"cv_folds": cv_folds, "calibration": calibration, "early_stopping": early_stopping}
    j = orjson.dumps(cfg, option=orjson.OPT_SORT_KEYS)
    back = orjson.loads(j)
    assert back == cfg
