"""Wave 11a monolith-split sensor for ``mlframe.models.ensembling.score``.

Carve pattern: input-validation prelude extracted to sibling. score_ensemble dispatcher stays in parent. Identity / behavioural equivalence preserved for the single-member / no-members short-circuit.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture(scope="module")
def parent_module():
    """Parent module."""
    from mlframe.models.ensembling import score as _ensembling_score

    return _ensembling_score


@pytest.fixture(scope="module")
def sibling_validate():
    """Sibling validate."""
    from mlframe.models.ensembling import score_validate as _ensembling_score_validate

    return _ensembling_score_validate


def test_validate_identity(parent_module, sibling_validate):
    """Validate identity."""
    assert parent_module._validate_score_ensemble_inputs is sibling_validate._validate_score_ensemble_inputs


def test_facade_loc_budget(parent_module):
    """Facade loc budget."""
    path = Path(parent_module.__file__)
    n_lines = len(path.read_text(encoding="utf-8").splitlines())
    assert n_lines < 1000, f"facade is {n_lines} LOC, expected < 1000"


def test_single_member_returns_sentinel(parent_module):
    """The carved single-member short-circuit must return the exact sentinel dict shape that callers (finalize, suite metadata) depend on."""
    m = MagicMock()
    m.oof_probs = None
    m.val_probs = np.array([[0.5, 0.5]])
    m.test_probs = None
    m.train_probs = None

    res = parent_module.score_ensemble(
        models_and_predictions=[m],
        ensemble_name="solo",
        verbose=False,
    )
    assert isinstance(res, dict)
    assert res.get("_reason") == "single_member"
    assert res.get("_n_members") == 1


def test_no_members_returns_sentinel(parent_module):
    """No members returns sentinel."""
    res = parent_module.score_ensemble(
        models_and_predictions=[],
        ensemble_name="empty",
        verbose=False,
    )
    assert isinstance(res, dict)
    assert res.get("_reason") == "no_members"
    assert res.get("_n_members") == 0


def test_validate_rejects_mixed_clf_reg(sibling_validate):
    """Mixed classifier/regressor members must still raise ValueError after the carve."""
    clf = MagicMock()
    clf.oof_probs = np.array([[0.5, 0.5]])
    clf.val_probs = None
    clf.test_probs = None
    clf.train_probs = None

    reg = MagicMock()
    reg.oof_probs = None
    reg.val_probs = None
    reg.test_probs = None
    reg.train_probs = None
    # Also clear any other probs MagicMock would fabricate.
    reg.oof_preds = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="uniform member types"):
        sibling_validate._validate_score_ensemble_inputs(
            level_models_and_predictions=[clf, reg],
            ensembling_methods=["mean"],
            ensure_prob_limits=True,
            max_ensembling_level=1,
            verbose=False,
        )


def test_validate_filters_rrf_on_regression(sibling_validate):
    """RRF must be silently filtered when target inference picks regression."""
    m1 = MagicMock()
    m2 = MagicMock()
    for m in (m1, m2):
        m.oof_probs = None
        m.val_probs = None
        m.test_probs = None
        m.train_probs = None

    early_res, is_regression, methods, ensure_prob = sibling_validate._validate_score_ensemble_inputs(
        level_models_and_predictions=[m1, m2],
        ensembling_methods=["mean", "rrf", "median"],
        ensure_prob_limits=True,
        max_ensembling_level=1,
        verbose=False,
    )
    assert early_res == {}
    assert is_regression is True
    assert "rrf" not in methods
    assert ensure_prob is False
