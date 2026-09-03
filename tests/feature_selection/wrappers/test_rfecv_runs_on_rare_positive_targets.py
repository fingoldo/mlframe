"""RFECV refused to run whenever the minority class was smaller than the requested fold count.

A rare-positive target is the normal case for a feature selector, not an error: at a 1% positive rate on 1000
rows there are about ten positives, and after the outer train/val/test split the inner CV sees fewer than the
requested five per fold. The wrapper raised rather than adapting, so RFECV simply did not work on imbalanced
targets.

The fuzz harness had papered over exactly this with a canonicalisation rule -- `if target_type ==
"binary_classification" and imbalance_ratio != "balanced": return None` -- so no combo ever exercised RFECV
against a rare-positive target and the suite stayed green indefinitely. The harness's own contract forbids that
shape: "If you catch yourself writing a canon rule whose justification references a CrashID ... STOP. Find the
prod fix." The rule is deleted here alongside the fix.
"""

from __future__ import annotations

import logging
import pathlib

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier


def _frame(n_pos, n, seed=0):
    """A binary target with EXACTLY `n_pos` positives, so the fold arithmetic under test is deterministic."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=int)
    y[rng.choice(n, size=n_pos, replace=False)] = 1
    X = pd.DataFrame(rng.normal(0, 1, (n, 5)), columns=[f"f{i}" for i in range(5)])
    return X, y


def _fit(X, y, cv=5):
    """One RFECV fit with a cheap forest."""
    from mlframe.feature_selection.wrappers.rfecv import RFECV

    sel = RFECV(estimator=RandomForestClassifier(n_estimators=5, random_state=0), cv=cv, verbose=0)
    sel.fit(X, y)
    return sel


class TestItRunsWhereItUsedToRefuse:
    """The defect, at the sizes the fuzz harness was excluding."""

    def test_a_one_percent_target_at_the_small_tier_fits(self):
        """n=1000 at 1% is roughly ten positives -- the exact combination the canon rule disabled."""
        X, y = _frame(10, 1000)
        assert _fit(X, y).support_.any()

    def test_the_fold_count_drops_to_what_the_minority_supports(self):
        """Two positives cannot fill five folds; the selector reduces rather than refusing."""
        X, y = _frame(2, 2000)
        assert _fit(X, y).cv == 2

    def test_the_reduction_is_announced(self):
        """Silently changing the CV depth would be its own surprise."""
        X, y = _frame(3, 2000)
        with _capture(logging.WARNING) as records:
            _fit(X, y)
        assert any("reducing to cv" in r for r in records), records

    def test_a_single_positive_still_raises(self):
        """Below two there is no stratified split at all, so refusing is the honest answer."""
        X, y = _frame(1, 300)
        with pytest.raises(ValueError, match="cannot be split at all"):
            _fit(X, y)

    def test_a_balanced_target_is_unaffected(self):
        """The common path must not change."""
        rng = np.random.default_rng(1)
        n = 400
        y = rng.integers(0, 2, n)
        X = pd.DataFrame(rng.normal(0, 1, (n, 5)), columns=[f"f{i}" for i in range(5)])
        assert _fit(X, y).cv == 5


def test_the_fuzz_canon_rule_is_gone():
    """The rule hid the defect; leaving it would keep RFECV untested against every imbalanced target."""
    src = (pathlib.Path(__file__).resolve().parents[2] / "training" / "_fuzz_combo" / "combo.py").read_text(encoding="utf-8")
    assert 'self.imbalance_ratio != "balanced"' not in src, "the RFECV imbalance canon rule is back"


class _capture:
    """Collect log messages at or above `level` for the duration of the block."""

    def __init__(self, level):
        self.level = level
        self.records: list = []

    def __enter__(self):
        """Attach a collecting handler to the root logger."""
        outer = self

        class _H(logging.Handler):
            def emit(self, record):
                """Record the formatted message."""
                outer.records.append(record.getMessage())

        self.handler = _H(self.level)
        logging.getLogger().addHandler(self.handler)
        self.prev = logging.getLogger().level
        logging.getLogger().setLevel(self.level)
        return self.records

    def __exit__(self, *exc):
        """Detach the handler and restore the level."""
        logging.getLogger().removeHandler(self.handler)
        logging.getLogger().setLevel(self.prev)
        return False
