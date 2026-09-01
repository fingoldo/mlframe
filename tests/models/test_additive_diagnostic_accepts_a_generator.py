"""`cv_splits` was documented as an Iterable and consumed as one on every call.

`_cv_score` is invoked at least twice -- once for the full model, once for the additive one -- and again twice
per feature under `per_feature_report`. Handed the natural argument, `KFold(...).split(X)`, the generator was
exhausted by the first call; every later call iterated nothing, `np.mean([])` returned NaN with only a
suppressible RuntimeWarning, and the diagnostic's recommendation flipped from True to False with no error.

The discriminating assertion is the comparison between a generator and a list: pre-fix they disagree.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import KFold

from mlframe.models.additive_interaction_diagnostic import additive_interaction_diagnostic


def _interacting(n: int = 400, seed: int = 0):
    """A target driven by an XOR-ish product, so an additive-only model genuinely scores worse."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    y = a * b + rng.normal(0, 0.1, n)
    return np.column_stack([a, b]), y


def _r2(y_true, y_pred):
    """Higher-is-better metric, matching the function's documented contract."""
    y_true = np.asarray(y_true, dtype=float)
    resid = float(np.sum((y_true - np.asarray(y_pred, dtype=float)) ** 2))
    total = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - resid / total if total > 0 else 0.0


class TestAOneShotIteratorIsAcceptedAsDocumented:
    """The parameter says Iterable; a generator is the most natural Iterable to hand it."""

    def test_a_generator_gives_the_same_answer_as_a_list(self):
        """Pre-fix the generator arm returned NaN for every quantity after the first."""
        X, y = _interacting()
        from_gen = additive_interaction_diagnostic(X, y, KFold(3, shuffle=True, random_state=0).split(X), _r2, objective="regression")
        from_list = additive_interaction_diagnostic(X, y, list(KFold(3, shuffle=True, random_state=0).split(X)), _r2, objective="regression")
        assert np.isfinite(from_gen["additive_model_cv_score"]), "the additive score came back non-finite"
        assert from_gen["additive_model_cv_score"] == pytest.approx(from_list["additive_model_cv_score"])
        assert from_gen["recommend_interaction_engineering"] == from_list["recommend_interaction_engineering"]

    def test_every_reported_quantity_is_finite(self):
        """`np.mean([])` is NaN, and a NaN ratio silently reads as 'no interaction signal'."""
        X, y = _interacting()
        out = additive_interaction_diagnostic(X, y, KFold(3, shuffle=True, random_state=0).split(X), _r2, objective="regression")
        for key in ("full_model_cv_score", "additive_model_cv_score", "additive_signal_ratio"):
            assert np.isfinite(out[key]), f"{key} came back {out[key]!r}"

    def test_the_per_feature_table_survives_the_extra_passes(self):
        """`per_feature_report` calls the scorer twice more per feature -- the deepest reuse of the iterator."""
        X, y = _interacting()
        out = additive_interaction_diagnostic(X, y, KFold(3, shuffle=True, random_state=0).split(X), _r2, objective="regression", per_feature_report=True)
        per_feature = out.get("per_feature_interaction_report")
        assert per_feature, "per_feature_report=True must produce a table"
        rows = per_feature.values() if isinstance(per_feature, dict) else per_feature
        numbers = [v for row in rows for v in (row.values() if isinstance(row, dict) else [row]) if isinstance(v, float)]
        assert numbers, "the per-feature table carried no numbers to check"
        assert all(np.isfinite(v) for v in numbers), f"non-finite entries in the per-feature table: {per_feature}"


class TestAnEmptyFoldSetIsLoud:
    """Silence is what let the exhausted-generator bug read as a legitimate 'no signal' answer."""

    def test_no_folds_raises(self):
        """`np.mean([])` returning NaN is the mechanism; refusing the input is the fix."""
        X, y = _interacting()
        with pytest.raises(ValueError, match="cv_splits is empty"):
            additive_interaction_diagnostic(X, y, iter([]), _r2, objective="regression")
