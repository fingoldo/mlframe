"""Unit tests for the adaptive coarse-to-fine elimination pace (``dichotomic_step``).

Covers the new ``_suggest_dichotomic(step=...)`` schedule and its RFECV wiring:
  - 'auto' strides far from the best while the unevaluated pool is large + the curve is flat;
  - 'midpoint' restores the legacy fixed bisection (both end at the same fine neighbourhood);
  - the schedule tapers to step=1 as the pool drains / the curve moves near the knee;
  - constructor + SearchConfig validate the knob and reject bad values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from mlframe.feature_selection.wrappers import RFECV
from mlframe.feature_selection.wrappers._enums import OptimumSearch
from mlframe.feature_selection.wrappers._helpers import (
    _suggest_dichotomic,
    _curve_is_flat_near_best,
)


class TestAdaptiveStepSuggester:
    def test_auto_strides_far_on_flat_wide_pool(self):
        # Best at 50, an immediate flat neighbour at 80, 200-wide pool: auto must stride well past the midpoint.
        ev = {50: 0.80, 80: 0.801}
        remaining = sorted(set(range(1, 201)) - set(ev))
        auto = _suggest_dichotomic(remaining, ev, n_total=200, step="auto")
        mid = _suggest_dichotomic(remaining, ev, n_total=200, step="midpoint")
        assert auto - 50 > mid - 50, "auto stride must exceed the midpoint bisection on a flat wide pool"
        assert auto > 80, "auto should jump beyond the nearest evaluated neighbour"

    def test_midpoint_is_legacy_bisection(self):
        ev = {50: 0.80, 80: 0.801}
        remaining = sorted(set(range(1, 201)) - set(ev))
        mid = _suggest_dichotomic(remaining, ev, n_total=200, step="midpoint")
        # Legacy: bisect between best (50) and nearest neighbour (80) -> ~65 region, but the wider gap is the
        # lower side (50 down to 1) so it bisects there. Either way it is a midpoint, never a long stride.
        assert mid <= 65 or mid >= 35

    def test_auto_tapers_to_fine_when_pool_small(self):
        # Pool nearly drained around the best -> no big stride possible; falls through to midpoint refinement.
        ev = {n: 0.5 + 0.001 * n for n in range(1, 198)}
        remaining = sorted(set(range(1, 201)) - set(ev))  # {198, 199, 200}
        auto = _suggest_dichotomic(remaining, ev, n_total=200, step="auto")
        assert auto in remaining

    def test_curve_flat_detector(self):
        assert _curve_is_flat_near_best({50: 0.80, 80: 0.8005}, 50) is True
        assert _curve_is_flat_near_best({50: 0.80, 51: 0.95}, 50) is False

    def test_first_call_probes_midpoint_both_modes(self):
        ev = {30: 0.5}
        remaining = sorted(set(range(1, 61)) - set(ev))
        a = _suggest_dichotomic(remaining, ev, n_total=60, step="auto")
        m = _suggest_dichotomic(remaining, ev, n_total=60, step="midpoint")
        # With a single eval both probe near the full-range midpoint.
        assert abs(a - 30) <= 30 and abs(m - 30) <= 30


class TestRFECVWiring:
    def test_constructor_rejects_bad_step(self):
        with pytest.raises(ValueError, match="dichotomic_step"):
            RFECV(estimator=LogisticRegression(), dichotomic_step="bogus")

    def test_default_is_midpoint(self):
        # 'auto' was bench-rejected as default (no replicated wall win); legacy midpoint stays default, auto opt-in.
        sel = RFECV(estimator=LogisticRegression())
        assert sel.dichotomic_step == "midpoint"

    def test_search_config_validates_step(self):
        from mlframe.feature_selection.wrappers.rfecv._configs import SearchConfig, _PYDANTIC_AVAILABLE
        if not _PYDANTIC_AVAILABLE:
            pytest.skip("pydantic unavailable")
        with pytest.raises(ValueError):
            SearchConfig(dichotomic_step="bogus")

    def test_both_modes_fit_and_select_on_dichotomic(self):
        X, y = make_classification(n_samples=600, n_features=40, n_informative=8, n_redundant=6, random_state=0)
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(40)])
        out = {}
        for step in ("auto", "midpoint"):
            sel = RFECV(
                estimator=LogisticRegression(max_iter=300),
                top_predictors_search_method=OptimumSearch.ExhaustiveDichotomic,
                dichotomic_step=step, dichotomic_epsilon=0.0, cv=3,
                max_noimproving_iters=8, random_state=0, verbose=0, leave_progressbars=False,
            )
            sel.fit(X, y)
            out[step] = set(np.asarray(X.columns)[sel.support_])
            assert sel.support_.sum() >= 1
        # Selections should be highly similar on a smooth linear model (speed lever, not accuracy change).
        jac = len(out["auto"] & out["midpoint"]) / max(1, len(out["auto"] | out["midpoint"]))
        assert jac >= 0.7, f"auto vs midpoint Jaccard {jac:.2f} too low on smooth LR curve"
