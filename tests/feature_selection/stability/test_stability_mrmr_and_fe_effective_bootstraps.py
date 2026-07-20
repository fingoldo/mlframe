"""Regression tests for mrmr_audit_2026-07-20 B-14/B-15: one degenerate bootstrap subsample used to
crash the WHOLE stability sweep in both ``StabilityMRMR`` (stability.py) and ``stability_select_fe``
(_stability_fe.py), unlike their sibling ``_stability_cluster.py`` implementations (which already
exclude failed draws and divide frequencies by the effective/successful count -- see
``test_stability_cluster_effective_bootstraps.py``). Both now mirror that pattern.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator


class _FlakyEstimator(BaseEstimator):
    """A tiny sklearn-clone-compatible selector whose .fit() raises on every Nth call, always
    selecting feature 0 on success -- mirrors the injected-failure pattern used for
    _stability_cluster.py's sibling regression test."""

    def __init__(self, fail_every: int = 2):
        self.fail_every = fail_every

    def fit(self, X, y):
        """Raise on every ``fail_every``-th call (process-wide counter); otherwise select feature 0 only."""
        _FlakyEstimator._calls += 1
        if _FlakyEstimator._calls % self.fail_every == 0:
            raise ValueError("injected degenerate-bootstrap failure")
        self.support_ = np.zeros(self.n_features_, dtype=bool)
        self.support_[0] = True
        return self

    _calls = 0
    n_features_ = 4


def _xy(n=200, p=4, seed=0):
    """Small synthetic classification fixture."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, p)), columns=[f"f{i}" for i in range(p)])
    y = pd.Series((X["f0"] > 0).astype(int))
    return X, y


class TestStabilityMRMRSurvivesFailedBootstrap:
    """B-14: StabilityMRMR.fit() must not crash when some bootstraps fail, and frequencies must divide by the effective (successful) count."""

    def test_fit_does_not_raise_with_some_failed_bootstraps(self):
        """A degenerate bootstrap that makes the inner estimator raise must not propagate out of .fit()."""
        from mlframe.feature_selection.filters.stability import StabilityMRMR

        _FlakyEstimator._calls = 0
        X, y = _xy()
        m = StabilityMRMR(estimator=_FlakyEstimator(fail_every=3), n_bootstraps=9, sample_fraction=0.5, n_jobs=1, stratify=False)
        m.fit(X, y)  # must not raise
        assert m.n_features_in_ == 4

    def test_frequency_divides_by_effective_not_nominal_count(self):
        """Feature 0 is selected in EVERY successful bootstrap -- its frequency must be 1.0 (divide by
        the effective/successful count), not < 1.0 (which a nominal-n_bootstraps denominator bug would give)."""
        from mlframe.feature_selection.filters.stability import StabilityMRMR

        _FlakyEstimator._calls = 0
        X, y = _xy()
        m = StabilityMRMR(estimator=_FlakyEstimator(fail_every=3), n_bootstraps=9, sample_fraction=0.5, n_jobs=1, stratify=False)
        m.fit(X, y)
        assert m.selection_probabilities_[0] == pytest.approx(
            1.0
        ), f"expected feature 0's frequency to be 1.0 (effective-count denominator), got {m.selection_probabilities_[0]}"
        assert 0 in m.support_

    def test_all_bootstraps_failing_raises_not_silently_succeeds(self):
        """When EVERY bootstrap fails (the input is fundamentally too small/degenerate, not just
        unlucky), .fit() must raise a clear RuntimeError -- not silently return a meaningless
        all-zero-frequency result. Partial failures (tested above) are tolerated; total failure is not."""
        from mlframe.feature_selection.filters.stability import StabilityMRMR

        _FlakyEstimator._calls = 0
        X, y = _xy()
        m = StabilityMRMR(estimator=_FlakyEstimator(fail_every=1), n_bootstraps=5, sample_fraction=0.5, n_jobs=1, stratify=False)
        with pytest.raises(RuntimeError, match="all .* bootstraps failed"):
            m.fit(X, y)


class TestStabilitySelectFeSurvivesFailedBootstrap:
    """B-15: stability_select_fe's _run_bootstraps must not crash when one MRMR sub-fit fails."""

    def test_run_bootstraps_excludes_failed_and_continues(self, monkeypatch):
        """Monkeypatch the resolved MRMR class so every 3rd bootstrap's .fit() raises; _run_bootstraps
        must return only the successful bootstraps' results, not crash."""
        from mlframe.feature_selection.filters import _stability_fe as mod

        calls = {"n": 0}

        class _FlakyMRMR:
            """A minimal MRMR stand-in whose .fit() raises on every 3rd call."""

            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def fit(self, X, y):
                """Raise on every 3rd call to simulate a degenerate-bootstrap crash; otherwise no-op."""
                calls["n"] += 1
                if calls["n"] % 3 == 0:
                    raise ValueError("injected degenerate-bootstrap failure")
                return self

            def get_feature_names_out(self):
                """Report a single stable engineered feature name on success."""
                return ["f0"]

        monkeypatch.setattr(mod, "_resolve_mrmr_cls", lambda: _FlakyMRMR)
        monkeypatch.setattr(mod, "_engineered_union", lambda m: {"f0": "raw"})

        X, y = _xy()
        rng = np.random.default_rng(0)
        per_boot = mod._run_bootstraps(X, y, base_mrmr_params={}, n_bootstraps=9, sample_fraction=0.5, rng=rng)

        # 9 calls, every 3rd fails -> 3 failures, 6 successes.
        assert len(per_boot) == 6, f"expected 6 successful bootstraps (9 - 3 failures), got {len(per_boot)}: {per_boot}"
        assert all(d == {"f0": "raw"} for d in per_boot)
