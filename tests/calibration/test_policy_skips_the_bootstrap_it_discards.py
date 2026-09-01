"""Under the default `selection="inner_cv"`, `pick_best_calibrator` ran a bootstrap whose result it threw away.

Both the point estimate and the interval were reassigned from the held-out inner-CV block a few lines later, so
per candidate the function paid 1000 ECE resamples plus a BCa jackknife for a number nothing read. Worse, the
shared resample matrix is refused above a 1 GiB ceiling: at `n_oof = 300_000` with the defaults the build is
1.2 GiB, so the whole call died with `MemoryError` computing a value that was going to be discarded.

The second finding in the same function: `_bootstrap_ece_with_indices` derived the BCa acceleration term with
the generic O(max_n * n) gather even though the O(n) closed form for ECE ships in the same package and is
already wired into two other bootstrap call sites.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.calibration import policy as pol


@pytest.fixture
def oof():
    """Miscalibrated binary OOF probabilities with both classes well represented."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    p = np.clip(0.5 + 0.25 * (y - 0.5) + rng.normal(0, 0.12, 4000), 0.001, 0.999)
    return np.column_stack([1.0 - p, p]), y.astype(float)


class TestTheDiscardedBootstrapIsNotRun:
    """The default path must not pay for a number it overwrites."""

    def test_the_resample_matrix_is_never_built_under_inner_cv(self, oof, monkeypatch):
        """The build is the part that can raise MemoryError, so it is the part that must not happen."""
        calls = []
        monkeypatch.setattr(pol, "_build_resample_indices", lambda *a, **k: calls.append(a) or np.zeros((2, len(a[0]) if a else 1), dtype=np.int64))
        pol.pick_best_calibrator(None, None, *oof, selection="inner_cv")
        assert not calls, "the discarded bootstrap's index matrix is still being built"

    def test_no_resample_ece_is_computed_under_inner_cv(self, oof, monkeypatch):
        """The 1000-resample loop itself, per candidate."""
        calls = []
        real = pol._bootstrap_ece_with_indices
        monkeypatch.setattr(pol, "_bootstrap_ece_with_indices", lambda *a, **k: calls.append(1) or real(*a, **k))
        pol.pick_best_calibrator(None, None, *oof, selection="inner_cv")
        assert not calls, f"{len(calls)} discarded bootstrap runs"

    def test_a_size_that_would_breach_the_matrix_ceiling_still_completes(self, oof, monkeypatch):
        """The user-visible failure: a MemoryError for a value nobody reads."""
        monkeypatch.setattr(pol, "_build_resample_indices", lambda *a, **k: (_ for _ in ()).throw(MemoryError("projected 1.2 GiB")))
        res = pol.pick_best_calibrator(None, None, *oof, selection="inner_cv")
        assert res["alternatives"], "inner_cv selection died on a bootstrap it does not use"


class TestTheSameOofPathIsUnchanged:
    """The bootstrap is still the source of the number on the path that reads it."""

    def test_same_oof_still_bootstraps(self, oof, monkeypatch):
        """Removing waste from one branch must not disable the other."""
        calls = []
        real = pol._bootstrap_ece_with_indices
        monkeypatch.setattr(pol, "_bootstrap_ece_with_indices", lambda *a, **k: calls.append(1) or real(*a, **k))
        pol.pick_best_calibrator(None, None, *oof, selection="same_oof", n_bootstrap=50)
        assert calls, "the same-OOF path no longer computes its interval"

    def test_same_oof_is_numerically_unchanged_by_the_new_jackknife(self, oof, monkeypatch):
        """The closed form must be a speedup, not a different answer. Forcing the old path must give the same CI.

        Note this path does NOT return an interval bracketing its own point estimate, and that is not a defect
        introduced here: an isotonic fit scored on the data it was fitted on has an in-sample ECE of ~1e-15 while
        every resample of it scores ~0.01. That optimism is exactly what `selection="inner_cv"` exists to remove.
        """
        fast = pol.pick_best_calibrator(None, None, *oof, selection="same_oof", n_bootstrap=100)
        monkeypatch.setattr(pol, "_jackknife_ece", lambda *a, **k: None)
        slow = pol.pick_best_calibrator(None, None, *oof, selection="same_oof", n_bootstrap=100)
        assert fast["ece_ci"] == slow["ece_ci"] and fast["chosen"] == slow["chosen"]

    def test_same_oof_returns_an_ordered_finite_interval(self, oof):
        """The weaker property that does hold on this path."""
        lo, hi = pol.pick_best_calibrator(None, None, *oof, selection="same_oof", n_bootstrap=100)["ece_ci"]
        assert np.isfinite([lo, hi]).all() and lo <= hi


class TestTheClosedFormJackknife:
    """`_jackknife_ece` must be the acceleration term, and must not change the interval."""

    def test_the_closed_form_is_used(self, monkeypatch):
        """The generic gather re-runs the metric over 2000 leave-one-out copies of the full array."""
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 800).astype(float)
        p = np.clip(0.5 + 0.2 * (y - 0.5) + rng.normal(0, 0.1, 800), 0.01, 0.99)
        gather = []
        monkeypatch.setattr(pol, "_jackknife_metric", lambda *a, **k: gather.append(1) or np.zeros(len(a[0])))
        idx = pol._build_resample_indices(800, 40, y, 0)
        pol._bootstrap_ece_with_indices(y, p, idx, lambda yy, pp: pol._ece_score(yy, pp, n_bins=10), 0.05, n_bins=10)
        assert not gather, "the generic O(max_n * n) gather jackknife is still being used for ECE"

    def test_the_interval_matches_the_generic_path(self):
        """Bit-identical is the claim; assert it rather than trust it."""
        from mlframe.evaluation._bootstrap_jackknife import _jackknife_ece
        from mlframe.evaluation.bootstrap import _jackknife_metric

        rng = np.random.default_rng(2)
        y = rng.integers(0, 2, 600).astype(float)
        p = np.clip(rng.random(600), 0.01, 0.99)
        def fn(yy, pp):
            """The ECE metric the generic jackknife re-runs per leave-one-out subset."""
            return pol._ece_score(yy, pp, n_bins=10)

        assert np.allclose(_jackknife_ece(y, p, n_bins=10), _jackknife_metric(y, p, fn), rtol=0, atol=1e-12)

    def test_a_degenerate_input_falls_back(self, monkeypatch):
        """`_jackknife_ece` returns None on n < 3 / non-binary labels; the generic path must still cover those."""
        gather = []
        real = pol._jackknife_metric
        monkeypatch.setattr(pol, "_jackknife_metric", lambda *a, **k: gather.append(1) or real(*a, **k))
        y = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        p = np.array([0.2, 0.8, 0.3, 0.7, 0.1, 0.9])
        monkeypatch.setattr(pol, "_jackknife_ece", lambda *a, **k: None)
        pol._bootstrap_ece_with_indices(y, p, pol._build_resample_indices(6, 20, y, 0), lambda yy, pp: pol._ece_score(yy, pp, n_bins=4), 0.05, n_bins=4)
        assert gather, "the fallback to the generic jackknife was lost"
