"""Two metrics defects that produced a confident number from a degenerate or unmeasured input.

  * `assess_prediction_band_stability` substituted the MEANINGFUL value 1.0 ("no correction") into the bootstrap
    distribution whenever a resample's in-band mean prediction was exactly zero. That point is not a draw from
    the estimator's sampling distribution, so `bootstrap_std` measured the spread between the real factor and a
    pile of 1.0s, and `is_stable` was decided from that mixture -- invisibly, because the point estimate uses the
    same 1.0 convention.
  * `_ktc_dispatch`'s three `except Exception` handlers logged at DEBUG, so a persistent kernel-tuning-cache
    failure silently downgraded `odds_ratio_combine`'s backend choice from the measured per-host winner to a
    hardcoded size threshold for the life of the process. The measured spread between backends is up to 9x, and
    the module the first handler imports probes CUDA at import time -- the exact mechanism that once downgraded
    this process's whole MI backend.

The `triage_cv_delta` band is checked here too, because the fix it received differs from what the audit
suggested and the difference is worth pinning.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest


class TestADegenerateResampleIsDroppedNotSubstituted:
    """A band straddling zero makes `mean(y_pred) == 0` reachable on a real fraction of resamples."""

    def _report(self, y_true, y_pred, band, **kw):
        """Stability of the correction factor over the given prediction band."""
        from mlframe.calibration.prediction_band_correction import assess_prediction_band_stability

        lo, hi = band
        return assess_prediction_band_stability(np.asarray(y_true, float), np.asarray(y_pred, float), lo, hi, random_state=0, **kw)

    def _straddling(self, n=60):
        """Predictions symmetric about zero, so a resample's mean can land exactly on it."""
        pred = np.tile([-1.0, 1.0], n // 2)
        true = pred * 1.5
        return true, pred

    def test_a_band_straddling_zero_is_not_reported_stable(self):
        """The old form filled those slots with 1.0 and let `is_stable` be decided from the mixture."""
        true, pred = self._straddling()
        rep = self._report(true, pred, band=(-2.0, 2.0), min_band_n=1)
        assert rep.is_stable is False, rep

    def test_the_degenerate_resamples_are_announced(self, caplog):
        """Silence is what made the contamination invisible."""
        true, pred = self._straddling()
        with caplog.at_level(logging.WARNING, logger="mlframe.calibration.prediction_band_correction"):
            self._report(true, pred, band=(-2.0, 2.0), min_band_n=1)
        assert any("exactly-zero in-band mean" in r.message for r in caplog.records), [r.message for r in caplog.records]

    def test_the_fixture_really_produces_degenerate_resamples(self):
        """Without this the two assertions above could pass for an unrelated reason."""
        rng = np.random.default_rng(0)
        _, pred = self._straddling()
        hits = sum(1 for _ in range(500) if pred[rng.integers(0, pred.size, size=pred.size)].mean() == 0.0)
        assert hits > 0, "no resample degenerates; this fixture proves nothing"

    def test_a_well_conditioned_band_is_unaffected(self):
        """The fix must not start rejecting bands that were fine."""
        rng = np.random.default_rng(1)
        pred = rng.uniform(5.0, 10.0, 400)
        true = pred * 1.2
        rep = self._report(true, pred, band=(5.0, 10.0), min_band_n=10)
        assert rep.is_stable is True and rep.bootstrap_std == pytest.approx(0.0, abs=1e-9), rep

    def test_the_factor_itself_is_still_reported(self):
        """Dropping resamples changes the uncertainty, never the point estimate."""
        rng = np.random.default_rng(2)
        pred = rng.uniform(5.0, 10.0, 400)
        rep = self._report(pred * 1.2, pred, band=(5.0, 10.0), min_band_n=10)
        assert rep.factor == pytest.approx(1.2, rel=1e-9)


class TestTheBackendDowngradeIsAudible:
    """A hardcoded size threshold standing in for a measured winner must not be a debug line."""

    PATH = "src/mlframe/calibration/_ktc_dispatch.py"

    def _src(self):
        """The dispatch module's source."""
        import pathlib

        return (pathlib.Path(__file__).resolve().parents[2] / self.PATH).read_text(encoding="utf-8")

    def test_the_import_guard_is_narrowed_to_import_error(self):
        """The module it imports probes CUDA at import time, so a device fault is not a missing package."""
        src = self._src()
        assert "except ImportError as exc:" in src

    def test_every_handler_reaches_at_least_warning(self):
        """The three handlers on the DISPATCH path logged at debug, which production logging does not emit.

        Scoped to `_get_cache` and `choose_odds_combine_backend`. The handler inside `_make_tuner` stays at debug
        on purpose: it probes an optional cupy backend during a benchmark sweep, where absence is expected and is
        already recorded by that backend simply being missing from the timings.
        """
        import ast

        tree = ast.parse(self._src())
        scoped = [f for f in ast.walk(tree) if isinstance(f, ast.FunctionDef) and f.name in ("_get_cache", "choose_odds_combine_backend")]
        assert len(scoped) == 2, [f.name for f in scoped]
        handlers = [n for f in scoped for n in ast.walk(f) if isinstance(n, ast.ExceptHandler)]
        assert len(handlers) == 4, len(handlers)  # one ImportError guard plus the three that must be audible
        for h in handlers:
            names = {n.id for n in ast.walk(h.type) if isinstance(n, ast.Name)} if h.type else set()
            if names == {"ImportError"}:
                continue  # the genuine package-absent case stays at debug
            calls = [c for c in ast.walk(h) if isinstance(c, ast.Call)]
            attrs = {c.func.attr for c in calls if isinstance(c.func, ast.Attribute)}
            ids = {c.func.id for c in calls if isinstance(c.func, ast.Name)}
            assert attrs & {"warning", "error"} or "log_throttle" in ids, ast.dump(h)[:300]

    def test_the_handlers_are_throttled(self):
        """One line per process, not one per call -- the lookup runs on a hot dispatch path."""
        src = self._src()
        for key in ("odds_combine_ktc_import_failed", "odds_combine_ktc_singleton_failed", "odds_combine_ktc_lookup_failed"):
            assert src.count(f'"{key}"') >= 1, key

    def test_the_module_still_imports_and_dispatches(self):
        """Narrowed excepts and new logging calls must not break the dispatch itself."""
        from mlframe.calibration._ktc_dispatch import choose_odds_combine_backend

        assert isinstance(choose_odds_combine_backend(n=1000, k=4, fallback="numpy"), str)


class TestTheTriageBandAccountsForBothArms:
    """The band brackets a DIFFERENCE of two means, so both arms' spreads belong in it."""

    def _triage(self, b, c):
        """Feature-engineering triage on two paired fold-score arrays."""
        from mlframe.evaluation.cv_delta_triage import triage_cv_delta

        return triage_cv_delta(np.asarray(b, float), np.asarray(c, float), "feature_engineering")

    def test_a_candidate_with_a_huge_fold_spread_is_not_actionable_on_a_tiny_delta(self):
        """The audit's fixture: baseline std 7e-4, candidate std 0.036, delta 0.001. Band measured at 0.0365."""
        r = self._triage([0.800, 0.801, 0.799, 0.800, 0.800], [0.760, 0.840, 0.770, 0.830, 0.805])
        assert r["delta"] == pytest.approx(0.001, abs=1e-9)
        assert r["actionable"] is False, r

    def test_the_candidate_arm_actually_moves_the_band(self):
        """Same baseline and same delta, two candidate spreads -- a baseline-only band would tie them."""
        tight = self._triage([0.800, 0.801, 0.799, 0.800, 0.800], [0.801, 0.802, 0.800, 0.801, 0.801])
        wide = self._triage([0.800, 0.801, 0.799, 0.800, 0.800], [0.760, 0.840, 0.770, 0.830, 0.805])
        assert wide["band"] > 10 * tight["band"], (tight["band"], wide["band"])

    def test_a_perfectly_consistent_delta_does_not_get_a_zero_band(self):
        """Why the paired form was not adopted: `std(candidate - baseline)` is exactly 0 here, so a paired band
        would declare a delta actionable with infinite confidence off five folds."""
        b = [0.70, 0.75, 0.72, 0.80, 0.68]
        r = self._triage(b, [x + 0.01 for x in b])
        assert r["band"] > 0.0, r
        assert np.std(np.asarray([0.01] * 5), ddof=1) == 0.0  # the paired band's own numerator, for the record
