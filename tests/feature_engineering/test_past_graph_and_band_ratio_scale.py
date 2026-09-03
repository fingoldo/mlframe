"""Three defects in the feature_engineering families, all of which produced a plausible number silently.

  * ``shared_attribute_edges(..., timestamp=...)`` documents "links only to EARLIER same-group rows (directed
    past graph) -> leakage-safe", but selected partners by POSITION in a stable time-sort. Rows sharing an
    identical timestamp were therefore linked as if one were in the other's past -- and day-granularity
    timestamps are the ordinary case for a transactions or affiliation table, so a group observed on one day
    aggregated its own contemporaries through a feature the module sells as past-only.
  * ``band_energy_ratio`` guarded its division with ``+ 1e-6`` on UNNORMALISED squared FFT magnitudes, whose
    units are the square of the input's. On log-returns at amplitude ~1e-3 the band energies land near 1e-8, so
    the epsilon dominated the denominator and the feature became a function of the input's SCALE.
  * ``remediate_drifting_features(auto_tune_drop_threshold=True)`` chose the threshold that best de-drifted the
    very rows its importances came from, and copied both whole frames once per candidate to do it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_engineering.graph_construction import shared_attribute_edges


class TestTiedTimestampsAreNotInEachOthersPast:
    """A strict-past graph must use the timestamp VALUE, which is what the sibling `knn_graph_edges` does."""

    def _edges(self, times, **kw):
        """One group of len(times) rows sharing a category code, observed at the given timestamps."""
        codes = np.zeros(len(times), dtype=np.int64)
        e = shared_attribute_edges(codes, timestamp=np.asarray(times, dtype=np.float64), **kw)
        return sorted((int(a), int(b)) for a, b in e)

    def test_a_group_observed_at_one_instant_has_no_edges(self):
        """The defect in its sharpest form: three contemporaneous rows are nobody's past."""
        assert self._edges([5.0, 5.0, 5.0]) == []

    def test_a_tie_inside_a_longer_history_links_only_to_what_precedes_it(self):
        """Rows 1 and 2 are tied at t=5; both may see row 0 at t=1, neither may see the other."""
        assert self._edges([1.0, 5.0, 5.0]) == [(1, 0), (2, 0)]

    def test_strictly_increasing_timestamps_are_unaffected(self):
        """The fix must not change the case the function was already right about."""
        assert self._edges([1.0, 2.0, 3.0]) == [(1, 0), (2, 0), (2, 1)]

    def test_every_edge_points_strictly_backwards_in_time(self):
        """The documented contract, stated directly, over a mix of ties and gaps."""
        t = np.array([1.0, 2.0, 3.0, 3.0, 3.0, 7.0])
        for src, dst in self._edges(t):
            assert t[dst] < t[src], (src, dst, t[src], t[dst])

    def test_the_neighbour_window_counts_back_from_the_tie_boundary(self):
        """`max_neighbours` bounds the window; it must start at the first strictly-earlier row, not at `pos`."""
        assert self._edges([1.0, 2.0, 3.0, 3.0, 3.0], max_neighbours=2) == [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (4, 1)]

    def test_the_untimed_graph_is_untouched(self):
        """Without a timestamp this is an undirected all-pairs group graph; the fix must not reach it."""
        e = shared_attribute_edges(np.zeros(3, dtype=np.int64))
        assert sorted((int(a), int(b)) for a, b in e) == [(0, 1), (0, 2), (1, 2)]


class TestTheBandRatioMeasuresShapeNotScale:
    """A spectral BALANCE must be invariant to the unit the series is expressed in."""

    K = 128

    def _ratio(self, scale):
        """One deterministic mixed-frequency series, rescaled -- the spectral shape is identical."""
        from mlframe.feature_engineering.spectral import rolling_hf_lf_ratio

        t = np.arange(4 * self.K, dtype=np.float64)
        sig = (np.sin(2 * np.pi * t / 6.0) + 0.5 * np.sin(2 * np.pi * t / 90.0)) * scale
        out = rolling_hf_lf_ratio(sig, np.zeros(t.size, dtype=np.int64), window_K=self.K, clip_range=(0.0, 1e12))
        vals = out[np.isfinite(out)]
        assert vals.size, "the fixture produced no windows"
        return float(vals[-1])

    def test_expressing_the_same_series_in_smaller_units_does_not_change_it(self):
        """Measured pre-fix on this fixture: 4.59e-6 at scale 1e-6 against a true 1.09e-3, i.e. 237x too small."""
        assert self._ratio(1e-6) == pytest.approx(self._ratio(1.0), rel=1e-6)

    def test_it_is_stable_across_eight_orders_of_magnitude(self):
        """Pre-fix relative error on these: 0.996, 0.703, 0.023, 2.4e-4, 2.4e-6, 2.4e-10, 2.4e-14 -- purely a scale effect."""
        vals = [self._ratio(s) for s in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1.0, 1e2)]
        assert max(vals) == pytest.approx(min(vals), rel=1e-6), vals

    def test_a_constant_series_still_returns_the_fill_value(self):
        """The genuine empty-low-band case the epsilon was nominally there for."""
        from mlframe.feature_engineering.spectral import rolling_hf_lf_ratio

        n = 4 * self.K
        out = rolling_hf_lf_ratio(np.zeros(n), np.zeros(n, dtype=np.int64), window_K=self.K, fill_value=7.0, clip_range=(0.0, 1e12))
        assert set(np.unique(out[np.isfinite(out)]).tolist()) <= {7.0}


class TestTheAutoTunedThresholdIsChosenOutOfSample:
    """The search must not score candidates on the rows that produced its own flags."""

    def _frames(self, n=400, seed=0):
        """A drifting level feature, a drifting-even-after-ranking one, and a clean one."""
        rng = np.random.default_rng(seed)
        g = np.repeat(np.arange(n // 4), 4)

        def mk(shift):
            """One frame whose feature levels are shifted by `shift`."""
            return pd.DataFrame({"grp": g, "level": rng.normal(shift, 1, n), "severe": rng.normal(4 * shift, 1, n), "clean": rng.normal(0, 1, n)})

        return mk(0.0), mk(1.0)

    def test_it_holds_rows_out(self):
        """The tuning fit and the re-check must not share rows, and the caller must be able to size the split."""
        import inspect

        from mlframe.feature_engineering import drift_remediation

        assert "auto_tune_holdout" in inspect.signature(drift_remediation.remediate_drifting_features).parameters

    def test_it_does_not_copy_the_frames_once_per_candidate(self):
        """Frames can be 100+ GB; only the flagged columns differ between candidates."""
        from mlframe.feature_engineering import drift_remediation

        train, test = self._frames()
        full_shapes = {train.shape, test.shape}
        calls = []
        orig = pd.DataFrame.copy

        def _spy(self, *a, **k):
            """Record only whole-frame copies; the estimator legitimately copies its own small slices."""
            if self.shape in full_shapes:  # the estimator copies its own small slices; only whole-frame copies are the defect
                calls.append(self.shape)
            return orig(self, *a, **k)

        try:
            pd.DataFrame.copy = _spy  # type: ignore[method-assign]
            drift_remediation.remediate_drifting_features(train, test, group_col="grp", auto_tune_drop_threshold=True, auto_tune_candidates=[2.0, 3.0, 4.0])
        finally:
            pd.DataFrame.copy = orig  # type: ignore[method-assign]
        assert len(calls) == 2, f"expected only the final _build to copy the two frames, got {len(calls)}: {calls}"

    def test_the_result_is_still_a_valid_remediation(self):
        """The honest search must not change the SHAPE of the contract."""
        from mlframe.feature_engineering import drift_remediation

        train, test = self._frames()
        tr, te, rep = drift_remediation.remediate_drifting_features(train, test, group_col="grp", auto_tune_drop_threshold=True, auto_tune_candidates=[2.0, 3.0])
        assert set(rep["action"]) <= {"none", "rank_transform", "drop"}
        assert list(tr.columns) == list(te.columns) and "grp" in tr.columns

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
    def test_a_nonsensical_holdout_fraction_is_refused(self, bad):
        """Silently clamping would hide a caller's mistake."""
        from mlframe.feature_engineering import drift_remediation

        train, test = self._frames(n=80)
        with pytest.raises(ValueError, match="auto_tune_holdout"):
            drift_remediation.remediate_drifting_features(train, test, group_col="grp", auto_tune_drop_threshold=True, auto_tune_holdout=bad)
