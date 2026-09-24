"""Warnings a production run printed that were false, contradictory or repeated with no new information.

Of 240 WARNING lines in one run, 189 said "failed to cap n_jobs=1 ... oversubscription risk" about a Ridge pipeline that
has no n_jobs and cannot oversubscribe, 21 repeated one commit-pressure fact every heartbeat because the throttle keyed
on a message whose number changed each time, and one advised a switch that could not do what it promised.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

from mlframe.training.composite.discovery._screening_tiny import _build_tiny_model, cap_inner_n_jobs


def _tiny(family: str):
    return _build_tiny_model(family, n_estimators=5, num_leaves=7, learning_rate=0.1, random_state=0, deterministic=False, inner_n_jobs=-1)


class TestInnerNJobsCap:
    """Only knobs a model exposes are set; a model with none is left alone, silently."""

    def test_a_pipeline_without_n_jobs_is_not_a_warning(self, caplog):
        """The defect: set_params(n_jobs=1) raised on Pipeline(SimpleImputer, Ridge) and logged a false risk."""
        model = _tiny("linear")
        with caplog.at_level(logging.WARNING):
            cap_inner_n_jobs(model, 16)
        assert not [r for r in caplog.records if "cap" in r.getMessage()]

    def test_a_booster_is_capped(self):
        """The case the cap exists for."""
        model = _tiny("lgb")
        cap_inner_n_jobs(model, 16)
        assert model.get_params()["n_jobs"] == 1

    def test_a_nested_step_is_capped(self):
        """A pipeline whose step does thread is capped through the step's own parameter."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.pipeline import Pipeline

        model = Pipeline([("rf", RandomForestRegressor(n_jobs=-1))])
        cap_inner_n_jobs(model, 16)
        assert model.get_params(deep=True)["rf__n_jobs"] == 1

    def test_serial_folds_leave_the_model_alone(self):
        """Nothing to protect against when the folds do not run in parallel."""
        model = _tiny("lgb")
        before = model.get_params()["n_jobs"]
        cap_inner_n_jobs(model, 1)
        assert model.get_params()["n_jobs"] == before

    def test_a_model_that_exposes_the_knob_but_rejects_it_still_warns(self, caplog):
        """That one is a real oversubscription risk and must stay visible."""

        class _Stubborn:
            def get_params(self, deep=True):
                return {"n_jobs": -1}

            def set_params(self, **kw):
                raise ValueError("n_jobs is read-only here")

        with caplog.at_level(logging.WARNING):
            cap_inner_n_jobs(_Stubborn(), 16)
        assert [r for r in caplog.records if "failed to cap n_jobs=1" in r.getMessage()]


class TestCommitPressureRepetition:
    """The heartbeat repeats a condition when it changes, not every beat."""

    def _run(self, monkeypatch, private_gb: float, avail_gb: float = 200.0):
        import mlframe.training._commit_headroom as ch

        monkeypatch.setattr(
            "mlframe.training.crash_diagnostics.windows_commit_status",
            lambda: {"commit_limit_gb": 300.0, "commit_avail_gb": avail_gb},
        )

        class _MI:
            rss = int(1.0 * 1024**3)
            private = int(private_gb * 1024**3)

        monkeypatch.setattr("psutil.Process", lambda *a, **k: SimpleNamespace(memory_info=lambda: _MI()))
        return ch.check_and_warn(throttle_key="heartbeat_commit_pressure")

    def test_retained_commit_is_not_repeated_each_beat(self, monkeypatch, caplog):
        """The defect: 21 identical warnings, because the throttle key carried the changing number."""
        import mlframe.training._commit_headroom as ch

        monkeypatch.setattr(ch, "_last_retained_reported_gb", None)
        with caplog.at_level(logging.WARNING):
            for gb in (58.3, 58.4, 58.6, 58.7, 58.8):
                self._run(monkeypatch, gb)
        assert len([r for r in caplog.records if "private commit" in r.getMessage()]) == 1

    def test_material_growth_is_reported_again(self, monkeypatch, caplog):
        """Growth that can end the run is new information."""
        import mlframe.training._commit_headroom as ch

        monkeypatch.setattr(ch, "_last_retained_reported_gb", None)
        with caplog.at_level(logging.WARNING):
            self._run(monkeypatch, 58.3)
            self._run(monkeypatch, 67.0)
        assert len([r for r in caplog.records if "private commit" in r.getMessage()]) == 2


class TestStructuralNaNAdvice:
    """The "set auto_drop=False to keep them" advice is only given where the switch would keep them."""

    @staticmethod
    def _report(fraction: float, cols):
        return SimpleNamespace(feature_warnings={c: [f"nan_fraction={fraction:.4f} >= 0.99"] for c in cols})

    def test_columns_the_pre_screen_drops_anyway_get_no_false_advice(self, caplog):
        """The defect: the advice followed a line saying the same columns would be dropped regardless."""
        from mlframe.training.core._main_train_suite_target_distribution import _warn_structural_nan_drops

        cols = [f"c{i}" for i in range(9)]
        with caplog.at_level(logging.WARNING):
            _warn_structural_nan_drops(cols, self._report(0.999, cols), SimpleNamespace(), 0.99)
        assert not caplog.records

    def test_columns_the_auto_drop_decided_get_the_advice_with_one_percent_sign(self, caplog):
        """Where the switch does keep them, say so -- and render 99%, not 99%%."""
        from mlframe.training.core._main_train_suite_target_distribution import _warn_structural_nan_drops

        cols = [f"c{i}" for i in range(6)]
        with caplog.at_level(logging.WARNING):
            _warn_structural_nan_drops(cols, self._report(0.985, cols), SimpleNamespace(), 0.99)
        (record,) = caplog.records
        assert ">=99% missing" in record.getMessage() and "%%" not in record.getMessage()
