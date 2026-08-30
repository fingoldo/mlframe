"""``prefer_polarsds=False`` alone must not force a polars->pandas conversion.

The deferral gate treated "the polars-ds pipeline was not applied" as "the fitted pipeline state lives in
pandas". Those coincide only when a transform was actually REQUESTED. A CatBoost-only run asking for no
encoder, no scaler and no imputer has no fitted state at all, so there is nothing for pandas to carry -- yet a
production run converted a 2.2M x 113 frame and reported the reason as ``prefer_polarsds=False``.

CatBoost consumes polars natively, so that conversion bought nothing and cost RAM and wall time.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mlframe.training.core._main_train_suite_polars_gate import any_pipeline_stage_requested as _any_pipeline_stage_requested


class TestWhatCountsAsARequestedStage:
    """Only a transform whose fitted state must travel with the frames counts."""

    def test_no_stages_requested(self):
        """The configuration from the production run."""
        cfg = SimpleNamespace(categorical_encoding=None, scaler_name=None, imputer_strategy=None, dim_reducer_name=None)
        assert _any_pipeline_stage_requested(cfg) is False

    @pytest.mark.parametrize(
        "field, value",
        [
            ("categorical_encoding", "ordinal"),
            ("scaler_name", "standard"),
            ("imputer_strategy", "median"),
            ("dim_reducer_name", "pca"),
        ],
    )
    def test_any_single_stage_counts(self, field, value):
        """One fitted transform is enough: its state has to reach predict time."""
        cfg = SimpleNamespace(categorical_encoding=None, scaler_name=None, imputer_strategy=None, dim_reducer_name=None)
        setattr(cfg, field, value)
        assert _any_pipeline_stage_requested(cfg) is True

    def test_dict_form_is_read_too(self):
        """The suite accepts both the dataclass and the dict form of the config."""
        assert _any_pipeline_stage_requested({"categorical_encoding": None, "scaler_name": None}) is False
        assert _any_pipeline_stage_requested({"scaler_name": "robust"}) is True

    def test_none_config_requests_nothing(self):
        """No config at all is the strongest form of "no transforms"."""
        assert _any_pipeline_stage_requested(None) is False

    def test_an_unreadable_config_is_conservative(self):
        """Not being able to tell must not silently disable a conversion something may depend on."""

        class Hostile:
            """A config whose attribute access raises."""

            def __getattr__(self, name):
                """Always raises, standing in for an exotic config object."""
                raise RuntimeError("nope")

        assert _any_pipeline_stage_requested(Hostile()) is True


class TestTheGateItself:
    """The decision the log reported, exercised through the real phase helper."""

    def _call(self, *, polars_applied: bool, stages: bool):
        """Run the conversion phase on a small polars frame and report whether it deferred."""
        pl = pytest.importorskip("polars")
        from mlframe.training.core._phase_helpers import _phase_pandas_conversion_and_cat_prep

        frame = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        result = _phase_pandas_conversion_and_cat_prep(
            train_df=frame,
            val_df=frame,
            test_df=frame,
            train_df_polars_pre=frame,
            val_df_polars_pre=frame,
            test_df_polars_pre=frame,
            cat_features=[],
            was_polars_input=True,
            all_models_polars_native=True,
            needs_polars_pre_clone=False,
            mlframe_models=["cb"],
            recurrent_models=[],
            rfecv_models=[],
            baseline_rss_mb=0.0,
            df_size_mb=0.0,
            verbose=False,
            polars_pipeline_applied=polars_applied,
            pipeline_stages_requested=stages,
        )
        return result[-2]  # defer_pandas_conv

    def test_no_stages_requested_keeps_polars(self):
        """The fix: nothing was requested, so an unapplied pipeline is not a reason to convert."""
        assert self._call(polars_applied=False, stages=False) is True

    def test_requested_but_unapplied_still_converts(self):
        """The case the gate was written for stays intact: fitted state exists only in pandas."""
        assert self._call(polars_applied=False, stages=True) is False

    def test_applied_pipeline_keeps_polars(self):
        """Unchanged behaviour when the polars pipeline did run."""
        assert self._call(polars_applied=True, stages=True) is True
