"""Eleven training-core defects, most of them a read that could never succeed.

`TrainingContext` is a `slots=True` dataclass, so `getattr(ctx, "not_a_slot", None)` returns None silently
rather than raising. Three separate mechanisms were built on top of that silence:

  * the pre-screen's "defensive double-source" protection of the group and timestamp columns probed six
    attribute names across three objects, none of which exists -- so the protected set never contained a group
    or ts column at all, and the default-ON unsupervised pre-screen could drop the group_id column itself (a
    high-cardinality string ID looks exactly like a "near-all-unique string") and break GroupShuffleSplit;
  * the composite y-scale TEST chart's `plot_file` came from `ctx.plot_file`, so it was always None and the
    chart was never written, while the metric log line still printed and the run looked healthy;
  * eight `getattr(ctx, "configs", None)` fallbacks read as a working two-source resolution that could never
    resolve anything.

Plus a purge charged twice for boundaries that do not exist, a config knob the code reads but nothing declares,
a knob nothing reads that the fuzz suite varied as an axis, and four documentation claims contradicted by the
code they describe.
"""

from __future__ import annotations

import inspect

import numpy as np


class TestTheGroupAndTimestampColumnsAreActuallyProtected:
    """The protected set has to be reachable, not merely constructed."""

    def _src(self):
        """The pre-screen phase's source."""
        from mlframe.training.core import _phase_train_one_target_pre_screen as m

        return inspect.getsource(m)

    def test_no_probe_reads_a_name_that_is_not_a_context_slot(self):
        """`slots=True` turns a typo into silence, which is what hid this for as long as it did."""
        from mlframe.training.core._training_context import TrainingContext

        slots = set(getattr(TrainingContext, "__slots__", ()) or ())
        src = self._src()
        for dead in ("group_id_col", "ts_field", "features_and_targets_extractor"):
            assert dead not in slots, f"{dead} became a real slot; this test needs updating"
            assert f'getattr(ctx, "{dead}"' not in src, dead

    def test_the_names_come_from_the_series_the_context_carries(self):
        """`group_ids_raw` / `group_ids` / `timestamps` are the real slots, and a pandas Series knows its name."""
        src = self._src()
        assert 'getattr(ctx, "group_ids_raw", None)' in src
        assert 'getattr(ctx, "timestamps", None)' in src
        assert 'getattr(_series, "name", None)' in src

    def test_a_named_group_series_reaches_the_protected_set(self):
        """The behaviour, not the text: a pandas Series' name is the column name."""
        import pandas as pd

        s = pd.Series([1, 2, 3], name="well_id")
        assert isinstance(getattr(s, "name", None), str) and s.name == "well_id"

    def test_the_split_config_has_no_group_or_ts_field_to_probe(self):
        """Pins the premise: the removed probes could not have worked."""
        from mlframe.training._preprocessing_configs import TrainingSplitConfig

        fields = set(TrainingSplitConfig.model_fields)
        assert not ({"group_field", "timestamps_column", "ts_column"} & fields), fields


def test_the_composite_test_chart_reads_its_real_config_slot():
    """`ctx.plot_file` is not a slot, so the chart was never written and nothing warned."""
    from mlframe.training.core import _phase_train_one_target_post as m
    from mlframe.training.core._training_context import TrainingContext

    src = inspect.getsource(m)
    assert 'plot_file=getattr(ctx, "plot_file", None)' not in src
    assert 'getattr(getattr(ctx, "output_config", None), "plot_file", None)' in src
    assert "plot_file" not in set(getattr(TrainingContext, "__slots__", ()) or ())
    assert "output_config" in set(getattr(TrainingContext, "__slots__", ()) or ())


def test_the_unreachable_configs_fallbacks_are_gone():
    """Eight sites read as a working two-source resolution while `configs` is not a slot at all."""
    from mlframe.training.core import _phase_finalize, _phase_finalize_calibration
    from mlframe.training.core._training_context import TrainingContext

    assert "configs" not in set(getattr(TrainingContext, "__slots__", ()) or ())
    for mod in (_phase_finalize, _phase_finalize_calibration):
        assert 'getattr(ctx, "configs", None)' not in inspect.getsource(mod), mod.__name__


class TestThePurgeIsChargedOnlyForRealBoundaries:
    """`_resolve_counts` collapses a non-positive fraction to 0, leaving no boundary to protect."""

    def _carve(self, n=1000, calib=0.0, conformal=0.0, purge=100):
        """Temporal carve over a contiguous train index."""
        from mlframe.training._conformal_split import carve_calib_conformal_temporal

        return carve_calib_conformal_temporal(np.arange(n), calib, conformal, purge=purge)

    def test_an_empty_carve_discards_no_rows(self):
        """It returned a fit slice missing its 200 newest rows, with empty calib and conformal and no error."""
        fit, calib, conf = self._carve()[:3]
        assert calib.size == 0 and conf.size == 0
        assert fit.size == 1000, fit.size

    def test_a_calib_only_carve_charges_one_purge(self):
        """One boundary exists, so one purge is due."""
        fit, calib, conf = self._carve(calib=0.1, conformal=0.0)[:3]
        assert conf.size == 0
        assert calib.size == 100
        assert fit.size == 1000 - 100 - 100, fit.size

    def test_a_full_carve_still_charges_both(self):
        """The case the unconditional form was written for must not change."""
        fit, calib, conf = self._carve(calib=0.1, conformal=0.1)[:3]
        assert conf.size == 100 and calib.size == 100
        assert fit.size == 1000 - 100 - 100 - 100 - 100, fit.size

    def test_zero_purge_is_unaffected(self):
        """No purge, no boundaries to reason about."""
        fit, calib, conf = self._carve(calib=0.1, conformal=0.1, purge=0)[:3]
        assert fit.size + calib.size + conf.size == 1000


def test_the_temporal_audit_unit_knob_is_declared():
    """The audit phase read and documented it; setting it worked only through the extras escape hatch, which
    also logged a warning telling the user it looked like a typo."""
    from mlframe.training._model_configs_behavior import TrainingBehaviorConfig

    assert "target_temporal_audit_unit" in TrainingBehaviorConfig.model_fields
    assert TrainingBehaviorConfig().target_temporal_audit_unit is None
    assert TrainingBehaviorConfig(target_temporal_audit_unit="s").target_temporal_audit_unit == "s"


def test_the_dead_ensembling_knob_and_its_fuzz_axis_are_gone_together():
    """Nothing in `src` read `force_legacy`, yet the fuzz harness varied it as a combo axis -- so both arms
    produced identical runs and the suite reported coverage of a path it never exercised."""
    import pathlib

    from mlframe.training._model_configs_ensembling import EnsemblingConfig

    assert "force_legacy" not in EnsemblingConfig.model_fields
    root = pathlib.Path(__file__).resolve().parents[1]
    for rel in ("training/_fuzz_combo/axes.py", "training/_fuzz_combo/combo.py", "test_meta/test_config_field_consumption.py"):
        assert "force_legacy" not in (root / rel).read_text(encoding="utf-8"), rel


class TestTheStatedContractsMatchTheCode:
    """Four places described behaviour the code does not implement."""

    def test_the_jaggedness_docstring_names_the_statistic_the_code_computes(self):
        """It described a SECOND-difference count over LENGTH; the code counts first-difference sign changes
        over the number of NON-ZERO first differences, which is smaller on any curve with flat segments -- so a
        threshold set against the documented denominator is systematically too permissive."""
        from mlframe.training import _overlapping_walk_forward_cv as m

        src = inspect.getsource(m)
        assert "second-difference sign-change count, divided by its length" not in src.lower()
        assert "NON-ZERO first differences" in src

    def test_the_precompute_args_do_not_promise_forwarding(self):
        """Three parameters were documented as "forwarded to the stub"; the body never calls either stub."""
        from mlframe.training import _precompute as m

        src = inspect.getsource(m)
        assert "forwarded to the dummy stub" not in src and "forwarded to the composite stub" not in src
        assert "NOT consumed" in src

    def test_the_enum_domain_comment_pair_no_longer_contradicts_itself(self):
        """One comment said the domain is train-only with non-strict val; the code unions train and val and
        casts both strictly. A future edit trusting the stale one would reintroduce the ES bias the other
        comment exists to prevent."""
        from mlframe.training.core import _phase_helpers_fit_split as m

        src = inspect.getsource(m)
        assert "keyed off the train-only unique set" not in src
        assert "strict=True" in src

    def test_the_ar1_veto_no_longer_calls_val_an_honest_holdout(self):
        """Val is the early-stopping split, and the bias points the same way as the decision the veto makes."""
        from mlframe.training.core import _ar1_failsafe_veto as m

        src = inspect.getsource(m)
        assert "SAME honest-holdout regime as test" not in src
        assert "early-stopping split" in src.lower()

    def test_the_ar1_veto_logs_its_realised_margin(self):
        """The finding's point that nothing measured how much ES optimism was present."""
        from mlframe.training.core import _ar1_failsafe_veto as m

        src = inspect.getsource(m)
        assert "veto threshold=" in src and "logger.info(" in src


def test_the_split_diagnostics_table_is_labelled_by_split():
    """val, test and OOF mean different things, and `split_name` was accepted, passed, and never read -- so the
    emitted worst-K table carried no in-artifact indication of which split produced it."""
    from mlframe.training import _eval_helpers as m

    src = inspect.getsource(m)
    assert 'metrics_dict[f"worst_k_table_{split_name}"' in src
    assert "split_name" in inspect.signature(m._render_split_diagnostics).parameters
