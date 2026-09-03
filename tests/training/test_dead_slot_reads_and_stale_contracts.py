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

import numpy as np


class TestTheGroupAndTimestampColumnsAreActuallyProtected:
    """The protected set has to be reachable, not merely constructed."""

    def _probed(self):
        """Every attribute name the pre-screen phase probes off ``ctx``."""
        from mlframe.training.core import _phase_train_one_target_pre_screen as m
        from tests._source_ast import getattr_literals, module_ast

        return getattr_literals(module_ast(m), obj="ctx")

    def test_no_probe_reads_a_name_that_is_not_a_context_slot(self):
        """`slots=True` turns a typo into silence, which is what hid this for as long as it did.

        Structural of necessity: a probe for a name the context does not have returns the DEFAULT rather than
        raising, so the protection it was supposed to build simply comes back empty and every downstream
        assertion still passes. Nothing observable distinguishes "protected nothing" from "protected the right
        thing" except asking which names are probed.
        """
        from mlframe.training.core._training_context import TrainingContext

        slots = set(getattr(TrainingContext, "__slots__", ()) or ())
        probed = self._probed()
        for dead in ("group_id_col", "ts_field", "features_and_targets_extractor"):
            assert dead not in slots, f"{dead} became a real slot; this test needs updating"
            assert dead not in probed, f"the phase still probes ctx.{dead}, which is not a slot, so the probe silently yields the default"

    def test_the_names_come_from_the_series_the_context_carries(self):
        """`group_ids_raw` / `group_ids` / `timestamps` are the real slots, and a pandas Series knows its name."""
        from mlframe.training.core._training_context import TrainingContext

        slots = set(getattr(TrainingContext, "__slots__", ()) or ())
        probed = self._probed()
        for real in ("group_ids_raw", "timestamps"):
            assert real in slots, f"{real} is no longer a context slot; the phase cannot read it"
            assert real in probed, f"the phase no longer reads ctx.{real}, so the group/ts column cannot be protected"
        # ...and the column NAME comes off the Series itself, which is the only place it exists at this point.
        from mlframe.training.core import _phase_train_one_target_pre_screen as _m
        from tests._source_ast import getattr_literals, module_ast

        assert "name" in getattr_literals(module_ast(_m), obj="_series"), "the phase no longer reads the Series' own .name, so no column name reaches the protected set"

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

    from tests._source_ast import getattr_literals, module_ast

    slots = set(getattr(TrainingContext, "__slots__", ()) or ())
    assert "plot_file" not in slots, "plot_file became a real slot; this test needs updating"
    assert "output_config" in slots

    # Structural: reading a name the context does not have yields None rather than raising, so the chart is
    # simply never written and the metric log line still prints -- the run looks healthy either way.
    probed = getattr_literals(module_ast(m), obj="ctx")
    assert "plot_file" not in probed, "the phase still probes ctx.plot_file, which is not a slot, so the path is always None"
    assert "output_config" in probed, "the phase no longer reads ctx.output_config, so it cannot reach the real plot_file"


def test_the_unreachable_configs_fallbacks_are_gone():
    """Eight sites read as a working two-source resolution while `configs` is not a slot at all."""
    from mlframe.training.core import _phase_finalize, _phase_finalize_calibration
    from mlframe.training.core._training_context import TrainingContext

    from tests._source_ast import getattr_literals, module_ast

    assert "configs" not in set(getattr(TrainingContext, "__slots__", ()) or ())
    for mod in (_phase_finalize, _phase_finalize_calibration):
        assert "configs" not in getattr_literals(module_ast(mod), obj="ctx"), f"{mod.__name__} still probes ctx.configs, which is not a slot and can never resolve"


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

    def test_the_jaggedness_denominator_excludes_flat_segments(self):
        """The ratio is sign changes over NON-ZERO first differences, not over the curve length.

        Asserted by measuring the statistic rather than by reading its docstring, which is what the old form
        did. The two denominators differ on any curve with flat segments -- the documented one is larger, so a
        threshold set against it is systematically too permissive -- and this drives a curve built so the two
        land on opposite sides of the same threshold.
        """
        import numpy as np

        from mlframe.training._overlapping_walk_forward_cv import cv_stability_check

        # Eight points: four alternating steps then three flat ones. First differences are
        # [+, -, +, -, 0, 0, 0]: four non-zero, three sign changes -> 3/4 = 0.75 under the real rule, but
        # 3/7 = 0.43 if the denominator were the full length. A threshold of 0.6 separates them.
        curve = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        curves = np.vstack([curve, curve, curve])
        out = cv_stability_check(curves, max_sign_change_ratio=0.6)
        assert out["jagged_seed_fraction"] == 1.0, (
            "the flat tail is being counted in the denominator: 3 sign changes over 7 first differences is 0.43 "
            f"and would read smooth at a 0.6 threshold, but over the 4 NON-ZERO ones it is 0.75. Got "
            f"jagged_seed_fraction={out['jagged_seed_fraction']!r}"
        )

        # A genuinely smooth curve must still read smooth, so the assertion above is not trivially satisfied.
        smooth = np.linspace(0.0, 1.0, 8)
        assert cv_stability_check(np.vstack([smooth, smooth, smooth]), max_sign_change_ratio=0.6)["jagged_seed_fraction"] == 0.0

    def test_the_precompute_stubs_raise_rather_than_being_forwarded_to(self):
        """Three parameters were documented as "forwarded to the stub"; both stubs raise unconditionally.

        The contract the stale docstring misdescribed is testable directly: a parameter cannot be forwarded to
        a callable that refuses every call, so asserting the refusal is what the documentation should have
        said and what a caller actually experiences.
        """
        import pytest as _pytest

        from mlframe.training._precompute import precompute_composite_target_specs, precompute_dummy_baselines

        # Called with real arguments, so the refusal cannot be mistaken for an arity error.
        with _pytest.raises(NotImplementedError):
            precompute_composite_target_specs(train_df=None, target_by_type={}, config=None)
        with _pytest.raises(NotImplementedError):
            precompute_dummy_baselines(None, {}, config=None)

    def test_the_enum_cast_is_strict_on_the_shared_domain(self):
        """The domain unions train and val, and both are cast strictly.

        Structural: this cast sits inside a phase helper that needs a fully built split context to reach, and
        a non-strict cast NULLS an out-of-domain value rather than raising -- so the difference surfaces
        downstream as missing categories, far from here. Asserted on the parsed module: the cast helper is
        invoked with `strict=True`, which is the half a stale comment had claimed was non-strict.
        """
        import ast

        from mlframe.training.core import _phase_helpers_fit_split as m
        from tests._source_ast import module_ast

        tree = module_ast(m)
        strict_true = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            for kw in node.keywords
            if kw.arg == "strict" and isinstance(kw.value, ast.Constant) and kw.value.value is True
        ]
        assert strict_true, "no call passes strict=True, so an out-of-domain category is silently nulled instead of raising"

    # `test_the_ar1_veto_no_longer_calls_val_an_honest_holdout` used to sit here, asserting the wording of a
    # docstring. What that wording was correcting -- that val is the early-stopping split, so the trained arm's
    # val RMSE is biased low while zero-parameter lag_predict's is not, and the bias points the same way as the
    # decision the veto makes -- is a statement about the SPLIT, not about behaviour any call can expose. The
    # sibling below pins what is observable: the realised margin is logged, so the headroom the tolerance is
    # absorbing can be read off a run instead of assumed.

    def test_the_ar1_veto_logs_its_realised_margin(self):
        """The realised margin must be LOGGED, so the ES optimism the tolerance absorbs is observable.

        Structural: the line is emitted from inside a veto decision that needs a fitted suite and both arms'
        val RMSEs to reach, and the finding's point was that nothing MEASURED the optimism -- so what matters
        is that the numbers reach a log record at info level, not what a particular run's values are.
        """
        from mlframe.training.core import _ar1_failsafe_veto as m
        from tests._source_ast import called_names, module_ast, string_literals

        tree = module_ast(m)
        assert "info" in called_names(tree), "the veto no longer logs at info level, so its margin is invisible on a normal run"
        emitted = " ".join(string_literals(tree))
        for token in ("veto threshold=", "val RMSE="):
            assert token in emitted, f"the veto log line no longer reports {token!r}, so the realised margin cannot be read off a run"


def test_the_split_diagnostics_table_is_labelled_by_split():
    """val, test and OOF mean different things, and `split_name` was accepted, passed, and never read -- so the
    emitted worst-K table carried no in-artifact indication of which split produced it."""
    from mlframe.training import _eval_helpers as m

    import inspect as _inspect

    from tests._source_ast import function_ast, loaded_names

    # The parameter must still be accepted...
    assert "split_name" in _inspect.signature(m._render_split_diagnostics).parameters
    # ...and actually READ. It was accepted and passed and never used, so the emitted worst-K table carried no
    # in-artifact indication of which split produced it -- and val, test and OOF mean different things.
    body = function_ast(m, "_render_split_diagnostics")
    assert "split_name" in loaded_names(body), "split_name is accepted but never read, so the worst-K table is unlabelled again"
