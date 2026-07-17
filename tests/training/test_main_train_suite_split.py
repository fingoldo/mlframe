"""Sensor: _main_train_suite.py W14A phase-grouping carve preserves identity + facade under budget.

Carve lifts 9 sub-bodies from ``train_mlframe_models_suite`` into ``_main_train_suite_phases.py``:
- ``validate_suite_inputs`` (4 required positional kwargs)
- ``apply_module_global_patches`` (apply_loky / apply_third_party_patches)
- ``warn_on_empty_target_by_type`` (empty extractor warning)
- ``check_precomputed_fingerprint`` (PRECOMP-NO-FP-CHECK)
- ``compute_or_fetch_trainset_features_stats`` (4-way routing)
- ``maybe_apply_composite_target_specs_precomputed``
- ``maybe_apply_dummy_baselines_precomputed`` (with config.enabled flip)
- ``apply_polars_cat_fixes_and_back_write_ctx`` (polars cat fills + ctx back-write)
- ``run_recurrent_finalize_and_composite_post`` (post-loop tail)
- ``export_votenrank_leaderboards`` (post-loop votenrank CSV export)

Bodies lifted verbatim so behavioural equivalence is preserved by construction.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


def test_w14a_main_train_suite_facade_under_budget():
    parent = Path(__file__).parent.parent.parent / "src" / "mlframe" / "training" / "core" / "_main_train_suite.py"
    facade_loc = sum(1 for _ in parent.open(encoding="utf-8"))
    # Budget tracks the irreducible orchestration shell plus the comprehensive 34-parameter
    # facade docstring (~95 lines, intentionally kept rich). The carveable helpers were lifted:
    # the 9 phase helpers live in `_main_train_suite_phases.py` and the return-shape contract +
    # string-target encoding in `_main_train_suite_encoding.py`. The remaining body is phase-call
    # glue over a shared local namespace -- further splitting would thread dozens of locals.
    assert facade_loc < 765, f"_main_train_suite.py LOC={facade_loc} exceeds 765 budget"


def test_w14a_main_train_suite_phases_identity():
    from mlframe.training.core import _main_train_suite as parent
    from mlframe.training.core import _main_train_suite_phases as phases

    assert parent.validate_suite_inputs is phases.validate_suite_inputs
    assert parent.apply_module_global_patches is phases.apply_module_global_patches
    assert parent.warn_on_empty_target_by_type is phases.warn_on_empty_target_by_type
    assert parent.check_precomputed_fingerprint is phases.check_precomputed_fingerprint
    assert parent.compute_or_fetch_trainset_features_stats is phases.compute_or_fetch_trainset_features_stats
    assert parent.maybe_apply_composite_target_specs_precomputed is phases.maybe_apply_composite_target_specs_precomputed
    assert parent.maybe_apply_dummy_baselines_precomputed is phases.maybe_apply_dummy_baselines_precomputed
    assert parent.apply_polars_cat_fixes_and_back_write_ctx is phases.apply_polars_cat_fixes_and_back_write_ctx
    assert parent.run_recurrent_finalize_and_composite_post is phases.run_recurrent_finalize_and_composite_post
    assert parent.export_votenrank_leaderboards is phases.export_votenrank_leaderboards


def test_encoding_sibling_reexport_identity():
    from mlframe.training.core import _main_train_suite as parent
    from mlframe.training.core import _main_train_suite_encoding as enc

    assert parent._assert_suite_return_shape is enc._assert_suite_return_shape
    assert parent._encode_string_multiclass_target is enc._encode_string_multiclass_target
    assert parent.SuiteResult is enc.SuiteResult


def test_assert_suite_return_shape_body_callable():
    from mlframe.training.core._main_train_suite_encoding import _assert_suite_return_shape
    import pytest

    assert _assert_suite_return_shape(({"m": 1}, {"meta": 2}), source="t") == ({"m": 1}, {"meta": 2})
    with pytest.raises(TypeError):
        _assert_suite_return_shape(["not", "dicts"], source="t")


def test_encode_string_multiclass_target_body_callable():
    import numpy as np
    from mlframe.training.core._main_train_suite_encoding import _encode_string_multiclass_target
    from mlframe.training.configs import TargetTypes

    md = {}
    codes = _encode_string_multiclass_target(TargetTypes.MULTICLASS_CLASSIFICATION, "y", np.array(["b", "a", "c", "a"]), md)
    assert list(codes) == [1, 0, 2, 0]
    assert md["target_label_classes"]["y"] == ["a", "b", "c"]
    # non-multiclass / numeric passes through unchanged
    same = _encode_string_multiclass_target(TargetTypes.REGRESSION, "y", np.array([1.0, 2.0]), {})
    assert list(same) == [1.0, 2.0]


def test_w14a_validate_suite_inputs_pathlike_coerced_to_str():
    from mlframe.training.core._main_train_suite_phases import validate_suite_inputs

    p = Path("data.parquet")
    result = validate_suite_inputs(p, "y", "exp1", SimpleNamespace())
    assert result == str(p)


def test_w14a_validate_suite_inputs_rejects_non_parquet_str():
    from mlframe.training.core._main_train_suite_phases import validate_suite_inputs

    import pytest

    with pytest.raises(ValueError, match="must be a .parquet file"):
        validate_suite_inputs("data.csv", "y", "exp1", SimpleNamespace())


def test_w14a_validate_suite_inputs_rejects_empty_target_name():
    from mlframe.training.core._main_train_suite_phases import validate_suite_inputs

    import pandas as pd
    import pytest

    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="target_name cannot be empty"):
        validate_suite_inputs(df, "   ", "exp1", SimpleNamespace())


def test_w14a_validate_suite_inputs_rejects_none_extractor():
    from mlframe.training.core._main_train_suite_phases import validate_suite_inputs

    import pandas as pd
    import pytest

    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="features_and_targets_extractor is required"):
        validate_suite_inputs(df, "y", "exp1", None)


def test_w14a_warn_on_empty_target_by_type_warns_caplog(caplog):
    """``warn_on_empty_target_by_type`` emits the WARN line on empty input."""
    import logging
    from mlframe.training.core._main_train_suite_phases import warn_on_empty_target_by_type

    caplog.set_level(logging.WARNING, logger="mlframe.training.core._main_train_suite_phases")
    warn_on_empty_target_by_type({})
    assert any("empty target_by_type" in rec.message for rec in caplog.records)


def test_w14a_warn_on_empty_target_by_type_silent_on_nonempty(caplog):
    """Non-empty target_by_type does NOT trigger the warning."""
    import logging
    from mlframe.training.core._main_train_suite_phases import warn_on_empty_target_by_type

    caplog.set_level(logging.WARNING, logger="mlframe.training.core._main_train_suite_phases")
    warn_on_empty_target_by_type({"binary": {"y": [0, 1, 0]}})
    assert not any("empty target_by_type" in rec.message for rec in caplog.records)


def test_w14a_check_precomputed_fingerprint_none_precomputed_returns_true():
    from mlframe.training.core._main_train_suite_phases import check_precomputed_fingerprint

    assert check_precomputed_fingerprint(None, None) is True


def test_w14a_maybe_apply_composite_target_specs_returns_false_on_empty():
    """Falsy precomputed.composite_target_specs returns False so caller runs inline discovery."""
    from mlframe.training.core._main_train_suite_phases import maybe_apply_composite_target_specs_precomputed

    pre = SimpleNamespace(composite_target_specs=None)
    assert maybe_apply_composite_target_specs_precomputed(True, pre, {}, verbose=False) is False
    pre2 = SimpleNamespace(composite_target_specs={})
    assert maybe_apply_composite_target_specs_precomputed(True, pre2, {}, verbose=False) is False


def test_w14a_maybe_apply_composite_target_specs_true_when_present():
    """Truthy precomputed.composite_target_specs returns True and deep-copies into metadata."""
    from mlframe.training.core._main_train_suite_phases import maybe_apply_composite_target_specs_precomputed

    pre = SimpleNamespace(composite_target_specs={"k": {"v": 1}})
    md = {}
    assert maybe_apply_composite_target_specs_precomputed(True, pre, md, verbose=False) is True
    assert md["composite_target_specs"] == {"k": {"v": 1}}
    # Deep-copy: mutating the caller's bundle does not bleed into metadata
    pre.composite_target_specs["k"]["v"] = 999
    assert md["composite_target_specs"]["k"]["v"] == 1
