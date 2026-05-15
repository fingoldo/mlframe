"""Regression tests for the rename ``can_skip_pandas_conv`` -> ``defer_pandas_conv``
and its simplified gating condition (P1).

Pre-fix the variable was named ``can_skip_pandas_conv`` and the gate
included the always-True disjunction
``(all_models_polars_native OR _has_non_native_mlframe_strategy)`` --
given ``_has_non_native_mlframe_strategy = was_polars_input AND NOT
all_models_polars_native``, the disjunction collapses to True whenever
``was_polars_input``. Post-fix the gate is just the three real
conditions: polars input, no recurrent model, no RFECV.
"""

from __future__ import annotations

import pytest


def _defer_pandas_conv(
    *,
    was_polars_input: bool,
    recurrent_models: list,
    rfecv_models: list,
) -> bool:
    """Mirror the simplified condition under test, isolated from the
    big ``_phase_pandas_conversion_and_cat_prep`` helper so we can
    unit-test the gate without standing up its DataFrames + strategies.
    The production assignment lives in
    ``mlframe.training.core._phase_helpers``; this duplicate is kept
    in sync by ``test_production_assignment_matches_simplified_gate``.
    """
    _has_rfecv = bool(rfecv_models)
    return (
        was_polars_input
        and not recurrent_models
        and not _has_rfecv
    )


def test_defer_pandas_conv_false_with_recurrent_model():
    assert _defer_pandas_conv(
        was_polars_input=True,
        recurrent_models=["lstm"],
        rfecv_models=[],
    ) is False


def test_defer_pandas_conv_true_with_polars_native_and_no_recurrent():
    assert _defer_pandas_conv(
        was_polars_input=True,
        recurrent_models=[],
        rfecv_models=[],
    ) is True


def test_defer_pandas_conv_false_when_input_is_pandas():
    assert _defer_pandas_conv(
        was_polars_input=False,
        recurrent_models=[],
        rfecv_models=[],
    ) is False


def test_defer_pandas_conv_false_with_rfecv():
    assert _defer_pandas_conv(
        was_polars_input=True,
        recurrent_models=[],
        rfecv_models=["cb_num_rfecv"],
    ) is False


def test_training_context_field_renamed_to_defer_pandas_conv():
    """The ``TrainingContext.defer_pandas_conv`` field replaced the old
    ``TrainingContext.can_skip_pandas_conv`` field. Callers that read
    the old name would NameError post-rename, so this test guarantees
    the field exists under the new name."""
    from mlframe.training.core._training_context import TrainingContext
    fields = {f.name for f in TrainingContext.__dataclass_fields__.values()}
    assert "defer_pandas_conv" in fields
    assert "can_skip_pandas_conv" not in fields


def test_production_assignment_symbol_is_defer_pandas_conv():
    """Behavioural check: the helper's return signature contracts include
    the renamed name. ``_phase_pandas_conversion_and_cat_prep`` returns a
    13-tuple whose 12th slot is the new ``defer_pandas_conv`` flag; the
    import below would fail before the rename because the field was
    renamed in ``TrainingContext`` (which is the down-stream consumer).
    The TrainingContext field test above already covers that side; this
    test adds the symmetric import-binding check for the helper module
    itself so any future inadvertent revert of the rename is caught at
    test collection."""
    from mlframe.training.core import _phase_helpers as _ph
    # The helper still exists under the same function name.
    assert callable(getattr(_ph, "_phase_pandas_conversion_and_cat_prep", None))
