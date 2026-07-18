"""Sensor: TrainingSplitConfig fields that are caller-side only must
be excluded from the ``model_dump()`` passed to ``make_train_test_split``.

Prod TVT 2026-05-25: ``composite_cardinality_cap`` shipped without
being added to the exclude set in ``_phase_helpers_fit_split.py``, so
the splitter raised
``TypeError: make_train_test_split() got an unexpected keyword
argument 'composite_cardinality_cap'`` on a real run, killing the
entire suite at phase 2.

This sensor enumerates the documented caller-side-only fields and
verifies the production exclude clause covers each one. Adding a new
caller-side field means updating BOTH this list AND the exclude set
in lockstep, so the test fails early if a future contributor forgets
one side.
"""

from __future__ import annotations

from pathlib import Path

import inspect

# Fields on TrainingSplitConfig that are CALLER-SIDE behaviour knobs:
# they configure surrounding logic but are NOT make_train_test_split
# kwargs. Adding to this list requires updating
# _phase_helpers_fit_split.py's exclude={} set on the model_dump call.
# ``calib_size`` is NOT here: the splitter now consumes it directly (carves a disjoint calib slice via
# return_calib=True), so it is a legitimate splitter kwarg, not a caller-side-only field.
_CALLER_SIDE_FIELDS = (
    "use_groups",  # derived into _groups upstream
    "composite_cardinality_cap",  # bucket-stratify gate (consumed pre-call)
)


def test_phase_helpers_fit_split_uses_signature_derived_filter() -> None:
    """The model_dump filter MUST be derived from the splitter's signature
    at runtime, not a hardcoded list. Hardcoded lists drift: prod TVT
    2026-05-25 surfaced TWO consecutive TypeErrors when caller-side fields
    were added without exclude updates. The runtime-signature filter
    catches any future field addition automatically."""
    import mlframe.training.core._phase_helpers_fit_split as ph

    src = Path(ph.__file__).read_text(encoding="utf-8")
    assert (
        "inspect.signature(make_train_test_split).parameters" in src
    ), "phase_helpers_fit_split must inspect the splitter signature at runtime to filter the model_dump kwargs; hardcoded exclude lists drift out of sync."


def test_phase_helpers_fit_split_filter_drops_all_caller_side_fields() -> None:
    """End-to-end: simulate the suite filter behaviour and verify each
    documented caller-side field gets dropped before the splitter call."""
    import inspect
    from mlframe.training._preprocessing_configs import TrainingSplitConfig
    from mlframe.training.splitting import make_train_test_split

    cfg = TrainingSplitConfig()
    splitter_kwargs = set(inspect.signature(make_train_test_split).parameters)
    explicit = {"df", "timestamps", "stratify_y", "groups"}
    filtered = {k: v for k, v in cfg.model_dump().items() if k in splitter_kwargs and k not in explicit}
    for field in _CALLER_SIDE_FIELDS:
        assert field not in filtered, (
            f"Caller-side field {field!r} survived the signature-derived "
            f"filter; check that make_train_test_split.signature doesn't "
            f"accept it and TrainingSplitConfig still declares it."
        )


def test_make_train_test_split_signature_does_not_accept_caller_side_fields() -> None:
    """Pin: the splitter's signature must NOT accept any of the
    documented caller-side fields. If a future change ADDS a field
    to the splitter (legitimately consuming it), this test fails on
    purpose so the maintainer revisits the exclude set."""
    from mlframe.training.splitting import make_train_test_split

    params = inspect.signature(make_train_test_split).parameters
    for field in _CALLER_SIDE_FIELDS:
        assert field not in params, (
            f"make_train_test_split signature now accepts {field!r}; "
            f"either the field is no longer caller-side-only (remove "
            f"from _CALLER_SIDE_FIELDS in this test AND from the "
            f"exclude set in _phase_helpers_fit_split.py), or the "
            f"signature edit was unintentional."
        )


def test_training_split_config_has_each_caller_side_field() -> None:
    """Defense: the caller-side field list itself stays in sync with
    the config. A typo'd field name would silently bypass the sensor
    above (a field that doesn't exist is trivially 'excluded')."""
    from mlframe.training._preprocessing_configs import TrainingSplitConfig

    fields = getattr(TrainingSplitConfig, "model_fields", {})
    for field in _CALLER_SIDE_FIELDS:
        assert (
            field in fields
        ), f"TrainingSplitConfig has no field {field!r}; the sensor won't catch real regressions. Rename or remove from _CALLER_SIDE_FIELDS."
