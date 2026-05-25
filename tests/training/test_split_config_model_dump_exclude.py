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
_CALLER_SIDE_FIELDS = (
    "use_groups",  # derived into _groups upstream
    "calib_size",  # downstream post-train carve
    "composite_cardinality_cap",  # bucket-stratify gate (consumed pre-call)
)


def test_phase_helpers_fit_split_excludes_caller_side_fields() -> None:
    import mlframe.training.core._phase_helpers_fit_split as ph
    src = Path(ph.__file__).read_text(encoding="utf-8")
    # Locate the exclude clause via a substring sensor. We don't AST-
    # parse because the exclude set may be on a single line or split
    # across lines; the substring test catches the common shapes.
    for field in _CALLER_SIDE_FIELDS:
        assert f'"{field}"' in src or f"'{field}'" in src, (
            f"TrainingSplitConfig caller-side field {field!r} not "
            f"excluded in _phase_helpers_fit_split.py. Add it to the "
            f"exclude={{}} set on the model_dump() call so the "
            f"splitter doesn't TypeError on the kwarg."
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
        assert field in fields, (
            f"TrainingSplitConfig has no field {field!r}; the sensor "
            f"won't catch real regressions. Rename or remove from "
            f"_CALLER_SIDE_FIELDS."
        )
