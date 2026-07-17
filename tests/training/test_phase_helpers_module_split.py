"""Wave 105 (2026-05-21): split training/core/_phase_helpers.py
(1812 lines) into _phase_helpers.py (now 872 lines) + new
_phase_helpers_fit_split.py (976 lines).

Moved to the sibling file: three big phase functions
(_phase_auto_detect_feature_types, _phase_fit_pipeline,
_phase_train_val_test_split). Parent keeps the other phase
functions (_phase_global_outlier_detection,
_phase_pandas_conversion_and_cat_prep, _phase_load_and_preprocess,
_log_cardinality_and_drift_snapshot, _build_suite_common_params_dict,
_maybe_dispatch_to_ltr_ranker_suite) and the three NamedTuple
result types.
"""

from __future__ import annotations

from pathlib import Path


def test_moved_phase_functions_importable_from_facade() -> None:
    from mlframe.training.core._phase_helpers import (
        _phase_auto_detect_feature_types,
        _phase_fit_pipeline,
        _phase_train_val_test_split,
    )

    for fn in (_phase_auto_detect_feature_types, _phase_fit_pipeline, _phase_train_val_test_split):
        assert callable(fn), fn


def test_other_phase_helpers_still_importable() -> None:
    from mlframe.training.core._phase_helpers import (
        _phase_load_and_preprocess,
        _phase_pandas_conversion_and_cat_prep,
        _phase_global_outlier_detection,
        _build_suite_common_params_dict,
        _maybe_dispatch_to_ltr_ranker_suite,
    )

    for fn in (
        _phase_load_and_preprocess,
        _phase_pandas_conversion_and_cat_prep,
        _phase_global_outlier_detection,
        _build_suite_common_params_dict,
        _maybe_dispatch_to_ltr_ranker_suite,
    ):
        assert callable(fn), fn


def test_facade_below_1k_line_threshold() -> None:
    root = Path(__file__).resolve().parent.parent.parent / "src" / "mlframe" / "training" / "core"
    facade = root / "_phase_helpers.py"
    n = len(facade.read_text(encoding="utf-8").splitlines())
    assert n < 1000, f"_phase_helpers.py is {n} lines, still over the 1k threshold"


def test_sibling_owns_the_moved_functions() -> None:
    """Identity: facade and sibling expose the SAME function objects."""
    from mlframe.training.core import _phase_helpers, _phase_helpers_fit_split

    for name in (
        "_phase_auto_detect_feature_types",
        "_phase_fit_pipeline",
        "_phase_train_val_test_split",
    ):
        assert getattr(_phase_helpers, name) is getattr(_phase_helpers_fit_split, name), name
