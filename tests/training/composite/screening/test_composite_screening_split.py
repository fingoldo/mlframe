"""Wave 98 (2026-05-21): split training/composite_screening.py
(1160 lines) into composite_screening.py (now 539 lines) + new
_composite_screening_tiny.py (667 lines).

Moved to the sibling file: the tiny-model RMSE / CV helpers
(_silence_tiny_model_output, _build_tiny_model, _tiny_cv_rmse_raw_y,
_tiny_cv_rmse_y_scale_multiseed, _tiny_cv_rmse_raw_y_multiseed,
_per_bin_rmse, _tiny_cv_rmse_y_scale).

Original re-exports all 7 so existing
``from mlframe.training.composite.discovery.screening import _build_tiny_model``
imports continue to work.
"""

from __future__ import annotations

from pathlib import Path


def test_tiny_model_symbols_still_importable_from_facade() -> None:
    from mlframe.training.composite.discovery.screening import (
        _silence_tiny_model_output,
        _build_tiny_model,
        _tiny_cv_rmse_raw_y,
        _tiny_cv_rmse_y_scale_multiseed,
        _tiny_cv_rmse_raw_y_multiseed,
        _per_bin_rmse,
        _tiny_cv_rmse_y_scale,
    )

    for fn in (
        _silence_tiny_model_output,
        _build_tiny_model,
        _tiny_cv_rmse_raw_y,
        _tiny_cv_rmse_y_scale_multiseed,
        _tiny_cv_rmse_raw_y_multiseed,
        _per_bin_rmse,
        _tiny_cv_rmse_y_scale,
    ):
        assert callable(fn), fn


def test_mi_symbols_still_in_parent_module() -> None:
    """The MI / correlation helpers stay in the original file."""
    from mlframe.training.composite.discovery.screening import (
        _is_polars_df,
        _extract_column_array,
        _is_numeric_column,
        _safe_corr,
        _safe_abs_corr_all,
        _residualise,
        _mi_pair_bin,
        _prebin_feature_columns,
        _mi_to_target_prebinned,
        _mi_from_binned_pair,
        _mi_per_feature_y_fixed,
        _mi_to_target,
        _sample_indices,
    )

    for fn in (
        _is_polars_df,
        _extract_column_array,
        _is_numeric_column,
        _safe_corr,
        _safe_abs_corr_all,
        _residualise,
        _mi_pair_bin,
        _prebin_feature_columns,
        _mi_to_target_prebinned,
        _mi_from_binned_pair,
        _mi_per_feature_y_fixed,
        _mi_to_target,
        _sample_indices,
    ):
        assert callable(fn), fn


def test_facade_below_1k_line_threshold() -> None:
    root = Path(__file__).resolve().parents[4] / "src" / "mlframe" / "training" / "composite" / "discovery"
    facade = root / "screening.py"
    n = len(facade.read_text(encoding="utf-8").splitlines())
    assert n < 1000, f"screening.py is {n} lines, still over the 1k threshold"


def test_sibling_module_owns_the_moved_symbols() -> None:
    """Identity: facade and sibling module expose the SAME objects."""
    from mlframe.training.composite.discovery import screening as cs
    from mlframe.training.composite.discovery import _screening_tiny as tiny

    for name in (
        "_silence_tiny_model_output",
        "_build_tiny_model",
        "_tiny_cv_rmse_raw_y",
        "_tiny_cv_rmse_y_scale_multiseed",
        "_tiny_cv_rmse_raw_y_multiseed",
        "_per_bin_rmse",
        "_tiny_cv_rmse_y_scale",
    ):
        assert getattr(cs, name) is getattr(tiny, name), name


def test_silence_context_manager_works() -> None:
    """Functional smoke: the @contextlib.contextmanager decorator
    survived the split."""
    from mlframe.training.composite.discovery.screening import _silence_tiny_model_output

    with _silence_tiny_model_output():
        # Just exercising the with-block exit path.
        pass
