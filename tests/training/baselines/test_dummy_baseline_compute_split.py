"""Wave 92 (2026-05-21): split _dummy_baseline_compute.py (1028 lines)
into 4 sibling files, keeping the original as a thin facade that
re-exports the moved symbols.

Pattern validated here for follow-up monolith splits:
  1. Move large self-contained functions to sibling files
     (`_dummy_baseline_regression.py`, `_classification.py`, `_quantile.py`).
  2. Each sibling lazy-imports back from the facade inside its function
     body so circular-load at module-load time is impossible.
  3. The facade top-level imports the moved symbols so every existing
     `from ._dummy_baseline_compute import X` import still resolves.

After wave 92:
  _dummy_baseline_compute.py:        1028 -> 648 (under the 1k threshold)
  _dummy_baseline_regression.py:     new, 202 lines
  _dummy_baseline_classification.py: new, 139 lines
  _dummy_baseline_quantile.py:       new, 113 lines
"""
from __future__ import annotations

from pathlib import Path


def test_all_public_symbols_still_importable_from_facade() -> None:
    """Backward-compat: every name previously importable from
    `_dummy_baseline_compute` is still there."""
    from mlframe.training.baselines._dummy_baseline_compute import (
        _per_target_seed,
        _to_pandas_for_baseline,
        _pick_per_group_categorical,
        _is_polars_frame,
        _per_group_predict_polars,
        _per_group_predict,
        _safe_metric,
        _compute_regression_baselines,
        _compute_classification_baselines,
        _compute_quantile_baselines,
        compute_dummy_baselines,
    )
    for fn in (
        _per_target_seed,
        _to_pandas_for_baseline,
        _pick_per_group_categorical,
        _is_polars_frame,
        _per_group_predict_polars,
        _per_group_predict,
        _safe_metric,
        _compute_regression_baselines,
        _compute_classification_baselines,
        _compute_quantile_baselines,
        compute_dummy_baselines,
    ):
        assert callable(fn), fn


def test_sibling_files_exist_and_define_the_moved_symbol() -> None:
    """Each sibling file owns one of the moved symbols (the facade just re-exports)."""
    from mlframe.training.baselines import (
        _dummy_baseline_regression,
        _dummy_baseline_classification,
        _dummy_baseline_quantile,
    )
    assert hasattr(_dummy_baseline_regression, "_compute_regression_baselines")
    assert hasattr(_dummy_baseline_classification, "_compute_classification_baselines")
    assert hasattr(_dummy_baseline_quantile, "_compute_quantile_baselines")


def test_facade_below_1k_line_threshold() -> None:
    """The user's directive: monoliths >1k lines must be split.
    After wave 92, _dummy_baseline_compute.py is below the threshold."""
    root = Path(__file__).resolve().parents[3] / "src" / "mlframe" / "training" / "baselines"
    facade = root / "_dummy_baseline_compute.py"
    assert facade.exists()
    n = len(facade.read_text(encoding="utf-8").splitlines())
    assert n < 1000, f"_dummy_baseline_compute.py is {n} lines, still over the 1k threshold"


def test_per_target_seed_still_deterministic_after_split() -> None:
    """Sanity: import via the facade and exercise one of the helpers
    that the moved sub-modules lazy-import back."""
    from mlframe.training.baselines._dummy_baseline_compute import _per_target_seed
    a = _per_target_seed(42, "revenue")
    b = _per_target_seed(42, "revenue")
    c = _per_target_seed(42, "churn")
    assert a == b, "seed must be deterministic per (base_seed, target_name)"
    assert a != c, "different target name must yield different seed"


def test_compute_quantile_baselines_round_trips_via_facade() -> None:
    """Quick functional check: the moved _compute_quantile_baselines
    behaves the same when called via the facade re-export."""
    import numpy as np
    from mlframe.training.baselines._dummy_baseline_compute import _compute_quantile_baselines

    class _Config:
        pass

    train_y = np.linspace(0.0, 1.0, 101)
    val_y = train_y.copy()
    test_y = train_y.copy()
    val_preds, test_preds, extras = _compute_quantile_baselines(
        target_name="t",
        train_y=train_y,
        val_y=val_y,
        test_y=test_y,
        alphas=[0.25, 0.5, 0.75],
        config=_Config(),
    )
    assert "quantile_alpha_0.250" in val_preds
    assert "median_for_all" in val_preds
    # Predictions are (N, K) with K=3 alphas.
    assert val_preds["median_for_all"].shape == (101, 3)
    assert test_preds["median_for_all"].shape == (101, 3)
