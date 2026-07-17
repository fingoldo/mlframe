"""Wave 107 (2026-05-21): split feature_engineering/numerical.py
(1597 lines) into numerical.py (now 865 lines) + new
_numerical_numba.py (785 lines).

Moved: compute_numerical_aggregates_numba (~348 lines) +
_make_compute_moments_slope_mi / compute_moments_slope_mi (~388 lines).
NUMBA_NJIT_PARAMS mirrored on the sibling; _EMPTY_FLOAT32 imported
back so the parent's compute_mutual_info_regression default arg
resolves.
"""

from __future__ import annotations

from pathlib import Path


def test_moved_symbols_still_importable() -> None:
    from mlframe.feature_engineering.numerical import (
        compute_numerical_aggregates_numba,
        _make_compute_moments_slope_mi,
        compute_moments_slope_mi,
    )

    for fn in (
        compute_numerical_aggregates_numba,
        _make_compute_moments_slope_mi,
        compute_moments_slope_mi,
    ):
        assert callable(fn), fn


def test_other_numerical_api_still_importable() -> None:
    from mlframe.feature_engineering.numerical import (
        compute_numaggs,
        get_numaggs_names,
        compute_simple_stats_numba,
        compute_simple_stats_numba_arr,
        get_simple_stats_names,
        compute_distributional_features,
        compute_entropy_features,
        compute_nunique_modes_quantiles_numpy,
        rolling_moving_average,
    )

    for fn in (
        compute_numaggs,
        get_numaggs_names,
        compute_simple_stats_numba,
        compute_simple_stats_numba_arr,
        get_simple_stats_names,
        compute_distributional_features,
        compute_entropy_features,
        compute_nunique_modes_quantiles_numpy,
        rolling_moving_average,
    ):
        assert callable(fn), fn


def test_facade_below_1k_line_threshold() -> None:
    root = Path(__file__).resolve().parent.parent.parent / "src" / "mlframe" / "feature_engineering"
    facade = root / "numerical.py"
    n = len(facade.read_text(encoding="utf-8").splitlines())
    assert n < 1000, f"numerical.py is {n} lines, still over the 1k threshold"


def test_sibling_module_identity() -> None:
    from mlframe.feature_engineering import numerical, _numerical_numba

    assert numerical.compute_numerical_aggregates_numba is _numerical_numba.compute_numerical_aggregates_numba
    assert numerical.compute_moments_slope_mi is _numerical_numba.compute_moments_slope_mi
