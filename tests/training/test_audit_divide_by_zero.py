"""Wave 47 (2026-05-20): divide-by-zero in metric / feature / kernel kernels.

Audit class: a / b / 1/x / sqrt(a/b) where b can be zero on realistic inputs
(all-zero sample_weight, empty post-filter slice, smoothing=0 + zero-positive
category, user-supplied zero temperature). Either crashes the njit kernel
(ZeroDivisionError) or silently propagates NaN.

4 P1 fixes + 5 P2 fixes:

  P1:
    1. feature_engineering/numerical.py:475 (weighted_arithmetic_mean / sum_weights)
    2. feature_engineering/numerical.py:493 (sqrt(weighted_quadratic_mean / sum_weights))
    3. feature_engineering/numerical.py:1031 (sqrt(weighted_std / sum_weights) + weighted_mad)
    4. metrics/core.py:3900 (fast_r2_score variance-weighted multioutput, wmean / wsum)

  P2:
    5. training/feature_handling/target_encoders.py:707 (WoE log(p) - log(q) clip)
    6. calibration/quality.py:405 (anderson_darling_statistic (1/n) on empty PIT)
    7. feature_selection/mi.py:90 (grok_compute_mutual_information 1/n_samples on empty data)
    8. feature_selection/filters/info_theory.py:353 (sibling njit inv_n)
    9. feature_selection/filters/batch_pair_mi_gpu.py:365 (sibling host-side inv_n; CUDA kernel inv_n at :194 protected via the host guard)
    10. feature_engineering/transformer/_kernels_njit.py:189 (row_attention softmax_temp inv guard)
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

MLFRAME_ROOT = Path(importlib.import_module("mlframe").__file__).parent


def _read(rel: str) -> str:
    # info_theory.py was carved into the ``info_theory/`` subpackage; the
    # empty-factors guard now lives in a submodule (``_batch_kernels.py``).
    # Concat every submodule so the source-grep sensor matches the relocated
    # guard regardless of which submodule owns it now.
    """Read."""
    pkg_dir = MLFRAME_ROOT / "feature_selection" / "filters" / "info_theory"
    if rel == "feature_selection/filters/info_theory.py" and pkg_dir.is_dir():
        return "\n".join(p.read_text(encoding="utf-8") for p in sorted(pkg_dir.glob("*.py")))
    return (MLFRAME_ROOT / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Source-level sensors
# ---------------------------------------------------------------------------


def test_numerical_weighted_arithmetic_mean_guards_zero_sum() -> None:
    """The numba kernels for weighted_arithmetic_mean / quadratic / std moved
    into the sibling _numerical_numba.py during the 2026-05-21 monolith
    split (numerical.py re-exports via from ._numerical_numba import ...)."""
    src = _read("feature_engineering/numerical.py") + "\n" + _read("feature_engineering/_numerical_numba.py")
    assert "if sum_weights == 0.0:\n            weighted_arithmetic_mean = np.nan" in src
    # Two more sites in the same kernel family (quadratic + std) must also
    # have the guard.
    assert src.count("if sum_weights == 0.0:") >= 4


def test_metrics_fast_r2_guards_zero_wsum() -> None:
    # ``fast_r2_score`` was moved to ``_regression_metrics.py`` when
    # ``metrics/core.py`` was split into siblings.
    """Metrics fast r2 guards zero wsum."""
    src = _read("metrics/regression/_regression_metrics.py")
    # The fix introduces an explicit `if wsum <= 0.0: ss_tots[j] = 0.0; continue`.
    assert "if wsum <= 0.0:" in src and "ss_tots[j] = 0.0" in src and "continue" in src


def test_target_encoders_woe_clips_p_and_q() -> None:
    """Target encoders woe clips p and q."""
    src = _read("training/feature_handling/target_encoders.py")
    # The fix clips p, q with the same Laplace cushion (1e-12) the kfold path uses.
    assert "p_safe = float(min(max(p, 1e-12), 1.0 - 1e-12))" in src
    assert "q_safe = float(min(max(q, 1e-12), 1.0 - 1e-12))" in src


def test_calibration_quality_guards_empty_pit() -> None:
    """Calibration quality guards empty pit."""
    src = _read("calibration/quality.py")
    # The fix early-returns nan on empty PIT.
    assert "if n == 0:" in src and 'return float("nan")' in src


def test_mi_grok_guards_empty_data() -> None:
    """Mi grok guards empty data."""
    src = _read("feature_selection/mi.py")
    # The fix returns the zero mi_results early.
    assert "if n_samples == 0:\n        return mi_results" in src


def test_info_theory_guards_empty_factors_data() -> None:
    """Info theory guards empty factors data."""
    src = _read("feature_selection/filters/info_theory.py")
    assert "if n_samples == 0:\n        out[:] = 0.0\n        return out" in src


def test_batch_pair_mi_gpu_host_guards_empty() -> None:
    """Batch pair mi gpu host guards empty."""
    src = _read("feature_selection/filters/batch_pair_mi_gpu.py")
    assert "if n_samples == 0:\n        return np.zeros(n_pairs, dtype=np.float64)" in src


def test_kernels_njit_softmax_temp_guarded() -> None:
    """Kernels njit softmax temp guarded."""
    src = _read("feature_engineering/transformer/_kernels_njit.py")
    # The fix mirrors the sibling kernel's pattern: temp > eps else 1.0.
    assert "1.0 / softmax_temp if softmax_temp > 1e-12 else 1.0" in src


# ---------------------------------------------------------------------------
# Behavioural sensors: trigger the divide-by-zero path and assert no crash.
# ---------------------------------------------------------------------------


def test_anderson_darling_empty_pit_returns_nan() -> None:
    """Anderson darling empty pit returns nan."""
    from mlframe.calibration.quality import anderson_darling_statistic

    result = anderson_darling_statistic(np.array([], dtype=np.float64))
    assert np.isnan(result)


def test_grok_mi_empty_data_returns_zero_matrix() -> None:
    """Grok mi empty data returns zero matrix."""
    from mlframe.feature_selection.mi import grok_compute_mutual_information

    empty = np.empty((0, 3), dtype=np.int8)
    out = grok_compute_mutual_information(
        data=empty,
        target_indices=[0],
        n_bins=15,
    )
    assert out.shape == (1, 3)
    np.testing.assert_array_equal(out, np.zeros_like(out))


def test_batch_pair_mi_gpu_host_module_loads() -> None:
    """The host-side guard is asserted source-level above; this confirms the
    module imports cleanly (no syntax regression from the fix edit)."""
    import importlib

    mod = importlib.import_module("mlframe.feature_selection.filters.batch_pair_mi_gpu")
    assert mod is not None
