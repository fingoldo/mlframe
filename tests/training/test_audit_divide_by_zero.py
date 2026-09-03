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
import pytest

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


def test_fast_r2_survives_a_fold_whose_weights_are_all_zero() -> None:
    """A zero total weight makes the weighted mean 0/0, and every ss_tot term follows it to NaN.

    Behavioural since 2026-09-04. This asserted that `if wsum <= 0.0:`, `ss_tots[j] = 0.0` and
    `continue` all appear in the module -- three fragments that can sit in three unrelated
    functions and say nothing about what the score does when handed such a fold. Sample weights
    that sum to zero arise from a fold where every row was filtered out by a recency or
    importance weighting, which is a data condition, not a programming error.
    """
    import numpy as np

    from mlframe.metrics.regression._regression_metrics import fast_r2_score

    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])

    score = fast_r2_score(y_true, y_pred, sample_weight=np.zeros(3))

    assert not np.isinf(np.asarray(score)).any(), f"zero-weight fold scored {score}"


def test_fast_r2_still_matches_sklearn_on_ordinary_input() -> None:
    """The guard must not have bought its safety by changing the ordinary answer."""
    import numpy as np

    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    from mlframe.metrics.regression._regression_metrics import fast_r2_score

    rng = np.random.default_rng(0)
    y_true = rng.normal(size=200)
    y_pred = y_true + rng.normal(scale=0.3, size=200)
    weights = rng.random(200)

    assert fast_r2_score(y_true, y_pred, sample_weight=weights) == pytest.approx(sklearn_metrics.r2_score(y_true, y_pred, sample_weight=weights))


def _woe_encode(train_cats, train_y, query):
    """Fit a WoE encoder with smoothing off and encode ``query``."""
    import numpy as _np

    from mlframe.training.feature_handling.target_encoders import LeakageSafeEncoder

    encoder = LeakageSafeEncoder(method="woe", smoothing=0.0, woe_smoothing=0.0)
    encoder.fit(_np.array(train_cats), _np.array(train_y, dtype=float))
    return encoder.transform(_np.array(query))


def test_woe_stays_finite_for_a_category_with_no_positives() -> None:
    """log(0) is -inf and the subtraction is nan; both reach the model as a feature value.

    Behavioural since 2026-09-04. This asserted that the two clip lines
    `p_safe = float(min(max(p, 1e-12), 1.0 - 1e-12))` and its q twin appear in the module. Two
    spellings of one cushion, silent about the number that reaches the caller, and passing just as
    well if the branch holding them were unreachable.

    With smoothing=0 a category whose train rows are all negative has p == 0 exactly, which is the
    input the cushion exists for.
    """
    import numpy as np

    out = _woe_encode(["a", "a", "b", "b"], [0.0, 0.0, 1.0, 1.0], ["a", "b"])

    assert np.all(np.isfinite(out)), f"WoE emitted {out} -- an infinite feature poisons every downstream fit"


def test_woe_still_orders_the_clipped_extremes_correctly() -> None:
    """Clipping must not flatten the two ends onto each other: an all-negative category still has
    to encode below an all-positive one, or the cushion has destroyed the signal it protects."""
    out = _woe_encode(["a", "a", "b", "b"], [0.0, 0.0, 1.0, 1.0], ["a", "b"])

    assert out[0] < out[1]


def test_woe_unseen_category_uses_the_prior_log_odds_not_zero() -> None:
    """0.0 is the neutral log-odds only for a balanced target. On a 99/1 split the true no-
    information point is log(99) ~= 4.6, so a 0.0 baseline was wildly biased toward the minority
    class for every test-time unseen string."""
    import numpy as np

    cats = ["a"] * 99 + ["b"]
    ys = [1.0] * 99 + [0.0]
    out = _woe_encode(cats, ys, ["never_seen_in_train"])

    assert np.isfinite(out[0])
    assert out[0] > 1.0, f"unseen encoded as {out[0]}, i.e. near the balanced-target neutral rather than the prior"


# test_calibration_quality_guards_empty_pit and test_mi_grok_guards_empty_data were removed on
# 2026-09-04 rather than rewritten. Each asserted a two-line source spelling of a guard that the
# behavioural sensors below already drive -- test_anderson_darling_empty_pit_returns_nan hands the
# function an empty array and reads the NaN back, and test_grok_mi_empty_data_returns_zero_matrix
# does the same for the MI matrix. Keeping both forms means the spelling breaks on a reindent while
# the twin goes on passing, which is churn with no added signal.


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
