"""Wave-21 sensor cluster: mlframe.utils.nan_safe helper + 11 production
call sites migrated to it.

Audit wave 21 found a missing central nan-safe helper across the
codebase: ~200 raw `np.argmax` / `np.median` / `np.percentile` /
`np.quantile` calls vs ~32 `nan*` variants. Each non-nan call was a
potential silent-NaN-propagation bug.

This sensor pins both:
1. The helper's contract (``argmax_classes_safe`` / ``quantile_safe`` /
   ``median_safe`` in ``mlframe.utils.nan_safe``).
2. The 11 production sites that now use the helper or the nan-aware
   variants (training/core/predict.py x2, training/_reporting.py,
   evaluation/reports.py, reporting/charts/multiclass.py x2,
   reporting/charts/ltr.py, metrics/core.py, models/ensembling.py x4,
   feature_selection/general.py, feature_engineering/numerical.py,
   training/_classif_helpers.py, calibration/quality.py x2).
"""
from __future__ import annotations

import logging

import numpy as np
import pytest


# ---- mlframe.utils.nan_safe contract -----------------------------------


def test_argmax_classes_safe_all_finite():
    from mlframe.utils.nan_safe import argmax_classes_safe
    p = np.array([[0.1, 0.7, 0.2], [0.4, 0.3, 0.3]])
    out = argmax_classes_safe(p, context="test")
    np.testing.assert_array_equal(out, [1, 0])


def test_argmax_classes_safe_with_all_nan_row_uses_fallback(caplog):
    from mlframe.utils.nan_safe import argmax_classes_safe
    p = np.array([
        [0.1, 0.7, 0.2],
        [np.nan, np.nan, np.nan],
        [0.5, 0.4, 0.3],
    ])
    with caplog.at_level(logging.WARNING, logger="mlframe.utils.nan_safe"):
        out = argmax_classes_safe(p, fallback_class=99, context="test")
    np.testing.assert_array_equal(out, [1, 99, 0])
    assert any("1/3 rows contain NO finite probabilities" in r.message
               for r in caplog.records)


def test_argmax_classes_safe_mixed_finite_nan_row():
    """A row with SOME finite entries uses nanargmax (picks max-finite)."""
    from mlframe.utils.nan_safe import argmax_classes_safe
    p = np.array([
        [0.1, np.nan, 0.5],
        [np.nan, 0.7, 0.2],
    ])
    out = argmax_classes_safe(p, context="test")
    np.testing.assert_array_equal(out, [2, 1])


def test_argmax_classes_safe_1d_array():
    from mlframe.utils.nan_safe import argmax_classes_safe
    p = np.array([0.1, 0.7, 0.2])
    out = argmax_classes_safe(p, context="test")
    assert int(out) == 1


def test_argmax_classes_safe_1d_all_nan(caplog):
    from mlframe.utils.nan_safe import argmax_classes_safe
    p = np.array([np.nan, np.nan, np.nan])
    with caplog.at_level(logging.WARNING, logger="mlframe.utils.nan_safe"):
        out = argmax_classes_safe(p, fallback_class=7, context="test")
    assert int(out) == 7


def test_quantile_safe_finite_input():
    from mlframe.utils.nan_safe import quantile_safe
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert quantile_safe(arr, 0.5) == 3.0


def test_quantile_safe_with_nan_input():
    from mlframe.utils.nan_safe import quantile_safe
    arr = np.array([1.0, 2.0, np.nan, 4.0])
    # nanquantile interpolation: median of [1,2,4] is 2.0.
    assert quantile_safe(arr, 0.5) == 2.0


def test_quantile_safe_all_nan_returns_fallback(caplog):
    from mlframe.utils.nan_safe import quantile_safe
    arr = np.array([np.nan, np.nan])
    with caplog.at_level(logging.WARNING, logger="mlframe.utils.nan_safe"):
        out = quantile_safe(arr, 0.5, fallback=-1.0)
    assert out == -1.0


def test_quantile_safe_q_sequence():
    from mlframe.utils.nan_safe import quantile_safe
    arr = np.array([1.0, 2.0, 3.0, 4.0])
    out = quantile_safe(arr, [0.25, 0.5, 0.75])
    np.testing.assert_array_almost_equal(out, [1.75, 2.5, 3.25])


def test_median_safe_finite():
    from mlframe.utils.nan_safe import median_safe
    assert median_safe(np.array([1.0, 2.0, 3.0])) == 2.0


def test_median_safe_with_nan():
    from mlframe.utils.nan_safe import median_safe
    assert median_safe(np.array([1.0, np.nan, 3.0])) == 2.0


def test_median_safe_all_nan_fallback(caplog):
    from mlframe.utils.nan_safe import median_safe
    with caplog.at_level(logging.WARNING, logger="mlframe.utils.nan_safe"):
        out = median_safe(np.array([np.nan, np.nan]), fallback=0.0)
    assert out == 0.0


# ---- Production-site source-level guards (11 migrations) ----------------


@pytest.mark.parametrize("rel,must_contain", [
    # training/core/predict.py - two argmax sites migrated:
    ("training/core/predict.py",
     "argmax_classes_safe"),
    # training/_reporting_probabilistic.py - one site (moved out of _reporting.py
    # during the probabilistic-report monolith split):
    ("training/_reporting_probabilistic.py",
     "argmax_classes_safe"),
    # evaluation/reports.py - one site:
    ("evaluation/reports.py",
     "argmax_classes_safe"),
    # reporting/charts/multiclass.py - two sites use replace_all:
    ("reporting/charts/multiclass.py",
     "argmax_classes_safe"),
    # reporting/charts/ltr.py uses inline nanargmax pattern:
    ("reporting/charts/ltr.py",
     "np.nanargmax(scores_q)"),
    # ``compute_fairness_metrics`` moved from ``metrics/core.py`` to the
    # ``_fairness_metrics.py`` sibling when ``core.py`` was split below 1k LOC.
    ("metrics/_fairness_metrics.py",
     "np.nanquantile(performances"),
    # models/ensembling.py:538/540/542/575/576 -> nan* variants:
    ("models/ensembling.py",
     "np.nanmedian(per_member_mae)"),
    ("models/ensembling.py",
     "np.nanmedian(per_member_std)"),
    # feature_selection/general.py:200/202 -> nanmax/nanquantile:
    ("feature_selection/general.py",
     "np.nanquantile(all_permuted_mis"),
    # feature_engineering/numerical.py:694 -> nanquantile:
    ("feature_engineering/numerical.py",
     "np.nanquantile(arr, q"),
    # training/_classif_helpers.py:135 -> nan-safe argmax:
    ("training/_classif_helpers.py",
     "np.nanargmax(arr, axis=1)"),
    # calibration/quality.py:200,201 -> nanmean:
    ("calibration/quality.py",
     "np.nanmean(y_pred[indices[l:r]])"),
    ("calibration/quality.py",
     "np.nanmean(y_true[indices[l:r]])"),
])
def test_wave21_production_site_migrated(rel, must_contain):
    """Source-level guard that each of the 13 migration sites now contains
    the post-fix idiom (the helper call or the nan-aware variant)."""
    import pathlib
    import mlframe as _mlframe
    src = (pathlib.Path(_mlframe.__file__).resolve().parent / rel).read_text(encoding="utf-8")
    assert must_contain in src, (
        f"Wave 21 P1/P2 regression: {rel} no longer contains {must_contain!r}. "
        f"Pre-fix raw np.argmax/np.quantile/np.median over potentially-NaN "
        f"input is the bug class."
    )
