"""Wave 91 (2026-05-21): NaN-safe argmax at the 4 per-target + suite-wide
ensemble sites in predict.py.

Pre-fix, ``_combine_probs`` (RRF / geomean / harmonic / quadratic) could
emit NaN rows when an underlying member's row was NaN; the downstream
``np.argmax(_combined, axis=1)`` silently routed those rows to class 0,
poisoning the confusion matrix / per-class P/R/F1. The fix swaps each
plain np.argmax for ``argmax_classes_safe`` (already used at the per-model
sites in the same module).

This sensor pins:
  1. ``np.argmax(_combined|avg_probs, ...)`` no longer appears in predict.py.
  2. Each of the 4 fixed sites now resolves to ``argmax_classes_safe``.
  3. ``argmax_classes_safe`` handles an all-NaN row by falling back to
     ``fallback_class`` (default 0) AND emits a logged warning, so the
     misclassification is visible instead of silent.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np


def test_no_bare_argmax_on_combined_or_avg_probs() -> None:
    src_path = (
        Path(__file__).resolve().parent.parent.parent
        / "src" / "mlframe" / "training" / "core" / "predict.py"
    )
    src = src_path.read_text(encoding="utf-8")
    # The 4 audit-flagged sites (per-target + suite-wide in both predict
    # entries) must NOT use np.argmax(_combined, ...) or np.argmax(avg_probs, ...).
    assert "np.argmax(_combined" not in src, (
        "bare np.argmax(_combined) found in predict.py -- one of the 4 "
        "wave-91 audit sites regressed; use argmax_classes_safe instead."
    )
    assert "np.argmax(avg_probs" not in src, (
        "bare np.argmax(avg_probs) found in predict.py -- one of the 4 "
        "wave-91 audit sites regressed; use argmax_classes_safe instead."
    )


def test_argmax_classes_safe_imports_present() -> None:
    """The 4 fixed sites each import argmax_classes_safe lazily; together
    with the 2 pre-existing per-model sites that's 6 lazy imports total."""
    src_path = (
        Path(__file__).resolve().parent.parent.parent
        / "src" / "mlframe" / "training" / "core" / "predict.py"
    )
    src = src_path.read_text(encoding="utf-8")
    n_imports = src.count("from ...utils.nan_safe import argmax_classes_safe")
    # Wave-21 added 2 per-model sites; wave-91 adds 4 ensemble sites = 6 total.
    assert n_imports >= 6, (
        f"expected >= 6 argmax_classes_safe imports (2 per-model + 4 "
        f"ensemble); found {n_imports}"
    )


def test_argmax_classes_safe_all_nan_row_falls_back_to_zero(caplog) -> None:
    """Functional smoke: feed an all-NaN row through argmax_classes_safe and
    confirm it lands on class 0 (the configured fallback) WITH a logged
    warning -- that's the contract the wave-91 swap relies on."""
    from mlframe.utils.nan_safe import argmax_classes_safe

    probs = np.array([
        [0.1, 0.7, 0.2],
        [np.nan, np.nan, np.nan],
        [0.6, 0.3, 0.1],
    ])
    with caplog.at_level(logging.WARNING, logger="mlframe.utils.nan_safe"):
        preds = argmax_classes_safe(probs, context="wave91_sensor")
    assert preds.shape == (3,)
    assert int(preds[0]) == 1  # argmax of [0.1, 0.7, 0.2]
    assert int(preds[1]) == 0  # all-NaN -> fallback_class=0
    assert int(preds[2]) == 0  # argmax of [0.6, 0.3, 0.1]
    # The NaN row must be visible in the log so silent misclassification is impossible.
    assert any("wave91_sensor" in rec.message for rec in caplog.records), (
        "argmax_classes_safe must log a warning when falling back on NaN rows"
    )
