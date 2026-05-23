"""Regression test for the (bool, float64) prewarm of parallel reduction kernels.

c0023 iter190 profile (multilabel cb+linear+xgb, 200k) attributed 4.156s of
``_compile_for_args`` to ``fast_brier_score_loss`` across 2 fresh compiles --
the (bool, float64) signature emitted by multilabel per-class loops
(``y_true = targets == class_name`` -> ndarray[bool]) was NOT covered by the
existing prewarm in ``mlframe.metrics.core._prewarm_numba_cache_body``.

Adding the prewarm pays the same 4s upfront at import time but moves it OUT
of the first-fit hot path. This test verifies the (bool, f64) parallel
variants are warm AFTER prewarm by asserting they execute in well under the
fresh-compile time bound.
"""
import time

import numpy as np
import pytest


def _ensure_prewarmed():
    """Idempotent prewarm; pays cost ONCE per test session."""
    from mlframe.metrics.core import prewarm_numba_cache
    prewarm_numba_cache()


def test_fast_brier_score_loss_par_bool_dtype_is_warm():
    """``_fast_brier_score_loss_par`` with (bool, float64) signature MUST be
    JIT-cached by prewarm. First call after prewarm should be <10ms (warm),
    NOT 1000s of ms (which would mean compile happened here)."""
    from mlframe.metrics.core import _fast_brier_score_loss_par

    _ensure_prewarmed()
    y_true = np.random.randint(0, 2, 1000).astype(np.bool_)
    y_pred = np.random.random(1000).astype(np.float64)

    t = time.perf_counter()
    result = _fast_brier_score_loss_par(y_true, y_pred)
    elapsed_ms = (time.perf_counter() - t) * 1000

    assert np.isfinite(result)
    assert 0.0 <= result <= 1.0
    # Generous upper bound: post-prewarm calls are typically <1ms; first-cold
    # JIT compile would be 1500-4000ms. 50ms gives plenty of margin for
    # cprofile overhead / CI noise without masking a missed prewarm.
    assert elapsed_ms < 50.0, (
        f"_fast_brier_score_loss_par(bool, float64) took {elapsed_ms:.1f}ms; "
        f">50ms suggests the (bool, f64) signature is NOT prewarmed and a "
        f"fresh JIT compile fired here. Verify prewarm in metrics/core.py "
        f"_prewarm_numba_cache_body includes the bool->f64 path."
    )


def test_fast_log_loss_binary_par_bool_dtype_is_warm():
    """Same regression for ``_fast_log_loss_binary_par`` -- multilabel
    per-class loops emit (bool, float64) here too."""
    from mlframe.metrics.core import _fast_log_loss_binary_par

    _ensure_prewarmed()
    y_true = np.random.randint(0, 2, 1000).astype(np.bool_)
    y_pred = np.random.random(1000).astype(np.float64)

    t = time.perf_counter()
    result = _fast_log_loss_binary_par(y_true, y_pred, 1e-15)
    elapsed_ms = (time.perf_counter() - t) * 1000

    assert np.isfinite(result)
    assert elapsed_ms < 50.0, (
        f"_fast_log_loss_binary_par(bool, float64) took {elapsed_ms:.1f}ms; "
        f">50ms suggests the (bool, f64) signature is NOT prewarmed."
    )


def test_brier_bool_matches_float64_semantics():
    """Equivalence sanity: bool y_true must produce the same brier as the
    equivalent float64 y_true (defensive against numba dtype coercion bugs)."""
    from mlframe.metrics.core import _fast_brier_score_loss_par

    _ensure_prewarmed()
    rng = np.random.default_rng(20260523)
    y_true_bool = rng.integers(0, 2, 5000).astype(np.bool_)
    y_true_f64 = y_true_bool.astype(np.float64)
    y_pred = rng.random(5000).astype(np.float64)

    b1 = _fast_brier_score_loss_par(y_true_bool, y_pred)
    b2 = _fast_brier_score_loss_par(y_true_f64, y_pred)
    assert abs(b1 - b2) < 1e-12, (
        f"brier(bool) = {b1} differs from brier(float64) = {b2} by "
        f"{abs(b1 - b2):.2e}; possible dtype-coercion semantic divergence"
    )
