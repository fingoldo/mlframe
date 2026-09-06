"""Regression test for the (bool, float64) prewarm of parallel reduction kernels.

c0023 iter190 profile (multilabel cb+linear+xgb, 200k) attributed 4.156s of
``_compile_for_args`` to ``fast_brier_score_loss`` across 2 fresh compiles --
the (bool, float64) signature emitted by multilabel per-class loops
(``y_true = targets == class_name`` -> ndarray[bool]) was NOT covered by the
existing prewarm in ``mlframe.metrics.core._prewarm_numba_cache_body``.

Adding the prewarm pays the same 4s upfront at import time but moves it OUT
of the first-fit hot path. This test verifies the (bool, f64) parallel
variants are warm AFTER prewarm by asserting the signature is already in the dispatcher's compiled table.
"""

import numpy as np

from tests.conftest import skip_under_numba_disabled_jit


def _bool_signature_present(fn) -> bool:
    """Whether numba has already compiled ``fn`` for a boolean first argument.

    ``nopython_signatures`` grows one entry per argument-type combination the dispatcher compiles, so this
    answers the contract these tests exist for directly. A wall-clock bound cannot: it reads the same on a
    prewarmed dispatcher and on a box that merely compiles the tiny kernel inside the budget, and it is
    breached on a healthy build whenever the box is contended.
    """
    sigs = getattr(fn, "nopython_signatures", None)
    if sigs is None:
        sigs = getattr(fn, "signatures", None)
    assert sigs is not None, f"{fn!r} is not a numba dispatcher; this check has lost its subject"
    return any("bool" in str(sig.args[0]) for sig in sigs if getattr(sig, "args", None))


def _ensure_prewarmed():
    """Idempotent prewarm; pays cost ONCE per test session."""
    from mlframe.metrics.core import prewarm_numba_cache

    prewarm_numba_cache()


@skip_under_numba_disabled_jit
def test_fast_brier_score_loss_par_bool_dtype_is_warm():
    """``_fast_brier_score_loss_par`` with a (bool, float64) signature MUST be JIT-cached by prewarm.

    Asked of the dispatcher's compiled-signature table rather than of the clock: a 50ms bound was both a
    false red on a contended box and a false green on any box that compiles the tiny kernel inside it."""
    from mlframe.metrics.core import _fast_brier_score_loss_par

    _ensure_prewarmed()
    y_true = np.random.randint(0, 2, 1000).astype(np.bool_)
    y_pred = np.random.random(1000).astype(np.float64)

    assert _bool_signature_present(_fast_brier_score_loss_par), (
        "_fast_brier_score_loss_par has no compiled (bool, float64) signature after prewarm; verify "
        "_prewarm_numba_cache_body in metrics/core.py includes the bool->f64 path"
    )
    n_sigs = len(_fast_brier_score_loss_par.nopython_signatures)

    result = _fast_brier_score_loss_par(y_true, y_pred)

    assert len(_fast_brier_score_loss_par.nopython_signatures) == n_sigs, "the call compiled a fresh signature, so prewarm did not cover it"
    assert np.isfinite(result)
    assert 0.0 <= result <= 1.0


@skip_under_numba_disabled_jit
def test_fast_log_loss_binary_par_bool_dtype_is_warm():
    """Same regression for ``_fast_log_loss_binary_par``: multilabel per-class loops emit (bool, float64) here too."""
    from mlframe.metrics.core import _fast_log_loss_binary_par

    _ensure_prewarmed()
    y_true = np.random.randint(0, 2, 1000).astype(np.bool_)
    y_pred = np.random.random(1000).astype(np.float64)

    assert _bool_signature_present(_fast_log_loss_binary_par), "_fast_log_loss_binary_par has no compiled (bool, float64) signature after prewarm"
    n_sigs = len(_fast_log_loss_binary_par.nopython_signatures)

    result = _fast_log_loss_binary_par(y_true, y_pred, 1e-15)

    assert len(_fast_log_loss_binary_par.nopython_signatures) == n_sigs, "the call compiled a fresh signature, so prewarm did not cover it"
    assert np.isfinite(result)


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
    assert abs(b1 - b2) < 1e-12, f"brier(bool) = {b1} differs from brier(float64) = {b2} by {abs(b1 - b2):.2e}; possible dtype-coercion semantic divergence"
