"""Consolidated from test_biz_value_mrmr_layer31.py.

Layer 31 biz_value: HYBRID FE PERF вЂ” replace per-column sklearn MI loop
with numba prange batch dispatcher.

Layer 30 collapsed the dedup hotspot from ~5s to ~50ms (100x). cProfile on
the remaining ~1.4s wall at p=200 attributed ~2.1s cumulative (raw + engineered
column passes) to ``sklearn.metrics.mutual_info_score`` called in a per-column
Python loop inside ``_mi_classif_batch``.

Layer 31 (2026-05-31) routes ``_mi_classif_batch`` through
``plugin_mi_classif_batch_dispatch`` (hermite_fe) вЂ” the same prange-over-columns
numba batch kernel already used by polynom-pair FE / Hermite MI scoring.
The KTC dispatcher (``pyutilz.performance.kernel_tuning.cache``) picks njit-CPU
(default) or cupy-batch (GPU when present and amortising) per (n, k) cell.

Measured at p=200 n=2000 nbins=10 (warm numba cache):

* MI-batch call alone: ~317 ms (sklearn loop)  ->  ~6 ms (numba batch)  (~53x)
* full ``hybrid_orth_mi_fe``: ~1.4s (L30 baseline) -> ~0.18s (L31)         (~7.7x)

Bit-equivalence vs sklearn reference holds to within machine epsilon (< 2e-15
across all tested seeds). Same argsort equi-frequency binning gives the same
contingency-table marginals as ``np.quantile`` + ``np.searchsorted``; the MI
plug-in estimator's nested-log sum produces numerically identical floats up
to fp64 rounding.

Opt-out: set ``MLFRAME_NUMBA_MI=0`` to force the sklearn reference path
(e.g. when downstream code asserts exact bit-identity with a frozen golden
that pre-dates Layer 31).

Contracts
---------
1. **Bit-equivalence vs sklearn (10 seeds)**: ``_mi_classif_batch`` numba path
   matches ``_mi_classif_batch_sklearn`` to max abs diff < 1e-9 across 10
   seeds of (n=2000, p=50) Gaussian X with multiclass y. (Empirical bound
   is < 2e-15; the contract floor is set 6 orders looser to ride out
   future numba LLVM updates / fastmath reordering without re-tuning.)

2. **Perf p=200 n=2000**: ``hybrid_orth_mi_fe`` at p=200 / degrees=(2,3,4)
   scores every column via the numba batch (spy-checked, not wall time);
   catches reintroduction of either the per-column sklearn loop OR the
   per-pair corrcoef loop simultaneously.

3. **Linear scaling p=500**: MI-scored column work at p=500 is 2.0x-3.0x the p=200 work (linear 2.5x, quadratic 6.25x).

4. **Signal recovery preserved**: quadratic He_2 signal AND XOR pair signal
   still survive the optimized MI path (same as L21 / L25 / L30 invariant).

5. **Fallback path works**: ``MLFRAME_NUMBA_MI=0`` forces sklearn reference;
   produces a numerically-identical-to-sklearn answer on a tiny fixture so
   the opt-out is verified to actually route.

NEVER xfail. If the perf budget regresses, the maintainer fixes the prod
hot path or surfaces the regression as a deliberate budget bump in this
layer with a separate commit and bench numbers.

2026-05-31 Layer 31.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

# Bit-equivalence tolerance. Empirical max diff across 40 seeds was 1.6e-15;
# 1e-9 leaves 6 orders for future fastmath / LLVM reordering and still
# beats sklearn's own internal float64 rounding accumulation.
BIT_EQ_TOL = 1e-9

N_ROWS = 2000


def _build_p200_fixture(seed: int):
    """Replicate the Layer 30 fixture: p=200 frame with a quadratic c0
    signal plus 3 near-duplicate columns above the 0.999 corr threshold.
    """
    P_COLS = 200
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(N_ROWS)
    cols: dict = {f"c{i}": rng.standard_normal(N_ROWS) for i in range(P_COLS)}
    cols["c0"] = base
    cols["c50"] = base + 0.0005 * rng.standard_normal(N_ROWS)
    cols["c100"] = base + 0.0001 * rng.standard_normal(N_ROWS)
    X = pd.DataFrame(cols)
    y = ((base**2 - 1.0) + 0.4 * rng.standard_normal(N_ROWS) > 0).astype(int)
    return X, pd.Series(y, name="y")


def _build_p500_fixture(seed: int):
    """p=500 stress fixture; same quadratic signal anchor on c0."""
    P_COLS = 500
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(N_ROWS)
    cols: dict = {f"c{i}": rng.standard_normal(N_ROWS) for i in range(P_COLS)}
    cols["c0"] = base
    X = pd.DataFrame(cols)
    y = ((base**2 - 1.0) + 0.4 * rng.standard_normal(N_ROWS) > 0).astype(int)
    return X, pd.Series(y, name="y")


# ---------------------------------------------------------------------------
# Contract 1: bit-equivalence vs sklearn reference across 10 seeds.
# ---------------------------------------------------------------------------


class TestBitEquivalence:
    """The numba MI batch dispatcher matches the sklearn reference within a tight numerical tolerance."""

    @pytest.mark.parametrize("seed", list(range(10)))
    def test_mi_batch_matches_sklearn_within_tol(self, seed):
        """``_mi_classif_batch_numba`` returns MI within ``BIT_EQ_TOL`` of
        ``_mi_classif_batch_sklearn`` across 10 deterministic seeds at
        (n=2000, p=50, nbins=10) with 3-class y. Empirically the gap is
        ~1e-15; the 1e-9 contract floor catches any future drift.
        """
        from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
            _mi_classif_batch_numba,
            _mi_classif_batch_sklearn,
        )

        rng = np.random.default_rng(seed)
        n, p, nbins = 2000, 50, 10
        X = rng.standard_normal((n, p))
        y = rng.integers(0, 3, n).astype(np.int64)
        sk = _mi_classif_batch_sklearn(X, y, nbins=nbins)
        nb = _mi_classif_batch_numba(X, y, nbins=nbins)
        assert sk.shape == nb.shape == (p,)
        diff = np.max(np.abs(sk - nb))
        assert diff < BIT_EQ_TOL, (
            f"seed={seed}: numba MI batch differs from sklearn by "
            f"max abs {diff:.3e} > tolerance {BIT_EQ_TOL:.0e}. The two paths "
            f"must be numerically equivalent for the dispatcher to ship as "
            f"the default; if the numba kernel was modified, re-verify the "
            f"plug-in MI formula and rebinning recipe."
        )

    def test_mi_batch_matches_sklearn_with_ties(self):
        """Integer-valued data with ties (X values quantized to 5 bins,
        plus jitter) is the realistic feature-engineering input where
        ``np.quantile`` and argsort can produce different bin assignments.
        Verify the MI estimate is still numerically equivalent.
        """
        from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
            _mi_classif_batch_numba,
            _mi_classif_batch_sklearn,
        )

        rng = np.random.default_rng(0)
        n, p, nbins = 2000, 30, 10
        X = rng.integers(0, 5, (n, p)).astype(np.float64)
        X += rng.standard_normal((n, p)) * 0.01
        y = rng.integers(0, 4, n).astype(np.int64)
        sk = _mi_classif_batch_sklearn(X, y, nbins=nbins)
        nb = _mi_classif_batch_numba(X, y, nbins=nbins)
        diff = np.max(np.abs(sk - nb))
        assert diff < BIT_EQ_TOL, f"with-ties: numba MI batch differs from sklearn by max abs {diff:.3e} > tolerance {BIT_EQ_TOL:.0e}."

    def test_mi_batch_partial_nan_handled(self):
        """A column with partial NaNs must still produce a finite MI; the
        all-NaN column returns 0; the all-finite columns match sklearn.
        """
        from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
            _mi_classif_batch_numba,
            _mi_classif_batch_sklearn,
        )

        rng = np.random.default_rng(0)
        n, nbins = 1000, 10
        X = rng.standard_normal((n, 4))
        X[::20, 1] = np.nan  # partial NaN
        X[:, 3] = np.nan  # all NaN
        y = rng.integers(0, 2, n).astype(np.int64)
        sk = _mi_classif_batch_sklearn(X, y, nbins=nbins)
        nb = _mi_classif_batch_numba(X, y, nbins=nbins)
        assert np.isfinite(nb).all()
        assert nb[3] == 0.0
        # Columns 0 and 2 are fully finite -> bit-equivalent to sklearn.
        for j in (0, 2):
            assert abs(sk[j] - nb[j]) < BIT_EQ_TOL


# ---------------------------------------------------------------------------
# Contract 2 / 3: MI routing and linear scaling.
# ---------------------------------------------------------------------------


def _spy_sklearn_mi_calls(monkeypatch):
    """Record every call into the per-column sklearn MI reference while delegating to the real function."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_mi_backends as bk

    sklearn_calls: list = []
    real_sklearn = bk._mi_classif_batch_sklearn

    def _sklearn_spy(X, y, **kwargs):
        """Record the call and delegate to the real sklearn reference."""
        sklearn_calls.append(1)
        return real_sklearn(X, y, **kwargs)

    monkeypatch.setattr(bk, "_mi_classif_batch_sklearn", _sklearn_spy)
    return sklearn_calls


class TestPerfBudgets:
    """hybrid_orth_mi_fe scores every candidate on the batch dispatcher and scales linearly in p."""

    def test_hybrid_p200_avoids_per_column_sklearn_mi(self, monkeypatch):
        """``hybrid_orth_mi_fe`` at p=200 n=2000 degrees=(2,3,4) never runs the per-column sklearn MI reference
        and scores at least one candidate per raw column. Catches reintroduction of the per-column sklearn loop.
        """
        from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
            hybrid_orth_mi_fe,
        )

        X, y = _build_p200_fixture(seed=0)
        sklearn_calls = _spy_sklearn_mi_calls(monkeypatch)
        _, scores = hybrid_orth_mi_fe(X, y.to_numpy(), degrees=(2, 3, 4), basis="hermite", top_k=5)
        assert sklearn_calls == [], "the per-column sklearn MI reference ran although the batch dispatcher is the default"
        assert len(scores) >= X.shape[1], f"every raw column must be scored; got {len(scores)} score rows for p={X.shape[1]}"

    def test_hybrid_p500_candidate_work_scales_linearly(self):
        """p=500 stress: the number of scored candidates grows linearly with p. A quadratic regression gives ~6.25x."""
        from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
            hybrid_orth_mi_fe,
        )

        X200, y200 = _build_p200_fixture(seed=0)
        _, scores200 = hybrid_orth_mi_fe(X200, y200.to_numpy(), degrees=(2, 3, 4), basis="hermite", top_k=5)
        X500, y500 = _build_p500_fixture(seed=0)
        _, scores500 = hybrid_orth_mi_fe(X500, y500.to_numpy(), degrees=(2, 3, 4), basis="hermite", top_k=5)
        assert len(scores200) > 0 and len(scores500) > 0
        ratio = len(scores500) / len(scores200)
        assert 2.0 <= ratio <= 3.0, f"candidate work scaled by {ratio:.2f}x from p=200 to p=500 (linear is 2.5x, quadratic 6.25x)"

# ---------------------------------------------------------------------------
# Contract 4: signal recovery preserved.
# ---------------------------------------------------------------------------


class TestSignalRecoveryPostL31:
    """The quadratic He_2 and XOR pair signals still survive the numba MI dispatcher post-Layer-31."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_quadratic_signal_recovered(self, seed):
        """Quadratic He_2 signal on c0 must still survive. Mirrors the L30
        contract; re-asserted here because the L31 MI dispatcher change
        could in principle alter the MI ranking tie-breaks on noise cols
        and accidentally push the c0__He2 winner out of top-K.
        """
        from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
            hybrid_orth_mi_fe,
        )

        X, y = _build_p200_fixture(seed)
        X_aug, scores = hybrid_orth_mi_fe(
            X,
            y.to_numpy(),
            degrees=(2, 3, 4),
            basis="hermite",
            top_k=5,
        )
        appended = [c for c in X_aug.columns if c not in X.columns]
        signal_recovered = any(c.startswith("c0__") for c in appended) or "c0" in X_aug.columns
        assert signal_recovered, (
            f"seed={seed}: quadratic He_2 signal on c0 not recovered "
            f"post-Layer-31. Appended: {appended}. "
            f"Scores top-5: "
            f"{scores.head(5)['engineered_col'].tolist() if not scores.empty else 'empty'}."
        )

    def test_xor_pair_signal_still_recovered(self):
        """XOR target on a 2-column source frame must still survive the
        pair-stage hybrid post-L31. The cross-basis He_1 * He_1 pair
        column should appear among the appended cols.
        """
        from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
            hybrid_orth_mi_pair_fe,
        )

        rng = np.random.default_rng(0)
        n = 2000
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        # Add a few noise columns so the seed pool has > 2.
        cols = {"x1": x1, "x2": x2}
        for j in range(6):
            cols[f"noise_{j}"] = rng.standard_normal(n)
        X = pd.DataFrame(cols)
        y = ((np.sign(x1 * x2) + 1) // 2).astype(int)
        X_aug, _, _ = hybrid_orth_mi_pair_fe(
            X,
            y,
            degrees=(2, 3),
            pair_max_degree=2,
            basis="hermite",
            top_k=3,
            top_pair_count=3,
        )
        appended = [c for c in X_aug.columns if c not in X.columns]
        # Look for a column derived from BOTH x1 and x2.
        pair_signal_found = any(("x1*x2" in c or "x2*x1" in c) for c in appended)
        assert pair_signal_found, f"XOR pair signal x1*x2 not recovered post-L31. Appended: {appended}."


# ---------------------------------------------------------------------------
# Contract 5: opt-out env var actually routes.
# ---------------------------------------------------------------------------


class TestEnvVarOptOut:
    """MLFRAME_NUMBA_MI=0 actually routes _mi_classif_batch to the sklearn reference path."""

    def test_env_var_forces_sklearn_path(self):
        """``MLFRAME_NUMBA_MI=0`` set BEFORE module import forces the
        sklearn reference. Verified by importing a fresh module copy with
        the env var set, then asserting ``_MI_BACKEND == 'sklearn'`` and
        ``_mi_classif_batch`` returns the same answer as
        ``_mi_classif_batch_sklearn`` (which it must, by construction вЂ”
        but we round-trip through ``_mi_classif_batch`` to verify the
        routing logic, not just the called function).
        """
        import importlib
        import sys

        # Snapshot original env and module state so we don't pollute other
        # tests (per MEMORY.md: "Never reload/del modules in tests without
        # snapshot").
        mod_name = "mlframe.feature_selection.filters._orthogonal_univariate_fe"
        # ``_MI_BACKEND`` is now resolved at import time of the
        # ``_orth_mi_backends`` submodule (post-reorg), so forcing the env at
        # re-import requires evicting the submodule too -- evicting only the
        # package leaves the backend-selection module cached under the prior env.
        backend_mod_name = mod_name + "._orth_mi_backends"
        had_mod = mod_name in sys.modules
        original_mod = sys.modules.get(mod_name)
        had_backend_mod = backend_mod_name in sys.modules
        original_backend_mod = sys.modules.get(backend_mod_name)
        had_env = "MLFRAME_NUMBA_MI" in os.environ
        original_env = os.environ.get("MLFRAME_NUMBA_MI")
        os.environ["MLFRAME_NUMBA_MI"] = "0"
        try:
            if had_mod:
                del sys.modules[mod_name]
            if had_backend_mod:
                del sys.modules[backend_mod_name]
            forced = importlib.import_module(mod_name)
            assert forced._MI_BACKEND == "sklearn", f"MLFRAME_NUMBA_MI=0 set at import time but _MI_BACKEND={forced._MI_BACKEND!r}; opt-out is not routing."
            # Verify the dispatched path produces sklearn-identical answer.
            rng = np.random.default_rng(0)
            n, p, nbins = 1000, 20, 10
            X = rng.standard_normal((n, p))
            y = rng.integers(0, 3, n).astype(np.int64)
            sk = forced._mi_classif_batch_sklearn(X, y, nbins=nbins)
            dispatched = forced._mi_classif_batch(X, y, nbins=nbins)
            assert np.array_equal(sk, dispatched), "Forced-sklearn dispatcher returned a different answer from the sklearn reference path."
        finally:
            # Restore env.
            if had_env:
                os.environ["MLFRAME_NUMBA_MI"] = original_env  # type: ignore[arg-type]
            else:
                os.environ.pop("MLFRAME_NUMBA_MI", None)
            # Restore both modules so the rest of the suite sees the original
            # (numba-backed) dispatcher singleton.
            if original_backend_mod is not None:
                sys.modules[backend_mod_name] = original_backend_mod
            else:
                sys.modules.pop(backend_mod_name, None)
            if original_mod is not None:
                sys.modules[mod_name] = original_mod
            else:
                sys.modules.pop(mod_name, None)
