"""Regression sensors for the 2026-05-20 polynom-FE follow-up wave.

Three protect-against-revert tests:

- NEW-A: passing ``precomputed_trivial_baseline`` MUST elide the
  internal ``best_trivial_pair`` call inside ``optimise_hermite_pair``.
  Counts ``best_trivial_pair`` invocations across a 5-restart loop and
  asserts they drop from 5 to 0 when the caller hoists the baseline.

- NEW-B: the basis-matrix GEMV gate MUST stay ON for polynomial bases
  under ``multi_fidelity=True``. Pre-flip the gate was
  ``not _multi_fidelity_active``; post-flip ``always when basis is
  polynomial``. A revert to the old gating would silently regress the
  1.13-1.19x speedup measured 2026-05-20.

- NEW-D: passing a small ``early_stop_no_improve`` MUST measurably
  shorten ``optimise_hermite_pair`` wall time vs an effectively-infinite
  value on a target where the warm-start finds the optimum early.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import numpy as np
import pytest


class TestNewATrivialBaselineHoist:
    """precomputed_trivial_baseline must skip the internal
    best_trivial_pair call inside optimise_hermite_pair."""

    def test_precomputed_baseline_elides_internal_call(self) -> None:
        """Precomputed baseline elides internal call."""
        from mlframe.feature_selection.filters.hermite_fe import (
            optimise_hermite_pair,
        )
        from mlframe.feature_selection.filters import fe_baselines

        rng = np.random.default_rng(11)
        n = 3000
        x_a = rng.normal(size=n)
        x_b = rng.normal(size=n)
        y = (x_a * x_b > 0).astype(np.int64)

        # Without precomputed baseline: best_trivial_pair runs once per
        # optimise_hermite_pair call. We loop 5 times to mirror the
        # fe_smart_polynom_iters=5 restart pattern.
        original = fe_baselines.best_trivial_pair
        call_count_without = {"n": 0}

        def _spy_without(*args, **kwargs):
            """Spy without."""
            call_count_without["n"] += 1
            return original(*args, **kwargs)

        with patch.object(fe_baselines, "best_trivial_pair", _spy_without):
            for s in range(5):
                optimise_hermite_pair(
                    x_a=x_a,
                    x_b=x_b,
                    y=y,
                    n_trials=10,
                    max_degree=2,
                    multi_fidelity=False,
                    sweep_degrees=False,
                    seed=42 + s,
                )
        assert call_count_without["n"] == 5, f"expected 5 best_trivial_pair calls (one per restart) without precompute; got {call_count_without['n']}"

        # WITH precomputed baseline: the caller pre-runs best_trivial_pair
        # ONCE then feeds the result via precomputed_trivial_baseline.
        # Internal calls must drop to ZERO.
        from mlframe.feature_selection.filters.fe_baselines import (
            best_trivial_pair,
        )

        trivial = best_trivial_pair(
            x_a,
            x_b,
            y,
            discrete_target=True,
            mi_estimator="plugin",
            plugin_n_bins=20,
            n_neighbors=None,
        )
        assert trivial is not None
        trivial_name, _, trivial_mi = trivial

        call_count_with = {"n": 0}

        def _spy_with(*args, **kwargs):
            """Spy with."""
            call_count_with["n"] += 1
            return original(*args, **kwargs)

        with patch.object(fe_baselines, "best_trivial_pair", _spy_with):
            for s in range(5):
                optimise_hermite_pair(
                    x_a=x_a,
                    x_b=x_b,
                    y=y,
                    n_trials=10,
                    max_degree=2,
                    multi_fidelity=False,
                    sweep_degrees=False,
                    seed=42 + s,
                    precomputed_trivial_baseline=trivial_mi,
                    precomputed_trivial_name=trivial_name,
                )
        assert (
            call_count_with["n"] == 0
        ), f"precomputed_trivial_baseline did NOT elide internal call: got {call_count_with['n']} calls (expected 0). NEW-A regressed."


class TestNewBBasisMatrixGate:
    """The 2026-05-20 gate flip enables the basis-matrix GEMV path
    REGARDLESS of multi_fidelity. A revert to the pre-2026-05-20 gating
    (``not _multi_fidelity_active``) would silently leave the 1.13-1.19x
    speedup on the table at the 1500-sample CMA inner scale.

    We don't time the call (flaky on CI); we ASSERT the structural fact
    that ``_eval_coef_pair`` is invoked with non-None B_a/B_b when
    multi_fidelity=True at n>=4000 with a polynomial basis.
    """

    @pytest.mark.parametrize("basis", ["hermite", "legendre", "chebyshev", "laguerre"])
    def test_b_matrices_built_under_multi_fidelity(self, basis: str) -> None:
        """B matrices built under multi fidelity."""
        from mlframe.feature_selection.filters import hermite_fe

        rng = np.random.default_rng(11)
        n = 5_000  # >= 4000 so multi_fidelity activates
        if basis == "hermite":
            x_a = rng.normal(size=n)
            x_b = rng.normal(size=n)
        elif basis in ("legendre", "chebyshev"):
            x_a = rng.uniform(-1.0, 1.0, n)
            x_b = rng.uniform(-1.0, 1.0, n)
        else:  # laguerre
            x_a = rng.exponential(2.0, n)
            x_b = rng.exponential(2.0, n)
        y = (x_a * x_b > 0).astype(np.int64)

        # After the hermite_fe monolith split, _eval_coef_pair lives in the
        # _hermite_fe_optimise sibling; the parent re-exports it but the actual
        # call site inside optimise_hermite_pair resolves to the sibling's own
        # globals. The default ``cma_batch`` optimiser (since 2026-05-22) calls
        # ``_eval_coef_pair_batch`` instead of ``_eval_coef_pair`` for the
        # inner CMA loop, so the gate sensor must intercept BOTH.
        from mlframe.feature_selection.filters import _hermite_fe_optimise as _opt_mod

        captured = {"saw_non_none_B": False, "calls": 0}
        original_eval = _opt_mod._eval_coef_pair
        original_eval_batch = _opt_mod._eval_coef_pair_batch

        def _spy_eval(*args, **kwargs):
            """Spy eval."""
            captured["calls"] += 1
            if kwargs.get("B_a") is not None and kwargs.get("B_b") is not None:
                captured["saw_non_none_B"] = True
            return original_eval(*args, **kwargs)

        def _spy_eval_batch(*args, **kwargs):
            """Spy eval batch."""
            captured["calls"] += 1
            if kwargs.get("B_a") is not None and kwargs.get("B_b") is not None:
                captured["saw_non_none_B"] = True
            return original_eval_batch(*args, **kwargs)

        with patch.object(_opt_mod, "_eval_coef_pair", _spy_eval), patch.object(_opt_mod, "_eval_coef_pair_batch", _spy_eval_batch):
            hermite_fe.optimise_hermite_pair(
                x_a=x_a,
                x_b=x_b,
                y=y,
                n_trials=20,
                max_degree=3,
                multi_fidelity=True,
                sweep_degrees=False,
                seed=42,
                basis=basis,
                warm_start=False,
            )

        assert captured["calls"] > 0, "spy did not capture any _eval_coef_pair* calls"
        assert captured["saw_non_none_B"], (
            f"basis={basis}: _eval_coef_pair* received None for B_a/B_b under "
            f"multi_fidelity=True. NEW-B gate flip regressed -- production "
            f"is paying the 1.13-1.19x Horner cost again."
        )


class TestNewDPlateauEarlyStop:
    """Plateau early-stop in _run_cma_search MUST measurably shorten
    wall time on a target where CMA converges quickly (warm-start
    already at optimum).

    biz_value sensor: with ``early_stop_no_improve=20`` (small) vs
    ``=10000`` (effectively off), the small value runs strictly less
    work. We don't pin a speedup ratio (flaky on shared CI) but assert
    a non-trivial wall-time gap (at least 1.5x).
    """

    def test_early_stop_shortens_wall_time(self) -> None:
        """Early stop shortens wall time."""
        from mlframe.feature_selection.filters.hermite_fe import (
            optimise_hermite_pair,
        )

        rng = np.random.default_rng(11)
        # XOR-like: CMA converges in 2-3 generations from warm-start.
        n = 5_000
        x_a = rng.uniform(-1.0, 1.0, n)
        x_b = rng.uniform(-1.0, 1.0, n)
        y = ((x_a * x_b > 0) ^ (rng.random(n) < 0.05)).astype(np.int64)

        # Warmup
        optimise_hermite_pair(
            x_a=x_a[:1000],
            x_b=x_b[:1000],
            y=y[:1000],
            n_trials=20,
            max_degree=2,
            multi_fidelity=False,
            sweep_degrees=False,
            basis="chebyshev",
            use_trivial_baseline=False,
            baseline_uplift_threshold=0.5,
        )

        # Median of 3 trials each (cheap; n=5000, n_trials=300)
        t_on, t_off = [], []
        for s in range(3):
            t0 = time.perf_counter()
            optimise_hermite_pair(
                x_a=x_a,
                x_b=x_b,
                y=y,
                n_trials=300,
                max_degree=3,
                multi_fidelity=False,
                sweep_degrees=False,
                seed=42 + s,
                basis="chebyshev",
                use_trivial_baseline=False,
                baseline_uplift_threshold=0.5,
                early_stop_no_improve=20,
            )
            t_on.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            optimise_hermite_pair(
                x_a=x_a,
                x_b=x_b,
                y=y,
                n_trials=300,
                max_degree=3,
                multi_fidelity=False,
                sweep_degrees=False,
                seed=42 + s,
                basis="chebyshev",
                use_trivial_baseline=False,
                baseline_uplift_threshold=0.5,
                early_stop_no_improve=10000,
            )
            t_off.append(time.perf_counter() - t0)

        med_on = float(np.median(t_on))
        med_off = float(np.median(t_off))
        # Floor: 1.5x. Measured 5.29x in dev; 1.5x catches a full revert
        # of the early-stop branch without being flaky on CI.
        assert med_off / max(med_on, 1e-9) >= 1.5, (
            f"NEW-D plateau early-stop did not shorten wall time: "
            f"on={med_on:.3f}s, off={med_off:.3f}s "
            f"(ratio={med_off / max(med_on, 1e-9):.2f}x, floor=1.5x). "
            f"Either the ``early_stop_no_improve_gens`` thread is broken "
            f"or _run_cma_search no longer checks it."
        )
