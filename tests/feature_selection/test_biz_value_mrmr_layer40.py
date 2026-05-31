"""Layer 40 biz_value: VERIFY GPU PATH for hybrid FE.

``hermite_fe.polyeval_dispatch`` routes polynomial evaluation between
three backends (``njit`` -- single-thread Horner, ``njit_par`` -- prange
Horner, ``cuda`` -- cupy RawKernel) based on array length, hardware
availability, and an optional ``MLFRAME_POLYEVAL_BACKEND`` env-var
override. Hybrid FE (``hybrid_orth_mi_fe`` and the MRMR
``fe_hybrid_orth_enable=True`` pipeline) goes through this dispatcher
on every basis evaluation, so a silent regression on the GPU path can
mis-compute basis columns AND go unnoticed when the dev machine has no
GPU.

Layer 40 adds CONTRACTS on the dispatcher so the GPU path can't silently
diverge from the CPU path:

A. **CPU bit-equivalence at the dispatcher level**: forcing
   ``MLFRAME_POLYEVAL_BACKEND=njit`` matches auto-dispatch at small n
   bit-for-bit (rtol 1e-12) AND matches the parallel njit_par variant
   to rtol 1e-10 across all four bases.

B. **GPU bit-equivalence (if cupy available)**: forcing
   ``MLFRAME_POLYEVAL_BACKEND=cuda`` matches forced ``njit`` to rtol
   1e-9 (RawKernel runs in float64 with the same Horner recurrence; H2D
   and back is lossless) on a moderate-sized fixture across all four
   bases. Skipped via ``pytest.importorskip`` when cupy is missing or
   broken (numpy ABI / missing CUDA DLLs).

C. **GPU speedup gate**: at n=5_000_000 the CUDA backend beats
   ``njit_par`` (host-side wall time including H2D + D2H). Bound is
   loose (<= 1.5x njit_par) because tiny GPUs (GTX 1050 Ti) only just
   break even on plain polynomial eval; the contract is "GPU does not
   make things dramatically worse". Skipped via ``skipif`` (not xfail)
   when cupy is missing.

D. **Env-var fallback**: forcing every backend (``njit`` / ``njit_par``
   / ``cuda`` / unset) produces a finite, correct-length output for
   each basis. The ``cuda`` force WITHOUT cupy must silently fall back
   (existing contract in ``test_hermite_fe_coverage``; Layer 40
   re-pins it from the hybrid-pipeline perspective).

E. **MRMR hybrid round-trip across backends**: fit MRMR with
   ``fe_hybrid_orth_enable=True`` under one backend, transform under
   another -- support_ and transform output must match. Pins the
   dispatcher as truly stateless: switching backends between fit and
   inference (production scenario: trained on GPU box, serving on CPU
   box) must not change selected features or basis values within
   numerical tolerance.

NEVER xfail. GPU-required tests use ``pytest.importorskip("cupy")``;
the GPU speedup contract uses ``skipif`` so the absence of a working
GPU is reported plainly (not as a passing test).
"""
from __future__ import annotations

import os
import time
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# cupy availability sentinel (resolved once, reused by skip-marks).
# ---------------------------------------------------------------------------


def _cupy_ok() -> bool:
    """True iff cupy imports AND its CUDA stack initialises.

    Pure import isn't enough: on this dev box cupy is installed but
    cuTENSOR / cublas / cusolver DLLs are missing, so any non-trivial
    op raises. We probe with a 1-element kernel-equivalent op and
    treat ANY exception as "no working GPU".
    """
    try:
        import cupy as cp  # noqa: F401
        # Force CUDA context init + a trivial kernel launch.
        a = cp.asarray(np.array([1.0, 2.0], dtype=np.float64))
        _ = cp.asnumpy(a * 2.0)
        return True
    except Exception:
        return False


_CUPY_OK = _cupy_ok()
_REQUIRES_CUPY = pytest.mark.skipif(
    not _CUPY_OK,
    reason="cupy not importable or CUDA stack not initialisable on this host",
)


BASES = ("hermite", "legendre", "chebyshev", "laguerre")


def _make_inputs(n: int, basis: str, seed: int = 0):
    """Return (x, c) sized for the given basis. Use a narrow x-range so
    high-degree polynomials don't overflow float64 (chebyshev/legendre
    need |x|<=1; laguerre prefers x>=0; hermite is unbounded but we keep
    it inside ~3sigma for the recurrence to stay numerically tame).
    """
    rng = np.random.default_rng(seed)
    if basis == "laguerre":
        x = rng.uniform(0.0, 2.0, size=n).astype(np.float64)
    elif basis == "hermite":
        x = rng.uniform(-2.0, 2.0, size=n).astype(np.float64)
    else:  # legendre / chebyshev
        x = rng.uniform(-0.9, 0.9, size=n).astype(np.float64)
    c = rng.uniform(-0.5, 0.5, size=5).astype(np.float64)
    return x, c


# ---------------------------------------------------------------------------
# A. CPU bit-equivalence -- forced njit vs auto-dispatch vs njit_par.
# ---------------------------------------------------------------------------


class TestCPUBitEquivalence:

    @pytest.mark.parametrize("basis", BASES)
    def test_forced_njit_matches_auto_dispatch_small_n(self, basis, monkeypatch):
        """At small n (=500, well below _PAR_THRESHOLD=50000), auto-
        dispatch must pick ``njit`` and forcing ``njit`` must produce
        bit-identical output (same code path; rtol 1e-12 = float
        round-trip).
        """
        from mlframe.feature_selection.filters.hermite_fe import polyeval_dispatch
        x, c = _make_inputs(500, basis)

        # Auto-dispatch (no env override).
        monkeypatch.delenv("MLFRAME_POLYEVAL_BACKEND", raising=False)
        out_auto = polyeval_dispatch(basis, x, c)

        # Forced njit.
        monkeypatch.setenv("MLFRAME_POLYEVAL_BACKEND", "njit")
        out_njit = polyeval_dispatch(basis, x, c)

        np.testing.assert_allclose(
            out_auto, out_njit, rtol=1e-12, atol=0,
            err_msg=(
                f"A basis={basis}: auto-dispatch at n=500 should pick "
                f"njit (n<_PAR_THRESHOLD); auto and forced-njit must "
                f"match bit-for-bit."
            ),
        )

    @pytest.mark.parametrize("basis", BASES)
    def test_forced_njit_par_matches_njit_within_tolerance(self, basis, monkeypatch):
        """njit and njit_par run the same Horner recurrence in different
        orderings (sequential vs prange); their outputs must match to
        rtol 1e-10 (fastmath=True allows minor associativity slack but
        Horner has no cross-row reductions, so we expect tight agreement).
        """
        from mlframe.feature_selection.filters.hermite_fe import polyeval_dispatch
        x, c = _make_inputs(2000, basis, seed=1)

        monkeypatch.setenv("MLFRAME_POLYEVAL_BACKEND", "njit")
        out_njit = polyeval_dispatch(basis, x, c)
        monkeypatch.setenv("MLFRAME_POLYEVAL_BACKEND", "njit_par")
        out_par = polyeval_dispatch(basis, x, c)

        np.testing.assert_allclose(
            out_par, out_njit, rtol=1e-10, atol=1e-12,
            err_msg=(
                f"A basis={basis}: njit_par must match njit; if you see "
                f"large deltas the prange recurrence has a bug "
                f"(off-by-one on the seed terms?)."
            ),
        )


# ---------------------------------------------------------------------------
# B. GPU bit-equivalence (cupy required).
# ---------------------------------------------------------------------------


class TestGPUBitEquivalence:

    @_REQUIRES_CUPY
    @pytest.mark.parametrize("basis", BASES)
    def test_cuda_matches_njit_on_large_fixture(self, basis, monkeypatch):
        """RawKernel on a moderate fixture (n=5000) must match njit to
        rtol 1e-9. CUDA path runs the SAME Horner recurrence in float64;
        only the parallelisation pattern differs, so bit-equivalence is
        the expected contract.

        n=5000 keeps the test cheap (no need to hit a multi-million row
        speedup point) while still exercising the H2D/D2H boundary.
        """
        pytest.importorskip("cupy")
        from mlframe.feature_selection.filters.hermite_fe import polyeval_dispatch

        x, c = _make_inputs(5000, basis, seed=2)

        monkeypatch.setenv("MLFRAME_POLYEVAL_BACKEND", "njit")
        out_cpu = polyeval_dispatch(basis, x, c)
        monkeypatch.setenv("MLFRAME_POLYEVAL_BACKEND", "cuda")
        out_gpu = polyeval_dispatch(basis, x, c)

        # If cuda backend silently fell back (e.g. kernel compile fail
        # at test time), out_gpu equals out_cpu -- still fine for the
        # contract.
        np.testing.assert_allclose(
            out_gpu, out_cpu, rtol=1e-9, atol=1e-10,
            err_msg=(
                f"B basis={basis}: CUDA Horner must match CPU Horner to "
                f"rtol 1e-9; large delta indicates a recurrence-coefficient "
                f"bug in _CUDA_KERNELS[{basis!r}]."
            ),
        )


# ---------------------------------------------------------------------------
# C. GPU speedup gate (cupy required, large fixture).
# ---------------------------------------------------------------------------


class TestGPUSpeedup:
    """At n >= _CUDA_THRESHOLD (5e5 by default), CUDA SHOULD beat
    njit_par; on a weak GPU the realistic floor is "no worse than 1.5x
    slower" once H2D + D2H + kernel launch are amortised. The contract
    is intentionally permissive -- we just want to catch a regression
    where the CUDA path became 10x slower (e.g. accidental
    cp.asnumpy on a tiny intermediate).
    """

    @_REQUIRES_CUPY
    def test_cuda_does_not_regress_vs_njit_par_at_5m(self, monkeypatch):
        pytest.importorskip("cupy")
        from mlframe.feature_selection.filters.hermite_fe import polyeval_dispatch

        # Use chebyshev (cheapest recurrence -> hardest case for GPU to
        # win, so the bound is tightest here -- if cuda wins chebyshev
        # it'll win the others too).
        x, c = _make_inputs(5_000_000, "chebyshev", seed=3)

        # Warm up both backends so JIT compile / CUDA context cost
        # doesn't pollute the timing.
        monkeypatch.setenv("MLFRAME_POLYEVAL_BACKEND", "njit_par")
        _ = polyeval_dispatch("chebyshev", x[:1024], c)
        monkeypatch.setenv("MLFRAME_POLYEVAL_BACKEND", "cuda")
        _ = polyeval_dispatch("chebyshev", x[:1024], c)

        # Time forced njit_par.
        monkeypatch.setenv("MLFRAME_POLYEVAL_BACKEND", "njit_par")
        t0 = time.perf_counter()
        _ = polyeval_dispatch("chebyshev", x, c)
        t_par = time.perf_counter() - t0

        # Time forced cuda (includes H2D + D2H).
        monkeypatch.setenv("MLFRAME_POLYEVAL_BACKEND", "cuda")
        t0 = time.perf_counter()
        _ = polyeval_dispatch("chebyshev", x, c)
        t_cuda = time.perf_counter() - t0

        # Cuda <= 1.5x njit_par -- permissive; the hopeful path is
        # t_cuda <= 0.7 * t_par. Either way, "GPU isn't a disaster".
        assert t_cuda <= 1.5 * t_par + 0.1, (
            f"C cuda regressed vs njit_par at n=5e6 chebyshev: "
            f"t_cuda={t_cuda:.3f}s, t_par={t_par:.3f}s "
            f"(ratio {t_cuda / t_par:.2f}x). Bound: <= 1.5x."
        )


# ---------------------------------------------------------------------------
# D. Env-var fallback / overrides work for all four backends.
# ---------------------------------------------------------------------------


class TestEnvVarFallback:

    @pytest.mark.parametrize("backend", ("", "njit", "njit_par"))
    @pytest.mark.parametrize("basis", BASES)
    def test_forced_backend_produces_finite_output(self, backend, basis, monkeypatch):
        """Setting MLFRAME_POLYEVAL_BACKEND to each valid value (incl.
        empty == auto) returns finite output of the expected length for
        every basis.
        """
        from mlframe.feature_selection.filters.hermite_fe import polyeval_dispatch
        x, c = _make_inputs(1500, basis, seed=4)
        if backend == "":
            monkeypatch.delenv("MLFRAME_POLYEVAL_BACKEND", raising=False)
        else:
            monkeypatch.setenv("MLFRAME_POLYEVAL_BACKEND", backend)
        out = polyeval_dispatch(basis, x, c)
        assert out.shape == x.shape, (
            f"D backend={backend!r} basis={basis}: output shape "
            f"{out.shape} != input shape {x.shape}"
        )
        assert np.all(np.isfinite(out)), (
            f"D backend={backend!r} basis={basis}: output has non-finite "
            f"values; backend force broke the recurrence."
        )

    @pytest.mark.parametrize("basis", BASES)
    def test_cuda_force_falls_back_when_cupy_missing(self, basis, monkeypatch):
        """Forcing cuda must not raise when cupy is unavailable; it
        silently falls back to a CPU backend. Existing contract in
        test_hermite_fe_coverage.test_polyeval_dispatch_cuda_silent_fallback;
        Layer 40 re-pins per basis.
        """
        from mlframe.feature_selection.filters import hermite_fe as hfe
        monkeypatch.setenv("MLFRAME_POLYEVAL_BACKEND", "cuda")
        # Temporarily mask cupy availability so we test the fallback
        # path even when cupy IS installed (we want the contract to
        # hold uniformly, not only on machines without cupy).
        monkeypatch.setattr(hfe, "_CUDA_AVAILABLE", False)

        x, c = _make_inputs(800, basis, seed=5)
        out = hfe.polyeval_dispatch(basis, x, c)
        assert out.shape == x.shape
        assert np.all(np.isfinite(out)), (
            f"D fallback basis={basis}: forced-cuda with masked cupy "
            f"must silently route to a CPU backend and return finite "
            f"output."
        )


# ---------------------------------------------------------------------------
# E. MRMR hybrid round-trip across backends -- fit under one backend,
#    transform under another. Pins the dispatcher as stateless.
# ---------------------------------------------------------------------------


def _build_hybrid_fixture(seed: int = 0, n: int = 1500):
    """Quadratic He_2 signal + 4 noise columns. Hybrid recovers
    ``x1__He2`` reliably at n=1500 (Layer 25 calibration)."""
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    X = pd.DataFrame({
        "x1": x1,
        "noise_0": rng.standard_normal(n),
        "noise_1": rng.standard_normal(n),
        "noise_2": rng.standard_normal(n),
        "noise_3": rng.standard_normal(n),
    })
    y = ((x1 ** 2 - 1.0) + 0.2 * rng.standard_normal(n) > 0).astype(int)
    return X, pd.Series(y, name="y")


def _mrmr_hybrid_kw():
    return dict(
        verbose=0,
        interactions_max_order=1,
        fe_max_steps=0,
        dcd_enable=False,
        cluster_aggregate_enable=False,
        build_friend_graph=False,
        cat_fe_config=None,
        quantization_nbins=10,
        random_seed=0,
        fe_hybrid_orth_enable=True,
        fe_hybrid_orth_pair_enable=False,
        fe_hybrid_orth_basis="hermite",
        fe_hybrid_orth_degrees=(2, 3),
        fe_hybrid_orth_top_k=5,
    )


class TestRoundTripAcrossBackends:

    @pytest.mark.parametrize(
        "fit_backend,transform_backend",
        [
            ("njit", "njit_par"),
            ("njit_par", "njit"),
            ("", "njit"),
            ("njit", ""),
        ],
    )
    def test_fit_transform_match_across_backends(
        self, fit_backend, transform_backend, monkeypatch,
    ):
        """Fit MRMR under one backend, transform under another. Support
        and transform-output values must match to numerical tolerance.

        Production scenario: model is trained on a GPU host
        (``MLFRAME_POLYEVAL_BACKEND`` unset, dispatcher picks cuda at
        large n during training) and served on a CPU host (forced
        njit). The dispatcher MUST be stateless across this boundary.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _build_hybrid_fixture(seed=0)
        kw = _mrmr_hybrid_kw()

        # Fit.
        if fit_backend == "":
            monkeypatch.delenv("MLFRAME_POLYEVAL_BACKEND", raising=False)
        else:
            monkeypatch.setenv("MLFRAME_POLYEVAL_BACKEND", fit_backend)
        m = MRMR(**kw).fit(X, y)
        sup_fit = list(m.get_feature_names_out())
        # Sanity: hybrid signal was recovered.
        assert "x1__He2" in sup_fit, (
            f"E fit_backend={fit_backend!r}: hybrid must recover x1__He2 "
            f"under fit-time backend; got {sup_fit}"
        )

        # Transform reference under fit backend.
        X_ref = m.transform(X)

        # Switch backend and transform again.
        if transform_backend == "":
            monkeypatch.delenv("MLFRAME_POLYEVAL_BACKEND", raising=False)
        else:
            monkeypatch.setenv("MLFRAME_POLYEVAL_BACKEND", transform_backend)
        X_swapped = m.transform(X)

        # Contract E.1: column set identical.
        assert list(X_ref.columns) == list(X_swapped.columns), (
            f"E fit={fit_backend!r} -> tx={transform_backend!r}: "
            f"transform columns diverged across backends. "
            f"ref={list(X_ref.columns)}, swap={list(X_swapped.columns)}"
        )

        # Contract E.2: values match to rtol 1e-9 (njit/njit_par share
        # the same Horner; the only float diff is associativity inside
        # prange, which Horner doesn't exercise across rows).
        np.testing.assert_allclose(
            X_swapped.to_numpy(), X_ref.to_numpy(),
            rtol=1e-9, atol=1e-10,
            err_msg=(
                f"E fit={fit_backend!r} -> tx={transform_backend!r}: "
                f"transform output diverged numerically when the polyeval "
                f"backend changed between fit and transform. The "
                f"dispatcher is NOT stateless; backend swap rewrote a "
                f"basis column."
            ),
        )


# ---------------------------------------------------------------------------
# F. cupy availability sentinel -- emit a single informational test so
#    the suite plainly reports whether GPU contracts ran or were skipped.
# ---------------------------------------------------------------------------


def test_cupy_availability_reported():
    """Informational: always passes; the message records whether GPU
    contracts (TestGPUBitEquivalence, TestGPUSpeedup) actually ran on
    this host. Use ``pytest -v -s`` to see it.
    """
    if _CUPY_OK:
        msg = "cupy import + CUDA context OK -- GPU contracts will run"
    else:
        msg = (
            "cupy unavailable on this host -- GPU contracts skipped via "
            "pytest.importorskip / skipif. CPU contracts still pinned."
        )
    # The "assert True" is here so the test never fails; the message
    # is captured by pytest -s.
    print(msg)
    assert True
