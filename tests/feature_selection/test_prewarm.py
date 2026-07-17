"""Unit tests for ``mlframe.feature_selection.filters._prewarm``.

The pre-warm entry point triggers numba JIT compilation across every dispatcher path used by the screening / cat-FE / discretisation hot loops. The risk
pattern: when a new dispatcher path is added in production but NOT in the prewarm matrix, the first real fit pays the full cold-start cost (~17s for
``parallel_mi*``, ~9s per discretisation dtype). These tests cover:

  - Smoke: ``prewarm_fs_numba_cache()`` runs without raising and is idempotent.
  - Coverage: known critical dispatchers (``compute_mi_from_classes``, discretisation kernels) have at least one compiled signature after prewarm.
  - Biz-value: a downstream ``compute_mi_from_classes`` call hits cache (post-warm >= 2x faster than its own first cold compile).
"""

from __future__ import annotations

import time

import numpy as np
import pytest


# Module-scoped fixture: prewarm exactly once per test session for the coverage / biz-value tests that need a warmed dispatcher graph.
@pytest.fixture(scope="module")
def warmed():
    """Run prewarm_fs_numba_cache once for the module and return True."""
    from mlframe.feature_selection.filters._prewarm import prewarm_fs_numba_cache

    prewarm_fs_numba_cache(verbose=False)
    return True


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Smoke
# ----------------------------------------------------------------------------------------------------------------------------------------------------


class TestPrewarmSmoke:
    """Entry-point sanity: importable, callable, idempotent, returns None."""

    def test_importable(self):
        """prewarm_fs_numba_cache is importable and callable."""
        from mlframe.feature_selection.filters import _prewarm

        assert hasattr(_prewarm, "prewarm_fs_numba_cache")
        assert callable(_prewarm.prewarm_fs_numba_cache)

    def test_public_api(self):
        """The module's __all__ exposes exactly prewarm_fs_numba_cache."""
        from mlframe.feature_selection.filters import _prewarm

        assert _prewarm.__all__ == ["prewarm_fs_numba_cache"]

    def test_runs_without_exception(self):
        """prewarm_fs_numba_cache runs without raising."""
        from mlframe.feature_selection.filters._prewarm import prewarm_fs_numba_cache

        # Must NOT raise even if individual kernels fail (the body wraps each call in try/except).
        prewarm_fs_numba_cache(verbose=False)

    def test_returns_none(self):
        """prewarm_fs_numba_cache returns None."""
        from mlframe.feature_selection.filters._prewarm import prewarm_fs_numba_cache

        out = prewarm_fs_numba_cache(verbose=False)
        assert out is None

    def test_idempotent(self):
        """A second prewarm call is near-instant (numba dispatcher cache hit) and does not raise."""
        # Second call must be near-instant (numba dispatcher cache hit) and again must not raise.
        from mlframe.feature_selection.filters._prewarm import prewarm_fs_numba_cache

        prewarm_fs_numba_cache(verbose=False)
        t0 = time.perf_counter()
        prewarm_fs_numba_cache(verbose=False)
        elapsed = time.perf_counter() - t0
        # Idempotent call should be far faster than the initial compile budget (~60s). Generous 30s upper bound to avoid CI flakiness.
        assert elapsed < 30.0, f"second prewarm took {elapsed:.2f}s -- cache not effective"

    def test_verbose_flag_accepts_true(self):
        """verbose=True emits a log line and does not raise."""
        # Verbose path emits a log line; must not raise.
        from mlframe.feature_selection.filters._prewarm import prewarm_fs_numba_cache

        prewarm_fs_numba_cache(verbose=True)

    def test_regression_module_level_guard_short_circuits_second_call(self, monkeypatch):
        """A second prewarm call short-circuits on the module-level guard instead of re-running the sweep."""
        # Wave 13 finding 7: prewarm_fs_numba_cache had no self-idempotency guard (unlike its sibling
        # ``_numba_warmup.warmup_typed_dict``'s ``_warmup_done`` flag), so a second call in the same
        # process re-ran the entire synthetic-input sweep instead of returning immediately.
        # Fails pre-fix (second call's ``np.random.default_rng`` count > first call's) and passes
        # post-fix (second call is a true no-op: same count).
        import numpy as np
        from mlframe.feature_selection.filters import _prewarm as pw

        calls = {"n": 0}
        orig_default_rng = np.random.default_rng

        def counted(*a, **kw):
            """Wrap default_rng, counting invocations to detect a re-run prewarm body."""
            calls["n"] += 1
            return orig_default_rng(*a, **kw)

        monkeypatch.setattr(np.random, "default_rng", counted)
        # Other tests in this module (the ``warmed`` fixture) may have already prewarmed the process;
        # force a clean "first call" state for this test.
        monkeypatch.setattr(pw, "_fs_numba_prewarmed", False)
        pw.prewarm_fs_numba_cache(verbose=False)
        n1 = calls["n"]
        assert n1 > 0, "prewarm body did not run at all on the first call"
        pw.prewarm_fs_numba_cache(verbose=False)
        n2 = calls["n"]
        assert n2 == n1, f"second call re-ran the prewarm body (n1={n1}, n2={n2}); guard not effective"

    def test_regression_cupy_prewarm_has_module_level_guard(self):
        """The CuPy prewarm twin has the same module-level idempotency guard."""
        # Same guard pattern must exist on the CuPy twin; the guard is decided before any cupy import,
        # so this is checkable even on a CUDA-less CI host.
        from mlframe.feature_selection.filters import _prewarm as pw

        assert hasattr(pw, "_fs_cupy_prewarmed")
        pw._fs_cupy_prewarmed = False
        pw.prewarm_fs_cupy_kernels(verbose=False)
        assert pw._fs_cupy_prewarmed is True
        # Second call must return immediately (guard true) regardless of CUDA availability.
        pw.prewarm_fs_cupy_kernels(verbose=False)


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Coverage: known dispatchers compiled
# ----------------------------------------------------------------------------------------------------------------------------------------------------


class TestPrewarmCoverage:
    """After prewarm, key dispatchers must have at least one compiled signature -- otherwise the production hot path still pays JIT cost on first use."""

    def test_compute_mi_from_classes_has_signatures(self, warmed):
        """compute_mi_from_classes has at least one compiled signature after prewarm."""
        from mlframe.feature_selection.filters.info_theory import compute_mi_from_classes

        # ``signatures`` is the public list of (dtype tuple) -> compiled-impl entries on a numba Dispatcher.
        assert len(compute_mi_from_classes.signatures) >= 1, "compute_mi_from_classes was not compiled by prewarm"

    def test_numba_utils_compiled(self, warmed):
        """The _numba_utils dispatchers have at least one compiled signature after prewarm."""
        from mlframe.feature_selection.filters._numba_utils import arr2str, count_cand_nbins, unpack_and_sort

        assert len(arr2str.signatures) >= 1
        assert len(count_cand_nbins.signatures) >= 1
        assert len(unpack_and_sort.signatures) >= 1

    def test_permutation_kernels_compiled(self, warmed):
        """The permutation-null njit kernels have at least one compiled signature after prewarm."""
        from mlframe.feature_selection.filters.permutation import parallel_mi, parallel_mi_prange, shuffle_arr

        assert len(parallel_mi.signatures) >= 1
        assert len(parallel_mi_prange.signatures) >= 1
        assert len(shuffle_arr.signatures) >= 1

    def test_marginal_screen_njit_compiled_in_fresh_process(self):
        """_marginal_screen_njit is compiled by prewarm in a fresh subprocess (regression: wrong-arity call left it cold)."""
        # Regression sensor: the prewarm body called ``_marginal_screen_njit`` with the wrong arity (5 args, omitting ``candidate_idxs``); the swallowing
        # ``except: pass`` hid the TypeError, leaving this hot prange marginal-MI screening kernel cold so it paid full JIT compile on the first real MRMR.fit.
        # Run in a fresh subprocess so no other test / import incidentally compiles the kernel first; assert prewarm populated a signature.
        import subprocess  # nosec B404 -- test-only local trusted subprocess invocation (fixed argv, no shell, no untrusted input)
        import sys

        code = (
            "import sys; sys.modules['cupy']=None\n"
            "from mlframe.feature_selection.filters._prewarm import prewarm_fs_numba_cache\n"
            "prewarm_fs_numba_cache(verbose=False)\n"
            "from mlframe.feature_selection.filters.cat_interactions import _marginal_screen_njit\n"
            "assert len(_marginal_screen_njit.signatures) >= 1, 'prewarm left _marginal_screen_njit cold'\n"
            "print('OK')\n"
        )
        env = dict(__import__("os").environ, CUDA_VISIBLE_DEVICES="", NUMBA_DISABLE_CUDA="1", MPLBACKEND="Agg")
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300)  # nosec B603 -- fixed local argv (sys.executable/git + literal args), no shell, no untrusted input
        assert res.returncode == 0 and "OK" in res.stdout, f"stdout={res.stdout!r} stderr={res.stderr[-1500:]!r}"

    def test_discretization_dtype_matrix_compiled(self, warmed):
        """_discretize_array_impl has at least one compiled signature after prewarm."""
        # The public ``discretize_array`` is a regular Python wrapper; its njit kernel is ``_discretize_array_impl``.
        # Guards the dtype-matrix prewarm regression (a missed dtype leaves a ~9s cold compile in prod).
        from mlframe.feature_selection.filters.discretization import _discretize_array_impl

        assert len(_discretize_array_impl.signatures) >= 1

    def test_renumber_joint_kernels_compiled(self, warmed):
        """The _mi_greedy_cmi_fe renumber/joint-entropy njit kernels have at least one compiled signature after prewarm."""
        # Regression sensor: create_unary_transformations' njit-wrapped LAMBDA entries (a bare numpy ufunc
        # like the OLD "cos"/"sin"/"abs" entries fails njit-wrapping silently in njit_functions_dict and
        # stays a raw ufunc -- only the lambda-bodied transforms below actually become numba Dispatchers)
        # were not covered by prewarm; the first FE round's _build_operand_table call paid their JIT
        # compile inline (~14s cumulative, measured via a saved cProfile .prof on the canonical 100k-row
        # wellbore fit). Includes both the original lambda-native transforms (identity/sqr/reciproc/...)
        # and the measured-win bare-ufunc-to-lambda conversions (see bench_unary_transform_njit_wrap.py):
        # sign/tanh/neg/rint/cbrt/arccos/arctan/cosh/arcsinh.
        from mlframe.feature_selection.filters.feature_engineering import create_unary_transformations

        d = create_unary_transformations(preset="maximal")
        for name in (
            "identity",
            "sqr",
            "reciproc",
            "sqrt",
            "qubed",
            "invsquared",
            "invqubed",
            "invcbrt",
            "invsqrt",
            "sign",
            "tanh",
            "neg",
            "rint",
            "cbrt",
            "arccos",
            "arctan",
            "cosh",
            "arcsinh",
        ):
            fn = d[name]
            assert len(fn.signatures) >= 1, f"unary transform {name!r} was not compiled by prewarm"

    def test_unary_transform_bare_ufunc_entries_stay_unwrapped(self, warmed):
        """The deliberately-unwrapped bare-ufunc unary transforms remain unwrapped (no njit signatures)."""
        # Companion to the above: bench_unary_transform_njit_wrap.py measured these entries FLAT-TO-
        # REGRESSING when njit-wrapped (numpy's C loop already wins), so they were deliberately left as
        # bare ufuncs. A future edit accidentally converting one without re-benchmarking would silently
        # regress -- this pins the current (intentional) unwrapped state so such a change is caught.
        from mlframe.feature_selection.filters.feature_engineering import create_unary_transformations

        d = create_unary_transformations(preset="maximal")
        for name in ("abs", "sin", "exp", "cos", "tan", "sinh", "arcsin", "arccosh", "arctanh"):
            fn = d[name]
            assert not hasattr(fn, "signatures"), (
                f"unary transform {name!r} is now njit-wrapped -- update the rejected-conversion bench/comment if this is an intentional, re-measured win"
            )

    def test_post_warm_no_inf_or_nan_in_smoke_kernel(self, warmed):
        """After prewarm, compute_mi_from_classes on a tiny input returns a finite, non-negative scalar."""
        # Spot-check the canonical kernel: after prewarm, computing MI on tiny input must return a finite non-negative scalar.
        from mlframe.feature_selection.filters.info_theory import compute_mi_from_classes

        rng = np.random.default_rng(0)
        cx = rng.integers(0, 4, 32).astype(np.int32)
        cy = rng.integers(0, 3, 32).astype(np.int32)
        fx = np.bincount(cx, minlength=4).astype(np.float64) / 32
        fy = np.bincount(cy, minlength=3).astype(np.float64) / 32
        mi = compute_mi_from_classes(classes_x=cx, freqs_x=fx, classes_y=cy, freqs_y=fy, dtype=np.int32)
        assert np.isfinite(mi)
        assert mi >= 0.0


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# biz_value: prewarm reduces cold-start
# ----------------------------------------------------------------------------------------------------------------------------------------------------


class TestPrewarmBizValue:
    """Quantitative win: a representative downstream ``@njit`` call must be measurably faster after prewarm than the first (cold) call would have been."""

    def test_biz_prewarm_reduces_cold_start(self, warmed):
        """After prewarm, compute_mi_from_classes on a fresh tiny fixture completes well under the cold-compile budget."""
        # We cannot truly time "cold vs warm" inside one process because numba caches process-wide. Instead: after prewarm, the call on a fresh tiny fixture
        # must complete well under the documented cold-compile budget (~17s for parallel_mi, ~3-5s for compute_mi_from_classes). A warm dispatch is sub-ms.
        from mlframe.feature_selection.filters.info_theory import compute_mi_from_classes

        rng = np.random.default_rng(42)
        n = 1000
        cx = rng.integers(0, 8, n).astype(np.int32)
        cy = rng.integers(0, 4, n).astype(np.int32)
        fx = np.bincount(cx, minlength=8).astype(np.float64) / n
        fy = np.bincount(cy, minlength=4).astype(np.float64) / n

        # Warm call must be sub-100ms -- if prewarm did its job the dispatcher is cached, the actual numeric work on 1000 rows is microseconds, and we have a
        # solid 100x margin over the cold-compile floor (~3s) which is what the assertion is really gating.
        t0 = time.perf_counter()
        mi1 = compute_mi_from_classes(classes_x=cx, freqs_x=fx, classes_y=cy, freqs_y=fy, dtype=np.int32)
        warm_elapsed = time.perf_counter() - t0

        # Second call -- pure cache hit, must be at most as slow as the first warm call (allow 2x slack for timer noise on tiny calls).
        t0 = time.perf_counter()
        mi2 = compute_mi_from_classes(classes_x=cx, freqs_x=fx, classes_y=cy, freqs_y=fy, dtype=np.int32)
        second_elapsed = time.perf_counter() - t0

        assert np.isfinite(mi1) and np.isfinite(mi2)
        assert mi1 == mi2, "deterministic kernel must produce identical output on identical input"
        # Warm call well under the cold-compile floor. Generous bound (0.5s) to absorb CI jitter; the real cold-compile is ~3-5s on the same machine.
        assert warm_elapsed < 0.5, f"first post-prewarm call took {warm_elapsed:.3f}s -- prewarm did not cover this dispatcher signature"
        # Second call must be at least as fast as warm (in expectation). With tiny-call timer noise we just require it stays under the same envelope.
        assert second_elapsed < 0.5, f"cached call took {second_elapsed:.3f}s -- unexpected"
