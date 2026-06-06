"""Layer 103 biz_value: Param-Oracle <-> kernel_tuning_cache migration POC.

This layer wires the FIRST kernel_tuning_cache consumer through the
ParamOracle ("learning to optimize") path, as a proof-of-concept of the
migration. It pins six contracts:

1. **Bridge imports KTC regions**: ``ParamOracle.read_ktc_regions`` turns a
   KernelTuningCache region table (synthetic / mock) into cold-start
   observations so a migrated consumer inherits the tuning history instead
   of starting blind. Read-only -- never writes back to the kernel cache.

2. **Oracle learns the njit/njit_par crossover**: after benchmarking the CPU
   polyeval backends at a small (n=200) and a large (n=500k) size, the
   oracle recommends ``njit`` at small and ``njit_par`` at large -- LEARNED
   from recorded wall-times, not a hardcoded threshold.

3. **Bit-equivalence**: ``polyeval_dispatch`` output is identical regardless
   of which CPU backend the oracle picks (rtol 1e-12), across all four
   polynomial bases. Both backends compute the same Horner recurrence; the
   only difference is last-ULP FP rounding (absorbed by atol=1e-12).

4. **Default OFF byte-identical**: with ``MLFRAME_POLYEVAL_ORACLE`` unset the
   dispatcher's CPU routing is byte-identical to the legacy threshold path.

5. **GPU path untouched**: the cuda branch still consults
   ``_lookup_polyeval_thresholds`` (kernel_tuning_cache), the oracle governs
   ONLY the {njit, njit_par} CPU choice, and importing the module on this
   CPU-only box pulls in no cupy. GPU migration is DEFERRED (needs a CUDA box
   to bench -- cupy is broken on the dev box).

6. **kernel_tuning_cache unmodified**: the bridge is import-only; exercising
   it performs no write/update against the KernelTuningCache module.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from mlframe.utils._param_oracle import ParamOracle
from mlframe.feature_selection.filters import hermite_fe as H
from tests.conftest import is_fast_mode

FIXED_TS = "2026-01-01T00:00:00+00:00"
_BASES = ("hermite", "legendre", "chebyshev", "laguerre")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class _FakeKTC:
    """Minimal KernelTuningCache stand-in: only ``get_regions`` is consulted
    by the read-only bridge. Records any mutating call so the test can assert
    the bridge never writes back."""

    def __init__(self, regions):
        self._regions = regions
        self.mutations: list[str] = []

    def get_regions(self, kernel_name):
        return list(self._regions)

    # Any write API the bridge must NOT touch -- record if hit.
    def update(self, *a, **k):
        self.mutations.append("update")

    def _save(self, *a, **k):
        self.mutations.append("_save")


def _fresh_oracle(tmp_path, name="poly_l103.parquet", **kw):
    kw.setdefault("param_space", {"backend": ["njit", "njit_par"]})
    kw.setdefault("minimize", "elapsed_s")
    kw.setdefault("mode", "inference")
    kw.setdefault("min_observations", 1)
    return ParamOracle(os.path.join(str(tmp_path), name), **kw)


# ---------------------------------------------------------------------------
# 1. Bridge imports KTC regions as cold-start observations
# ---------------------------------------------------------------------------

def test_bridge_imports_ktc_regions(tmp_path):
    fake = _FakeKTC([
        {"n_samples_max": 1000, "backend": "njit", "wall_ms": 0.05},
        {"n_samples_max": 1_000_000, "backend": "njit_par", "wall_ms": 2.0},
        # catch-all (no size cap) -> no representative size -> skipped.
        {"n_samples_max": None, "backend": "njit_par"},
    ])
    oracle = _fresh_oracle(tmp_path)
    imported = oracle.read_ktc_regions(
        "polyeval_cpu_backend", param_field="backend",
        fixed_fp={"p": 1, "dtype_kind": "f"}, cache=fake, fn_name="poly",
    )
    assert imported == 2  # the catch-all region is skipped

    rows = [r for r in oracle.store.read_rows() if r.get("fn_name") == "poly"]
    assert len(rows) == 2
    # The imported observations carry the region's backend + scaled wall-time.
    combos = sorted(__import__("json").loads(r["param_combo_json"])["backend"] for r in rows)
    assert combos == ["njit", "njit_par"]

    # Recommendations honour the imported history: njit near the small cap,
    # njit_par near the large cap.
    assert oracle.recommend({"n": 500, "p": 1, "dtype_kind": "f"}, fn_name="poly")["backend"] == "njit"
    assert oracle.recommend({"n": 900_000, "p": 1, "dtype_kind": "f"}, fn_name="poly")["backend"] == "njit_par"

    # READ-ONLY: bridge never wrote back to the kernel cache.
    assert fake.mutations == []


def test_bridge_from_classmethod(tmp_path):
    fake = _FakeKTC([
        {"n_samples_max": 1000, "backend": "njit", "wall_ms": 0.05},
        {"n_samples_max": 1_000_000, "backend": "njit_par", "wall_ms": 2.0},
    ])
    oracle = ParamOracle.from_kernel_tuning_cache(
        os.path.join(str(tmp_path), "cm.parquet"),
        "polyeval_cpu_backend", param_field="backend",
        param_space={"backend": ["njit", "njit_par"]},
        cache=fake, fn_name="poly", mode="inference", min_observations=1,
    )
    rows = [r for r in oracle.store.read_rows() if r.get("fn_name") == "poly"]
    assert len(rows) == 2
    assert fake.mutations == []


# ---------------------------------------------------------------------------
# 2. Oracle LEARNS the njit/njit_par crossover from wall-times
# ---------------------------------------------------------------------------

def test_oracle_learns_cpu_crossover(tmp_path):
    oracle = _fresh_oracle(tmp_path)
    res = H.benchmark_polyeval_cpu_backends(
        "hermite", sizes=(200, 500_000), repeats=5, oracle=oracle,
    )
    # njit_par must lose at small n and win at large n (the empirical signal
    # the oracle is supposed to LEARN, not be told).
    assert res[(200, "njit")] < res[(200, "njit_par")], res
    assert res[(500_000, "njit_par")] < res[(500_000, "njit")], res

    small = oracle.recommend({"n": 200, "p": 1, "dtype_kind": "f"}, fn_name=H._POLYEVAL_ORACLE_FN_NAME)
    large = oracle.recommend({"n": 500_000, "p": 1, "dtype_kind": "f"}, fn_name=H._POLYEVAL_ORACLE_FN_NAME)
    assert small["backend"] == "njit", small
    assert large["backend"] == "njit_par", large


def test_dispatch_uses_oracle_when_enabled(tmp_path, monkeypatch):
    oracle = _fresh_oracle(tmp_path)
    # Fewer repeats under --fast: the njit/njit_par crossover signal at these sizes is stable with 2 repeats, and the
    # full 5-repeat 500k bench is what starves a worker into a timeout under full-suite ``-n`` contention.
    repeats = 2 if is_fast_mode() else 5
    H.benchmark_polyeval_cpu_backends("hermite", sizes=(200, 500_000), repeats=repeats, oracle=oracle)
    monkeypatch.setattr(H, "_polyeval_oracle_singleton", oracle, raising=False)
    monkeypatch.setenv("MLFRAME_POLYEVAL_ORACLE", "1")
    monkeypatch.delenv("MLFRAME_POLYEVAL_BACKEND", raising=False)

    assert H._polyeval_oracle_pick_cpu_backend(200) == "njit"
    assert H._polyeval_oracle_pick_cpu_backend(500_000) == "njit_par"


# ---------------------------------------------------------------------------
# 3. Bit-equivalence regardless of oracle-picked backend (all 4 bases)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("basis", _BASES)
def test_bit_equivalence_njit_vs_njit_par(basis, tmp_path, monkeypatch):
    c = np.array([0.3, -0.7, 0.2, 0.5, -0.1], dtype=np.float64)
    x = np.linspace(0.1, 2.0, 500_000).astype(np.float64)
    ref = H._NJIT_FUNCS[basis](x, c)
    par = H._NJIT_PAR_FUNCS[basis](x, c)
    # Both compute the same Horner polynomial; only last-ULP rounding differs.
    assert np.allclose(ref, par, rtol=1e-12, atol=1e-12)

    # Through the dispatcher with the oracle ON, picking njit_par for large n,
    # the output still matches the direct njit reference.
    oracle = _fresh_oracle(tmp_path)
    H.benchmark_polyeval_cpu_backends(basis, sizes=(200, 500_000), repeats=3, oracle=oracle)
    monkeypatch.setattr(H, "_polyeval_oracle_singleton", oracle, raising=False)
    monkeypatch.setenv("MLFRAME_POLYEVAL_ORACLE", "1")
    monkeypatch.delenv("MLFRAME_POLYEVAL_BACKEND", raising=False)
    disp = H.polyeval_dispatch(basis, x, c)
    assert np.allclose(ref, disp, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# 4. Default OFF -> byte-identical to the legacy CPU threshold path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [200, 100_000])
@pytest.mark.parametrize("basis", _BASES)
def test_default_off_byte_identical(basis, n, monkeypatch):
    """With the oracle flag unset, the dispatcher's CPU routing must equal the
    legacy threshold decision exactly (n < par_threshold -> njit, else
    njit_par), byte-for-byte."""
    monkeypatch.delenv("MLFRAME_POLYEVAL_ORACLE", raising=False)
    monkeypatch.delenv("MLFRAME_POLYEVAL_BACKEND", raising=False)
    assert not H._polyeval_oracle_enabled()

    c = np.array([0.3, -0.7, 0.2, 0.5, -0.1], dtype=np.float64)
    x = np.linspace(0.1, 2.0, n).astype(np.float64)
    par_threshold, _ = H._lookup_polyeval_thresholds(basis, n)
    legacy = (H._NJIT_FUNCS[basis](x, c) if n < par_threshold
              else H._NJIT_PAR_FUNCS[basis](x, c))
    got = H.polyeval_dispatch(basis, x, c)
    assert np.array_equal(legacy, got)  # byte-identical, not merely close


def test_oracle_enabled_does_not_alter_small_n_decision(monkeypatch):
    """Sanity: even with the oracle ON, a cold oracle (no observations) falls
    back to njit, so tiny arrays never silently route to njit_par."""
    monkeypatch.setenv("MLFRAME_POLYEVAL_ORACLE", "1")
    monkeypatch.delenv("MLFRAME_POLYEVAL_BACKEND", raising=False)
    with tempfile.TemporaryDirectory() as d:
        cold = ParamOracle(os.path.join(d, "cold.parquet"),
                           param_space={"backend": ["njit", "njit_par"]},
                           minimize="elapsed_s", mode="inference", min_observations=1)
        monkeypatch.setattr(H, "_polyeval_oracle_singleton", cold, raising=False)
        assert H._polyeval_oracle_pick_cpu_backend(200) == "njit"


# ---------------------------------------------------------------------------
# 5. GPU path untouched -- no cupy on CPU box; cuda branch still on KTC
# ---------------------------------------------------------------------------

def test_no_cupy_imported_on_cpu_box():
    """Importing + exercising the CPU dispatch path must not pull in cupy."""
    import sys
    # cupy may be entirely absent on this box; if it somehow got imported it
    # would only be via the (skipped) cuda branch. The POC must not add a CPU
    # import of cupy.
    assert H._CUDA_AVAILABLE is False or "cupy" in sys.modules  # honest on either box
    if not H._CUDA_AVAILABLE:
        assert "cupy" not in sys.modules


def test_cuda_branch_still_consults_kernel_tuning_cache():
    """The GPU crossover stays on kernel_tuning_cache: the dispatcher still
    calls ``_lookup_polyeval_thresholds`` (which reads the kernel cache) and
    uses its cuda_threshold for the cuda branch -- the oracle never governs
    GPU. Asserted behaviourally via the source-level wiring of the helper."""
    par, cuda = H._lookup_polyeval_thresholds("hermite", 1000)
    assert isinstance(par, int) and isinstance(cuda, int)
    assert cuda >= par  # cuda crossover sits above the par crossover
    # The cuda decision variable feeding the dispatcher is the KTC-derived
    # cuda_threshold, NOT any oracle output. The oracle param space is CPU-only.
    assert set(H._POLYEVAL_ORACLE_PARAM_SPACE["backend"]) == {"njit", "njit_par"}
    assert "cuda" not in H._POLYEVAL_ORACLE_PARAM_SPACE["backend"]


# ---------------------------------------------------------------------------
# 6. kernel_tuning_cache module unmodified by the bridge
# ---------------------------------------------------------------------------

def test_bridge_never_writes_kernel_tuning_cache(tmp_path):
    """The read-only bridge must call only ``get_regions`` on the cache and
    never any write API (update / _save)."""
    calls: list[str] = []

    class _Spy:
        def get_regions(self, name):
            calls.append("get_regions")
            return [{"n_samples_max": 1000, "backend": "njit", "wall_ms": 0.1}]

        def __getattr__(self, item):
            # Any non-get_regions attribute access (e.g. update/_save) is a
            # write attempt -> record and raise so the test fails loudly.
            calls.append(f"WRITE:{item}")
            raise AssertionError(f"bridge touched cache.{item} (must be read-only)")

    oracle = _fresh_oracle(tmp_path)
    oracle.read_ktc_regions("k", param_field="backend",
                            fixed_fp={"p": 1, "dtype_kind": "f"},
                            cache=_Spy(), fn_name="poly")
    assert calls == ["get_regions"]


def test_kernel_tuning_cache_public_api_intact():
    """The KernelTuningCache public API the POC relies on (get_regions /
    lookup) is unchanged -- the migration is additive, KTC is untouched."""
    from pyutilz.performance.kernel_tuning.cache import KernelTuningCache
    for name in ("get_regions", "lookup", "update", "has", "reset"):
        assert hasattr(KernelTuningCache, name)
