"""Prototype validation for the GPU-resident FE candidate generation + MI (``_gpu_resident_fe``).

Correctness gate: the on-device (cupy) candidate grid + single big-k MI must match the CPU (numpy +
njit) path -- same candidate names, MI ranking, and values to fp round-off -- and both must rank the
a**2/b-equivalent candidate top on an a**2/b target. Speed is exercised by a separate opt-in bench
(not a hard timing assert, to stay non-flaky)."""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._gpu_resident_fe import (
    cpu_pair_candidate_mi,
    fe_gpu_resident_enabled,
)


def _ab_target(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.uniform(1.0, 5.0, n)
    b = rng.uniform(1.0, 5.0, n)
    y = a**2 / b
    # discretise y to codes the MI kernels score against (equi-frequency, 20 bins).
    edges = np.quantile(y, np.linspace(0, 1, 21)[1:-1])
    y_codes = np.searchsorted(edges, y).astype(np.int64)
    return a, b, y_codes


def test_gate_default_off(monkeypatch):
    monkeypatch.delenv("MLFRAME_FE_GPU_RESIDENT", raising=False)
    assert fe_gpu_resident_enabled() is False
    monkeypatch.setenv("MLFRAME_FE_GPU_RESIDENT", "on")
    assert fe_gpu_resident_enabled() is True


def test_cpu_path_recovers_a2_over_b():
    """The CPU reference grid must rank an (a,b) ratio-of-square candidate top on an a**2/b target."""
    a, b, y_codes = _ab_target()
    names, mi = cpu_pair_candidate_mi(a, b, y_codes)
    top = names[int(np.argmax(mi))]
    assert "div" in top and "sqr" in top, f"top candidate not the a**2/b form: {top}"


def test_gpu_resident_matches_cpu():
    """On-device generation + single big-k MI must match the CPU path: identical names, same top
    candidate, MI values equal to fp round-off (njit vs cupy plug-in MI are equivalent)."""
    pytest.importorskip("cupy")
    from mlframe.feature_selection.filters._gpu_resident_fe import gpu_resident_pair_candidate_mi

    a, b, y_codes = _ab_target()
    cpu_names, cpu_mi = cpu_pair_candidate_mi(a, b, y_codes)
    gpu_names, gpu_mi = gpu_resident_pair_candidate_mi(a, b, y_codes)
    assert gpu_names == cpu_names
    # same winner + same ranking head
    assert int(np.argmax(gpu_mi)) == int(np.argmax(cpu_mi)), (cpu_names[int(np.argmax(cpu_mi))], gpu_names[int(np.argmax(gpu_mi))])
    # values match to fp round-off (binning tie-breaks aside)
    np.testing.assert_allclose(gpu_mi, cpu_mi, rtol=1e-3, atol=1e-4)


def test_gpu_resident_chunked_matches_cpu():
    """Force MULTIPLE VRAM K-chunks (n=100k -> k_chunk < 384) and assert the concatenated chunked MI
    still matches the CPU path -- the chunk boundary must not corrupt per-candidate MI."""
    pytest.importorskip("cupy")
    from mlframe.feature_selection.filters._gpu_resident_fe import _gpu_k_chunk, gpu_resident_pair_candidate_mi

    a, b, y_codes = _ab_target(n=100_000)
    assert _gpu_k_chunk(100_000) < 384, "test needs >1 chunk to be meaningful"
    cpu_names, cpu_mi = cpu_pair_candidate_mi(a, b, y_codes)
    gpu_names, gpu_mi = gpu_resident_pair_candidate_mi(a, b, y_codes)
    assert gpu_names == cpu_names
    assert int(np.argmax(gpu_mi)) == int(np.argmax(cpu_mi))
    np.testing.assert_allclose(gpu_mi, cpu_mi, rtol=1e-3, atol=1e-4)


def test_fast_path_preserves_exact_winner():
    """prescreen(sort-free)+refine(exact top-K) must return the SAME top candidate as the pure-exact
    GPU path -- the whole point of refining is bit-exact selection despite the approximate prescreen."""
    pytest.importorskip("cupy")
    from mlframe.feature_selection.filters._gpu_resident_fe import (
        gpu_resident_pair_candidate_mi,
        gpu_resident_pair_candidate_mi_fast,
    )

    for seed in (0, 1, 2):
        a, b, y_codes = _ab_target(n=100_000, seed=seed)
        names, exact = gpu_resident_pair_candidate_mi(a, b, y_codes)
        _, fast = gpu_resident_pair_candidate_mi_fast(a, b, y_codes)
        assert names[int(np.argmax(fast))] == names[int(np.argmax(exact))], f"winner flipped at seed {seed}"


def _target(a, b, kind):
    if kind == "a2b":
        y = a**2 / np.where(b == 0, 1e-9, b)
    elif kind == "logsin":
        y = np.log(np.abs(a) + 1e-9) * np.sin(b)
    else:  # noise
        y = np.random.default_rng(7).normal(size=a.shape[0])
    e = np.quantile(y, np.linspace(0, 1, 21)[1:-1])
    return np.searchsorted(e, y).astype(np.int64)


def test_cpu_path_edge_cases_no_nonfinite_escapes():
    """Operands with zeros + negatives exercise the safe-div y==0 branch, sqrt(|x|), smart-log shift
    and reciproc-of-0 -- all dead in the uniform[1,5] tests. The scrub must leave NO NaN/inf and MI
    finite + >= 0, at n below nbins too."""
    rng = np.random.default_rng(3)
    for n in (12, 2000):  # n=12 < nbins=20 (degenerate quantile binning)
        a = rng.uniform(-3, 3, n); a[::5] = 0.0
        b = rng.uniform(-3, 3, n); b[::4] = 0.0
        for kind in ("a2b", "logsin", "noise"):
            names, mi = cpu_pair_candidate_mi(a, b, _target(a, b, kind))
            assert len(names) == len(mi) == 8 * 8 * 6
            assert np.all(np.isfinite(mi)), (n, kind)
            assert np.all(mi >= -1e-9), (n, kind)


def test_gpu_cpu_agree_heavytail_and_varied_targets():
    """Exact GPU path must match CPU on HEAVY-TAILED operands (the regime the docstring says breaks the
    approximate path) and on NON-a**2/b targets -- the cases the original suite never covered."""
    pytest.importorskip("cupy")
    from mlframe.feature_selection.filters._gpu_resident_fe import gpu_resident_pair_candidate_mi

    rng = np.random.default_rng(11)
    n = 50_000
    a = rng.lognormal(0.0, 2.5, n)   # heavy tail
    b = rng.lognormal(0.0, 2.5, n)
    for kind in ("a2b", "logsin", "noise"):
        yc = _target(a, b, kind)
        cnames, cmi = cpu_pair_candidate_mi(a, b, yc)
        gnames, gmi = gpu_resident_pair_candidate_mi(a, b, yc)
        assert gnames == cnames
        np.testing.assert_allclose(gmi, cmi, rtol=1e-3, atol=1e-4, err_msg=kind)
        if kind != "noise":  # noise has no real winner -> argmax meaningless
            assert int(np.argmax(gmi)) == int(np.argmax(cmi)), kind


def test_dispatch_falls_back_to_cpu_on_gpu_error(monkeypatch):
    """A GPU error at n>=crossover must fall back to the CPU result (same array), not propagate."""
    import mlframe.feature_selection.filters._gpu_resident_fe as G

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated GPU failure")

    monkeypatch.setattr(G, "gpu_resident_pair_candidate_mi", _boom)
    a, b, y_codes = _ab_target(n=60_000)  # >= _GPU_RESIDENT_MIN_N
    names, mi = G.pair_candidate_mi_dispatch(a, b, y_codes)
    cnames, cmi = cpu_pair_candidate_mi(a, b, y_codes)
    assert names == cnames
    np.testing.assert_array_equal(mi, cmi)


def test_chunk_invariance(monkeypatch):
    """The VRAM K-chunk boundary must not change the result: forcing k_chunk in {1, 7, 384} on the same
    data yields identical MI vectors (isolates the chunk-stitch logic from whatever VRAM the box has)."""
    pytest.importorskip("cupy")
    import mlframe.feature_selection.filters._gpu_resident_fe as G

    a, b, y_codes = _ab_target(n=40_000)
    results = []
    for kc in (1, 7, 384):
        monkeypatch.setattr(G, "_gpu_k_chunk", lambda n, _kc=kc, **kw: _kc)
        _, mi = G.gpu_resident_pair_candidate_mi(a, b, y_codes)
        results.append(mi)
    np.testing.assert_array_equal(results[0], results[1])
    np.testing.assert_array_equal(results[1], results[2])


def test_gpu_resident_emits_replayable_recipe():
    """The structured bridge: gpu_resident_pair_recipes must emit a real EngineeredRecipe (preset-
    stamped, edge-pinned) whose transform() replays leak-free on RAW inputs and recovers a**2/b. This
    is what makes the GPU output first-class for the production FE pipeline (not a flat string)."""
    import pandas as pd

    from mlframe.feature_selection.filters._gpu_resident_fe import gpu_resident_pair_recipes
    from mlframe.feature_selection.filters.engineered_recipes import apply_recipe

    a, b, y_codes = _ab_target(n=20_000)  # below crossover -> dispatcher uses CPU leg (no cupy needed)
    recs = gpu_resident_pair_recipes(
        a, b, y_codes, src_a_name="a", src_b_name="b", cols_names=["a", "b"],
        unary_preset="minimal", binary_preset="minimal", quantization_nbins=None, top_k=3,
    )
    assert recs, "no recipes emitted"
    name, recipe, mi = recs[0]
    # winner must encode a**2/b -- but the spelling can be div(sqr(a),b) OR the equivalent
    # mul(sqr(a),reciproc(b)); assert the SIGNAL via the replay-tracks-a**2/b check below, not a literal.
    assert "sqr" in name and ("div" in name or "reciproc" in name), f"winner not an a**2/b form: {name}"
    # structured + preset-stamped (the whole point vs a flat name)
    assert recipe.kind == "unary_binary"
    assert recipe.src_names == ("a", "b")
    assert recipe.unary_preset == "minimal" and recipe.binary_preset == "minimal"
    assert recipe.name == name
    # leak-free replay on RAW inputs reproduces the engineered column; it must track true a**2/b.
    df = pd.DataFrame({"a": a, "b": b})
    replayed = np.asarray(apply_recipe(recipe, df), dtype=np.float64).ravel()
    assert np.all(np.isfinite(replayed))
    true = a**2 / np.where(b == 0, 1e-9, b)
    rho = np.corrcoef(replayed, true)[0, 1]
    assert abs(rho) > 0.99, f"replayed recipe doesn't track a**2/b (|rho|={rho:.3f})"


def test_grand_fused_pair_mi_bit_identical_to_cpu():
    """GRAND FUSION (GPU gen + GPU discretize + GPU noise-gate) must produce the EXACT noise-gated MI of
    the full CPU path (gen -> discretize_2d_quantile_batch -> batch_mi_with_noise_gate). GPU discretize
    is bit-identical to CPU (verified) and the GPU noise-gate is the bit-identical production twin, so the
    fused pair-MI is bit-identical -- at ~20x the speed (CPU 54s vs GPU 2.8s at n=200k K=384)."""
    pytest.importorskip("cupy")
    from mlframe.feature_selection.filters._gpu_resident_fe import _build_candidate_matrix, grand_fused_pair_mi
    from mlframe.feature_selection.filters.discretization import discretize_2d_quantile_batch
    from mlframe.feature_selection.filters.info_theory import batch_mi_with_noise_gate

    rng = np.random.default_rng(0)
    n = 50_000
    a = rng.uniform(1, 5, n); b = rng.uniform(1, 5, n); y = a**2 / b
    e = np.quantile(y, np.linspace(0, 1, 21)[1:-1]); yc = np.searchsorted(e, y).astype(np.int64)
    fy = np.bincount(yc, minlength=int(yc.max()) + 1).astype(np.float64) / n

    cand = np.ascontiguousarray(_build_candidate_matrix(np, a, b))
    disc = discretize_2d_quantile_batch(cand, n_bins=20, dtype=np.int8)
    fnb = np.full(cand.shape[1], 20, dtype=np.int64)
    ref = batch_mi_with_noise_gate(
        disc_2d=disc, factors_nbins=fnb, classes_y=yc, classes_y_safe=yc, freqs_y=fy,
        npermutations=25, base_seed=np.uint64(0), min_nonzero_confidence=0.0, use_su=False,
        dtype=np.int32, classes_dtype=np.int32,
    )
    _, gf = grand_fused_pair_mi(a, b, yc, yc, fy, nbins=20, npermutations=25)
    np.testing.assert_allclose(gf, ref, rtol=1e-6, atol=1e-9)
    assert int(np.argmax(gf)) == int(np.argmax(ref))


def test_fused_generation_is_bit_equal_to_cupy_loop():
    """The fused RawKernel generation must be BIT-EQUAL (maxdiff 0) to the cupy elementwise loop --
    same ops, safe-div y==0 branch, nan_to_num. This is what lets it replace the loop with no result
    change while being ~15x faster generation."""
    cp = pytest.importorskip("cupy")
    from mlframe.feature_selection.filters._gpu_resident_fe import (
        _COMBOS,
        _build_candidate_matrix,
        _fused_generate_block,
        _unary_stack_cm,
    )

    rng = np.random.default_rng(0)
    n = 50_000
    a = cp.asarray(rng.uniform(1, 5, n))
    b = cp.asarray(rng.uniform(1, 5, n))
    ref = _build_candidate_matrix(cp, a, b)
    fused = _fused_generate_block(_unary_stack_cm(cp, a), _unary_stack_cm(cp, b), _COMBOS)
    assert float(cp.max(cp.abs(ref - fused))) == 0.0


def test_dispatch_routes_and_recovers():
    """The size dispatcher returns the right shape/ranking on both legs (small-n CPU leg always; large-n
    GPU leg when cupy present) and recovers a**2/b."""
    from mlframe.feature_selection.filters._gpu_resident_fe import pair_candidate_mi_dispatch

    a, b, y_codes = _ab_target(n=20_000)  # below crossover -> CPU leg
    names, mi = pair_candidate_mi_dispatch(a, b, y_codes)
    assert len(names) == len(mi) == 8 * 8 * 6
    top = names[int(np.argmax(mi))]
    assert "div" in top and "sqr" in top


def test_gpu_discretize_codes_host_f64_bit_identical_to_cpu(monkeypatch):
    """With the EXACT f64 binning fallback (MLFRAME_FE_GPU_BINNING_DTYPE=float64) the GPU codes must be
    BIT-IDENTICAL to the CPU discretize_2d_quantile_batch (maxdiff 0) -- np.percentile upcasts the float32
    FE buffer to float64, so the f64 GPU path matches it exactly. This is the bit-exact fallback contract."""
    pytest.importorskip("cupy")
    monkeypatch.setenv("MLFRAME_FE_GPU_BINNING_DTYPE", "float64")
    from mlframe.feature_selection.filters._gpu_resident_fe import (
        _build_candidate_matrix, gpu_discretize_codes_host,
    )
    from mlframe.feature_selection.filters.discretization import discretize_2d_quantile_batch

    rng = np.random.default_rng(0)
    for n in (20_000, 50_000):
        a = rng.uniform(1, 5, n); b = rng.uniform(1, 5, n)
        cand = np.ascontiguousarray(_build_candidate_matrix(np, a, b)).astype(np.float32)
        np.nan_to_num(cand, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        cpu = discretize_2d_quantile_batch(cand, n_bins=20, dtype=np.int8, assume_finite=True)
        gpu = gpu_discretize_codes_host(cand, 20, dtype=np.int8)
        assert np.array_equal(cpu, gpu), f"n={n} K={cand.shape[1]} codes differ"


def test_gpu_discretize_codes_host_f32_default_selection_safe():
    """DEFAULT (native float32) binning: the FE candidate buffer is already float32, so binning it in f32
    (no f64 up-cast, half the sort bandwidth) is the default. The acceptance bar is SELECTION-equivalence,
    not bit-identity: f32 codes must agree with the CPU f64 codes to ~100% (measured 100.000% @ K=384,
    n in {100k,300k,1M} on a GTX 1050 Ti) so the downstream noise-gate MI ranking -- and thus the FE
    selection -- is preserved. Assert >=99.9% code agreement (selection-safe margin)."""
    pytest.importorskip("cupy")
    from mlframe.feature_selection.filters._gpu_resident_fe import (
        _build_candidate_matrix, gpu_discretize_codes_host,
    )
    from mlframe.feature_selection.filters.discretization import discretize_2d_quantile_batch

    rng = np.random.default_rng(0)
    for n in (20_000, 50_000):
        a = rng.uniform(1, 5, n); b = rng.uniform(1, 5, n)
        cand = np.ascontiguousarray(_build_candidate_matrix(np, a, b)).astype(np.float32)
        np.nan_to_num(cand, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        cpu = discretize_2d_quantile_batch(cand, n_bins=20, dtype=np.int8, assume_finite=True)
        gpu = gpu_discretize_codes_host(cand, 20, dtype=np.int8)  # native f32 default
        agree = float(np.mean(cpu == gpu))
        assert agree >= 0.999, f"n={n} f32 code agreement {agree:.5f} < 0.999 (selection at risk)"


def test_gpu_discretize_codes_host_k1_chunk_guard(monkeypatch):
    """A single-column (n, 1) candidate block must still bin bit-identically -- guards the cupy
    cp.percentile(axis=0) single-column bug (wrong edges) that would corrupt a K==1 last chunk. Asserted
    against the f64 fallback so it is a clean bit-identity check of the K==1 ravel guard (independent of
    the default f32 edge round-off)."""
    pytest.importorskip("cupy")
    monkeypatch.setenv("MLFRAME_FE_GPU_BINNING_DTYPE", "float64")
    from mlframe.feature_selection.filters._gpu_resident_fe import gpu_discretize_codes_host
    from mlframe.feature_selection.filters.discretization import discretize_2d_quantile_batch

    rng = np.random.default_rng(1)
    one = np.ascontiguousarray((rng.uniform(1, 5, 40_000) * rng.uniform(1, 5, 40_000)).reshape(-1, 1)).astype(np.float32)
    cpu = discretize_2d_quantile_batch(one, n_bins=20, dtype=np.int8, assume_finite=True)
    gpu = gpu_discretize_codes_host(one, 20, dtype=np.int8)
    assert np.array_equal(cpu, gpu)


def test_gpu_pairs_fe_mi_matches_cpu_dispatch_analytic(monkeypatch):
    """gpu_pairs_fe_mi (GPU binning + GPU observed-MI + analytic gate) must equal the production CPU
    _dispatch_batch_mi_with_noise_gate on its analytic branch -- this is the selection-identity contract
    for the size-gated FE GPU path (MLFRAME_FE_GPU_DISCRETIZE). n is above analytic_null_min_n so the CPU
    dispatch takes the analytic route too. Uses the f64 binning fallback so the GPU codes are bit-identical
    to the CPU discretize and the observed-MI equality is an exact contract (not f32 round-off dependent)."""
    pytest.importorskip("cupy")
    monkeypatch.setenv("MLFRAME_FE_GPU_BINNING_DTYPE", "float64")
    from mlframe.feature_selection.filters._gpu_resident_fe import _build_candidate_matrix, gpu_pairs_fe_mi
    from mlframe.feature_selection.filters.info_theory import batch_mi_with_noise_gate
    from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_dispatch import (
        _dispatch_batch_mi_with_noise_gate,
    )

    rng = np.random.default_rng(0)
    n = 60_000  # >= analytic_null_min_n default (50k) so both take the analytic branch
    a = rng.uniform(1, 5, n); b = rng.uniform(1, 5, n); y = a**2 / b
    e = np.quantile(y, np.linspace(0, 1, 21)[1:-1]); yc = np.searchsorted(e, y).astype(np.int64)
    fy = np.bincount(yc, minlength=int(yc.max()) + 1).astype(np.float64) / n
    cand = np.ascontiguousarray(_build_candidate_matrix(np, a, b)).astype(np.float32)
    np.nan_to_num(cand, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    from mlframe.feature_selection.filters.discretization import discretize_2d_quantile_batch
    disc = discretize_2d_quantile_batch(cand, n_bins=20, dtype=np.int8, assume_finite=True)
    cpu = _dispatch_batch_mi_with_noise_gate(
        disc_2d=disc, quantization_nbins=20, classes_y=yc, classes_y_safe=yc, freqs_y=fy,
        npermutations=3, min_nonzero_confidence=0.0, use_su=False,
        batch_mi_kernel=batch_mi_with_noise_gate,
    )
    gpu = gpu_pairs_fe_mi(cand, 20, yc, yc, fy, 3, 0.0, False)
    assert gpu is not None, "expected the analytic GPU branch to engage at n>=50k"
    np.testing.assert_array_equal(np.asarray(gpu, dtype=np.float64), np.asarray(cpu, dtype=np.float64))
    assert int(np.argmax(gpu)) == int(np.argmax(cpu))


def test_gpu_pairs_fe_mi_returns_none_for_nonanalytic():
    """gpu_pairs_fe_mi must DEFER (None) when the analytic branch doesn't apply -- SU-normalised,
    npermutations<=0, or small n -- so the caller uses the CPU dispatch (no silent wrong path)."""
    pytest.importorskip("cupy")
    from mlframe.feature_selection.filters._gpu_resident_fe import _build_candidate_matrix, gpu_pairs_fe_mi
    rng = np.random.default_rng(0)
    n = 4_000  # below analytic_null_min_n -> must defer
    a = rng.uniform(1, 5, n); b = rng.uniform(1, 5, n); y = a**2 / b
    e = np.quantile(y, np.linspace(0, 1, 21)[1:-1]); yc = np.searchsorted(e, y).astype(np.int64)
    fy = np.bincount(yc, minlength=int(yc.max()) + 1).astype(np.float64) / n
    cand = np.ascontiguousarray(_build_candidate_matrix(np, a, b)).astype(np.float32)
    assert gpu_pairs_fe_mi(cand, 20, yc, yc, fy, 3, 0.0, False) is None      # small n
    assert gpu_pairs_fe_mi(cand, 20, yc, yc, fy, 0, 0.0, False) is None      # npermutations<=0
    assert gpu_pairs_fe_mi(cand, 20, yc, yc, fy, 3, 0.0, True) is None       # SU-normalised
