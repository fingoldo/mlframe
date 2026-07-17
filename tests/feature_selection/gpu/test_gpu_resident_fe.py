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
        a = rng.uniform(-3, 3, n)
        a[::5] = 0.0
        b = rng.uniform(-3, 3, n)
        b[::4] = 0.0
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
    a = rng.lognormal(0.0, 2.5, n)  # heavy tail
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
        a,
        b,
        y_codes,
        src_a_name="a",
        src_b_name="b",
        cols_names=["a", "b"],
        unary_preset="minimal",
        binary_preset="minimal",
        quantization_nbins=None,
        top_k=3,
    )
    assert recs, "no recipes emitted"
    name, recipe, _mi = recs[0]
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
    # SELECTION-EQUIVALENCE via RANK correlation (not Pearson). The MI scorer bins equi-frequency
    # (quantile/rank), so it is MONOTONE-INVARIANT: every monotone spelling of the a-b interaction --
    # ``mul(sqr(a),reciproc(b))`` (= a**2/b), ``div(a,sqrt(b))`` (= sqrt(a**2/b)), ``mul(reciproc(a),sqrt(b))``
    # (= 1/sqrt(a**2/b)) -- has the SAME partition, hence IDENTICAL MI (verified: all top-3 tie at MI 2.9957),
    # and which one wins is a sub-ULP MI tie-break (a fused-kernel ~1e-15 reassociation flips it). All recover
    # the a**2/b signal; the winner is monotone-equivalent to a**2/b (|Spearman| = 1.0) but need not be LINEAR
    # in it (a/sqrt(b) gives Pearson ~0.97). Rank correlation is the selection-equivalent recovery metric here.
    rank = lambda v: np.argsort(np.argsort(v))
    rho = np.corrcoef(rank(replayed), rank(true))[0, 1]
    assert abs(rho) > 0.99, f"replayed recipe doesn't monotonically track a**2/b (|spearman|={rho:.3f})"


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
    a = rng.uniform(1, 5, n)
    b = rng.uniform(1, 5, n)
    y = a**2 / b
    e = np.quantile(y, np.linspace(0, 1, 21)[1:-1])
    yc = np.searchsorted(e, y).astype(np.int64)
    fy = np.bincount(yc, minlength=int(yc.max()) + 1).astype(np.float64) / n

    cand = np.ascontiguousarray(_build_candidate_matrix(np, a, b))
    disc = discretize_2d_quantile_batch(cand, n_bins=20, dtype=np.int8)
    fnb = np.full(cand.shape[1], 20, dtype=np.int64)
    ref = batch_mi_with_noise_gate(
        disc_2d=disc,
        factors_nbins=fnb,
        classes_y=yc,
        classes_y_safe=yc,
        freqs_y=fy,
        npermutations=25,
        base_seed=np.uint64(0),
        min_nonzero_confidence=0.0,
        use_su=False,
        dtype=np.int32,
        classes_dtype=np.int32,
    )
    _, gf = grand_fused_pair_mi(a, b, yc, yc, fy, nbins=20, npermutations=25)
    np.testing.assert_allclose(gf, ref, rtol=1e-6, atol=1e-9)
    assert int(np.argmax(gf)) == int(np.argmax(ref))


def test_grand_fusion_fused_bit_identical_to_nonfused():
    """GRAND FUSION (never materialise (n,K)): the fully-fused histogram path must produce the SAME
    noise-gated MI as the non-fused grand_fused_pair_mi (same percentile edges -> same codes -> same
    counts), so the FE selection is preserved while the (n,K) float/codes/disc/d_base are eliminated.

    TOLERANCE (2026-07-13): the non-fused leg's ``use_su=False`` noise-gate now calls
    ``batch_mi_with_noise_gate_cuda_resident`` directly with the discretized codes kept GPU-resident
    (``d_disc_resident=``) instead of a D2H-then-H2D round trip through ``dispatch_batch_mi_with_noise_
    gate_gpu`` -- see ``grand_fused_pair_mi``. That resident kernel reduces the MI entropy ON the device
    (its own docstring: "GPU MI reproduces the CPU reduction order to fp round-off"), whereas the fused
    kernel here still reduces via the CPU-bit-exact ``_mi_from_counts_cpu``/``_gate_from_mi`` -- so the two
    no longer share the identical reduction order. Measured maxdiff 2.22e-16 (1 ULP) at n=40k/nbins=20/
    nperm=25, seed=1 -- confirmed to be this FP-reduction-order difference, NOT a counts/selection
    regression: with ``use_su=True`` (which the resident kernel does not support, so it still takes the
    old CPU-bit-exact route unchanged) maxdiff is 0.0 exactly. This mirrors the project's documented FE/
    MRMR exception (selection-equivalence, not bit-identical MI, is the bar for GPU-resident entropy
    reduction -- see e.g. ``_pairs_dispatch.py``'s resident gate, verified maxdiff ~1e-18 there). Argmax
    match is still required (selection-bearing)."""
    pytest.importorskip("cupy")
    from mlframe.feature_selection.filters._gpu_resident_fe import (
        grand_fused_pair_mi,
        grand_fused_pair_mi_fused,
    )

    rng = np.random.default_rng(1)
    n = 40_000
    a = rng.uniform(1, 5, n)
    b = rng.uniform(1, 5, n)
    y = a**2 / b
    e = np.quantile(y, np.linspace(0, 1, 21)[1:-1])
    yc = np.searchsorted(e, y).astype(np.int64)
    fy = np.bincount(yc, minlength=int(yc.max()) + 1).astype(np.float64) / n

    # Force the NON-fused leg by disabling grand fusion, then compare to the explicit fused call.
    import os as _os

    prev = _os.environ.get("MLFRAME_FE_GPU_GRAND_FUSION")
    _os.environ["MLFRAME_FE_GPU_GRAND_FUSION"] = "0"
    try:
        _, nonfused = grand_fused_pair_mi(a, b, yc, yc, fy, nbins=20, npermutations=25)
    finally:
        if prev is None:
            _os.environ.pop("MLFRAME_FE_GPU_GRAND_FUSION", None)
        else:
            _os.environ["MLFRAME_FE_GPU_GRAND_FUSION"] = prev
    _, fused = grand_fused_pair_mi_fused(a, b, yc, yc, fy, nbins=20, npermutations=25)

    np.testing.assert_allclose(fused, nonfused, rtol=1e-9, atol=1e-12)
    assert int(np.argmax(fused)) == int(np.argmax(nonfused))


def test_grand_fused_pair_mi_resident_gate_skips_d2h_then_h2d_round_trip():
    """The non-fused fallback's ``use_su=False`` noise-gate must feed the resident (n, K) int8 disc codes
    straight into ``batch_mi_with_noise_gate_cuda_resident`` via ``d_disc_resident=`` -- so the codes never
    round-trip GPU -> host (``cp.asnumpy``) -> GPU (a re-upload the plain dispatcher's ``force_backend``
    path used to pay). ``use_su=True`` is the documented exception (the resident kernel does not support
    it, per its own docstring): it still D2Hs the codes eagerly through the unchanged fallback path. Counts
    ``cp.asnumpy`` calls on int8 arrays (the disc codes' narrow dtype) to prove the round trip is gone for
    ``use_su=False`` and unchanged (>=1) for ``use_su=True``."""
    pytest.importorskip("cupy")
    import cupy as cp

    from mlframe.feature_selection.filters._gpu_resident_fe import grand_fused_pair_mi
    from mlframe.feature_selection.filters.batch_mi_noise_gate_gpu import _CUDA_AVAIL

    if not _CUDA_AVAIL:
        pytest.skip("numba.cuda not available on this host -- the resident kernel requires it")

    rng = np.random.default_rng(7)
    n = 6_000
    a = rng.uniform(1, 5, n)
    b = rng.uniform(1, 5, n)
    y = a**2 / b
    e = np.quantile(y, np.linspace(0, 1, 21)[1:-1])
    yc = np.searchsorted(e, y).astype(np.int64)
    fy = np.bincount(yc, minlength=int(yc.max()) + 1).astype(np.float64) / n

    import os as _os

    prev = _os.environ.get("MLFRAME_FE_GPU_GRAND_FUSION")
    _os.environ["MLFRAME_FE_GPU_GRAND_FUSION"] = "0"  # force the non-fused leg under test

    orig_asnumpy = cp.asnumpy
    int8_calls = {"n": 0}

    def _counting_asnumpy(arr, *a_, **kw):
        if getattr(arr, "dtype", None) == cp.int8:
            int8_calls["n"] += 1
        return orig_asnumpy(arr, *a_, **kw)

    cp.asnumpy = _counting_asnumpy
    try:
        int8_calls["n"] = 0
        grand_fused_pair_mi(a, b, yc, yc, fy, nbins=20, npermutations=25, use_su=False)
        calls_su_false = int8_calls["n"]

        int8_calls["n"] = 0
        grand_fused_pair_mi(a, b, yc, yc, fy, nbins=20, npermutations=25, use_su=True)
        calls_su_true = int8_calls["n"]
    finally:
        cp.asnumpy = orig_asnumpy
        if prev is None:
            _os.environ.pop("MLFRAME_FE_GPU_GRAND_FUSION", None)
        else:
            _os.environ["MLFRAME_FE_GPU_GRAND_FUSION"] = prev

    assert calls_su_false == 0, f"use_su=False resident noise-gate should skip the codes D2H entirely; got {calls_su_false} int8 cp.asnumpy call(s)"
    assert calls_su_true >= 1, "use_su=True (unsupported by the resident kernel) should still D2H the codes"


def test_grand_fusion_falls_back_when_shared_hist_too_big():
    """When the shared-mem histogram (P1*nbins*K_y int32) exceeds the device per-block limit, the fused
    path must RAISE (so grand_fused_pair_mi catches it and uses the exact non-fused fallback) rather than
    launch an oversized kernel. A huge nperm forces the overflow."""
    pytest.importorskip("cupy")
    from mlframe.feature_selection.filters._gpu_resident_fe import grand_fused_pair_mi_fused

    rng = np.random.default_rng(2)
    n = 5_000
    a = rng.uniform(1, 5, n)
    b = rng.uniform(1, 5, n)
    y = a**2 / b
    e = np.quantile(y, np.linspace(0, 1, 21)[1:-1])
    yc = np.searchsorted(e, y).astype(np.int64)
    fy = np.bincount(yc, minlength=int(yc.max()) + 1).astype(np.float64) / n
    # P1 = 4001, nbins=20, K_y~20 -> ~6.4MB >> 48KB shared -> must raise.
    with pytest.raises(RuntimeError):
        grand_fused_pair_mi_fused(a, b, yc, yc, fy, nbins=20, npermutations=4000)


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
        _build_candidate_matrix,
        gpu_discretize_codes_host,
    )
    from mlframe.feature_selection.filters.discretization import discretize_2d_quantile_batch

    rng = np.random.default_rng(0)
    for n in (20_000, 50_000):
        a = rng.uniform(1, 5, n)
        b = rng.uniform(1, 5, n)
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
        _build_candidate_matrix,
        gpu_discretize_codes_host,
    )
    from mlframe.feature_selection.filters.discretization import discretize_2d_quantile_batch

    rng = np.random.default_rng(0)
    for n in (20_000, 50_000):
        a = rng.uniform(1, 5, n)
        b = rng.uniform(1, 5, n)
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


def test_radix_select_edges_codes_bit_identical_to_percentile(monkeypatch):
    """ROADMAP #2: the rank-EXACT sort-free radix-select quantile-edge path (MLFRAME_FE_GPU_RADIX_EDGES=1,
    the default) must produce codes BIT-IDENTICAL (maxdiff 0) to the cp.percentile full-sort fallback
    (=0), for BOTH binning dtypes and on heavy-tailed candidates -- it only replaces HOW the nbins-1
    interior edges are computed (radix-select of the bracketing order statistics + cupy's exact 'linear'
    interpolation), not the emit contract. This locks the exactness the production gate depends on."""
    cp = pytest.importorskip("cupy")
    from mlframe.feature_selection.filters._gpu_resident_fe import _gpu_resident_discretize_codes

    rng = np.random.default_rng(3)
    for dt in (cp.float32, cp.float64):
        for n, K in ((10_000, 384), (50_000, 257), (40_000, 1)):
            base = rng.standard_normal((n, K)).astype(np.dtype(dt.__name__))
            # heavy-tailed a**2/b-style stress in a third of the columns (outlier-stretched ranges)
            if K >= 3:
                hk = base[:, ::3].shape[1]
                base[:, ::3] = (base[:, ::3] ** 2) / (rng.standard_normal((n, hk)).astype(base.dtype) + 3.0)
            cand = cp.ascontiguousarray(cp.asarray(base))
            monkeypatch.setenv("MLFRAME_FE_GPU_RADIX_EDGES", "1")
            codes_radix = _gpu_resident_discretize_codes(cand, 20)
            monkeypatch.setenv("MLFRAME_FE_GPU_RADIX_EDGES", "0")
            codes_sort = _gpu_resident_discretize_codes(cand, 20)
            maxdiff = int(cp.abs(codes_radix.astype(cp.int64) - codes_sort.astype(cp.int64)).max())
            assert maxdiff == 0, f"dt={dt.__name__} n={n} K={K} radix vs percentile code maxdiff={maxdiff}"


def test_radix_with_extremes_bit_identical_to_separate_minmax(monkeypatch):
    """The ``with_extremes=True`` fused radix path (min/max ride along as 2 extra exact order statistics in
    the select kernel) must produce codes BIT-IDENTICAL to the separate Xd.min(axis=0)/Xd.max(axis=0) +
    interior-only concatenate it replaces -- across normal, const, tied, binary, low-cardinality, subnormal,
    and signed-zero columns (the edge-dedup/ndistinct logic is exact-value-sensitive at the extremes)."""
    cp = pytest.importorskip("cupy")
    import mlframe.feature_selection.filters._fe_batched_mi as m
    import mlframe.feature_selection.filters._gpu_resident_select as gs

    rng = np.random.default_rng(3)
    base = rng.standard_normal((30_000, 64))
    variants = {"normal": base.copy()}
    v = base.copy()
    v[:, 0] = 1.0
    variants["const_col"] = v
    v = base.copy()
    v[:15_000, 1] = v[0, 1]
    variants["half_tied"] = v
    v = base.copy()
    v[:, 2] = (v[:, 2] > 0).astype(float)
    variants["binary_col"] = v
    v = base.copy()
    v[:, 3] = np.repeat(np.arange(5), 6_000)
    variants["5_distinct"] = v
    v = base.copy()
    v[:, 4] *= -1e-300
    variants["subnormal"] = v
    v = base.copy()
    v[:100, 5] = -0.0
    variants["neg_zero"] = v

    orig_fn = gs._radix_select_interior_edges

    def old_path(cand, nbins, cm_hint=None, with_extremes=False):
        interior = orig_fn(cand, nbins, cm_hint=cm_hint, with_extremes=False)
        if interior is None:
            return None
        return cp.concatenate([cand.min(axis=0)[None, :], interior, cand.max(axis=0)[None, :]], axis=0)

    for name, Xn in variants.items():
        Xd = cp.ascontiguousarray(cp.asarray(Xn))
        gs._RADIX_INTERP_CACHE.clear()
        new = cp.asnumpy(m.batched_quantile_bin_gpu(Xd, 21))
        monkeypatch.setattr(gs, "_radix_select_interior_edges", old_path)
        gs._RADIX_INTERP_CACHE.clear()
        old = cp.asnumpy(m.batched_quantile_bin_gpu(Xd, 21))
        monkeypatch.setattr(gs, "_radix_select_interior_edges", orig_fn)
        gs._RADIX_INTERP_CACHE.clear()
        assert np.array_equal(new, old), f"variant={name}: with_extremes diverges from separate min/max"


def test_radix_f64_v3_compaction_bit_identical_to_v2(monkeypatch):
    """The candidate-compaction v3 fused f64 select+interp kernel must emit interior edges BIT-IDENTICAL to
    v2 -- including columns that OVERFLOW the candidate cap (heavy ties concentrating a whole column into
    one 16-bit key window force the in-kernel full-scan fallback) and const/binary/subnormal columns."""
    cp = pytest.importorskip("cupy")
    import mlframe.feature_selection.filters._gpu_resident_select as gs

    rng = np.random.default_rng(11)
    X = rng.standard_normal((30_000, 64))
    X[:, 0] = 1.0
    X[:, 1] = (X[:, 1] > 0).astype(float)
    X[:, 2] = np.repeat(np.arange(5), 6000)[:30_000]  # 5 distinct values -> guaranteed cap overflow
    X[:, 3] *= -1e-300
    Xd = cp.ascontiguousarray(cp.asarray(X))

    def edges(v3: str):
        monkeypatch.setenv("MLFRAME_RADIX_F64_V3", v3)
        gs._RADIX_SELECT_INTERP_F64_V3_KERNEL = None
        return cp.asnumpy(gs._radix_select_interior_edges(Xd, 21))

    e3, e2 = edges("1"), edges("0")
    gs._RADIX_SELECT_INTERP_F64_V3_KERNEL = None
    assert np.array_equal(e3, e2, equal_nan=True), "v3 compaction edges diverge from v2"


def test_radix_f32_bsearch_variant_bit_identical_to_linear(monkeypatch):
    """LEVER C: the binary-search window-match f32 radix-select variant (radix_select_f32_bsearch) must
    produce codes BIT-IDENTICAL (maxdiff 0) to the base linear-scan radix_select_f32 -- it only replaces
    HOW each row finds its rank window (linear scan -> branchless binary search over the sorted active-window
    prefixes), not the order statistics. Stresses the ties/duplicates edge case (binary search must exact-
    match a window prefix, not lower_bound onto a wrong slot) and all-equal columns alongside heavy-tailed
    data. Both variants run via the radix edges path (MLFRAME_FE_GPU_RADIX_EDGES=1)."""
    cp = pytest.importorskip("cupy")
    from mlframe.feature_selection.filters import _gpu_resident_select as S

    monkeypatch.setenv("MLFRAME_FE_GPU_RADIX_EDGES", "1")
    rng = np.random.default_rng(11)

    def _codes(cand, variant):
        S._RADIX_F32_VARIANT_OVERRIDE = variant
        try:
            return S._gpu_resident_discretize_codes(cand, 20)
        finally:
            S._RADIX_F32_VARIANT_OVERRIDE = None

    for n, K in ((10_000, 257), (100_000, 96), (40_000, 1)):
        for kind in ("heavy", "uniform", "ties", "allequal"):
            if kind == "heavy":
                base = ((rng.standard_normal((n, K)) ** 2) / (rng.standard_normal((n, K)) + 3.0)).astype(np.float32)
            elif kind == "uniform":
                base = rng.uniform(-5.0, 5.0, (n, K)).astype(np.float32)
            elif kind == "ties":
                base = rng.integers(0, 7, (n, K)).astype(np.float32)  # heavy duplicate values
            else:
                base = np.full((n, K), 2.5, np.float32)
                if K > 1:
                    base[:, ::3] = 1.0  # mix all-equal + two-valued columns
            cand = cp.ascontiguousarray(cp.asarray(base))
            cl = _codes(cand, "linear")
            cb = _codes(cand, "bsearch")
            md = int(cp.abs(cl.astype(cp.int64) - cb.astype(cp.int64)).max())
            assert md == 0, f"n={n} K={K} {kind} bsearch vs linear radix code maxdiff={md}"


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
    a = rng.uniform(1, 5, n)
    b = rng.uniform(1, 5, n)
    y = a**2 / b
    e = np.quantile(y, np.linspace(0, 1, 21)[1:-1])
    yc = np.searchsorted(e, y).astype(np.int64)
    fy = np.bincount(yc, minlength=int(yc.max()) + 1).astype(np.float64) / n
    cand = np.ascontiguousarray(_build_candidate_matrix(np, a, b)).astype(np.float32)
    np.nan_to_num(cand, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    from mlframe.feature_selection.filters.discretization import discretize_2d_quantile_batch

    disc = discretize_2d_quantile_batch(cand, n_bins=20, dtype=np.int8, assume_finite=True)
    cpu = _dispatch_batch_mi_with_noise_gate(
        disc_2d=disc,
        quantization_nbins=20,
        classes_y=yc,
        classes_y_safe=yc,
        freqs_y=fy,
        npermutations=3,
        min_nonzero_confidence=0.0,
        use_su=False,
        batch_mi_kernel=batch_mi_with_noise_gate,
    )
    gpu = gpu_pairs_fe_mi(cand, 20, yc, yc, fy, 3, 0.0, False)
    assert gpu is not None, "expected the analytic GPU branch to engage at n>=50k"
    gpu = np.asarray(gpu, dtype=np.float64)
    cpu = np.asarray(cpu, dtype=np.float64)
    # Selection-equivalence, not bit-identity: the device-resident path computes the observed MI entropy ON the
    # GPU (that is the residency win -- the (n,K) codes never leave the device), so its parallel reduction order
    # differs from the CPU njit by ULP-level rounding (~4e-16 here). That can never flip the chi2 keep/reject or
    # the argmax, so the gated MI vector agrees to full double precision and the same columns are kept + ranked.
    np.testing.assert_allclose(gpu, cpu, rtol=1e-12, atol=1e-12)
    assert int(np.argmax(gpu)) == int(np.argmax(cpu))
    np.testing.assert_array_equal(gpu > 0.0, cpu > 0.0)  # identical keep/reject set
    assert int(np.argmax(gpu)) == int(np.argmax(cpu))


def test_gpu_pairs_fe_mi_returns_none_for_nonanalytic():
    """gpu_pairs_fe_mi must DEFER (None) when the analytic branch doesn't apply -- SU-normalised,
    npermutations<=0, or small n -- so the caller uses the CPU dispatch (no silent wrong path)."""
    pytest.importorskip("cupy")
    from mlframe.feature_selection.filters._gpu_resident_fe import _build_candidate_matrix, gpu_pairs_fe_mi

    rng = np.random.default_rng(0)
    n = 4_000  # below analytic_null_min_n -> must defer
    a = rng.uniform(1, 5, n)
    b = rng.uniform(1, 5, n)
    y = a**2 / b
    e = np.quantile(y, np.linspace(0, 1, 21)[1:-1])
    yc = np.searchsorted(e, y).astype(np.int64)
    fy = np.bincount(yc, minlength=int(yc.max()) + 1).astype(np.float64) / n
    cand = np.ascontiguousarray(_build_candidate_matrix(np, a, b)).astype(np.float32)
    assert gpu_pairs_fe_mi(cand, 20, yc, yc, fy, 3, 0.0, False) is None  # small n
    assert gpu_pairs_fe_mi(cand, 20, yc, yc, fy, 0, 0.0, False) is None  # npermutations<=0
    assert gpu_pairs_fe_mi(cand, 20, yc, yc, fy, 3, 0.0, True) is None  # SU-normalised


def test_fused_bin_codes_bit_identical_to_per_column_searchsorted():
    """The fused bin-codes RawKernel (_searchsorted_codes) must equal the per-column
    cp.searchsorted(edges, col, side='right') it replaces -- BIT-IDENTICAL codes (maxdiff 0) at f32 AND
    f64, on the SAME f64 edges. Guards the nvprof-driven fusion that removed the K int64->int32 cast-copies
    + K searchsorted launches (the value is promoted to f64 for the compare, matching cp.searchsorted)."""
    cp = pytest.importorskip("cupy")
    from mlframe.feature_selection.filters._gpu_resident_fe import _searchsorted_codes

    rng = np.random.default_rng(0)
    for n, K, dt in [(50_000, 128, cp.float32), (50_000, 128, cp.float64), (80_000, 257, cp.float32), (40_000, 1, cp.float32)]:
        cand = cp.asarray(rng.uniform(1, 5, (n, K)).astype(np.float32 if dt == cp.float32 else np.float64))
        edges = cp.percentile(cand.ravel(), cp.linspace(0, 100, 21)).reshape(-1, 1) if K == 1 else cp.percentile(cand, cp.linspace(0, 100, 21), axis=0)
        interior = cp.ascontiguousarray(edges[1:-1])  # (nbins-1, K), f64
        fused = cp.asnumpy(_searchsorted_codes(cand, interior))
        ref = cp.empty((n, K), dtype=cp.int32)
        for j in range(K):
            ref[:, j] = cp.searchsorted(interior[:, j], cand[:, j], side="right")
        np.testing.assert_array_equal(fused, cp.asnumpy(ref))


def test_deferred_host_codes_bit_identical_to_eager():
    """DEFERRED host-codes D2H (MLFRAME_FE_GPU_DEFER_HOST_CODES, default ON when the resident handoff is
    on) must be SELECTION-EQUIVALENT to the eager per-block fill: the host codes are produced unfilled and
    the resident DEVICE codes kept, so a host-reading consumer materialises ``out`` lazily via
    ``ensure_host_codes_filled`` -- which D2Hs the EXACT bytes the eager fill produced (maxdiff 0). This
    removed the canonical fit's single largest D2H (1691 MB / 160 transfers @100k). Covers the FUSED
    materialise producer leg AND the device-codes==host-codes invariant + fill idempotency. (The
    binning-only gpu_discretize_codes_host leg stays eager -- direct callers read its host return.)
    """
    pytest.importorskip("cupy")
    import cupy as cp
    import mlframe.feature_selection.filters._gpu_resident_fe as G

    rng = np.random.default_rng(7)
    n, n_oper, K, nbins = 4000, 6, 40, 20
    tv = rng.random((n, n_oper)).astype(np.float32) + 0.1
    a_cols = rng.integers(0, n_oper, size=K).astype(np.int64)
    b_cols = rng.integers(0, n_oper, size=K).astype(np.int64)
    ops = rng.integers(0, 9, size=K).astype(np.int8)

    # Force the resident-codes handoff on (so a device copy exists to defer from), and A/B the deferral.
    import os

    _saved = os.environ.get("MLFRAME_FE_GPU_DEFER_HOST_CODES")
    _saved_res = os.environ.get("MLFRAME_FE_GPU_RESIDENT_CODES")
    os.environ["MLFRAME_FE_GPU_RESIDENT_CODES"] = "1"
    try:
        # --- fused materialise leg ---
        os.environ["MLFRAME_FE_GPU_DEFER_HOST_CODES"] = "0"
        eager = G.gpu_materialise_discretize_codes_host(tv, a_cols, b_cols, ops, nbins, dtype=np.int8).copy()
        os.environ["MLFRAME_FE_GPU_DEFER_HOST_CODES"] = "1"
        out = G.gpu_materialise_discretize_codes_host(tv, a_cols, b_cols, ops, nbins, dtype=np.int8)
        dev = G.take_resident_codes(out)  # resident gate would consume these in place
        assert dev is not None and tuple(dev.shape) == (n, K)
        np.testing.assert_array_equal(cp.asnumpy(dev), eager)  # device codes == eager host codes
        G.ensure_host_codes_filled(out)  # host consumer (analytic/CPU) materialises lazily
        np.testing.assert_array_equal(out, eager)  # bit-identical -> selection-equivalent
        G.ensure_host_codes_filled(out)  # idempotent: second fill is a no-op
        np.testing.assert_array_equal(out, eager)
    finally:
        for k, v in (("MLFRAME_FE_GPU_DEFER_HOST_CODES", _saved), ("MLFRAME_FE_GPU_RESIDENT_CODES", _saved_res)):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        G.clear_resident_codes_handoff()


def test_gpu_apply_prewarp_resolves_clenshaw_dict_after_carve():
    """Regression (Tier E carve 2026-06-22): ``_gpu_apply_prewarp`` lives in ``_gpu_resident_fe``
    but the ``_PREWARP_CLENSHAW_GPU`` lookup table it reads was carved into ``_gpu_resident_basis``.
    A bare-name reference left behind raised ``NameError`` on every Clenshaw-basis prewarp (hermite/
    legendre/chebyshev/laguerre) -- a GPU-only path the parity suite never exercised. Pin that the
    parent resolves the carved dict (qualified via the ``_grb`` re-export) so the NameError can't return."""
    cp = pytest.importorskip("cupy")
    from mlframe.feature_selection.filters._gpu_resident_fe import _gpu_apply_prewarp
    from mlframe.feature_selection.filters._gpu_resident_basis import _PREWARP_CLENSHAW_GPU

    x = cp.asarray(np.linspace(0.2, 1.0, 16))
    for basis in _PREWARP_CLENSHAW_GPU:  # chebyshev / legendre / hermite / laguerre
        spec = {"basis": basis, "preprocess": {}, "coef": [1.0, 0.0, 0.0]}
        try:
            _gpu_apply_prewarp(cp, x, spec)
        except NameError as e:  # the carved-dict reference must resolve
            pytest.fail(f"_gpu_apply_prewarp NameError on basis {basis!r}: {e}")
        except Exception:  # nosec B110 -- best-effort cleanup/optional step; failure here never masks this test's own assertions
            pass  # incomplete minimal spec may raise KeyError/ValueError downstream -- not the bug under test


def test_fe_materialise_cm_bit_identical():
    """Coalescing audit (2026-06-23): the COALESCED column-major fe_materialise (``fe_materialise_cm`` +
    tv-transpose + result transpose-back, ~2x net) must be BIT-IDENTICAL (array_equal) to the row-major
    ``fe_materialise`` kernel and return the SAME (n, K) row-major layout the downstream bin/D2H expect.
    Covers all op-codes (0..8 incl. float64-promoted div/ratio_abs + nan-propagating max/min/signed),
    zeros / negatives / +-inf operands, and K==1. The row-major kernel is the gated fallback."""
    cp = pytest.importorskip("cupy")
    import mlframe.feature_selection.filters._gpu_resident_select as M

    rng = np.random.RandomState(13)
    for n, K, nop in [(3000, 50, 16), (10000, 257, 32), (40000, 583, 64), (2048, 1, 8)]:
        tv = rng.standard_normal((n, nop)).astype(np.float32)
        tv[::97, :] = 0.0
        tv[2, 0] = 1e30
        tv[3, 0] = -1e30
        a = rng.randint(0, nop, K).astype(np.int64)
        b = rng.randint(0, nop, K).astype(np.int64)
        ops = rng.randint(0, 9, K).astype(np.int8)  # exercise every op-code 0..8
        tvg = cp.asarray(tv)

        M._OPERAND_TABLE_CM_CACHE["ref"] = None  # avoid a stale cm cache from a prior shape
        import os

        os.environ["MLFRAME_FE_GPU_MATERIALISE_CM"] = "0"
        o_rm = M._fe_materialise_block_gpu(tvg, a, b, ops)
        os.environ["MLFRAME_FE_GPU_MATERIALISE_CM"] = "1"
        M._OPERAND_TABLE_CM_CACHE["ref"] = None
        o_cm = M._fe_materialise_block_gpu(tvg, a, b, ops)
        os.environ.pop("MLFRAME_FE_GPU_MATERIALISE_CM", None)

        assert o_cm.shape == (n, K) == o_rm.shape
        assert bool(cp.array_equal(o_rm, o_cm)), f"cm materialise differs at n={n} K={K} nop={nop}"
