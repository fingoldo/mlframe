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
