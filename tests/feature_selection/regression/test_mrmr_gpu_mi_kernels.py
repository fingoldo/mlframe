"""Behavioural tests for the GPU mutual-information kernels and their host-side helpers."""

from __future__ import annotations
import numpy as np
import pytest


def test_regression_compute_pairs_mis_canonicalizes_pair_key_order(monkeypatch):
    """Pre-fix: compute_pairs_mis([(b, a)], ...) cached the pair under the RAW (unsorted) key (b, a); calling
    it again with the same logical pair given as (a, b) would then miss the cache and recompute, creating a
    second divergent dict entry for one physical pair. Post-fix: the key is canonicalized to sorted order,
    so both call orders resolve to the identical dict entry. mi_direct itself is monkeypatched to a trivial
    stub so this test isolates the key-canonicalization logic from the real MI kernel.
    """
    import mlframe.feature_selection.filters.feature_engineering as fe_mod

    monkeypatch.setattr(fe_mod, "mi_direct", lambda *a, **kw: (0.5, 1.0))

    kwargs = dict(
        data=np.zeros((10, 3)),
        target_indices=(2,),
        nbins=np.array([5, 5, 2]),
        classes_y=np.array([0, 1]),
        classes_y_safe=np.array([0, 1]),
        freqs_y=np.array([0.5, 0.5]),
        fe_min_nonzero_confidence=0.0,
        fe_npermutations=0,
        fe_min_pair_mi=0.0,
        fe_min_pair_mi_prevalence=0.0,
    )

    cached_MIs_a: dict = {}
    fe_mod.compute_pairs_mis(all_pairs=[(1, 0)], cached_confident_MIs={}, cached_MIs=cached_MIs_a, **kwargs)
    pair_keys_a = [k for k in cached_MIs_a if len(k) == 2]
    assert pair_keys_a == [(0, 1)], f"expected the pair key stored in sorted order, got {pair_keys_a!r}"

    # Simulate a second FE step's candidate pool handing the SAME logical pair in the OTHER order --
    # must resolve to the SAME dict entry, not a second divergent one.
    cached_MIs_b: dict = dict(cached_MIs_a)
    fe_mod.compute_pairs_mis(all_pairs=[(0, 1)], cached_confident_MIs={}, cached_MIs=cached_MIs_b, **kwargs)
    pair_keys_b = [k for k in cached_MIs_b if len(k) == 2]
    assert pair_keys_b == [(0, 1)], f"the same logical pair given in the other order must not create a second entry, got {pair_keys_b!r}"


# ---------------------------------------------------------------------------
# SCREEN_CONFIRM_A-1 (P1): evaluate_candidate's baseline mi_direct/mi_direct_gpu
# calls never received a seed derived from random_seed, so the per-candidate
# relevance permutation-null gate was CPU-seed-insensitive (always base_seed=0)
# and GPU-nondeterministic (fresh OS entropy every call).
# SCREEN_CONFIRM_A-3 (P1): _cmi_plugin_njit's dense (K_x,K_y,K_z) histogram cap
# was 1_000_000 (K_x=K_y=10 alone -> ~800MB per call, rebuilt every permutation
# draw) -- lowered to 10_000.
# ---------------------------------------------------------------------------


def test_regression_batch_joint_entropy_pairs_rejects_nonpositive_nbins():
    """Pre-fix: a non-positive nbins for a referenced column reached the njit kernel unchecked,
    risking a silent OOB write. Post-fix: a host-side ValueError guard fails loudly first."""
    from mlframe.feature_selection.filters._dcd_pair_su_batch import _validate_batch_joint_entropy_pairs_inputs

    with pytest.raises(ValueError, match="nbins must be >= 1"):
        _validate_batch_joint_entropy_pairs_inputs(
            a_arr=np.array([0], dtype=np.int64),
            b_arr=np.array([1], dtype=np.int64),
            nb_arr=np.array([2, 0], dtype=np.int64),
        )


# ---------------------------------------------------------------------------
# GPU_INFRA_A-12 (P2): the module-level _DY_DEVICE_CACHE/_DY_DEVICE_CACHE_CUPY OrderedDict LRU
# caches performed a non-atomic get -> move_to_end -> popitem sequence with no lock, risking the
# LRU eviction discipline being violated under concurrent multi-thread MRMR.fit() calls.
# ---------------------------------------------------------------------------


def test_regression_gpu_resident_materialise_entrypoints_do_not_blanket_clear_handoff():
    """Pre-fix: both entry points called clear_resident_codes_handoff() with no argument, which could
    silently drop another thread's in-flight deferred-fill entry. Post-fix: the blanket call is gone
    entirely (relies on the bounded-FIFO eviction instead), so a stashed entry from a prior call survives."""
    pytest.importorskip("cupy")
    import mlframe.feature_selection.filters._gpu_resident_fe as gfe

    gfe.clear_resident_codes_handoff()
    sentinel_host = np.zeros((4, 4), dtype=np.int8)
    sentinel_device = np.zeros((4, 4), dtype=np.int8)  # stand-in device codes; only identity/shape matter here
    gfe._stash_deferred_host_fill(sentinel_host, sentinel_device)
    assert gfe._DEFERRED_HOST_FILL, "test setup must actually stash an entry"
    stashed_key = next(iter(gfe._DEFERRED_HOST_FILL.keys()))

    import mlframe.feature_selection.filters._gpu_resident_materialise as gm

    rng = np.random.default_rng(0)
    cand = rng.standard_normal((512, 3)).astype(np.float32)
    try:
        gm.gpu_discretize_codes_host(cand, 8)
        assert stashed_key in gfe._DEFERRED_HOST_FILL, "gpu_discretize_codes_host silently dropped another call's pending handoff entry"
        tv = rng.standard_normal((512, 3)).astype(np.float32)
        gm.gpu_materialise_discretize_codes_host(tv, np.array([0]), np.array([1]), np.array([0], dtype=np.int8), 8)
        assert stashed_key in gfe._DEFERRED_HOST_FILL, "gpu_materialise_discretize_codes_host silently dropped another call's pending handoff entry"
    finally:
        gfe.clear_resident_codes_handoff()  # cleanup


# ---------------------------------------------------------------------------
# GPU_INFRA_B-4 (P2): gpu_materialise_discretize_codes_host / gpu_discretize_codes_host's docstrings/comments
# claimed "nbins<=255 -> int8 cannot overflow", but _BIN_CODES_OUTTYPE maps "int8" to C signed char
# (range -128..127), so the correct bound is nbins<=128, not 255 -- a code of 150 would silently wrap to -106.
# ---------------------------------------------------------------------------


def test_regression_gpu_materialise_discretize_rejects_narrow_dtype_for_nbins():
    """Pre-fix: no validation -- a dtype too narrow for nbins would silently wrap around instead of raising.
    Post-fix: both public entry points raise ValueError immediately for an under-sized integer dtype."""
    import mlframe.feature_selection.filters._gpu_resident_materialise as gm

    with pytest.raises(ValueError, match="cannot represent codes"):
        gm.gpu_discretize_codes_host(
            cand=np.zeros((4, 2), dtype=np.float32),
            nbins=150,
            dtype=np.int8,
        )
    with pytest.raises(ValueError, match="cannot represent codes"):
        gm.gpu_materialise_discretize_codes_host(
            transformed_vars=np.zeros((4, 2), dtype=np.float32),
            a_cols=np.array([0], dtype=np.int64),
            b_cols=np.array([1], dtype=np.int64),
            op_codes=np.array([0], dtype=np.int8),
            nbins=150,
            dtype=np.int8,
        )


# ---------------------------------------------------------------------------
# GPU_INFRA_B-5 (P2): grand_fused_pair_mi/grand_fused_pair_mi_fused hardcoded base_seed=np.uint64(0) at
# every permutation-null call site, never threading through a caller-supplied seed -- the same
# "ignores the estimator's random_state" bug class the 2026-07-20 audit flagged at P1 for the production
# confirmation path. Currently dead/unreachable prototype code, fixed before any future wiring reuses it.
# ---------------------------------------------------------------------------


def test_regression_fe_gpu_strict_auto_fit_shape_is_thread_local():
    """Pre-fix: _AUTO_FIT_N/_AUTO_FIT_P were bare module globals, visible to every thread. Post-fix: each
    thread gets its own (n, p) via threading.local(), so one thread's set_auto_fit_n does not leak into another."""
    import threading

    from mlframe.feature_selection.filters import _fe_gpu_strict as strict

    strict.clear_auto_fit_n()
    other_thread_saw = {}

    def _other_thread():
        # A fresh thread must start with NO fit shape set, regardless of what the main thread set below.
        """Thread body: exercises the call under test and records its result or error."""
        other_thread_saw["n"] = getattr(strict._auto_fit_state, "n", None)
        strict.set_auto_fit_n(999_999, 5)
        other_thread_saw["after_set"] = strict._auto_fit_state.n

    strict.set_auto_fit_n(123, 7)
    assert strict._auto_fit_state.n == 123

    t = threading.Thread(target=_other_thread)
    t.start()
    t.join()

    assert other_thread_saw["n"] is None, "a fresh thread must not inherit another thread's fit shape"
    assert other_thread_saw["after_set"] == 999_999
    # The main thread's own state must be UNCHANGED by the other thread's set_auto_fit_n call.
    assert strict._auto_fit_state.n == 123
    strict.clear_auto_fit_n()


# ---------------------------------------------------------------------------
# GPU_INFRA_C-2 (P1): install_cuda_teardown_guard's idempotency check was a bare unlocked
# `if _installed: return` -- two threads racing through it before either set _installed=True could let the
# second thread capture the FIRST thread's own hook wrapper as "the previous hook", risking infinite
# self-recursion later. Fixed via a threading.Lock guarding the whole check-then-install sequence.
# ---------------------------------------------------------------------------


def _cuda_device_count(cp):
    """Visible CUDA device count, or 0 when the CUDA runtime cannot be queried."""
    try:
        return int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        return 0


def test_regression_fe_additive_fusion_gpu_ols_margin_uses_full_n_rows(monkeypatch):
    """Pre-fix: the OLS separability margin was divided by sqrt of the SCORING subsample size, so above the subsample cap
    the GPU twin's threshold sat at 2/sqrt(n_sc) rather than the CPU sibling's 2/sqrt(n_rows). Post-fix: the margin uses
    the true fit row count. The fixture (n=4000, cap=1000) places the fused R-gain between 2/sqrt(4000) and 2/sqrt(1000),
    so the fusion is admitted only under the full-n margin. The fixture has to satisfy three bounds at once: both halves clear their relevance floor on the
    1000-row scoring sample, the OLS gain stays inside (2/sqrt(4000), 2/sqrt(1000)), and the unweighted sum keeps the gain (the keeps-both-halves rule). Equal
    weights cannot do it (the relevance floor alone puts the gain just above 2/sqrt(1000)); the second half at 0.6 of the first can."""
    cp = pytest.importorskip("cupy")
    if _cuda_device_count(cp) < 1:
        pytest.skip("no usable CUDA device")

    import pandas as pd

    from mlframe.feature_selection.filters import _fe_additive_fusion_gpu_resident as fus
    from mlframe.feature_selection.filters.engineered_recipes import build_orth_univariate_recipe
    from mlframe.feature_selection.filters.mrmr import MRMR

    n, cap = 4000, 1000
    rng = np.random.default_rng(0)
    c = rng.standard_normal(n)
    d = rng.standard_normal(n)
    h_c = c**2 - 1.0
    h_d = d**2 - 1.0
    # On the scoring subsample (every 4th row, binned y) the OLS gain of the pair over its best half is 0.053 and the unweighted sum's is 0.049: both above 2/sqrt(4000) = 0.032,
    # below 2/sqrt(1000) = 0.063.
    y_cont = h_c + 0.6 * h_d + 5.0 * rng.standard_normal(n)
    y_dense = pd.qcut(y_cont, q=8, labels=False, duplicates="drop").astype(np.int64)
    name_c, name_d = "c__He2", "d__He2"
    recipes = {
        name_c: build_orth_univariate_recipe(name=name_c, src_name="c", basis="hermite", degree=2),
        name_d: build_orth_univariate_recipe(name=name_d, src_name="d", basis="hermite", degree=2),
    }
    fs = MRMR(verbose=0)
    fs.fe_additive_fusion_floor_margin = 1e9  # the binned-MI gate can never pass, so only the OLS margin decides
    monkeypatch.setenv("MLFRAME_FE_FUSION_MAX_ROWS", str(cap))

    def run() -> list:
        """Names of the fusions admitted for the fixture under the current margin settings."""
        admitted, _subsumed, _subsumed_raws = fus.propose_additive_fusions_gpu(
            fs,
            engineered_recipes=recipes,
            engineered_continuous={name_c: h_c, name_d: h_d},
            newly_engineered_names=[name_c, name_d],
            raw_name_set={"c", "d"},
            cols=["c", "d"],
            classes_y=y_dense,
            X=pd.DataFrame({"c": c, "d": d}),
            nbins=10,
            seed=0,
        )
        return [a["name"] for a in admitted]

    assert run() == ["add(c__He2,d__He2)"], "the full-n OLS margin must admit this fusion under the cap"
    # Control: the pre-fix margin 2/sqrt(n_scoring) is twice as wide here (4/sqrt(4000)); the same gain must then be rejected.
    fs.fe_additive_fusion_ols_r_margin_sd = 4.0
    assert run() == [], "a margin scaled to the scoring subsample rejects this fusion, so the test separates the two margins"


# ---------------------------------------------------------------------------
# GPU_INFRA_D-5 (P1): _gpu_pairs.py's per-pair joint-MI reduction was a pure-Python triple-nested loop
# (O(total joint cells) at native Python speed), defeating the point of the batched CUDA histogram launch
# at realistic cell counts. Vectorized with numpy, same formula.
# ---------------------------------------------------------------------------


def test_regression_gpu_pairs_joint_mi_reduction_matches_reference_loop():
    """The vectorized per-pair MI reduction must produce IDENTICAL results to the original triple-nested
    Python loop formula, for a hand-built joint-count table with zero cells in various positions.

    Calls production. The earlier form re-typed the vectorised reduction inline and never imported
    `_gpu_pairs`, so a regression in the shipped code left it green.
    """
    import numpy as _np

    from mlframe.feature_selection.filters._gpu_pairs import joint_mi_from_flat_counts

    def _reference_loop(joint_counts_host, joint_offsets, pair_merged_sizes, nbins_y, n_pairs, n_total):
        """The straightforward reference implementation the optimised path must agree with."""
        out = _np.zeros(n_pairs, dtype=_np.float64)
        for k in range(n_pairs):
            off = int(joint_offsets[k])
            merged_size = int(pair_merged_sizes[k])
            joint_2d = joint_counts_host[off : off + merged_size * nbins_y].reshape(merged_size, nbins_y)
            marg_m = joint_2d.sum(axis=1)
            marg_y = joint_2d.sum(axis=0)
            mi = 0.0
            for m in range(merged_size):
                mm = marg_m[m]
                if mm == 0:
                    continue
                for y in range(nbins_y):
                    jc = joint_2d[m, y]
                    if jc == 0:
                        continue
                    my = marg_y[y]
                    if my == 0:
                        continue
                    jf = jc / n_total
                    mi += jf * _np.log(jc * n_total / (mm * my))
            out[k] = mi
        return out

    rng = _np.random.default_rng(0)
    nbins_y = 4
    n_total = 1000.0
    # Two pairs, merged sizes 3 and 5, with some deliberately-zero cells.
    merged_sizes = [3, 5]
    offsets = [0, 3 * nbins_y]
    total_cells = sum(m * nbins_y for m in merged_sizes)
    joint_counts_host = rng.integers(0, 50, size=total_cells).astype(np.int64)
    joint_counts_host[rng.choice(total_cells, size=total_cells // 4, replace=False)] = 0  # sprinkle zeros

    ref = _reference_loop(joint_counts_host, offsets, merged_sizes, nbins_y, 2, n_total)
    joint_mi_out = joint_mi_from_flat_counts(joint_counts_host, offsets, merged_sizes, nbins_y, 2, n_total, _np.zeros(2, dtype=_np.float64))

    np.testing.assert_allclose(joint_mi_out, ref, rtol=1e-12, atol=1e-12)
    assert (joint_mi_out > 0).all(), "the fixture produced degenerate zero MI, so the comparison is vacuous"


def test_regression_gpu_pairs_joint_mi_zeroes_a_pair_with_no_valid_cell():
    """The `if not np.any(valid): continue` branch: an all-zero pair leaves its slot untouched.

    Uncovered before, and it is the branch that decides whether an empty pair reads as MI 0.0 or as
    whatever the caller's buffer happened to hold.
    """
    import numpy as _np

    from mlframe.feature_selection.filters._gpu_pairs import joint_mi_from_flat_counts

    counts = _np.zeros(2 * 3, dtype=_np.int64)
    out = joint_mi_from_flat_counts(counts, [0], [2], 3, 1, 100.0, _np.full(1, -7.0))
    assert out[0] == -7.0, "an all-zero pair must be left alone rather than written with a computed value"


# ---------------------------------------------------------------------------
# GPU_INFRA_D-6 (P2): run_cupy_kernel_search / run_numba_kernel_search's docstrings claimed an identical
# 5-tuple return contract, but the 5th element differs (best_score vs an evaluation count).
# ---------------------------------------------------------------------------


def test_regression_complementary_pairs_stability_raises_on_total_failure():
    """Pre-fix: 100% pair failure silently returned an empty selection. Post-fix: raises RuntimeError."""
    from mlframe.feature_selection.filters._stability_cluster import complementary_pairs_stability

    rng = np.random.default_rng(0)
    n, p = 40, 5
    X = rng.standard_normal((n, p))
    y = rng.standard_normal(n)

    def _always_fails(Xb, yb):
        """Stand-in that raises ValueError, so the caller's failure path is the one under test."""
        raise ValueError("simulated systematic selector_fn failure")

    with pytest.raises(RuntimeError, match="all .* pairs failed"):
        complementary_pairs_stability(X, y, _always_fails, n_pairs=5, rng_seed=0)


def test_regression_scan_integer_lattice_pairs_uses_one_shared_rng(monkeypatch):
    """Pre-fix: np.random.default_rng(rng_seed) was constructed fresh per candidate, so every candidate was
    tested against identical shuffles. Post-fix: one Generator is built per scan and threaded into every
    permutation-null call. Observed at the null-band call boundary across several candidates."""
    from mlframe.feature_selection.filters import _integer_lattice_fe_proto as proto

    rng_objects: list = []
    _orig_null = proto._perm_null_hi

    def _null_spy(feat, yi, **kwargs):
        """Record the Generator handed to each candidate's permutation null, then delegate."""
        rng_objects.append(kwargs["rng"])
        return _orig_null(feat, yi, **kwargs)

    monkeypatch.setattr(proto, "_perm_null_hi", _null_spy)
    rng = np.random.default_rng(0)
    n = 1500
    a = rng.integers(0, 8, n).astype(np.int64)
    b = rng.integers(0, 8, n).astype(np.int64)
    c = rng.integers(0, 8, n).astype(np.int64)
    X = np.column_stack([a, b, c])
    y = ((a + b) % 3).astype(np.int64)
    proto.scan_integer_lattice_pairs(X, y, ["a", "b", "c"], nbins=8, n_perm=12, rng_seed=7)

    assert len(rng_objects) >= 2, "fixture must reach the permutation null for several candidates"
    assert all(r is rng_objects[0] for r in rng_objects), "each candidate got a freshly seeded Generator instead of the shared one"


# ---------------------------------------------------------------------------
# ORTH_BASIS_B-3 (P2): _fourier_detect_gpu_resident.py's _SPLIT_MASK_CACHE/_SUBSAMPLE_IDX_CACHE performed
# an unlocked read-then-write (get() then [key]=), the same hand-rolled unlocked module-level cache pattern
# that reproduced a real crash elsewhere in this codebase under concurrent access.
# ---------------------------------------------------------------------------


def test_regression_gate_med_rejects_constant_operand():
    """Post-fix: a constant, non-NaN operand is now rejected (matching the comment's promise), not
    registered under gate_med as a dead all-zero pseudo-unary. Observed on the returned median table:
    the constant operand gets no gate_med entry, the varying operand does."""
    from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_setup import _fit_prewarp_and_gate_med

    rng = np.random.default_rng(0)
    n = 200
    columns = {0: np.full(n, 3.0), 1: rng.standard_normal(n)}
    _, _, gate_med_active, medians = _fit_prewarp_and_gate_med(
        prospective_pairs={((0, 1), "add"): None},
        prewarp_enable=False,
        prewarp_y=None,
        prewarp_y_continuous=False,
        prewarp_basis="chebyshev",
        prewarp_max_degree=3,
        prewarp_min_val_corr=0.0,
        fe_gate_med_enable=True,
        original_cols=[0, 1],
        _use_subsample=False,
        _full_n_rows=n,
        _sample_idx=None,
        _extval_raw_col=lambda v: columns[v],
    )
    assert gate_med_active is True
    assert 0 not in medians, "a constant operand must not be registered under gate_med"
    assert 1 in medians


# ---------------------------------------------------------------------------
# FE_PAIRS_CORE-4 (P2): the subsample freqs_y recompute used np.bincount without minlength=, so dropping
# every row of the highest class label would silently under-count k_y.
# ---------------------------------------------------------------------------


def test_regression_bincount_minlength_prevents_class_undercounting():
    """Direct pin of the underlying bincount behavior this fix relies on: minlength= keeps k_y honest even
    when the highest class label has zero occurrences in a subsample."""
    classes_y_full_k = 5  # classes 0..4 existed pre-subsample
    classes_y_subsample = np.array([0, 1, 2, 1, 0], dtype=np.int64)  # classes 3, 4 dropped by the draw
    counts_without_minlength = np.bincount(classes_y_subsample)
    counts_with_minlength = np.bincount(classes_y_subsample, minlength=classes_y_full_k)
    assert counts_without_minlength.shape[0] == 3, "sanity: the bug's premise -- bare bincount undercounts k_y"
    assert counts_with_minlength.shape[0] == classes_y_full_k

    from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_core_helpers import _check_prospective__total_get_proportions_matching

    freqs_y_full = np.full(classes_y_full_k, 1.0 / classes_y_full_k)
    freqs_out = _check_prospective__total_get_proportions_matching(classes_y_subsample, freqs_y_full)
    assert freqs_out.shape[0] == classes_y_full_k, "the subsample's k_y shrank below the pre-subsample class count"
    np.testing.assert_allclose(freqs_out, [0.4, 0.4, 0.2, 0.0, 0.0])


# ---------------------------------------------------------------------------
# FE_ORCH_BUDGET-1 (P1): raws_linearly_explain_y's regression skip-gate scored IN-SAMPLE R^2, which
# crosses the default 0.92 threshold purely from overfitting once p numeric raw columns approaches n --
# a false "nothing left to find" verdict that silently disables the discrete-structural operators.
# ---------------------------------------------------------------------------


def test_regression_gpu_binary_greater_less_equal_match_cpu_int_dtype():
    """Post-fix: the GPU-resident twin now casts to cp.int64, matching the CPU registry's .astype(int).
    Observed on the returned device arrays: integer dtype with the 0/1 comparison values."""
    cp = pytest.importorskip("cupy")
    from mlframe.feature_selection.filters.engineered_recipes._recipe_unary_binary_gpu import _gpu_binary

    a = cp.asarray([1.0, 2.0, 3.0], dtype=cp.float32)
    b = cp.asarray([2.0, 2.0, 1.0], dtype=cp.float32)
    for name, expected in (("greater", [0, 0, 1]), ("less", [1, 0, 0]), ("equal", [0, 1, 0])):
        out = _gpu_binary(cp, name, a, b)
        assert out.dtype == cp.int64, f"{name}: GPU result dtype {out.dtype} must be int64 to match the CPU registry"
        assert cp.asnumpy(out).tolist() == expected


# ---------------------------------------------------------------------------
# MI_GREEDY_RECIPES-6 (P2): _fe_batched_mi.py's docstring claimed "imported by nothing in production yet"
# but batched_cmi_gpu/batched_quantile_bin_gpu/cmi_device_argmax ARE called from
# _mi_greedy_cmi_fe.py's greedy_cmi_fe_construct, production-reachable via fe_mi_greedy_cmi_enable=True.
# ---------------------------------------------------------------------------


def test_regression_fit_constant_memmap_cache_is_bounded():
    """Post-fix: caching more than the LRU cap of distinct-content arrays evicts the oldest and unlinks
    its backing file instead of leaking it forever."""
    import gc
    import glob
    import os
    import tempfile

    from mlframe.feature_selection.filters._joblib_safe import (
        _FIT_MEMMAP_CACHE_MAX_ENTRIES,
        fit_constant_memmap,
    )

    tmp_dir = tempfile.gettempdir()
    before = set(glob.glob(os.path.join(tmp_dir, "mlframe_fitconst_*.mmap")))
    for i in range(_FIT_MEMMAP_CACHE_MAX_ENTRIES + 4):
        arr = np.random.default_rng(i).standard_normal((64, 8))
        view = fit_constant_memmap(arr)
        del view
    gc.collect()
    after = set(glob.glob(os.path.join(tmp_dir, "mlframe_fitconst_*.mmap")))
    new_files = after - before
    assert (
        len(new_files) <= _FIT_MEMMAP_CACHE_MAX_ENTRIES
    ), f"expected at most {_FIT_MEMMAP_CACHE_MAX_ENTRIES} live fit-constant files, found {len(new_files)}: {new_files}"
    for p in new_files:
        try:
            os.unlink(p)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# USABILITY_A-12 (P1): _fit_constant_key hashed only a bounded sample (first/last 64KB + coarse stride),
# so two different-content arrays agreeing at the sampled points could collide and alias to the wrong
# cached memmap.
# ---------------------------------------------------------------------------


def test_regression_fit_constant_key_hashes_full_buffer_not_sample():
    """Post-fix: two arrays differing ONLY in the middle (outside the old sampled-stride pattern) must
    get different keys -- pre-fix, a carefully-crafted middle-only diff at non-sampled offsets could
    collide."""
    from mlframe.feature_selection.filters._joblib_safe import _fit_constant_key

    n = 200_000
    a = np.zeros(n, dtype=np.float64)
    b = a.copy()
    b[n // 2 + 3] = 1.0
    assert _fit_constant_key(a) != _fit_constant_key(b)


# ---------------------------------------------------------------------------
# USABILITY_A-5 (P2): threading.stack_size() is process-global; concurrent run_in_big_stack_thread()
# callers racing on save/set/restore could have the loser's finally restore a stale value.
# USABILITY_A-6 (P2, coverage_gap): zero direct unit tests exercised run_in_big_stack_thread by name.
# ---------------------------------------------------------------------------


def test_regression_run_in_big_stack_thread_returns_value():
    """Direct coverage: a function that returns a value round-trips it through the big-stack thread."""
    from mlframe.feature_selection.filters._joblib_safe import run_in_big_stack_thread

    assert run_in_big_stack_thread(lambda a, b: a + b, 3, 4) == 7


def test_regression_run_in_big_stack_thread_propagates_exception():
    """Direct coverage: a function that raises propagates the same exception type/message to the caller."""
    from mlframe.feature_selection.filters._joblib_safe import run_in_big_stack_thread

    def _boom():
        """Stand-in that raises ValueError, so the caller's failure path is the one under test."""
        raise ValueError("boom-marker")

    with pytest.raises(ValueError, match="boom-marker"):
        run_in_big_stack_thread(_boom)


def test_regression_run_in_big_stack_thread_serializes_stack_size_races():
    """Post-fix: concurrent callers no longer race on the process-global threading.stack_size() --
    every concurrent call must still see the ORIGINAL stack size restored once all calls complete."""
    import threading as _threading

    from mlframe.feature_selection.filters._joblib_safe import run_in_big_stack_thread

    original = _threading.stack_size()
    results = []
    errors = []

    def _worker(i):
        """Thread body: exercises the call under test and records its result or error."""
        try:
            r = run_in_big_stack_thread(lambda x: x * 2, i)
            results.append(r)
        except Exception as e:  # pragma: no cover - failure path only
            errors.append(e)

    threads = [_threading.Thread(target=_worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert sorted(results) == sorted(i * 2 for i in range(8))
    assert _threading.stack_size() == original


# ---------------------------------------------------------------------------
# USABILITY_A-2 (P2): boruta_select(n_iterations=0) silently produced a NaN win_rate then an unhelpful
# scipy ValueError instead of a clear parameter-validation message.
# ---------------------------------------------------------------------------


def test_regression_residency_audit_nested_same_thread_still_works():
    """Sanity: same-thread nested residency_audit() (the legitimate reentrant case) must not deadlock now
    that a lock guards the monkeypatch region."""
    from mlframe.feature_selection.filters._gpu_strict_fe._audit import residency_audit

    with residency_audit() as outer:
        with residency_audit() as inner:
            pass
    assert outer is not None and inner is not None


# ---------------------------------------------------------------------------
# X_EDGE_CASES_BEST_PRACTICES-5 (P2): _ksg_mi_1d hardcoded random_state=42 regardless of caller intent
# (dead code today, but a latent instance of a bug class fixed elsewhere this audit wave).
# ---------------------------------------------------------------------------


def test_regression_ksg_mi_1d_accepts_random_state():
    """Post-fix: _ksg_mi_1d threads its random_state through to the estimator instead of a hardcoded 42.
    Observed on the output: the caller's seed changes the estimate, a repeated seed reproduces it, and the
    omitted-argument call equals the historical default of 42."""
    from mlframe.feature_selection.filters.hermite_fe._hermite_prewarp import _ksg_mi_1d

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    # Rounded so the continuous feature has ties: the estimator's tie-breaking jitter is what the seed controls.
    x = np.round(y + 0.8 * rng.standard_normal(200), 1)
    v1 = _ksg_mi_1d(x, y, discrete_target=True, random_state=1)
    v1_again = _ksg_mi_1d(x, y, discrete_target=True, random_state=1)
    v2 = _ksg_mi_1d(x, y, discrete_target=True, random_state=2)
    v_default = _ksg_mi_1d(x, y, discrete_target=True)
    v_42 = _ksg_mi_1d(x, y, discrete_target=True, random_state=42)
    assert np.isfinite(v1) and np.isfinite(v2)
    assert v1 == v1_again, "same random_state must reproduce the estimate"
    assert v1 != v2, "random_state is ignored: different seeds give identical estimates"
    assert v_default == v_42, "omitted random_state must keep the historical default of 42"


# ---------------------------------------------------------------------------
# X_EDGE_CASES_BEST_PRACTICES-6 (P2): pack_blocks_to_devices treated a genuinely empty speeds list
# (zero visible devices) identically to the single-device case, returning a bogus device index 0.
# ---------------------------------------------------------------------------


def test_regression_gpu_resident_basis_mi_docstring_not_self_contradicting():
    """X_EFFICIENCY_ARCHITECTURE-6: fe_gpu_resident_basis_mi_enabled's docstring must not claim both
    'DEFAULT OFF' and 'DEFAULT ON when CUDA is present' for the same flag."""
    from mlframe.feature_selection.filters._gpu_resident_fe import fe_gpu_resident_basis_mi_enabled

    doc = fe_gpu_resident_basis_mi_enabled.__doc__ or ""
    # Pre-fix: the opening summary line unqualifiedly asserted "DEFAULT OFF" while a later paragraph
    # asserted "DEFAULT ON when CUDA is present" for the SAME flag. Post-fix, any "DEFAULT OFF" mention
    # must be qualified (e.g. "DEFAULT OFF otherwise"), not a bare unqualified claim.
    assert "(Piece 3). DEFAULT OFF." not in doc
    assert "DEFAULT ON when CUDA is present" in doc
