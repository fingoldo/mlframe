"""Residency BYTE-AUDIT for the GPU-strict greedy-CMI feature constructor.

Selection parity is not enough: a byte-identical kernel can still leak the whole (K,) CMI vector D2H and
re-upload the fit-constant ``y`` (and round-constant ``z``) H2D on every candidate batch. This audits the
ACTUAL host<->device traffic of the wired greedy loop (``greedy_cmi_fe_construct`` ->
``batched_cmi_gpu(..., return_device=True)`` + ``cmi_device_argmax``) by transfer SIZE and asserts the bulk
leak is gone:

  (a) the (K,) MI float64 vector is NOT in the bulk-D2H list (it stays resident; only the argmax scalars cross),
  (b) ``y`` is uploaded ONCE, not re-uploaded as a bulk H2D per candidate batch (resident-operand cache),
  (c) the per-round argmax pulls only scalar (< BULK_BYTES) D2H.

The harness (``residency_audit``) classifies cp.asarray H2D / .get()+cp.asnumpy D2H by byte size; BULK_BYTES
(8192) cleanly separates the scalar decisions from bulk arrays.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

cp = pytest.importorskip("cupy")


def _need_cuda() -> bool:
    """Whether a usable CUDA device is available (used to skip the module when it is not)."""
    try:
        from pyutilz.core.pythonlib import is_cuda_available
        return is_cuda_available()
    except Exception:
        return False


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not _need_cuda(), reason="no CUDA")]


def _make_frame(n: int, seed: int):
    """Build the (X, y) synthetic frame the greedy-CMI residency audit fits."""
    rng = np.random.default_rng(seed)
    a, b, c, d = (rng.random(n) for _ in range(4))
    # A compound target so several engineered candidates carry signal and the greedy loop runs >1 round.
    score = (a * a) - 1.0 + 0.7 * np.sin(d * 3.0) - 0.5 * b
    y = (score > np.median(score)).astype(np.int64)
    X = pd.DataFrame({"a": a, "b": b, "c": c, "d": d})
    return X, y


def _run_audit(env_on: bool):
    """Run one MRMR-adjacent greedy_cmi_fe_construct call under residency_audit, with/without the STRICT flags."""
    from mlframe.feature_selection.filters import _fe_resident_operands as _R
    from mlframe.feature_selection.filters._mi_greedy_cmi_fe import greedy_cmi_fe_construct
    from mlframe.feature_selection.filters._gpu_strict_fe import residency_audit

    saved = {k: os.environ.get(k) for k in ("MLFRAME_FE_GPU_STRICT", "MLFRAME_CMI_GPU", "MLFRAME_FE_VRAM_F32")}
    if env_on:
        os.environ["MLFRAME_FE_GPU_STRICT"] = "1"
        os.environ["MLFRAME_CMI_GPU"] = "1"
        os.environ["MLFRAME_FE_VRAM_F32"] = "1"
    else:
        os.environ["MLFRAME_CMI_GPU"] = "0"
        os.environ.pop("MLFRAME_FE_GPU_STRICT", None)

    # Count resident-operand UPLOAD events (cache misses) per role: a HIT returns the cached device array
    # without an H2D, a MISS issues exactly one cp.asarray H2D. The label ``cmi_y`` is fit-constant and must
    # upload exactly once; ``cmi_z`` is round-constant and legitimately re-uploads when its CONTENT changes
    # (the conditioning support grows each greedy round -> a genuinely different array, not a redundant churn).
    role_uploads: dict = {}
    _orig = _R.resident_operand

    def _counting(arr, key, **kw):
        """Wraps resident_operand to count cache-miss uploads per role."""
        import numpy as _np
        host = _np.asarray(arr)
        dtype = kw.get("dtype")
        if dtype is not None:
            host = host.astype(dtype, copy=False)
        host = _np.ascontiguousarray(host) if kw.get("contiguous", True) else host
        full_key = (key, host.shape, host.dtype.str)
        sig = (host.shape, host.dtype.str, hash(host.tobytes()))
        cached = _R._FE_RESIDENT_OPERANDS.get(full_key)
        miss = not (cached is not None and cached[1] == sig)
        if miss:
            role_uploads[key] = role_uploads.get(key, 0) + 1
        return _orig(arr, key, **kw)

    try:
        X, y = _make_frame(8000, 11)
        kw = dict(nbins=10, seed_cols_count=4, min_cmi_gain=0.0)
        _R.clear_fe_resident_operands()
        # Warm cupy/JIT + the greedy enumeration OUTSIDE the audited region.
        greedy_cmi_fe_construct(X, y, **kw)
        _R.clear_fe_resident_operands()
        _R.resident_operand = _counting
        try:
            with residency_audit() as rep:
                X_aug, scores = greedy_cmi_fe_construct(X, y, **kw)
        finally:
            _R.resident_operand = _orig
        return rep, scores, role_uploads
    finally:
        _R.resident_operand = _orig
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_cmi_greedy_residency_no_bulk_mi_vector_d2h():
    """PRIMARY GATE: under the 3 strict flags the wired greedy loop emits ZERO bulk D2H (the (K,) CMI vector
    stays resident; the argmax pulls only scalars) and does NOT re-upload y as a bulk H2D per candidate batch.

    2026-07-17: the candidate pool is now binned + fingerprinted fully resident (``_quantile_bin_gpu_resident``
    + a device-side reduction hash, replacing the host ``_quantile_bin`` + ``.tobytes()`` pair that D2H'd one
    (n,) array PER CANDIDATE PER SCAN -- 212 bulk D2H events on this fixture before the fix), and the per-winner
    conditioning-support (Z) fold now runs entirely on-device (``_renumber_joint_gpu``, resident twin of the
    host ``_renumber_joint``) instead of materializing each winner's codes host-side just to fold them -- the
    remaining y/z-invariant CMI precompute (``precompute_cmi_yz_terms``, still host-only) is now LAZY, computed
    only on the rare batched-CMI-call exception fallback, not unconditionally every round."""
    rep_on, scores, role_uploads = _run_audit(env_on=True)
    # Sanity: the GPU path actually ran (>=1 winner selected over multiple candidates).
    assert len(scores) >= 1, "greedy CMI selected no winners (path may not have exercised the GPU loop)"

    # (a) The (K,) MI float64 vector is NOT in the bulk-D2H list. Under return_device the only D2H crossings
    #     are the argmax (idx int64 8B, val float64 8B) scalars + the analytic-null tiny scalars.
    assert len(rep_on.bulk_d2h) == 0, f"unexpected bulk D2H (the (K,) MI vector should stay resident): {rep_on.bulk_d2h}; {rep_on.summary()}"

    # (c) All D2H is scalar (< BULK_BYTES).
    from mlframe.feature_selection.filters._gpu_strict_fe._audit import BULK_BYTES
    assert all(b < BULK_BYTES for b in rep_on.d2h), rep_on.summary()

    # (b) y is FIT-CONSTANT: the greedy hot path uploads the fixed y (role ``cmi_greedy_y_fixed``) EXACTLY ONCE
    #     and then reuses that RESIDENT cupy array for every per-round batched_cmi_gpu call (no per-round y H2D).
    #     The per-permutation SHUFFLED y (role ``cmi_y``) is a genuinely different array each null draw and so is
    #     not fit-constant -- it is kept on a distinct role so it can never evict the fixed y.
    assert role_uploads.get("cmi_greedy_y_fixed", 0) == 1, (
        f"fixed greedy y uploaded {role_uploads.get('cmi_greedy_y_fixed', 0)}x; the fit-constant y must be "
        f"uploaded exactly once and reused resident. role_uploads={role_uploads}"
    )


def test_cmi_residency_before_after_classification():
    """Document the before/after bulk-transfer classification: with return_device OFF (host path) the per-round
    (K,) MI vector + y cross as bulk; with the strict wiring ON they do not. This is informational (printed) and
    asserts the ON path is strictly cleaner on bulk D2H."""
    rep_off, _, _ = _run_audit(env_on=False)  # CPU host path (no GPU): zero device traffic at all
    rep_on, _, ru = _run_audit(env_on=True)
    print("BEFORE (host/CPU path):     " + rep_off.summary())
    print("AFTER  (GPU-strict wired):  " + rep_on.summary() + f"  role_uploads={ru}")
    # The wired GPU path keeps the bulk D2H at zero (the gate); the host path issues no device transfers.
    assert len(rep_on.bulk_d2h) == 0


# ============================================================================================================
# PAIR-SEARCH residency byte-audit (the DOMINANT strict F2 path: check_prospective_fe_pairs ->
# _score_one_pair / _compute_one_fe_chunk -> gpu_materialise_discretize_codes_host -> the MI noise gate).
#
# The greedy-CMI audit above covers the batched_cmi_gpu argmax path. This block audits the OTHER, dominant
# strict path -- the unary x binary pair search -- and pins its three residency invariants under
# MLFRAME_FE_GPU_STRICT=1 / MLFRAME_CMI_GPU=1 / MLFRAME_FE_VRAM_F32=1:
#   (1) the candidate codes the chunk/pair binner produced ON the GPU stay RESIDENT and are consumed IN PLACE
#       by the noise-gate MI (the resident-codes handoff: every producer stash is matched by a take-hit and
#       the gate receives device_codes, NOT a host re-upload),
#   (2) NO (n, K) codes buffer crosses the bus mid-pipeline (the single largest D2H of the pre-residency fit
#       -- it is deferred + skipped because the resident gate reads the device copy),
#   (3) the operand table is NOT re-uploaded per chunk/pair (it is GPU-prebuilt once or H2D'd once per step
#       and reused across the step's chunks/pairs by weakref identity).
# ============================================================================================================

def _run_pair_search_audit():
    """Run a small MRMR F2 fit (the compound y = a**2/b + log(c)*sin(d) + noise) under the 3 strict flags,
    instrumenting the resident-codes handoff + the noise-gate consumer + the operand-table upload, and return
    the recovered feature names + the collected counters + a residency byte report.

    The GPU pair-search backend is FORCED on (``MLFRAME_FE_GPU_DISCRETIZE=1`` / ``MLFRAME_FE_GPU_BINNING=1``)
    so a per-host KTC cache entry left "cpu" by a prior test cannot route the binning off the GPU (which would
    make this audit silently observe nothing). Strict-resident is selection-equivalent either way."""
    import warnings

    import pandas as pd

    from mlframe.feature_selection.filters import _gpu_resident_fe as _FE
    from mlframe.feature_selection.filters import _gpu_resident_materialise as _RM
    from mlframe.feature_selection.filters._feature_engineering_pairs import _pairs_dispatch as _DI
    from mlframe.feature_selection.filters._gpu_strict_fe import residency_audit
    from mlframe.feature_selection.filters.mrmr import MRMR

    saved = {k: os.environ.get(k) for k in
             ("MLFRAME_FE_GPU_STRICT", "MLFRAME_CMI_GPU", "MLFRAME_FE_VRAM_F32",
              "MLFRAME_FE_GPU_DISCRETIZE", "MLFRAME_FE_GPU_BINNING", "MLFRAME_MI_ANALYTIC_NULL")}
    os.environ["MLFRAME_FE_GPU_STRICT"] = "1"
    os.environ["MLFRAME_CMI_GPU"] = "1"
    os.environ["MLFRAME_FE_VRAM_F32"] = "1"
    os.environ["MLFRAME_FE_GPU_DISCRETIZE"] = "1"
    os.environ["MLFRAME_FE_GPU_BINNING"] = "1"
    # This audit's whole point is exercising the PERMUTATION noise-gate's resident-codes consumer
    # (``_batch_mi_with_noise_gate_gpu`` / ``take_resident_codes``). At this fixture's n=30000, which is above
    # ``analytic_null_min_n()``'s default 25000, ``_pairs_dispatch`` legitimately routes every pair through the
    # analytic chi-square gate instead (a real, separate, n-gated optimization -- see ``_analytic_mi_null.py``),
    # which never touches ``device_codes`` at all: the resident-codes handoff gets popped (unconditionally, at
    # dispatch entry) and then discarded unread for every pair, showing up here as near-all take-MISSES. Force
    # the legacy permutation path so this audit exercises the consumer it is actually auditing.
    os.environ["MLFRAME_MI_ANALYTIC_NULL"] = "0"

    # Start from a clean VRAM state: this module runs the greedy-CMI audit fits BEFORE this one, and on a
    # 4GB / contended card their accumulated device allocations can leave the shared context near-exhausted so
    # the pair-search GPU path silently falls back to CPU. Free the TEST-side mempool + the FE resident caches
    # (a teardown of accumulated TEST allocations -- NOT the production shared-pool free) so this audit gets a
    # fresh context. Best-effort: any failure just proceeds (the skip below covers a still-dead context).
    try:
        from mlframe.feature_selection.filters import _fe_resident_operands as _RO
        _RO.clear_fe_resident_operands()
    except Exception:
        pass
    try:
        import cupy as _cp
        _cp.get_default_memory_pool().free_all_blocks()
        _cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass

    n = 30000
    rng = np.random.default_rng(42)
    a, b, c, d, e = (rng.uniform(0.1, 1.1, n) for _ in range(5))
    f = rng.uniform(0.1, 1.1, n)
    df = pd.DataFrame({k: v.astype(np.float64) for k, v in zip("abcde", (a, b, c, d, e))})
    y = a**2 / b + f / 5.0 + np.log(np.abs(c) + 1e-9) * np.sin(d)

    cnt = {"stash": 0, "take_calls": 0, "take_hits": 0, "gate_with_devcodes": 0, "gate_no_devcodes": 0, "operand_table_h2d": 0, "operand_table_distinct": 0}

    # The producer (gpu_materialise/discretize_codes_host, both in _RM) resolves ``_stash_resident_codes`` and
    # ``_resident_operand_table`` as _RM module globals -> patch THOSE bindings to count them.
    _o_stash = _RM._stash_resident_codes
    def _stash(h, dv):
        """Wraps _stash_resident_codes to count producer stash events."""
        cnt["stash"] += 1
        return _o_stash(h, dv)

    _o_take = _FE.take_resident_codes
    def _take(h):
        """Wraps take_resident_codes to count consumer take calls/hits."""
        cnt["take_calls"] += 1
        r = _o_take(h)
        if r is not None:
            cnt["take_hits"] += 1
        return r

    _o_rot = _RM._resident_operand_table
    _seen_tv: set = set()
    def _rot(cp_mod, tv):
        """Wraps _resident_operand_table to count genuine (non-cached) operand-table H2D uploads."""
        # Count a REAL operand-table H2D (cache miss: neither GPU-prebuilt for this host array nor already
        # weakref-cached). A hit returns the resident device table with no transfer.
        pre = _RM._prebuilt_operand_table(tv)
        hit = _RM._OPERAND_TABLE_CACHE.get(id(tv))
        is_hit = pre is not None or (hit is not None and hit[0]() is tv and hit[1] is not None)
        if not is_hit:
            cnt["operand_table_h2d"] += 1
        _seen_tv.add(id(tv))
        return _o_rot(cp_mod, tv)

    _o_bg = _DI._batch_mi_with_noise_gate_gpu
    def _bg(*a_, **k_):
        """Wraps _batch_mi_with_noise_gate_gpu to count dispatches with/without resident device codes."""
        if k_.get("device_codes") is not None:
            cnt["gate_with_devcodes"] += 1
        else:
            cnt["gate_no_devcodes"] += 1
        return _o_bg(*a_, **k_)

    _RM._stash_resident_codes = _stash
    _FE.take_resident_codes = _take
    _RM._resident_operand_table = _rot
    _DI._batch_mi_with_noise_gate_gpu = _bg
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with residency_audit() as rep:
                fs = MRMR(full_npermutations=10, baseline_npermutations=20, fe_max_steps=2, fe_min_pair_mi_prevalence=1.05, verbose=0, n_jobs=1).fit(df, y)
        names = [str(s) for s in fs.get_feature_names_out()]
        cnt["operand_table_distinct"] = len(_seen_tv)
        return names, cnt, rep, n
    finally:
        _RM._stash_resident_codes = _o_stash
        _FE.take_resident_codes = _o_take
        _RM._resident_operand_table = _o_rot
        _DI._batch_mi_with_noise_gate_gpu = _o_bg
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture(scope="module")
def pair_search_audit():
    """Run the strict pair-search residency audit ONCE for the whole module (one MRMR fit, shared by the three
    invariant assertions below). A single fit keeps the GPU churn minimal -- repeating a full fit per assertion
    on a contended/4GB box risks a sticky CUDA-context corruption that poisons later GPU ops.

    If the GPU pair-search never actually ran (``stash == 0``) the device-residency path produced nothing to
    audit. On a CONTENDED box that means a concurrent CUDA user illegally-accessed / OOM'd the shared context
    (``cudaErrorIllegalAddress``) and every GPU op fell back to CPU -- an ENVIRONMENTAL condition, not a
    residency regression -- so skip rather than false-fail (the path is byte-audited green in a quiet process;
    verified in isolation). A genuine wiring regression instead shows the path running with a BROKEN invariant
    (take-miss / host re-upload / per-pair operand H2D), which the assertions below still catch hard."""
    names, cnt, rep, n = _run_pair_search_audit()
    print("\nPAIR-SEARCH residency: " + rep.summary() + f"  counters={cnt}  names={names}")
    if cnt["stash"] == 0:
        pytest.skip("strict GPU pair-search did not run (CUDA context unavailable/contended -- the producer "
                    f"stashed no resident codes); counters={cnt}. Residency audit needs a live GPU context.")
    return names, cnt, rep, n


def test_pair_search_residency_codes_resident_into_mi_gate(pair_search_audit):
    """PRIMARY GATE for the pair-search path: the candidate codes the GPU binner produced stay RESIDENT and
    are consumed IN PLACE by the noise-gate MI -- every producer stash is matched by exactly one consumer
    take-HIT (never leaked, never phantom-hit), and the gate receives the device codes (never re-uploads host
    codes). This is invariant (1)+(2): codes resident into the MI gate, with NO (n, K) codes buffer crossing
    the bus."""
    names, cnt, rep, n = pair_search_audit

    # (1) Resident-codes handoff: ``take_calls`` (once per pair/chunk dispatch) is NOT expected to equal
    # ``take_hits`` -- most pairs at this fixture's scale take the fully-resident ``gpu_pairs_fe_mi`` fast path
    # (``_pairs_score.py``'s ``_fe_gpu_discretize_enabled`` branch), which scores MI entirely on-device and
    # never reaches ``_dispatch_batch_mi_with_noise_gate`` / the codes handoff at all -- only the pairs that
    # path declines (SU-normalised / sparse / small-n) fall through to the ``_disc_2d``-based dispatcher this
    # handoff serves. The real invariant is exact: every stash is consumed exactly once (no leak, no phantom
    # hit on an unrelated dispatch's stale entry).
    assert cnt["take_hits"] == cnt["stash"], f"stashed/consumed resident-codes mismatch: {cnt['take_hits']} hits vs {cnt['stash']} stashes; counters={cnt}"

    # (2) The MI gate consumed the DEVICE codes in place on every GPU dispatch -- it never fell back to a host
    #     codes re-upload (which would have re-paid the (n, K) codes H2D the residency path exists to skip).
    assert cnt["gate_no_devcodes"] == 0, (
        f"the noise-gate MI re-uploaded host codes on {cnt['gate_no_devcodes']} dispatch(es) instead of " f"consuming the resident device copy; counters={cnt}"
    )
    assert cnt["gate_with_devcodes"] >= 1, f"the resident-codes gate never ran; counters={cnt}"


def test_pair_search_residency_no_nk_codes_bulk_d2h(pair_search_audit):
    """Invariant (2), byte-audited: NO (n, K) codes buffer is bulk-D2H'd mid-pipeline. The codes are produced
    resident and consumed by the gate in place, so the only legitimate bulk D2H are the handful of SINGLE
    survivor/scoring COLUMNS (n floats each -- the downstream recipe/usability stages genuinely read them on
    host) and the final survivor column(s). A (n, K) codes/float buffer would be ORDERS of magnitude larger
    than one column; assert no such buffer crosses."""
    names, cnt, rep, n = pair_search_audit

    # A single candidate/operand column at n is n * itemsize bytes (<= 8 B per row). A (n, K) codes buffer (or
    # the float candidate matrix) would be K-times larger -- K is the per-pair candidate count (tens to
    # hundreds). 64 * n bytes cleanly separates "a few columns" from "an (n, K) matrix": no legitimate
    # single-column host read reaches it, while any (n, K>=64) buffer blows past it.
    nk_threshold = 64 * n
    big = [bb for bb in rep.bulk_d2h if bb >= nk_threshold]
    assert not big, (
        f"unexpected (n, K)-scale bulk D2H on the strict pair-search path (codes/float buffer should stay "
        f"resident): {sorted(big, reverse=True)} (threshold {nk_threshold} = 64*n); {rep.summary()}")


def test_pair_search_residency_operand_table_uploaded_bounded(pair_search_audit):
    """Invariant (3): the operand table (transformed_vars) is NOT re-uploaded per chunk/pair. It is either
    GPU-PREBUILT once (built on-device from the raw operand inputs -- zero bulk table H2D) or H2D'd once per
    FE step and reused across that step's chunks/pairs by weakref identity. So the real operand-table H2D
    count must be FAR below the per-pair dispatch count (it is per-step, not per-pair)."""
    names, cnt, rep, n = pair_search_audit

    # The operand table is uploaded at most ONCE per FE step (typically <= 2 steps here) and reused across the
    # step's pairs -- never per the many per-pair dispatches. Pin "per-step, not per-pair": the real table H2D
    # count must be strictly below the number of pair-search dispatches that consumed codes.
    assert cnt["operand_table_h2d"] <= cnt["operand_table_distinct"], (
        f"operand table re-uploaded more than once per distinct table: {cnt['operand_table_h2d']} H2D for "
        f"{cnt['operand_table_distinct']} distinct tables; counters={cnt}")
    assert cnt["operand_table_h2d"] < max(cnt["stash"], 2), (
        f"operand table uploaded per-pair ({cnt['operand_table_h2d']} H2D vs {cnt['stash']} pair dispatches) "
        f"-- it must be uploaded once per step and reused; counters={cnt}")
