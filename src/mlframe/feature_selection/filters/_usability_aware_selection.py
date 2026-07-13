"""Usability-aware feature selection -- a LINEAR-downstream selection list alongside
MRMR's model-agnostic MI selection (see tests/feature_selection/MRMR_USABILITY_AWARE_SELECTION_DESIGN.md).

MRMR's Fleuret objective ranks features by MI, which is rank-based and BLIND to linear
usability: on a magnitude-carrying target it picks a high-MI monotone warp a linear model
cannot use over a lower-MI form that is the linearly-aligned interaction. This module runs a
SEPARATE greedy whose relevance blends MI with the HELD-OUT |partial correlation of the
candidate's CONTINUOUS values with the RESIDUAL after the already-selected features|. The
residual is the key: on a heavy-tailed target the dominant term (e.g. ``a**2/b``) swamps a
raw-y correlation / R^2, but once it is selected and removed the residual is bounded and the
weak interaction's linear correlation is visible (measured: F2 linear MAE 0.092 -> 0.052).

The candidate pool is generated here (not reused from the main FE) so it is RICH ENOUGH to
contain the linearly-usable interaction forms the main FE's admission/one-best-per-pair prune
out -- the residual-partial-corr admission keeps a genuine ``mul(log(c),sin(d))`` and rejects an
additive cross-mix ``(a,c)``. Each selected feature is a replayable ``EngineeredRecipe`` (or a
raw column), so ``transform()`` can reproduce the linear feature space on test data.

Self-contained: it does NOT modify ``screen_predictors`` / the MI greedy; the caller runs it as
a second pass and exposes its selection (``support_linear_``) for the suite to route to linear
models.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np


def _scrub(v: np.ndarray, dtype: Any = np.float64) -> np.ndarray:
    """Cast ``v`` to ``dtype`` and replace every non-finite entry (NaN/+inf/-inf) with 0.0, returning a new array."""
    # ``np.where(isfinite, a, 0)`` is bit-identical to ``nan_to_num(nan=0, posinf=0, neginf=0)`` (isfinite
    # is False for exactly nan/+inf/-inf) but ~2.8x faster (no per-call isposinf/isneginf/_getmaxmin
    # machinery): 764us -> 269us on a 100k float32 column. _scrub is called ~17k+/retention fit on full-n
    # columns, so this is a direct cut to the pool-build cost. Verified bit-identical over float32/float64 +
    # nan/inf fuzz (2026-06-21).
    a = np.asarray(v, dtype=dtype)
    return np.where(np.isfinite(a), a, 0)


def _f64(v: np.ndarray) -> np.ndarray:
    """Upcast a stored (possibly float32) candidate column to float64 for MI / correlation /
    recipe-edge computation where the heavy-tail precision matters (transient; not stored)."""
    return np.asarray(v, dtype=np.float64)


def _abscorr(u: np.ndarray, v: np.ndarray) -> float:
    """Absolute Pearson correlation ``|corr(u, v)|`` in float64, used as the diversity / near-duplicate gate. Returns
    0.0 if either input is empty or near-constant (std < 1e-12), or if the raw correlation is non-finite."""
    # GATED GPU PATH (MLFRAME_FE_GPU_USABILITY, default OFF). The cupy twin is float64 + the SAME
    # std<1e-12 guard, but a cupy reduction can reassociate the last bits vs numpy -> a |corr| drift
    # that, on the ULP-sensitive clean-form demotion, could flip a pin. So it is ENABLED only on a host
    # where the gate-on pytest verified the SAME selection; on ANY cupy/device error we fall through to
    # the exact numpy path (the fit is never broken by a GPU problem).
    if _GPU_USABILITY():
        try:
            from ._usability_gpu import gpu_abscorr
            return gpu_abscorr(u, v)
        except Exception:  # nosec B110 - optional/best-effort path, rationale documented
            pass  # fall back to the exact CPU path
    u = _f64(u); v = _f64(v)  # precision for the heavy-tail correlation
    if u.size == 0 or float(np.std(u)) < 1e-12 or float(np.std(v)) < 1e-12:
        return 0.0
    r = np.corrcoef(u, v)[0, 1]
    return abs(float(r)) if np.isfinite(r) else 0.0


def _GPU_USABILITY() -> bool:
    """Whether the gated cupy usability-scoring path is active (``MLFRAME_FE_GPU_USABILITY`` + live
    cupy + global GPU not disabled). Default OFF; the CPU path is the proven, selection-exact default.
    Imported lazily so a no-cupy host never touches the GPU module."""
    try:
        from ._usability_gpu import fe_gpu_usability_enabled
        return fe_gpu_usability_enabled()
    except Exception:
        return False


# bench-attempt-rejected (2026-06-18): per-operand near-duplicate unary dedup (|corr|>0.999) to shrink
# the retention pool's unary^2*binary enumeration. Measured on a structured n=10000 fit: retention
# 68.6s -> 66.4s (~1.5% of fit), because the 'medium' unary set is genuinely DISTINCT functions (sqr /
# log / sin / sqrt / exp / reciproc / cbrt / ...) -- almost nothing dedups, and the dedup's own pairwise
# corr cost offsets most of the saving. The exhaustive unary^2*binary MI search is inherent to
# synergy-safe pair recovery (pruning unaries by marginal relevance would drop low-marginal synergy
# operands). Not worth the added complexity; reverted. Do not re-attempt without a cheaper-MI redesign.


@dataclass
class UsableCandidate:
    """One candidate feature (a raw column or a unary/binary-engineered pair form) evaluated for the linear-usability
    greedy. ``name`` is the display/recipe name, ``values`` the full-n scrubbed continuous column, ``mi`` its binned
    mutual information with the target, ``recipe`` a replayable ``EngineeredRecipe`` (``None`` for a raw column),
    ``src`` the raw operand name(s) the candidate was built from, and ``ops`` the ``(unary_a, unary_b, binary)`` op
    names used to build a pair form (empty for a raw column)."""

    name: str
    values: np.ndarray  # continuous, full-n, scrubbed
    mi: float  # binned MI with y
    recipe: Any = None  # EngineeredRecipe for a pair form; None for a raw column
    src: tuple = ()  # (op_a, op_b) raw names, for diagnostics
    ops: tuple = ()  # (unary_a, unary_b, binary) names, for the recipe builder


def _binned_mi(x: np.ndarray, y_codes: np.ndarray, nbins: int, y_terms: Any = None) -> float:
    """Mutual information of ``x`` with the target, estimated via equi-frequency quantile binning of ``x`` into
    ``nbins`` bins. When ``y_terms`` (the precomputed H(Y)/k_y terms for the fixed target) is supplied, uses the
    faster fixed-y marginal-MI path; otherwise falls back to the general conditional-MI-from-binned estimator."""
    from ._mi_greedy_cmi_fe import _cmi_from_binned, _quantile_bin, marginal_mi_binned_fixed_y
    xb = _quantile_bin(_f64(x), nbins)
    if y_terms is not None:
        # y is fixed across the candidate enumeration; reuse the precomputed H(Y)/k_y. Bit-identical.
        return float(marginal_mi_binned_fixed_y(xb, *y_terms))
    return float(_cmi_from_binned(xb, y_codes, None))


def build_usability_candidate_pool(
    X_df: "Any",
    y_cont: np.ndarray,
    base_names: Sequence[str],
    *,
    unary_preset: str = "medium",
    binary_preset: str = "minimal",
    quantization_nbins: int = 10,
    quantization_method: str = "quantile",
    quantization_dtype: Any = np.int32,
    feature_dtype: Any = np.float32,
    mi_floor: float = 0.02,
    max_per_pair: int = 12,
    diversity_corr: float = 0.97,
    max_pairs: int = 60,
    rank_pairs_by_joint_mi: bool = False,
) -> list[UsableCandidate]:
    """Enumerate raw + unary/binary-product candidates (continuous, replayable) for pairs among
    ``base_names``. Per pair, keep up to ``max_per_pair`` DISTINCT forms clearing ``mi_floor``
    (greedy by MI, dropping any |corr|>``diversity_corr`` near-duplicate). For wide p, restrict to
    the ``max_pairs`` highest-marginal-MI pairs. Each pair form is a replayable
    ``EngineeredRecipe`` (so ``transform`` can reproduce it).

    ``rank_pairs_by_joint_mi`` (default False -> OFF, byte-identical marginal rank): a smart-search prune.
    The per-pair unary x unary x binary enumeration (the ~O(pairs * |unary|^2 * |binary|) MI-kernel core,
    ~100s on a structured fit) is wasted on a pair with NO joint signal. When True, rank pairs by ONE cheap
    binned JOINT MI per pair and keep only the top ``max_pairs`` -- joint MI SURFACES low-marginal synergy
    pairs (XOR/ratio) the marginal-sum rank buries, so a SMALL ``max_pairs`` recovers them while the noise
    pairs (lower joint MI) drop out. One MI eval/pair instead of |unary|^2*|binary|.

    bench-attempt-rejected (FE-wall /loop, 2026-06-22): this is the wall core of BOTH retention passes
    (4.37s cumtime over 2 calls in the F2 100k warm fit). Profiled top-down, the residual time is in the
    already-njit'd parallel MI kernels (``score_pair_combos`` -> ``_pair_combo_mi_njit_table_parallel`` ~1.0s;
    ``_binned_mi`` -> ``_combine_factorize_njit`` ~0.87s) plus first-touch numba JIT compilation of the
    binary/unary dispatchers during the per-combo replay-verify (``apply_recipe``). The Python scaffolding
    here (the lazy-recompute combo loop, the diversity ``_abscorr`` filter, the ``_combo_replay_ok`` cache)
    is already minimal -- ~17k list-appends but <0.05s. No safe pure-overhead/vectorization win was found
    above the 0.5% ship floor that preserves the byte-identical retain/drop on this selection-critical path;
    the next real lever is the MI kernel / a JIT-warmth pre-touch, not this CPU dispatch glue."""
    import pandas as pd
    from .feature_engineering import create_unary_transformations, create_binary_transformations
    from .engineered_recipes import build_unary_binary_recipe, apply_recipe
    from ._mi_greedy_cmi_fe import _quantile_bin, marginal_mi_binned_fixed_y, precompute_marginal_y_terms
    from ._fe_mi_contract import quantize_mi_tiebreak

    # SELECTION-EQUIVALENT retention key: snap the MI sort key to the shared grid ONLY under the resident
    # GPU-strict path (where the resident MI differs from the CPU njit by ~1e-15 and the ULP-tie-sensitive
    # retention sort must not flip the kept set). On the default / non-resident path the CPU njit MI has no
    # such drift, so the sort stays on the raw MI -> BYTE-IDENTICAL default selection (the plan's non-STRICT
    # invariant). Stable sort: exact ties keep enumeration order either way; only sub-quantum near-ties differ.
    try:
        from ._gpu_strict_fe import fe_gpu_strict_resident_enabled
        _seleq = bool(fe_gpu_strict_resident_enabled())
    except Exception:
        _seleq = False
    _mi_key = (lambda m: quantize_mi_tiebreak(m)) if _seleq else (lambda m: m)

    if not isinstance(X_df, pd.DataFrame):
        X_df = pd.DataFrame(np.asarray(X_df))
    y_cont = _scrub(y_cont)
    y_codes = _quantile_bin(y_cont, quantization_nbins)
    # y is fixed for the whole candidate enumeration -> hoist H(Y)/k_y once (reused by every marginal-MI eval).
    y_terms = precompute_marginal_y_terms(y_codes)

    unary = create_unary_transformations(preset=unary_preset)
    binary = create_binary_transformations(preset=binary_preset)
    base_names = [b for b in base_names if b in X_df.columns]

    # Per-base-name RAW extraction + float64 scrub, computed ONCE and reused everywhere a base column
    # is needed at float64 precision: the marginal-MI pair ranking, the joint-MI pair ranking, the
    # resident-GPU pair rebuild, and the main per-pair combo loop all previously re-extracted / re-
    # scrubbed the SAME column (avg ~7-8x redundant per base name at the default pair cap). The raw
    # (unscrubbed) extraction is ALSO cached for the raw-candidate loop below, which needs a SEPARATE
    # ``feature_dtype`` scrub (default float32) -- not derived from ``base_f64`` because downcasting an
    # already-scrubbed float64 value can differ from scrubbing straight to the narrower dtype (a finite
    # float64 value that overflows to +/-inf in float32 must scrub to 0, which only holds when the
    # feature_dtype cast happens BEFORE the finite check).
    raw_np = {nm: X_df[nm].to_numpy() for nm in base_names}
    base_f64 = {nm: _scrub(raw_np[nm]) for nm in base_names}

    pool: list[UsableCandidate] = []
    # raw columns are always candidates (a linear model often wants a raw operand too).
    for nm in base_names:
        col = _scrub(raw_np[nm], feature_dtype)
        if float(np.std(col)) <= 1e-9:
            continue
        pool.append(UsableCandidate(nm, col, _binned_mi(col, y_codes, quantization_nbins, y_terms), None, (nm,), ()))

    pairs = list(itertools.combinations(base_names, 2))
    if rank_pairs_by_joint_mi:
        # SMART-SEARCH pair ranking: rank by binned JOINT MI (one eval/pair) and keep the top
        # ``max_pairs``, so the per-pair unary^2*binary enumeration (~100s core) runs on only the few
        # pairs with the strongest joint signal. Joint MI SURFACES low-marginal synergy pairs (XOR/ratio)
        # that the marginal-sum rank buries. This is a relative RANKING, so the raw (un-MM-corrected,
        # un-occupancy-floored) joint MI is the right tool: a genuine pair's real joint dependence ranks it
        # ABOVE the roughly-uniform finite-sample inflation of the independent-noise pairs, and top-K keeps
        # it -- whereas the MM/occupancy-floored estimator zeroes EVERY pair once rows/cell is small (the
        # 3000-row subsample x 10x10 grid), which would prune the genuine pairs too. Default OFF -> marginal
        # rank, byte-identical.
        _pj_codes = {nm: _quantile_bin(base_f64[nm], quantization_nbins).astype(np.int64) for nm in base_names}
        _nb = int(quantization_nbins)

        def _pair_joint_mi(p):
            """Binned joint MI of the pair ``p = (name_a, name_b)`` with the target: bins each operand independently
            then combines their quantile codes into one joint code (``code_a * nbins + code_b``) before scoring."""
            return float(marginal_mi_binned_fixed_y(_pj_codes[p[0]] * _nb + _pj_codes[p[1]], *y_terms))

        # DEVICE-BATCHED pair ranking (kernel-residency, 2026-07-02): under the resident strict path score ALL
        # pair joint MIs in ONE fused device call (binned_mi_from_codes_gpu computes the SAME plain plug-in MI,
        # no MM bias) instead of the per-pair host loop. The per-base codes are already the device binner's
        # partition (host copies of the strict _quantile_bin route), so the joint codes are identical; the MI
        # is snapped to the shared tie grid (the ~1e-15 device-vs-njit drift must not flip the hard
        # top-max_pairs cut). Any cupy fault -> the exact host loop.
        _pj = None
        if _seleq and pairs:
            try:
                import cupy as _cp
                from ._fe_batched_mi import binned_mi_from_codes_gpu
                _base_dev = {nm: _cp.asarray(_pj_codes[nm]) for nm in base_names}
                _joint = _cp.stack([_base_dev[a] * _nb + _base_dev[b] for a, b in pairs], axis=1)
                _ky = int(np.asarray(y_codes).max()) + 1
                _mis = np.asarray(binned_mi_from_codes_gpu(_joint, y_codes, ky=_ky, codes_trusted=True), dtype=np.float64)
                _pj = {p: quantize_mi_tiebreak(float(_mis[i])) for i, p in enumerate(pairs)}
            except Exception:
                _pj = None
        if _pj is None:
            _pj = {p: _mi_key(_pair_joint_mi(p)) for p in pairs}
        pairs.sort(key=lambda p: _pj[p], reverse=True)
        pairs = pairs[:max_pairs]
    else:
        # rank pairs by marginal-MI sum so a wide-p sweep keeps the most promising first.
        marg = {nm: _binned_mi(base_f64[nm], y_codes, quantization_nbins, y_terms) for nm in base_names}
        pairs.sort(key=lambda p: marg[p[0]] + marg[p[1]], reverse=True)
        pairs = pairs[:max_pairs]

    # FUSED njit PER-PAIR ENUMERATION (retention path only, 2026-06-18). On the retention path
    # (``rank_pairs_by_joint_mi=True``) the per-pair ``|unary|^2*|binary|`` value+quantile-bin+MI triple
    # is Python-dispatched per combo (~3.5s/pair at n=10000, ~62s of a structured fit). When every
    # preset op is njit-coded, score ALL combos for a pair in ONE njit(parallel) kernel
    # (``score_pair_combos``) -- bit-faithful to the Python MI (verified ~6e-15) -- then recompute the
    # numpy value only for the (bounded) combos clearing ``mi_floor`` so the diversity filter + recipe
    # replay are UNCHANGED. The default (marginal-rank) path stays byte-identical (Python loop below).
    _ua_codes = _ub_codes = _bn_codes = None
    if rank_pairs_by_joint_mi:
        from ._usability_njit_pool import (
            njit_unary_codes_or_none, njit_binary_codes_or_none, score_pair_combos,
        )
        _unary_names = list(unary.keys())
        _binary_names = list(binary.keys())
        _uc = njit_unary_codes_or_none(_unary_names)
        _bc = njit_binary_codes_or_none(_binary_names)
        if _uc is not None and _bc is not None:
            _ua_codes, _ub_codes, _bn_codes = _uc, _uc, _bc  # ua/ub share the unary code table

    # REPLAY-VERIFICATION CACHE (2026-06-18): the per-candidate ``apply_recipe`` replay + allclose was
    # ~0.35s each and a large chunk of the pool build. The fused/Python value path already produces the
    # candidate values bit-faithfully, and these are STANDARD ``build_unary_binary_recipe`` recipes whose
    # replayABILITY is a property of the (unary_a, unary_b, binary) op-combo + the recipe machinery, not of
    # the specific operand pair or the per-candidate edges (the only per-candidate state is the pinned
    # quantile/uniform edge array, which never changes WHETHER a recipe replays, only its exact values --
    # already covered by the recompute being bit-identical). So run the full apply_recipe + allclose check
    # ONCE per distinct op-combo; for later candidates of a verified combo, trust the recipe (skip the
    # expensive replay). A combo whose first verification FAILS is blacklisted -> all its candidates drop,
    # and any recipe whose ``build_unary_binary_recipe``/``apply_recipe`` RAISES still drops individually.
    # Contract preserved: every recipe that reaches the returned pool is replayable by ``transform()``.
    _combo_replay_ok: dict[tuple, bool] = {}

    # bench-attempt-rejected (iter17, 2026-06-23): GPU-RESIDENT batched pair-combo MI TABLE. The MI-table
    # computation IS cleanly separable from this retention/diversity bookkeeping (the loop only reads
    # ``mis[j]`` per pair), and a resident batched-across-pairs table was built + gated
    # (``_usability_pool_resident.py`` + ``_usability_pool_resident_ktc.py``, kept for a capable card).
    # NOT WIRED IN, for TWO independent reasons measured here: (1) it LOSES on the dev GTX 1050 Ti -- n=100k
    # npairs=4 nc=1734/pair CUDA-event A/B: 29.6s resident vs 14.7s CPU njit = 0.50x, because the bit-faithful
    # ``_gpu_quantile_bin_codes``/``_gpu_marginal_mi`` do a per-row device->host scalar sync (~14k tiny syncs)
    # the 6-SM card cannot hide (HW-bound regime). (2) MORE IMPORTANTLY it is NOT selection-equivalent: the
    # table is bit-faithful to ~6e-15, but the downstream STABLE MI-sort + greedy ``_abscorr`` diversity
    # filter is ULP-sensitive at MI ties -- a 6e-15 reassociation flips the tie ORDER, changing which of two
    # near-equal-MI combos is retained (verified: a 125-form structured pool had ~6 retained forms DIFFER,
    # e.g. mul(invsquared(a),neg(b)) vs mul(invsquared(a),identity(b))). Selection must stay byte-identical on
    # this path, so the resident MI is not fed in. NEEDS-X to ship: a BIT-EXACT (not just bit-faithful) GPU MI
    # matching the njit reduction order, AND a row-vectorised sync-free bin+MI kernel, AND a card where it wins.
    # RESIDENT PAIR-COMBO MI TABLE NOW WIRED (2026-06-27): ``score_pair_combos_table_resident`` is fed into the
    # retention loop under the resident GPU-strict flag (``_seleq``) only. The (3) blocker the iter17 note below
    # records -- the fused resident binning diverged from the njit on low-cardinality columns -- was fixed in
    # 71e31818 (the resident binner now matches the njit distinct-edge dedup), so the resident table is now
    # SELECTION-EQUIVALENT to the njit per-pair ``score_pair_combos`` (parity test green:
    # tests/feature_selection/gpu/test_usability_pool_resident_parity.py). The ULP-tie sensitivity is absorbed by
    # the ``_mi_key`` grid-snap (already engaged under ``_seleq``). The DEFAULT (flag-off) path is BYTE-IDENTICAL
    # -- it never computes the resident table and uses the per-pair njit ``score_pair_combos`` exactly as before.
    # If the resident table errors (no cupy / device fault) it returns None and we fall back per-pair. This is a
    # residency win (the MI runs on-device under the flag), not necessarily a wall win at the FE-subsample n.
    _resident_table = None
    if _seleq and _ua_codes is not None:
        try:
            from ._usability_pool_resident import score_pair_combos_table_resident

            assert _ub_codes is not None and _bn_codes is not None  # ua/ub/bn are set together at the same tuple-unpack site
            _res_ops = [(base_f64[n1], base_f64[n2]) for n1, n2 in pairs]
            _resident_table = score_pair_combos_table_resident(
                _res_ops, y_codes, y_terms, quantization_nbins, _ua_codes, _ub_codes, _bn_codes,
            )
        except Exception:
            _resident_table = None
    for _pidx, (n1, n2) in enumerate(pairs):
        x1 = base_f64[n1]
        x2 = base_f64[n2]
        cand_here: list[UsableCandidate] = []
        if _ua_codes is not None:
            # njit-scored retention path. The kernel enumerates ``for ua: for ub: for bn`` in the SAME
            # order as the Python loop, so the flat combo index maps 1:1 to (ua, ub, bn) below.
            _unary_names = list(unary.keys())
            _binary_names = list(binary.keys())
            if _resident_table is not None:
                # resident GPU table row p == score_pair_combos for pair p (selection-equivalent after the
                # distinct-edge dedup fix; ULP ties absorbed by ``_mi_key`` grid-snap engaged under ``_seleq``).
                mis = _resident_table[_pidx]
            else:
                mis = score_pair_combos(
                    x1, x2, y_codes, y_terms, quantization_nbins, _ua_codes, _ub_codes, _bn_codes,
                )
            nu = len(_unary_names)
            nb = len(_binary_names)
            # LAZY-RECOMPUTE (2026-06-21): the njit kernel already produced the MI of EVERY combo, and the
            # only use of the recomputed numpy value is (a) the per-pair diversity filter that keeps the
            # top ``max_per_pair`` MI-ranked DISTINCT forms and (b) the recipe replay for those few kept
            # forms. The prior code materialised the float64 value + ``_scrub`` for EVERY mi_floor-clearing
            # combo (~1700/pair here) only to discard all but 3 -- ~17k full-n value+scrub builds/fit, the
            # second-largest retention cost after the kernel. Instead, collect only the cheap combo METADATA
            # (mi + op indices), sort by MI (STABLE -> identical tie order to the old append-order +
            # ``cand_here.sort(key=mi, reverse=True)``), then recompute the numpy value LAZILY while building
            # the diverse ``kept`` set, stopping at ``max_per_pair``. Selection-identical: the same MI order,
            # the same diversity gate, the same kept forms -- just ~10-15 value builds/pair instead of ~1700.
            metas = []  # (mi, ia, ib, ibn) for floor-clearing combos, in enumeration order
            j = 0
            for ia in range(nu):
                for ib in range(nu):
                    for ibn in range(nb):
                        m = float(mis[j]); j += 1
                        if m < mi_floor:   # also rejects the -1.0 std<=1e-9 sentinel
                            continue
                        metas.append((m, ia, ib, ibn))
            # stable sort by MI desc; under the resident path the key is grid-snapped (``_mi_key``) so a
            # sub-quantum (~1e-15) CPU-vs-GPU MI difference can't flip the kept set. Default path: raw MI
            # (byte-identical). Exact ties keep enumeration order; only within-quantum near-ties differ.
            metas.sort(key=lambda t: _mi_key(t[0]), reverse=True)
            _ta_cache: dict = {}  # unary(x1) by ua index -- reused across combos sharing ua
            _tb_cache: dict = {}  # unary(x2) by ub index
            _njit_kept: list[UsableCandidate] = []
            for m, ia, ib, ibn in metas:
                if len(_njit_kept) >= max_per_pair:
                    break
                ua = _unary_names[ia]; ub = _unary_names[ib]; bn = _binary_names[ibn]
                ta = _ta_cache.get(ia)
                if ta is None:
                    ta = unary[ua](x1); _ta_cache[ia] = ta
                tb = _tb_cache.get(ib)
                if tb is None:
                    tb = unary[ub](x2); _tb_cache[ib] = tb
                try:
                    val = _scrub(binary[bn](ta, tb), feature_dtype)
                except Exception:  # nosec B112 - best-effort path
                    continue
                if any(_abscorr(val, k.values) > diversity_corr for k in _njit_kept):
                    continue
                name = f"{bn}({ua}({n1}),{ub}({n2}))"
                _njit_kept.append(UsableCandidate(name, val, m, None, (n1, n2), (ua, ub, bn)))
            # already MI-sorted + diversity-filtered + capped -> feed straight to the recipe builder.
            cand_here = _njit_kept
        else:
            # Default / fallback Python loop (``rank_pairs_by_joint_mi=False``, the shipped default).
            #
            # bench-attempt-rejected (2026-07-13, Wave 11 audit item M11): swapping this branch to the
            # SAME ``score_pair_combos`` njit kernel the ``True`` branch uses is NOT a safe drop-in here,
            # despite looking like one. ``score_pair_combos`` is documented "bit-faithful to the Python MI
            # (verified ~6e-15)" -- NOT bit-identical -- and this function's own docstring states the
            # DEFAULT (marginal-rank) path is "byte-identical" by design (the ``_mi_key`` grid-snap that
            # absorbs sub-quantum ties is deliberately gated to the resident-GPU-strict path only, see
            # ``_seleq`` above). The retention loop below is a STABLE sort by MI keeping only the top
            # ``max_per_pair`` DISTINCT forms; a prior investigation into feeding njit-computed MI into this
            # exact retention/diversity logic on the non-resident path (see the "RESIDENT PAIR-COMBO MI
            # TABLE" bench-attempt-rejected note above, ~90 lines up) measured ~6e-15 reassociation flipping
            # the retained SET on a real 125-form pool (~6 forms differed) -- selection-altering, not a pure
            # FP-reorder. Feeding ``score_pair_combos`` output into the default path's un-snapped ``_mi_key``
            # would reproduce that exact regression for the SHIPPED DEFAULT config, not an opt-in one.
            # A snap-then-batch variant (grid-snapping this path's MI too) would trade the "byte-identical
            # default" contract for a "selection-equivalent-only" one -- a real behavior-contract change,
            # not a pure perf refactor, so it is out of scope for this pass; not applied.
            ta_by_ua: dict = {}
            for _ua in unary:
                try:
                    ta_by_ua[_ua] = unary[_ua](x1)
                except Exception:  # nosec B110 - best-effort path  # noqa: PERF203 -- per-iteration fault isolation is intentional, not a hoisting candidate
                    pass
            tb_by_ub: dict = {}
            for _ub in unary:
                try:
                    tb_by_ub[_ub] = unary[_ub](x2)
                except Exception:  # nosec B110 - best-effort path  # noqa: PERF203 -- per-iteration fault isolation is intentional, not a hoisting candidate
                    pass
            for ua, ta in ta_by_ua.items():
                for ub, tb in tb_by_ub.items():
                    for bn, bf in binary.items():
                        try:
                            val = _scrub(bf(ta, tb), feature_dtype)
                        except Exception:  # nosec B112 - best-effort path
                            continue
                        if float(np.std(val)) <= 1e-9:
                            continue
                        m = _binned_mi(val, y_codes, quantization_nbins, y_terms)
                        if m < mi_floor:
                            continue
                        name = f"{bn}({ua}({n1}),{ub}({n2}))"
                        cand_here.append(UsableCandidate(name, val, m, None, (n1, n2), (ua, ub, bn)))
        # keep diverse top-MI forms for this pair. ``_mi_key`` grid-snaps under the resident path (invariant to
        # sub-quantum CPU-vs-GPU MI reassociation); raw MI on the default path (byte-identical). Stable sort.
        cand_here.sort(key=lambda c: _mi_key(c.mi), reverse=True)
        kept: list[UsableCandidate] = []
        for c in cand_here:
            if len(kept) >= max_per_pair:
                break
            if any(_abscorr(c.values, k.values) > diversity_corr for k in kept):
                continue
            kept.append(c)
        # build replayable recipes only for the kept forms (cheap: bounded count). Use the stored
        # (ua, ub, bn) ops directly -- never re-parse the display name.
        for c in kept:
            ua, ub, bn = c.ops
            combo = (ua, ub, bn)
            try:
                recipe = build_unary_binary_recipe(
                    name=c.name, src_a_name=c.src[0], src_b_name=c.src[1],
                    unary_a_name=ua, unary_b_name=ub,
                    binary_name=bn, unary_preset=unary_preset, binary_preset=binary_preset,
                    quantization_nbins=quantization_nbins, quantization_method=quantization_method,
                    quantization_dtype=quantization_dtype,
                    fit_values_for_edges=_f64(c.values),  # edges need float64 precision
                )
            except Exception:  # nosec B112 - optional/best-effort path, rationale documented
                continue  # recipe could not even be built -> not replayable, drop.
            # Verify the replay ONCE per distinct op-combo (see cache note above); trust verified combos
            # for later candidates. A recipe whose replay RAISES or MISMATCHES on its first sighting
            # blacklists the combo (and drops). This keeps the "non-replayable recipe never reaches output"
            # contract while paying the ~0.35s apply_recipe at most once per (ua, ub, bn).
            ok = _combo_replay_ok.get(combo)
            if ok is None:
                try:
                    replay = _scrub(apply_recipe(recipe, X_df), feature_dtype)
                    ok = bool(replay.shape == c.values.shape and np.allclose(_f64(replay), _f64(c.values), atol=1e-4, equal_nan=True))
                except Exception:
                    ok = False
                _combo_replay_ok[combo] = ok
            if ok:
                c.recipe = recipe
                pool.append(c)
    return pool


def usability_greedy(
    pool: list[UsableCandidate],
    y_cont: np.ndarray,
    *,
    w: float = 0.7,
    K: int = 8,
    seed: int = 0,
    n_folds: int = 4,
    mae_improve_rel: float = 0.01,
    shortlist: int = 40,
    classification: bool = False,
) -> list[UsableCandidate]:
    """CROSS-VALIDATED forward selection for the LINEAR downstream: greedily add the candidate that
    most reduces the K-fold CV mean-absolute-error of a linear model on the selected set, stopping
    when no candidate improves it by ``mae_improve_rel`` (relative). This is the gold-standard
    linear wrapper -- it directly optimises the deployed objective, so it inherently (a) prefers the
    LINEARLY-usable form over a high-MI monotone warp a linear model cannot use, (b) drops redundant
    forms (no CV gain), and (c) is robust to OVERFITTING a single held-out slice: a feature that
    helps one fold by chance but drags in a noise operand (e.g. ``min(log(d),e)``) does NOT lower
    the AVERAGE CV MAE, so it is rejected (a single-split gate let it through and regressed F2 n=80k
    MAE 0.054 -> 0.102; CV keeps it ~0.055). The stop replaces a hardcoded feature count.

    A cheap usability pre-rank (``MI + |corr with the post-dominant residual|``) shortlists the pool
    to ``shortlist`` candidates so the per-step CV cost is bounded; ``w`` weights the MI vs the
    residual-corr in that pre-rank only (the COMMIT decision is always the CV-MAE improvement).

    GPU-RESIDENT DISPATCH (``MLFRAME_FE_GPU_STRICT`` + ``MLFRAME_FE_GPU_STRICT_RESIDENT``, default OFF):
    under the resident flag the REGRESSION greedy is computed by the resident twin
    (:func:`_usability_greedy_gpu_resident.usability_greedy_gpu_resident`) with the candidate value
    matrix + target uploaded ONCE and the per-round residual/|corr| shortlist + bordered CV-MAE solve
    kept resident -- killing the per-candidate value D2H the gated ``_usability_gpu`` primitives incur.
    It is SELECTION-EQUIVALENT (same algorithm; only float reduction order differs ~1e-12) and returns
    ``None`` -> the exact CPU body below for classification, a degenerate pool, a singular border, or any
    cupy/device error. The default (flag-off) path never imports it -> byte-identical."""
    try:
        from ._gpu_strict_fe import fe_gpu_strict_resident_enabled
        if fe_gpu_strict_resident_enabled():
            from ._usability_greedy_gpu_resident import usability_greedy_gpu_resident
            _res = usability_greedy_gpu_resident(
                pool, y_cont, w=w, K=K, seed=seed, n_folds=n_folds,
                mae_improve_rel=mae_improve_rel, shortlist=shortlist, classification=classification,
            )
            if _res is not None:
                return _res
    except Exception:  # nosec B110 - optional/best-effort path, rationale documented
        pass  # any import/device error -> the exact CPU greedy below
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    # CLASSIFICATION mode (2026-06-18): the LINEAR downstream for a classification target is a LOGISTIC
    # model, and the deployed objective is a CLASSIFICATION metric -- so the wrapper must score by
    # CROSS-VALIDATED LOGLOSS of a logistic model, not CV-MAE of a linear regression. Mirrors the
    # regression structure exactly (lower-is-better metric, majority-of-folds improvement gate); the
    # regression path (classification=False) is byte-identical.
    if classification:
        y_enc = np.asarray(y_cont).ravel()
        # encode to dense 0..C-1 class codes
        _classes, y_enc = np.unique(y_enc, return_inverse=True)
        n_classes = int(_classes.size)
        if n_classes < 2:
            return []

        def _mk():
            """Build a fresh scaled-logistic-regression pipeline (classification path CV scorer/pre-rank model)."""
            return make_pipeline(StandardScaler(), LogisticRegression(max_iter=200))

        from sklearn.metrics import log_loss as _log_loss
        _labels = np.arange(n_classes)

        def _logloss(y_true, proba):
            """Multi-class log loss of ``proba`` against ``y_true``, evaluated over the fixed full label set
            ``_labels`` (so a fold missing a class still scores against the complete class set)."""
            return float(_log_loss(y_true, proba, labels=_labels))
    else:
        # bench-attempt-rejected (2026-06-13): an audit flagged OLS min-norm as fragile on a singular/wide
        # selected set and suggested a small Ridge in the CV. Benched OLS vs Ridge(alpha=1) on F2: n=8000
        # IDENTICAL (0.0554 per-seed -- the design is never singular here, K<=8 features vs folds of 1000s
        # of rows), n=500 marginally WORSE (0.0670 -> 0.0672). The singular regime needs per-fold rows <
        # selected-feature count, which cannot happen given the shortlist + the small K, so Ridge buys
        # nothing. Kept OLS (the gold-standard wrapper that matches the deployed objective). Do not re-add.
        def _mk():
            """Build a fresh scaled-OLS-linear-regression pipeline (regression path CV scorer/pre-rank model)."""
            return make_pipeline(StandardScaler(), LinearRegression())

    if not pool:
        return []
    if classification:
        n = y_enc.shape[0]
    else:
        y_cont = _scrub(y_cont)
        n = y_cont.shape[0]
    if n < 2:
        return []  # cannot cross-validate a usability greedy on < 2 rows

    # MEM-2 RAM GOVERNOR (2026-06-18). The forward selection builds ``np.column_stack`` design
    # matrices of up to ``K`` float64 columns (``Xsel``, ``Xs``) -- an (n, K) transient that, on a
    # very-wide x very-large-n retention pass, can spike RAM with no psutil check. Best-effort cap:
    # ask the SAME governor the FE buffers use (``_can_hoist_shared_buffer``) whether an (n, K)
    # float64 design fits the LIVE budget; if not, shrink ``K`` (and the shortlist with it) to the
    # largest width that fits (floored at 1) so the greedy still runs. This only triggers under
    # genuine memory pressure: on a normal-RAM host the (n, K<=8) design always fits and ``K`` is
    # unchanged, so selection is identical. Any psutil/import failure -> proceed with the full ``K``.
    try:
        from .feature_engineering import _can_hoist_shared_buffer, _fe_effective_buffer_budget_bytes

        _k_eff = max(1, min(int(K), len(pool)))
        _can, _need, _avail = _can_hoist_shared_buffer(n * _k_eff * 8, n_workers=1)
        if (not _can) and _avail > 0:
            # Cap K to the largest float64 (n, K) design that fits the SAME overhead-aware budget the
            # gate used (not the raw available), flooring at 1 so the greedy always makes progress.
            _budget = _fe_effective_buffer_budget_bytes(_avail, n_workers=1)
            _k_fit = int(_budget // (n * 8)) if _budget > 0 else 1
            if _k_fit < _k_eff:
                K = max(1, _k_fit)
                shortlist = min(int(shortlist), max(int(K), 1))
    except Exception:  # nosec B110 - best-effort path
        pass

    rng = np.random.default_rng(int(seed))
    # BALANCED PARTITION (audit fix, 2026-06-13): a random ``rng.integers(0, n_folds)`` multinomial
    # assignment can leave a fold EMPTY at small n / large n_folds -> an empty TRAIN fold crashes
    # ``fit`` and an empty TEST fold yields a NaN MAE that poisons the per-fold consistency gate. A
    # shuffled ``arange(n) % k`` partition guarantees every fold has floor/ceil(n/k) >= 1 rows.
    n_folds = max(2, min(int(n_folds), n))
    folds = np.arange(n) % n_folds
    rng.shuffle(folds)
    mi_max = max((c.mi for c in pool), default=1.0) or 1.0

    # INCREMENTAL CV (2026-06-18, was PERF TODO 2026-06-13): the regression scorer no longer refits a
    # StandardScaler+LinearRegression for every (candidate, fold). ``StandardScaler -> LinearRegression
    # (fit_intercept=True)`` predictions are INVARIANT to per-column affine scaling, so they equal a raw
    # mean-CENTERED OLS-with-intercept fit; that lets us work with the centered normal equations directly.
    # Within one greedy step ``selected`` is fixed, so per fold the selected-set centered Gram
    # ``Gs = Xc_tr.T @ Xc_tr`` and ``bs = Xc_tr.T @ yc_tr`` are computed ONCE; each shortlist candidate is a
    # rank-1 BORDER (one extra row/col of the Gram + one dot with y), solved as a (k+1)x(k+1) system
    # (k <= K, trivial) instead of an O(n*k^2) refit. The per-candidate work is then O(n) for the borders +
    # O(k^3) for the solve. The selection is the SAME as the full-refit path: centered OLS is the unique
    # minimiser the SVD-based LinearRegression also finds (matched to ~1e-9 on the gate datasets). The
    # classification (logistic / CV-logloss) path and the no-selection MAE baseline stay on ``_cv_per_fold``
    # (byte-identical), and ``_cv_per_fold`` remains the exact fallback for any fold whose bordered Gram is
    # singular (degenerate / collinear column) so correctness never depends on the fast path.

    # Per-fold train/val masks + train-centered selected design, cached for the current step.
    _fold_tr = [folds != fo for fo in range(n_folds)]
    _fold_va = [folds == fo for fo in range(n_folds)]

    # Candidates get re-visited across greedy steps (shortlisted again, re-scored in _cv_per_fold /
    # _shortlist) and pool[i].values never changes for the life of this call, so cache each candidate's
    # float64 upcast once instead of re-casting its (often float32) stored array on every visit.
    _f64_by_idx: dict[int, np.ndarray] = {}

    def _pv(i: int) -> np.ndarray:
        """Cache-backed ``_f64(pool[i].values)`` for the life of this ``usability_greedy`` call."""
        v = _f64_by_idx.get(i)
        if v is None:
            v = _f64(pool[i].values)
            _f64_by_idx[i] = v
        return v

    def _cv_candidates_incremental(sel_idx, cand_list) -> dict:
        """Return {cand_i: per-fold MAE array} for the regression path via the bordered normal equations.
        Falls back to a per-candidate ``_cv_per_fold`` for any fold where the centered border is singular."""
        # Precompute, ONCE per step per fold: centered selected design (train+val), ybar, Gs, bs.
        Xsel = np.column_stack([_pv(j) for j in sel_idx]) if sel_idx else None
        per_fold = []
        for fo in range(n_folds):
            tr, va = _fold_tr[fo], _fold_va[fo]
            ytr = y_cont[tr]
            ybar = float(ytr.mean())
            yc = ytr - ybar
            if sel_idx:
                assert Xsel is not None  # Xsel is built from the same sel_idx truthiness check above
                Str = Xsel[tr]; Sva = Xsel[va]
                mu = Str.mean(axis=0)
                Sc_tr = Str - mu
                Sc_va = Sva - mu
                Gs = Sc_tr.T @ Sc_tr
                bs = Sc_tr.T @ yc
            else:
                Sc_tr = Sc_va = None; mu = None; Gs = None; bs = None
            per_fold.append((tr, va, ybar, yc, Sc_tr, Sc_va, Gs, bs))

        out: dict = {}
        for i in cand_list:
            ci = _pv(i)
            errs = np.empty(n_folds, dtype=np.float64)
            singular = False
            for fo in range(n_folds):
                tr, va, ybar, yc, Sc_tr, Sc_va, Gs, bs = per_fold[fo]
                ctr = ci[tr]; cva = ci[va]
                cmu = float(ctr.mean())
                cc_tr = ctr - cmu
                cc_va = cva - cmu
                if Sc_tr is None:
                    d = float(cc_tr @ cc_tr)
                    if d <= 1e-12:
                        singular = True; break
                    beta = float(cc_tr @ yc) / d
                    pred = ybar + cc_va * beta
                else:
                    g = Sc_tr.T @ cc_tr  # cross terms (k,)
                    d = float(cc_tr @ cc_tr)  # new diagonal
                    bn = float(cc_tr @ yc)  # new rhs entry
                    k = Gs.shape[0]
                    G = np.empty((k + 1, k + 1), dtype=np.float64)
                    G[:k, :k] = Gs; G[:k, k] = g; G[k, :k] = g; G[k, k] = d
                    rhs = np.empty(k + 1, dtype=np.float64)
                    rhs[:k] = bs; rhs[k] = bn
                    try:
                        beta_vec = np.linalg.solve(G, rhs)
                    except np.linalg.LinAlgError:
                        singular = True; break
                    pred = ybar + Sc_va @ beta_vec[:k] + cc_va * beta_vec[k]
                errs[fo] = float(np.mean(np.abs(y_cont[va] - pred)))
            if singular:
                # exact fallback: refit through the standard pipeline for this candidate.
                out[i] = _cv_per_fold([*sel_idx, i])
            else:
                out[i] = errs
        return out

    def _cv_per_fold(sel_idx) -> np.ndarray:
        """Exact (full-refit) K-fold CV score of the candidate set ``sel_idx``: per fold, fit ``_mk()`` on the
        train rows and score the held-out rows, returning one error value per fold (CV log loss when
        ``classification``, else CV mean-absolute-error). With an empty ``sel_idx`` scores the no-selection
        baseline (train-fold class-prior probabilities for classification, train-fold mean for regression). This
        is the exact fallback for any fold where the incremental bordered-Gram solve in
        ``_cv_candidates_incremental`` is singular."""
        if classification:
            # CV LOGLOSS of a logistic model (lower-is-better, same gate semantics as MAE). The
            # no-selection baseline is the constant train-fold class-PRIOR probability.
            if not sel_idx:
                errs = []
                for fo in range(n_folds):
                    trm, vam = folds != fo, folds == fo
                    prior = np.bincount(y_enc[trm], minlength=n_classes).astype(np.float64)
                    prior = prior / max(prior.sum(), 1.0)
                    prior = np.clip(prior, 1e-12, 1.0)
                    proba = np.tile(prior, (int(vam.sum()), 1))
                    errs.append(_logloss(y_enc[vam], proba))
                return np.asarray(errs, dtype=np.float64)
            Xs = np.column_stack([_pv(i) for i in sel_idx])
            errs = []
            for fo in range(n_folds):
                trm, vam = folds != fo, folds == fo
                if np.unique(y_enc[trm]).size < 2:
                    errs.append(np.inf)
                    continue
                m = _mk().fit(Xs[trm], y_enc[trm])
                proba = m.predict_proba(Xs[vam])
                errs.append(_logloss(y_enc[vam], proba))
            return np.asarray(errs, dtype=np.float64)
        if not sel_idx:
            return np.array([float(np.mean(np.abs(y_cont[folds == fo] - float(np.mean(y_cont[folds != fo]))))) for fo in range(n_folds)])
        Xs = np.column_stack([_pv(i) for i in sel_idx])
        errs = []
        for fo in range(n_folds):
            trm, vam = folds != fo, folds == fo
            m = _mk().fit(Xs[trm], y_cont[trm])
            errs.append(float(np.mean(np.abs(y_cont[vam] - m.predict(Xs[vam])))))
        return np.asarray(errs, dtype=np.float64)

    # cheap residual-aware pre-rank to a bounded shortlist (so per-step CV stays cheap).
    def _shortlist(sel_idx) -> list[int]:
        """Rank every not-yet-selected pool candidate by a cheap pre-rank score -- ``(1-w) * (mi / mi_max) + w *
        |corr(candidate, held-out residual)|`` -- and return the indices of the top ``shortlist`` (default 40)
        candidates. The residual is computed on the fold-0 held-out rows only (fit on the other folds, to avoid
        leakage): for regression it is ``y - model.predict(X)``, for classification the positive/majority-class
        indicator minus its predicted probability; with no prior selection it falls back to the (train-fold) mean/
        prior-centered residual over all rows. This pre-rank only bounds the candidate pool the greedy's expensive
        per-step CV evaluates -- the actual commit decision is always the CV-MAE/logloss improvement."""
        # HELD-OUT residual (audit fix, 2026-06-13): fit on the fold-0-out train rows but score the
        # candidate correlation on the HELD-OUT fold-0 residual only -- the prior code predicted over
        # ALL rows (in-sample for the ~(k-1)/k training rows), which is the leakage the module's
        # "held-out residual" design explicitly avoids. The no-selection case uses the mean residual
        # over all rows (no model fit -> no leakage).
        if classification:
            # CLASSIFICATION residual: correlate each candidate with the POSITIVE-class indicator
            # residual (point-biserial-style). For binary y the indicator is 1{y==last class}; once a
            # logistic model is selected, the residual is indicator - P(positive). For multiclass we
            # fall back to the one-vs-rest indicator of the majority class (a cheap pre-rank only -- the
            # COMMIT decision is always the CV-logloss improvement).
            if n_classes == 2:
                pos = (y_enc == 1).astype(np.float64)
            else:
                _maj = int(np.argmax(np.bincount(y_enc, minlength=n_classes)))
                pos = (y_enc == _maj).astype(np.float64)
            if sel_idx:
                Xs = np.column_stack([_pv(i) for i in sel_idx])
                ho = folds == 0
                tr = ~ho
                if np.unique(y_enc[tr]).size >= 2:
                    m = _mk().fit(Xs[tr], y_enc[tr])
                    proba = m.predict_proba(Xs[ho])
                    if n_classes == 2:
                        phat = proba[:, 1]
                    else:
                        phat = proba[:, _maj]
                    resid = pos[ho] - phat
                else:
                    resid = pos[ho] - float(np.mean(pos[tr]))
                rows = ho
            else:
                resid = pos - float(np.mean(pos))
                rows = slice(None)
        elif sel_idx:
            Xs = np.column_stack([_pv(i) for i in sel_idx])
            ho = folds == 0
            tr = ~ho
            m = _mk().fit(Xs[tr], y_cont[tr])
            resid = y_cont[ho] - m.predict(Xs[ho])
            rows = ho
        else:
            resid = y_cont - float(np.mean(y_cont))
            rows = slice(None)
        cand_ids = [i for i in range(len(pool)) if i not in sel_idx]
        # GATED GPU BATCH (MLFRAME_FE_GPU_USABILITY, default OFF): the per-candidate |corr(values,
        # resid)| is the n-scaling inner loop of the shortlist -- batch all candidates' columns vs the
        # one resid in ONE cupy GEMV (centered dot / sqrt(ss)) instead of a Python loop of np.corrcoef.
        # BIT-FAITHFUL to the per-candidate ``_abscorr`` (same float64 estimator + std<1e-12 guard), so
        # the shortlist ORDER is unchanged. Any cupy/device error -> the exact per-candidate CPU loop.
        uses = None
        if _GPU_USABILITY() and cand_ids:
            try:
                from ._usability_gpu import gpu_abscorr_batch
                cols = np.column_stack([_pv(i)[rows] for i in cand_ids])
                uses = gpu_abscorr_batch(cols, _f64(np.asarray(resid)))
            except Exception:
                uses = None  # fall back to the exact CPU path
        scored = []
        for k, i in enumerate(cand_ids):
            use = float(uses[k]) if uses is not None else _abscorr(pool[i].values[rows], resid)
            scored.append((i, (1.0 - w) * (pool[i].mi / mi_max) + w * use))
        scored.sort(key=lambda t: t[1], reverse=True)
        return [i for i, _ in scored[: max(1, shortlist)]]

    import math
    # a committed feature must improve a MAJORITY of folds (>=75%), not just the mean -- a noise-
    # contaminated feature lowers some folds by chance and raises others (net ~0); requiring
    # consistency across folds rejects it and stops the greedy at the genuinely useful set.
    min_improving_folds = max(1, math.ceil(0.75 * n_folds))
    selected: list[int] = []
    folds_cur = _cv_per_fold(selected)
    mae_cur = float(folds_cur.mean())
    for _ in range(min(K, len(pool))):
        cand_idx = _shortlist(selected)
        best_i, best_mean, best_folds = -1, mae_cur, folds_cur
        # regression: score the whole shortlist via the incremental bordered solve (one selected-set
        # Gram per fold, reused across candidates). classification stays on the per-candidate refit.
        # ``_USAB_FORCE_FULL_REFIT`` (test-only) bypasses the incremental solve to A/B the selection.
        import os as _os
        _force_full = bool(_os.environ.get("_USAB_FORCE_FULL_REFIT"))
        _mf_by_i = None if (classification or _force_full) else _cv_candidates_incremental(selected, cand_idx)
        for i in cand_idx:
            mf = _mf_by_i[i] if _mf_by_i is not None else _cv_per_fold([*selected, i])
            if int(np.sum(mf < folds_cur)) < min_improving_folds:
                continue  # not a consistent improvement across folds
            if float(mf.mean()) < best_mean:
                best_mean, best_i, best_folds = float(mf.mean()), i, mf
        if best_i < 0 or best_mean >= mae_cur * (1.0 - mae_improve_rel):
            break
        selected.append(best_i)
        folds_cur, mae_cur = best_folds, best_mean
    return [pool[i] for i in selected]


def select_usability_aware_features(
    X_df: "Any",
    y_cont: np.ndarray,
    base_names: Sequence[str],
    *,
    w: float = 0.7,
    K: int = 8,
    seed: int = 0,
    classification: bool = False,
    pool_kwargs: Optional[dict] = None,
    greedy_kwargs: Optional[dict] = None,
) -> list[UsableCandidate]:
    """End-to-end: build the replayable candidate pool, then run the usability greedy. Returns the
    selected ``UsableCandidate`` list (each with a replayable ``recipe`` for pair forms, ``None``
    for raw columns), in selection order. ``classification=True`` routes the greedy to the logistic /
    CV-logloss scorer (the regression CV-MAE path is the default, byte-identical)."""
    pool = build_usability_candidate_pool(X_df, y_cont, base_names, **(pool_kwargs or {}))
    return usability_greedy(pool, y_cont, w=w, K=K, seed=seed, classification=classification, **(greedy_kwargs or {}))
