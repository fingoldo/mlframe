"""CuPy (GPU-resident) polynom-pair FE optimizer -- the device twin of ``_numba_polynom_optimizer``.

One generation of P coefficient candidates is scored as TWO GEMMs plus batched elementwise/binning/MI
device work, instead of P independent host evaluations:

  1. Basis matrices ``B_a (n, D+1)`` / ``B_b`` are built ONCE per (pair, degree) on the device via the
     exact numpy.polynomial recurrences (He/P/T/L -- parity with the ``_*val_njit`` kernels).
  2. ``H_A = B_a @ C_a^T`` -> (n, P) for the whole generation in one cuBLAS GEMM (same trick as the
     host BLAS fastpath, but for all candidates at once).
  3. The six binary functions are cupy elementwise ops at BIT-PARITY with ``_DEFAULT_BIN_FUNCS``
     (incl. the sign-preserving ``_safe_div`` and the sum-of-factor-logs ``_log_abs_signed``).
  4. Rank-based equi-frequency binning of every combined column in ONE batched ``cp.argsort(axis=0)``
     (the same block-size rule as ``_quantile_bin_njit``), then plug-in MI for every column in one
     fused launch via the existing ``_fe_batched_mi.binned_mi_from_codes_gpu``.
  5. The saturating L2 penalty (``lambda * s / (s + 1)``) matches ``_l2_penalty_value``'s default.

Search structure mirrors the numba kernel: random init (+ warm seeds), elitism, Gaussian perturbation;
all random draws come from a HOST ``np.random.default_rng(seed)`` so runs are deterministic and
seed-comparable across backends regardless of device.

CORRECTNESS BAR: selection-equivalence, not bit-identity -- ``cp.argsort`` breaks ties differently from
the host quicksort ``np.argsort``, so tied feature values may bin differently (ties in a GEMM-produced
continuous feature are measure-zero in practice; the e2e regression pins winner recovery).

DEFAULTS (batch_size=100, elitism_k=10, sigma_frac=0.1) come from the 2026-07-15 sweep
(scratch harness, cases ratio_regime + cubic_inner, 3 seeds x 3 restarts): bs=100 lifted cubic_inner
0.4901 -> 0.5834 at 39.6s (vs bs=20's 8-10s and bs=300's 138s/0.5267 -- the sweet spot), and the
speed->restarts conversion closes the ratio_regime gap entirely (bs=100 x 15 restarts finds it 3/3
seeds at 0.465-0.487 in 102.9s vs cma_batch's 4/5 at 492.5s full-budget wall).

Public entry point: ``run_cupy_kernel_search`` -- same return contract as ``run_numba_kernel_search`` /
``_run_cma_search_batch`` so the pair-optimiser dispatch swaps it in via ``optimizer="cupy_kernel"``.
Limitations (same set as the numba kernel): plugin MI only, the 4 polynomial bases only, no
multi-fidelity closures.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

_BASIS_IDS = {"hermite": 0, "legendre": 1, "chebyshev": 2, "laguerre": 3}
_BF_ORDER = ("mul", "add", "sub", "div", "atan2", "logabs")


def _basis_matrix_gpu(cp, x, degree: int, basis_id: int):
    """(n, degree+1) device basis matrix via the exact numpy.polynomial recurrences.

    float32 (2026-07-20, GPU-residency pass): built and consumed in float32, not float64 -- the
    GEMM output feeds straight into rank-based equi-frequency binning (a small fixed bin count),
    which absorbs float32-vs-float64 rounding noise long before it could move a bin assignment.
    Verified (see ``_benchmarks/bench_cupy_polynom_f32_gemm.py``) bit-identical winning MI across
    multiple seeds on the module's own cubic-inner regression scenario when BOTH this matrix and
    the coefficient population upload are float32 -- cupy promotes float32 @ float64 back to
    float64 silently, so half-measures (only one side float32) would give zero benefit."""
    n = x.shape[0]
    B = cp.empty((n, degree + 1), dtype=cp.float32)
    B[:, 0] = 1.0
    if degree >= 1:
        B[:, 1] = x
    for k in range(2, degree + 1):
        if basis_id == 0:  # hermite_e: He_k = x*He_{k-1} - (k-1)*He_{k-2}
            B[:, k] = x * B[:, k - 1] - (k - 1) * B[:, k - 2]
        elif basis_id == 1:  # legendre: k*P_k = (2k-1)*x*P_{k-1} - (k-1)*P_{k-2}
            B[:, k] = ((2 * k - 1) * x * B[:, k - 1] - (k - 1) * B[:, k - 2]) / k
        elif basis_id == 2:  # chebyshev: T_k = 2x*T_{k-1} - T_{k-2}
            B[:, k] = 2.0 * x * B[:, k - 1] - B[:, k - 2]
        else:  # laguerre: k*L_k = (2k-1-x)*L_{k-1} - (k-1)*L_{k-2}
            B[:, k] = ((2 * k - 1 - x) * B[:, k - 1] - (k - 1) * B[:, k - 2]) / k
    return B


def _bf_apply_gpu(cp, bf_name: str, A, Bm):
    """Elementwise binary function on (n, P) device matrices, bit-parity with _DEFAULT_BIN_FUNCS."""
    if bf_name == "mul":
        return A * Bm
    if bf_name == "add":
        return A + Bm
    if bf_name == "sub":
        return A - Bm
    if bf_name == "div":  # _safe_div: exact sign-preserving divide, eps only at exact zero
        return A / cp.where(Bm == 0.0, 1e-9, Bm)
    if bf_name == "atan2":
        return cp.arctan2(A, Bm)
    # logabs: sign(a*b + eps) * (log(|a|+eps) + log(|b|+eps))
    eps = 1e-9
    return cp.sign(A * Bm + eps) * (cp.log(cp.abs(A) + eps) + cp.log(cp.abs(Bm) + eps))


def _rank_bin_batched_gpu(cp, M, n_bins: int):
    """Column-wise equi-frequency codes for (n, P) M via the sort-free radix-edge binner.

    nsys on the production-shaped search (n=99401, P=100): the original argsort-based rank binning was
    ~80%% of ALL kernel GPU time (52.4%% cub sort at 5040 x 4.2ms + 16.1%% put_along_axis scatter + 8.1%%
    more sort segments), while the MI itself was 4.5%%. ``batched_quantile_bin_gpu`` computes the SAME
    equi-frequency partition from radix-selected value edges + a vectorized coder (~nbins+3 fused
    elementwise launches, no sort at all). Value-edge vs rank coding differs only on TIED feature values
    (ties get one bin instead of an arbitrary rank split) -- measure-zero on continuous GEMM outputs and
    arguably more correct on genuine ties; the e2e winner-recovery regressions pin the quality bar."""
    from ._fe_batched_mi import batched_quantile_bin_gpu

    return batched_quantile_bin_gpu(M, n_bins)


_ALL_FINITE_AXIS0_KERNEL = None


def _get_all_finite_axis0_kernel(cp):
    """Lazy-compiled ``cp.ReductionKernel`` fusing ``cp.isfinite(x).all(axis=0)`` into ONE kernel launch
    (nsys, 2026-07-15: the two-kernel form -- elementwise isfinite materializing a full boolean array, then
    a separate .all() reduce -- cost cupy_isfinite 5.9%% + cupy_all 10%% of GPU time in the cupy-search
    microbench, ~480 launches each). Bit-identical (verified incl. NaN columns): 'isfinite(x)' as the map
    expression, '&&' as the reduce, avoids ever materializing the intermediate boolean array. ~7x faster
    isolated at (20, 20000)."""
    global _ALL_FINITE_AXIS0_KERNEL
    if _ALL_FINITE_AXIS0_KERNEL is None:
        _ALL_FINITE_AXIS0_KERNEL = cp.ReductionKernel(
            "T x", "bool y", "isfinite(x)", "a && b", "y = a", "true", "all_finite_axis0",
        )
    return _ALL_FINITE_AXIS0_KERNEL


def _score_generation_gpu(cp, Ba, Bb, Ca, Cb, y_codes_dev, ky: int, n_bins: int, l2_penalty: float, direction_only: bool, bf_names: Sequence[str]):
    """Score P candidates across all binary funcs; returns host (score, raw_mi, bf_idx) arrays of len P.

    GPU-RESIDENCY NOTE (2026-07-20): everything inside this function -- the finite masks, the per-bf MI,
    the running best-score/raw/bf accumulators -- stays a cupy device array through the WHOLE bf_names
    loop; the single ``cp.asnumpy`` call at the very end is the ONLY device->host sync per generation.
    Pre-fix this was up to ``2 + 2*len(bf_names)`` separate syncs (one per finite-mask check and one per
    ``binned_mi_from_codes_gpu`` call, which itself unconditionally synced) -- nsys on the wellbore
    100k GPU-strict trace showed cudaStreamSynchronize/cudaMemcpyAsync call counts consistent with this
    exact multiplicity. Selection logic (upd/where) is bit-identical, just evaluated on-device."""
    from ._fe_batched_mi import binned_mi_from_codes_gpu

    all_finite0 = _get_all_finite_axis0_kernel(cp)

    P = Ca.shape[0]
    if direction_only:
        norm = cp.sqrt((Ca * Ca).sum(axis=1) + (Cb * Cb).sum(axis=1))
        norm = cp.where(norm > 1e-12, norm, 1.0)
        Ca = Ca / norm[:, None]
        Cb = Cb / norm[:, None]
        penalty = cp.zeros(P, dtype=cp.float64)
    elif l2_penalty > 0.0:
        s_norm = (Ca * Ca).sum(axis=1) + (Cb * Cb).sum(axis=1)
        penalty = l2_penalty * s_norm / (s_norm + 1.0)  # saturating parity with _l2_penalty_value
    else:
        penalty = cp.zeros(P, dtype=cp.float64)

    HA = Ba @ Ca.T  # (n, P)
    HB = Bb @ Cb.T
    col_finite = all_finite0(HA, axis=0) & all_finite0(HB, axis=0)

    best_score = cp.full(P, -cp.inf)
    best_raw = cp.zeros(P)
    best_bf = cp.full(P, -1, dtype=cp.int64)
    for k, bf in enumerate(bf_names):
        C = _bf_apply_gpu(cp, bf, HA, HB)
        finite = col_finite & all_finite0(C, axis=0)
        codes = _rank_bin_batched_gpu(cp, C, n_bins)
        # kx_per_col=n_bins (codes are always in [0, n_bins) by construction of the quantile binner) skips
        # binned_mi_from_codes_gpu's int(C.max())+1 fallback -- a BLOCKING device sync that nsys showed
        # costing more than the MI kernel itself (cupy_max ~9.1% of GPU time, one sync every generation).
        mi = binned_mi_from_codes_gpu(codes, y_codes_dev, kx_per_col=n_bins, ky=ky, codes_trusted=True, as_device=True)
        score = mi - penalty
        upd = finite & (score > best_score)
        best_score = cp.where(upd, score, best_score)
        best_raw = cp.where(upd, mi, best_raw)
        best_bf = cp.where(upd, k, best_bf)
    return cp.asnumpy(best_score), cp.asnumpy(best_raw), cp.asnumpy(best_bf)


def run_cupy_kernel_search(*, ca_size: int, cb_size: int, coef_range: tuple, n_trials: int, seed: int,
                           direction_only: bool, warm_start_seeds: Optional[Sequence[np.ndarray]],
                           eval_kwargs: dict, batch_size: int = 100, elitism_k: int = 10,
                           perturb_sigma_frac: float = 0.1) -> Optional[tuple]:
    """GPU generation-batched random+elitism search.
    Returns ``(coef_a_best, coef_b_best, bf_idx_best, raw_mi_best, best_score)`` or ``None``.

    GPU_INFRA_D-6 fix: the 5th element is NOT the same contract as
    ``run_numba_kernel_search``'s 5th element -- this backend returns ``best_score`` (``mi - penalty``),
    the numba twin returns an evaluation *count* (``int(out_n_evals[0])``). Currently benign because the
    sole caller (``_hermite_fe_optimise_pair.py``) discards element 5 for every optimizer branch, but do
    not rely on element 5 meaning the same thing across backends without fixing this divergence first."""
    import cupy as cp

    fn_name = getattr(eval_kwargs.get("eval_func"), "__name__", "")
    basis = next((b for b in _BASIS_IDS if b[:4] in fn_name or b in fn_name), None)
    if basis is None:
        for key, frag in (("hermite", "herm"), ("legendre", "leg"), ("chebyshev", "cheb"), ("laguerre", "lag")):
            if frag in fn_name:
                basis = key
                break
    if basis is None:
        raise ValueError(f"cannot infer basis from eval_func={fn_name!r}")
    bf_names = list(eval_kwargs["bf_names"])
    if any(b not in _BF_ORDER for b in bf_names):
        raise ValueError(f"unsupported bf in {bf_names}; cupy kernel supports {_BF_ORDER}")
    if eval_kwargs.get("mi_estimator", "plugin") != "plugin" or not bool(eval_kwargs["discrete_target"]):
        raise ValueError("cupy kernel supports plugin MI on a discrete target only")

    # float32 (2026-07-20): see _basis_matrix_gpu's docstring -- both the operand upload and the
    # basis matrix it feeds must be float32 together, or cupy silently promotes the GEMM back to
    # float64 and this is a no-op.
    z_a = np.ascontiguousarray(eval_kwargs["z_a"], dtype=np.float32)
    z_b = np.ascontiguousarray(eval_kwargs["z_b"], dtype=np.float32)
    y = np.asarray(eval_kwargs["y_njit"]).astype(np.int64).ravel()
    y_min = int(y.min()) if y.size else 0
    y_codes = y - y_min  # dense 0-based labels for the fused-MI shared tile
    ky = int(y_codes.max()) + 1 if y.size else 1
    n_bins = int(eval_kwargs["plugin_n_bins"])
    l2_penalty = float(eval_kwargs["l2_penalty"]) if not direction_only else 0.0

    xa_d = cp.asarray(z_a)
    xb_d = cp.asarray(z_b)
    y_d = cp.asarray(y_codes)
    Ba = _basis_matrix_gpu(cp, xa_d, ca_size - 1, _BASIS_IDS[basis])
    Bb = _basis_matrix_gpu(cp, xb_d, cb_size - 1, _BASIS_IDS[basis])

    rng = np.random.default_rng(seed)
    lo, hi = float(coef_range[0]), float(coef_range[1])
    dim = ca_size + cb_size

    pop = rng.uniform(lo, hi, size=(batch_size, dim))
    if warm_start_seeds:
        for i, s in enumerate(warm_start_seeds[: min(len(warm_start_seeds), batch_size)]):
            sv = np.asarray(s, dtype=np.float64).ravel()
            if sv.size == dim:
                pop[i] = np.clip(sv, lo, hi)

    best_vec = None
    best_score = -np.inf
    best_raw = 0.0
    best_bf = -1
    sigma = perturb_sigma_frac * (hi - lo)
    n_gens = max(1, n_trials // batch_size)
    for _gen in range(n_gens):
        Ca = cp.asarray(np.ascontiguousarray(pop[:, :ca_size], dtype=np.float32))
        Cb = cp.asarray(np.ascontiguousarray(pop[:, ca_size:], dtype=np.float32))
        scores, raws, bfs = _score_generation_gpu(
            cp, Ba, Bb, Ca, Cb, y_d, ky, n_bins, l2_penalty, direction_only, bf_names,
        )
        gi = int(np.argmax(scores))
        if scores[gi] > best_score:
            best_score = float(scores[gi])
            best_raw = float(raws[gi])
            best_bf = int(bfs[gi])
            best_vec = pop[gi].copy()
        order = np.argsort(scores)[::-1]
        elites = pop[order[:elitism_k]]
        n_perturb = batch_size - elitism_k - max(1, batch_size // 4)
        perturbed = elites[rng.integers(0, elitism_k, size=n_perturb)] + rng.normal(0.0, sigma, size=(n_perturb, dim))
        fresh = rng.uniform(lo, hi, size=(max(1, batch_size // 4), dim))
        pop = np.clip(np.vstack([elites, perturbed, fresh]), lo, hi)[:batch_size]

    if best_vec is None or not np.isfinite(best_score):
        return None
    return (best_vec[:ca_size].copy(), best_vec[ca_size:].copy(), best_bf, best_raw, best_score)
