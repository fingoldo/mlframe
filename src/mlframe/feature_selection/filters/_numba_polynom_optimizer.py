"""Numba-only polynom-pair FE optimizer.

Fully replaces the joblib + cma/optuna stack with one ``@njit(parallel=True)``
kernel that:

  1. ``prange`` over feature pairs -- outer parallelism via numba threads
     (no GIL, no joblib worker spawn, no pickle, no memmap).
  2. Inside one pair: random batch + elitism + Gaussian perturbation,
     fully njit, no Python decisions.
  3. Inline polyeval (4 polynomial bases: Hermite / Legendre / Chebyshev /
     Laguerre) dispatched by integer ``basis_id``.
  4. Inline binary-function dispatch (6 funcs: mul / add / sub / div /
     atan2 / log_abs_signed) by integer ``bf_id``.
  5. Inline plugin MI for the single-column case (reuses the existing
     numba-compiled ``_plugin_mi_classif_njit`` / ``_plugin_mi_regression_njit``).

Random draws (uniform sampling + Gaussian perturbation) are pre-
generated outside numba and passed as flat streams so the inner kernel
has no global RNG state to share across threads.

Public entry point: ``run_numba_kernel_search`` -- single-pair drop-in
matching the ``_run_cma_search`` / ``_run_random_batch_search`` return
shape. The full multi-pair kernel is exposed via
``optimize_all_pairs_numba_kernel`` for callers that can hand over
all pair data at once (the polynom-pair FE dispatch loop is the
target consumer).

Limitations:
  - Factory bases (RBF, Sigmoid) with Python preprocess closures NOT
    supported -- falls back to caller.
  - BLAS GEMV fastpath (B_a / B_b precomputed basis matrices) NOT
    supported -- uses inlined Horner via the existing serial @njit
    polyeval routines.
  - ``eval_pair_fn`` closures (multi-fidelity, custom warm-start
    selectors) NOT supported -- the kernel uses the full dataset for
    every evaluation. Warm-start seeds CAN be passed as an explicit
    ndarray.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
from numba import njit, prange

# Import the existing @njit polyeval routines for the 4 polynomial bases.
# Using the SERIAL versions (not _parallel) -- we already parallelise OVER
# pairs at the outer kernel layer via prange; nested parallel kernels
# inside prange bodies oversubscribe and slow down.
from .hermite_fe import (
    _hermeval_njit, _legval_njit, _chebval_njit, _lagval_njit,
)
from ._hermite_fe_mi import _plugin_mi_classif_njit, _plugin_mi_regression_njit
from .hermite_fe import _plugin_mi_classif_batch_njit, _plugin_mi_regression_batch_njit

logger = logging.getLogger(__name__)


# Integer dispatch enum for basis_id -- matches polynomial families.
BASIS_HERMITE = 0
BASIS_LEGENDRE = 1
BASIS_CHEBYSHEV = 2
BASIS_LAGUERRE = 3

_BASIS_NAME_TO_ID = {
    "hermite": BASIS_HERMITE,
    "legendre": BASIS_LEGENDRE,
    "chebyshev": BASIS_CHEBYSHEV,
    "laguerre": BASIS_LAGUERRE,
}


# Integer dispatch enum for bf_id -- matches DEFAULT_BIN_FUNCS in hermite_fe.
BF_MUL = 0
BF_ADD = 1
BF_SUB = 2
BF_DIV = 3
BF_ATAN2 = 4
BF_LOGABS = 5

_BF_NAME_TO_ID = {
    "mul": BF_MUL,
    "add": BF_ADD,
    "sub": BF_SUB,
    "div": BF_DIV,
    "atan2": BF_ATAN2,
    "logabs": BF_LOGABS,
}


@njit(cache=True, fastmath=True)
def _polyeval_dispatch_njit(basis_id: int, x: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Integer-dispatched polyeval. Calls the appropriate serial @njit kernel.

    Why serial (not _parallel): the caller runs prange over pairs at the
    outer kernel, so inner prange would oversubscribe. The serial Horner
    loop here is already cache-friendly on contiguous (n,) arrays.
    """
    if basis_id == BASIS_HERMITE:
        return np.asarray(_hermeval_njit(x, c))
    if basis_id == BASIS_LEGENDRE:
        return np.asarray(_legval_njit(x, c))
    if basis_id == BASIS_CHEBYSHEV:
        return np.asarray(_chebval_njit(x, c))
    # BASIS_LAGUERRE
    return np.asarray(_lagval_njit(x, c))


@njit(cache=True, fastmath=True)
def _bf_dispatch_njit(bf_id: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Integer-dispatched binary function. Operates element-wise.

    Mirrors the six bf entries in ``_DEFAULT_BIN_FUNCS`` (mul / add /
    sub / div / atan2 / logabs). ``div`` uses the safe ``a / (|b| + 1e-9)``
    pattern that matches ``_safe_div`` in hermite_fe.
    """
    n = a.shape[0]
    out = np.empty(n, dtype=np.float64)
    if bf_id == BF_MUL:
        for i in range(n):
            out[i] = a[i] * b[i]
    elif bf_id == BF_ADD:
        for i in range(n):
            out[i] = a[i] + b[i]
    elif bf_id == BF_SUB:
        for i in range(n):
            out[i] = a[i] - b[i]
    elif bf_id == BF_DIV:
        for i in range(n):
            denom = b[i]
            denom_abs = denom if denom >= 0.0 else -denom
            out[i] = a[i] / (denom_abs + 1e-9)
    elif bf_id == BF_ATAN2:
        for i in range(n):
            out[i] = np.arctan2(a[i], b[i])
    else:  # BF_LOGABS
        for i in range(n):
            val = a[i] * b[i]
            sign = 1.0 if val >= 0.0 else -1.0
            abs_val = val if val >= 0.0 else -val
            out[i] = sign * np.log(abs_val + 1.0)
    return out


@njit(cache=True, fastmath=True)
def _all_finite_njit(arr: np.ndarray) -> bool:
    """Sequential isfinite scan; bails on first non-finite. Faster than
    ``np.all(np.isfinite(arr))`` on the common all-finite case because
    we don't materialise the boolean intermediate."""
    n = arr.shape[0]
    for i in range(n):
        v = arr[i]
        if not (v == v and v != np.inf and v != -np.inf):
            return False
    return True


@njit(cache=True, fastmath=True)
def _eval_one_candidate_njit(
    x_a: np.ndarray, x_b: np.ndarray, y: np.ndarray,
    coef_a: np.ndarray, coef_b: np.ndarray,
    basis_id: int, bf_ids: np.ndarray,
    n_bins: int, l2_penalty: float, direction_only: bool,
    discrete_target: bool,
):
    """Evaluate one (coef_a, coef_b) candidate across all bf_ids; return
    (best_score, best_raw_mi, best_bf_id). Mirrors the inner body of
    ``_eval_coef_pair`` from ``_hermite_fe_optimise.py`` but fully njit.
    """
    # L2-normalise if direction_only -- keep coef magnitudes from
    # exploding the polyeval output range.
    if direction_only:
        a_sq = 0.0
        b_sq = 0.0
        for i in range(coef_a.shape[0]):
            a_sq += coef_a[i] * coef_a[i]
        for i in range(coef_b.shape[0]):
            b_sq += coef_b[i] * coef_b[i]
        norm = np.sqrt(a_sq + b_sq)
        if norm > 1e-12:
            for i in range(coef_a.shape[0]):
                coef_a[i] = coef_a[i] / norm
            for i in range(coef_b.shape[0]):
                coef_b[i] = coef_b[i] / norm

    h_a = _polyeval_dispatch_njit(basis_id, x_a, coef_a)
    h_b = _polyeval_dispatch_njit(basis_id, x_b, coef_b)
    if not _all_finite_njit(h_a) or not _all_finite_njit(h_b):
        return -np.inf, 0.0, -1

    penalty = 0.0
    if not direction_only and l2_penalty > 0.0:
        a_sq = 0.0
        b_sq = 0.0
        for i in range(coef_a.shape[0]):
            a_sq += coef_a[i] * coef_a[i]
        for i in range(coef_b.shape[0]):
            b_sq += coef_b[i] * coef_b[i]
        penalty = l2_penalty * (a_sq + b_sq)

    best_score = -np.inf
    best_raw = 0.0
    best_bf = -1
    for k in range(bf_ids.shape[0]):
        bf_id = bf_ids[k]
        combined = _bf_dispatch_njit(bf_id, h_a, h_b)
        if not _all_finite_njit(combined):
            continue
        if discrete_target:
            mi = _plugin_mi_classif_njit(combined, y, n_bins)
        else:
            mi = _plugin_mi_regression_njit(combined, y, n_bins)
        score = mi - penalty
        if score > best_score:
            best_score = score
            best_raw = mi
            best_bf = bf_id
    return best_score, best_raw, best_bf


@njit(cache=True, fastmath=True)
def _eval_batch_candidates_njit(
    x_a: np.ndarray, x_b: np.ndarray, y: np.ndarray,
    coefs_a_batch: np.ndarray, coefs_b_batch: np.ndarray,
    basis_id: int, bf_ids: np.ndarray,
    n_bins: int, l2_penalty: float, direction_only: bool,
    discrete_target: bool,
    # output scratch (caller-allocated to avoid per-iter alloc):
    cols_scratch: np.ndarray,  # (n, B*K)
    col_valid_scratch: np.ndarray,  # (B*K,) bool
    out_scores: np.ndarray,  # (B,)
    out_raws: np.ndarray,  # (B,)
    out_bfs: np.ndarray,  # (B,)
) -> None:
    """Evaluate B candidates in ONE batched MI call. Replaces the
    per-candidate ``_plugin_mi_classif_njit`` call pattern -- batch MI
    pranges over columns (B * K), saturating cores in a single kernel
    launch instead of B sequential numba calls each holding its own
    quantile-bin sort.

    Memory: caller supplies ``cols_scratch`` (n, B*K) so this function
    never allocates the heavy column store per iter.
    """
    B = coefs_a_batch.shape[0]
    K = bf_ids.shape[0]
    n = x_a.shape[0]
    ca_size = coefs_a_batch.shape[1]
    cb_size = coefs_b_batch.shape[1]

    # Init outputs + invalid mask.
    for b in range(B):
        out_scores[b] = -np.inf
        out_raws[b] = 0.0
        out_bfs[b] = -1
    for c in range(B * K):
        col_valid_scratch[c] = False

    # Per-candidate polyeval + bf -> store columns.
    for b in range(B):
        ca = coefs_a_batch[b].copy()
        cb = coefs_b_batch[b].copy()
        if direction_only:
            a_sq = 0.0
            b_sq = 0.0
            for i in range(ca_size):
                a_sq += ca[i] * ca[i]
            for i in range(cb_size):
                b_sq += cb[i] * cb[i]
            norm = np.sqrt(a_sq + b_sq)
            if norm > 1e-12:
                for i in range(ca_size):
                    ca[i] = ca[i] / norm
                for i in range(cb_size):
                    cb[i] = cb[i] / norm
        h_a = _polyeval_dispatch_njit(basis_id, x_a, ca)
        h_b = _polyeval_dispatch_njit(basis_id, x_b, cb)
        if not _all_finite_njit(h_a) or not _all_finite_njit(h_b):
            # Leave cols zeroed; column_valid stays False; MI naturally low.
            continue
        for k in range(K):
            combined = _bf_dispatch_njit(bf_ids[k], h_a, h_b)
            if not _all_finite_njit(combined):
                continue
            col_idx = b * K + k
            for i in range(n):
                cols_scratch[i, col_idx] = combined[i]
            col_valid_scratch[col_idx] = True

    # Zero out columns that weren't filled (so MI on them == 0 from
    # zero-variance constant, gets filtered by argmax).
    for c in range(B * K):
        if not col_valid_scratch[c]:
            for i in range(n):
                cols_scratch[i, c] = 0.0

    # ONE batched MI call. _plugin_mi_classif_batch_njit pranges over
    # columns; with B*K typically 100-200 this saturates all cores.
    if discrete_target:
        mi_arr = _plugin_mi_classif_batch_njit(cols_scratch, y, n_bins)
    else:
        mi_arr = _plugin_mi_regression_batch_njit(cols_scratch, y, n_bins)

    # Per-candidate argmax over K bfs.
    for b in range(B):
        penalty = 0.0
        if not direction_only and l2_penalty > 0.0:
            a_sq = 0.0
            b_sq = 0.0
            for i in range(ca_size):
                a_sq += coefs_a_batch[b, i] * coefs_a_batch[b, i]
            for i in range(cb_size):
                b_sq += coefs_b_batch[b, i] * coefs_b_batch[b, i]
            penalty = l2_penalty * (a_sq + b_sq)
        for k in range(K):
            col_idx = b * K + k
            if not col_valid_scratch[col_idx]:
                continue
            raw = mi_arr[col_idx]
            score = raw - penalty
            if score > out_scores[b]:
                out_scores[b] = score
                out_raws[b] = raw
                out_bfs[b] = bf_ids[k]


@njit(cache=True, fastmath=True)
def _random_batch_one_pair_njit(
    x_a: np.ndarray, x_b: np.ndarray, y: np.ndarray,
    ca_size: int, cb_size: int,
    coef_lo: float, coef_hi: float,
    n_iters: int, batch_size: int, elitism_k: int, sigma_perturb: float,
    basis_id: int, bf_ids: np.ndarray,
    n_bins: int, l2_penalty: float, direction_only: bool,
    discrete_target: bool,
    uniform_stream: np.ndarray, normal_stream: np.ndarray,
    warm_start_a: np.ndarray, warm_start_b: np.ndarray, n_warm: int,
):
    """Run random-batch + elitism search for ONE pair using pre-generated
    RNG streams. Returns (best_coef_a, best_coef_b, best_score, best_raw,
    best_bf, n_evals).

    Parameters
    ----------
    uniform_stream : (n_iters * batch_size * (ca_size + cb_size),) float64
        Pre-generated U[coef_lo, coef_hi] draws. Consumed by cursor.
    normal_stream : (n_iters * elitism_k * (ca_size + cb_size),) float64
        Pre-generated N(0, sigma_perturb) draws for elitism perturbation.
    warm_start_a, warm_start_b : (W, ca_size) / (W, cb_size)
        Optional warm-start coefficients evaluated before random search.
    n_warm : int
        How many warm-start rows to use (W >= n_warm). 0 = skip warm.
    """
    best_score = -np.inf
    best_raw = 0.0
    best_bf = -1
    best_ca = np.zeros(ca_size, dtype=np.float64)
    best_cb = np.zeros(cb_size, dtype=np.float64)
    has_best = False
    n_evals = 0

    # Warm-start phase: evaluate the provided seeds first.
    for w in range(n_warm):
        ca = warm_start_a[w].copy()
        cb = warm_start_b[w].copy()
        score, raw, bf = _eval_one_candidate_njit(
            x_a, x_b, y, ca, cb, basis_id, bf_ids,
            n_bins, l2_penalty, direction_only, discrete_target,
        )
        n_evals += 1
        if score > best_score:
            best_score = score
            best_raw = raw
            best_bf = bf
            for j in range(ca_size):
                best_ca[j] = ca[j]
            for j in range(cb_size):
                best_cb[j] = cb[j]
            has_best = True

    # Scratch buffers for the candidate batch + batched-eval workspace.
    coefs_a_batch = np.empty((batch_size, ca_size), dtype=np.float64)
    coefs_b_batch = np.empty((batch_size, cb_size), dtype=np.float64)
    K = bf_ids.shape[0]
    n_rows = x_a.shape[0]
    cols_scratch = np.empty((n_rows, batch_size * K), dtype=np.float64)
    col_valid_scratch = np.empty(batch_size * K, dtype=np.bool_)
    batch_scores = np.empty(batch_size, dtype=np.float64)
    batch_raws = np.empty(batch_size, dtype=np.float64)
    batch_bfs = np.empty(batch_size, dtype=np.int64)

    u_cursor = 0
    n_cursor = 0
    dim = ca_size + cb_size

    for _it in range(n_iters):
        # Phase 1: sample batch_size candidates uniformly from coef_range.
        for i in range(batch_size):
            for j in range(ca_size):
                coefs_a_batch[i, j] = uniform_stream[u_cursor]
                u_cursor += 1
            for j in range(cb_size):
                coefs_b_batch[i, j] = uniform_stream[u_cursor]
                u_cursor += 1

        # Phase 2: elitism -- replace first elitism_k slots with
        # Gaussian perturbations of the current best (after iter 0).
        if has_best:
            k_eff = elitism_k if elitism_k < batch_size else batch_size
            for i in range(k_eff):
                for j in range(ca_size):
                    v = best_ca[j] + normal_stream[n_cursor]
                    n_cursor += 1
                    if v < coef_lo:
                        v = coef_lo
                    elif v > coef_hi:
                        v = coef_hi
                    coefs_a_batch[i, j] = v
                for j in range(cb_size):
                    v = best_cb[j] + normal_stream[n_cursor]
                    n_cursor += 1
                    if v < coef_lo:
                        v = coef_lo
                    elif v > coef_hi:
                        v = coef_hi
                    coefs_b_batch[i, j] = v
        else:
            # No best yet -- skip normal_stream cursor advance to keep
            # the per-iter consumption deterministic across pairs.
            n_cursor += elitism_k * dim

        # Phase 3: batched eval -- ONE MI batch call across all
        # (batch_size * K) columns. _plugin_mi_classif_batch_njit pranges
        # over columns, saturating cores in a single kernel launch.
        _eval_batch_candidates_njit(
            x_a, x_b, y, coefs_a_batch, coefs_b_batch,
            basis_id, bf_ids, n_bins, l2_penalty, direction_only,
            discrete_target,
            cols_scratch, col_valid_scratch,
            batch_scores, batch_raws, batch_bfs,
        )
        n_evals += batch_size

        # Phase 4: track per-iter best across the batch.
        for i in range(batch_size):
            if batch_scores[i] > best_score:
                best_score = batch_scores[i]
                best_raw = batch_raws[i]
                best_bf = batch_bfs[i]
                for j in range(ca_size):
                    best_ca[j] = coefs_a_batch[i, j]
                for j in range(cb_size):
                    best_cb[j] = coefs_b_batch[i, j]
                has_best = True

    return best_ca, best_cb, best_score, best_raw, best_bf, n_evals


@njit(parallel=True, cache=True, fastmath=True)
def _optimize_all_pairs_kernel(
    X_arr: np.ndarray, y: np.ndarray, pair_indices: np.ndarray,
    ca_size: int, cb_size: int,
    coef_lo: float, coef_hi: float,
    n_iters: int, batch_size: int, elitism_k: int, sigma_perturb: float,
    basis_id: int, bf_ids: np.ndarray,
    n_bins: int, l2_penalty: float, direction_only: bool,
    discrete_target: bool,
    uniform_streams: np.ndarray, normal_streams: np.ndarray,
    warm_start_a: np.ndarray, warm_start_b: np.ndarray, n_warm: int,
    # outputs:
    out_best_ca: np.ndarray, out_best_cb: np.ndarray,
    out_best_score: np.ndarray, out_best_raw: np.ndarray,
    out_best_bf: np.ndarray, out_n_evals: np.ndarray,
):
    """Multi-pair outer kernel: prange over pairs, each thread runs an
    independent random-batch search on its pair.

    All pair-shared inputs (X_arr, y, warm_start_*) are read-only;
    per-pair outputs are written to disjoint rows so there's no false
    sharing between threads. ``uniform_streams[p]`` / ``normal_streams[p]``
    give pair ``p`` its own RNG stream (pre-generated outside numba)
    so the kernel has no global RNG state to race on.
    """
    P = pair_indices.shape[0]
    for p in prange(P):
        col_a = pair_indices[p, 0]
        col_b = pair_indices[p, 1]
        # X_arr is (N, F) -- slice the two columns.
        n_rows = X_arr.shape[0]
        x_a = np.empty(n_rows, dtype=np.float64)
        x_b = np.empty(n_rows, dtype=np.float64)
        for i in range(n_rows):
            x_a[i] = X_arr[i, col_a]
            x_b[i] = X_arr[i, col_b]

        best_ca, best_cb, best_score, best_raw, best_bf, n_evals = _random_batch_one_pair_njit(
            x_a, x_b, y, ca_size, cb_size,
            coef_lo, coef_hi,
            n_iters, batch_size, elitism_k, sigma_perturb,
            basis_id, bf_ids,
            n_bins, l2_penalty, direction_only, discrete_target,
            uniform_streams[p], normal_streams[p],
            warm_start_a, warm_start_b, n_warm,
        )

        for j in range(ca_size):
            out_best_ca[p, j] = best_ca[j]
        for j in range(cb_size):
            out_best_cb[p, j] = best_cb[j]
        out_best_score[p] = best_score
        out_best_raw[p] = best_raw
        out_best_bf[p] = best_bf
        out_n_evals[p] = n_evals


def _basis_name_to_id(basis: str) -> int:
    name = basis.lower()
    if name not in _BASIS_NAME_TO_ID:
        raise ValueError(
            f"basis={basis!r} not supported by numba_kernel; expected one of "
            f"{list(_BASIS_NAME_TO_ID.keys())}. Factory bases (RBF, Sigmoid) "
            f"are not yet supported -- use the optuna / cma / cma_batch path."
        )
    return _BASIS_NAME_TO_ID[name]


def _bf_names_to_ids(bf_names: Sequence[str]) -> np.ndarray:
    """Translate bf_callable name strings to the integer ids the kernel
    dispatches on. Unknown names raise -- the caller is expected to pass
    the same names the existing ``_DEFAULT_BIN_FUNCS`` dict uses."""
    out = np.empty(len(bf_names), dtype=np.int64)
    for i, name in enumerate(bf_names):
        if name not in _BF_NAME_TO_ID:
            raise ValueError(f"bf_name={name!r} not supported by numba_kernel; expected one of " f"{list(_BF_NAME_TO_ID.keys())}.")
        out[i] = _BF_NAME_TO_ID[name]
    return out


def run_numba_kernel_search(*, ca_size: int, cb_size: int, coef_range: tuple, n_trials: int, seed: int,
                              direction_only: bool, warm_start_seeds: Optional[Sequence[np.ndarray]], eval_kwargs: dict,
                              batch_size: int = 20, elitism_k: int = 4,
                              perturb_sigma_frac: float = 0.1) -> Optional[tuple]:
    """Single-pair entry point matching the ``_run_cma_search`` /
    ``_run_random_batch_search`` return contract so the dispatcher can
    swap it in via ``optimizer="numba_kernel"``.

    Internally calls the multi-pair kernel with P=1, so the JIT cache
    is shared with the multi-pair path.
    """
    # Resolve basis + bf_names from eval_kwargs. The caller passes the
    # same eval_kwargs dict it would pass to _eval_coef_pair.
    eval_func = eval_kwargs["eval_func"]
    # Discover basis from the eval_func name -- the existing _NJIT_FUNCS
    # dispatch in hermite_fe maps function-name suffixes to basis names.
    fn_name = getattr(eval_func, "__name__", "")
    if "hermeval" in fn_name:
        basis = "hermite"
    elif "legval" in fn_name:
        basis = "legendre"
    elif "chebval" in fn_name:
        basis = "chebyshev"
    elif "lagval" in fn_name:
        basis = "laguerre"
    else:
        raise ValueError(f"Could not infer basis from eval_func={fn_name!r}; " f"numba_kernel supports polynomial bases only (no factory bases).")
    basis_id = _basis_name_to_id(basis)
    bf_ids = _bf_names_to_ids(eval_kwargs["bf_names"])

    # Build x_a / x_b from z_a / z_b (already standardised) -- the
    # njit polyeval operates on z directly.
    x_a = np.ascontiguousarray(eval_kwargs["z_a"], dtype=np.float64)
    x_b = np.ascontiguousarray(eval_kwargs["z_b"], dtype=np.float64)
    discrete_target = bool(eval_kwargs["discrete_target"])
    y_njit = eval_kwargs["y_njit"]
    if y_njit is None:
        # KSG path not supported here; require plugin MI.
        raise ValueError("numba_kernel requires mi_estimator='plugin'; y_njit was None " "(ksg path).")
    y_arr = np.ascontiguousarray(y_njit, dtype=np.int64) if discrete_target else np.ascontiguousarray(y_njit, dtype=np.float64)
    n_bins = int(eval_kwargs["plugin_n_bins"])
    l2_penalty = float(eval_kwargs["l2_penalty"]) if not direction_only else 0.0

    coef_lo, coef_hi = float(coef_range[0]), float(coef_range[1])
    sigma_perturb = perturb_sigma_frac * (coef_hi - coef_lo)
    n_iters = max(1, int(np.ceil(n_trials / max(1, batch_size))))

    # Pre-generate RNG streams. Per-iter cost: ~3-4 KB per pair -- trivial.
    rng = np.random.default_rng(seed if seed > 0 else 1)
    u_per_iter = batch_size * (ca_size + cb_size)
    n_per_iter = elitism_k * (ca_size + cb_size)
    uniform_stream = rng.uniform(coef_lo, coef_hi, size=u_per_iter * n_iters)
    normal_stream = rng.normal(0.0, sigma_perturb, size=n_per_iter * n_iters)

    # Warm-start seeds -- pad / truncate to ca_size / cb_size.
    if warm_start_seeds:
        n_warm = len(warm_start_seeds)
        warm_a = np.zeros((n_warm, ca_size), dtype=np.float64)
        warm_b = np.zeros((n_warm, cb_size), dtype=np.float64)
        for i, s in enumerate(warm_start_seeds):
            s_arr = np.asarray(s, dtype=np.float64)
            warm_a[i, : min(ca_size, s_arr.shape[0])] = s_arr[: min(ca_size, s_arr.shape[0])]
            if s_arr.shape[0] > ca_size:
                warm_b[i, : min(cb_size, s_arr.shape[0] - ca_size)] = s_arr[ca_size : ca_size + cb_size]
    else:
        n_warm = 0
        warm_a = np.zeros((1, ca_size), dtype=np.float64)
        warm_b = np.zeros((1, cb_size), dtype=np.float64)

    # Single-pair wrapper around the multi-pair kernel: P=1, X = [x_a, x_b]
    # column-stacked.
    n_rows = x_a.shape[0]
    X_arr = np.empty((n_rows, 2), dtype=np.float64)
    X_arr[:, 0] = x_a
    X_arr[:, 1] = x_b
    pair_indices = np.array([[0, 1]], dtype=np.int64)
    # Streams need a (P,) outer dim for the kernel; reshape.
    uniform_streams = uniform_stream.reshape(1, -1)
    normal_streams = normal_stream.reshape(1, -1)

    out_best_ca = np.zeros((1, ca_size), dtype=np.float64)
    out_best_cb = np.zeros((1, cb_size), dtype=np.float64)
    out_best_score = np.full(1, -np.inf, dtype=np.float64)
    out_best_raw = np.zeros(1, dtype=np.float64)
    out_best_bf = np.full(1, -1, dtype=np.int64)
    out_n_evals = np.zeros(1, dtype=np.int64)

    _optimize_all_pairs_kernel(
        X_arr, y_arr, pair_indices,
        ca_size, cb_size,
        coef_lo, coef_hi,
        n_iters, batch_size, elitism_k, sigma_perturb,
        basis_id, bf_ids,
        n_bins, l2_penalty, direction_only,
        discrete_target,
        uniform_streams, normal_streams,
        warm_a, warm_b, n_warm,
        out_best_ca, out_best_cb,
        out_best_score, out_best_raw,
        out_best_bf, out_n_evals,
    )

    if out_best_score[0] == -np.inf:
        return None
    return (
        out_best_ca[0].copy(),
        out_best_cb[0].copy(),
        int(out_best_bf[0]),
        float(out_best_raw[0]),
        int(out_n_evals[0]),
    )


def optimize_all_pairs_numba_kernel(
    X_arr: np.ndarray, y: np.ndarray, pair_indices: np.ndarray,
    *,
    ca_size: int, cb_size: int,
    coef_range: tuple, basis: str = "hermite",
    bf_names: Sequence[str] = ("mul", "add", "sub", "div"),
    n_trials: int = 200, batch_size: int = 20,
    elitism_k: int = 4, perturb_sigma_frac: float = 0.1,
    n_bins: int = 20, l2_penalty: float = 0.05,
    direction_only: bool = False, discrete_target: bool = True,
    seed: int = 0,
    warm_start_seeds: Optional[Sequence[np.ndarray]] = None,
) -> dict:
    """Multi-pair entry point: process ALL feature pairs in one kernel
    call via prange parallelism. Returns ``dict`` per pair so the
    caller can match against ``raw_vars_pair`` keys.

    This is the path the polynom-pair FE dispatch should call when
    ``fe_optimizer == "numba_kernel"`` to fully eliminate joblib.
    """
    P = pair_indices.shape[0]
    basis_id = _basis_name_to_id(basis)
    bf_ids = _bf_names_to_ids(bf_names)
    coef_lo, coef_hi = float(coef_range[0]), float(coef_range[1])
    sigma_perturb = perturb_sigma_frac * (coef_hi - coef_lo)
    n_iters = max(1, int(np.ceil(n_trials / max(1, batch_size))))

    rng = np.random.default_rng(seed if seed > 0 else 1)
    u_per_iter = batch_size * (ca_size + cb_size)
    n_per_iter = elitism_k * (ca_size + cb_size)
    uniform_streams = rng.uniform(
        coef_lo, coef_hi,
        size=(P, u_per_iter * n_iters),
    )
    normal_streams = rng.normal(
        0.0, sigma_perturb,
        size=(P, n_per_iter * n_iters),
    )

    if warm_start_seeds:
        n_warm = len(warm_start_seeds)
        warm_a = np.zeros((n_warm, ca_size), dtype=np.float64)
        warm_b = np.zeros((n_warm, cb_size), dtype=np.float64)
        for i, s in enumerate(warm_start_seeds):
            s_arr = np.asarray(s, dtype=np.float64)
            warm_a[i, : min(ca_size, s_arr.shape[0])] = s_arr[: min(ca_size, s_arr.shape[0])]
            if s_arr.shape[0] > ca_size:
                warm_b[i, : min(cb_size, s_arr.shape[0] - ca_size)] = s_arr[ca_size : ca_size + cb_size]
    else:
        n_warm = 0
        warm_a = np.zeros((1, ca_size), dtype=np.float64)
        warm_b = np.zeros((1, cb_size), dtype=np.float64)

    out_best_ca = np.zeros((P, ca_size), dtype=np.float64)
    out_best_cb = np.zeros((P, cb_size), dtype=np.float64)
    out_best_score = np.full(P, -np.inf, dtype=np.float64)
    out_best_raw = np.zeros(P, dtype=np.float64)
    out_best_bf = np.full(P, -1, dtype=np.int64)
    out_n_evals = np.zeros(P, dtype=np.int64)

    X_contig = np.ascontiguousarray(X_arr, dtype=np.float64)
    y_arr = np.ascontiguousarray(y, dtype=np.int64) if discrete_target else np.ascontiguousarray(y, dtype=np.float64)
    pair_indices_arr = np.ascontiguousarray(pair_indices, dtype=np.int64)

    _optimize_all_pairs_kernel(
        X_contig, y_arr, pair_indices_arr,
        ca_size, cb_size,
        coef_lo, coef_hi,
        n_iters, batch_size, elitism_k, sigma_perturb,
        basis_id, bf_ids,
        n_bins, l2_penalty, direction_only,
        discrete_target,
        uniform_streams, normal_streams,
        warm_a, warm_b, n_warm,
        out_best_ca, out_best_cb,
        out_best_score, out_best_raw,
        out_best_bf, out_n_evals,
    )

    return {
        "best_coefs_a": out_best_ca,
        "best_coefs_b": out_best_cb,
        "best_scores": out_best_score,
        "best_raws": out_best_raw,
        "best_bfs": out_best_bf,
        "n_evals": out_n_evals,
    }
