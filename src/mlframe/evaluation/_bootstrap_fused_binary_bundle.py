"""Fully-njit prange-parallel batch bootstrap for the roc_auc/brier/log_loss/ece bundle
``honest_diagnostics._bootstrap_block`` computes together over ONE shared resample.

``bootstrap_metrics`` (``evaluation/bootstrap.py``) already shares one resample loop across
these four metrics, but each resample still pays a Python-level round-trip per njit kernel
call -- the GIL-bound cost its own docstring flags as the real bottleneck (see
``bootstrap_metrics``'s "MEASURED (2026-07, n=200k/R=1000)" note). This module fuses all four
metrics' per-resample computation into ONE ``numba.njit(parallel=True)`` call: the index
matrix is materialised in Python (matching ``bootstrap_metrics``'s exact RNG draw order for
bit-identical resamples), then every resample's brier/log_loss/ece/auc is computed truly
concurrently across OS threads with zero per-resample Python overhead.

Handles both tie-free and tied base scores (the grouped/distinct-score-group AUC counting
``make_bootstrap_auc_resampler`` uses for its own tied fallback, batched here too) -- real
predicted probabilities routinely repeat at scale (quantised tree-model leaf outputs, float32
rounding), so a tie-free-only gate would silently defeat this fusion on exactly the large runs
it targets most. :func:`bootstrap_auc_brier_ll_ece_batch` still returns ``None`` for a
degenerate ``n < 2`` input, so the caller falls back to the generic ``bootstrap_metrics`` path.
"""

from __future__ import annotations

import logging
from typing import Optional

import numba
import numpy as np

from mlframe.calibration.policy import DEFAULT_ECE_NBINS, _ece_score_numba_serial
from mlframe.metrics import NUMBA_NJIT_PARAMS, fast_roc_auc_unstable
# _fast_brier_score_loss_seq / _fast_log_loss_binary_seq are the njit-compiled sequential kernels
# specifically needed for calls from INSIDE this module's own @numba.njit(parallel=True) bootstrap
# kernel -- the public fast_brier_score_loss/fast_log_loss_binary dispatchers pick seq vs. parallel
# at plain-Python runtime and are not njit-callable, so the private variant is the only usable one
# here; not promotable to the public surface without misleading general callers.
from mlframe.metrics._core_auc_brier import _fast_brier_score_loss_seq
from mlframe.metrics._log_loss_and_separation import _fast_log_loss_binary_seq

from ._bootstrap_jackknife import _ci_from_samples, _jackknife_auc, _jackknife_ece, _jackknife_mean_metric, _jackknife_metric

logger = logging.getLogger(__name__)

_LOG_LOSS_EPS_F64 = float(np.finfo(np.float64).eps)


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _bootstrap_batch_auc_brier_ll_ece(
    idxs: np.ndarray,
    y_f64: np.ndarray,
    p_f64: np.ndarray,
    base_rank: np.ndarray,
    y_by_rank: np.ndarray,
    n: int,
    n_bins: int,
):
    """One ``prange`` pass over ``idxs`` (``(R, m)`` int64 resamples): every resample's
    brier/log_loss/ece/auc computed concurrently, no per-resample Python dispatch.

    Calls the existing SEQUENTIAL kernels (``_fast_brier_score_loss_seq`` /
    ``_fast_log_loss_binary_seq`` / ``_ece_score_numba_serial``) for brier/log_loss/ece --
    deliberately the seq variants, not the size-adaptive public dispatchers, because calling a
    ``parallel=True`` njit kernel from inside this function's own outer ``prange`` loop would be
    nested parallelism (numba does not usefully support it; the outer loop already supplies all
    the parallelism). AUC is inlined (the same counting + descending-walk as
    :func:`_fused_resample_auc` in ``_core_auc_brier.py``) rather than calling that function, to
    keep the per-resample scratch (``counts``/``ones``) private to this loop body without a
    caller-owned-buffer signature mismatch across the two call sites.
    """
    r_count = idxs.shape[0]
    auc_out = np.empty(r_count, dtype=np.float64)
    brier_out = np.empty(r_count, dtype=np.float64)
    ll_out = np.empty(r_count, dtype=np.float64)
    ece_out = np.empty(r_count, dtype=np.float64)
    for r in numba.prange(r_count):
        idx = idxs[r]
        yy = y_f64[idx]
        pp = p_f64[idx]
        brier_out[r] = _fast_brier_score_loss_seq(yy, pp)
        ll_out[r] = _fast_log_loss_binary_seq(yy, pp, _LOG_LOSS_EPS_F64)
        ece_out[r] = _ece_score_numba_serial(yy, pp, n_bins)

        counts = np.zeros(n, dtype=np.int64)
        ones = np.zeros(n, dtype=np.int64)
        m = idx.shape[0]
        for k in range(m):
            rk = base_rank[idx[k]]
            counts[rk] += 1
            ones[rk] += y_by_rank[rk]
        last_fps = 0
        last_tps = 0
        tps = 0
        fps = 0
        auc = 0
        for rk in range(n - 1, -1, -1):
            c = counts[rk]
            if c == 0:
                continue
            pos = ones[rk]
            neg = c - pos
            tps += pos
            fps += neg
            auc += (fps - last_fps) * (last_tps + tps)
            last_fps = fps
            last_tps = tps
        tmp = tps * fps * 2
        auc_out[r] = auc / tmp if tmp > 0 else np.nan
    return auc_out, brier_out, ll_out, ece_out


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _bootstrap_batch_auc_brier_ll_ece_grouped(
    idxs: np.ndarray,
    y_f64: np.ndarray,
    p_f64: np.ndarray,
    group_of_base: np.ndarray,
    y_base: np.ndarray,
    ngroups: int,
    n_bins: int,
):
    """Tie-aware twin of :func:`_bootstrap_batch_auc_brier_ll_ece` for base scores with duplicate values.

    Real predicted probabilities routinely repeat at scale (quantised tree-model leaf outputs, float32
    rounding, near-2M-row combos) -- the tie-free gate this bundle used to hard-require on ANY duplicate
    score silently defeated the whole 2.9-3.4x fused-parallel win in favour of ``bootstrap_metrics``'s
    fully serial GIL-bound loop on exactly the large real-world runs where it matters most. brier/log_loss/
    ece are order-invariant per-row reductions (unaffected by ties); only AUC needs the tie-aware
    treatment, via :func:`_fused_resample_auc_grouped`'s DISTINCT-score-group counting instead of
    per-base-rank counting -- bit-identical to ``fast_roc_auc_unstable`` on tied OR tie-free scores (see
    that function's docstring for why: AUC is invariant to within-tie argsort order)."""
    r_count = idxs.shape[0]
    auc_out = np.empty(r_count, dtype=np.float64)
    brier_out = np.empty(r_count, dtype=np.float64)
    ll_out = np.empty(r_count, dtype=np.float64)
    ece_out = np.empty(r_count, dtype=np.float64)
    for r in numba.prange(r_count):
        idx = idxs[r]
        yy = y_f64[idx]
        pp = p_f64[idx]
        brier_out[r] = _fast_brier_score_loss_seq(yy, pp)
        ll_out[r] = _fast_log_loss_binary_seq(yy, pp, _LOG_LOSS_EPS_F64)
        ece_out[r] = _ece_score_numba_serial(yy, pp, n_bins)

        counts = np.zeros(ngroups, dtype=np.int64)
        ones = np.zeros(ngroups, dtype=np.int64)
        m = idx.shape[0]
        for k in range(m):
            bi = idx[k]
            g = group_of_base[bi]
            counts[g] += 1
            ones[g] += y_base[bi]
        last_fps = 0
        last_tps = 0
        tps = 0
        fps = 0
        auc = 0
        for g in range(ngroups - 1, -1, -1):
            c = counts[g]
            if c == 0:
                continue
            pos = ones[g]
            neg = c - pos
            tps += pos
            fps += neg
            auc += (fps - last_fps) * (last_tps + tps)
            last_fps = fps
            last_tps = tps
        tmp = tps * fps * 2
        auc_out[r] = auc / tmp if tmp > 0 else np.nan
    return auc_out, brier_out, ll_out, ece_out


def _generate_resample_idxs(
    rng: np.random.Generator,
    n_bootstrap: int,
    n: int,
    stratify: "np.ndarray | None",
) -> np.ndarray:
    """Materialise the ``(n_bootstrap, n)`` index matrix with the EXACT same RNG call order as
    ``bootstrap_metrics``'s serial resample loop, so the resulting resamples -- and every metric's
    CI -- are bit-identical to that path for the same ``random_state``."""
    idxs = np.empty((n_bootstrap, n), dtype=np.int64)
    if stratify is None:
        for i in range(n_bootstrap):
            idxs[i] = rng.integers(0, n, size=n, dtype=np.int64)
        return idxs
    groups = {int(c): np.flatnonzero(stratify == c) for c in np.unique(stratify)}
    groups_list = list(groups.values())
    class_sizes = np.array([g.shape[0] for g in groups_list], dtype=np.int64)
    class_offsets = np.empty(class_sizes.shape[0] + 1, dtype=np.int64)
    class_offsets[0] = 0
    class_offsets[1:] = np.cumsum(class_sizes)
    for i in range(n_bootstrap):
        for c in range(class_sizes.shape[0]):
            sz = int(class_sizes[c])
            rand = rng.integers(0, sz, size=sz, dtype=np.int64)
            idxs[i, class_offsets[c] : class_offsets[c + 1]] = groups_list[c][rand]
    return idxs


def bootstrap_auc_brier_ll_ece_batch(
    y_true: np.ndarray,
    p_pos: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    stratify: "np.ndarray | None" = None,
    random_state: Optional[int] = None,
    method: str = "bca",
    n_bins: int = DEFAULT_ECE_NBINS,
    chunk_size: int = 200,
) -> "Optional[dict]":
    """Fused fast path for ``honest_diagnostics._bootstrap_block``'s roc_auc/brier/log_loss/ece bundle.

    Returns ``{"roc_auc": {...}, "brier": {...}, "log_loss": {...}, "ece": {...}}`` -- same shape as
    ``bootstrap_metrics`` for this exact metric set. Handles BOTH tie-free and tied base scores (see
    :func:`_bootstrap_batch_auc_brier_ll_ece_grouped`'s docstring for why the tie-free-only gate this
    function used to hard-require was itself the bigger, more damaging pattern: real predicted
    probabilities routinely repeat at scale, silently forcing the fully serial ``bootstrap_metrics``
    fallback on exactly the large real-world runs this fusion targets). Still returns ``None`` for
    ``n < 2`` so the caller falls back to ``bootstrap_metrics`` on a degenerate input.

    Bit-identical to ``bootstrap_metrics(y_true, p_pos, {"brier": ..., "log_loss": ..., "ece": ...},
    metric_fns_idx={"roc_auc": make_bootstrap_auc_resampler(y_true, p_pos)}, stratify=stratify,
    random_state=random_state, method=method)`` for the same inputs: identical resample index
    sequence (:func:`_generate_resample_idxs` matches the generic loop's RNG call order exactly),
    identical per-resample metric values (same underlying njit kernels -- the grouped AUC counting is
    bit-identical to the exact/tie-free resampler on ANY base scores, tied or not, per
    ``_fused_resample_auc_grouped``'s docstring), identical CI reduction (the same ``_ci_from_samples``
    / jackknife helpers, called unchanged).
    """
    y_true = np.ascontiguousarray(y_true)
    p_pos = np.ascontiguousarray(p_pos)
    n = p_pos.shape[0]
    if n < 2:
        return None

    asc_order = np.argsort(p_pos)
    sorted_score = p_pos[asc_order]
    tie_free = n < 2 or bool(np.all(sorted_score[1:] != sorted_score[:-1]))

    y_f64 = np.ascontiguousarray(y_true, dtype=np.float64)
    p_f64 = np.ascontiguousarray(p_pos, dtype=np.float64)

    rng = np.random.default_rng(random_state)
    idxs = _generate_resample_idxs(rng, n_bootstrap, n, stratify)

    auc_samples = np.empty(n_bootstrap, dtype=np.float64)
    brier_samples = np.empty(n_bootstrap, dtype=np.float64)
    ll_samples = np.empty(n_bootstrap, dtype=np.float64)
    ece_samples = np.empty(n_bootstrap, dtype=np.float64)
    if tie_free:
        base_rank = np.empty(n, dtype=np.int64)
        base_rank[asc_order] = np.arange(n, dtype=np.int64)
        y_by_rank = np.ascontiguousarray(y_true[asc_order].astype(np.int64))
        for lo in range(0, n_bootstrap, chunk_size):
            hi = min(lo + chunk_size, n_bootstrap)
            auc_c, brier_c, ll_c, ece_c = _bootstrap_batch_auc_brier_ll_ece(idxs[lo:hi], y_f64, p_f64, base_rank, y_by_rank, n, n_bins)
            auc_samples[lo:hi] = auc_c
            brier_samples[lo:hi] = brier_c
            ll_samples[lo:hi] = ll_c
            ece_samples[lo:hi] = ece_c
    else:
        # DISTINCT-score-group counting (see _fused_resample_auc_grouped's docstring): group_of_base[i] =
        # index of base i's distinct score in ascending order, 0..ngroups-1.
        boundaries = sorted_score[1:] != sorted_score[:-1]
        group_by_rank = np.empty(n, dtype=np.int64)
        group_by_rank[0] = 0
        if n > 1:
            group_by_rank[1:] = np.cumsum(boundaries)
        ngroups = int(group_by_rank[-1]) + 1 if n > 0 else 0
        group_of_base = np.empty(n, dtype=np.int64)
        group_of_base[asc_order] = group_by_rank
        y_base = np.ascontiguousarray(y_true.astype(np.int64))
        for lo in range(0, n_bootstrap, chunk_size):
            hi = min(lo + chunk_size, n_bootstrap)
            auc_c, brier_c, ll_c, ece_c = _bootstrap_batch_auc_brier_ll_ece_grouped(idxs[lo:hi], y_f64, p_f64, group_of_base, y_base, ngroups, n_bins)
            auc_samples[lo:hi] = auc_c
            brier_samples[lo:hi] = brier_c
            ll_samples[lo:hi] = ll_c
            ece_samples[lo:hi] = ece_c

    points = {
        "roc_auc": float(fast_roc_auc_unstable(y_true, p_pos)),
        "brier": float(_fast_brier_score_loss_seq(y_f64, p_f64)),
        "log_loss": float(_fast_log_loss_binary_seq(y_f64, p_f64, _LOG_LOSS_EPS_F64)),
        "ece": float(_ece_score_numba_serial(y_f64, p_f64, n_bins)),
    }

    results: dict = {}
    _raw = {"roc_auc": auc_samples, "brier": brier_samples, "log_loss": ll_samples, "ece": ece_samples}
    for name, raw in _raw.items():
        finite = raw[np.isfinite(raw)]
        if finite.shape[0] == 0:
            results[name] = {"error": f"all {n_bootstrap} resamples failed for {name!r}"}
            continue
        jackknife = None
        if method == "bca":
            if name == "roc_auc":
                jackknife = _jackknife_auc(y_true, p_pos)
            elif name == "brier":
                per_row = (p_f64 - y_f64) ** 2
                jackknife = _jackknife_mean_metric(y_true, per_row, requires_both_classes=False)
            elif name == "log_loss":
                _pc = np.clip(p_f64, _LOG_LOSS_EPS_F64, 1.0 - _LOG_LOSS_EPS_F64)
                per_row = np.where(y_f64 == 1, -np.log(_pc), -np.log(1.0 - _pc))
                jackknife = _jackknife_mean_metric(y_true, per_row, requires_both_classes=True)
            elif name == "ece":
                # _jackknife_ece is the SAME O(n) closed-form honest_diagnostics._bootstrap_block's
                # bootstrap_metrics fallback path already wires into jackknife_fns["ece"] (2026-07-31,
                # ~800x vs the generic gather jackknife) -- this bundle used to duplicate the OLD, pre-fix
                # bootstrap_metrics behaviour (a bare _jackknife_metric gather call) instead of the closed
                # form, caught live via a 2M-row cProfile: _jackknife_metric cost 24.0s/12 calls here, now
                # that the tied-scores fix above routes nearly every _bootstrap_block call through this
                # module. Mirrors bootstrap_metrics's own custom-jackknife-then-gather-fallback order: try
                # the closed form first, fall back to the generic gather only on its documented degeneracy
                # (n<3, non-binary labels, non-finite total).
                jackknife = _jackknife_ece(y_true, p_pos, n_bins=n_bins)
                if jackknife is None:
                    jackknife = _jackknife_metric(y_true, p_pos, lambda yy, pp: float(_ece_score_numba_serial(np.ascontiguousarray(yy, dtype=np.float64), np.ascontiguousarray(pp, dtype=np.float64), n_bins)))
        lo_ci, hi_ci = _ci_from_samples(finite, points[name], alpha, method, jackknife)
        results[name] = {"point": points[name], "lo": lo_ci, "hi": hi_ci, "samples": finite}
    return results
