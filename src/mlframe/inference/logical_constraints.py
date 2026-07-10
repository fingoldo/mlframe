"""Enforce declared label-implication rules on multi-label prediction matrices.

Real multi-label problems often have known domain constraints of the form "label A is never true without
label B" (A implies B) — e.g. a premium product tier implies the base tier, a fraud sub-category implies the
parent fraud flag, an upsell implies the base subscription. A trained model has no way to know this unless it
is baked into the loss, so its raw predictions can rank A above B even when the constraint is known to hold.
``apply_logical_constraints`` is a rank-preserving post-processor: for every row that violates a declared
implication, it SWAPS the two predicted values (rather than clipping/zeroing one), which fixes the ordering
while leaving the marginal distribution of predicted scores completely unchanged — no recalibration needed.

Backend: fused ``numba.njit`` kernels (single-thread and ``prange``-parallel) plus a cupy GPU path, with a
numpy fallback. The winning backend depends on shape AND hardware (measured: njit_single wins at n=1,000,
njit_parallel at n=100,000, cupy at n=1,000,000 — see ``_ktc_dispatch.py``), so the choice routes through
``kernel_tuning_cache`` rather than a hardcoded threshold. All backends are bit-identical by construction
(integer-index swaps, no floating-point reduction reordering).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    import numba
    from numba import prange

    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - numba is a core mlframe dependency; exercised only if absent
    _NUMBA_AVAILABLE = False

_PARALLEL_THRESHOLD = 20_000


def _apply_numpy(out: np.ndarray, rules: Sequence[tuple[int, int]]) -> np.ndarray:
    for child_idx, parent_idx in rules:
        violates = out[:, child_idx] > out[:, parent_idx]
        child_vals = out[violates, child_idx].copy()
        out[violates, child_idx] = out[violates, parent_idx]
        out[violates, parent_idx] = child_vals
    return out


if _NUMBA_AVAILABLE:

    @numba.njit(cache=True)
    def _apply_njit(out: np.ndarray, rules_arr: np.ndarray) -> np.ndarray:
        n = out.shape[0]
        n_rules = rules_arr.shape[0]
        for i in range(n):
            for r in range(n_rules):
                c = rules_arr[r, 0]
                p = rules_arr[r, 1]
                if out[i, c] > out[i, p]:
                    tmp = out[i, c]
                    out[i, c] = out[i, p]
                    out[i, p] = tmp
        return out

    @numba.njit(cache=True, parallel=True)
    def _apply_njit_parallel(out: np.ndarray, rules_arr: np.ndarray) -> np.ndarray:
        n = out.shape[0]
        n_rules = rules_arr.shape[0]
        for i in prange(n):
            for r in range(n_rules):
                c = rules_arr[r, 0]
                p = rules_arr[r, 1]
                if out[i, c] > out[i, p]:
                    tmp = out[i, c]
                    out[i, c] = out[i, p]
                    out[i, p] = tmp
        return out


def _apply_cupy(out: np.ndarray, rules_arr: np.ndarray) -> np.ndarray:
    import cupy as cp

    out_gpu = cp.asarray(out)
    for r in range(rules_arr.shape[0]):
        c, p = int(rules_arr[r, 0]), int(rules_arr[r, 1])
        violates = out_gpu[:, c] > out_gpu[:, p]
        tmp = out_gpu[violates, c].copy()
        out_gpu[violates, c] = out_gpu[violates, p]
        out_gpu[violates, p] = tmp
    return np.asarray(cp.asnumpy(out_gpu))


def apply_logical_constraints(preds: np.ndarray, rules: Sequence[tuple[int, int]]) -> np.ndarray:
    """Swap child/parent prediction columns wherever a declared implication rule is violated.

    Parameters
    ----------
    preds
        ``(n_samples, n_labels)`` array of per-label scores/probabilities.
    rules
        Sequence of ``(child_idx, parent_idx)`` column-index pairs, each meaning "label ``child_idx`` implies
        label ``parent_idx``" (child is never true without parent — e.g. a premium tier implies the base
        tier). A row VIOLATES the rule when ``preds[row, child_idx] > preds[row, parent_idx]``: the model
        predicts the dependent label more strongly than the label it requires, which is domain-inconsistent.

    Returns
    -------
    np.ndarray
        A copy of ``preds`` with, for every violating row and rule, the child/parent columns swapped. Rules
        are applied in order; a later rule can re-touch a row/column adjusted by an earlier rule (this is
        intentional — declare rules in dependency order for a rule chain, e.g. tier3->tier2 before
        tier2->tier1).

    Notes
    -----
    Swapping (not clipping) preserves the marginal distribution of predicted values exactly — a strictly
    rank-preserving fix, unlike zeroing/clamping the violating value which would distort calibration.
    """
    out = np.array(preds, dtype=np.float64, copy=True)
    if out.ndim != 2:
        raise ValueError(f"apply_logical_constraints: preds must be 2D (n_samples, n_labels); got shape {out.shape}")
    n_labels = out.shape[1]
    for child_idx, parent_idx in rules:
        if not (0 <= child_idx < n_labels) or not (0 <= parent_idx < n_labels):
            raise ValueError(f"apply_logical_constraints: rule ({child_idx}, {parent_idx}) out of bounds for {n_labels} labels")
        if child_idx == parent_idx:
            raise ValueError(f"apply_logical_constraints: rule ({child_idx}, {parent_idx}) has identical child/parent index")

    if not rules:
        return out
    if not _NUMBA_AVAILABLE:
        return _apply_numpy(out, rules)

    rules_arr = np.asarray(rules, dtype=np.int64)
    n = out.shape[0]
    from mlframe.inference._ktc_dispatch import choose_logical_constraints_backend

    fallback = "njit_parallel" if n >= _PARALLEL_THRESHOLD else "njit_single"
    backend = choose_logical_constraints_backend(n, n_labels, len(rules), fallback=fallback)
    if backend == "cupy":
        try:
            return _apply_cupy(out, rules_arr)
        except Exception:  # GPU path failed at runtime (OOM, driver hiccup) -> CPU fallback, never raise
            backend = "njit_parallel"
    if backend == "njit_parallel":
        return np.asarray(_apply_njit_parallel(out, rules_arr))
    return np.asarray(_apply_njit(out, rules_arr))


def discover_logical_constraints(
    y: np.ndarray,
    min_child_support: int = 10,
) -> list[tuple[int, int]]:
    """Auto-discover ``(child, parent)`` implication rules from historical labels with ZERO counter-examples.

    A rule ``child implies parent`` is proposed only when the training data contains not one single row where
    ``child == 1`` and ``parent == 0`` — i.e. the implication holds with certainty over the observed history,
    not just "mostly". This is deliberately conservative: with real, noisy labels a threshold like "child=1,
    parent=0 in <1% of rows" would flag spurious co-occurrence patterns as hard domain rules and start
    corrupting predictions that were actually correct. Pass rules discovered here straight into
    ``apply_logical_constraints``.

    Parameters
    ----------
    y
        ``(n_samples, n_labels)`` binary (0/1) label matrix — historical/training labels, NOT predictions.
    min_child_support
        Minimum number of positive (``child == 1``) rows required before a column is even considered as a
        child in a candidate rule. Guards against trivially "discovering" a rule from a handful of positives
        (e.g. 2 out of 2 positives happen to also have parent=1 -- not enough evidence to call it a law).

    Returns
    -------
    list[tuple[int, int]]
        ``(child_idx, parent_idx)`` pairs with zero observed counter-examples and adequate child support,
        sorted by child support descending (most-evidenced rules first). Self-pairs are never returned.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    if y_arr.ndim != 2:
        raise ValueError(f"discover_logical_constraints: y must be 2D (n_samples, n_labels); got shape {y_arr.shape}")
    n_labels = y_arr.shape[1]
    if n_labels < 2:
        return []

    child_support = y_arr.sum(axis=0)
    # counter_examples[c, p] = count of rows where child c == 1 AND parent p == 0.
    counter_examples = y_arr.T @ (1.0 - y_arr)

    # Vectorised candidate mask: zero counter-examples, adequate child support, no self-pairs. Replaces an
    # O(n_labels^2) Python double loop (the dominant cost at n_labels>=100 -- profiled: 100 labels =
    # 10,000 pure-Python iterations per call) with boolean matrix ops + a single ``nonzero``/``argsort`` pass.
    candidate_mask = (counter_examples == 0.0) & (child_support[:, None] >= min_child_support)
    np.fill_diagonal(candidate_mask, False)
    c_idx, p_idx = np.nonzero(candidate_mask)
    if c_idx.size == 0:
        return []
    order = np.argsort(-child_support[c_idx], kind="stable")
    return list(zip(c_idx[order].tolist(), p_idx[order].tolist()))


__all__ = ["apply_logical_constraints", "discover_logical_constraints"]
