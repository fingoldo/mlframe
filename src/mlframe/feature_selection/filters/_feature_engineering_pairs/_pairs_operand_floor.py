"""Operand-floor check of the pair-FE joint-prevalence gate, split out of ``_pairs_score`` to keep that module under the size budget."""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

import numpy as np
from numba import njit, prange

__all__ = ["_beats_the_larger_operand", "operand_uplift_null_bar"]


@njit(cache=True, nogil=True)
def _plugin_mi(codes_x: np.ndarray, codes_y: np.ndarray, n_x: int, n_y: int) -> float:
    """Plug-in mutual information (nats) of two integer code vectors, the value ``sklearn.metrics.mutual_info_score`` gives, in one pass over the rows."""
    n = codes_x.shape[0]
    joint = np.zeros((n_x, n_y), dtype=np.int64)
    for i in range(n):
        joint[codes_x[i], codes_y[i]] += 1
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    total = float(n)
    mi = 0.0
    for a in range(n_x):
        if px[a] == 0:
            continue
        for b in range(n_y):
            c = joint[a, b]
            if c > 0:
                mi += (c / total) * np.log(c * total / (float(px[a]) * float(py[b])))
    return mi


@njit(cache=True, nogil=True, parallel=True)
def _null_uplift_values(codes_a: np.ndarray, codes_b: np.ndarray, perms: np.ndarray, y_codes: np.ndarray, n_b: int, n_joint: int, n_y: int, floor: float) -> np.ndarray:
    """``_plugin_mi(codes_a * n_b + codes_b[perm], y_codes, n_joint, n_y) / floor`` for every row ``perm`` of ``perms``, the permutations in parallel; each one runs the serial ``_plugin_mi`` code, so the values are the serial loop's."""
    n_perm = perms.shape[0]
    out = np.empty(n_perm, dtype=np.float64)  # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
    for k in prange(n_perm):
        joint = codes_a * n_b + codes_b[perms[k]]
        out[k] = _plugin_mi(joint, y_codes, n_joint, n_y) / floor
    return out


_PERM_BANK: "OrderedDict[tuple[int, int, int], np.ndarray]" = OrderedDict()
_PERM_BANK_LOCK = threading.Lock()
_PERM_BANK_MAX_BYTES = 64 * 1024 * 1024


def _permutation_bank(n: int, n_perm: int, seed: int) -> np.ndarray:
    """The ``n_perm`` permutations of ``range(n)`` that ``np.random.default_rng(seed)`` yields one after the other: the same for every pair of rows ``n`` (each call used to build a fresh generator), so they are drawn once
    and kept (an LRU of at most ``_PERM_BANK_MAX_BYTES``)."""
    key = (int(n), int(n_perm), int(seed))
    with _PERM_BANK_LOCK:
        bank = _PERM_BANK.get(key)
        if bank is not None:
            _PERM_BANK.move_to_end(key)
            return bank
    rng = np.random.default_rng(seed)
    bank = np.stack([rng.permutation(n) for _ in range(int(n_perm))]) if n_perm > 0 else np.empty((0, n), dtype=np.int64)
    with _PERM_BANK_LOCK:
        _PERM_BANK[key] = bank
        while len(_PERM_BANK) > 1 and sum(b.nbytes for b in _PERM_BANK.values()) > _PERM_BANK_MAX_BYTES:
            _PERM_BANK.popitem(last=False)  # evict-ok: a miss redraws the same permutations
    return bank


def operand_uplift_null_bar(codes_a: np.ndarray | None, codes_b: np.ndarray | None, codes_y: np.ndarray, floor: float, n_perm: int = 20, quantile: float = 0.95, seed: int = 0) -> float:
    """Data-driven uplift bar over the larger operand: the null quantile of joint-MI uplift when the partner is noise.

    The partner's discretised codes are permuted, which keeps its marginal and destroys its relation to y. The joint MI
    of that noise pair, divided by *floor*, is the uplift a plug-in estimate reaches by chance; the bar is its *quantile*
    and never below 1.0. Returns 1.0 when the operand codes are unavailable or the floor is not positive.
    """
    if codes_a is None or codes_b is None or floor <= 0.0:
        return 1.0
    codes_a = np.asarray(codes_a, dtype=np.int64)
    codes_b = np.asarray(codes_b, dtype=np.int64)
    y_codes = np.unique(np.asarray(codes_y), return_inverse=True)[1].astype(np.int64).ravel()
    if codes_a.shape != codes_b.shape or codes_a.ndim != 1 or codes_a.shape[0] != y_codes.shape[0]:
        raise ValueError(f"operand codes and target must be equal-length 1-D vectors, got {codes_a.shape}, {codes_b.shape}, {y_codes.shape}")
    if codes_a.size and (int(codes_a.min()) < 0 or int(codes_b.min()) < 0):
        raise ValueError("operand codes must be non-negative ordinals (a negative code indexes outside the joint table)")
    n_b = int(codes_b.max()) + 1 if codes_b.size else 1
    n_joint = (int(codes_a.max()) + 1) * n_b if codes_a.size else 1
    n_y = int(y_codes.max()) + 1 if y_codes.size else 1
    perms = _permutation_bank(codes_b.shape[0], n_perm, seed)
    from mlframe._numba_parallel_guard import parallel_kernel_entry

    with parallel_kernel_entry():  # a parallel=True kernel is entered by one thread at a time unless the threading layer allows concurrent entry
        null = _null_uplift_values(codes_a, codes_b, perms, y_codes, n_b, n_joint, n_y, float(floor))
    return max(1.0, float(np.quantile(null, quantile)))



def _operand_uplift_null_bar_reference(codes_a: np.ndarray | None, codes_b: np.ndarray | None, codes_y: np.ndarray, floor: float, n_perm: int = 20, quantile: float = 0.95, seed: int = 0) -> float:
    """The serial reference of :func:`operand_uplift_null_bar` (a fresh generator and a Python loop per call), kept for the equivalence test.

    Data-driven uplift bar over the larger operand: the null quantile of joint-MI uplift when the partner is noise.

    The partner's discretised codes are permuted, which keeps its marginal and destroys its relation to y. The joint MI
    of that noise pair, divided by *floor*, is the uplift a plug-in estimate reaches by chance; the bar is its *quantile*
    and never below 1.0. Returns 1.0 when the operand codes are unavailable or the floor is not positive.
    """
    if codes_a is None or codes_b is None or floor <= 0.0:
        return 1.0
    codes_a = np.asarray(codes_a, dtype=np.int64)
    codes_b = np.asarray(codes_b, dtype=np.int64)
    y_codes = np.unique(np.asarray(codes_y), return_inverse=True)[1].astype(np.int64).ravel()
    if codes_a.shape != codes_b.shape or codes_a.ndim != 1 or codes_a.shape[0] != y_codes.shape[0]:
        raise ValueError(f"operand codes and target must be equal-length 1-D vectors, got {codes_a.shape}, {codes_b.shape}, {y_codes.shape}")
    if codes_a.size and (int(codes_a.min()) < 0 or int(codes_b.min()) < 0):
        raise ValueError("operand codes must be non-negative ordinals (a negative code indexes outside the joint table)")
    n_b = int(codes_b.max()) + 1 if codes_b.size else 1
    n_joint = (int(codes_a.max()) + 1) * n_b if codes_a.size else 1
    n_y = int(y_codes.max()) + 1 if y_codes.size else 1
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=np.float64)  # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
    for i in range(n_perm):
        joint = codes_a * n_b + codes_b[rng.permutation(codes_b.shape[0])]
        null[i] = _plugin_mi(joint, y_codes, n_joint, n_y) / floor
    return max(1.0, float(np.quantile(null, quantile)))


def operand_uplift_null_bar_for_pair(raw_vars_pair: Any, operand_marginal_mi: Any, operand_discretized: Any, codes_y: np.ndarray, floor: float, **kwargs: Any) -> float:
    """``operand_uplift_null_bar`` for a raw pair, permuting the operand with the SMALLER marginal MI whichever slot it sits in.

    The null asks what a noise partner reaches next to the stronger operand; permuting the stronger one instead destroys the signal the bar is measured against, so the
    bar collapsed to 1.0 and a degraded copy of the strong operand passed whenever it was listed second.
    """
    first, second = raw_vars_pair[0], raw_vars_pair[1]
    if operand_marginal_mi(first) < operand_marginal_mi(second):
        first, second = second, first  # ``first`` is now the stronger operand, kept intact; ``second`` is permuted
    return operand_uplift_null_bar(operand_discretized(first), operand_discretized(second), codes_y, floor, **kwargs)


def _beats_the_larger_operand(best_mi: float, operand_marginal_mi, raw_vars_pair, messages, pair_mi: float = 0.0, uplift_bar: float = 1.0) -> bool:
    """Whether the engineered MI clears *uplift_bar* over the larger operand's own MI.

    With one strong operand and one noise operand the pair joint MI is about the strong operand's own MI, so the joint
    gate alone passed noise-degraded copies of it (uplift 1.011-1.032); genuine pairs measured 1.36-2.32. Appends the
    reason to *messages* when rejecting and *messages* is not None. The floor only guards a pair whose partner adds nothing:
    when the raw pair's joint MI itself clears the same uplift over the larger operand, the partner carries real joint
    information and the engineered column is not a degraded copy of one operand.
    """
    floor = max(operand_marginal_mi(raw_vars_pair[0]), operand_marginal_mi(raw_vars_pair[1]))
    bar = floor * uplift_bar
    if floor <= 0.0 or best_mi > bar or pair_mi > bar:
        return True
    if messages is not None:
        messages.append(
            f"joint gate operand floor: best engineered MI={best_mi:.4f} does not beat the larger operand marginal "
            f"MI={floor:.4f}; a degraded copy of one operand, not a pair feature."
        )
    return False
