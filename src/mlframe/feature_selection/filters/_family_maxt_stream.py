"""Memory-bounded family-wise maxT gate: the stacked code matrix of the family is built one group of columns at a time.

``family_pass_batches`` stacks every configuration scored for a pair into one ``(K_total, n)`` code matrix, one byte per code at best; at a million rows a wide family is tens of GB. Every statistic
of the gate decomposes over column groups without changing a value:

* the observed MI of a column and the number of occupied bins of a column (the input of the analytic triage) do not depend on the other columns;
* the null maximum of the family under one label permutation is the maximum over the groups of the groups' maxima, when every group is scored against the SAME permutation block: the generator is
  rewound to the start of the block for each group, so the block is drawn once (the permutation cache turns the repeats into hits) and the stream the caller sees is the one a single pass produces;
* the early rejection and the exact early stop read the null maxima of the whole family on every call, so they are two-pass by construction: each call walks all the groups.

A family whose stack fits the budget keeps the single-pass path (and its column deduplication) unchanged; only a family that does not fit is streamed.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from mlframe._numba_parallel_guard import parallel_kernel_entry

# A group of columns may hold this share of the free host RAM (the stacked group, and the transposed copies the kernels make of it).
_STACK_FREE_FRACTION = 0.25


def _code_itemsize(mats: "list[np.ndarray]") -> int:
    """Bytes per code of the narrowest integer width that holds the codes of ``mats`` (as :func:`_stack_codes_transposed` chooses it)."""
    # A matrix whose dtype cannot exceed 127 (int8) needs no scan of its values: its maximum is bounded by the type. The family holds thousands of int8 matrices per call, so the scan cost seconds.
    scan = [m for m in mats if m.size and not (m.dtype.kind == "i" and np.iinfo(m.dtype).max <= 127)]
    top = max((int(m.max()) for m in scan), default=0)
    return 1 if top <= 127 else (2 if top <= 32767 else 8)


def plan_stream_groups(mats: "list[np.ndarray]", widths: "list[int]", n_rows: int) -> "list[list[tuple[int, int, int]]] | None":
    """Groups of column pieces ``(batch, first, stop)`` whose stacked matrix fits the budget, or ``None`` when the whole family fits (the single-pass path then runs)."""
    from ._orthogonal_univariate_fe._streamed_gpu_mi import _host_free_bytes

    per_column = max(1, int(n_rows)) * _code_itemsize(mats)
    budget_columns = max(1, int(_STACK_FREE_FRACTION * _host_free_bytes() // per_column))
    if int(sum(widths)) <= budget_columns:
        return None
    groups: "list[list[tuple[int, int, int]]]" = [[]]
    held = 0
    for i, w in enumerate(widths):
        for first in range(0, int(w), budget_columns):
            stop = min(int(w), first + budget_columns)
            if held + (stop - first) > budget_columns and groups[-1]:
                groups.append([])
                held = 0
            groups[-1].append((i, first, stop))
            held += stop - first
    return groups


def _stack_group(mats: "list[np.ndarray]", group: "list[tuple[int, int, int]]") -> "tuple[np.ndarray, np.ndarray]":
    """The ``(columns, n)`` code matrix of one group and the bin count of every row."""
    from ._family_maxt_gate import _stack_codes_transposed

    pieces = [mats[i][:, first:stop] for i, first, stop in group]
    cols = _stack_codes_transposed(pieces, [stop - first for _i, first, stop in group])
    return cols, cols.max(axis=1).astype(np.int64) + 1


def streamed_null_max(mats: "list[np.ndarray]", groups: "list[list[tuple[int, int, int]]]", y: np.ndarray, n_y: int):
    """``null_max(m, rng)``: the next ``m`` null maxima over the whole family, with one group of columns stacked at a time.

    Every group is scored against the same permutation draw (the generator is rewound to the start of the draw for each group, and ends where a single pass would leave it)."""
    from ._family_maxt_gate import family_null_max

    def null_max(m: int, rng: np.random.Generator) -> np.ndarray:
        """The next ``m`` null maxima over the whole family.

        Device groups fold their maxima into one device accumulator, so the family's result crosses the bus once; host groups
        fold into the host vector. The two are combined at the end (the maximum is associative, so the order does not matter)."""
        from ._family_maxt_gate import _null_device_wanted

        out = np.zeros(int(m), dtype=np.float64)  # Intentional float64: statistic vector (null max), exact permutation ties
        acc = None  # device accumulator, created when the first group runs on the device
        state = rng.bit_generator.state
        for group in groups:
            cols, n_x = _stack_group(mats, group)
            replay = np.random.default_rng()
            replay.bit_generator.state = state
            if _null_device_wanted(cols, y, int(m)):
                import cupy as cp

                if acc is None:
                    acc = cp.zeros(int(m), dtype=cp.float64)  # Intentional float64: statistic vector, exact permutation ties
                family_null_max(cols, n_x, y, n_y, int(m), replay, acc=acc)
            else:
                out = np.maximum(out, cast(np.ndarray, family_null_max(cols, n_x, y, n_y, int(m), replay)))  # no accumulator: the maxima come back
            rng.bit_generator.state = replay.bit_generator.state
            del cols
        if acc is not None:
            import cupy as cp

            out = np.maximum(out, cp.asnumpy(acc))
        return out

    return null_max


def family_pass_streamed(mats: "list[np.ndarray]", widths: "list[int]", y: np.ndarray, n_y: int, alpha: float, n_perm: int, seed: int, groups: "list[list[tuple[int, int, int]]]") -> list:
    """Pass flag per batch of one family, with the same decisions as the single pass, holding one group of columns at a time."""
    from ._analytic_mi_null import analytic_null_enabled
    from ._family_maxt_gate import _analytic_triage_core, _occupied_bins, _permutation_pass_flags_with, mi_all_columns_njit

    n_rows = int(mats[0].shape[0])
    starts = np.cumsum([0, *widths])
    total = int(starts[-1])
    observed = np.empty(total, dtype=np.float64)  # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
    occupied = np.empty(total, dtype=np.int64)
    at = 0
    for group in groups:
        cols, n_x = _stack_group(mats, group)
        with parallel_kernel_entry():
            observed[at : at + cols.shape[0]] = mi_all_columns_njit(cols, n_x, y, n_y)
            occupied[at : at + cols.shape[0]] = _occupied_bins(cols, n_x)
        at += cols.shape[0]
        del cols
    n_batches = len(widths)
    observed_max = np.array([float(observed[starts[j] : starts[j + 1]].max()) if starts[j + 1] > starts[j] else 0.0 for j in range(n_batches)], dtype=np.float64)  # Intentional float64: statistic vector
    flags = np.zeros(n_batches, dtype=np.bool_)

    null_max = streamed_null_max(mats, groups, y, n_y)

    status = _analytic_triage_core(n_rows, occupied, y, n_y, observed, starts, alpha, None) if (analytic_null_enabled() and n_rows > 0) else None
    if status is None:
        flags[:] = _permutation_pass_flags_with(null_max, observed_max, alpha, n_perm, seed)
        return [bool(f) for f in flags]
    flags[status > 0] = True
    gray = status == 0
    if gray.any():
        flags[gray] = _permutation_pass_flags_with(null_max, observed_max[gray], alpha, n_perm, seed)
    return [bool(f) for f in flags]


__all__ = ["family_pass_streamed", "plan_stream_groups", "streamed_null_max"]
