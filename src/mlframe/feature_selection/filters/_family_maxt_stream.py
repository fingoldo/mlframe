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
from typing import Any

# A group of columns may hold this share of the free host RAM (the stacked group, and the transposed copies the kernels make of it).
_STACK_FREE_FRACTION = 0.25
# What a stacked group of codes costs per code, in bytes, beyond the code itself: its column bit sets (a bin set of ``n / 8`` bytes per bin, about one and a half bytes per code at the
# usual twelve bins) and the transposed copy the kernels take of it. Measured at 1000000 rows x 100 columns: the group budget counted only the codes, so the groups of a wide family
# held about three and a half times the share of the free RAM they were planned for (peak growth 8.2 GB = 20 x the frame).
_GROUP_FOOTPRINT_PER_CODE = 3.5


def _code_itemsize(mats: "list[np.ndarray]") -> int:
    """Bytes per code of the narrowest integer width that holds the codes of ``mats`` (as :func:`_stack_codes_transposed` chooses it)."""
    # A matrix whose dtype cannot exceed 127 (int8) needs no scan of its values: its maximum is bounded by the type. The family holds thousands of int8 matrices per call, so the scan cost seconds.
    scan = [m for m in mats if m.size and not (m.dtype.kind == "i" and np.iinfo(m.dtype).max <= 127)]
    # The family's matrices are mostly single columns of a few block matrices (views of the binner's output): scanning a column view walks the whole parent at its row stride, once per
    # column (75741 scans, 19.6 s over a default 100000 x 100 fit). The question asked is only which width the maximum needs, so when a parent is read at most twice over by its
    # views, its own maximum (one contiguous pass) bounds them all; a parent over that, or one whose bound crosses a width, is answered by the views themselves.
    by_root: "dict[int, list[np.ndarray]]" = {}
    for m in scan:
        by_root.setdefault(id(m.base) if isinstance(m.base, np.ndarray) else id(m), []).append(m)
    top = 0
    for views in by_root.values():
        root = views[0].base
        if isinstance(root, np.ndarray) and root.size <= 2 * sum(v.size for v in views):
            bound = int(root.max())
            if bound <= 127:
                top = max(top, bound)
                continue
        top = max(top, max(int(v.max()) for v in views))
    return 1 if top <= 127 else (2 if top <= 32767 else 8)


def plan_stream_groups(mats: "list[np.ndarray]", widths: "list[int]", n_rows: int, max_group_bytes: "int | None" = None) -> "list[list[tuple[int, int, int]]] | None":
    """Groups of column pieces ``(batch, first, stop)`` whose stacked matrix fits the budget, or ``None`` when the whole family fits (the single-pass path then runs).

    ``max_group_bytes`` narrows the group to that many bytes of stacked codes (with their footprint) on top of the free-memory share: a caller whose columns are regenerated on demand streams even when the RAM is free."""
    from ._orthogonal_univariate_fe._streamed_gpu_mi import _host_free_bytes

    per_column = max(1, int(n_rows * _code_itemsize(mats) * _GROUP_FOOTPRINT_PER_CODE))
    budget_columns = max(1, int(_STACK_FREE_FRACTION * _host_free_bytes() // per_column))
    if max_group_bytes is not None:
        budget_columns = max(1, min(budget_columns, int(max_group_bytes) // per_column))
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


def streamed_null_max(mats: "list[np.ndarray]", groups: "list[list[tuple[int, int, int]]]", y: np.ndarray, n_y: int) -> "Any":
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
            replay = np.random.default_rng()  # rng-entropy-ok: the bit-generator state is overwritten with the caller's state on the next line
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


def family_pass_columns_streamed(mats: "list[Any]", y: np.ndarray, alpha: float, n_perm: int, seed: int, group_bytes: int) -> np.ndarray:
    """Per-column pass flags of a family whose columns ``mats`` (each ``(n, 1)``, read through ``mats[i][:, a:b]``) are produced on demand; one group of at most ``group_bytes`` of codes is alive at a time.

    The decisions are those of ``family_maxt_pass_codes`` on the stacked family: the label encoding is the same, every column is its own batch, and the groups are scored against one permutation draw."""
    from ._y_encoding import encode_y_for_classif_mi

    _, y_codes = np.unique(encode_y_for_classif_mi(np.asarray(y).ravel()), return_inverse=True)
    y_codes = np.ascontiguousarray(y_codes.astype(np.int64).ravel())
    n_y = int(y_codes.max()) + 1
    widths = [1] * len(mats)
    groups = plan_stream_groups(mats, widths, int(mats[0].shape[0]), max_group_bytes=group_bytes)
    if groups is None:
        groups = [[(i, 0, 1) for i in range(len(mats))]]
    return np.asarray(family_pass_streamed(mats, widths, y_codes, n_y, alpha, n_perm, seed, groups), dtype=np.bool_)


__all__ = ["family_pass_columns_streamed", "family_pass_streamed", "plan_stream_groups", "streamed_null_max"]
