"""Column blocks around the one-shot batched CMI: ``batched_cmi_gpu`` of ``_fe_batched_mi`` (which re-exports it) runs a candidate matrix that does not fit the room under the calling
thread's pool in blocks of columns and joins the pieces in order.

The columns are independent, so the blocks give the same values, cards and return forms as one call on the whole matrix; a matrix that fits is one call, as before.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_CMI_BLOCK_COPIES = 3  # copies of the (rows, block) int64 candidate codes that the count and entropy buffers of one block take


def batched_cmi_gpu(
    x_cols: Any,
    y: np.ndarray,
    z: Any = None,
    return_cards: bool = False,
    codes_trusted: bool = False,
    return_device: bool = False,
    precomp_yz: Any = None,
    kx: int = 0,
    ky: int = 0,
    precomp_xz: Any = None,
) -> Any:
    """Miller-Madow plug-in CMI(x_k; y | z) in nats for EVERY column of ``x_cols`` (the arguments and the returns are those of ``_fe_batched_mi._batched_cmi_gpu_once``), in column blocks.

    A block is sized from the room under the calling thread's pool (``_device_blocks.first_block``: int64 codes plus the count and entropy buffers, about three copies of the block), a block that
    still runs out of memory is halved (``run_column_blocks``), and the pieces are joined in column order: values, the occupied-cell counts of the ``return_cards`` form and the resident or host
    return all match the one-shot call."""
    import cupy as cp

    from . import _fe_batched_mi as batched_mi
    from ._device_blocks import first_block, run_column_blocks

    shape = getattr(x_cols, "shape", None)
    n_cols = int(shape[1]) if shape is not None and len(shape) == 2 else 1
    n_rows = int(shape[0]) if shape is not None else 0
    if n_cols <= 1:
        return batched_mi._batched_cmi_gpu_once(x_cols, y, z, return_cards, codes_trusted, return_device, precomp_yz, kx, ky, precomp_xz)
    block = first_block(cp, n_rows, n_cols, 8, _CMI_BLOCK_COPIES)
    if block >= n_cols:
        return batched_mi._batched_cmi_gpu_once(x_cols, y, z, return_cards, codes_trusted, return_device, precomp_yz, kx, ky, precomp_xz)

    def _piece(lo: int, hi: int) -> Any:
        """One block of columns, resident."""
        return batched_mi._batched_cmi_gpu_once(x_cols[:, lo:hi], y, z, return_cards, codes_trusted, True, precomp_yz, kx, ky, precomp_xz)

    pieces = run_column_blocks(cp, n_cols, block, _piece)
    if return_cards:
        cmi_d = cp.concatenate([pc[0] for pc in pieces])
        k_xz = cp.concatenate([pc[2] for pc in pieces])
        k_xyz = cp.concatenate([pc[4] for pc in pieces])
        k_z, k_yz = pieces[0][1], pieces[0][3]
        if return_device:
            return (cmi_d, k_z, k_xz, k_yz, k_xyz)
        return (cp.asnumpy(cmi_d), k_z, cp.asnumpy(k_xz).astype(np.int64), k_yz, cp.asnumpy(k_xyz).astype(np.int64))
    merged = cp.concatenate(pieces)
    return merged if return_device else cp.asnumpy(merged)
