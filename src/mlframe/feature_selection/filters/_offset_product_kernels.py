"""CPU kernels of the offset-product operator ``(u + s) * (v + t)``.

Why the shift matters: a product ``f(c) * g(d)`` whose factor ``f`` changes sign inside the data range (``f = ln 2 + ln c`` crosses zero at ``c = 0.5``) is an interaction no
fixed unary pair can express, because the preset forms have their sign change fixed at 0. ``y ~ a + b*u + c*v + d*u*v`` is exactly ``d * (u + c/d) * (v + b/d)`` plus a constant, so one
4x4 normal-equation solve gives both shifts in closed form, with no grid.

``scan_offset_products`` is the fused scan: per (column pair, unary pair) task it fits the shifts on the even rows, then regenerates ``(u+s)*(v+t)`` on the fly (nothing is stored per
candidate) to bin it and score the plug-in MI on the even rows (the rows the shifts were fitted on) and on the odd rows (held out), next to five baselines that need no shift: ``u*v``,
``u+v``, ``u``, ``v`` and the least-squares weighted sum ``b*u + c*v`` (a product with large shifts degenerates into exactly that sum, which is no interaction). Bin edges are quantiles of a fixed-size row subsample of the even rows, so the held-out half never influences them.
"""

from __future__ import annotations

# The kernels live in ``_numba_utils`` (the shared host of every cross-module ``@njit`` helper); re-exported here for existing import paths.
from ._numba_utils import N_BASELINES, N_EDGE_SUBSAMPLE, NACC, ols2_shift, scan_offset_products, weighted_sum_heldout_mi

__all__ = ["NACC", "N_EDGE_SUBSAMPLE", "N_BASELINES", "ols2_shift", "scan_offset_products", "weighted_sum_heldout_mi"]
