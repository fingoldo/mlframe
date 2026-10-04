"""Numeric member-matrix resolution for the DCD swap evaluation, split out of ``_dcd_swap`` to keep that module under the size budget."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def _resolve_member_matrix(X_raw: Any, members: list, member_names: list) -> Optional[tuple]:
    """The float64 matrix of the cluster members' raw columns with its surviving ``members`` / ``member_names``, or ``None`` when fewer than two
    usable columns remain (the swap is then declined)."""
    # Resolve raw columns; X_raw may be a DataFrame or ndarray.
    if hasattr(X_raw, "columns"):
        keep = [i for i, c in enumerate(member_names) if c in X_raw.columns]
        # A cluster can anchor on a mix of numeric and categorical columns (DCD clusters by
        # correlation/co-membership, not dtype); a raw string/object column made the blanket
        # ``to_numpy(dtype=float64)`` below raise "could not convert string to float", which the
        # broad except caught but degraded to declining the WHOLE swap even when 2+ numeric
        # members remained. Narrow to numeric-castable columns first so the swap still runs on
        # those (same "need >=2 columns" contract as the finite-value filter just below).
        keep = [i for i in keep if X_raw[member_names[i]].dtype.kind in "iuf"]
        if len(keep) < 2:
            return None
        members = [members[i] for i in keep]
        member_names = [member_names[i] for i in keep]
        M = X_raw[member_names].to_numpy(dtype=np.float64, copy=True)
    else:
        arr = np.asarray(X_raw)
        if arr.ndim != 2 or arr.shape[1] < max(members) + 1:
            return None
        M = arr[:, members].astype(np.float64, copy=True)
    # Drop columns containing NaN / Inf to keep PC1 stable. ``members``/``member_names`` are kept in
    # lockstep with M's surviving columns (they used to stay at their PRE-filter length while mean/
    # std/signs below were built from the post-filter M - a length mismatch that corrupted
    # recipe_obj["members"] for replay/naming whenever a member was actually dropped here).
    finite_mask = np.isfinite(M).all(axis=0)
    if finite_mask.sum() < 2:
        return None
    M = M[:, finite_mask]
    members = [m for m, keep_col in zip(members, finite_mask) if keep_col]
    member_names = [n for n, keep_col in zip(member_names, finite_mask) if keep_col]
    return M, members, member_names
