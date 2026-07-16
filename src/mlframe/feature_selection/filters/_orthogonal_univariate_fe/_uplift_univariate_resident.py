"""DEVICE-BORN orth-univariate MI-uplift scorer (the SF1b :311 H2D collapse, 2026-06-30).

``score_features_by_mi_uplift`` is the shared HOST scorer that ranks each engineered column by its MI uplift
vs its raw source column. It accepts an already-materialised host ``engineered_X`` DataFrame and scores it with
``mi_classif_batch_chunked`` (the engineered matrix) + ``_mi_classif_batch`` (the raw baseline). Under
``MLFRAME_FE_GPU_STRICT`` both route through the resident plug-in MI, so the WHOLE host engineered matrix is
``cp.asarray``-uploaded at ``_orth_mi_backends.py:311`` (the SF1 share of a 300k STRICT F2 byte-audit:
~72 MB = 64 MB engineered + 8 MB raw).

When EVERY engineered column name parses to a poly basis leg (``"{src}__{code}{degree}"`` with
``code in {He, T, L, LL}``), the engineered matrix is a stack of arity-1 orthogonal-polynomial legs the device
batched Clenshaw evaluator supports. This module rebuilds that matrix ON the device from the small raw operand
columns (uploaded once via the resident-operand cache) and scores it -- plus the raw baseline -- through the
SAME percentile-edge resident plug-in MI ``score_features_by_mi_uplift`` already uses under STRICT, so the host
engineered matrix is never materialised/uploaded. It REUSES the shipped device-born cross-basis builder
``_gpu_resident_cross_basis.build_leg_product_matrix_gpu`` (arity-1 specs) + the resident MI; the cross-basis
twin is the arity>=2 sibling of this scorer.

EXTRA-BASIS columns (spline ``__sp`` / Fourier ``__sin`` ``__cos`` / chirp ``__qsin`` ``__qcos`` / wavelet)
are NOT GPU-ported by the basis evaluator -- when ANY engineered column is one, this returns ``None`` and the
caller keeps the engineered matrix on the host path (irreducible born-fresh transient, SF1c). The raw baseline
is collapsed separately by the class-B ``_resident_raw_mi`` route regardless.

PARITY (selection-equivalence): the device backward-Clenshaw matches the host forward recurrence to ~1e-12 at
the default low degrees; BOTH the engineered matrix AND the raw baseline route through the SAME resident
estimator so the uplift RATIO ``engineered_mi / baseline_mi`` is internally consistent (no estimator switch
that could flip selection). GATE: ``fe_gpu_device_born_uplift_univariate_enabled`` (DEFAULT ON under STRICT,
opt-out ``MLFRAME_FE_GPU_DEVICE_BORN_UPLIFT_UNIVARIATE=0``). On ANY cupy error / no cupy / a non-poly column it
returns ``None`` -> the EXACT host scorer (byte-identical default path untouched). NEVER ``free_all_blocks``.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["uplift_univariate_eng_mi_resident", "_specs_from_engineered_names"]

_POLY_CODES = ("LL", "He", "T", "L")  # longest-first so "LL" is not shadowed by "L"
# Engineered-name basis CODE -> the basis NAME the device evaluator expects (the inverse of
# ``_orthogonal_univariate_fe._BASIS_CODE``). The name carries the EXACT basis the host generator chose, so the
# device must reproduce THAT basis rather than re-deriving from moments (which can route differently).
_CODE_TO_BASIS = {"He": "hermite", "LL": "laguerre", "T": "chebyshev", "L": "legendre"}


def _specs_from_engineered_names(eng_names, raw_cols):
    """Parse each engineered column name ``"{src}__{code}{degree}"`` into an arity-1 leg spec
    ``{"legs": [(src, degree, basis_name)]}`` aligned 1:1 with ``eng_names`` (the basis is pinned from the name's
    code so the device reproduces the host's EXACT basis, not a moment re-route). Returns ``None`` when ANY name
    does NOT parse to a poly leg (an extra-basis emit / a source not in ``raw_cols``) -- the signal that the
    device builder cannot reproduce the matrix and the caller must keep it on the host."""
    from . import _source_from_engineered_name

    raw_set = set(raw_cols)
    specs = []
    for name in eng_names:
        # Pass the already-built set (not raw_cols) -> _source_from_engineered_name's O(1) fast path.
        src = _source_from_engineered_name(name, raw_set)
        if src not in raw_set:
            return None
        if not name.startswith(src + "__"):
            return None
        suffix = name[len(src) + 2 :]
        deg = None
        basis_name = None
        for code in _POLY_CODES:
            if suffix.startswith(code):
                rest = suffix[len(code) :]
                if rest.isdigit():
                    deg = int(rest)
                    basis_name = _CODE_TO_BASIS[code]
                break
        if deg is None:
            return None  # extra-basis emit (spline / Fourier / chirp / wavelet) -> host path.
        specs.append({"legs": [(src, deg, basis_name)]})
    return specs


def uplift_univariate_eng_mi_resident(
    raw_X: pd.DataFrame,
    engineered_X: pd.DataFrame,
    y: Any,
    *,
    nbins: int,
    basis: str = "auto",
) -> Optional[np.ndarray]:
    """DEVICE-BORN twin of the ENGINEERED-matrix MI call inside ``score_features_by_mi_uplift``
    (``mi_classif_batch_chunked(engineered_X)``).

    Rebuilds the engineered poly-leg matrix ON the device from the small resident raw operand columns
    (collapsing the host engineered-matrix upload at ``_orth_mi_backends.py:311``) and scores it through the
    SAME percentile-edge resident plug-in MI the host STRICT path uses. Returns the (K,) host float64 MI array
    in ``engineered_X.columns`` order, OR ``None`` when STRICT-residency is off / cupy is unavailable / any
    column is not a poly leg / any cupy fault -- in which case the caller keeps the engineered matrix on the
    EXACT host ``mi_classif_batch_chunked`` scorer (byte-identical default path untouched)."""
    try:
        from .._gpu_strict_fe import fe_gpu_device_born_uplift_univariate_enabled

        if not fe_gpu_device_born_uplift_univariate_enabled():
            return None
    except Exception:
        return None
    if engineered_X is None or engineered_X.shape[1] == 0:
        return None
    specs = _specs_from_engineered_names(list(engineered_X.columns), list(raw_X.columns))
    if specs is None:
        return None  # extra-basis / unparseable -> host path (SF1c irreducible transient).
    try:
        import cupy as cp

        from ._gpu_resident_cross_basis import _resident_mi, build_leg_product_matrix_gpu

        mat_gpu = build_leg_product_matrix_gpu(cp, raw_X, specs, basis=basis)
        if mat_gpu.shape[1] != engineered_X.shape[1]:
            return None  # spec / column-count mismatch -> host scorer (never emit a misaligned ratio).
        return _resident_mi(cp, mat_gpu, y, int(nbins))
    except Exception as _exc:
        logger.debug("uplift_univariate_eng_mi_resident: GPU path failed (%s); host fallback", _exc)
        return None
