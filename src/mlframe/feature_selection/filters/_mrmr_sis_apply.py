"""Layer A-gate — SIS front-screen application for ``MRMR`` (carved out of the class body).

``MRMR._apply_sis_screen(X, y)`` is implemented here as a module-level function taking ``self`` as its
first argument and bound onto the class in ``mrmr/__init__.py`` the same way ``_fit_impl`` / ``_run_fe_step``
are. Living in a sibling shrinks the LOC-exempt ``_mrmr_class.py`` estimator class body. The body is moved
VERBATIM (no logic / threshold / RNG change); only the leading indentation level was removed. The downstream
selection path is byte-for-byte identical -- this is pure I/O + column subsetting around the standalone
``sis_screen`` kernel, not a kernel rewrite.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")


def _apply_sis_screen(self, X, y):
    """Gate A: run the chunked SIS screen and subset ``X`` to the survivor columns.

    Returns a column-subset of ``X`` (same container type: pandas / polars / numpy). Reuses the
    standalone ``sis_screen`` (filters/_mrmr_sis_screen) -- no kernel logic here, just I/O + subsetting.
    ``k_target`` feeds the survivor floor (the requested number of finally-selected features)."""
    from ._mrmr_sis_screen import sis_screen  # filters-level sibling (was ``from .._mrmr_sis_screen`` in the class body)

    # ndarray view of X for the screen (the screen reads it in column blocks; a memmap stays on disk).
    # NON-NUMERIC SAFETY (2026-06-19, critique P0-2): the screen casts each column block to float32, so any
    # string/object/categorical column would raise and the outer try would SILENTLY fall back to full-width
    # MRMR (defeating the gate). Factorise non-numeric columns to integer codes here so categoricals are
    # SCORED by the marginal-MI channel (codes are valid MI input) rather than crashing. Numeric columns
    # pass through unchanged (float). The 2nd-moment channel will score nominal codes ~uninformatively, which
    # is acceptable -- categorical interaction is not what that channel targets.
    def _numeric_matrix(df):
        """Coerce ``df`` to an all-float ndarray for the SIS scoring channels: numeric/bool-excluded columns pass through as float64, non-numeric columns are factorized to integer codes so they're scored (not silently dropped or crashed on)."""
        cols_out = []
        for c in df.columns:
            s = df[c]
            if pd.api.types.is_numeric_dtype(s.dtype) and s.dtype != bool:
                cols_out.append(np.asarray(s.to_numpy(), dtype=np.float64))
            else:
                cols_out.append(pd.factorize(s, sort=False)[0].astype(np.float64))
        return np.column_stack(cols_out) if cols_out else np.empty((len(df), 0))

    if isinstance(X, pd.DataFrame):
        Xmat = X.to_numpy()
        if Xmat.dtype.kind in "USO" or Xmat.dtype == object:  # mixed/object frame -> factorise per column
            Xmat = _numeric_matrix(X)
    elif str(type(X).__module__).startswith("polars"):
        Xmat = X.to_numpy()
        if Xmat.dtype.kind in "USO" or Xmat.dtype == object:
            Xmat = _numeric_matrix(X.to_pandas())
    else:
        Xmat = np.asarray(X)
        if Xmat.dtype.kind in "USO" or Xmat.dtype == object:  # object ndarray -> factorise per column
            Xmat = _numeric_matrix(pd.DataFrame(Xmat))

    k_target = getattr(self, "n_features", None)
    try:
        k_target = int(k_target) if k_target is not None else None
    except (TypeError, ValueError):
        k_target = None

    # return_scores=True is FREE (the scores are already computed for survivor selection). We STASH the
    # survivor marginal-MI as a relevance prior so the screen's most expensive output is no longer discarded
    # (reuse audit RU-2). NB it is NOT fed into screen_predictors' cached_MIs as a warm-start: SIS bins
    # quantile-nbins-10 on RAW columns BEFORE categorize, whereas screen_predictors scores MI on categorize's
    # (default MDLP) codes -- the two MI values differ, so substituting would CHANGE selection. The recompute
    # it would save is ~3.6s at 100k (CK kernel audit) -- second-order vs the ~290s Fleuret CMI loop (CK-1) --
    # so the cached_MIs warm-start is deferred behind CK-1 rather than destabilising the 900-line screen for
    # ~1%. The prior is exposed for diagnostics / a future binning-aligned warm-start.
    survivors, _sis_scores = sis_screen(
        Xmat, y, k_target=k_target,
        dedup_corr_thr=float(getattr(self, "sis_dedup_corr_thr", 0.92) or 0.0),
        return_scores=True,
    )
    survivors = np.asarray(survivors, dtype=np.int64)
    _mi_full = _sis_scores.get("mi")
    self.sis_relevance_prior_ = {int(s): float(_mi_full[int(s)]) for s in survivors} if _mi_full is not None else {}
    self.sis_survivors_ = survivors
    self.sis_n_input_features_ = int(Xmat.shape[1])
    logger.info(
        "[MRMR] SIS front gate: %d -> %d survivors (k_target=%s)",
        int(Xmat.shape[1]), int(survivors.size), k_target,
    )

    # Subset in the caller's container type so the downstream path is identical to a natively narrow frame.
    if isinstance(X, pd.DataFrame):
        return X.iloc[:, survivors]
    if str(type(X).__module__).startswith("polars"):
        return X[:, survivors.tolist()]
    return X[:, survivors]
