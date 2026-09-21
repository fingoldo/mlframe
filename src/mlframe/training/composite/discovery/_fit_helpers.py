"""Self-contained helpers for ``fit`` carved out of ``_fit.py`` to keep that module under the 1000-LOC house limit.
Each takes the discovery instance explicitly and is imported back into ``_fit``.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def no_base_candidates_report_entry() -> list[dict]:
    """A single diagnostic ``report_`` entry emitted when every base candidate was filtered out, so a caller inspecting
    ``report_`` sees WHY discovery produced nothing (not an ambiguous empty list)."""
    return [{
        "name": "__no_base_candidates__",
        "kept": False,
        "rejected": True,
        "reason": "no usable base candidates: all excluded by forbidden-pattern / corr / ptp / numeric filters",
        "base_column": "",
        "transform_name": "",
        "mi_gain": float("nan"),
        "valid_domain_frac": float("nan"),
    }]


def maybe_boost_mi_strata_for_heavy_tail(self, y_train: np.ndarray) -> None:
    """Auto-boost ``mi_n_strata`` on a heavy-tail train target (skew/kurtosis), in place on ``self.config``.

    The default 10 strata gives unstable MI estimates when the tail dominates (one or two tail rows per bin); when
    skew/kurtosis is high, bump to ``mi_n_strata_heavy_tail`` (default 30) -- but only when the boost is HIGHER than
    the user-configured floor, and only on a per-target config clone (model_copy) so callers' shared config never leaks.
    """
    y_finite_for_check = y_train[np.isfinite(y_train)]
    if y_finite_for_check.size < 100:
        return
    y_std = float(y_finite_for_check.std())
    if y_std <= 1e-12:
        return
    z_centered = (y_finite_for_check - y_finite_for_check.mean()) / y_std
    # z**3 / z**4 via chained mul (faster than np.power dispatch; same antipattern as the target-distribution analyzer).
    z2 = z_centered * z_centered
    skew = float(np.mean(z2 * z_centered))
    kurt = float(np.mean(z2 * z2) - 3.0)
    if not (abs(skew) > 2.0 or kurt > 5.0):
        return
    boost = int(getattr(self.config, "mi_n_strata_heavy_tail", 30))
    cur_n_strata = int(getattr(self.config, "mi_n_strata", 10))
    if boost <= cur_n_strata:
        return
    try:
        self.config = self.config.model_copy(update={"mi_n_strata": boost})
        logger.info(
            "[CompositeTargetDiscovery] heavy-tail y detected (skew=%.2f, kurt=%.2f); boosted mi_n_strata %d -> %d.",
            skew, kurt, cur_n_strata, boost,
        )
    except Exception as e:  # nosec B110 - optional/best-effort path, rationale documented
        logger.debug("heavy-tail y skew/kurt boost computation failed, leaving mi_n_strata at user-configured value: %s", e)


def take_screen_matrix(self, df, columns, rows) -> np.ndarray:
    """The screening feature matrix for ``rows``, reusing the one auto-base already gathered when it covers them.

    Auto-base draws the same screening sample and gathers the same columns before ``fit`` does; ``fit`` only reorders the
    rows when a time key is given. The stashed matrix is reused when the frame object and the column list are the same
    and its rows are the same set, permuted into ``fit``'s order; anything else gathers from the frame as before. The
    stash is released either way, since it can be a few hundred MB on a wide screen.
    """
    stash = getattr(self, "_screen_matrix_stash", None)
    self._screen_matrix_stash = None
    rows = np.asarray(rows)
    if stash is not None:
        s_df, s_cols, s_rows, s_matrix = stash
        if s_df is df and s_cols == tuple(columns) and s_rows.shape == rows.shape:
            if np.array_equal(s_rows, rows):
                return s_matrix
            order = np.argsort(s_rows, kind="stable")
            pos = order[np.clip(np.searchsorted(s_rows[order], rows), 0, s_rows.size - 1)]
            if np.array_equal(s_rows[pos], rows):
                return s_matrix[pos]
    return self._build_feature_matrix(df, columns, rows)
