"""Verbatim-carved ``_fit_body`` block helpers of ``MRMR`` (restore, partial-fit reset, budgets, caches, cleanup) as a mixin.

``MRMR._fit_body`` calls these through ``self``; they resolve via the MRO of the concrete class. ``_safe_restore`` lives here
(it has no dependency on the class) and is re-exported from ``_mrmr_class`` for its other callers and tests.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")


def _safe_restore(action: Callable[[], Any], description: str) -> None:
    """Run a single ``fit()``-finally restore action, swallowing any exception it raises into a debug
    log line instead of letting a failed restore mask the fit's real outcome (success or a genuine
    error) or abort the remaining restores. Replaces 11 near-identical
    ``try: ... except Exception as e: logger.debug("suppressed in _mrmr_class.py:<N>: %s", e)`` blocks
    that each hardcoded their own source line number - numbers which drifted out of sync with the
    actual line on every subsequent edit, silently giving false diagnostics forever once stale.
    ``description`` is a short human-readable label instead, which cannot go stale the same way."""
    try:
        action()
    except Exception as exc:  # nosec B110 - a restore-step failure must never break/mask the fit's real outcome
        logger.debug("mrmr: fit()-finally restore failed (%s): %r", description, exc, exc_info=True)


class _MRMRFitBodyBlocksMixin:
    """Mixin carrying the small ``_fit_body`` block helpers; see the module docstring."""

    def _fit_body_clone_pickle_repeated_fit(self, _default_screen_saved):
        """Block of _fit_body starting at ``if _default_screen_saved:``."""
        if _default_screen_saved:

            def _make_default_screen_restorer(_k: str, _v: Any) -> Callable[[], None]:
                """Bind (_k, _v) into a niladic restorer so the loop variables are captured by value, not by reference."""
                return lambda: setattr(self, _k, _v)

            for _k, _v in _default_screen_saved.items():
                _safe_restore(_make_default_screen_restorer(_k, _v), f"default-screen-subsample knob {_k!r}")

    def _fit_body_getattr_self_partial_fit(self):
        """Block of _fit_body starting at ``if not getattr(self, "_in_partial_fit_", False):``."""
        if not getattr(self, "_in_partial_fit_", False):
            # An explicit fit() replaces the model, so a partial_fit stream restarts from its next batch; resuming the old buffer made partial_fit(A); fit(B);
            # partial_fit(C) silently refit on A + C.
            for _pf_attr in (
                "_partial_fit_X_buffer_",
                "_partial_fit_y_buffer_",
                "_partial_fit_batch_sizes_",
                "_partial_fit_n_seen_",
                "_partial_fit_n_since_refit_",
            ):
                if getattr(self, _pf_attr, None) is not None:
                    setattr(self, _pf_attr, None)

    def _fit_body_isinstance_fe_budget_setting(self, _fe_budget_setting, X, _fe_budget_learning_effective, _fe_loaded_budgets):
        """Block of _fit_body starting at ``if isinstance(_fe_budget_setting, str):``."""
        if isinstance(_fe_budget_setting, str):
            if _fe_budget_setting != "auto":
                raise ValueError(f"fe_budget_learning must be True, False, or 'auto'; got {_fe_budget_setting!r}")
            try:
                from mlframe.feature_selection.filters._fe_family_budget import dataset_fingerprint as _fe_budget_fp, load_budgets as _fe_load_budgets

                _fe_budget_cols = list(X.columns) if hasattr(X, "columns") else [str(i) for i in range(np.asarray(X).shape[1])]
                self._fe_budget_fingerprint_ = _fe_budget_fp(len(_fe_budget_cols), _fe_budget_cols)
                _fe_loaded_budgets = _fe_load_budgets(fingerprint=self._fe_budget_fingerprint_)
                # "auto" fires ONLY when a previous explicit opt-in already persisted a budget for this
                # exact dataset fingerprint - otherwise it is a strict no-op, identical to False.
                _fe_budget_learning_effective = _fe_loaded_budgets is not None
            except Exception as _fe_budget_probe_exc:
                logger.warning("[MRMR] fe_budget_learning='auto': cache probe failed (%s); treating as disabled this fit.", _fe_budget_probe_exc)
                _fe_budget_learning_effective = False
        return _fe_budget_learning_effective, _fe_loaded_budgets

    @staticmethod
    def _fit_body_int_paths_get_validated(_y_check):
        """Block of _fit_body starting at ``if _y_check.dtype.kind in "fc":``."""
        if _y_check.dtype.kind in "fc":
            _n_nan = int(np.isnan(_y_check).sum())
            _n_inf = int(np.isinf(_y_check).sum())
            if _n_nan or _n_inf:
                raise ValueError(
                    f"MRMR.fit: y contains {_n_nan} NaN and {_n_inf} +/-inf values. "
                    f"MI estimation silently degrades on NaN; drop or impute these rows "
                    f"before fitting."
                )

    @staticmethod
    def _fit_body_ycorr_thr(_ycorr_thr, _prior_y_sample, y, _measured_corr, _ycorr_ok):
        """Block of _fit_body starting at ``if _ycorr_thr > 0.0:``."""
        from .._mrmr_fingerprints import _mrmr_y_corr, _mrmr_y_corr_sample  # lazy: keeps this leaf module out of the package import cycle

        if _ycorr_thr > 0.0:
            if _prior_y_sample is not None:
                _measured_corr = _mrmr_y_corr(_mrmr_y_corr_sample(y), _prior_y_sample)
                # NaN corr (constant sample / mismatched length) is treated as "cannot confirm" -> refuse.
                _ycorr_ok = _measured_corr is not None and abs(_measured_corr) >= _ycorr_thr
            else:
                # The user asked for the y-correlation safety gate (thr > 0) but the cached entry is the
                # legacy bool format with no prior y-sample to check against - we cannot confirm the new
                # target matches the one that produced the cached identity selection. Refuse the shortcut
                # and run a full fit rather than emit a selection that never saw this y.
                _ycorr_ok = False
        return _measured_corr, _ycorr_ok

    def _fit_body_best_effort_any_failure(self, X, y, _sis_input_space_info):
        """Block of _fit_body starting at ``try:``."""
        try:
            _sis_thr = int(getattr(self, "sis_screen_threshold", 0) or 0)
            _p_in = int(X.shape[1]) if hasattr(X, "shape") and getattr(X, "ndim", 1) > 1 else 0
            if _sis_thr and _p_in >= _sis_thr:
                from mlframe.feature_selection.filters._mrmr_sis_apply import _sis_input_space

                _sis_space = _sis_input_space(X)
                X = self._apply_sis_screen(X, y)
                _sis_input_space_info = _sis_space
        except Exception as _sis_exc:
            warnings.warn(
                f"MRMR SIS front gate raised {type(_sis_exc).__name__}: {_sis_exc}; falling back to the "
                "full-width MRMR path (safe -- the screen is a fast-path optimisation, not a correctness "
                "requirement; the fit still runs, just without the pre-screen speedup).",
                UserWarning,
                stacklevel=2,
            )
        return X, _sis_input_space_info

    def _fit_body_their_package_default_shrunk(self, X, _default_screen_saved):
        """Block of _fit_body starting at ``try:``."""
        try:
            _n_rows_for_screen = int(X.shape[0]) if hasattr(X, "shape") else None
            if _n_rows_for_screen is not None:
                _default_screen_saved = self._apply_default_screen_subsample(_n_rows_for_screen)
        except Exception as exc:
            logger.warning("mrmr: default screen-subsample application failed (%s: %s); screening at full n instead (slower than the subsampled screen)", type(exc).__name__, exc, exc_info=True)
            _default_screen_saved = {}
        return _default_screen_saved

    def _fit_body_these_alone_because_resolved(self, _fast_search_saved, _fast_on, _fast_values):
        """Block of _fit_body starting at ``for _auto_attr, _pkg_value in self._AUTO_KNOB_DEFAULTS:``."""
        for _auto_attr, _pkg_value in self._AUTO_KNOB_DEFAULTS:
            if getattr(self, _auto_attr, None) is None:
                _fast_search_saved[_auto_attr] = None
                setattr(self, _auto_attr, _fast_values[_auto_attr] if (_fast_on and _auto_attr in _fast_values) else _pkg_value)

    def _fit_body_fast(self, _fast_on, _fast_search_saved):
        """Block of _fit_body starting at ``if _fast_on:``."""
        if _fast_on:
            try:
                _fast_search_saved.update(self._apply_fast_search_profile())
            except Exception as exc:
                # No reset of the saved map: it already holds the auto-knob entries above, which still need restoring.
                logger.debug("mrmr: fast-search profile application failed; using unmodified knobs: %r", exc, exc_info=True)

    def _fit_body_freshly_constructed_instance_never(self, _skip_shape, _orig_skip_content):
        """Block of _fit_body starting at ``if _skip_shape is not None:``."""
        if _skip_shape is not None:
            _eff_skip = bool(_skip_shape)
            if _eff_skip != getattr(self, "skip_retraining_on_same_content", True):
                _orig_skip_content = getattr(self, "skip_retraining_on_same_content", True)
                self.skip_retraining_on_same_content = _eff_skip
        return _orig_skip_content

    def _fit_body_stash_fingerprint_identity_bool(self, _identity_skip, _x_fp, X, y, _cache_dict):
        """Block of _fit_body starting at ``if _identity_skip and _x_fp is not None:``."""
        from .._mrmr_fingerprints import _MRMR_IDENTITY_FP_LOCK, _mrmr_y_corr_sample  # lazy: keeps this leaf module out of the package import cycle

        if _identity_skip and _x_fp is not None:
            try:
                _is_id = (
                    getattr(self, "support_", None) is not None
                    and len(self.support_) == X.shape[1]
                    and len(getattr(self, "_engineered_features_", []) or []) == 0
                )
                # Store (is_id, y_sample) so a later candidate hit can apply the y-correlation gate
                # (A1-06): the "prior identity implies new identity" assumption only holds for a target
                # correlated with the one that produced this entry. y_sample is a deterministic strided
                # numeric sample (cheap; never the full y). The value stays back-compatible: legacy bool
                # readers still see truthiness, and the read path below handles both bool and tuple.
                with _MRMR_IDENTITY_FP_LOCK:
                    _cache_dict[_x_fp] = (bool(_is_id), _mrmr_y_corr_sample(y) if _is_id else None)
                if _is_id:
                    # Remember which (X, y) fingerprint THIS instance's own real fit produced, so a
                    # later self-refit on the identical pair defers to _fit_impl's precise same-instance
                    # shortcuts instead of the coarse cross-target identity-shortcut above.
                    self._own_last_identity_fp_ = _x_fp
                    logger.info(
                        "[MRMR] cross-target identity cache STORED for X fingerprint=%s "
                        "(no features dropped, no engineered features); subsequent "
                        "fits on this X will short-circuit.",
                        _x_fp,
                    )
            except Exception as exc:
                logger.debug("mrmr: cross-target identity-cache store failed (optimisation only): %r", exc, exc_info=True)

    def _fit_body_fast_search_saved(self, _fast_search_saved):
        """Block of _fit_body starting at ``if _fast_search_saved:``."""
        from .._fourier_detect_cap import clear_fourier_detect_cap, set_fourier_detect_cap  # lazy: keeps this leaf module out of the package import cycle

        if _fast_search_saved:

            def _restore_fast_search_knob(_k: str, _v: Any) -> None:
                """Restore one saved fast-search knob (plain attr or the Fourier-detect cap)."""
                if _k == "__fdcap__":
                    if _v is None:
                        clear_fourier_detect_cap()
                    else:
                        set_fourier_detect_cap(_v)
                else:
                    setattr(self, _k, _v)

            def _make_fast_search_restorer(_k: str, _v: Any) -> Callable[[], None]:
                """Bind (_k, _v) into a niladic restorer so the loop variables are captured by value, not by reference."""
                return lambda: _restore_fast_search_knob(_k, _v)

            for _k, _v in _fast_search_saved.items():
                _safe_restore(_make_fast_search_restorer(_k, _v), f"fast-search knob {_k!r}")

    @staticmethod
    def _fit_body_frame_none_names(frame, names):
        """Block of _fit_body starting at ``if frame is not None and names:``."""
        if frame is not None and names:
            # Drop only columns that actually exist (success path already removed them).
            present = [c for c in names if c in frame.columns]
            if present:
                # Restore the caller's frame in place (it stored this exact object for cleanup); option_context
                # silences the conservative SettingWithCopy heuristic on a possibly-viewed frame, no copy.
                def _drop_target_cleanup_columns() -> None:
                    """Drop the temporary targ_* columns this fit injected into the caller's own frame."""
                    with pd.option_context("mode.chained_assignment", None):
                        # `inplace=True` is required here, not a habit: this must mutate the caller's stored frame OBJECT
                        # by identity, and rebinding a local would silently not clean up the caller's actual frame.
                        frame.drop(columns=present, inplace=True)  # noqa: PD002 - must mutate the caller's stored frame object by identity

                _safe_restore(_drop_target_cleanup_columns, "target-cleanup column drop on caller frame")
