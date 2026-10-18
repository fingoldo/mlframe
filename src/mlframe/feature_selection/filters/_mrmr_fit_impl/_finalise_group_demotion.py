"""Group-aware FE demotion stage of ``_finalise_fs_results``."""

from __future__ import annotations

import numpy as np


def _demote_group_leaks_final(self, cols, classes_y, data, nbins, X, verbose):
    """Group-aware FE demotion final choke point, split out of ``_finalise_fs_results`` to keep it under its length ceiling."""
    # Imported here, not at module top: ``_finalise`` imports this module, so a top-level import would be circular.
    from ._finalise import logger

    try:
        from mlframe.feature_selection.filters.info_theory.shared import get_group_mi as _get_group_mi_final
        _gmi_final_payload = _get_group_mi_final()
    except Exception as e:
        logger.warning("get_group_mi_final() failed: %s", e)
        _gmi_final_payload = None
    _eng_recipes_final = getattr(self, "_engineered_recipes_", None) or []
    if _gmi_final_payload is not None and _eng_recipes_final:
        try:
            from mlframe.feature_selection.filters.info_theory.shared import group_leak_verdict as _group_leak_verdict_final
            from mlframe.feature_selection.filters.info_theory.shared import group_relevance_mi as _group_relevance_mi_final

            _cols_idx_f = {nm: i for i, nm in enumerate(cols)}
            _gsi_f, _goff_f, _gmr_f, _gsw_f = _gmi_final_payload
            _classes_y_arr_f = np.asarray(classes_y)
            _g_n_bins_y_f = int(_classes_y_arr_f.max()) + 1
            _group_dropped_final: set = set()
            for _recipe in _eng_recipes_final:
                _rname = str(getattr(_recipe, "name", ""))
                _cidx = _cols_idx_f.get(_rname)
                if _cidx is not None:
                    # Fast path: the engineered column is still materialised in ``data``/``cols``.
                    _grp_obs_f = _group_relevance_mi_final(
                        data, (int(_cidx),), _classes_y_arr_f, np.asarray(nbins), _g_n_bins_y_f,
                        _gsi_f, _goff_f, min_rows=_gmr_f, size_weighted=_gsw_f, dtype=self.quantization_dtype,
                    )
                    _is_leak_f = False
                    if _grp_obs_f == 0.0:
                        _is_leak_f = True
                    elif _grp_obs_f == _grp_obs_f:  # NaN is neither a leak nor a verdict, so the recipe is kept
                        _is_leak_f = _group_leak_verdict_final(
                            np.asarray(data[:, int(_cidx)]), _classes_y_arr_f, _gsi_f, _goff_f,
                            int(np.asarray(data[:, int(_cidx)]).max()) + 1, _g_n_bins_y_f, min_rows=_gmr_f, size_weighted=_gsw_f,
                        )[1]
                else:
                    # Retention-only recipe (usability-aware retention re-attaches it from its OWN
                    # recompute, bypassing ``cols``/``data`` entirely - see the comment above). Replay
                    # the SAME recompute + discretise the retention pass itself uses, then group-block
                    # directly (a single already-discretised column needs no ``merge_vars`` combination).
                    try:
                        from mlframe.feature_selection.filters.engineered_recipes.shared import apply_recipe as _apply_recipe_final
                        from mlframe.feature_selection.filters.discretization import discretize_array as _discretize_array_final
                        from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype

                        _cv_f = np.asarray(_apply_recipe_final(_recipe, X), dtype=_crit_np_dtype()).ravel()
                        _cv_f = np.nan_to_num(_cv_f, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                        if _cv_f.shape[0] != _classes_y_arr_f.shape[0]:
                            continue  # can't align - leave this recipe alone
                        _codes_f = _discretize_array_final(_cv_f, n_bins=int(self.quantization_nbins), method=self.quantization_method, dtype=self.quantization_dtype)
                        _codes_mat_f = np.ascontiguousarray(np.asarray(_codes_f, dtype=self.quantization_dtype).reshape(-1, 1))
                        _nbins_mat_f = np.asarray([int(_codes_mat_f.max()) + 1])
                        _grp_obs_f = _group_relevance_mi_final(
                            _codes_mat_f, (0,), _classes_y_arr_f, _nbins_mat_f, _g_n_bins_y_f,
                            _gsi_f, _goff_f, min_rows=_gmr_f, size_weighted=_gsw_f, dtype=self.quantization_dtype,
                        )
                        _is_leak_f = False
                        if _grp_obs_f == 0.0:
                            _is_leak_f = True
                        elif _grp_obs_f == _grp_obs_f:  # NaN is neither a leak nor a verdict, so the recipe is kept
                            _is_leak_f = _group_leak_verdict_final(
                                _codes_f, _classes_y_arr_f, _gsi_f, _goff_f,
                                int(_nbins_mat_f[0]), _g_n_bins_y_f, min_rows=_gmr_f, size_weighted=_gsw_f,
                            )[1]
                    except Exception as e:
                        logger.debug("_fit: within-group MI replay/discretise failed for recipe %r, leaving it alone: %s", _rname, e)
                        continue  # can't replay/discretise - leave this recipe alone (conservative)
                # No within-group signal beyond the within-group permutation null (see group_leak_verdict).
                if _is_leak_f:
                    _group_dropped_final.add(_rname)
            if _group_dropped_final:
                self._engineered_recipes_ = [_r for _r in _eng_recipes_final if str(getattr(_r, "name", "")) not in _group_dropped_final]
                _eng_features_final = getattr(self, "_engineered_features_", None) or []
                self._engineered_features_ = [_fn for _fn in _eng_features_final if str(_fn) not in _group_dropped_final]
                self.n_features_ = (
                    len(self.support_) + len(self._engineered_recipes_)
                    if hasattr(self, "support_")
                    else int(getattr(self, "n_features_", 0)) - len(_group_dropped_final)
                )
                if verbose:
                    logger.info(
                        "MRMR: demoted %d engineered feature(s) with zero within-group relevance MI "
                        "under group_aware_mi (between-group-only leak signature reaching the final selection): %s",
                        len(_group_dropped_final), sorted(_group_dropped_final),
                    )
        except Exception as _group_final_exc:
            logger.debug(
                "MRMR group-aware FE demotion (final choke point) failed (%s: %s); keeping the naive-MI selection.",
                type(_group_final_exc).__name__, _group_final_exc,
            )

    # The demotion also removed names from ``_engineered_features_``; re-reconcile the public rosters so none still lists a demoted column.
