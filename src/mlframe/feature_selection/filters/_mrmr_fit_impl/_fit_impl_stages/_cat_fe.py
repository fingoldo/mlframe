"""Categorical feature-engineering stages of ``_fit_impl``."""

from __future__ import annotations

# --- imports (managed) ---
import logging
import numpy as np
from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_core import logger
from mlframe.utils.log_throttle import log_throttle
# --- end imports ---


def _run_categorical_fe(
    self,
    cat_fe_cfg,
    _cat_fe_pool_size,
    data,
    target_indices,
    nbins,
    dtype,
    cols,
    categorical_vars,
    _num_raw_values,
    verbose,
    _is_polars_input,
    _x_for_cat,
    engineered_recipes,
):
    """Run categorical feature engineering and append its engineered codes; returns the updated vars, columns, data and nbins."""
    if cat_fe_cfg.enable and _cat_fe_pool_size >= 2:
        from mlframe.feature_selection.filters.cat_interactions import run_cat_interaction_step
        from mlframe.feature_selection.filters.info_theory import merge_vars as _merge_vars_for_cat_fe

        # Pre-compute classes_y / freqs_y for cat-FE (avoids re-binning the target inside every kernel call).
        _classes_y, _freqs_y, _ = _merge_vars_for_cat_fe(
            factors_data=data,
            vars_indices=target_indices,
            var_is_nominal=None,
            factors_nbins=nbins,
            dtype=dtype,
        )
        _classes_y_safe = _classes_y.copy()

        # Pull cached cat-FE state from prior fit (if any).
        _prev_cache = getattr(self, "_cat_fe_cache_", None)
        _n_cols_before_cat_fe = data.shape[1]
        data, cols, nbins, cat_fe_state = run_cat_interaction_step(
            data=data,
            cols=cols,
            nbins=nbins,
            target_indices=target_indices,
            classes_y=_classes_y,
            classes_y_safe=_classes_y_safe,
            freqs_y=_freqs_y,
            categorical_vars=categorical_vars,
            cfg=cat_fe_cfg,
            streaming_cache=_prev_cache,
            numeric_raw_values=_num_raw_values,
            dtype=dtype,
            verbose=verbose,
        )
        self._cat_fe_state_ = cat_fe_state
        # Register engineered cat features as categorical_vars so the downstream numeric-FE step excludes them
        # from numeric_vars_to_consider; without this, k-way cat engineered cols enter prospective_pairs and
        # check_prospective_fe_pairs hits KeyError reading them from X (which lacks engineered cols).
        # Engineered cat cols are appended at the end of data/cols at positions [_n_cols_before_cat_fe..].
        _n_cat_fe_added = data.shape[1] - _n_cols_before_cat_fe
        if _n_cat_fe_added > 0:
            categorical_vars = list(categorical_vars) + list(range(_n_cols_before_cat_fe, data.shape[1]))
        # Persist cache for next fit() call
        if cat_fe_state.streaming_cache_out:
            self._cat_fe_cache_ = cat_fe_state.streaming_cache_out
        # Stamp the fit-time categorical -> integer-code mapping onto every cat-FE recipe whose source columns are
        # categorical / string. Without this, ``transform`` on a raw frame routes string source values through
        # ``astype(int64)`` -> ValueError -> all-zero codes, so the carefully-discovered cat-interaction (factorize /
        # target_encoding) feature collapses to a CONSTANT column at serving time - a silent train/serve skew (the
        # FS-side analog of the 4b299e25 neural ``_apply_cat_codes`` bug). ``categorize_dataset`` codes Categorical via
        # ``.cat.codes`` (category order) and object/string via ``pd.factorize`` (first-appearance order, training-data
        # dependent); only a stored map can reproduce those codes at transform. The map is built ONCE per distinct source
        # column from the raw ``_x_for_cat`` frame and shared across recipes referencing that column.
        if cat_fe_state.recipes and not _is_polars_input and hasattr(_x_for_cat, "columns"):
            from mlframe.feature_selection.filters.engineered_recipes.shared import build_category_code_map as _build_cat_code_map

            # ``categorize_dataset`` factorises ALL categorical columns as ONE block and applies the NaN +1
            # shift to the WHOLE block when ANY column in it has a NaN. So even a NaN-FREE categorical source
            # gets its codes shifted +1 at fit time. Compute the block-level NaN flag ONCE (mirroring
            # ``categorize_dataset``'s ``select_dtypes`` block selection exactly) and thread it into every map
            # build; a per-column flag would off-by-one the NaN-free partner of a NaN-bearing column - the
            # same silent train/serve skew, for the mixed-block case the per-column path never handled.
            _block_has_nan: bool | None = None
            try:
                _cat_block = _x_for_cat.select_dtypes(include=("category", "object", "string", "bool"))
                if _cat_block.shape[1] > 0:
                    _block_has_nan = bool(_cat_block.isna().to_numpy().any())
            except Exception as exc:
                logger.debug("mrmr: NaN-block detection failed; treating as unknown: %r", exc, exc_info=True)
                _block_has_nan = None
            _src_map_cache: dict = {}
            for _ri, r in enumerate(cat_fe_state.recipes):
                _maps_for_recipe: dict = {}
                for _src in getattr(r, "src_names", ()) or ():
                    if _src not in _src_map_cache:
                        if _src in _x_for_cat.columns:
                            try:
                                _src_map_cache[_src] = _build_cat_code_map(_x_for_cat[_src], block_has_nan=_block_has_nan)
                            except Exception as exc:
                                logger.debug("mrmr: source-map cache build failed for this recipe source; treating as empty: %r", exc, exc_info=True)
                                _src_map_cache[_src] = {}
                        else:
                            _src_map_cache[_src] = {}
                    if _src_map_cache[_src]:
                        _maps_for_recipe[_src] = _src_map_cache[_src]
                if _maps_for_recipe:
                    # ``extra`` is a read-only MappingProxyType on a frozen recipe; ``with_extra`` returns a fresh copy carrying the maps.
                    try:
                        cat_fe_state.recipes[_ri] = r.with_extra(cat_code_maps=_maps_for_recipe)
                    except Exception as e:
                        # Without its train-time code maps the recipe replays categories with fresh codes at transform: silent train/serve skew.
                        log_throttle(
                            logger,
                            "mrmr_cat_code_maps_attach_failed",
                            logging.WARNING,
                            "mrmr: attaching cat_code_maps to recipe %r failed (%s: %s); its transform-time category codes may not match fit",
                            getattr(r, "name", "?"),
                            type(e).__name__,
                            e,
                        )
        # Cat-FE recipes feed the same engineered_recipes dict numeric FE uses; the fit-end splitter copies
        # any recipe whose engineered name appears in selected_vars_names into ``self._engineered_recipes_``.
        for r in cat_fe_state.recipes:
            engineered_recipes[r.name] = r
        if verbose and cat_fe_state.recipes:
            logger.info(
                "MRMR cat-FE produced %d engineered feature(s); data extended from %d to %d cols.",
                len(cat_fe_state.recipes),
                data.shape[1] - len(cat_fe_state.recipes),
                data.shape[1],
            )
    return categorical_vars, cols, data, nbins


def _collect_numeric_raw_values(self, cat_fe_cfg, categorical_vars, target_indices, cols, _include_numeric_input_nan_cols, X, _num_raw_values):
    """Collect raw numeric input values keyed by data-column index for the cat-FE numeric candidate pool."""
    if cat_fe_cfg.enable and getattr(cat_fe_cfg, "include_numeric", False):
        from mlframe.feature_selection.filters.engineered_recipes.shared import extract_column as _extract_col_for_num

        _cat_idx_set = set(int(c) for c in categorical_vars)
        _tgt_idx_set = set(int(t) for t in target_indices)
        # RAW input columns only: pre-FE recipes (haar / ratio / grouped-agg ...) appended engineered numeric
        # columns to data / cols / X before this step. Crossing those is unreplayable - the engineered source
        # is absent from the user's raw frame at transform time -> NaN column / silent feature drop. Restrict to
        # ``feature_names_in_`` (the raw user columns, set above, excludes engineered names).
        # feature_names_in_ is an ndarray; "or []" would test truthiness and raise on a multi-element array.
        _fni_raw = getattr(self, "feature_names_in_", None)
        _raw_name_set = set(_fni_raw) if _fni_raw is not None else set()
        _num_raw_values = {}
        for _ci in range(len(cols)):
            if _ci in _cat_idx_set or _ci in _tgt_idx_set:
                continue
            if _raw_name_set and cols[_ci] not in _raw_name_set:
                continue
            # Skip columns the user supplied with NaN (snapshot at fit entry, robust to any downstream impute):
            # the quantile-edge replay has no NaN bin, so crossing them would skew serving.
            if cols[_ci] in _include_numeric_input_nan_cols:
                continue
            try:
                _num_raw_values[_ci] = np.asarray(_extract_col_for_num(X, cols[_ci]))
            except Exception as e:  # nosec B112 - swallow converted to debug-log, non-fatal by design
                logger.debug("mrmr: extracting raw numeric column %r values failed: %r", cols[_ci], e, exc_info=True)
                continue
    return _num_raw_values
