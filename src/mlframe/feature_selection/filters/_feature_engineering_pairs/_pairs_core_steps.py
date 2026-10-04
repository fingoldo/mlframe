"""Helpers carved out of ``_pairs_core`` to keep that module under its size budget."""

from __future__ import annotations


import numba
import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionDtype

# A column is near-constant when its std is within this factor of machine epsilon of its largest |value|: its spread is then rounding noise.
_DEGENERATE_REL_TOL = 32.0 * np.finfo(np.float64).eps


from ._pairs_core_helpers import (  # noqa: F401  -- carved helpers
    _short_fe_name,
    _check_prospective__use_subsample,
    _check_prospective__behaviour_change_wrapped_experimental,
    _check_prospective__scores_same_rows_fall,
    _check_prospective__shared_idx_none,
    _check_prospective__total_get_proportions_matching,
    _check_prospective__win_do_re_implement,
    _check_prospective__ignored_thread_multiplication_byte,
    _check_prospective__classes_codes_still_usable,
    _check_prospective__only_worth_chunking_least,
    _check_prospective__read_weakref_cache_no,
    _check_prospective__pair_res_entry_none,
    _check_prospective__no_config_nan_edge,
    _check_prospective__pair_scoped_copies_spec,
    _check_prospective__fitted_specs,
    _check_prospective__value_single_float_per,
)

_DEGENERATE_REL_TOL = 32.0 * np.finfo(np.float64).eps


@numba.njit(cache=True, fastmath=False)
def _abs_corr_finite_njit(a, y, yfin, min_n=8):
    """|Pearson corr| of ``a`` vs ``y`` over rows where both are finite, in one pass (no boolean-index temporaries,
    no 2x2 corrcoef matrix). Returns 0.0 when fewer than ``min_n`` joint-finite rows or either side is
    (near-)constant. FP-equivalent to the numpy ``abs(corrcoef(a[m], y[m])[0,1])`` to ~1e-15 - selection-safe for
    the noise-wrap |corr| gate. ``min_n`` defaults to 8 (small-sample-noise protection for the y-correlation call
    sites); callers replicating a masked ``np.corrcoef`` call site with no such floor (e.g. the ratio/log-ratio FE
    redundancy gate, which rejects on ANY finite overlap corrcoef defines, however small) pass ``min_n=2`` -
    the minimum sample size for which variance - and hence Pearson r - is even defined."""
    # TWO passes, not one. The single-pass form accumulated raw power sums and recovered the variance by
    # subtraction (``saa - sa*sa/n``), which is catastrophic cancellation whenever the data carries an offset
    # large relative to its spread -- an epoch timestamp, a price, a count. Measured on this kernel: at
    # offset/spread 1e7 a true |r| of 0.300 was reported as 0.767, and on one minute of epoch-second ticks a
    # true |r| of 0.497 came back as EXACTLY 0.0, because the destroyed variance tripped the near-constant
    # branch below. That 0.0 means "not redundant, keep" in the dedup gate and "no signal, drop" in the
    # y-gate, so the wrong answer was silently actionable in both directions.
    n = 0
    sa = 0.0
    sy = 0.0
    for i in range(a.shape[0]):
        av = a[i]
        if yfin[i] and np.isfinite(av):
            n += 1
            sa += av
            sy += y[i]
    if n < min_n:
        return 0.0
    ma = sa / n
    my = sy / n
    va = 0.0
    vy = 0.0
    cay = 0.0
    amax = 0.0
    ymax = 0.0
    for i in range(a.shape[0]):
        av = a[i]
        if yfin[i] and np.isfinite(av):
            da = av - ma
            dy = y[i] - my
            va += da * da
            vy += dy * dy
            cay += da * dy
            if abs(av) > amax:
                amax = abs(av)
            if abs(y[i]) > ymax:
                ymax = abs(y[i])
    # Near-constant when the spread is rounding noise of the column's own values. An absolute floor (va <= 1e-24 * n) declared every genuinely
    # tiny-scale column (values ~1e-13) constant and returned 0.0 against its perfect correlate.
    if va <= n * (_DEGENERATE_REL_TOL * amax) ** 2 or vy <= n * (_DEGENERATE_REL_TOL * ymax) ** 2:
        return 0.0
    denom = (va * vy) ** 0.5
    if denom <= 0.0:
        return 0.0
    r = cay / denom
    return -r if r < 0.0 else r


def _check_prospective_fe_step1_def_extval_raw(_extval_raw_col_cache, original_cols, X, _densify_nullable, allow_engineered_operands, cols, engineered_operand_values, _use_subsample, _full_n_rows, _sample_idx):
    """Step 1 of check_prospective_fe_pairs: lines starting at ``def _extval_raw_col(_var):``."""
    from mlframe.feature_selection.filters.feature_engineering import (
        logger,
    )

    def _extval_raw_col(_var):
        """Memoised operand-values ndarray for var ``_var`` (cols-space index).

        For a RAW operand (``_var in original_cols``) returns ``X``'s column at the
        ``original_cols[_var]`` position (the RAW position into ``feature_names_in_``),
        bit-identical to the legacy ``X.iloc[...].values`` / ``.to_numpy()`` extract.

        ENGINEERED-OPERAND FEED-FORWARD: at FE step k>1 the operand pool
        also carries the engineered columns appended by the prior step(s)
        (``selected_vars`` includes their cols-space indices, so the pair-MI sweep
        surfaces ``(eng_i, eng_j)`` pairs - e.g. the additive composite of the two
        real step-1 features that captures ~the entire deterministic signal). Those
        columns are NOT in ``original_cols`` (which holds raw ``feature_names_in_``
        positions only), but they ARE present in the AUGMENTED frame ``X`` under their
        ``cols[_var]`` name (``_mrmr_fe_step`` appends each engineered column to BOTH
        ``cols`` and ``X`` in lockstep). When ``allow_engineered_operands`` is on we
        fetch them by NAME so ``(eng_i, eng_j)`` can produce a real composite candidate.
        Returns ``None`` only when the var is neither a raw position nor a resolvable
        augmented-frame column (the caller then skips it, exactly as before)."""
        if _var in _extval_raw_col_cache:
            return _extval_raw_col_cache[_var]
        if _var in original_cols:
            if isinstance(X, pd.DataFrame):
                _raw_dtype = X.dtypes.iloc[original_cols[_var]]
                if not pd.api.types.is_numeric_dtype(_raw_dtype):
                    # Defense in depth: ``numeric_vars_to_consider`` is built once, upstream, via
                    # ``_non_numeric_column_indices`` (see that helper's docstring) and is meant to
                    # already exclude every non-numeric raw column (datetime/object/categorical) from
                    # ever reaching a var-index here. If a non-numeric index nonetheless reaches this
                    # point (e.g. a stale/misaligned pool from an earlier FE round), extracting it
                    # RAW and feeding it straight into ``binary_transformations`` (plain numpy ufuncs
                    # like ``np.multiply``) crashes with a dtype-resolution error instead of a clean
                    # skip. Re-validate at the point of use, the same invariant every other operand
                    # pool touch point already enforces, rather than let a raw datetime/object column
                    # reach a numeric ufunc.
                    logger.debug(
                        "_extval_raw_col: var %r resolved to non-numeric raw column dtype %s; skipping " "(should have been excluded upstream by _non_numeric_column_indices).",
                        _var,
                        _raw_dtype,
                    )
                    _extval_raw_col_cache[_var] = None
                    return None
                _vals = _densify_nullable(X.iloc[:, original_cols[_var]].values)
            else:
                _vals = X[:, original_cols[_var]].to_numpy()
            _extval_raw_col_cache[_var] = _vals
            return _vals
        # Engineered operand: resolve by name. PREFER the CONTINUOUS engineered values
        # (``engineered_operand_values[name]``) over the augmented frame's column, which
        # holds the DISCRETISED bin codes - combining bin codes (e.g. ``add(codes_a,
        # codes_b)``) is severely lossy and sinks the composite below the engineered-MI
        # gate (measured: 0.88 from codes vs 1.81 - the full signal - from continuous
        # values). Fall back to the by-name frame extract when no continuous value is
        # stored (e.g. an engineered column produced by a stage that did not register one).
        if allow_engineered_operands and 0 <= _var < len(cols):
            _name = cols[_var]
            _vals = None
            if engineered_operand_values is not None:
                _cv = engineered_operand_values.get(_name)
                if _cv is not None:
                    _cv = np.asarray(_cv)
                    # The continuous store is full-n; align to the (possibly subsampled) X.
                    if _cv.shape[0] == len(X):
                        _vals = _cv
                    elif _use_subsample and _cv.shape[0] == _full_n_rows:
                        _vals = _cv[_sample_idx]
            if _vals is None:
                try:
                    if isinstance(X, pd.DataFrame):
                        _vals = _densify_nullable(X[_name]) if isinstance(X[_name].dtype, ExtensionDtype) else (X[_name].to_numpy() if hasattr(X[_name], "to_numpy") else X[_name].values)
                    elif hasattr(X, "columns") and _name in getattr(X, "columns", []):
                        _vals = X[_name].to_numpy()  # polars
                    else:
                        _vals = None
                except Exception as e:
                    logger.debug("reading column %r as numpy failed, skipping: %s", _name, e)
                    _vals = None
            if _vals is not None:
                _vals = np.asarray(_vals)
                _extval_raw_col_cache[_var] = _vals
                return _vals
        _extval_raw_col_cache[_var] = None
        return None
    return _extval_raw_col


def _check_prospective_fe_step2_classes_codes_still(_corr_y_cont, _corr_y_cont_finite):
    """Step 2 of check_prospective_fe_pairs: lines starting at ``def _safe_abs_corr(_v) -> float:``."""
    from mlframe.feature_selection.filters.feature_engineering import (
        logger,
    )

    def _safe_abs_corr(_v) -> float:
        """|Pearson corr| of a column with the (subsample-aligned) target over their jointly-finite rows;
        0.0 when the guard target is unavailable or either side is degenerate. Cheap (one corrcoef)."""
        if _corr_y_cont is None:
            return 0.0
        try:
            _a = np.ascontiguousarray(np.asarray(_v, dtype=np.float64).ravel())
            if _a.shape[0] != _corr_y_cont.shape[0]:
                return 0.0
            # One-pass njit |corr| over jointly-finite rows - replaces isfinite-mask + boolean-index copies + two
            # np.std + a 2x2 np.corrcoef (~23-35x on the 8k+ noise-wrap-gate calls); FP-equivalent to ~1e-15.
            return float(_abs_corr_finite_njit(_a, _corr_y_cont, _corr_y_cont_finite, 8))
        except Exception as e:
            logger.debug("_safe_abs_corr: |corr| computation failed, treating as uncorrelated (0.0): %s", e)
            return 0.0
    return _safe_abs_corr


def _check_prospective_fe_step3_gpu_clock_variance(_operand_marginal_mi_cache, vars_transformations, transformed_vars, quantization_nbins, quantization_method, quantization_dtype, classes_y, classes_y_safe, freqs_y, fe_min_nonzero_confidence, fe_npermutations):
    """Step 3 of check_prospective_fe_pairs: lines starting at ``def _operand_marginal_mi(_var) -> float:``."""
    from mlframe.feature_selection.filters.feature_engineering import (
        discretize_array,
        logger,
        mi_direct,
    )

    def _operand_marginal_mi(_var) -> float:
        """Memoised single-operand MI against the target, used as the marginal-uplift fallback gate's baseline. Fails CLOSED
        (returns +inf, never 0.0) on a computation error so an unknown marginal can only tighten admission, never loosen it."""
        if _var in _operand_marginal_mi_cache:
            return float(_operand_marginal_mi_cache[_var])
        _mi_val = 0.0
        _idx = vars_transformations.get((_var, "identity"))
        if _idx is not None:
            try:
                _disc = discretize_array(
                    arr=transformed_vars[:, _idx],
                    n_bins=quantization_nbins,
                    method=quantization_method,
                    dtype=quantization_dtype,
                )
                _m, _ = mi_direct(
                    _disc.reshape(-1, 1),
                    x=np.array([0], dtype=np.int64),  # type: ignore[arg-type]  # same reason as `y` below: the annotation is stricter than the accepted call shape
                    y=None,  # type: ignore[arg-type]  # mi_direct (permutation.py, sibling-owned) accepts this call shape at runtime; its x/y annotation (tuple) is stricter than actual usage
                    factors_nbins=np.array([quantization_nbins], dtype=np.int64),
                    classes_y=classes_y,
                    classes_y_safe=classes_y_safe,
                    freqs_y=freqs_y,
                    min_nonzero_confidence=fe_min_nonzero_confidence,
                    npermutations=fe_npermutations,
                )
                _mi_val = float(_m)
            except Exception as _mm_exc:
                # FAIL-CLOSED (audit A3, 2026-06-13): the previous ``0.0`` was FAIL-OPEN - it fed
                # the marginal-uplift gate's ``max(operand marginals)``, so a FAILED marginal on the
                # operand that actually has the LARGER marginal would shrink that max and LOOSEN the
                # admission bar (``best_nonprewarp_mi >= max_marginal * _FE_MARGINAL_UPLIFT_MIN_RATIO``),
                # wrongly admitting a feature whose uplift was never validated. Return +inf instead so
                # an UNKNOWN marginal can only TIGHTEN the gate (the pair fails the uplift fallback and
                # is dropped-on-uncertainty); it can still be admitted by the joint/prewarp gates, which
                # do not use this marginal. The whole-pair both-operand-fail case was already
                # fail-closed via the ``_max_operand_marginal > 0.0`` guard.
                logger.debug(
                    "MRMR FE: operand %s marginal-MI computation failed (%s); failing the marginal-uplift gate CLOSED (+inf) so it cannot loosen admission.",
                    _var,
                    type(_mm_exc).__name__,
                )
                _mi_val = float("inf")
        _operand_marginal_mi_cache[_var] = _mi_val
        return _mi_val
    return _operand_marginal_mi


def _check_prospective_fe_step4_int_code_array(_operand_disc_cache, vars_transformations, transformed_vars, quantization_nbins, quantization_method, quantization_dtype):
    """Step 4 of check_prospective_fe_pairs: lines starting at ``def _operand_discretized(_var):``."""
    from mlframe.feature_selection.filters.feature_engineering import (
        discretize_array,
        logger,
    )

    def _operand_discretized(_var):
        """Memoised per-operand discretised codes (same binning as the raw pair's joint MI), or None if the operand has no identity transform / discretisation fails."""
        if _var in _operand_disc_cache:
            return _operand_disc_cache[_var]
        _codes = None
        _idx = vars_transformations.get((_var, "identity"))
        if _idx is not None:
            try:
                _codes = discretize_array(
                    arr=transformed_vars[:, _idx],
                    n_bins=quantization_nbins,
                    method=quantization_method,
                    dtype=quantization_dtype,
                )
            except Exception as e:
                logger.debug("discretizing operand %r failed, caching None: %s", _var, e)
                _codes = None
        _operand_disc_cache[_var] = _codes
        return _codes
    return _operand_discretized
