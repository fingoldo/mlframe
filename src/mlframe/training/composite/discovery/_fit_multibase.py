"""Multi-base forward-stepwise auto-promotion, carved out of ``_fit.fit`` to keep
that module under the 1k-line monolith threshold.

After single-base discovery + raw-y baseline gate + tiny-model rerank, look at
each kept ``linear_residual`` spec and try greedily adding more bases from the
auto-base candidate pool. When the marginal RMSE reduction clears
``multi_base_min_marginal_rmse_gain`` (default 0.02 = 2%), upgrade the spec to
``linear_residual_multi`` with the expanded base list. Measure-first benchmark
in ``benchmarks/composite_multi_base_benchmark.py`` validates: geo-mean gain
83% on positive scenarios, no-harm on negative scenarios -> auto-promote=True.
Gated by ``self.config.multi_base_enabled``; opt-out via config.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from . import CompositeTargetDiscovery

from ..spec import CompositeSpec
from ..transforms import _linear_residual_multi_fit, compose_target_name
from .forward_stepwise import forward_stepwise_multi_base

logger = logging.getLogger(__name__)


def apply_multi_base_forward_stepwise(
    self: "CompositeTargetDiscovery",
    kept_specs: list[CompositeSpec],
    df: Any,
    target_col: str,
    train_idx: np.ndarray,
    y_train: np.ndarray,
) -> list[CompositeSpec]:
    """Greedily upgrade single-base ``linear_residual`` specs to multi-base, then dedup by base set."""
    from .screening import _extract_column_array

    # Auto-skip multi-base promotion when the base pool is uniformly highly-correlated: stacking two
    # near-identical bases into a multi-base residual adds NO orthogonal signal but DOUBLES the
    # base-shift amplification of the inverse on unseen groups. (On the prod TVT pool every base was
    # >=0.999 correlated -- a multi-base upgrade there would have made the collapse strictly worse.)
    _multibase_pool_corr_skip = False
    if kept_specs and getattr(self.config, "multi_base_enabled", False) and getattr(self, "_auto_base_pool", None):
        _pool_corr_thresh = float(getattr(self.config, "multi_base_skip_when_pool_corr_above", 0.98))
        if _pool_corr_thresh < 1.0:
            try:
                _pa = [np.asarray(v, dtype=np.float64).ravel() for v in self._auto_base_pool.values() if v is not None]
                _pa = [a for a in _pa if a.size > 2 and float(a.std()) > 0]
                if len(_pa) >= 2:
                    _M = np.vstack(_pa)
                    # np.corrcoef's return type is unioned with float64 (the 1-row degenerate case);
                    # len(_pa) >= 2 above guarantees a real (>=2, >=2) matrix at runtime, but mypy
                    # can't see that -- np.asarray narrows the static type back to ndarray.
                    _C = np.asarray(np.corrcoef(_M))
                    _off = _C[~np.eye(_C.shape[0], dtype=bool)]
                    _off = np.abs(_off[np.isfinite(_off)])
                    if _off.size and float(_off.mean()) > _pool_corr_thresh:
                        _multibase_pool_corr_skip = True
                        logger.info(
                            "[CompositeTargetDiscovery] multi-base promotion SKIPPED: base pool is "
                            "uniformly highly-correlated (mean |pair-corr|=%.4f > %.4g) -- a multi-base "
                            "residual would add no orthogonal signal and double the inverse's base-shift "
                            "amplification.", float(_off.mean()), _pool_corr_thresh,
                        )
            except Exception:  # -- the corr guard is a heuristic; never abort discovery on it
                _multibase_pool_corr_skip = False
    if not (kept_specs and getattr(self.config, "multi_base_enabled", False) and getattr(self, "_auto_base_pool", None) and not _multibase_pool_corr_skip):
        return kept_specs

    _multi_max_k = int(getattr(self.config, "multi_base_max_k", 3))
    _multi_min_gain = float(getattr(self.config, "multi_base_min_marginal_rmse_gain", 0.02))
    _cv_sel_mode = str(getattr(self.config, "cv_selector_mode", "mean"))
    _cv_sel_alpha = float(getattr(self.config, "cv_selector_alpha", 1.0))
    _cv_sel_conf = float(getattr(self.config, "cv_selector_confidence", 0.9))
    _cv_sel_qlevel = float(getattr(self.config, "cv_selector_quantile_level", 0.9))
    _cv_persist = bool(getattr(self.config, "cv_persist_fold_scores", False))
    _upgraded_specs: list[CompositeSpec] = []
    # Hoist the (base_column, pool_signature) -> pool_arrays
    # build outside the per-spec loop so K linear_residual specs that
    # share the same auto_base_pool + base_column do ONE pool build
    # (and one _extract_column_array call), not K. Cache key includes
    # the pool signature (frozenset of pool keys) so config-driven
    # pool changes invalidate cleanly.
    _pool_arrays_cache: dict[tuple[str, frozenset], dict[str, np.ndarray]] = {}
    _base_pool_keys_frozen = frozenset(self._auto_base_pool.keys())
    _y_train_local = y_train
    for _spec in kept_specs:
        if _spec.transform_name != "linear_residual":
            _upgraded_specs.append(_spec)
            continue
        _cache_key = (_spec.base_column, _base_pool_keys_frozen)
        _pool_arrays = _pool_arrays_cache.get(_cache_key)
        if _pool_arrays is None:
            # Build candidate pool: the auto-base candidates (top-K MI-ranked bases) PLUS the spec's own seed base.
            _pool_cols = list(self._auto_base_pool.keys())
            if _spec.base_column not in _pool_cols:
                _pool_cols.append(_spec.base_column)
            # Materialise arrays once (the pool stores arrays).
            _pool_arrays = {c: v for c in _pool_cols if (v := self._auto_base_pool.get(c)) is not None}
            if _spec.base_column not in _pool_arrays:
                _pool_arrays[_spec.base_column] = _extract_column_array(df, _spec.base_column)[train_idx]
            _pool_arrays_cache[_cache_key] = _pool_arrays
        try:
            _kept_bases, _fwd_diag = forward_stepwise_multi_base(
                _y_train_local,
                _pool_arrays,
                seed_bases=[_spec.base_column],
                max_k=_multi_max_k,
                min_marginal_rmse_gain=_multi_min_gain,
                cv_selector_mode=_cv_sel_mode,
                cv_selector_alpha=_cv_sel_alpha,
                cv_selector_confidence=_cv_sel_conf,
                cv_selector_quantile_level=_cv_sel_qlevel,
                cv_persist_fold_scores=_cv_persist,
            )
        except Exception as _multi_err:
            logger.warning(
                "[CompositeTargetDiscovery] multi-base forward-stepwise failed on spec=%s: %s. Keeping single-base spec.",
                _spec.name,
                _multi_err,
            )
            _upgraded_specs.append(_spec)
            continue
        if len(_kept_bases) <= 1:
            # No additional bases survived the gate; keep the original single-base spec.
            _upgraded_specs.append(_spec)
            continue
        # Upgrade: fit the linear_residual_multi joint OLS on the kept base set and stamp a NEW spec with extra_base_columns populated.
        _base_matrix = np.column_stack([_pool_arrays[n] for n in _kept_bases])
        _multi_params = _linear_residual_multi_fit(_y_train_local, _base_matrix)
        _new_name = compose_target_name(target_col, "linear_residual_multi", "+".join(_kept_bases))
        _upgraded_spec = CompositeSpec(
            name=_new_name,
            target_col=target_col,
            transform_name="linear_residual_multi",
            base_column=_kept_bases[0],
            fitted_params=_multi_params,
            mi_gain=_spec.mi_gain,
            mi_y=_spec.mi_y,
            mi_t=_spec.mi_t,
            valid_domain_frac=_spec.valid_domain_frac,
            n_train_rows=_spec.n_train_rows,
            extra_base_columns=tuple(_kept_bases[1:]),
        )
        _upgraded_specs.append(_upgraded_spec)
        # The upgraded spec carries a NEW name (``...-linear_residual_multi-<bases>``); carry the
        # seed's tiny-rerank CV-RMSE over to it so the raw-y baseline gate and the public
        # ``tiny_rerank_scores_`` diagnostic can look it up by the new name. The multi-base composite
        # is built by ADDING bases to a seed that already cleared the raw-y baseline gate, and every
        # added base only reduced the joint-OLS residual, so the seed's score is a valid conservative
        # stand-in. Without this the new name has no score entry and any ``tiny_rerank_scores_[name]``
        # lookup KeyErrors (regressed the sklearn-matrix composite sweep).
        _seed_score = getattr(self, "_tiny_rerank_scores", None)
        if isinstance(_seed_score, dict) and _spec.name in _seed_score:
            _seed_score[_new_name] = _seed_score[_spec.name]
        _accepted_steps = [d for d in _fwd_diag if d.get("accepted")]
        logger.info(
            "[CompositeTargetDiscovery.multi_base] upgraded spec='%s' -> '%s' with %d base(s); accepted_steps=%s",
            _spec.name,
            _new_name,
            len(_kept_bases),
            [(d["candidate_added"], f"{d['marginal_gain'] * 100:.1f}%") for d in _accepted_steps],
        )
    # Two seeds can converge on the same multi-base set yet emit name 'X+Y' vs 'Y+X' for one identical joint-OLS transform; dedup on the unordered base set so we don't train + ensemble two perfectly-correlated members.
    _seen_base_sets: set[tuple[str, frozenset]] = set()
    _deduped_specs: list[CompositeSpec] = []
    for _s in _upgraded_specs:
        _set_key = (
            _s.transform_name,
            frozenset((_s.base_column, *tuple(getattr(_s, "extra_base_columns", ()) or ()))),
        )
        if _set_key in _seen_base_sets:
            continue
        _seen_base_sets.add(_set_key)
        _deduped_specs.append(_s)
    return _deduped_specs
