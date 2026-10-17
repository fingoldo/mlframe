"""Per-stage blocks of the tiny-model rerank (raw-y families, per-bin passes, job fan-out, gate enumeration), split out of ``_tiny_rerank``
to keep that module under the size budget. Logger name is kept as ``_tiny_rerank``'s so log filters and captures are unchanged."""
from __future__ import annotations

import logging
import math

import numpy as np

from ._per_base_x import base_ordered
from ._rejection_ledger import RejectStage, ledger_append
from ._tiny_rerank_honest import _tiny_rerank_ram_checkpoint
from ._tiny_rerank_process import make_worker_task, score_specs_in_processes
from .screening import (
    _per_bin_from_fold_preds,
    _tiny_cv_rmse_raw_y,
    _tiny_cv_rmse_raw_y_multiseed,
    _tiny_cv_rmse_y_scale,
)
from ..transforms import get_transform
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger("mlframe.training.composite.discovery._tiny_rerank")


def _tiny_model_rerank_family_families(self, families, use_wilcoxon, y_screen, x_full, _tiny_n_jobs_auto, n_seed_repeats_raw, _early_any_base_monotone, _groups_screen, _cv_sel_mode, _cv_sel_alpha, _cv_sel_conf, _cv_sel_qlevel, raw_rmse_per_family, raw_per_seed_per_family):
    """Block of _tiny_model_rerank starting at ``for family in families:``."""
    for family in families:
        if use_wilcoxon:
            res = _tiny_cv_rmse_raw_y_multiseed(
                y_train=y_screen,
                x_train_matrix=x_full,
                family=family,
                n_estimators=self.config.tiny_model_n_estimators,
                num_leaves=self.config.tiny_model_num_leaves,
                learning_rate=self.config.tiny_model_learning_rate,
                cv_folds=self.config.tiny_model_cv_folds,
                n_jobs=_tiny_n_jobs_auto,
                deterministic=getattr(
                    self.config, "deterministic_screening_models", False,
                ),
                n_seed_repeats=n_seed_repeats_raw,
                base_random_state=self.config.random_state,
                return_per_seed=True,
                time_aware=_early_any_base_monotone,
                groups=_groups_screen,
                cv_selector_mode=_cv_sel_mode,
                cv_selector_alpha=_cv_sel_alpha,
                cv_selector_confidence=_cv_sel_conf,
                cv_selector_quantile_level=_cv_sel_qlevel,
            )
            raw_rmse_per_family[family] = res[0]
            raw_per_seed_per_family[family] = res[-1]
        else:
            raw_rmse_per_family[family] = _tiny_cv_rmse_raw_y_multiseed(
                y_train=y_screen,
                x_train_matrix=x_full,
                family=family,
                n_estimators=self.config.tiny_model_n_estimators,
                num_leaves=self.config.tiny_model_num_leaves,
                learning_rate=self.config.tiny_model_learning_rate,
                cv_folds=self.config.tiny_model_cv_folds,
                n_jobs=_tiny_n_jobs_auto,
                deterministic=getattr(
                    self.config, "deterministic_screening_models", False,
                ),
                n_seed_repeats=n_seed_repeats_raw,
                base_random_state=self.config.random_state,
                time_aware=_early_any_base_monotone,
                groups=_groups_screen,
                cv_selector_mode=_cv_sel_mode,
                cv_selector_alpha=_cv_sel_alpha,
                cv_selector_confidence=_cv_sel_conf,
                cv_selector_quantile_level=_cv_sel_qlevel,
            )


def _tiny_model_rerank_per_bin_enabled_pre(self, per_bin_enabled_pre, y_screen, kept_specs, raw_per_bin_per_base, _per_base_cache, families, x_full, _tiny_n_jobs_auto, _groups_screen, _early_any_base_monotone, per_bin_n_bins_pre):
    """Block of _tiny_model_rerank starting at ``if per_bin_enabled_pre:``."""
    if per_bin_enabled_pre:
        _raw_fold_preds = None
        _y_screen_finite = np.isfinite(np.asarray(y_screen))
        _bin_var_needs_mask = not bool(_y_screen_finite.all())
        for spec in kept_specs:
            if spec.base_column in raw_per_bin_per_base:
                continue
            base_screen = _per_base_cache.base_screen(spec.base_column)
            if base_screen is None:
                continue
            family = families[0]
            if _raw_fold_preds is None:
                raw_result = _tiny_cv_rmse_raw_y(
                    y_train=y_screen,
                    x_train_matrix=x_full,
                    family=family,
                    n_estimators=self.config.tiny_model_n_estimators,
                    num_leaves=self.config.tiny_model_num_leaves,
                    learning_rate=self.config.tiny_model_learning_rate,
                    cv_folds=self.config.tiny_model_cv_folds,
                    random_state=self.config.random_state,
                    n_jobs=_tiny_n_jobs_auto,
                    deterministic=getattr(
                        self.config, "deterministic_screening_models", False,
                    ),
                    return_fold_preds=True,
                    groups=_groups_screen,
                    time_aware=_early_any_base_monotone,
                )
                _raw_fold_preds = raw_result[1] if isinstance(raw_result, tuple) else []
            if not _raw_fold_preds:
                continue
            _bin_var_clean = base_screen[_y_screen_finite] if _bin_var_needs_mask else base_screen
            raw_per_bin = _per_bin_from_fold_preds(
                _raw_fold_preds, _bin_var_clean, n_bins=per_bin_n_bins_pre,
            )
            raw_per_bin_per_base[spec.base_column] = raw_per_bin


def _tiny_model_rerank_aggregate_single_per(kept_specs, families, per_family_scores, agg_scores, consensus):
    """Block of _tiny_model_rerank starting at ``for i, _spec in enumerate(kept_specs):``."""
    for i, _spec in enumerate(kept_specs):
        family_rmses = [per_family_scores[f][i] for f in families]
        finite = [r for r in family_rmses if math.isfinite(r)]
        if not finite:
            agg_scores.append(float("inf"))
            continue
        _tiny_model_rerank_consensus_union(consensus, agg_scores, finite)


def _tiny_model_rerank_np_arange_filtered_train(_groups_full_for_rerank, train_idx_screen, _groups_screen):
    """Block of _tiny_model_rerank starting at ``if _groups_full_for_rerank is not None:``."""
    if _groups_full_for_rerank is not None:
        try:
            _ga = np.asarray(_groups_full_for_rerank)
            if _ga.shape[0] >= int(np.max(train_idx_screen) + 1):
                _groups_screen = _ga[train_idx_screen]
        except (TypeError, ValueError, IndexError):
            _groups_screen = None
    return _groups_screen


def _tiny_model_rerank_self_config_tiny_screening(self):
    """Block of _tiny_model_rerank starting at ``if self.config.tiny_screening_models == "single_lgbm":``."""
    if self.config.tiny_screening_models == "single_lgbm":
        families = ["lightgbm"]
    else:  # per_family
        families = [f for f in self.config.tiny_screening_families]
        if not families:
            families = ["lightgbm"]
    return families


def _tiny_model_rerank_finite_raw_early(self, _finite_raw_early, raw_baseline):
    """Block of _tiny_model_rerank starting at ``if _finite_raw_early:``."""
    if _finite_raw_early:
        if self.config.tiny_consensus == "union":
            raw_baseline = min(_finite_raw_early)
        else:
            raw_baseline = float(np.mean(_finite_raw_early))
    return raw_baseline


def _tiny_model_rerank_rerank_jobs_len_kept(_rerank_n_jobs, kept_specs, _backend, _task_common, _worker_fold_n_jobs, _rerank_inner_n_jobs, _skip_cv_names, _per_base_cache, _rerank_one_spec):
    """Block of _tiny_model_rerank starting at ``if _rerank_n_jobs > 1 and len(kept_specs) > 1 and _backend == "process``."""
    if _rerank_n_jobs > 1 and len(kept_specs) > 1 and _backend == "processes":
        _common = dict(_task_common, fold_n_jobs=_worker_fold_n_jobs, inner_n_jobs=_rerank_inner_n_jobs)
        _rerank_results = score_specs_in_processes(
            [make_worker_task(s, get_transform(s.transform_name), _common, s.name in _skip_cv_names, _per_base_cache) for s in kept_specs],
            _rerank_n_jobs,
        )
    elif _rerank_n_jobs > 1 and len(kept_specs) > 1:
        from joblib import Parallel as _Parallel, delayed as _delayed

        # Grouped by base so the bounded per-base cache gathers each base once; results go back to spec order.
        _dispatch = base_ordered(range(len(kept_specs)), lambda i: kept_specs[i].base_column)
        _rerank_results = _Parallel(
            n_jobs=_rerank_n_jobs, backend="threading", prefer="threads",
        )(_delayed(_rerank_one_spec)(kept_specs[i]) for i in _dispatch)
        _rerank_results = [r for _, r in sorted(zip(_dispatch, _rerank_results))]
    else:
        # Sequential path: log every other spec so the kill-point is bracketed
        # without flooding the log on a 100-spec rerank. The parallel path
        # can't checkpoint mid-loop without contention on the logger.
        _rerank_results = []
        for _i, _spec in enumerate(kept_specs):
            _rerank_results.append(_rerank_one_spec(_spec))
            if _i % 2 == 1 or _i == len(kept_specs) - 1:
                _tiny_rerank_ram_checkpoint(f"after_spec[{_i + 1}/{len(kept_specs)}]={_spec.name[:40]}")
    return _rerank_results


def _tiny_model_rerank_spec_name_fam_errs(self, _rerank_results, families, per_family_scores, _per_bin_first_pass):
    """Block of _tiny_model_rerank starting at ``for _spec_name, _fam_rmses, _per_seed_by_family, _per_bin_first in _re``."""
    for _spec_name, _fam_rmses, _per_seed_by_family, _per_bin_first in _rerank_results:
        for family in families:
            per_family_scores[family].append(_fam_rmses.get(family, float("nan")))
        if _per_bin_first is not None:
            _per_bin_first_pass[_spec_name] = _per_bin_first
        for family, _per_seed in _per_seed_by_family.items():
            self._wilcoxon_per_seed_composite[(_spec_name, family)] = _per_seed


def _tiny_model_rerank_consensus_union(consensus, agg_scores, finite):
    """Block of _tiny_model_rerank starting at ``if consensus == "union":``."""
    if consensus == "union":
        # Best (lowest) family RMSE. "Union" = "kept if any
        # family ranks it well".
        agg_scores.append(min(finite))
    elif consensus == "borda":
        # Borda needs ranks per family.
        # Build rank tables per family, sum ranks per spec
        # below at the after-loop step. For simplicity use
        # mean RMSE here as a Borda proxy on a per-spec
        # basis -- for a 2-3 family setup the Borda result
        # collapses to mean rank, equivalent to mean RMSE.
        agg_scores.append(float(np.mean(finite)))
    else:
        agg_scores.append(min(finite))


def _tiny_model_rerank_per_spec_per_bin(self, per_bin_enabled, kept_specs, _per_bin_first_pass, spec_per_bin_rmse, _per_base_cache, families, y_screen, _tiny_n_jobs_auto, per_bin_n_bins, _early_any_base_monotone, _groups_screen):
    """Block of _tiny_model_rerank starting at ``if per_bin_enabled:``."""
    if per_bin_enabled:
        # REUSE the per-bin breakdown captured during the
        # first-pass multiseed sweep instead of re-running the K-fold
        # LGBM fits. Falls back to a recompute only if first-pass
        # was disabled (e.g. when ``per_bin_n_bins`` was changed
        # mid-run) or a spec is missing from the cache (NaN result).
        for _i, spec in enumerate(kept_specs):
            cached_pb = _per_bin_first_pass.get(spec.name)
            if cached_pb is not None:
                spec_per_bin_rmse[spec.name] = cached_pb
                continue
            cached = _per_base_cache.get(spec.base_column)
            if cached is None:
                continue
            base_screen, x_remaining_matrix = cached
            transform = get_transform(spec.transform_name)
            family = families[0]
            result = _tiny_cv_rmse_y_scale(
                y_train=y_screen, base_train=base_screen,
                transform=transform, fitted_params=spec.fitted_params,
                x_train_matrix=x_remaining_matrix,
                family=family,
                n_estimators=self.config.tiny_model_n_estimators,
                num_leaves=self.config.tiny_model_num_leaves,
                learning_rate=self.config.tiny_model_learning_rate,
                cv_folds=self.config.tiny_model_cv_folds,
                random_state=self.config.random_state,
                n_jobs=_tiny_n_jobs_auto,
                deterministic=getattr(
                    self.config, "deterministic_screening_models", False,
                ),
                return_per_bin=True,
                n_bins=per_bin_n_bins,
                time_aware=_early_any_base_monotone,
                groups=_groups_screen,
            )
            if isinstance(result, tuple):
                _, per_bin = result
                spec_per_bin_rmse[spec.name] = per_bin


def _tiny_model_rerank_spec_enumerate_kept_specs(self, kept_specs, agg_scores, threshold, gate_rejected_names, use_wilcoxon, families, raw_per_seed_per_family, gate_alpha, wilcoxon_rejected, per_bin_enabled, spec_per_bin_rmse, raw_per_bin_per_base, per_bin_tol, per_bin_rejected_names, survivors):
    """Block of _tiny_model_rerank starting at ``for i, spec in enumerate(kept_specs):``."""
    for i, spec in enumerate(kept_specs):
        score = agg_scores[i]
        if score >= threshold:  # every score is finite here: _reject_unscored_specs ran first
            gate_rejected_names.append((spec.name, score, threshold))
            ledger_append(
                self, spec_name=spec.name, stage=RejectStage.TINY_RERANK_THRESHOLD,
                reason=f"tiny-rerank CV-RMSE {score:.4g} >= raw-baseline threshold {threshold:.4g}",
                base_column=getattr(spec, "base_column", ""), transform_name=getattr(spec, "transform_name", ""),
                numbers={"cv_rmse": float(score), "threshold": float(threshold)},
            )
            continue
        # Paired Wilcoxon signed-rank test
        # on per-seed RMSE diffs (composite - raw). Reject
        # spec unless the median diff is significantly
        # negative at level gate_alpha.
        if use_wilcoxon and hasattr(self, "_wilcoxon_per_seed_composite"):
            family = families[0]
            comp_per_seed = self._wilcoxon_per_seed_composite.get((spec.name, family))
            raw_per_seed = raw_per_seed_per_family.get(family)
            # The one-sided Wilcoxon signed-rank test cannot reach a
            # p-value below gate_alpha unless there are enough paired
            # seeds: the most extreme configuration (all diffs favouring
            # the composite) gives min-p = 1/2^n, so n must satisfy
            # 1/2^n <= gate_alpha. At the default n_seed_repeats=3 and
            # gate_alpha=0.05 the floor is 0.125 -> the gate is
            # UNPASSABLE and silently rejects every spec. Require the
            # statistically-minimum seed count; below it, skip the
            # Wilcoxon rejection (threshold gate still applies) and warn.
            _min_seeds_wilcoxon = math.ceil(math.log2(1.0 / max(gate_alpha, 1e-12)))
            # Composite and raw per-seed arrays are fixed-length
            # NaN-padded on the SAME seed schedule, so pair by seed
            # INDEX and keep only positions finite on both sides. A
            # compacted (finite-only) layout would let a failed composite
            # seed and a failed raw seed at different positions produce
            # equal-length-but-mis-paired vectors -- the diff would then
            # subtract unrelated seeds. ``n`` for the min-seed gate is the
            # jointly-finite pair count, not the raw length.
            _both_finite = (
                np.isfinite(comp_per_seed) & np.isfinite(raw_per_seed)
                if (comp_per_seed is not None and raw_per_seed is not None and len(comp_per_seed) == len(raw_per_seed))
                else None
            )
            _n_paired = int(_both_finite.sum()) if _both_finite is not None else 0
            if _both_finite is not None and _n_paired < _min_seeds_wilcoxon:
                log_throttle(
                    logger,
                    "tiny_rerank_wilcoxon_gate_skipped",
                    logging.WARNING,
                    "[CompositeTargetDiscovery] Wilcoxon gate skipped: "
                    "jointly-finite paired seeds=%d < %d, the minimum for "
                    "a one-sided test to reach p<=gate_alpha=%.3g "
                    "(min-p=1/2^n). Raise tiny_model_n_seed_repeats "
                    "(and/or fix the seeds that degenerated) to enable "
                    "the gate; the threshold gate still applies.",
                    _n_paired, _min_seeds_wilcoxon, gate_alpha,
                )
            elif _both_finite is not None and _n_paired >= _min_seeds_wilcoxon:
                try:
                    from scipy.stats import wilcoxon

                    assert comp_per_seed is not None and raw_per_seed is not None
                    diff = comp_per_seed[_both_finite] - raw_per_seed[_both_finite]
                    # One-sided: composite better (less RMSE)
                    # so we want diff < 0; alternative='less'.
                    stat_res = wilcoxon(
                        diff, alternative="less",
                        zero_method="wilcox",
                    )
                    p_value = float(stat_res.pvalue)
                    if p_value > gate_alpha:
                        wilcoxon_rejected.append((spec.name, p_value))
                        continue
                except (ImportError, ValueError) as _wx_err:
                    # Scipy missing or all-zero diffs ->
                    # fall through to the threshold-only
                    # gate (no Wilcoxon rejection).
                    logger.debug(
                        "[CompositeTargetDiscovery] " "Wilcoxon gate skipped for spec=%s: %s",
                        spec.name,
                        _wx_err,
                    )
        # Per-bin gate. Composite
        # passes the global mean test; now check that
        # per-bin RMSE doesn't blow out vs the raw-y
        # per-bin baseline on any quintile of base.
        if per_bin_enabled and spec.name in spec_per_bin_rmse and spec.base_column in raw_per_bin_per_base:
            spec_pb = spec_per_bin_rmse[spec.name]
            raw_pb = raw_per_bin_per_base[spec.base_column]
            # Element-wise compare, ignoring NaN bins.
            worst_ratio = 0.0
            worst_bin_idx = -1
            for b in range(len(spec_pb)):
                if math.isfinite(spec_pb[b]) and math.isfinite(raw_pb[b]) and raw_pb[b] > 0:
                    ratio = spec_pb[b] / raw_pb[b]
                    if ratio > worst_ratio:
                        worst_ratio = ratio
                        worst_bin_idx = b
            if worst_ratio >= per_bin_tol:
                per_bin_rejected_names.append((
                    spec.name,
                    f"bin_{worst_bin_idx}",
                    float(worst_ratio),
                    float(per_bin_tol),
                ))
                continue
        survivors.append((i, spec, score))
