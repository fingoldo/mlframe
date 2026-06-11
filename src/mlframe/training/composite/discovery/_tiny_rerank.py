"""Tiny-model rerank phase for ``CompositeTargetDiscovery``.

``_tiny_model_rerank`` is bound back onto the ``CompositeTargetDiscovery``
class at the parent's module top, so call sites that invoke
``self._tiny_model_rerank(...)`` continue to work unchanged.
"""
from __future__ import annotations

import logging
import math
import re
from typing import Any, Optional, Sequence

import numpy as np

from ..spec import CompositeSpec
from ..ensemble import _is_monotone_nondecreasing
from .screening import (
    _extract_column_array,
    _per_bin_from_fold_preds,
    _sample_indices,
    _tiny_cv_rmse_raw_y,
    _tiny_cv_rmse_raw_y_multiseed,
    _tiny_cv_rmse_y_scale,
    _tiny_cv_rmse_y_scale_multiseed,
)
from ..transforms import get_transform

logger = logging.getLogger(__name__)


def _tiny_rerank_ram_checkpoint(label: str) -> None:
    """Log the current process memory triple right at a tiny-rerank boundary.

    Mirrors the discovery profiler format so the prod log presents a single
    coherent thread of RAM checkpoints across discovery sub-phases AND the
    tiny-rerank internals.

    The user observed a kernel-kill INSIDE tiny_model_rerank with ~20 GB of
    physical RAM still free -- not a classical OOM. Most likely cause on
    Windows: the system-wide commit-charge limit (physical + pagefile) was
    exhausted by mlframe + system baseline + the next LightGBM Dataset's
    transient allocation; the kernel-level C alloc fails, LightGBM doesn't
    handle it gracefully, and the process crashes with an access violation.
    Per-step checkpoints pin the exact step that pushed commit over the
    edge.
    """
    try:
        from ._fit import _process_mem_mb
        rss_mb, uss_mb, commit_mb = _process_mem_mb()
    except Exception:
        return
    logger.info(
        "[CompositeTargetDiscovery.tiny_rerank.RAM] %s USS=%.0f MB (RSS=%.0f MB, commit=%.0f MB)",
        label, uss_mb, rss_mb, commit_mb,
    )


def _tiny_model_rerank(
    self,
    kept_specs: list[CompositeSpec],
    df: Any,
    target_col: str,
    usable_features: Sequence[str],
    train_idx: np.ndarray,
    y_full: np.ndarray,
) -> list[CompositeSpec]:
    """Phase B: re-rank MI-survivors by CV-RMSE on y-scale.

    For each surviving spec:
    1. Build the feature matrix (X-without-base) on a screening
       sample of train rows.
    2. Compute CV-RMSE per family in
       ``self.config.tiny_screening_families``.
    3. Aggregate per-spec score by ``tiny_consensus``:
       - "union": min CV-RMSE across families (best-case).
       - "borda": Borda-count rank aggregation.
    4. Re-sort, take top-``top_m_after_tiny``.
    """
    # Emergency-skip path. The user observed kernel-kill INSIDE this function
    # with ~20 GB physical RAM free on a Windows host, consistent with system
    # commit-charge limit exhaustion when a LightGBM Dataset transient
    # allocation pushes the system over (physical + pagefile). Setting
    # MLFRAME_DISCOVERY_SKIP_TINY_RERANK=1 returns kept_specs unchanged so the
    # MI-survivor list ships without the y-scale CV rerank. Operators trade
    # spec-ranking accuracy for the ability to complete discovery at all.
    import os as _os
    if _os.environ.get("MLFRAME_DISCOVERY_SKIP_TINY_RERANK", "").strip().lower() in (
        "1", "true", "yes", "on",
    ):
        logger.warning(
            "[CompositeTargetDiscovery.tiny_rerank] SKIPPED via MLFRAME_DISCOVERY_SKIP_TINY_RERANK=1; "
            "returning %d MI-ranked spec(s) without y-scale CV rerank. "
            "Spec ranking accuracy is reduced; set the env var to 0 (or unset) to restore the rerank pass.",
            len(kept_specs),
        )
        return kept_specs
    _tiny_rerank_ram_checkpoint("entry")
    sample_n = min(self.config.tiny_model_sample_n, train_idx.size)
    # Phase B benefits from stratified sampling on heavy-tail y
    # for the same reason Phase A does -- tiny-model CV-RMSE on a
    # tail-empty sample mis-ranks transforms that only matter in
    # the tail.
    y_train_for_strat = y_full[train_idx]
    sample_idx = _sample_indices(
        train_idx.size, sample_n, self.config.random_state,
        strategy=getattr(self.config, "mi_sample_strategy", "random"),
        y=y_train_for_strat,
        n_strata=getattr(self.config, "mi_n_strata", 10),
    )
    train_idx_screen = train_idx[sample_idx]
    y_screen = y_full[train_idx_screen]

    # Group-aware tiny CV: when ``self._group_ids_for_rerank`` is set
    # (production split is group-aware), slice the per-spec sample
    # back to the same groups so the rerank ranks specs by the same
    # OOF distribution the production split will evaluate. Random
    # KFold on a group-aware production split rates per-group memorisers
    # high; production then catches that as catastrophic test failure
    # (observed in prod: 3 composite specs promoted by random-KFold
    # rerank, all 9 trained models failed dummy-floor on group-aware
    # test). ``train_idx`` from the discovery entry point is
    # ``np.arange(N_filtered_train)`` so ``sample_idx == train_idx_screen``.
    _groups_full_for_rerank = getattr(self, "_group_ids_for_rerank", None)
    _groups_screen = None
    if _groups_full_for_rerank is not None:
        try:
            _ga = np.asarray(_groups_full_for_rerank)
            if _ga.shape[0] >= int(np.max(train_idx_screen) + 1):
                _groups_screen = _ga[train_idx_screen]
        except (TypeError, ValueError, IndexError):
            _groups_screen = None

    if self.config.tiny_screening_models == "single_lgbm":
        families = ["lightgbm"]
    else:  # per_family
        families = [f for f in self.config.tiny_screening_families]
        if not families:
            families = ["lightgbm"]

    # Hoist per-bin-enabled check above the first pass so we
    # request per-bin RMSE during the SAME multiseed sweep that produces
    # the global scores. The legacy second pass refit every fold to get
    # per-bin breakdowns; with this change the K-fold LGBM fit count is
    # halved when the regime-aware gate is on.
    per_bin_n_bins_pre = int(getattr(self.config, "per_bin_n_bins", 0) or 0)
    per_bin_enabled_pre = (
        per_bin_n_bins_pre > 0
        and getattr(self.config, "require_beats_raw_baseline", True)
    )

    # Per-spec CV-RMSE per family. When K specs share a base
    # (the typical case: auto-base picks one lag-style
    # dominant feature, all K transforms operate on it), the
    # per-base ``x_remaining`` matrix and ``base_screen`` array
    # are recomputable from the same inputs. Cache them by base
    # to avoid K redundant builds (each ~50 ndarray copies on a
    # 200K-row sample).
    per_family_scores: dict[str, list[float]] = {f: [] for f in families}
    # Parallel buffer for per-bin RMSE captured during the
    # first pass. Keyed by spec.name -> per-bin ndarray. Only populated
    # when ``per_bin_enabled_pre`` is True.
    _per_bin_first_pass: dict[str, np.ndarray] = {}
    _per_base_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # Pre-build per-base cache serially (one entry per unique
    # base_column). Doing this before the parallel rerank avoids a
    # lazy-init serialization point inside the threaded loop and
    # ensures the read-only cache is fully populated by the time
    # workers run.
    # Build the full screen-sample feature matrix ONCE, then derive each
    # base's "all features except this base" matrix via np.delete on the
    # in-RAM matrix instead of re-extracting B+1 full-column passes from the
    # frame (the prior _build_feature_matrix per base re-gathered ~all columns
    # of a 4M-row frame for each unique base, differing by one column).
    # np.delete preserves the surviving column order, so x_full minus the base
    # column index is bit-identical to building from ``usable_features`` minus
    # that column.
    _usable_list = list(usable_features)
    _x_full = self._build_feature_matrix(df, _usable_list, train_idx_screen)
    _col_index = {c: i for i, c in enumerate(_usable_list)}
    for spec in kept_specs:
        if spec.base_column in _per_base_cache:
            continue
        # Unary (``requires_base=False``) specs carry an empty
        # ``base_column`` sentinel -- they ignore the base entirely. Extracting
        # a column named "" would crash, so synthesise a zeros ``base_screen``
        # placeholder (never read by a unary transform's forward) and score
        # against the FULL feature matrix (no base column dropped), mirroring
        # the dedicated unary context built in ``discovery/_fit.py``.
        if spec.base_column == "":
            base_screen = np.zeros(train_idx_screen.size, dtype=np.float32)
            _per_base_cache[spec.base_column] = (base_screen, _x_full)
            continue
        base_screen = _extract_column_array(
            df, spec.base_column, rows=train_idx_screen,
        )
        if spec.base_column in _col_index:
            x_matrix = np.delete(_x_full, _col_index[spec.base_column], axis=1)
        else:
            # Base is not among the screened features (e.g. a lag column the
            # user excluded); no column to drop -- reuse x_full read-only.
            x_matrix = _x_full
        _per_base_cache[spec.base_column] = (base_screen, x_matrix)
    _tiny_rerank_ram_checkpoint(f"per_base_cache_built(n_unique_bases={len(_per_base_cache)})")

    n_seed_repeats = max(1, int(getattr(
        self.config, "tiny_model_n_seed_repeats", 1,
    )))
    use_wilcoxon = bool(getattr(
        self.config, "use_wilcoxon_gate", False,
    ))
    # Forward the CV fold-score selector knobs (honoured by forward stepwise) into every tiny-CV call so the rerank ranks specs by the same selector the production split uses; default 'mean' keeps this bit-identical.
    _cv_sel_mode = str(getattr(self.config, "cv_selector_mode", "mean"))
    _cv_sel_alpha = float(getattr(self.config, "cv_selector_alpha", 1.0))
    _cv_sel_conf = float(getattr(self.config, "cv_selector_confidence", 0.9))
    _cv_sel_qlevel = float(getattr(self.config, "cv_selector_quantile_level", 0.9))

    def _rerank_one_spec(spec: CompositeSpec):
        """Per-spec worker for the rerank loop.

        Returns a tuple ``(spec.name, family_rmses, per_seed_by_family,
        per_bin_first_or_none)`` so the parallel reduce can rebuild
        ``per_family_scores`` / ``_wilcoxon_per_seed_composite`` /
        ``_per_bin_first_pass`` in spec order on the main thread.
        """
        base_screen_local, x_matrix_local = _per_base_cache[spec.base_column]
        transform = get_transform(spec.transform_name)
        # Switch this spec's tiny-CV to TimeSeriesSplit when the data is
        # temporal. Random K-fold on time-correlated rows leaks future->past,
        # over-rating ``linres-lag1``-style specs. Prefer the EXPLICIT signal:
        # when fit() time-ordered the screening sample (caller passed
        # time_ordering), every spec is time-aware regardless of base shape --
        # the canonical non-monotone lag(y) base is exactly the case the
        # base-monotonicity heuristic (the None-time fallback) missed.
        base_t_aware = bool(
            getattr(self, "_screen_time_ordered_", False)
            or _is_monotone_nondecreasing(base_screen_local)
        )
        fam_rmses: dict[str, float] = {}
        per_seed_by_family: dict[str, np.ndarray] = {}
        per_bin_first_local: Optional[np.ndarray] = None
        for family in families:
            # Capture per-bin alongside RMSE in the SAME pass
            # for the first family only (per-bin breakdown only checks
            # families[0] in the legacy second pass).
            is_first_family = (family == families[0])
            want_per_bin = bool(per_bin_enabled_pre and is_first_family)
            if use_wilcoxon:
                result = _tiny_cv_rmse_y_scale_multiseed(
                    y_train=y_screen,
                    base_train=base_screen_local,
                    transform=transform,
                    fitted_params=spec.fitted_params,
                    x_train_matrix=x_matrix_local,
                    family=family,
                    n_estimators=self.config.tiny_model_n_estimators,
                    num_leaves=self.config.tiny_model_num_leaves,
                    learning_rate=self.config.tiny_model_learning_rate,
                    cv_folds=self.config.tiny_model_cv_folds,
                    n_jobs=getattr(self.config, "tiny_model_n_jobs", 1),
                    deterministic=getattr(
                        self.config, "deterministic_screening_models", False,
                    ),
                    n_seed_repeats=n_seed_repeats,
                    base_random_state=self.config.random_state,
                    inner_n_jobs=_rerank_inner_n_jobs,
                    return_per_seed=True,
                    return_per_bin=want_per_bin,
                    n_bins=per_bin_n_bins_pre or 5,
                    time_aware=base_t_aware,
                    groups=_groups_screen,
                    cv_selector_mode=_cv_sel_mode,
                    cv_selector_alpha=_cv_sel_alpha,
                    cv_selector_confidence=_cv_sel_conf,
                    cv_selector_quantile_level=_cv_sel_qlevel,
                )
                if want_per_bin:
                    rmse, per_bin_first, per_seed = (
                        result[0], result[1], result[-1],
                    )
                    per_bin_first_local = per_bin_first
                else:
                    rmse, per_seed = result[0], result[-1]
                per_seed_by_family[family] = per_seed
            else:
                result = _tiny_cv_rmse_y_scale_multiseed(
                    y_train=y_screen,
                    base_train=base_screen_local,
                    transform=transform,
                    fitted_params=spec.fitted_params,
                    x_train_matrix=x_matrix_local,
                    family=family,
                    n_estimators=self.config.tiny_model_n_estimators,
                    num_leaves=self.config.tiny_model_num_leaves,
                    learning_rate=self.config.tiny_model_learning_rate,
                    cv_folds=self.config.tiny_model_cv_folds,
                    n_jobs=getattr(self.config, "tiny_model_n_jobs", 1),
                    deterministic=getattr(
                        self.config, "deterministic_screening_models", False,
                    ),
                    n_seed_repeats=n_seed_repeats,
                    base_random_state=self.config.random_state,
                    inner_n_jobs=_rerank_inner_n_jobs,
                    return_per_bin=want_per_bin,
                    n_bins=per_bin_n_bins_pre or 5,
                    time_aware=base_t_aware,
                    groups=_groups_screen,
                    cv_selector_mode=_cv_sel_mode,
                    cv_selector_alpha=_cv_sel_alpha,
                    cv_selector_confidence=_cv_sel_conf,
                    cv_selector_quantile_level=_cv_sel_qlevel,
                )
                if want_per_bin and isinstance(result, tuple):
                    rmse, per_bin_first = result[0], result[1]
                    per_bin_first_local = per_bin_first
                else:
                    rmse = result
            fam_rmses[family] = rmse
        return spec.name, fam_rmses, per_seed_by_family, per_bin_first_local

    # ``tiny_rerank_n_jobs=0`` is the documented sentinel for "auto-pick"
    # (the branch right below this assignment). The previous ``or 1`` form
    # collapsed 0->1 BEFORE the sentinel check ran, making the auto-pick
    # branch unreachable. ``None`` (Pydantic-default-unset) still folds to
    # 1 (the historical default).
    _rerank_raw = getattr(self.config, "tiny_rerank_n_jobs", 1)
    _rerank_n_jobs_cfg = int(1 if _rerank_raw is None else _rerank_raw)
    if _rerank_n_jobs_cfg == 0:
        # Auto: cap at len(kept_specs) and at cpu_count to avoid
        # oversubscription when tiny_model_n_jobs > 1 internally.
        try:
            import os as _os
            _cpu = _os.cpu_count() or 1
        except Exception:
            _cpu = 1
        _rerank_n_jobs = max(1, min(len(kept_specs), _cpu))
    else:
        _rerank_n_jobs = max(1, _rerank_n_jobs_cfg)
    # Cap each inner LGBM/XGB to its fair share of cores when the OUTER rerank
    # runs N spec-workers in parallel (threading backend). Without this, N
    # workers x all-core boosters demand N*cpu threads on the dominant rerank
    # phase: the existing fold-level cap only fires when tiny_model_n_jobs>1
    # (default 1), so it never applied here. -1 (all cores) when sequential.
    if _rerank_n_jobs > 1:
        try:
            import os as _os
            _cpu_total = _os.cpu_count() or 1
        except Exception:
            _cpu_total = 1
        _rerank_inner_n_jobs = max(1, _cpu_total // _rerank_n_jobs)
    else:
        _rerank_inner_n_jobs = -1
    _tiny_rerank_ram_checkpoint(f"pre_parallel_loop(n_specs={len(kept_specs)}, n_families={len(families)}, rerank_n_jobs={_rerank_n_jobs}, inner_n_jobs={_rerank_inner_n_jobs})")
    if _rerank_n_jobs > 1 and len(kept_specs) > 1:
        from joblib import Parallel as _Parallel, delayed as _delayed
        _rerank_results = _Parallel(
            n_jobs=_rerank_n_jobs, backend="threading", prefer="threads",
        )(_delayed(_rerank_one_spec)(s) for s in kept_specs)
    else:
        # Sequential path: log every other spec so the kill-point is bracketed
        # without flooding the log on a 100-spec rerank. The parallel path
        # can't checkpoint mid-loop without contention on the logger.
        _rerank_results = []
        for _i, _spec in enumerate(kept_specs):
            _rerank_results.append(_rerank_one_spec(_spec))
            if _i % 2 == 1 or _i == len(kept_specs) - 1:
                _tiny_rerank_ram_checkpoint(f"after_spec[{_i + 1}/{len(kept_specs)}]={_spec.name[:40]}")
    _tiny_rerank_ram_checkpoint("post_parallel_loop_done")

    # Serial reduce — preserve spec order for per_family_scores
    # (joblib.Parallel preserves input order).
    self._wilcoxon_per_seed_composite = getattr(
        self, "_wilcoxon_per_seed_composite", {},
    )
    for _spec_name, _fam_rmses, _per_seed_by_family, _per_bin_first in _rerank_results:
        for family in families:
            per_family_scores[family].append(_fam_rmses.get(family, float("nan")))
        if _per_bin_first is not None:
            _per_bin_first_pass[_spec_name] = _per_bin_first
        for family, _per_seed in _per_seed_by_family.items():
            self._wilcoxon_per_seed_composite[(_spec_name, family)] = _per_seed

    # Aggregate -> single score per spec.
    consensus = self.config.tiny_consensus
    agg_scores: list[float] = []
    for i, _spec in enumerate(kept_specs):
        family_rmses = [per_family_scores[f][i] for f in families]
        finite = [r for r in family_rmses if math.isfinite(r)]
        if not finite:
            agg_scores.append(float("inf"))
            continue
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

    # Persist tiny CV-RMSE keyed by spec name -- callers read it
    # via :attr:`CompositeTargetDiscovery.tiny_rerank_scores_`.
    # CompositeSpec is frozen; we keep the per-spec scoring on the
    # discovery instance instead of mutating the spec.
    self._tiny_rerank_scores: dict[str, float] = {
        kept_specs[i].name: float(agg_scores[i])
        for i in range(len(kept_specs))
    }

    # Regime-aware gate. In addition to the
    # global mean RMSE, compute per-quintile-of-base RMSE for each
    # spec AND for the raw-y baseline (binned by the SAME variable
    # for apples-to-apples). A spec is rejected if it loses to
    # raw on any quintile by more than ``per_bin_tolerance`` --
    # catches "two-regime" failures where logratio is correct on
    # multiplicative rows but actively wrong on additive rows.
    per_bin_n_bins = int(getattr(self.config, "per_bin_n_bins", 0) or 0)
    per_bin_tol = float(getattr(
        self.config, "raw_baseline_per_bin_tolerance", 1.10,
    ))
    per_bin_enabled = (
        per_bin_n_bins > 0
        and getattr(self.config, "require_beats_raw_baseline", True)
    )
    # Per-spec per-bin RMSE: spec_name -> ndarray(n_bins,)
    spec_per_bin_rmse: dict[str, np.ndarray] = {}
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
                n_jobs=getattr(self.config, "tiny_model_n_jobs", 1),
                deterministic=getattr(
                    self.config, "deterministic_screening_models", False,
                ),
                return_per_bin=True, n_bins=per_bin_n_bins,
                time_aware=bool(
                    getattr(self, "_screen_time_ordered_", False)
                    or _is_monotone_nondecreasing(base_screen)
                ),
                groups=_groups_screen,
            )
            if isinstance(result, tuple):
                _, per_bin = result
                spec_per_bin_rmse[spec.name] = per_bin

    # Raw-y baseline gate. Train a tiny model directly on raw y
    # using the SAME folds / sample / family as the composite
    # rerank above, so the comparison is apples-to-apples. Reject
    # any composite whose tiny RMSE >= raw_baseline * tolerance.
    # Configured via ``require_beats_raw_baseline`` /
    # ``raw_baseline_tolerance``.
    raw_rmse_per_family: dict[str, float] = {}
    raw_per_bin_per_base: dict[str, np.ndarray] = {}
    raw_baseline: float = float("nan")
    gate_rejected_names: list[tuple[str, float, float]] = []
    per_bin_rejected_names: list[tuple[str, str, float, float]] = []
    if getattr(self.config, "require_beats_raw_baseline", True):
        # Build a feature matrix using ALL usable_features on the
        # screening sample (raw-y training has no special "base"
        # to drop, so include everything).
        x_full = self._build_feature_matrix(
            df, list(usable_features), train_idx_screen,
        )
        n_seed_repeats_raw = max(1, int(getattr(
            self.config, "tiny_model_n_seed_repeats", 1,
        )))
        use_wilcoxon = bool(getattr(
            self.config, "use_wilcoxon_gate", False,
        ))
        # Time-aware raw-y baseline: TimeSeriesSplit folds when the data is
        # temporal, so the raw-vs-composite tiny-CVs stay apples-to-apples.
        # Explicit time_ordering (screen sorted by fit) forces it for ALL
        # specs; otherwise fall back to "any spec's base is monotone". Using
        # the same predicate as the per-spec side (base_t_aware) avoids the
        # cross-scheme mismatch where a monotone base put the raw baseline
        # on TSS while non-monotone specs were scored on shuffled KFold.
        _any_base_monotone = bool(getattr(self, "_screen_time_ordered_", False)) or any(
            _is_monotone_nondecreasing(
                _per_base_cache.get(spec.base_column, (None, None))[0]
            )
            for spec in kept_specs
            if _per_base_cache.get(spec.base_column, (None, None))[0] is not None
        )
        raw_per_seed_per_family: dict[str, np.ndarray] = {}
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
                    n_jobs=getattr(self.config, "tiny_model_n_jobs", 1),
                    deterministic=getattr(
                        self.config, "deterministic_screening_models", False,
                    ),
                    n_seed_repeats=n_seed_repeats_raw,
                    base_random_state=self.config.random_state,
                    return_per_seed=True,
                    time_aware=_any_base_monotone,
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
                    n_jobs=getattr(self.config, "tiny_model_n_jobs", 1),
                    deterministic=getattr(
                        self.config, "deterministic_screening_models", False,
                    ),
                    n_seed_repeats=n_seed_repeats_raw,
                    base_random_state=self.config.random_state,
                    time_aware=_any_base_monotone,
                    groups=_groups_screen,
                    cv_selector_mode=_cv_sel_mode,
                    cv_selector_alpha=_cv_sel_alpha,
                    cv_selector_confidence=_cv_sel_conf,
                    cv_selector_quantile_level=_cv_sel_qlevel,
                )
        # Per-base raw-y per-bin breakdown for the regime gate. The raw-y model
        # is trained on ``x_full`` and is INDEPENDENT of the per-base bin_var,
        # so fit the K-fold raw-y model ONCE (capturing per-fold predictions)
        # and re-bin those cached predictions for each distinct base. Memoised
        # by base column. Bit-identical to the prior per-base refit (the binning
        # operates on the same fold predictions either way), but the dominant
        # K-fold LGBM fit no longer repeats per base.
        if per_bin_enabled:
            _raw_fold_preds = None
            # bin_var aligns to the isfinite(y_screen)-masked space, matching the
            # masking _tiny_cv_rmse_raw_y applies to bin_var internally.
            _y_screen_finite = np.isfinite(np.asarray(y_screen))
            _bin_var_needs_mask = not bool(_y_screen_finite.all())
            for spec in kept_specs:
                if spec.base_column in raw_per_bin_per_base:
                    continue
                cached = _per_base_cache.get(spec.base_column)
                if cached is None:
                    continue
                base_screen, _ = cached
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
                        n_jobs=getattr(self.config, "tiny_model_n_jobs", 1),
                        deterministic=getattr(
                            self.config, "deterministic_screening_models", False,
                        ),
                        return_fold_preds=True,
                        groups=_groups_screen,
                        time_aware=_any_base_monotone,
                    )
                    # (mean_rmse, fold_preds) when return_fold_preds=True.
                    _raw_fold_preds = (
                        raw_result[1] if isinstance(raw_result, tuple) else []
                    )
                if not _raw_fold_preds:
                    continue
                _bin_var_clean = (
                    base_screen[_y_screen_finite]
                    if _bin_var_needs_mask else base_screen
                )
                raw_per_bin = _per_bin_from_fold_preds(
                    _raw_fold_preds, _bin_var_clean, n_bins=per_bin_n_bins,
                )
                raw_per_bin_per_base[spec.base_column] = raw_per_bin
        finite_raw = [r for r in raw_rmse_per_family.values()
                      if math.isfinite(r)]
        if finite_raw:
            # Apples-to-apples with consensus aggregation above.
            if consensus == "union":
                raw_baseline = min(finite_raw)
            else:
                raw_baseline = float(np.mean(finite_raw))
        tol = float(getattr(self.config, "raw_baseline_tolerance", 1.02))
        threshold = (raw_baseline * tol
                     if math.isfinite(raw_baseline) else float("inf"))
        # Skip the composite-target block when
        # raw is already SO close to perfect that composite has no
        # headroom. Compute raw_baseline / dummy_std as a proxy for
        # "fraction of dummy's variance the raw model could not
        # explain". If that fraction is below the configured ratio,
        # short-circuit to "no kept specs" -- this avoids 5+ min per
        # spec of full per-target training that would produce
        # composite metrics indistinguishable from raw.
        _raw_skip_ratio = float(getattr(
            self.config, "composite_skip_when_raw_dominates_ratio", 0.0,
        ))
        # Complementary skip via BaselineDiagnostics ablation delta%.
        # When the top hint feature's drop balloons
        # ablation RMSE by more than the configured fraction, the raw
        # model is essentially auto-regressive on that one feature --
        # composite discovery in this regime is wasted compute (observed
        # in prod: top_ablation_delta%=3209% on the lag feature meant the raw
        # model literally IS ``y ~ lag``; 15.6 min of discovery
        # produced 1 spec that scored identically to raw).
        _ablation_skip_pct = float(getattr(
            self.config, "composite_skip_when_ablation_delta_pct", 0.0,
        ))
        _hint_strengths_pct = getattr(self, "_hint_strengths_pct", None)
        _max_ablation_pct = (
            float(max(_hint_strengths_pct))
            if (_hint_strengths_pct is not None
                and len(_hint_strengths_pct) > 0
                and all(math.isfinite(_x) for _x in _hint_strengths_pct))
            else float("nan")
        )
        if _raw_skip_ratio > 0.0 and math.isfinite(raw_baseline):
            _y_std = float(np.std(y_screen)) if y_screen.size > 1 else 0.0
            if _y_std > 0:
                _ratio = raw_baseline / _y_std
                # Skip if EITHER the raw-dominance ratio OR the BD
                # ablation delta% signal trips. They're complementary:
                # ratio catches "raw R^2 already > threshold" globally,
                # ablation catches "one single feature explains nearly
                # everything" even when the raw ridge couldn't fully
                # capture it.
                _ablation_trips = (
                    _ablation_skip_pct > 0.0
                    and math.isfinite(_max_ablation_pct)
                    and _max_ablation_pct >= _ablation_skip_pct
                    and _ratio < (_raw_skip_ratio * 3.0)  # safety: R^2 must already be > 0.99
                )
                if _ratio < _raw_skip_ratio or _ablation_trips:
                    _reason = (
                        f"ratio={_ratio:.4f} < {_raw_skip_ratio:.4f}"
                        if _ratio < _raw_skip_ratio
                        else (
                            f"BD ablation delta%={_max_ablation_pct:.1f} "
                            f">= {_ablation_skip_pct:.1f} (raw R^2 already "
                            f">0.99: ratio={_ratio:.4f})"
                        )
                    )
                    logger.info(
                        "[CompositeTargetDiscovery] target='%s' raw model "
                        "already dominates: raw_baseline=%.4f, y_std=%.4f, "
                        "%s. Composite block skipped.",
                        getattr(self, "_target_col", "?"),
                        raw_baseline, _y_std, _reason,
                    )
                    # Returning an empty kept-spec list signals to the caller that the
                    # composite block is skipped; fit's existing empty-handling path
                    # still populates ``self.report_`` from the screening-phase
                    # ``candidates`` so per-(base, transform) rejection diagnostics ARE
                    # preserved. ``candidates`` lives in fit (not _tiny_model_rerank) and
                    # the underlying attr is ``_tiny_rerank_scores`` (``tiny_rerank_scores_``
                    # is a read-only property), so rerank-level report_ entries cannot be
                    # added from this scope -- no diagnostic loss.
                    self._tiny_rerank_scores = {}
                    return []
        self._raw_y_baseline_rmse = (
            float(raw_baseline) if math.isfinite(raw_baseline)
            else float("nan")
        )
        if math.isfinite(raw_baseline):
            survivors = []
            gate_alpha = float(getattr(
                self.config, "gate_alpha", 0.05,
            ))
            wilcoxon_rejected: list[tuple[str, float]] = []
            for i, spec in enumerate(kept_specs):
                score = agg_scores[i]
                if math.isfinite(score) and score >= threshold:
                    gate_rejected_names.append(
                        (spec.name, score, threshold)
                    )
                    continue
                # Paired Wilcoxon signed-rank test
                # on per-seed RMSE diffs (composite - raw). Reject
                # spec unless the median diff is significantly
                # negative at level gate_alpha.
                if (use_wilcoxon
                        and hasattr(self, "_wilcoxon_per_seed_composite")):
                    family = families[0]
                    comp_per_seed = self._wilcoxon_per_seed_composite.get(
                        (spec.name, family)
                    )
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
                    _min_seeds_wilcoxon = int(math.ceil(
                        math.log2(1.0 / max(gate_alpha, 1e-12))
                    ))
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
                        if (comp_per_seed is not None
                            and raw_per_seed is not None
                            and len(comp_per_seed) == len(raw_per_seed))
                        else None
                    )
                    _n_paired = int(_both_finite.sum()) if _both_finite is not None else 0
                    if (_both_finite is not None
                            and _n_paired < _min_seeds_wilcoxon):
                        logger.warning(
                            "[CompositeTargetDiscovery] Wilcoxon gate skipped: "
                            "jointly-finite paired seeds=%d < %d, the minimum for "
                            "a one-sided test to reach p<=gate_alpha=%.3g "
                            "(min-p=1/2^n). Raise tiny_model_n_seed_repeats "
                            "(and/or fix the seeds that degenerated) to enable "
                            "the gate; the threshold gate still applies.",
                            _n_paired, _min_seeds_wilcoxon, gate_alpha,
                        )
                    elif (_both_finite is not None
                            and _n_paired >= _min_seeds_wilcoxon):
                        try:
                            from scipy.stats import wilcoxon
                            diff = (
                                comp_per_seed[_both_finite]
                                - raw_per_seed[_both_finite]
                            )
                            # One-sided: composite better (less RMSE)
                            # so we want diff < 0; alternative='less'.
                            stat_res = wilcoxon(
                                diff, alternative="less",
                                zero_method="wilcox",
                            )
                            p_value = float(stat_res.pvalue)
                            if p_value > gate_alpha:
                                wilcoxon_rejected.append(
                                    (spec.name, p_value)
                                )
                                continue
                        except (ImportError, ValueError) as _wx_err:
                            # Scipy missing or all-zero diffs ->
                            # fall through to the threshold-only
                            # gate (no Wilcoxon rejection).
                            logger.debug(
                                "[CompositeTargetDiscovery] "
                                "Wilcoxon gate skipped for spec=%s: %s",
                                spec.name, _wx_err,
                            )
                # Per-bin gate. Composite
                # passes the global mean test; now check that
                # per-bin RMSE doesn't blow out vs the raw-y
                # per-bin baseline on any quintile of base.
                if (per_bin_enabled
                        and spec.name in spec_per_bin_rmse
                        and spec.base_column in raw_per_bin_per_base):
                    spec_pb = spec_per_bin_rmse[spec.name]
                    raw_pb = raw_per_bin_per_base[spec.base_column]
                    # Element-wise compare, ignoring NaN bins.
                    worst_ratio = 0.0
                    worst_bin_idx = -1
                    for b in range(len(spec_pb)):
                        if (math.isfinite(spec_pb[b])
                                and math.isfinite(raw_pb[b])
                                and raw_pb[b] > 0):
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
            if not survivors:
                logger.warning(
                    "[CompositeTargetDiscovery] raw-y baseline gate "
                    "rejected ALL %d composite candidate(s) "
                    "(raw_baseline=%.4f, tolerance=%.2f). Examples: %s. "
                    "Falling back to raw target only -- discovery "
                    "yields no specs.",
                    len(gate_rejected_names),
                    raw_baseline, tol,
                    ", ".join(
                        f"{n}=RMSE{r:.4f}>{t:.4f}"
                        for n, r, t in gate_rejected_names[:3]
                    ),
                )
                return []
            if gate_rejected_names:
                logger.info(
                    "[CompositeTargetDiscovery] raw-y baseline gate "
                    "rejected %d/%d composite(s) (raw_baseline=%.4f, "
                    "tolerance=%.2f): %s",
                    len(gate_rejected_names), len(kept_specs),
                    raw_baseline, tol,
                    ", ".join(
                        f"{n}(RMSE={r:.4f}>{t:.4f})"
                        for n, r, t in gate_rejected_names
                    ),
                )
            if per_bin_rejected_names:
                logger.info(
                    "[CompositeTargetDiscovery] regime-aware per-bin "
                    "gate rejected %d composite(s) (passed global "
                    "mean but blew out on a base quintile, "
                    "tolerance=%.2f): %s",
                    len(per_bin_rejected_names), per_bin_tol,
                    ", ".join(
                        f"{n}@{b}(ratio={r:.2f}>{t:.2f})"
                        for n, b, r, t in per_bin_rejected_names[:5]
                    ),
                )
            if wilcoxon_rejected:
                logger.info(
                    "[CompositeTargetDiscovery] Wilcoxon gate "
                    "rejected %d composite(s) (paired one-sided test "
                    "on per-seed RMSE diffs vs raw, alpha=%.3f): %s",
                    len(wilcoxon_rejected), gate_alpha,
                    ", ".join(
                        f"{n}(p={p:.3f})"
                        for n, p in wilcoxon_rejected[:5]
                    ),
                )
            # Replace kept_specs/agg_scores with survivors only.
            kept_specs = [s for _, s, _ in survivors]
            agg_scores = [sc for _, _, sc in survivors]

    # Sort by aggregated score (ascending: lowest RMSE wins).
    # Stable sort + spec-name tiebreak so tied RMSE
    # (common on small synthetic / regression tests) doesn't make top-M
    # pick depend on dict iteration order.
    _names = [getattr(s, "name", str(i)) for i, s in enumerate(kept_specs)]
    order = np.lexsort((_names, agg_scores))
    reranked = [kept_specs[i] for i in order]
    # Trim to top-M.
    top_m = max(1, self.config.top_m_after_tiny)
    reranked = reranked[:top_m]
    # Logging: show the rerank effect. The "top-%d" label reflects the ACTUAL length of the survivor set (which may be smaller than ``top_m`` after the raw-y baseline gate / Wilcoxon filter), not the configured target.
    original_top = [s.name for s in kept_specs[: top_m]]
    new_top = [s.name for s in reranked]
    if original_top != new_top:
        logger.info(
            "[CompositeTargetDiscovery] tiny-model rerank changed top-%d "
            "(configured top_m=%d). Before (by mi_gain): %s. "
            "After (by CV-RMSE on y-scale): %s.",
            len(new_top), top_m, original_top, new_top,
        )
    else:
        logger.info(
            "[CompositeTargetDiscovery] tiny-model rerank kept %d spec(s) "
            "(configured top_m=%d): %s.",
            len(new_top), top_m, new_top,
        )
    _tiny_rerank_ram_checkpoint("exit")
    return reranked

