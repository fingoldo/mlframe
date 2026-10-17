"""Tiny-model rerank phase for ``CompositeTargetDiscovery``.

``_tiny_model_rerank`` is bound back onto the ``CompositeTargetDiscovery``
class at the parent's module top, so call sites that invoke
``self._tiny_model_rerank(...)`` continue to work unchanged.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Sequence

import numpy as np

from ..spec import CompositeSpec
from mlframe.training.composite.ensemble.shared import is_monotone_nondecreasing as _is_monotone_nondecreasing
from ._rejection_ledger import RejectStage, ledger_append
from ._per_base_x import PerBaseMatrices
from ._tiny_rerank_process import make_spec_task, rerank_backend, score_spec
from ._score import Score, rank_specs
from ._tiny_rerank_waic import _apply_waic_tiebreak
from .screening import (
    _extract_column_array,
    _sample_indices,
)
from ..transforms import get_transform
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)


from mlframe.utils.env_flags import env_flag

from ._tiny_rerank_blocks import (
    _tiny_model_rerank_aggregate_single_per,
    _tiny_model_rerank_family_families,
    _tiny_model_rerank_finite_raw_early,
    _tiny_model_rerank_np_arange_filtered_train,
    _tiny_model_rerank_per_bin_enabled_pre,
    _tiny_model_rerank_per_spec_per_bin,
    _tiny_model_rerank_rerank_jobs_len_kept,
    _tiny_model_rerank_self_config_tiny_screening,
    _tiny_model_rerank_spec_enumerate_kept_specs,
    _tiny_model_rerank_spec_name_fam_errs,
)
from ._tiny_rerank_honest import (
    _tiny_rerank_ram_checkpoint,
    _honest_oof_prepass,
    _apply_honest_oof_ordering,
)


def _reject_unscored_specs(self, kept_specs: list, agg_scores: list) -> tuple[list, list]:
    """Drop, with a ledger entry, every spec whose aggregated tiny-CV score is not finite.

    A non-finite score means the tiny CV failed in every family (non-finite T on domain-valid rows makes each family's
    RMSE NaN) or the honest reconstruction collapsed. The raw-baseline gate compared only finite scores, and with a
    non-finite raw baseline no gate ran at all, so such a spec sorted last and still shipped when fewer than top_m
    candidates survived: a failure inside the gate disabled the gate for the spec that failed.
    """
    kept, scores = [], []
    for spec, score in zip(kept_specs, agg_scores):
        if math.isfinite(score):
            kept.append(spec)
            scores.append(score)
            continue
        log_throttle(logger, "composite_tiny_rerank_reject", logging.WARNING, "[CompositeTargetDiscovery] rejecting %s: its tiny-CV score is %s (the CV failed in every family).", spec.name, score)
        ledger_append(
            self, spec_name=spec.name, stage=RejectStage.TINY_RERANK_THRESHOLD,
            reason=f"tiny-rerank CV-RMSE is {score} (failed in every family or collapsed)",
            base_column=getattr(spec, "base_column", ""), transform_name=getattr(spec, "transform_name", ""),
        )
    return kept, scores


def _record_rerank_sample_time(self, train_idx_screen: np.ndarray) -> None:
    """Keep the time key of the rerank sample as the folds will see it, on ``_rerank_sample_time_``.

    A forward split over this sample is a forward split in time only when this is non-decreasing, and that is the property a
    caller (or a test) can check after the fact; the "screen time ordered" flag says the MI screen was sorted, which is a
    different sample.
    """
    time_all = getattr(self, "_time_ordering_", None)
    self._rerank_sample_time_ = None if time_all is None else np.asarray(time_all)[train_idx_screen]

def _resolve_rerank_n_jobs(raw, n_specs: int) -> int:
    """Outer spec-worker count for the rerank: ``0`` auto-picks, ``None`` is the historical 1, anything else is taken as given (floored at 1).

    Auto caps at the spec count and at the PHYSICAL core count: logical/SMT siblings do not help GIL-releasing booster threads and only add scheduler churn.
    """
    cfg = int(1 if raw is None else raw)
    if cfg == 0:
        from pyutilz.parallel import cpu_count_physical

        return max(1, min(n_specs, int(cpu_count_physical())))
    return max(1, cfg)


def _tiny_n_jobs_auto_cores(self) -> int:
    """CV-fold joblib parallelism for the tiny models: ``tiny_model_n_jobs=0`` (or unset) means the physical core count.

    Physical, NOT logical: oversubscribing CV-fold joblib past physical cores only adds context-switch churn. The serial raw-y baseline gate
    runs AFTER the per-spec parallel barrier, so it gets the full machine; the per-spec worker (already under an outer threading Parallel)
    keeps fold-joblib at 1 to avoid 3-level nesting (outer spec-threads x fold-joblib x LightGBM inner_n_jobs).
    """
    from pyutilz.parallel import cpu_count_physical

    raw_njobs_cfg = getattr(self.config, "tiny_model_n_jobs", 0)
    return int(cpu_count_physical()) if raw_njobs_cfg in (0, None) else max(1, int(raw_njobs_cfg))


def _rerank_screen_sample(self, train_idx: np.ndarray, y_full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Draw the rerank screening sample of train rows (time-ordered when the caller supplied a time key); returns ``(train_idx_screen, y_screen)``."""
    sample_n = min(self.config.tiny_model_sample_n, train_idx.size)
    # Phase B benefits from stratified sampling on heavy-tail y
    # for the same reason Phase A does -- tiny-model CV-RMSE on a
    # tail-empty sample mis-ranks transforms that only matter in
    # the tail.
    y_train_for_strat = y_full[train_idx]
    sample_idx = _sample_indices(
        train_idx.size, sample_n, self.config.random_state,
        strategy=getattr(self.config, "mi_sample_strategy", 'stratified_quantile'),
        y=y_train_for_strat,
        n_strata=getattr(self.config, "mi_n_strata", 10),
    )
    train_idx_screen = train_idx[sample_idx]
    # This sample is drawn here, so the MI screen's time sort does not reach it. Re-apply the caller's time key, or the
    # TimeSeriesSplit below walks the sampler's row order while reporting itself as time-aware.
    from ._fit_temporal import order_rows_by_time

    _time_order = order_rows_by_time(train_idx_screen, getattr(self, "_time_ordering_", None))
    if _time_order is not None:
        train_idx_screen = train_idx_screen[_time_order]
    _record_rerank_sample_time(self, train_idx_screen)
    return train_idx_screen, y_full[train_idx_screen]


def _build_rerank_base_cache(
    self, df: Any, usable_features: Sequence[str], kept_specs: list[CompositeSpec], train_idx_screen: np.ndarray
) -> tuple[np.ndarray, PerBaseMatrices]:
    """Build the screen-sample feature matrix once plus the per-base cache (base vectors extracted serially before the parallel rerank).

    The full matrix is built ONCE, then each base's "all features except this base" matrix is derived via np.delete on the in-RAM matrix instead
    of re-extracting full-column passes from the frame. np.delete preserves the surviving column order, so x_full minus the base column index is
    bit-identical to building from ``usable_features`` minus that column.
    """
    _usable_list = list(usable_features)
    _x_full = self._build_feature_matrix(df, _usable_list, train_idx_screen)
    # Base vectors are extracted here; each base's X-without-base is gathered from _x_full on first use and at most two
    # stay cached (``PerBaseMatrices``), so the rerank no longer holds one copy per distinct base.
    _base_screens: dict = {}
    for spec in kept_specs:
        if spec.base_column in _base_screens:
            continue
        # Unary (``requires_base=False``) specs carry an empty
        # ``base_column`` sentinel -- they ignore the base entirely. Extracting
        # a column named "" would crash, so synthesise a zeros ``base_screen``
        # placeholder (never read by a unary transform's forward) and score
        # against the FULL feature matrix (no base column dropped), mirroring
        # the dedicated unary context built in ``discovery/_fit.py``.
        if spec.base_column == "":
            _base_screens[spec.base_column] = np.zeros(train_idx_screen.size, dtype=np.float32)
            continue
        # A base not among the screened features (e.g. a lag column the user excluded) keeps the full matrix; a
        # synthetic interaction base drops its parents, as the screen does.
        _base_screens[spec.base_column] = _extract_column_array(df, spec.base_column, rows=train_idx_screen)
    _per_base_cache = PerBaseMatrices(_x_full, _usable_list, _base_screens)
    _tiny_rerank_ram_checkpoint(f"per_base_cache_built(n_unique_bases={len(_per_base_cache)})")
    return _x_full, _per_base_cache


def _cv_selector_knobs(self) -> tuple[str, float, float, float]:
    """The CV fold-score selector knobs ``(mode, alpha, confidence, quantile_level)`` forwarded into every tiny-CV call.

    The rerank ranks specs by the same selector the production split uses; default 'mean' keeps this bit-identical.
    """
    return (
        str(getattr(self.config, "cv_selector_mode", "mean")),
        float(getattr(self.config, "cv_selector_alpha", 1.0)),
        float(getattr(self.config, "cv_selector_confidence", 0.9)),
        float(getattr(self.config, "cv_selector_quantile_level", 0.9)),
    )


def _early_raw_baseline(
    self,
    kept_specs: list[CompositeSpec],
    families: Sequence[str],
    use_wilcoxon: bool,
    y_screen: np.ndarray,
    x_full: np.ndarray,
    tiny_n_jobs_auto: int,
    groups_screen: Any,
    per_base_cache: PerBaseMatrices,
    per_bin_enabled_pre: bool,
    per_bin_n_bins_pre: int,
    cv_sel: tuple[str, float, float, float],
) -> tuple[dict[str, float], dict[str, np.ndarray], float, dict[str, np.ndarray], bool]:
    """Early raw-y baseline pass, run BEFORE the per-candidate scoring so its rejection threshold can feed the sequential multiseed early-stop.

    A candidate whose running seed-median is already guaranteed to clear the threshold (see ``_seed_median_lower_bound`` in ``_screening_tiny.py``)
    skips its remaining seeds. The later raw-baseline gate reuses these values instead of recomputing them, so total raw-y compute is unchanged.
    Returns ``(raw_rmse_per_family, raw_per_bin_per_base, raw_baseline, raw_per_seed_per_family, any_base_monotone)``.
    """
    _cv_sel_mode, _cv_sel_alpha, _cv_sel_conf, _cv_sel_qlevel = cv_sel
    raw_rmse_per_family: dict[str, float] = {}
    raw_per_bin_per_base: dict[str, np.ndarray] = {}
    raw_baseline: float = float("nan")
    raw_per_seed_per_family: dict[str, np.ndarray] = {}
    # One fold scheme for raw-y and every spec (per-spec schemes made the threshold and Wilcoxon compare different fold RMSEs).
    _early_any_base_monotone = bool(getattr(self, "_screen_time_ordered_", False) or (
        groups_screen is None
        and any(_is_monotone_nondecreasing(_base_arr) for spec in kept_specs if (_base_arr := per_base_cache.base_screen(spec.base_column)) is not None)
    ))
    if bool(getattr(self.config, "require_beats_raw_baseline", False)):
        n_seed_repeats_raw = max(1, int(getattr(
            self.config, "tiny_model_n_seed_repeats", 3,
        )))
        _tiny_model_rerank_family_families(self, families, use_wilcoxon, y_screen, x_full, tiny_n_jobs_auto, n_seed_repeats_raw, _early_any_base_monotone, groups_screen, _cv_sel_mode, _cv_sel_alpha, _cv_sel_conf, _cv_sel_qlevel, raw_rmse_per_family, raw_per_seed_per_family)
        _tiny_model_rerank_per_bin_enabled_pre(self, per_bin_enabled_pre, y_screen, kept_specs, raw_per_bin_per_base, per_base_cache, families, x_full, tiny_n_jobs_auto, groups_screen, _early_any_base_monotone, per_bin_n_bins_pre)
        _finite_raw_early = [r for r in raw_rmse_per_family.values() if math.isfinite(r)]
        raw_baseline = _tiny_model_rerank_finite_raw_early(self, _finite_raw_early, raw_baseline)
    return raw_rmse_per_family, raw_per_bin_per_base, raw_baseline, raw_per_seed_per_family, _early_any_base_monotone


def _multiseed_early_stop_threshold(self, raw_baseline: float, honest_oof_will_run: bool, use_wilcoxon: bool) -> float:
    """Sequential multiseed early-stop threshold, ``inf`` (no early stop) unless it is provably sound.

    Sound only when (a) the flag is on, (b) a finite raw baseline was measured, and (c) honest-OOF selection will NOT later override the threshold
    with a different (possibly higher) value -- honest-OOF's reconstruction RMSE is only known AFTER candidates are scored, so the raw-baseline
    threshold could be unsafely tight if honest-OOF ends up more lenient. Candidates then run all seeds.
    """
    if (
        bool(getattr(self.config, "enable_multiseed_early_stop", True))
        and math.isfinite(raw_baseline)
        and not honest_oof_will_run
        and not use_wilcoxon  # Wilcoxon needs the FULL per-seed sample for the paired test power count; skip early-stop rather than prove that interaction safe too.
    ):
        return raw_baseline * float(getattr(self.config, "raw_baseline_tolerance", 1.02))
    return float("inf")


def _score_specs_parallel(
    self,
    kept_specs: list[CompositeSpec],
    families: Sequence[str],
    task_common: dict[str, Any],
    skip_cv_names: set[str],
    per_base_cache: PerBaseMatrices,
    tiny_n_jobs_auto: int,
) -> list:
    """Resolve the outer / inner / fold job split and score every kept spec, in spec order; returns the per-spec worker results."""
    from pyutilz.parallel import cpu_count_physical

    # ``tiny_rerank_n_jobs=0`` is the documented sentinel for "auto-pick"
    # (the branch right below this assignment). The previous ``or 1`` form
    # collapsed 0->1 BEFORE the sentinel check ran, making the auto-pick
    # branch unreachable. ``None`` (Pydantic-default-unset) still folds to
    # 1 (the historical default).
    _rerank_n_jobs = _resolve_rerank_n_jobs(getattr(self.config, "tiny_rerank_n_jobs", 0), len(kept_specs))
    # Cap each inner LGBM/XGB to its fair share of cores when the OUTER rerank
    # runs N spec-workers in parallel (threading backend). Without this, N
    # workers x all-core boosters demand N*cpu threads on the dominant rerank
    # phase: the existing fold-level cap only fires when tiny_model_n_jobs>1
    # (default 1), so it never applied here. -1 (all cores) when sequential.
    if _rerank_n_jobs > 1:
        _cpu_total = cpu_count_physical()
        _rerank_inner_n_jobs = max(1, _cpu_total // _rerank_n_jobs)
    else:
        _rerank_inner_n_jobs = -1
    # Per-spec worker fold-joblib: under an outer threading Parallel keep it serial (1) so the
    # core budget is outer-spec-threads x inner_n_jobs, never x a third fold-joblib layer. Only
    # the sequential rerank path (outer == 1) lets the worker parallelise CV folds across cores.
    _worker_fold_n_jobs = 1 if _rerank_n_jobs > 1 else tiny_n_jobs_auto

    def _rerank_one_spec(spec: CompositeSpec):
        """Per-spec worker for the in-process rerank loop: ``(spec.name, family_rmses, per_seed_by_family, per_bin_first)``."""
        base_screen_local, x_matrix_local = per_base_cache[spec.base_column]
        task = make_spec_task(
            spec, get_transform(spec.transform_name), dict(task_common, fold_n_jobs=_worker_fold_n_jobs, inner_n_jobs=_rerank_inner_n_jobs),
            spec.name in skip_cv_names, base_screen_local, x_matrix=x_matrix_local,
        )
        return score_spec(task)

    _tiny_rerank_ram_checkpoint(f"pre_parallel_loop(n_specs={len(kept_specs)}, n_families={len(families)}, rerank_n_jobs={_rerank_n_jobs}, inner_n_jobs={_rerank_inner_n_jobs}, worker_fold_n_jobs={_worker_fold_n_jobs})")
    _backend = rerank_backend(getattr(self.config, "tiny_rerank_backend", "auto"))
    _rerank_results: list = _tiny_model_rerank_rerank_jobs_len_kept(_rerank_n_jobs, kept_specs, _backend, task_common, _worker_fold_n_jobs, _rerank_inner_n_jobs, skip_cv_names, per_base_cache, _rerank_one_spec)
    _tiny_rerank_ram_checkpoint("post_parallel_loop_done")
    return _rerank_results


def _raw_dominates_skip(self, raw_baseline: float, y_screen: np.ndarray, kept_specs: list[CompositeSpec]) -> bool:
    """True (after logging and ledgering every spec) when the raw model already dominates, so the composite block has no headroom.

    Compute raw_baseline / dummy_std as a proxy for "fraction of dummy's variance the raw model could not explain"; below the configured ratio
    the composite block is skipped -- this avoids 5+ min per spec of full per-target training that would produce composite metrics
    indistinguishable from raw. A complementary skip fires via the BaselineDiagnostics ablation delta%: when the top hint feature's drop balloons
    ablation RMSE by more than the configured fraction, the raw model is essentially auto-regressive on that one feature (observed in prod:
    top_ablation_delta%=3209% on the lag feature meant the raw model literally IS ``y ~ lag``).
    """
    _raw_skip_ratio = float(getattr(
        self.config, "composite_skip_when_raw_dominates_ratio", 0.0,
    ))
    _ablation_skip_pct = float(getattr(
        self.config, "composite_skip_when_ablation_delta_pct", 0.0,
    ))
    _hint_strengths_pct = getattr(self, "_hint_strengths_pct", None)
    _max_ablation_pct = (
        float(max(_hint_strengths_pct))
        if (_hint_strengths_pct is not None and len(_hint_strengths_pct) > 0 and all(math.isfinite(_x) for _x in _hint_strengths_pct))
        else float("nan")
    )
    if not (_raw_skip_ratio > 0.0 and math.isfinite(raw_baseline)):
        return False
    _y_std = float(np.std(y_screen)) if y_screen.size > 1 else 0.0
    if _y_std <= 0:
        return False
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
    if not (_ratio < _raw_skip_ratio or _ablation_trips):
        return False
    _reason = (
        f"ratio={_ratio:.4f} < {_raw_skip_ratio:.4f}"
        if _ratio < _raw_skip_ratio
        else (f"BD ablation delta%={_max_ablation_pct:.1f} " f">= {_ablation_skip_pct:.1f} (raw R^2 already " f">0.99: ratio={_ratio:.4f})")
    )
    logger.info(
        "[CompositeTargetDiscovery] target='%s' raw model " "already dominates: raw_baseline=%.4f, y_std=%.4f, " "%s. Composite block skipped.",
        getattr(self, "_target_col", "?"),
        raw_baseline, _y_std, _reason,
    )
    # An empty kept-spec list signals to the caller that the
    # composite block is skipped; fit's existing empty-handling path
    # still populates ``self.report_`` from the screening-phase
    # ``candidates`` so per-(base, transform) rejection diagnostics ARE
    # preserved. ``candidates`` lives in fit (not _tiny_model_rerank) and
    # the underlying attr is ``_tiny_rerank_scores`` (``tiny_rerank_scores_``
    # is a read-only property), so rerank-level report_ entries cannot be
    # added from this scope -- no diagnostic loss.
    self._tiny_rerank_scores = {}
    for _s in kept_specs:
        ledger_append(
            self, spec_name=getattr(_s, "name", "?"), stage=RejectStage.RAW_DOMINATES_SKIP,
            reason=_reason, base_column=getattr(_s, "base_column", ""),
            transform_name=getattr(_s, "transform_name", ""),
        )
    return True


def _log_gate_rejections(
    n_kept: int,
    raw_baseline: float,
    tol: float,
    gate_rejected_names: list[tuple[str, float, float]],
    per_bin_rejected_names: list[tuple[str, str, float, float]],
    per_bin_tol: float,
    wilcoxon_rejected: list[tuple[str, float]],
    gate_alpha: float,
) -> None:
    """Log what the raw-baseline, regime-aware per-bin and Wilcoxon gates rejected (the all-rejected WARNING is emitted by the caller)."""
    if gate_rejected_names:
        logger.info(
            "[CompositeTargetDiscovery] raw-y baseline gate " "rejected %d/%d composite(s) (raw_baseline=%.4f, " "tolerance=%.2f): %s",
            len(gate_rejected_names),
            n_kept,
            raw_baseline,
            tol,
            ", ".join(f"{n}(RMSE={r:.4f}>{t:.4f})" for n, r, t in gate_rejected_names),
        )
    if per_bin_rejected_names:
        logger.info(
            "[CompositeTargetDiscovery] regime-aware per-bin "
            "gate rejected %d composite(s) (passed global "
            "mean but blew out on a base quintile, "
            "tolerance=%.2f): %s",
            len(per_bin_rejected_names),
            per_bin_tol,
            ", ".join(f"{n}@{b}(ratio={r:.2f}>{t:.2f})" for n, b, r, t in per_bin_rejected_names[:5]),
        )
    if wilcoxon_rejected:
        logger.info(
            "[CompositeTargetDiscovery] Wilcoxon gate " "rejected %d composite(s) (paired one-sided test " "on per-seed RMSE diffs vs raw, alpha=%.3f): %s",
            len(wilcoxon_rejected),
            gate_alpha,
            ", ".join(f"{n}(p={p:.3f})" for n, p in wilcoxon_rejected[:5]),
        )


def _apply_raw_baseline_gate(
    self,
    kept_specs: list[CompositeSpec],
    agg_scores: list[float],
    *,
    raw_baseline: float,
    honest_oof_baseline: float,
    raw_per_seed_per_family: dict[str, np.ndarray],
    raw_per_bin_per_base: dict[str, np.ndarray],
    spec_per_bin_rmse: dict[str, np.ndarray],
    per_bin_enabled: bool,
    per_bin_tol: float,
    families: Sequence[str],
    use_wilcoxon: bool,
    y_screen: np.ndarray,
) -> tuple[list[CompositeSpec], list[float]] | None:
    """Raw-y baseline gate: reject any composite whose tiny RMSE >= raw_baseline * tolerance (plus per-bin and Wilcoxon gates).

    The raw-y model was trained on the SAME folds / sample / family as the composite rerank, so the comparison is apples-to-apples; the values come
    from the early raw-y pass. Returns the surviving ``(kept_specs, agg_scores)``, or ``None`` when the composite block must yield no specs
    (raw model dominates, or every candidate was rejected).
    """
    gate_rejected_names: list[tuple[str, float, float]] = []
    per_bin_rejected_names: list[tuple[str, str, float, float]] = []
    tol = float(getattr(self.config, "raw_baseline_tolerance", 1.02))
    # When honest-OOF selection is active and produced a finite raw-y honest-OOF baseline, gate against the SAME
    # honest objective the ordering now uses, so the gate and the rank are consistent (a spec ranked by its honest
    # reconstruction RMSE is rejected against the raw honest reconstruction, not the group-internal raw CV).
    if math.isfinite(honest_oof_baseline):
        raw_baseline = honest_oof_baseline
        tol = float(getattr(self.config, "honest_oof_selection_tolerance", 1.05))
    threshold = raw_baseline * tol if math.isfinite(raw_baseline) else float("inf")
    if _raw_dominates_skip(self, raw_baseline, y_screen, kept_specs):
        return None
    self._raw_y_baseline_rmse = float(raw_baseline) if math.isfinite(raw_baseline) else float("nan")
    if not math.isfinite(raw_baseline):
        return kept_specs, agg_scores
    survivors: list[Any] = []
    gate_alpha = float(getattr(self.config, "gate_alpha", 0.05))
    wilcoxon_rejected: list[tuple[str, float]] = []
    _tiny_model_rerank_spec_enumerate_kept_specs(self, kept_specs, agg_scores, threshold, gate_rejected_names, use_wilcoxon, families, raw_per_seed_per_family, gate_alpha, wilcoxon_rejected, per_bin_enabled, spec_per_bin_rmse, raw_per_bin_per_base, per_bin_tol, per_bin_rejected_names, survivors)
    if not survivors:
        logger.warning(
            "[CompositeTargetDiscovery] raw-y baseline gate "
            "rejected ALL %d composite candidate(s) "
            "(raw_baseline=%.4f, tolerance=%.2f). Examples: %s. "
            "Falling back to raw target only -- discovery "
            "yields no specs.",
            len(gate_rejected_names),
            raw_baseline,
            tol,
            ", ".join(f"{n}=RMSE{r:.4f}>{t:.4f}" for n, r, t in gate_rejected_names[:3]),
        )
        return None
    _log_gate_rejections(len(kept_specs), raw_baseline, tol, gate_rejected_names, per_bin_rejected_names, per_bin_tol, wilcoxon_rejected, gate_alpha)
    # Replace kept_specs/agg_scores with survivors only.
    return [s for _, s, _ in survivors], [sc for _, _, sc in survivors]


def _rank_and_trim_reranked(
    self,
    kept_specs: list[CompositeSpec],
    agg_scores: list[float],
    *,
    y_screen: np.ndarray,
    per_base_cache: PerBaseMatrices,
    groups_screen: Any,
    time_aware: bool,
) -> list[CompositeSpec]:
    """Sort by aggregated score (lowest RMSE first), apply the WAIC tie-break, trim to top-M and log the rerank effect."""
    # Stable sort + spec-name tiebreak so tied RMSE
    # (common on small synthetic / regression tests) doesn't make top-M
    # pick depend on dict iteration order.
    _names = [getattr(s, "name", str(i)) for i, s in enumerate(kept_specs)]
    # One unit: the aggregated y-scale RMSE, unmeasured specs already put on the honest scale above.
    order = np.asarray(rank_specs(range(len(kept_specs)), lambda i: Score(agg_scores[i], "y_rmse", "tiny_rerank", "tiny_consensus", kept_specs[i].transform_name),
                                  name=lambda i: _names[i], transform=lambda i: kept_specs[i].transform_name), dtype=np.int64)

    # WAIC tie-break (config ``transform_waic_validation_enabled``). Tiny-CV RMSE on the small screening sample can
    # rank two transforms within measurement noise while one genuinely generalises and the other has merely memorised
    # the fold. Within a relative-RMSE noise band, re-order by WAIC (out-of-fold predictive density penalised for
    # across-fold instability; higher = better) so the overfit candidate sinks below the generalising one. Only a band
    # in which EVERY tied spec has a valid WAIC is re-ordered -- specs without a score keep their RMSE+name position, so
    # the tie-break never penalises an unscored candidate, and bands wider than the tolerance keep the raw RMSE order.
    if bool(getattr(self.config, "transform_waic_validation_enabled", True)) and len(order) > 1:
        order = _apply_waic_tiebreak(
            self, order, kept_specs, agg_scores, _names,
            y_screen=y_screen, per_base_cache=per_base_cache, groups=groups_screen, time_aware=time_aware,
        )

    reranked = [kept_specs[i] for i in order]
    # Trim to top-M.
    top_m = max(1, self.config.top_m_after_tiny)
    reranked = reranked[:top_m]
    # Logging: show the rerank effect. The "top-%d" label reflects the ACTUAL length of the survivor set (which may be smaller than ``top_m`` after the raw-y baseline gate / Wilcoxon filter), not the configured target.
    original_top = [s.name for s in kept_specs[:top_m]]
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
    return reranked


def _rerank_task_common(
    self,
    y_screen: np.ndarray,
    families: Sequence[str],
    per_bin_enabled_pre: bool,
    per_bin_n_bins_pre: int,
    use_wilcoxon: bool,
    n_seed_repeats: int,
    time_aware: bool,
    groups_screen: Any,
    cv_sel: tuple[str, float, float, float],
    early_stop_threshold: float,
) -> dict[str, Any]:
    """Everything a spec's scoring reads besides the spec itself, as values.

    The same task runs in this process (serial / threads) or in a worker process, through one implementation (``_tiny_rerank_process.score_spec``).
    """
    _cv_sel_mode, _cv_sel_alpha, _cv_sel_conf, _cv_sel_qlevel = cv_sel
    return dict(
        y_screen=y_screen, families=list(families), per_bin_enabled=bool(per_bin_enabled_pre), per_bin_n_bins=per_bin_n_bins_pre if per_bin_n_bins_pre is not None else 5,
        use_wilcoxon=bool(use_wilcoxon), n_estimators=self.config.tiny_model_n_estimators, num_leaves=self.config.tiny_model_num_leaves,
        learning_rate=self.config.tiny_model_learning_rate, cv_folds=self.config.tiny_model_cv_folds,
        deterministic=getattr(self.config, "deterministic_screening_models", False), n_seed_repeats=n_seed_repeats,
        random_state=self.config.random_state, time_aware=time_aware, groups=groups_screen,
        cv_selector_mode=_cv_sel_mode, cv_selector_alpha=_cv_sel_alpha, cv_selector_confidence=_cv_sel_conf,
        cv_selector_quantile_level=_cv_sel_qlevel, early_stop_threshold=early_stop_threshold,
    )


def _honest_prepass_stage(
    self,
    df: Any,
    target_col: str,
    kept_specs: list[CompositeSpec],
    usable_features: Sequence[str],
    train_idx: np.ndarray,
    y_full: np.ndarray,
    per_bin_enabled_pre: bool,
    use_wilcoxon: bool,
) -> tuple[bool, Any, set[str]]:
    """Run the honest-OOF prepass; returns ``(will_run, prepass_result, spec names whose CV score would be discarded)``."""
    _honest_oof_will_run = (
        bool(getattr(self.config, "honest_oof_selection", True))
        and getattr(self, "_group_ids_for_rerank", None) is not None
        and getattr(self, "honest_holdout_idx_", None) is not None
    )
    _honest_oof_pre = _honest_oof_prepass(
        self, df, target_col, kept_specs, usable_features, train_idx, y_full,
        will_run=_honest_oof_will_run, per_bin_enabled=per_bin_enabled_pre, use_wilcoxon=use_wilcoxon,
    )
    # Spec names whose CV score would be discarded: the worker returns immediately for these.
    return _honest_oof_will_run, _honest_oof_pre, (set(_honest_oof_pre) if _honest_oof_pre is not None else set())


def _per_spec_per_bin_stage(
    self,
    kept_specs: list[CompositeSpec],
    per_bin_first_pass: dict[str, np.ndarray],
    per_base_cache: PerBaseMatrices,
    families: Sequence[str],
    y_screen: np.ndarray,
    tiny_n_jobs_auto: int,
    any_base_monotone: bool,
    groups_screen: Any,
) -> tuple[bool, float, dict[str, np.ndarray]]:
    """Regime-aware gate inputs: ``(per_bin_enabled, per_bin_tolerance, per-spec per-bin RMSE)``.

    In addition to the global mean RMSE, compute per-quintile-of-base RMSE for each spec AND for the raw-y baseline (binned by the SAME variable
    for apples-to-apples). A spec is rejected if it loses to raw on any quintile by more than ``per_bin_tolerance`` -- catches "two-regime"
    failures where logratio is correct on multiplicative rows but actively wrong on additive rows.
    """
    per_bin_n_bins = int(getattr(self.config, "per_bin_n_bins", 0) or 0)
    per_bin_tol = float(getattr(self.config, "raw_baseline_per_bin_tolerance", 1.10))
    per_bin_enabled = per_bin_n_bins > 0 and getattr(self.config, "require_beats_raw_baseline", False)
    # Per-spec per-bin RMSE: spec_name -> ndarray(n_bins,)
    spec_per_bin_rmse: dict[str, np.ndarray] = {}
    _tiny_model_rerank_per_spec_per_bin(self, per_bin_enabled, kept_specs, per_bin_first_pass, spec_per_bin_rmse, per_base_cache, families, y_screen, tiny_n_jobs_auto, per_bin_n_bins, any_base_monotone, groups_screen)
    return per_bin_enabled, per_bin_tol, spec_per_bin_rmse


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
    if env_flag("MLFRAME_DISCOVERY_SKIP_TINY_RERANK"):
        logger.warning(
            "[CompositeTargetDiscovery.tiny_rerank] SKIPPED via MLFRAME_DISCOVERY_SKIP_TINY_RERANK=1; "
            "returning %d MI-ranked spec(s) without y-scale CV rerank. "
            "Spec ranking accuracy is reduced; set the env var to 0 (or unset) to restore the rerank pass.",
            len(kept_specs),
        )
        return kept_specs
    _tiny_rerank_ram_checkpoint("entry")
    _tiny_n_jobs_auto = _tiny_n_jobs_auto_cores(self)
    train_idx_screen, y_screen = _rerank_screen_sample(self, train_idx, y_full)

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
    _groups_screen = _tiny_model_rerank_np_arange_filtered_train(_groups_full_for_rerank, train_idx_screen, _groups_screen)

    families = _tiny_model_rerank_self_config_tiny_screening(self)

    # Hoist per-bin-enabled check above the first pass so we
    # request per-bin RMSE during the SAME multiseed sweep that produces
    # the global scores. The legacy second pass refit every fold to get
    # per-bin breakdowns; with this change the K-fold LGBM fit count is
    # halved when the regime-aware gate is on.
    per_bin_n_bins_pre = int(getattr(self.config, "per_bin_n_bins", 0) or 0)
    per_bin_enabled_pre = per_bin_n_bins_pre > 0 and getattr(self.config, "require_beats_raw_baseline", False)

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

    _x_full, _per_base_cache = _build_rerank_base_cache(self, df, usable_features, kept_specs, train_idx_screen)

    n_seed_repeats = max(1, int(getattr(
        self.config, "tiny_model_n_seed_repeats", 3,
    )))
    use_wilcoxon = bool(getattr(
        self.config, "use_wilcoxon_gate", False,
    ))
    _cv_sel = _cv_selector_knobs(self)

    _, raw_per_bin_per_base, raw_baseline, raw_per_seed_per_family, _early_any_base_monotone = _early_raw_baseline(
        self, kept_specs, families, use_wilcoxon, y_screen, _x_full, _tiny_n_jobs_auto, _groups_screen, _per_base_cache,
        per_bin_enabled_pre, per_bin_n_bins_pre, _cv_sel,
    )

    _honest_oof_will_run, _honest_oof_pre, _skip_cv_names = _honest_prepass_stage(
        self, df, target_col, kept_specs, usable_features, train_idx, y_full, per_bin_enabled_pre, use_wilcoxon,
    )
    _early_stop_threshold = _multiseed_early_stop_threshold(self, raw_baseline, _honest_oof_will_run, use_wilcoxon)

    _task_common = _rerank_task_common(
        self, y_screen, families, per_bin_enabled_pre, per_bin_n_bins_pre, use_wilcoxon, n_seed_repeats, _early_any_base_monotone, _groups_screen,
        _cv_sel, _early_stop_threshold,
    )
    _rerank_results = _score_specs_parallel(self, kept_specs, families, _task_common, _skip_cv_names, _per_base_cache, _tiny_n_jobs_auto)

    # Serial reduce — preserve spec order for per_family_scores
    # (joblib.Parallel preserves input order).
    self._wilcoxon_per_seed_composite = getattr(
        self, "_wilcoxon_per_seed_composite", {},
    )
    _tiny_model_rerank_spec_name_fam_errs(self, _rerank_results, families, per_family_scores, _per_bin_first_pass)

    # Aggregate -> single score per spec.
    consensus = self.config.tiny_consensus
    agg_scores: list[float] = []
    _tiny_model_rerank_aggregate_single_per(kept_specs, families, per_family_scores, agg_scores, consensus)

    # Persist tiny CV-RMSE keyed by spec name -- callers read it
    # via :attr:`CompositeTargetDiscovery.tiny_rerank_scores_`.
    # CompositeSpec is frozen; we keep the per-spec scoring on the
    # discovery instance instead of mutating the spec.
    self._tiny_rerank_scores = {kept_specs[i].name: float(agg_scores[i]) for i in range(len(kept_specs))}

    kept_specs, agg_scores, _honest_oof_baseline = _apply_honest_oof_ordering(
        self, df, target_col, kept_specs, agg_scores, usable_features, train_idx, y_full, _honest_oof_pre, raw_cv_baseline=raw_baseline,
    )
    kept_specs, agg_scores = _reject_unscored_specs(self, kept_specs, agg_scores)

    per_bin_enabled, per_bin_tol, spec_per_bin_rmse = _per_spec_per_bin_stage(
        self, kept_specs, _per_bin_first_pass, _per_base_cache, families, y_screen, _tiny_n_jobs_auto, _early_any_base_monotone, _groups_screen,
    )

    if getattr(self.config, "require_beats_raw_baseline", False):
        gated = _apply_raw_baseline_gate(
            self, kept_specs, agg_scores, raw_baseline=raw_baseline, honest_oof_baseline=_honest_oof_baseline,
            raw_per_seed_per_family=raw_per_seed_per_family, raw_per_bin_per_base=raw_per_bin_per_base, spec_per_bin_rmse=spec_per_bin_rmse,
            per_bin_enabled=per_bin_enabled, per_bin_tol=per_bin_tol, families=families, use_wilcoxon=use_wilcoxon, y_screen=y_screen,
        )
        if gated is None:
            return []
        kept_specs, agg_scores = gated

    reranked = _rank_and_trim_reranked(
        self, kept_specs, agg_scores, y_screen=y_screen, per_base_cache=_per_base_cache, groups_screen=_groups_screen, time_aware=_early_any_base_monotone,
    )
    _tiny_rerank_ram_checkpoint("exit")
    return reranked
