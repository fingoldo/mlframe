"""The main ``fit`` method for ``CompositeTargetDiscovery``.

Split out of ``composite_discovery.py`` to keep the parent below the 1k-line
monolith threshold. ``fit`` is bound back onto the
``CompositeTargetDiscovery`` class at the parent's module bottom, so call
sites that invoke ``disc.fit(...)`` continue to work unchanged.
"""
from __future__ import annotations

import logging
from timeit import default_timer as timer
from typing import Any, Sequence

import numpy as np

from .composite_spec import CompositeSpec
from .composite_forward_stepwise import forward_stepwise_multi_base
from .composite_screening import (
    _extract_column_array,
    _mi_to_target,
    _mi_to_target_prebinned,
    _prebin_feature_columns,
    _sample_indices,
)
from .composite_transforms import (
    UnknownTransformError,
    _linear_residual_fit,
    _linear_residual_multi_fit,
    compose_target_name,
    get_transform,
)
from ._ram_helpers import (
    get_process_rss_mb as _rss_mb,
)
from ._composite_discovery_eval import eval_one_transform

logger = logging.getLogger(__name__)


def _process_mem_mb() -> tuple[float, float, float]:
    """Return ``(rss_mb, uss_mb, commit_mb)`` for the current process.

    Three signals tell a complete Windows memory story:

    * **RSS** (working set): pages currently in physical RAM. Plummets to
      near-zero right after ``pyutilz.clean_ram()`` which invokes
      ``EmptyWorkingSet`` to evict pages to the page file -- the eviction is
      cosmetic; pages page back in on first touch. RSS alone reads as
      misleading "reclaimed 57 GB" lines that don't reflect real frees.
    * **USS** (Unique Set Size): pages this process uniquely owns,
      regardless of working-set residency. Immune to EmptyWorkingSet, so USS
      is the honest "what this process actually allocated" number.
    * **commit / private bytes**: the ``CommitCharge`` Windows uses to gate
      ``OutOfMemory`` decisions. The Windows commit limit is
      ``physical_RAM + pagefile_size``; when total committed memory across
      all processes hits the limit, new allocations fail and OOM-killer
      fires. USS<commit because USS excludes shared mappings and uncommitted
      reserved address space. On the user's 128 GB host with 35 GB system
      baseline, an mlframe process at USS=60 GB / commit=90 GB has only
      ``128 - 35 - 90 = 3 GB`` of commit-limit headroom -- the next 5 GB
      LightGBM Dataset triggers kernel-kill. Showing commit alongside USS
      makes the page-file pressure observable in the prod log.

    All three are floats in MB. When ``memory_full_info()`` is unavailable
    (psutil too old / sandboxed) we fall back through RSS for both USS and
    commit -- the call still doesn't raise.
    """
    rss = _rss_mb()
    uss = rss
    commit = rss
    try:
        import psutil as _psutil
        full = _psutil.Process().memory_full_info()
        uss = float(getattr(full, "uss", rss * 1024 ** 2)) / 1024 ** 2
        # ``private`` exists on Windows (psutil's MEMORY_PRIVATE_USAGE counter --
        # the actual CommitCharge); on Linux fall back to VMS minus shared.
        priv = getattr(full, "private", None)
        if priv is not None:
            commit = float(priv) / 1024 ** 2
        else:
            vms = float(getattr(full, "vms", rss * 1024 ** 2))
            shared = float(getattr(full, "shared", 0.0))
            commit = (vms - shared) / 1024 ** 2
    except Exception:
        pass
    return rss, uss, commit


def _phase_ram_report(state: dict, phase_name: str) -> None:
    """Emit one INFO log line per discovery sub-phase boundary with delta-vs-prev
    and cumulative delta vs the fit() entry baseline.

    Reports BOTH RSS (working set) and USS (unique set size, immune to
    EmptyWorkingSet eviction on Windows). When RSS << USS the process is
    page-thrashing -- a critical signal the prior version masked. State is a
    ``{'baseline_uss_mb': float, 'prev_uss_mb': float}`` dict the caller
    threads through the discovery fit.

    No GC is forced from inside the profiler. ``pyutilz.clean_ram()`` on
    Windows is harmful here (it evicts the working set without freeing real
    memory) and the discovery callers run gc collection at suite-level
    boundaries instead. Removing the call also removes the bogus post-GC
    log line that read "reclaimed 57 GB" right after each phase.
    """
    try:
        rss_mb, uss_mb, commit_mb = _process_mem_mb()
    except Exception:
        return
    if state.get("baseline_uss_mb") is None:
        state["baseline_uss_mb"] = uss_mb
        state["prev_uss_mb"] = uss_mb
        state["baseline_commit_mb"] = commit_mb
        state["prev_commit_mb"] = commit_mb
        logger.info(
            "[CompositeTargetDiscovery.RAM] phase=%s start USS=%.0f MB (RSS=%.0f MB, commit=%.0f MB)",
            phase_name, uss_mb, rss_mb, commit_mb,
        )
        return
    prev_uss = state["prev_uss_mb"]
    baseline_uss = state["baseline_uss_mb"]
    prev_commit = state.get("prev_commit_mb", commit_mb)
    # When RSS << USS by a non-trivial margin the process is page-thrashing
    # (working-set evicted by EmptyWorkingSet or external memory pressure).
    # When commit >> USS the process holds committed but rarely-touched
    # memory -- on Windows this consumes the system-wide commit limit and
    # is the proximate cause of OOM-kernel-kill even when USS itself is
    # well under physical RAM. Surface both hints inline.
    _hints = []
    if uss_mb > rss_mb * 2 and uss_mb > 1024:
        _hints.append(f"PAGE_THRASHING(uss/rss={uss_mb/max(rss_mb, 1):.1f}x)")
    if commit_mb > uss_mb * 1.4 and commit_mb > 4096:
        _hints.append(f"COMMIT_PRESSURE(commit/uss={commit_mb/max(uss_mb, 1):.1f}x)")
    _hint_suffix = (" " + " ".join(_hints)) if _hints else ""
    logger.info(
        "[CompositeTargetDiscovery.RAM] phase=%s USS=%.0f MB (RSS=%.0f MB, commit=%.0f MB; delta_uss_vs_prev=%+.0f MB, delta_commit_vs_prev=%+.0f MB, cum_uss=%+.0f MB)%s",
        phase_name,
        uss_mb,
        rss_mb,
        commit_mb,
        uss_mb - prev_uss,
        commit_mb - prev_commit,
        uss_mb - baseline_uss,
        _hint_suffix,
    )
    state["prev_uss_mb"] = uss_mb
    state["prev_commit_mb"] = commit_mb


def fit(
    self,
    df: Any,
    target_col: str,
    feature_cols: Sequence[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray | None = None,
    test_idx: np.ndarray | None = None,
) -> "CompositeTargetDiscovery":  # noqa: F821 -- forward ref to parent class
    """Discover composite-target specs.

    Parameters
    ----------
    df
        Pandas or polars frame containing ``target_col`` and
        ``feature_cols`` as columns.
    target_col
        Column name of the regression target.
    feature_cols
        Candidate feature columns. Base candidates are drawn from
        this set when ``config.base_candidates="auto"``.
    train_idx
        Row indices to use for fitting transform params and
        scoring. **Required** -- no implicit "use full df" shortcut.
    val_idx, test_idx
        Stored on the instance for later integrity checks; never
        touched during fit.
    """
    if not self.config.enabled:
        self.specs_: list[CompositeSpec] = []
        self.report_: list[dict[str, Any]] = []
        self.train_idx_ = np.asarray(train_idx)
        self._df_ref = df
        self._target_col = target_col
        return self

    train_idx = np.asarray(train_idx)
    # Stash the identifiers BEFORE the early-return paths so
    # _filter_features (which reads ``self._target_col``) and
    # iter_transform (which reads ``self._df_ref``) work even on
    # the no-spec degenerate cases.
    self._target_col = target_col
    self._df_ref = df
    self.train_idx_ = train_idx

    if train_idx.size < 50:
        logger.warning(
            "[CompositeTargetDiscovery] train_idx has only %d rows; "
            "MI estimates unreliable. Discovery yields no specs.",
            train_idx.size,
        )
        self.specs_ = []
        self.report_ = []
        return self

    t0 = timer()
    # Per-fit() RAM telemetry state. Each sub-phase report logs delta vs
    # prev + cumulative vs entry; opt out by setting
    # MLFRAME_DISCOVERY_RAM_PROFILER=0 (the helper checks the env once at
    # entry so the rest of the fit() body never tests the flag again).
    import os as _os
    _ram_state: dict = {}
    _ram_profiler_on = _os.environ.get("MLFRAME_DISCOVERY_RAM_PROFILER", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )
    if _ram_profiler_on:
        _phase_ram_report(_ram_state, "entry")

    # Pull target on train rows. We never touch val/test.
    y_full = _extract_column_array(df, target_col)
    y_train = y_full[train_idx]

    # Auto-boost mi_n_strata on heavy-tail y. The
    # default 10 strata produces unstable MI estimates when the
    # tail dominates the signal -- one or two tail rows per bin.
    # Detect via skew or kurtosis on train; if either is high,
    # bump n_strata to ``mi_n_strata_heavy_tail`` (default 30).
    # User-configured ``mi_n_strata`` remains the floor; we only
    # bump when the auto-detected boost is HIGHER.
    y_finite_for_check = y_train[np.isfinite(y_train)]
    if y_finite_for_check.size >= 100:
        y_std = float(y_finite_for_check.std())
        if y_std > 1e-12:
            z_centered = (y_finite_for_check - y_finite_for_check.mean()) / y_std
            # z**3 / z**4 via chained mul over np.power dispatch (~3x;
            # same antipattern as iter138 _target_distribution_analyzer).
            z2 = z_centered * z_centered
            skew = float(np.mean(z2 * z_centered))
            kurt = float(np.mean(z2 * z2) - 3.0)
            if abs(skew) > 2.0 or kurt > 5.0:
                boost = int(getattr(
                    self.config, "mi_n_strata_heavy_tail", 30,
                ))
                cur_n_strata = int(getattr(
                    self.config, "mi_n_strata", 10,
                ))
                if boost > cur_n_strata:
                    # Mutate config in-place ONLY if we own a
                    # copy (avoid leaking into callers' shared
                    # config). model_copy is safe here because
                    # discovery already gets a per-target config
                    # clone in core.py when hint is enabled.
                    try:
                        new_cfg = self.config.model_copy(
                            update={"mi_n_strata": boost}
                        )
                        self.config = new_cfg
                        logger.info(
                            "[CompositeTargetDiscovery] heavy-tail y "
                            "detected (skew=%.2f, kurt=%.2f); boosted "
                            "mi_n_strata %d -> %d.",
                            skew, kurt, cur_n_strata, boost,
                        )
                    except Exception:
                        pass  # leave at user-configured value.

    # Filter feature_cols by name patterns AND constancy on train.
    usable_features = self._filter_features(df, feature_cols, y_train, train_idx)
    if _ram_profiler_on:
        _phase_ram_report(_ram_state, "filter_features_done")

    # Resolve base candidates.
    base_candidates = self._resolve_base_candidates(
        df, target_col, usable_features, y_train, train_idx,
    )
    if _ram_profiler_on:
        _phase_ram_report(_ram_state, "resolve_base_candidates_done")
    if not base_candidates:
        logger.warning(
            "[CompositeTargetDiscovery] no usable base candidates after "
            "forbidden-pattern / corr / ptp / numeric filters. "
            "Discovery yields no specs."
        )
        self.specs_ = []
        self.report_ = []
        return self

    # Down-sample for MI screening. Stratified-quantile when
    # configured -- guarantees per-bin coverage on heavy-tail y.
    y_train_for_strat = y_full[train_idx]
    sample_idx = _sample_indices(
        train_idx.size, self.config.mi_sample_n, self.config.random_state,
        strategy=getattr(self.config, "mi_sample_strategy", "random"),
        y=y_train_for_strat,
        n_strata=getattr(self.config, "mi_n_strata", 10),
    )
    train_idx_screen = train_idx[sample_idx]
    y_screen = y_full[train_idx_screen]

    # mi_y baseline is computed PER-BASE because the X-without-base
    # feature set differs per candidate. Comparing MI(T, X_no_base)
    # against MI(y, X) (full X) confounds two effects: target
    # transformation AND removal of the dominant feature. We want
    # only the first effect, so both halves use the same feature
    # set: X without the base column.

    # Stash the per-candidate base arrays so the multi-base forward-stepwise extension (run after kept_specs is finalised) can pick from the SAME pool of MI-ranked bases that the single-base discovery considered. Keyed by column name; values are train-row-restricted ndarrays.
    self._auto_base_pool: dict[str, np.ndarray] = {}

    # Score each (base, transform).
    # Pack J: unary y-transforms (``requires_base=False``) ignore the base column;
    # tracking which unary names we've already evaluated lets us skip the redundant
    # re-fit on the second / third / ... base loop iteration. Bivariate + chain
    # transforms still iterate per base normally.
    _unary_evaluated: set[str] = set()

    candidates: list[dict[str, Any]] = []

    # 2026-05-20 #2: hoist the per-base setup OUT of the candidate
    # evaluation loop so a single parallel dispatch can span all
    # (base, transform) pairs. The previous serial outer loop over
    # bases bottlenecked total parallelism at ``n_transforms`` per
    # base; flattening lifts the cap to ``sum(transforms_per_base)``
    # and lets ``discovery_n_jobs`` saturate even when a single base
    # has fewer eligible transforms than CPU cores. Per-base setup
    # itself (column extraction, X-without-base matrix, pre-binning,
    # ``mi_y_for_base``) stays serial because it writes
    # ``self._auto_base_pool`` and is cheap relative to MI compute.
    # Build the full screen-sized feature matrix ONCE across all bases, then
    # per-base slice out the base column via np.delete. Avoids 10x polars->numpy
    # column extraction (and 10x prebinning if mi_estimator='bin') on the same
    # usable_features set. The polars columns themselves don't change between
    # iterations - only the choice of which one is the "base" does.
    _usable_features_list = list(usable_features)
    _col_index = {c: i for i, c in enumerate(_usable_features_list)}
    _full_x_matrix = self._build_feature_matrix(
        df, _usable_features_list, train_idx_screen,
    )
    if _ram_profiler_on:
        _phase_ram_report(_ram_state, "build_full_x_matrix_done")
    _full_x_prebinned = (
        _prebin_feature_columns(
            _full_x_matrix, nbins=int(self.config.mi_nbins),
        )
        if self.config.mi_estimator == "bin" else None
    )
    if _ram_profiler_on:
        _phase_ram_report(_ram_state, "prebin_features_done")

    _base_contexts: dict[str, dict[str, Any]] = {}
    for base in base_candidates:
        base_train = _extract_column_array(df, base)[train_idx]
        self._auto_base_pool[base] = base_train
        base_screen = base_train[sample_idx]
        if base in _col_index:
            _drop_idx = _col_index[base]
            x_remaining_matrix = np.delete(_full_x_matrix, _drop_idx, axis=1)
            _x_prebinned = (
                np.delete(_full_x_prebinned, _drop_idx, axis=1)
                if _full_x_prebinned is not None else None
            )
        else:
            # base wasn't in usable_features (rare: explicit base outside the FE pool)
            x_remaining_matrix = _full_x_matrix
            _x_prebinned = _full_x_prebinned
        if x_remaining_matrix.shape[1] == 0:
            continue
        _mi_kwargs: dict[str, Any] = dict(
            nbins=int(self.config.mi_nbins),
            aggregation=getattr(self.config, "mi_aggregation", "mean"),
        )
        if _x_prebinned is not None:
            mi_y_for_base = _mi_to_target_prebinned(
                _x_prebinned, y_screen, **_mi_kwargs,
            )
        else:
            mi_y_for_base = _mi_to_target(
                x_remaining_matrix, y_screen,
                n_neighbors=self.config.mi_n_neighbors,
                random_state=self.config.random_state,
                estimator=self.config.mi_estimator,
                **_mi_kwargs,
            )
        _base_contexts[base] = dict(
            base_train=base_train,
            base_screen=base_screen,
            x_remaining_matrix=x_remaining_matrix,
            _x_prebinned=_x_prebinned,
            mi_y_for_base=mi_y_for_base,
            _mi_kwargs=_mi_kwargs,
        )

    # Build flat (base, transform_name, transform) work list with
    # unary-dedup. Unary transforms ignore base, so a single
    # evaluation against the first base they appear under is
    # sufficient. Keeping dedup serial outside the parallel
    # dispatch preserves bit-for-bit identical behaviour vs the
    # serial path on the same input.
    _work_items: list[tuple[str, str, Any]] = []
    for base in base_candidates:
        if base not in _base_contexts:
            continue
        for transform_name in self.config.transforms:
            try:
                transform = get_transform(transform_name)
            except UnknownTransformError as exc:
                logger.warning(
                    "[CompositeTargetDiscovery] %s; skipping.", exc,
                )
                continue
            if not transform.requires_base:
                if transform_name in _unary_evaluated:
                    continue
                _unary_evaluated.add(transform_name)
            _work_items.append((base, transform_name, transform))

    # 2026-05-20 #2: single parallel dispatch over the flat
    # ``_work_items`` list. Joblib preserves input order so
    # ``candidates`` ends up in (base, transform) iteration order
    # identical to the legacy nested-loop serial path. joblib
    # threading backend keeps closure capture cheap (no pickling),
    # which is critical for the large ``x_remaining_matrix`` /
    # ``_x_prebinned`` arrays the body reads. Most of the compute
    # (transform.fit / transform.forward / _mi_to_target_prebinned
    # / bootstrap MI loop) is numpy / numba which releases the GIL,
    # so threading scales close to linearly up to cpu_count.
    # 0 = auto: cap at the number of work items and cpu_count. 1 = serial.
    _n_jobs_raw = getattr(self.config, "discovery_n_jobs", 1)
    _n_jobs_raw = 1 if _n_jobs_raw is None else int(_n_jobs_raw)
    if _n_jobs_raw == 0:
        import os as _os
        _n_jobs_disc = max(1, min(len(_work_items), _os.cpu_count() or 1))
    else:
        _n_jobs_disc = max(1, _n_jobs_raw)
    if _n_jobs_disc > 1 and len(_work_items) > 1:
        from joblib import Parallel as _Parallel, delayed as _delayed
        _results = _Parallel(
            n_jobs=_n_jobs_disc, backend="threading", prefer="threads",
        )(
            _delayed(eval_one_transform)(
                self, _b, _tn, _t,
                base_contexts=_base_contexts, y_train=y_train,
                y_screen=y_screen, target_col=target_col,
            )
            for _b, _tn, _t in _work_items
        )
    else:
        _results = [
            eval_one_transform(
                self, _b, _tn, _t,
                base_contexts=_base_contexts, y_train=y_train,
                y_screen=y_screen, target_col=target_col,
            )
            for _b, _tn, _t in _work_items
        ]
    for _r in _results:
        if _r:
            candidates.extend(_r)
    if _ram_profiler_on:
        _phase_ram_report(_ram_state, "transforms_evaluated")

    # Filter + sort.
    kept_specs: list[CompositeSpec] = []
    for entry in candidates:
        spec: CompositeSpec | None = entry.get("spec")
        if spec is None:
            continue  # already a reject
        # Gate compares LCB (lower CI bound), not point estimate,
        # when bootstrap is enabled. Falls back to point estimate
        # when LCB unavailable.
        mi_gain_for_gate = entry.get("mi_gain_lcb", spec.mi_gain)
        if mi_gain_for_gate <= self.config.eps_mi_gain:
            entry["reason"] = (
                f"mi_gain={spec.mi_gain:.4f} <= eps={self.config.eps_mi_gain:.4f}"
            )
            continue
        kept_specs.append(spec)
        entry["kept"] = True

    # Wave 57 (2026-05-20): plugin MI quantises to a fixed grid -> tied
    # mi_gain realistic. Secondary key on spec name for deterministic
    # top-K selection across runs.
    kept_specs.sort(key=lambda s: (-s.mi_gain, getattr(s, "name", "")))
    kept_specs = kept_specs[: self.config.top_k_after_mi]

    # R10b stat #6: rolling-origin alpha drift detection for
    # linear_residual specs. Fit alpha on first / second halves
    # of train; Chow-style z-score on the difference. Specs with
    # |z| > threshold either rejected or flagged depending on
    # config.
    if (getattr(self.config, "detect_linear_residual_alpha_drift", True)
            and any(s.transform_name == "linear_residual"
                    for s in kept_specs)):
        self._alpha_drift_flags: dict[str, dict[str, float]] = {}
        drift_threshold = float(getattr(
            self.config, "alpha_drift_z_threshold", 3.0,
        ))
        reject_on_drift = bool(getattr(
            self.config, "reject_on_alpha_drift", False,
        ))
        half = len(train_idx) // 2
        if half >= 50:
            drift_dropped: list[tuple[str, float]] = []
            drift_kept: list[CompositeSpec] = []
            for s in kept_specs:
                if s.transform_name != "linear_residual":
                    drift_kept.append(s)
                    continue
                base_full = _extract_column_array(df, s.base_column)
                y_full_arr = y_full
                idx1 = train_idx[:half]
                idx2 = train_idx[half:]
                try:
                    params1 = _linear_residual_fit(
                        y_full_arr[idx1], base_full[idx1],
                    )
                    params2 = _linear_residual_fit(
                        y_full_arr[idx2], base_full[idx2],
                    )
                except Exception:
                    drift_kept.append(s)
                    continue
                a1 = float(params1.get("alpha", 0.0))
                a2 = float(params2.get("alpha", 0.0))
                # ENS-Low-2: residual-based OLS slope SE. The previous
                # formula y_std / (sqrt(n) * base_std) used the marginal
                # y-variance which overstates SE when the regressor
                # explains most variance. Correct form is
                # SE(alpha) = sqrt(SSE / (n-2)) / (sqrt(n) * base_std).
                base_t = base_full[train_idx]
                finite_pair = np.isfinite(base_t) & np.isfinite(y_full_arr[train_idx])
                base_finite = base_t[finite_pair]
                base_std = (
                    float(base_finite.std()) if base_finite.size > 1
                    else 1.0
                )
                if base_std < 1e-12 or half < 2:
                    drift_kept.append(s)
                    continue
                # Use the pooled alpha/beta from full train_idx for a
                # single residual sum estimate; degrees-of-freedom
                # subtracts 2 (slope + intercept). ``half`` rows fit
                # alpha/beta on each half - reuse params1's beta on the
                # pooled segment for the residual scale.
                y_finite = y_full_arr[train_idx][finite_pair]
                n_pair = int(finite_pair.sum())
                if n_pair > 2:
                    # Use the average of (a1, a2) and average beta as a
                    # pooled OLS estimate to compute residuals; cheap
                    # robust pooled fit on (y, base).
                    b1 = float(params1.get("beta", 0.0))
                    b2 = float(params2.get("beta", 0.0))
                    alpha_pool = 0.5 * (a1 + a2)
                    beta_pool = 0.5 * (b1 + b2)
                    residuals = y_finite - (alpha_pool * base_finite + beta_pool)
                    sse = float(np.sum(residuals * residuals))
                    sigma_resid = float(np.sqrt(max(sse / (n_pair - 2), 0.0)))
                else:
                    # Fall back to marginal y-std if too few points to
                    # compute residual variance.
                    sigma_resid = float(y_finite.std()) if y_finite.size > 1 else 1.0
                se_alpha = sigma_resid / (np.sqrt(half) * base_std)
                z = abs(a1 - a2) / max(se_alpha, 1e-12)
                self._alpha_drift_flags[s.name] = {
                    "alpha_first_half": a1,
                    "alpha_second_half": a2,
                    "z_score": float(z),
                }
                if z > drift_threshold:
                    if reject_on_drift:
                        drift_dropped.append((s.name, float(z)))
                        continue
                    # Demoted to DEBUG: many drift-detected specs are subsequently rejected by the raw-y baseline gate / Wilcoxon filter; emitting a WARNING for each one before the gate produces dead-noise. A summary WARNING is emitted at the end of discovery ONLY for specs that survived all gates.
                    else:
                        logger.debug(
                            "[CompositeTargetDiscovery] alpha drift "
                            "candidate spec=%s (alpha first-half="
                            "%.4f, second-half=%.4f, z=%.2f > %.2f).",
                            s.name, a1, a2, z, drift_threshold,
                        )
                drift_kept.append(s)
            if drift_dropped:
                logger.info(
                    "[CompositeTargetDiscovery] alpha drift gate "
                    "dropped %d linear_residual spec(s): %s",
                    len(drift_dropped),
                    ", ".join(
                        f"{n}(z={z:.2f})"
                        for n, z in drift_dropped[:5]
                    ),
                )
            kept_specs = drift_kept

    # R10b improvement #6: collapse redundant linear_residual ->
    # diff when alpha ~ 1 and beta ~ 0 (linear_residual has zero
    # information advantage over diff but carries 2 fitted params).
    # Drop linear_residual specs whose alpha is close to 1.0 on
    # the data scale IF a diff spec for the same base also kept.
    # Skipped if the config eps is 0 (feature disabled).
    alpha_eps = float(getattr(
        self.config, "collapse_linear_residual_alpha_eps", 0.05,
    ))
    if alpha_eps > 0 and len(kept_specs) > 1:
        diff_bases = {
            s.base_column for s in kept_specs
            if s.transform_name == "diff"
        }
        collapsed: list[CompositeSpec] = []
        collapsed_dropped: list[tuple[str, float]] = []
        std_y = float(np.std(y_train[np.isfinite(y_train)])) or 1.0
        for s in kept_specs:
            if s.transform_name != "linear_residual" \
                    or s.base_column not in diff_bases:
                collapsed.append(s)
                continue
            alpha = float(s.fitted_params.get("alpha", float("nan")))
            beta = float(s.fitted_params.get("beta", 0.0))
            base_train = _extract_column_array(df, s.base_column)[train_idx]
            base_finite = np.isfinite(base_train)
            std_base = (
                float(np.std(base_train[base_finite]))
                if base_finite.any() else 1.0
            )
            if std_base < 1e-12:
                collapsed.append(s)
                continue
            # Scale-invariant alpha deviation: how much linear-residual
            # diverges from diff (which is alpha=1, beta=0) relative
            # to the data scale.
            alpha_dev = abs(alpha - 1.0) * std_base / std_y
            beta_dev = abs(beta) / std_y
            if alpha_dev < alpha_eps and beta_dev < alpha_eps:
                collapsed_dropped.append(
                    (s.name, float(alpha_dev))
                )
                continue
            collapsed.append(s)
        if collapsed_dropped:
            preview = ", ".join(
                f"{n}(alpha_dev={d:.4f})"
                for n, d in collapsed_dropped[:3]
            )
            logger.info(
                "[CompositeTargetDiscovery] collapsed %d "
                "linear_residual spec(s) into diff (alpha~1): %s",
                len(collapsed_dropped), preview,
            )
        kept_specs = collapsed

    # Phase B: tiny-model rerank. Re-rank the MI-survivors by
    # CV-RMSE on the y-scale (the actual prediction objective).
    # Skip when ``screening == "mi"`` -- callers who want only
    # MI ranking pay zero rerank cost.
    if (kept_specs and self.config.screening in ("tiny_model", "hybrid")
            and self.config.tiny_screening_models in ("single_lgbm",
                                                       "per_family")):
        kept_specs = self._tiny_model_rerank(
            kept_specs=kept_specs,
            df=df,
            target_col=target_col,
            usable_features=usable_features,
            train_idx=train_idx,
            y_full=y_full,
        )
        if _ram_profiler_on:
            _phase_ram_report(_ram_state, "tiny_model_rerank_done")

    if not kept_specs:
        mode = self.config.fail_on_no_gain
        msg = (
            f"[CompositeTargetDiscovery] no candidate cleared mi_gain > "
            f"{self.config.eps_mi_gain} on target='{target_col}'."
        )
        if mode == "raise":
            raise RuntimeError(msg)
        logger.warning(msg + f" (fail_on_no_gain={mode!r})")

    # OPEN-1 integration (2026-05-12): multi-base forward-stepwise auto-promotion of linear_residual specs. After single-base discovery + raw-y baseline gate + tiny-model rerank, look at each kept ``linear_residual`` spec and try greedily adding more bases from the auto-base candidate pool. When the marginal RMSE reduction clears ``multi_base_min_marginal_rmse_gain`` (default 0.02 = 2%), upgrade the spec to ``linear_residual_multi`` with the expanded base list. Measure-first benchmark in ``benchmarks/composite_multi_base_benchmark.py`` validates: geo-mean gain 83% on positive scenarios, no-harm on negative scenarios -> auto-promote=True. Gated by ``self.config.multi_base_enabled``; opt-out via config.
    if (kept_specs
            and getattr(self.config, "multi_base_enabled", False)
            and getattr(self, "_auto_base_pool", None)):
        _multi_max_k = int(getattr(self.config, "multi_base_max_k", 3))
        _multi_min_gain = float(getattr(self.config, "multi_base_min_marginal_rmse_gain", 0.02))
        _cv_sel_mode = str(getattr(self.config, "cv_selector_mode", "mean"))
        _cv_sel_alpha = float(getattr(self.config, "cv_selector_alpha", 1.0))
        _cv_sel_conf = float(getattr(self.config, "cv_selector_confidence", 0.9))
        _cv_sel_qlevel = float(getattr(self.config, "cv_selector_quantile_level", 0.9))
        _cv_persist = bool(getattr(self.config, "cv_persist_fold_scores", False))
        _upgraded_specs: list[CompositeSpec] = []
        # ENS-Low-6: hoist the (base_column, pool_signature) -> pool_arrays
        # build outside the per-spec loop so K linear_residual specs that
        # share the same auto_base_pool + base_column do ONE pool build
        # (and one _extract_column_array call), not K. Cache key includes
        # the pool signature (frozenset of pool keys) so config-driven
        # pool changes invalidate cleanly.
        _pool_arrays_cache: dict[tuple[str, frozenset], dict[str, np.ndarray]] = {}
        _base_pool_keys_frozen = frozenset(self._auto_base_pool.keys())
        _y_train_local = y_full[train_idx] if y_full is not None else _extract_column_array(df, target_col)[train_idx]
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
                _pool_arrays = {c: self._auto_base_pool.get(c) for c in _pool_cols if self._auto_base_pool.get(c) is not None}
                if _spec.base_column not in _pool_arrays:
                    _pool_arrays[_spec.base_column] = _extract_column_array(df, _spec.base_column)[train_idx]
                _pool_arrays_cache[_cache_key] = _pool_arrays
            try:
                _kept_bases, _fwd_diag = forward_stepwise_multi_base(
                    _y_train_local, _pool_arrays,
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
                    _spec.name, _multi_err,
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
            _accepted_steps = [d for d in _fwd_diag if d.get("accepted")]
            logger.info(
                "[CompositeTargetDiscovery.multi_base] upgraded spec='%s' -> '%s' with %d base(s); accepted_steps=%s",
                _spec.name, _new_name, len(_kept_bases),
                [(d["candidate_added"], f"{d['marginal_gain'] * 100:.1f}%") for d in _accepted_steps],
            )
        kept_specs = _upgraded_specs
        if _ram_profiler_on:
            _phase_ram_report(_ram_state, "forward_stepwise_done")

    elapsed = timer() - t0
    logger.info(
        "[CompositeTargetDiscovery] target='%s' discovered %d spec(s) "
        "from %d candidate(s) in %.2fs",
        target_col, len(kept_specs), len(candidates), elapsed,
    )
    if _ram_profiler_on:
        _phase_ram_report(_ram_state, "fit_exit")

    # Alpha-drift WARNINGs only for the SURVIVING specs. Inline emits during scoring are at DEBUG; the user sees a single, actionable warning at the end of discovery rather than a wall of warnings for specs that the raw-y baseline gate / Wilcoxon filter dropped anyway.
    _drift_flags = getattr(self, "_alpha_drift_flags", {})
    if _drift_flags and kept_specs:
        _drift_threshold = float(getattr(
            self.config, "alpha_drift_z_threshold", 3.0,
        ))
        _surviving_drift = [
            (s.name, _drift_flags[s.name])
            for s in kept_specs
            if s.name in _drift_flags
            and _drift_flags[s.name].get("z_score", 0.0) > _drift_threshold
        ]
        if _surviving_drift:
            for _spec_name, _info in _surviving_drift:
                logger.warning(
                    "[CompositeTargetDiscovery] alpha drift "
                    "detected for KEPT spec=%s (alpha first-half="
                    "%.4f, second-half=%.4f, z=%.2f > %.2f). "
                    "Concept drift -- linear_residual may "
                    "underperform on test. Set "
                    "reject_on_alpha_drift=True in "
                    "CompositeTargetDiscoveryConfig to drop "
                    "automatically.",
                    _spec_name,
                    _info["alpha_first_half"],
                    _info["alpha_second_half"],
                    _info["z_score"], _drift_threshold,
                )

    # Bookkeeping. (target_col + df_ref + train_idx already stashed.)
    self.specs_ = kept_specs
    self.report_ = [self._entry_to_report(e) for e in candidates]
    self.val_idx_ = np.asarray(val_idx) if val_idx is not None else None
    self.test_idx_ = np.asarray(test_idx) if test_idx is not None else None
    self.elapsed_seconds_ = elapsed
    return self

