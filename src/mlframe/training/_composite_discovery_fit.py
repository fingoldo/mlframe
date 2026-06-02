"""The main ``fit`` method for ``CompositeTargetDiscovery``.

Split out of ``composite_discovery.py`` to keep the parent below the 1k-line
monolith threshold. ``fit`` is bound back onto the
``CompositeTargetDiscovery`` class at the parent's module bottom, so call
sites that invoke ``disc.fit(...)`` continue to work unchanged.
"""
from __future__ import annotations

import logging
import re
import warnings
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
    maybe_clean_ram_adaptive as _maybe_clean_ram_adaptive,
)

logger = logging.getLogger(__name__)


def _phase_ram_report(state: dict, phase_name: str) -> None:
    """Emit one INFO log line per discovery sub-phase boundary with delta-vs-prev
    and cumulative delta vs the fit() entry baseline.

    State is a {'baseline_mb': float, 'prev_mb': float} dict the caller threads
    through the discovery fit. ``maybe_clean_ram_adaptive()`` is invoked AFTER
    reading the post-phase RSS so the report reflects the unfreed footprint of
    the sub-phase that just finished; the GC trigger then runs only when growth
    exceeded the helper's internal threshold (~500 MB).

    Designed for prod-debug: the user's observable was a kernel OOM somewhere
    inside one discovery call -- with each sub-phase stamped we can attribute
    the spike to ``_filter_features`` / ``_auto_base`` / per-base loop /
    ``_tiny_model_rerank`` / ``forward_stepwise`` rather than wave-handing at
    "discovery used too much RAM".
    """
    try:
        now_mb = _rss_mb()
    except Exception:
        return
    if state.get("baseline_mb") is None:
        state["baseline_mb"] = now_mb
        state["prev_mb"] = now_mb
        logger.info(
            "[CompositeTargetDiscovery.RAM] phase=%s start RSS=%.0f MB", phase_name, now_mb,
        )
        return
    prev_mb = state.get("prev_mb", now_mb)
    baseline_mb = state.get("baseline_mb", now_mb)
    logger.info(
        "[CompositeTargetDiscovery.RAM] phase=%s RSS=%.0f MB (delta_vs_prev=%+.0f MB, cumulative_vs_baseline=%+.0f MB)",
        phase_name,
        now_mb,
        now_mb - prev_mb,
        now_mb - baseline_mb,
    )
    state["prev_mb"] = now_mb
    # Defensive adaptive GC. Internal threshold (~500 MB) gates the call so
    # the cost is paid only when growth justified it. Re-reads RSS afterward
    # because clean_ram may have reclaimed buffers; the next sub-phase report
    # then measures delta from the post-GC baseline.
    try:
        _maybe_clean_ram_adaptive()
        post_gc = _rss_mb()
        if post_gc < now_mb - 100.0:
            logger.info(
                "[CompositeTargetDiscovery.RAM] phase=%s post-GC RSS=%.0f MB (reclaimed %.0f MB)",
                phase_name, post_gc, now_mb - post_gc,
            )
            state["prev_mb"] = post_gc
    except Exception:
        pass


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

    def _eval_one_transform(
        base: str, transform_name: str, transform,
    ) -> list[dict[str, Any]]:
        """Returns 0 or 1 candidate dict for one (base, transform) pair.

        Pulls per-base arrays from ``_base_contexts[base]`` (read-only
        once setup completes). Writes go to the returned list, never
        to the enclosing ``candidates`` list, so calling this
        concurrently from a thread pool is safe.
        """
        _ctx = _base_contexts[base]
        base_train = _ctx["base_train"]
        base_screen = _ctx["base_screen"]
        x_remaining_matrix = _ctx["x_remaining_matrix"]
        _x_prebinned = _ctx["_x_prebinned"]
        mi_y_for_base = _ctx["mi_y_for_base"]
        _mi_kwargs = _ctx["_mi_kwargs"]
        _local: list[dict[str, Any]] = []
        # Domain check on train, drop invalids, fit transform
        # params on the surviving rows only.
        valid = transform.domain_check(y_train, base_train)
        valid_frac = float(valid.mean()) if valid.size else 0.0
        if valid_frac < self.config.min_valid_domain_frac:
            _local.append(self._reject(
                base, transform_name, mi_y_for_base, valid_frac,
                reason=f"valid_domain_frac={valid_frac:.3f} "
                       f"< {self.config.min_valid_domain_frac:.3f}",
            ))
            return _local
        if not valid.any():
            return _local

        fitted_params = transform.fit(y_train[valid], base_train[valid])
        # Pack D 2026-05-18: reject identity / near-identity transforms early.
        # Some bivariate transforms can collapse to a constant residual
        # (T = y - const) when the base does not actually carry the
        # signal -- e.g. ``monotonic_residual`` on a base where the
        # fitted PCHIP knots are essentially flat. Discovery then
        # spends 5+ minutes training models that produce IDENTICAL
        # predictions to raw-y (observed in prod on a monres spec). The
        # transform's ``fit`` flags this via ``is_degenerate=True``
        # on the returned params dict; reject the spec here.
        if isinstance(fitted_params, dict) and fitted_params.get("is_degenerate"):
            _ve = fitted_params.get("var_explained", float("nan"))
            _local.append(self._reject(
                base, transform_name, mi_y_for_base, valid_frac,
                reason=(
                    f"transform fitted to a near-identity function: "
                    f"var_explained={_ve:.4f} -- T == y up to noise, "
                    f"downstream models will produce SAME predictions "
                    f"as on raw y"
                ),
            ))
            return _local
        # 2026-05-21: linres_robust dedup. When the MAD-trim step in
        # ``_linear_residual_robust_fit`` doesn't drop any rows, the
        # second-pass OLS produces alpha/beta identical to the first
        # pass -- i.e. the transform IS plain ``linear_residual``.
        # The fit stamps ``is_redundant_with_linres=True`` to signal
        # this; we skip the evaluation to avoid duplicate MI compute
        # + duplicate downstream rerank+training. Observed in a prod log:
        # ``linres-Y`` and ``linresR-Y`` produced identical
        # RMSE=21.5433 — 100% wasted compute on the duplicate.
        if (transform_name == "linear_residual_robust"
                and isinstance(fitted_params, dict)
                and fitted_params.get("is_redundant_with_linres")):
            _local.append(self._reject(
                base, transform_name, mi_y_for_base, valid_frac,
                reason=(
                    "linear_residual_robust MAD-trim found zero "
                    "outliers above 3*sigma_MAD; second-pass OLS "
                    "would be identical to plain linear_residual. "
                    "Skipping the duplicate evaluation."
                ),
            ))
            return _local
        # 2026-05-23: upper-bound degeneracy check. The pre-fix
        # ``is_degenerate`` flag in transform.fit only catches the
        # LOWER bound (transform explains <5% of y variance -- T ~= y).
        # The OPPOSITE pathology also exists: transform absorbs SO
        # much of y that the residual T is at or below the noise
        # floor (observed in prod on a logr spec: y_std=644,
        # T_std=0.001 -- ratio 644000:1). Even a tiny fitting error
        # on T compounds via inverse_transform into significant
        # y-scale error, AND downstream models train on essentially
        # white noise. Compute residual std on full train sample
        # (cheap: one transform.forward call) and reject when
        # T_std / y_std < 0.001 (T is below 0.1% of y scale -- below
        # typical noise floor for f32 tabular targets).
        try:
            _y_train_valid = y_train[valid].astype(np.float64)
            _base_train_valid = base_train[valid].astype(np.float64)
            _t_train_full = transform.forward(
                _y_train_valid, _base_train_valid, fitted_params,
            )
            _t_train_finite = _t_train_full[np.isfinite(_t_train_full)]
            _y_train_finite = _y_train_valid[np.isfinite(_y_train_valid)]
            if _t_train_finite.size > 1 and _y_train_finite.size > 1:
                _y_std = float(np.std(_y_train_finite))
                _t_std = float(np.std(_t_train_finite))
                _residual_ratio = (
                    _t_std / _y_std if _y_std > 0 else 1.0
                )
                if _residual_ratio < 0.001:
                    _local.append(self._reject(
                        base, transform_name, mi_y_for_base, valid_frac,
                        reason=(
                            f"residual T below noise floor: "
                            f"T_std={_t_std:.3g} vs y_std={_y_std:.3g} "
                            f"(ratio={_residual_ratio:.2e} < 0.001). "
                            f"Composite would train downstream models on "
                            f"essentially white noise AND amplify tiny "
                            f"T-errors into y-scale errors via "
                            f"inverse_transform."
                        ),
                    ))
                    return _local
        except Exception as _residual_err:
            # Probe failure is non-fatal -- continue to MI screening.
            logger.debug(
                "composite_discovery: residual-std probe failed "
                "for base=%s transform=%s: %s (continuing)",
                base, transform_name, _residual_err,
            )
        # T on the screening sample (which is a subset of train).
        valid_screen = transform.domain_check(y_screen, base_screen)
        if valid_screen.sum() < 50:
            _local.append(self._reject(
                base, transform_name, mi_y_for_base, valid_frac,
                reason="too few rows in screening sample after domain filter",
            ))
            return _local
        t_screen = transform.forward(
            y_screen[valid_screen], base_screen[valid_screen], fitted_params,
        )

        # MI(T, X_remaining) on the same valid rows -- comparable
        # to mi_y_for_base computed on the same x_remaining.
        x_screen_valid = x_remaining_matrix[valid_screen]
        if _x_prebinned is not None:
            _x_pb_valid = _x_prebinned[valid_screen]
            mi_t = _mi_to_target_prebinned(
                _x_pb_valid, t_screen, **_mi_kwargs,
            )
        else:
            mi_t = _mi_to_target(
                x_screen_valid, t_screen,
                n_neighbors=self.config.mi_n_neighbors,
                random_state=self.config.random_state,
                estimator=self.config.mi_estimator,
                **_mi_kwargs,
            )
        # When the screening sample shrunk after domain
        # filtering (logratio with negative rows in train),
        # the mi_y baseline for THIS base must also be
        # recomputed on the same valid_screen subset to keep
        # comparison fair.
        if valid_screen.sum() < y_screen.size:
            if _x_prebinned is not None:
                mi_y_compare = _mi_to_target_prebinned(
                    _x_pb_valid, y_screen[valid_screen], **_mi_kwargs,
                )
            else:
                mi_y_compare = _mi_to_target(
                    x_screen_valid, y_screen[valid_screen],
                    n_neighbors=self.config.mi_n_neighbors,
                    random_state=self.config.random_state,
                    estimator=self.config.mi_estimator,
                    **_mi_kwargs,
                )
        else:
            mi_y_compare = mi_y_for_base
        mi_gain = mi_t - mi_y_compare

        # Bootstrap CI on mi_gain. The
        # point-estimate has a noise floor that scales with
        # screening-sample size and y-tail heaviness; the
        # absolute eps_mi_gain threshold misses this. Bootstrap
        # produces a 95% CI; the gate compares against the
        # LOWER CI bound (LCB), not the point estimate. Spec
        # is rejected if LCB <= eps_mi_gain.
        bootstrap_n = int(getattr(
            self.config, "mi_gain_bootstrap_n", 0,
        ))
        mi_gain_lcb = mi_gain  # default: point estimate.
        if bootstrap_n > 0:
            boot_rng = np.random.default_rng(
                int(getattr(
                    self.config, "mi_gain_bootstrap_random_state", 12345,
                ))
            )
            n_screen = int(valid_screen.sum())
            boot_gains = np.empty(bootstrap_n)
            # Hoist the valid_screen slices once. The pre-fix re-sliced
            # ``y_screen[valid_screen]`` and ``_x_prebinned[valid_screen]`` per replicate
            # even though they are constants across replicates.
            _y_screen_valid = y_screen[valid_screen]
            _x_pb_valid_const = (
                _x_prebinned[valid_screen] if _x_prebinned is not None else None
            )
            for b in range(bootstrap_n):
                idx_b = boot_rng.integers(0, n_screen, size=n_screen)
                x_boot = x_screen_valid[idx_b]
                t_boot = t_screen[idx_b]
                y_boot = _y_screen_valid[idx_b]
                try:
                    if _x_pb_valid_const is not None:
                        _x_pb_boot = _x_pb_valid_const[idx_b]
                        mi_t_b = _mi_to_target_prebinned(
                            _x_pb_boot, t_boot, **_mi_kwargs,
                        )
                        mi_y_b = _mi_to_target_prebinned(
                            _x_pb_boot, y_boot, **_mi_kwargs,
                        )
                    else:
                        mi_t_b = _mi_to_target(
                            x_boot, t_boot,
                            n_neighbors=self.config.mi_n_neighbors,
                            random_state=self.config.random_state,
                            estimator=self.config.mi_estimator,
                            **_mi_kwargs,
                        )
                        mi_y_b = _mi_to_target(
                            x_boot, y_boot,
                            n_neighbors=self.config.mi_n_neighbors,
                            random_state=self.config.random_state,
                            estimator=self.config.mi_estimator,
                            **_mi_kwargs,
                        )
                    boot_gains[b] = mi_t_b - mi_y_b
                except Exception as _e_boot:
                    # Pre-fix: silent NaN on failure; CI shifted toward
                    # well-behaved bootstraps. Log first per-spec failure
                    # so operators see when the CI is computed over a
                    # reduced bootstrap sample (the `>= bootstrap_n // 2`
                    # guard below only protects against extreme
                    # under-sampling, not the partial-bias case).
                    if b == 0:
                        import logging as _logging
                        _logging.getLogger(__name__).warning(
                            "composite_discovery: MI-bootstrap iteration "
                            "failed (%s); per-bootstrap result reported "
                            "as NaN. Bootstrap CI will use surviving "
                            "samples; with sparse failures the LCB is "
                            "biased toward well-behaved bootstraps.",
                            _e_boot,
                        )
                    boot_gains[b] = float("nan")
            boot_finite = boot_gains[np.isfinite(boot_gains)]
            if boot_finite.size >= bootstrap_n // 2:
                mi_gain_lcb = float(np.percentile(boot_finite, 2.5))

        spec = CompositeSpec(
            name=compose_target_name(target_col, transform_name, base),
            target_col=target_col,
            transform_name=transform_name,
            base_column=base,
            fitted_params=dict(fitted_params),
            mi_gain=mi_gain,
            mi_y=mi_y_compare,
            mi_t=mi_t,
            valid_domain_frac=valid_frac,
            n_train_rows=int(valid.sum()),
        )
        _local.append({
            "spec": spec,
            "kept": False,  # set after filtering
            "reason": "",
            "mi_gain_lcb": float(mi_gain_lcb),
        })
        return _local

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
            _delayed(_eval_one_transform)(_b, _tn, _t)
            for _b, _tn, _t in _work_items
        )
    else:
        _results = [
            _eval_one_transform(_b, _tn, _t)
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

