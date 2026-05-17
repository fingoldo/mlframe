"""CompositeTargetDiscovery: main entry-point class that auto-finds the best (base, transform) pairs for a regression target. Orchestrates: base candidate ranking via residualised-MI, transform screening over the registry, optional tiny-model rerank, multi-base forward-stepwise auto-promotion, validation gating, and CompositeProvenance generation. Split out of composite.py to isolate discovery internals from the lightweight wrapper / spec / provenance surface; composite.py re-exports CompositeTargetDiscovery at its bottom for full back-compat."""


from __future__ import annotations

import hashlib
import logging
import math
import re
import warnings
from datetime import datetime
from timeit import default_timer as timer
from typing import (
    Any, Callable, Dict, FrozenSet, Iterator, List, Optional, Sequence, Tuple, Union,
)

import numpy as np
import pandas as pd

from .composite_spec import CompositeSpec
from .composite_auto_detect import (
    detect_time_column_candidates,
    sort_df_by_time_column,
    detect_group_column_candidates,
)
from .composite_bayesian import bayesian_alpha_fit
from .composite_cache import (
    DiscoveryCache,
    data_signature,
    make_discovery_cache_key,
)
from .composite_ensemble import (
    CompositeCrossTargetEnsemble,
    compute_oof_holdout_predictions,
    derive_seeds,
    detect_gpu_in_use,
    env_signature,
)
from .composite_estimator import CompositeTargetEstimator
from .composite_feature_stacking import (
    composite_oof_predictions,
    composite_predictions_as_feature,
)
from .composite_forward_stepwise import forward_stepwise_multi_base
from .composite_interaction_bases import generate_interaction_bases
from .composite_provenance import (
    CompositeProvenance,
    report_to_markdown,
)
from .composite_screening import (
    _build_tiny_model,
    _extract_column_array,
    _is_numeric_column,
    _mi_pair_bin,
    _mi_to_target,
    _mi_to_target_prebinned,
    _prebin_feature_columns,
    _per_bin_rmse,
    _residualise,
    _safe_abs_corr_all,
    _safe_corr,
    _sample_indices,
    _silence_tiny_model_output,
    _tiny_cv_rmse_raw_y,
    _tiny_cv_rmse_raw_y_multiseed,
    _tiny_cv_rmse_y_scale,
    _tiny_cv_rmse_y_scale_multiseed,
)
from .composite_stacking import (
    max_off_diagonal_correlation,
    residual_correlation_matrix,
    stacking_aware_gate,
)
from .composite_streaming import streaming_alpha_check_and_refit
from .composite_transforms import (
    DomainViolationError,
    Transform,
    UnknownTransformError,
    _linear_residual_fit,
    _linear_residual_multi_fit,
    _TRANSFORMS_REGISTRY,
    compose_target_name,
    get_transform,
    list_transforms,
)

logger = logging.getLogger(__name__)


class CompositeTargetDiscovery:
    """Auto-find the best (base, transform) pairs for a regression target.

    Workflow
    --------
    1. Resolve base candidates (auto via residualised-MI ranking, OR
       user-supplied list). Apply the forbidden-pattern + corr + ptp
       filters to drop columns that are leakage-prone (target encoding,
       derived-from-y, near-constant).
    2. For each (base, transform) pair, fit transform-specific params
       on **train_idx only**, compute T on the train sample, and score
       MI(T, X \\ {base}) against MI(y, X).
    3. Filter by ``min_valid_domain_frac`` and ``eps_mi_gain``, sort
       by MI gain descending, keep top ``top_k_after_mi``.

    Leakage discipline (CRITICAL)
    -----------------------------
    Every fitted parameter (alpha/beta for linear_residual, MAD for
    logratio, eps for ratio, MI bin edges for screening, y-clip
    quantiles, etc.) is computed from rows in ``train_idx`` ONLY. Test
    and validation rows are NEVER touched at fit. The unit test
    ``test_alpha_train_only_changes_with_train_idx`` proves this:
    fitting on two different ``train_idx`` slices of the same df
    yields different alpha, while fitting on the same train_idx
    yields identical alpha. If you ever change the implementation to
    add a "use full df for X" shortcut, that test will fail.
    """

    def __init__(self, config: Any) -> None:
        if isinstance(config, dict):
            from .configs import CompositeTargetDiscoveryConfig
            config = CompositeTargetDiscoveryConfig(**config)
        self.config = config
        self._patterns_compiled: list[re.Pattern] = [
            re.compile(p) for p in config.forbidden_base_patterns
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        df: Any,
        target_col: str,
        feature_cols: Sequence[str],
        train_idx: np.ndarray,
        val_idx: np.ndarray | None = None,
        test_idx: np.ndarray | None = None,
    ) -> CompositeTargetDiscovery:
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
                skew = float(np.mean(z_centered ** 3))
                kurt = float(np.mean(z_centered ** 4) - 3.0)
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

        # Resolve base candidates.
        base_candidates = self._resolve_base_candidates(
            df, target_col, usable_features, y_train, train_idx,
        )
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
        candidates: list[dict[str, Any]] = []
        for base in base_candidates:
            base_train = _extract_column_array(df, base)[train_idx]
            self._auto_base_pool[base] = base_train
            base_screen = base_train[sample_idx]
            x_remaining = [c for c in usable_features if c != base]
            if not x_remaining:
                continue
            x_remaining_matrix = self._build_feature_matrix(
                df, x_remaining, train_idx_screen,
            )

            # Pre-bin feature columns for this base so all transforms
            # evaluated against it reuse the same quantile edges + bin
            # indices (~50% of MI wall time).
            _x_prebinned = (
                _prebin_feature_columns(
                    x_remaining_matrix, nbins=int(self.config.mi_nbins),
                )
                if self.config.mi_estimator == "bin" else None
            )

            # MI(y, X_remaining) -- baseline for THIS base. The model
            # trained on raw y from X_remaining (base dropped from
            # features) sets the bar; a composite target only earns
            # its keep if MI(T, X_remaining) > this.
            _mi_fn = (
                _mi_to_target_prebinned
                if _x_prebinned is not None else _mi_to_target
            )
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

            for transform_name in self.config.transforms:
                try:
                    transform = get_transform(transform_name)
                except UnknownTransformError as exc:
                    logger.warning("[CompositeTargetDiscovery] %s; skipping.", exc)
                    continue

                # Domain check on train, drop invalids, fit transform
                # params on the surviving rows only.
                valid = transform.domain_check(y_train, base_train)
                valid_frac = float(valid.mean()) if valid.size else 0.0
                if valid_frac < self.config.min_valid_domain_frac:
                    candidates.append(self._reject(
                        base, transform_name, mi_y_for_base, valid_frac,
                        reason=f"valid_domain_frac={valid_frac:.3f} "
                               f"< {self.config.min_valid_domain_frac:.3f}",
                    ))
                    continue
                if not valid.any():
                    continue

                fitted_params = transform.fit(y_train[valid], base_train[valid])
                # T on the screening sample (which is a subset of train).
                valid_screen = transform.domain_check(y_screen, base_screen)
                if valid_screen.sum() < 50:
                    candidates.append(self._reject(
                        base, transform_name, mi_y_for_base, valid_frac,
                        reason="too few rows in screening sample after domain filter",
                    ))
                    continue
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
                        except Exception:
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
                candidates.append({
                    "spec": spec,
                    "kept": False,  # set after filtering
                    "reason": "",
                    "mi_gain_lcb": float(mi_gain_lcb),
                })

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

        kept_specs.sort(key=lambda s: -s.mi_gain)
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

        elapsed = timer() - t0
        logger.info(
            "[CompositeTargetDiscovery] target='%s' discovered %d spec(s) "
            "from %d candidate(s) in %.2fs",
            target_col, len(kept_specs), len(candidates), elapsed,
        )

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

    def iter_transform(self, df: Any) -> Iterator[tuple[str, np.ndarray]]:
        """Yield ``(spec_name, T_values)`` per discovered spec, applied
        to ALL rows of ``df``. Streaming generator: we never
        materialise more than one T column at a time -- on a 4M-row
        frame with K=8 specs that saves ~250 MB peak.

        Rows that fail ``domain_check`` get ``NaN`` in T, so downstream
        target-aware filters drop them automatically when fitting the
        per-spec model. The wrapper at predict time uses its own
        ``y_train_median`` fallback for those rows.
        """
        if not getattr(self, "specs_", None):
            return
        target_col = self._target_col
        y_full = _extract_column_array(df, target_col)
        for spec in self.specs_:
            base_full = _extract_column_array(df, spec.base_column)
            transform = get_transform(spec.transform_name)
            valid = transform.domain_check(y_full, base_full)
            t = np.full(y_full.shape[0], np.nan, dtype=np.float64)
            if valid.any():
                t[valid] = transform.forward(
                    y_full[valid], base_full[valid], spec.fitted_params,
                )
            yield spec.name, t

    def export_specs(self) -> list[dict[str, Any]]:
        """Plain-dict snapshot of discovered specs for ``metadata`` storage."""
        return [
            {
                "name": s.name,
                "target_col": s.target_col,
                "transform_name": s.transform_name,
                "base_column": s.base_column,
                "fitted_params": dict(s.fitted_params),
                "mi_gain": s.mi_gain,
                "mi_y": s.mi_y,
                "mi_t": s.mi_t,
                "valid_domain_frac": s.valid_domain_frac,
                "n_train_rows": s.n_train_rows,
            }
            for s in getattr(self, "specs_", [])
        ]

    def report(self) -> list[dict[str, Any]]:
        """All evaluated candidates including rejected ones with reasons."""
        return list(getattr(self, "report_", []))

    @property
    def tiny_rerank_scores_(self) -> dict[str, float]:
        """Per-spec tiny CV-RMSE on y-scale (after Phase B rerank).

        Empty when ``screening="mi"`` or rerank didn't run. Keyed by
        spec name. Useful for surfacing "why did this composite get
        kept / rejected" diagnostics.
        """
        return dict(getattr(self, "_tiny_rerank_scores", {}))

    @property
    def raw_y_baseline_rmse_(self) -> float:
        """Tiny CV-RMSE of a model trained directly on raw y on the
        same screening sample / folds / family used by Phase B rerank.

        ``nan`` when the raw-y baseline gate didn't run
        (``require_beats_raw_baseline=False``, screening="mi", or
        degenerate sample).
        """
        return float(getattr(self, "_raw_y_baseline_rmse", float("nan")))

    def filter_drops(self) -> list[dict[str, Any]]:
        """Columns that were filtered out before MI ranking, with reason
        and the offending value (corr, ptp, n_finite). Useful for audit
        when discovery seems to "miss" an obvious base candidate -- the
        most common cause is a corr-threshold false positive on a
        legitimate autoregressive lag feature.
        """
        return list(getattr(self, "_filter_drops", []))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _filter_features(
        self,
        df: Any,
        feature_cols: Sequence[str],
        y_train: np.ndarray,
        train_idx: np.ndarray,
    ) -> list[str]:
        """Drop columns that are non-numeric, near-constant on train,
        match a forbidden name pattern, or correlate suspiciously
        highly with y on train (likely derived-from-y leakage).

        Drops are recorded on ``self._filter_drops`` (list of dicts
        with name + reason + value) so :meth:`fit` can surface them
        in the report and so callers can audit false positives -- the
        corr filter in particular is prone to misfiring on legitimate
        autoregressive lag features such as ``TVT_prev``.
        """
        # First pass: cheap-fail filters (name patterns, type, finite
        # count, near-constant). Build a list of survivors + their
        # train-row arrays so the corr check can be vectorised across
        # all survivors in ONE matrix op (~2.2x faster vs per-column
        # ``_safe_corr`` loop on 200 cols x 80K rows).
        drops: list[dict[str, Any]] = []
        corr_drops: list[tuple[str, float]] = []
        candidates: list[str] = []
        candidate_arrays: list[np.ndarray] = []
        for col in feature_cols:
            if col == self._target_col:
                continue
            if any(p.search(col) for p in self._patterns_compiled):
                drops.append({"name": col, "reason": "forbidden_pattern"})
                continue
            if not _is_numeric_column(df, col):
                drops.append({"name": col, "reason": "non_numeric"})
                continue
            arr = _extract_column_array(df, col)[train_idx]
            finite_mask = np.isfinite(arr)
            if finite_mask.sum() < 50:
                drops.append({
                    "name": col, "reason": "insufficient_finite_rows",
                    "n_finite": int(finite_mask.sum()),
                })
                continue
            ptp = float(np.ptp(arr[finite_mask]))
            if ptp <= self.config.constant_base_eps:
                drops.append({
                    "name": col, "reason": "constant_or_near_constant",
                    "ptp": ptp,
                })
                continue
            candidates.append(col)
            candidate_arrays.append(arr)

        # Vectorised corr filter on survivors. Replaces the per-column
        # ``abs(_safe_corr(arr, y_train))`` loop. NaN rows in the
        # survivor matrix are imputed with column-mean before the
        # corr-vs-y dot product, which is a small approximation
        # versus per-column NaN masking but only matters for columns
        # with sparse NaN -- and those have already passed the
        # ``finite_mask.sum() < 50`` gate above with at least 50
        # finite rows. Acceptable trade-off for the ~600ms saving on
        # 200-feature filter calls.
        kept: list[str] = []
        if candidates:
            X_train = np.column_stack(candidate_arrays)
            # Impute non-finite cells with per-column mean to keep
            # the vectorised dot product well-defined.
            col_means = np.nanmean(
                np.where(np.isfinite(X_train), X_train, np.nan),
                axis=0,
            )
            non_finite_mask = ~np.isfinite(X_train)
            if non_finite_mask.any():
                X_train = X_train.copy()
                # Per-column mean fill (broadcast).
                X_train[non_finite_mask] = np.broadcast_to(
                    col_means, X_train.shape,
                )[non_finite_mask]
            abs_corrs = _safe_abs_corr_all(y_train, X_train)
            threshold = float(self.config.forbidden_base_corr_threshold)
            for col, corr_val in zip(candidates, abs_corrs.tolist()):
                if corr_val >= threshold:
                    drops.append({
                        "name": col, "reason": "forbidden_base_corr_threshold",
                        "corr": float(corr_val), "threshold": threshold,
                    })
                    corr_drops.append((col, float(corr_val)))
                else:
                    kept.append(col)
        self._filter_drops = drops
        # Loud warning for corr-threshold drops: this is the filter
        # most likely to misfire on legitimate strong predictors
        # (autoregressive lags, near-deterministic features). Make it
        # visible at INFO so users can spot a false positive.
        if corr_drops:
            corr_drops.sort(key=lambda t: -t[1])
            preview = ", ".join(f"{n}=|corr|{c:.6f}" for n, c in corr_drops[:5])
            logger.info(
                "[CompositeTargetDiscovery] corr-threshold filter dropped "
                "%d feature(s) (threshold=%.6f): %s%s. If a legitimate "
                "lag/strong predictor was dropped, raise "
                "forbidden_base_corr_threshold or pass it via "
                "base_candidates=[...] explicitly.",
                len(corr_drops),
                self.config.forbidden_base_corr_threshold,
                preview,
                "" if len(corr_drops) <= 5 else f" (+{len(corr_drops) - 5} more)",
            )
        return kept

    def _resolve_base_candidates(
        self,
        df: Any,
        target_col: str,
        usable_features: Sequence[str],
        y_train: np.ndarray,
        train_idx: np.ndarray,
    ) -> list[str]:
        """Return the base candidates to evaluate.

        For ``base_candidates="auto"``, rank features by *structural*
        MI gain: for each feature ``x``, residualise ``y`` against
        ``x`` (remove the linear contribution) and score
        ``MI(residual, X \\ {x})``. Features whose autoregressive
        contribution genuinely *opens up* the rest of the feature
        space rank higher than features that just correlate with ``y``
        through a global trend.
        """
        config = self.config
        if isinstance(config.base_candidates, str) and config.base_candidates == "auto":
            return self._auto_base(df, usable_features, y_train, train_idx)
        # Explicit list. Keep only entries that survived feature filters.
        explicit = list(config.base_candidates)
        kept = [c for c in explicit if c in usable_features]
        if len(kept) != len(explicit):
            dropped = sorted(set(explicit) - set(kept))
            logger.warning(
                "[CompositeTargetDiscovery] explicit base_candidates dropped "
                "by filters (forbidden/constant/non-numeric/leak-corr): %s", dropped,
            )
        return kept

    def _auto_base(
        self,
        df: Any,
        usable_features: Sequence[str],
        y_train: np.ndarray,
        train_idx: np.ndarray,
    ) -> list[str]:
        """Rank candidates by per-feature MI with y on the screening
        sample, take the top-K.

        Why pairwise MI(y, x) and not the more elaborate "residualised
        gain" of round-2 critique R2.27: the residualised metric
        ranks candidates by how predictable ``y - alpha*x - beta``
        is from the remaining features. On a feature whose linear
        contribution is small, the residual still contains the
        dominant feature itself (we did not subtract it), so the
        remaining feature set predicts the residual perfectly --
        which inverts the ranking versus what we want. Pairwise
        MI(y, x) directly measures "how much information about y
        does this single feature carry" and surfaces ``TVT_prev`` at
        top-1 on the canonical autoregressive case.

        The forbidden-base + ptp + corr filters elsewhere already
        catch the pathologies the residualised metric was meant to
        guard against (target encoding, near-constant features,
        derived-from-y).
        """
        if not usable_features:
            # Every feature was filtered out (forbidden / non-numeric /
            # constant / corr-threshold). Don't ask sklearn to do MI on
            # a 0-column matrix -- it raises ValueError. Return empty
            # cleanly so discovery falls through to the no-spec path.
            logger.info(
                "[CompositeTargetDiscovery] auto-base: 0 usable features "
                "after filtering; no base candidates available."
            )
            return []

        # Hint-aware ranking: BaselineDiagnostics ablation already
        # measured each feature's predictive contribution directly
        # (drop feature -> RMSE delta). That signal beats pairwise
        # MI(y, x), which gets fooled by features with global trend
        # but no structural residual signal (spatial coords on
        # geographically-trended y is the canonical case). When a
        # hint is provided, prepend hint features (preserving order)
        # then fill remaining slots with MI-ranked features.
        usable_set = set(usable_features)
        hint_raw = list(getattr(self.config, "dominant_features_hint", None) or [])
        hint_kept: list[str] = []
        hint_dropped: list[str] = []
        for c in hint_raw:
            if c in usable_set and c not in hint_kept:
                hint_kept.append(c)
            else:
                hint_dropped.append(c)
        if hint_dropped:
            logger.info(
                "[CompositeTargetDiscovery] dominant_features_hint dropped "
                "%d entries (filtered or not in feature_cols): %s",
                len(hint_dropped), hint_dropped[:5],
            )
        top_k = self.config.auto_base_top_k
        # R10c bug #5 fix: adaptive hint cap. Previous fixed cap of
        # ``max(1, top_k // 2)`` was too aggressive when BD ablation
        # confidently identified the dominant base (e.g. delta% > 100%
        # for the top-1 feature on production TVT). Now: if the user
        # supplied a hint AND we have ablation strengths in metadata,
        # check the strength signal. Strong hint (top-1 delta_pct >
        # ``hint_strength_threshold_pct``, default 50%) -> use FULL
        # hint (no cap). Weak/absent strength info -> fall back to the
        # half-slot cap so MI-leaders still get evaluated.
        #
        # Rationale: BD ablation directly measures "drop feature -> RMSE
        # delta%" which is a high-quality signal. When it screams +501%
        # for TVT_prev (real production case), trust it; don't dilute
        # with MI-leaders that may be lower-quality features.
        strong_hint_threshold = float(getattr(
            self.config, "hint_strength_threshold_pct", 50.0,
        ))
        # Strength info is plumbed via the suite-level hint precompute
        # at core.py and stored on the discovery instance for this fit.
        # Absent = treat as unknown strength -> use half-slot cap.
        hint_strengths = getattr(self, "_hint_strengths_pct", None)
        is_strong_hint = (
            hint_strengths is not None
            and len(hint_strengths) > 0
            and max(hint_strengths[:len(hint_kept)]) >= strong_hint_threshold
        )
        if is_strong_hint:
            # Full hint -- no cap. Log so it's auditable.
            logger.info(
                "[CompositeTargetDiscovery] auto-base using FULL hint "
                "(%d candidates, max ablation delta%% = %.1f%% >= %.1f%% "
                "threshold; trusting BD over MI ranking).",
                len(hint_kept), max(hint_strengths[:len(hint_kept)]),
                strong_hint_threshold,
            )
            hint_cap = top_k  # effectively no cap
        else:
            hint_cap = max(1, top_k // 2)
            if len(hint_kept) > hint_cap:
                logger.info(
                    "[CompositeTargetDiscovery] auto-base capping hint "
                    "contribution to %d/%d slots (was %d hint candidates; "
                    "strength signal weak or absent) so MI-leaders also "
                    "get evaluated; full hint list preserved as feature "
                    "ordering source.",
                    hint_cap, top_k, len(hint_kept),
                )
                hint_kept = hint_kept[:hint_cap]

        sample_idx = _sample_indices(
            train_idx.size, self.config.mi_sample_n, self.config.random_state,
            strategy=getattr(self.config, "mi_sample_strategy", "random"),
            y=y_train,
            n_strata=getattr(self.config, "mi_n_strata", 10),
        )
        train_idx_screen = train_idx[sample_idx]
        y_screen = y_train[sample_idx]

        x_matrix = self._build_feature_matrix(df, usable_features, train_idx_screen)
        finite = np.isfinite(y_screen) & np.all(np.isfinite(x_matrix), axis=1)
        if finite.sum() < 50:
            logger.warning(
                "[CompositeTargetDiscovery] auto-base: only %d finite rows in "
                "screening sample; falling back to feature-list order.", int(finite.sum()),
            )
            return list(usable_features)[: self.config.auto_base_top_k]
        # Per-feature MI honours config.mi_estimator: bin-based when
        # the screening pipeline opted for the fast estimator.
        if self.config.mi_estimator == "bin":
            mi_per_feature = np.array([
                _mi_pair_bin(x_matrix[finite, j], y_screen[finite],
                             nbins=self.config.mi_nbins)
                for j in range(x_matrix.shape[1])
            ])
        else:
            from sklearn.feature_selection import mutual_info_regression
            mi_per_feature = mutual_info_regression(
                x_matrix[finite], y_screen[finite],
                n_neighbors=self.config.mi_n_neighbors,
                random_state=self.config.random_state,
            )
        # R10b improvement #7: structural detectors for time-index
        # and spatial-coordinate features. Cheap heuristics applied
        # to the screening matrix; flagged features are demoted
        # (large MI penalty) so they only win base selection when
        # genuinely high-MI relative to alternatives.
        demote_set: set = set()
        if getattr(self.config, "auto_base_demote_time_index", True) \
                and finite.sum() >= 50:
            # Spearman(rank(x), arange(n)) computed as |corr(rankdata(x), arange(n))|.
            # ``scipy.stats.rankdata`` uses fractional (average) ranks for ties; the prior
            # ``argsort(argsort(x))`` assigned arbitrary integer positions to tied values,
            # which inflated |Spearman| toward 1.0 on columns with many duplicate values
            # (e.g. integer-encoded categoricals) and silently misfired the time-index demoter.
            try:
                from scipy.stats import rankdata as _rankdata
            except ImportError:  # pragma: no cover - scipy is a hard dep but allow graceful skip
                _rankdata = None
            n_screen = int(finite.sum())
            row_idx = np.arange(n_screen, dtype=np.float64)
            # R10c bug #2 extension: hint features are IMMUNE from
            # the time-index demoter too. BD ablation already proved
            # they predict y; demoting silently is wrong.
            time_hint_protected = set(hint_kept) if hint_kept else set()
            for j, col_name in enumerate(usable_features):
                if col_name in time_hint_protected:
                    continue
                col_finite = x_matrix[finite, j]
                if _rankdata is not None:
                    col_ranks = _rankdata(col_finite, method="average").astype(np.float64)
                else:
                    col_ranks = np.argsort(np.argsort(col_finite)).astype(np.float64)
                # Pearson on rank vs row-index = Spearman(x, time).
                spearman = abs(_safe_corr(col_ranks, row_idx))
                if spearman > 0.95:
                    demote_set.add(col_name)
            if demote_set:
                logger.info(
                    "[CompositeTargetDiscovery] auto-base detected %d "
                    "time-index-like feature(s) (rank ~ row order, "
                    "|Spearman| > 0.95): %s. Demoted in MI ranking.",
                    len(demote_set), sorted(demote_set)[:5],
                )
        if getattr(self.config, "auto_base_demote_spatial_coords", True) \
                and len(usable_features) >= 3 and finite.sum() >= 50:
            # R10c bug #1 fix: spatial-coord block detector tightened
            # after a production geological-data run demoted 17
            # features (entire feature set). Previously: ``>=2 cross-
            # correlations |corr|>0.5`` -- fires on any moderately-
            # correlated feature group.
            #
            # Tightened criteria for "spatial-coord block":
            #   1. Block size 3 <= K <= 6 (X/Y/Z triplet up to a
            #      5-coord positional spec; anything larger is a
            #      feature GROUP, not spatial coords).
            #   2. EVERY pair within the block has |corr| > 0.75
            #      (not 0.5 -- geological features routinely correlate
            #      at 0.5-0.7 from physics, not from being coords).
            #   3. Mean within-block |corr| > 0.80 (catches X/Y/Z
            #      typical corr range while rejecting lower-corr
            #      industrial feature groups).
            # All three must hold; otherwise the group is preserved.
            X_screen = x_matrix[finite]
            n_feats = X_screen.shape[1]
            # Vectorised |corr|: centre each column, normalise to unit-L2, then take Gram matrix
            # (~12x over the nested ``_safe_corr`` loop on 25 features x 50k rows). Constant
            # columns (zero variance) map to all-zero correlations, matching ``_safe_corr``'s
            # degenerate-input contract.
            corr_matrix = np.zeros((n_feats, n_feats))
            if n_feats >= 2 and X_screen.shape[0] >= 3:
                Xc = X_screen - X_screen.mean(axis=0)
                norms = np.sqrt((Xc ** 2).sum(axis=0))
                live = norms > 1e-12
                if live.sum() >= 2:
                    live_idx = np.where(live)[0]
                    Xn = Xc[:, live_idx] / norms[live_idx]
                    gram = np.abs(Xn.T @ Xn)
                    np.fill_diagonal(gram, 0.0)
                    corr_matrix[np.ix_(live_idx, live_idx)] = gram
            spatial_demoted: list[str] = []
            # For each feature j, find its "tight neighbourhood":
            # features k where |corr(j, k)| > 0.75. If that
            # neighbourhood (including j) is size 3-6 AND has mean
            # within-pair corr > 0.80, demote ALL members.
            for j, _col_name in enumerate(usable_features):
                tight_neighbours = np.where(corr_matrix[j] > 0.75)[0]
                if not (2 <= len(tight_neighbours) <= 5):
                    continue
                block_idx = np.r_[j, tight_neighbours]
                block_idx = np.unique(block_idx)
                if not (3 <= len(block_idx) <= 6):
                    continue
                # Mean within-block pairwise corr.
                sub = corr_matrix[np.ix_(block_idx, block_idx)]
                upper = sub[np.triu_indices_from(sub, k=1)]
                if upper.size == 0:
                    continue
                if float(upper.mean()) < 0.80:
                    continue
                # Also require EVERY pair > 0.75 (no weak edge in the
                # cluster).
                if float(upper.min()) < 0.75:
                    continue
                # Cluster qualifies -- demote every member EXCEPT
                # those on the hint list (BD ablation already proved
                # they predict y; demoting them silently is the same
                # production bug pattern as the dedup-vs-hint race).
                hint_protected = set(hint_kept) if hint_kept else set()
                for k in block_idx:
                    name_k = usable_features[k]
                    if name_k in hint_protected:
                        continue
                    if name_k not in demote_set:
                        demote_set.add(name_k)
                        spatial_demoted.append(name_k)
            if spatial_demoted:
                logger.info(
                    "[CompositeTargetDiscovery] auto-base detected "
                    "spatial-coord block of %d feature(s) (tight "
                    "cluster, |pair-corr| > 0.75, mean > 0.80, size "
                    "3-6): %s. Demoted in MI ranking.",
                    len(spatial_demoted),
                    sorted(spatial_demoted)[:8],
                )

        # R10b improvement #2: permutation-MI null filter. Catches
        # features whose MI(y, x) is non-trivial only because of a
        # shared monotonic component (time/spatial trend), not
        # structural information about y. Computes MI(y, shuffle(x))
        # with block shuffles to preserve marginal autocorrelation,
        # then requires MI(y, x) > mean_null + n_sigma * std_null.
        n_perms = int(getattr(self.config, "auto_base_null_perms", 0) or 0)
        if n_perms > 0:
            n_sigma = float(getattr(
                self.config, "auto_base_null_z_threshold", 3.0,
            ))
            block_len_cfg = getattr(
                self.config, "auto_base_null_block_length", "auto",
            )
            n_screen = int(finite.sum())
            if isinstance(block_len_cfg, str) and block_len_cfg == "auto":
                block_len = max(1, int(np.sqrt(n_screen)))
            else:
                try:
                    block_len = max(1, int(block_len_cfg))
                except (TypeError, ValueError):
                    block_len = max(1, int(np.sqrt(n_screen)))

            def _block_shuffle(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
                if block_len <= 1:
                    out = arr.copy()
                    rng.shuffle(out)
                    return out
                m = arr.size
                n_blocks = (m + block_len - 1) // block_len
                blocks = [arr[i * block_len:(i + 1) * block_len]
                          for i in range(n_blocks)]
                perm = rng.permutation(len(blocks))
                shuffled = np.concatenate([blocks[p] for p in perm])
                return shuffled[:m]

            rng_perm = np.random.default_rng(
                int(self.config.random_state) + 7919
            )
            y_finite = y_screen[finite]
            null_means = np.zeros(x_matrix.shape[1])
            null_stds = np.zeros(x_matrix.shape[1])
            for j in range(x_matrix.shape[1]):
                col = x_matrix[finite, j]
                null_mis = np.empty(n_perms)
                for p in range(n_perms):
                    shuffled = _block_shuffle(col, rng_perm)
                    if self.config.mi_estimator == "bin":
                        null_mis[p] = _mi_pair_bin(
                            shuffled, y_finite, nbins=self.config.mi_nbins,
                        )
                    else:
                        from sklearn.feature_selection import mutual_info_regression
                        null_mis[p] = float(mutual_info_regression(
                            shuffled.reshape(-1, 1), y_finite,
                            n_neighbors=self.config.mi_n_neighbors,
                            random_state=self.config.random_state,
                        )[0])
                null_means[j] = float(null_mis.mean())
                null_stds[j] = float(null_mis.std())
            null_threshold = null_means + n_sigma * np.maximum(
                null_stds, 1e-9,
            )
            passes_null = mi_per_feature > null_threshold
            null_dropped: list[tuple[str, float, float]] = []
            for j, (mi_val, col_name) in enumerate(
                zip(mi_per_feature.tolist(), usable_features)
            ):
                if not passes_null[j]:
                    null_dropped.append((
                        col_name, float(mi_val), float(null_threshold[j]),
                    ))
            if null_dropped:
                preview = ", ".join(
                    f"{n}(mi={m:.4f}<=null+{n_sigma:.0f}sigma={t:.4f})"
                    for n, m, t in null_dropped[:5]
                )
                logger.info(
                    "[CompositeTargetDiscovery] permutation-MI null "
                    "dropped %d feature(s) (z<%.0f, block_len=%d, "
                    "perms=%d): %s",
                    len(null_dropped), n_sigma, block_len, n_perms,
                    preview,
                )
            # Mask out features that didn't pass the null.
            mi_for_ranking = np.where(passes_null, mi_per_feature, -np.inf)
        else:
            mi_for_ranking = mi_per_feature.copy()
        # R10b improvement #7: apply demotion to time-index / spatial-
        # coord candidates. Subtract a large penalty so they sort
        # below all non-demoted features but stay reachable as a
        # last resort.
        if demote_set:
            for j, col_name in enumerate(usable_features):
                if col_name in demote_set:
                    mi_for_ranking[j] -= 1e6
        ranked = sorted(
            zip(mi_for_ranking.tolist(), usable_features),
            key=lambda t: -t[0],
        )
        # Strip features whose MI was masked out (-inf) so the ranking
        # tail doesn't include null-failed candidates.
        ranked = [(m, c) for m, c in ranked if math.isfinite(m)]
        # Cross-base correlation dedup. Two highly-correlated bases
        # (typical: ``TVT_prev``, ``TVT_prev_lag2``, ``TVT_smooth_3``)
        # produce near-identical composites that waste Phase B compute
        # AND inflate ensemble correlation, hurting cross-target
        # diversity. After ranking, drop a candidate if its absolute
        # corr against any already-kept candidate exceeds
        # ``auto_base_dedup_corr_threshold``. Skipped candidates are
        # logged at INFO. Configurable via
        # ``CompositeTargetDiscoveryConfig.auto_base_dedup_corr_threshold``;
        # set to 1.0 to disable.
        dedup_threshold = float(getattr(
            self.config, "auto_base_dedup_corr_threshold", 0.95,
        ))
        if 0 < dedup_threshold < 1.0 and len(ranked) > 1:
            kept_ranked: list[tuple[float, str]] = []
            kept_arrays: dict[str, np.ndarray] = {}
            dedup_dropped: list[tuple[str, str, float]] = []
            # R10c bug #2 fix: hint features are IMMUNE from dedup.
            # Otherwise on geological data with high feature
            # cross-correlation (e.g. Z ~ TVT_prev at |corr|=0.974),
            # the lower-MI hint candidate gets dropped against a
            # higher-MI non-hint one, then later re-injected by the
            # hint-merge step with a poisoned score from the demoter.
            # Hint features were chosen by the upstream BD ablation
            # specifically because they predict y; their relevance is
            # already established and shouldn't be filtered by
            # raw-feature redundancy.
            hint_set = set(hint_kept)
            # Pre-compute column-name -> matrix-index lookup once. ``usable_features.index(col)``
            # inside the loop was O(n) per iteration, O(n^2) over ``ranked``.
            _name_to_col_idx = {name: i for i, name in enumerate(usable_features)}
            for mi_score, col in ranked:
                col_arr = x_matrix[finite, _name_to_col_idx[col]]
                drop_due_to: tuple[str, float] | None = None
                if col in hint_set:
                    # Hint features always pass dedup.
                    pass
                else:
                    for kept_col, kept_arr in kept_arrays.items():
                        pair_corr = abs(_safe_corr(col_arr, kept_arr))
                        if pair_corr >= dedup_threshold:
                            drop_due_to = (kept_col, float(pair_corr))
                            break
                if drop_due_to is None:
                    kept_ranked.append((mi_score, col))
                    kept_arrays[col] = col_arr
                else:
                    dedup_dropped.append(
                        (col, drop_due_to[0], drop_due_to[1])
                    )
            if dedup_dropped:
                preview = ", ".join(
                    f"{c}~={ref}(|corr|={corr:.3f})"
                    for c, ref, corr in dedup_dropped[:5]
                )
                logger.info(
                    "[CompositeTargetDiscovery] auto-base dedup dropped "
                    "%d candidate(s) at |corr|>=%.3f: %s",
                    len(dedup_dropped), dedup_threshold, preview,
                )
            ranked = kept_ranked
        # Combine hint (priority) + MI-ranked tail. Hint always wins
        # the leading slots; MI fills up to auto_base_top_k.
        if hint_kept:
            mi_tail: list[str] = []
            for _, c in ranked:
                if c in hint_kept:
                    continue
                mi_tail.append(c)
                if len(hint_kept) + len(mi_tail) >= top_k:
                    break
            top = hint_kept + mi_tail
            top = top[:top_k]
            mi_lookup = {c: mi for mi, c in ranked}
            scores = ", ".join(
                f"{c}={mi_lookup.get(c, float('nan')):.4f}{'(hint)' if c in hint_kept else ''}"
                for c in top
            )
            logger.info(
                "[CompositeTargetDiscovery] auto-base top-%d (%d hint, %d MI): %s",
                len(top), len(hint_kept), len(mi_tail), scores,
            )
            return top

        top = [c for _, c in ranked[: top_k]]
        if top:
            scores = ", ".join(
                f"{c}={mi:.4f}" for mi, c in ranked[: top_k]
            )
            logger.info(
                "[CompositeTargetDiscovery] auto-base top-%d by MI(y, x): %s",
                len(top), scores,
            )
        return top

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

        if self.config.tiny_screening_models == "single_lgbm":
            families = ["lightgbm"]
        else:  # per_family
            families = [f for f in self.config.tiny_screening_families]
            if not families:
                families = ["lightgbm"]

        # ENS-P2-5: hoist per-bin-enabled check above the first pass so we
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
        # (the typical case: auto-base picks one TVT_prev-style
        # dominant feature, all K transforms operate on it), the
        # per-base ``x_remaining`` matrix and ``base_screen`` array
        # are recomputable from the same inputs. Cache them by base
        # to avoid K redundant builds (each ~50 ndarray copies on a
        # 200K-row sample).
        per_family_scores: dict[str, list[float]] = {f: [] for f in families}
        # ENS-P2-5: parallel buffer for per-bin RMSE captured during the
        # first pass. Keyed by spec.name -> per-bin ndarray. Only populated
        # when ``per_bin_enabled_pre`` is True.
        _per_bin_first_pass: dict[str, np.ndarray] = {}
        _per_base_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for spec in kept_specs:
            cached = _per_base_cache.get(spec.base_column)
            if cached is None:
                base_screen = (
                    _extract_column_array(df, spec.base_column)[train_idx_screen]
                )
                x_remaining = [
                    c for c in usable_features if c != spec.base_column
                ]
                x_matrix = self._build_feature_matrix(
                    df, x_remaining, train_idx_screen,
                )
                _per_base_cache[spec.base_column] = (base_screen, x_matrix)
            else:
                base_screen, x_matrix = cached
            transform = get_transform(spec.transform_name)
            n_seed_repeats = max(1, int(getattr(
                self.config, "tiny_model_n_seed_repeats", 1,
            )))
            use_wilcoxon = bool(getattr(
                self.config, "use_wilcoxon_gate", False,
            ))
            for family in families:
                # ENS-P2-5: capture per-bin alongside RMSE in the SAME pass
                # for the first family only (per-bin breakdown only checks
                # families[0] in the legacy second pass).
                _is_first_family = (family == families[0])
                want_per_bin = bool(per_bin_enabled_pre and _is_first_family)
                if use_wilcoxon:
                    result = _tiny_cv_rmse_y_scale_multiseed(
                        y_train=y_screen,
                        base_train=base_screen,
                        transform=transform,
                        fitted_params=spec.fitted_params,
                        x_train_matrix=x_matrix,
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
                        return_per_seed=True,
                        return_per_bin=want_per_bin,
                        n_bins=per_bin_n_bins_pre or 5,
                    )
                    if want_per_bin:
                        # (rmse, per_bin, per_seed)
                        rmse, per_bin_first, per_seed = result[0], result[1], result[-1]
                        _per_bin_first_pass[spec.name] = per_bin_first
                    else:
                        rmse, per_seed = result[0], result[-1]
                    # Stash per-seed array on the discovery instance
                    # (keyed by spec.name + family) for the Wilcoxon
                    # gate to compare against raw-y per-seed.
                    self._wilcoxon_per_seed_composite = getattr(
                        self, "_wilcoxon_per_seed_composite", {}
                    )
                    self._wilcoxon_per_seed_composite[
                        (spec.name, family)
                    ] = per_seed
                else:
                    result = _tiny_cv_rmse_y_scale_multiseed(
                        y_train=y_screen,
                        base_train=base_screen,
                        transform=transform,
                        fitted_params=spec.fitted_params,
                        x_train_matrix=x_matrix,
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
                        return_per_bin=want_per_bin,
                        n_bins=per_bin_n_bins_pre or 5,
                    )
                    if want_per_bin and isinstance(result, tuple):
                        rmse, per_bin_first = result[0], result[1]
                        _per_bin_first_pass[spec.name] = per_bin_first
                    else:
                        rmse = result
                per_family_scores[family].append(rmse)

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

        # R10b improvement #1: regime-aware gate. In addition to the
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
            # ENS-P2-5: REUSE the per-bin breakdown captured during the
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
                    )
            # Per-base raw-y per-bin breakdown for the regime gate.
            # Cached by base column so multiple specs sharing a base
            # compute baselines once.
            if per_bin_enabled:
                for spec in kept_specs:
                    if spec.base_column in raw_per_bin_per_base:
                        continue
                    cached = _per_base_cache.get(spec.base_column)
                    if cached is None:
                        continue
                    base_screen, _ = cached
                    family = families[0]
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
                        return_per_bin=True, n_bins=per_bin_n_bins,
                        bin_var=base_screen,
                    )
                    if isinstance(raw_result, tuple):
                        _, raw_per_bin = raw_result
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
                    # R10b stat #4: paired Wilcoxon signed-rank test
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
                        if (comp_per_seed is not None
                                and raw_per_seed is not None
                                and len(comp_per_seed) == len(raw_per_seed)
                                and len(comp_per_seed) >= 3):
                            try:
                                from scipy.stats import wilcoxon
                                diff = comp_per_seed - raw_per_seed
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
                    # R10b improvement #1: per-bin gate. Composite
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
        order = np.argsort(agg_scores)
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
        return reranked

    def _build_feature_matrix(
        self, df: Any, cols: Sequence[str], idx: np.ndarray,
    ) -> np.ndarray:
        """Materialise a 2-D ndarray of the requested columns at the
        requested rows. Used only for MI screening on the small
        sample slice -- never on the full frame."""
        if not cols:
            return np.zeros((idx.size, 0), dtype=np.float64)
        cols_arrays = [_extract_column_array(df, c)[idx] for c in cols]
        return np.column_stack(cols_arrays)

    def _reject(
        self, base: str, transform_name: str, mi_y: float, valid_frac: float,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "spec": None,
            "kept": False,
            "rejected": True,
            "base": base,
            "transform_name": transform_name,
            "valid_domain_frac": valid_frac,
            "mi_y": mi_y,
            "reason": reason,
        }

    def _entry_to_report(self, entry: dict[str, Any]) -> dict[str, Any]:
        spec = entry.get("spec")
        if spec is None:
            return {
                "name": f"__{entry['transform_name']}__{entry['base']}",
                "kept": False,
                "rejected": True,
                "reason": entry["reason"],
                "base_column": entry["base"],
                "transform_name": entry["transform_name"],
                "mi_gain": float("nan"),
                "valid_domain_frac": entry.get("valid_domain_frac", float("nan")),
            }
        return {
            "name": spec.name,
            "kept": entry.get("kept", False),
            "rejected": False,
            "reason": entry.get("reason", ""),
            "base_column": spec.base_column,
            "transform_name": spec.transform_name,
            "mi_gain": spec.mi_gain,
            "mi_y": spec.mi_y,
            "mi_t": spec.mi_t,
            "valid_domain_frac": spec.valid_domain_frac,
            "n_train_rows": spec.n_train_rows,
        }



# ----------------------------------------------------------------------
# Re-exports from the post-split sub-modules. Existing code that did
# ``from mlframe.training.composite import detect_group_column_candidates`` etc.
# continues to work without change. New code should prefer the direct
# sub-module import (`mlframe.training.composite_auto_detect`) for
# clearer dependency boundaries.
# ----------------------------------------------------------------------
from .composite_auto_detect import (  # noqa: E402,F401
    _GROUP_DETECT_DEFAULT_MIN_UNIQUE,
    _GROUP_DETECT_DEFAULT_MAX_UNIQUE,
    _GROUP_DETECT_DEFAULT_MIN_SIZE_RATIO,
)
from .composite_cache import (  # noqa: E402,F401
    _DISCOVERY_SIGNATURE_SAMPLE_N,
)
from .composite_interaction_bases import (  # noqa: E402,F401
    _INTERACTION_OPS_DEFAULT,
)

# Dependent helper re-exports.
from .composite_forward_stepwise import (  # noqa: E402,F401
    _MULTI_BASE_DEFAULT_MAX_K,
    _MULTI_BASE_DEFAULT_MIN_MARGINAL_GAIN,
)
from .composite_streaming import (  # noqa: E402,F401
    _STREAMING_DEFAULT_Z_THRESHOLD,
    _STREAMING_DEFAULT_MIN_BUFFER_N,
)
from .composite_bayesian import (  # noqa: E402,F401
    _BAYESIAN_ALPHA_DEFAULT_N_BOOTSTRAP,
    _BAYESIAN_ALPHA_DEFAULT_CI_LEVEL,
)


# ----------------------------------------------------------------------
