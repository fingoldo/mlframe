"""Per-transform candidate evaluator lifted out of ``_composite_discovery_fit.fit``.

``eval_one_transform`` was the ``_eval_one_transform`` nested closure inside ``fit``; it is lifted to module level so the parallel-dispatch wrapper can call it without re-capturing the (large) per-base arrays via closure cells. Every variable the closure captured from the enclosing ``fit`` scope is now an explicit parameter: the discovery instance (``self``, for ``self.config`` + ``self._reject``), the per-base context map (``base_contexts``), and the shared ``y_train`` / ``y_screen`` / ``target_col``. The function is pure w.r.t. shared state: it reads ``base_contexts[base]`` (read-only after setup) and returns a fresh list, so it is safe to call concurrently from the threading pool.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..spec import CompositeSpec
from .screening import _mi_to_target, _mi_to_target_prebinned
from ..transforms import compose_target_name

logger = logging.getLogger(__name__)


def eval_one_transform(
    self,
    base: str,
    transform_name: str,
    transform,
    *,
    base_contexts: dict,
    y_train: np.ndarray,
    y_screen: np.ndarray,
    target_col: str,
) -> list[dict[str, Any]]:
    """Returns 0 or 1 candidate dict for one (base, transform) pair.

    Pulls per-base arrays from ``base_contexts[base]`` (read-only once setup completes). Writes go to the returned list, never to the enclosing ``candidates`` list, so calling this concurrently from a thread pool is safe.
    """
    _ctx = base_contexts[base]
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

    # P13 (2026-06-11) bit-identical sub-fix: gather ``y_train[valid]`` /
    # ``base_train[valid]`` ONCE here and reuse the same arrays for both the
    # transform fit AND (below) the residual-std probe, instead of fancy-index
    # gathering the same rows a second time at the probe. ``transform.fit`` /
    # ``forward`` only READ these arrays (verified across linear/nonlinear
    # transforms), so sharing one gather is bit-identical. ``_valid_stale``
    # tracks whether the T15 block below shrinks ``valid`` -- if it does, the
    # reused gather is re-taken on the narrowed mask (still bit-identical,
    # just not shared). The dominant probe cost (``transform.forward`` over all
    # train rows) is unchanged here; moving it onto the smaller screen sample
    # is NOT bit-identical (different rows -> different T_std/y_std ratio vs the
    # 0.001 gate) and is left as a measured FUTURE (perf-measure-first).
    _y_train_valid = y_train[valid]
    _base_train_valid = base_train[valid]
    _valid_stale = False
    fitted_params = transform.fit(_y_train_valid, _base_train_valid)
    # T15 (2026-06-10): fitted-params-aware domain refinement. The pre-fit
    # ``domain_check`` above cannot see learned params (log_y's ``offset``,
    # centered_ratio's shift ``c`` + eps-floor), so it lets rows through that
    # are out of the TRUE fitted domain -- e.g. log_y rows with
    # ``y + offset <= 0`` produce NaN under ``forward(log)``. Re-evaluate the
    # valid mask now that params exist and drop the newly-invalid rows BEFORE
    # the residual-std probe / screening forward, so those NaN-T rows never
    # bias the MI gain (and ``n_train_rows`` reflects the real domain).
    _dcf = getattr(transform, "domain_check_fitted", None)
    if _dcf is not None and isinstance(fitted_params, dict):
        valid_fitted = np.asarray(_dcf(y_train, base_train, fitted_params), dtype=bool)
        if valid_fitted.shape == valid.shape and not bool(valid_fitted[valid].all()):
            valid = valid & valid_fitted
            _valid_stale = True  # P13: the cached gather no longer matches valid.
            valid_frac = float(valid.mean()) if valid.size else 0.0
            if valid_frac < self.config.min_valid_domain_frac:
                _local.append(self._reject(
                    base, transform_name, mi_y_for_base, valid_frac,
                    reason=(
                        f"fitted-domain valid_frac={valid_frac:.3f} "
                        f"< {self.config.min_valid_domain_frac:.3f} "
                        f"(rows out of domain only after fit set params)"
                    ),
                ))
                return _local
            if not valid.any():
                return _local
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
        # P13: reuse the gather hoisted above. Re-take it only when T15
        # narrowed ``valid`` (``_valid_stale``); otherwise the cached arrays
        # already hold exactly ``y_train[valid]`` / ``base_train[valid]``.
        if _valid_stale:
            _y_train_valid = y_train[valid]
            _base_train_valid = base_train[valid]
        _y_train_valid = _y_train_valid.astype(np.float64)
        _base_train_valid = _base_train_valid.astype(np.float64)
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
    # T15: apply the same fitted-domain refinement to the screening mask so
    # ``t_screen`` (mi_t) and ``y_screen[valid_screen]`` (mi_y_compare) score
    # the SAME row population -- otherwise mi_gain compares MI over different
    # rows (mi_t excludes NaN-T rows inside the binner, mi_y_compare keeps
    # them) and the gate sees an apples-to-oranges delta.
    if _dcf is not None and isinstance(fitted_params, dict):
        valid_screen_fitted = np.asarray(
            _dcf(y_screen, base_screen, fitted_params), dtype=bool,
        )
        if valid_screen_fitted.shape == valid_screen.shape:
            valid_screen = valid_screen & valid_screen_fitted
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
    # x_screen_valid (the full-precision float slice) is consumed ONLY on the
    # non-prebinned MI path (mi_t else, mi_y_compare else, and the bootstrap
    # else); the prebinned path -- the default config (mi_estimator='bin') --
    # uses _x_prebinned slices instead, so this was a dead ~80-200 MB copy per
    # work item. Gate it, and gate the prebinned slice on valid_screen.all().
    x_screen_valid = (
        x_remaining_matrix[valid_screen] if _x_prebinned is None else None
    )
    if _x_prebinned is not None:
        _x_pb_valid = (
            _x_prebinned if bool(valid_screen.all())
            else _x_prebinned[valid_screen]
        )
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
    #
    # P18 (2026-06-11) -- shrunk-domain ``mi_y_compare`` recompute, NOT memoised
    # across work items, deferred. Several transforms that share the SAME base
    # produce the SAME ``valid_screen`` mask (e.g. all the non-domain-shrinking
    # bivariate residuals on one base keep the full screen), so the
    # ``_mi_to_target_prebinned(_x_pb_valid, y_screen[valid_screen])`` call
    # below is recomputed identically per transform. A
    # ``(base, hash(valid_screen.tobytes())) -> mi_y_compare`` memo would save
    # ~0.2-0.5 s/mask. It is deferred because ``eval_one_transform`` runs
    # CONCURRENTLY from a threading pool (``discovery_n_jobs``) over a shared,
    # currently read-only ``base_contexts``; a cross-call memo needs a
    # thread-safe shared store (lock or per-base precompute) AND a measurement
    # to confirm the saving survives the lock overhead. The bit-identical
    # WITHIN-call sharing is already done (``_x_pb_valid`` is computed once and
    # reused for both ``mi_t`` and this ``mi_y_compare``); only the
    # ACROSS-call memo remains, and it cannot be added safely without the
    # shared-state design. Do not stash the memo on ``self`` here -- that races.
    #
    # D21 (2026-06-11) -- correlated-sibling inflation of ``mi_y_compare``,
    # deferred. ``x_remaining_matrix`` excludes only THIS base's column; a
    # ~0.99-correlated sibling of the base (a second lag, a near-duplicate
    # smooth) left in the remaining set inflates ``mi_y_compare`` (it carries
    # almost the same info as the removed base) while contributing little to
    # ``mi_t``, biasing ``mi_gain`` DOWN -- conservative (no leak, never an
    # over-keep) but it can wrongly sink the lag-family bases discovery most
    # wants. The fix masks out per-base high-corr siblings before the MI
    # baseline (gated on the existing ``auto_base_dedup_corr_threshold``),
    # which CHANGES which columns enter ``mi_y_compare`` -> changes the MI
    # numerics -> needs a biz_value measurement, and ties into D10's
    # ``exclude_col`` threading through ``_mi_to_target_prebinned`` (a sibling
    # cross-file contract). Deferred rather than half-applied because dropping
    # columns from ``x_remaining`` here without the calibrated threshold +
    # measurement could silently over-keep weak specs.
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
    #
    # M4 (2026-06-11) -- KNOWN GAP, no family-wise multiplicity control here.
    # ``eval_one_transform`` runs once per (base, transform) work item and the
    # LCB above is a PER-SPEC 95% CI. Across the ~19 candidates a single sweep
    # screens, that per-comparison error rate is NOT corrected for the family
    # of tests, so under the gates that actually filter on it
    # (eps_mi_gain >> -10, mi_gain_bootstrap_n > 0) the false-discovery rate
    # inflates with the candidate count. This is LATENT under the shipped
    # defaults (eps_mi_gain=-10.0 disables the MI prefilter and
    # mi_gain_bootstrap_n=0 disables the CI entirely), so no default user is
    # affected today. The honest fix is family-wise control (Benjamini-Hochberg
    # on the per-spec bootstrap p-values, or a maxT step-down over the joint
    # bootstrap of all candidates) applied at the gate in ``_fit.py`` AFTER all
    # work items return -- selective-inference machinery that needs its own
    # validation + a biz_value showing the FDR is actually controlled. It is
    # deferred rather than half-wired here because a per-spec p-value computed
    # in this function cannot see the family; the correction MUST be a
    # post-collection pass. Do not add a per-spec BH approximation here -- it
    # would be statistically wrong (BH needs the full p-value vector).
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
        _boot_fail_count = 0
        for b in range(bootstrap_n):
            idx_b = boot_rng.integers(0, n_screen, size=n_screen)
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
                    # Non-prebinned path is the only consumer of the float slice.
                    x_boot = x_screen_valid[idx_b]
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
                # Silent NaN on failure shifts the CI toward well-behaved bootstraps; warn on the FIRST failure (any replicate, not just b==0)
                # so operators see when the CI is computed over a reduced bootstrap sample. The `>= bootstrap_n // 2` guard below only
                # protects against extreme under-sampling, not the partial-bias case.
                _boot_fail_count += 1
                if _boot_fail_count == 1:
                    import logging as _logging
                    _logging.getLogger(__name__).warning(
                        "composite_discovery: MI-bootstrap iteration "
                        "failed (%s); per-bootstrap result reported "
                        "as NaN. Bootstrap CI will use surviving "
                        "samples; with sparse failures the LCB is "
                        "biased toward well-behaved bootstraps "
                        "(failures so far: %d).",
                        _e_boot, _boot_fail_count,
                    )
                boot_gains[b] = float("nan")
        boot_finite = boot_gains[np.isfinite(boot_gains)]
        if boot_finite.size >= bootstrap_n // 2:
            mi_gain_lcb = float(np.percentile(boot_finite, 2.5))

    # T21/D13 (2026-06-11) -- KNOWN naming defect for unary specs, deferred.
    # When ``transform.requires_base`` is False (cbrt_y / log_y / yeo_johnson_y
    # / quantile_normal_y / y_quantile_clip), the transform ignores ``base``
    # entirely, yet the spec is still stamped with ``base_column=base`` and
    # named ``y-cbrtY-<base>`` where ``<base>`` is whichever base the dispatch
    # loop happened to pair this unary with first (``_fit.py`` dedups unary
    # transforms to the first base in ``base_candidates`` order). The value is
    # DETERMINISTIC (always the first base) but semantically MISLEADING: the
    # name implies a base dependence that does not exist, and any report label
    # or dedup key derived from it is base-order-coupled. The correct fix is a
    # 2-segment unary name (``y-cbrtY``) + a sentinel ``base_column`` -- but
    # that changes the emitted spec name, which is load-bearing for dedup keys,
    # report headings, the wrapper's base-column extraction, AND v1 suite
    # pickles, so it must be a coordinated cross-file change (name/spec/wrapper/
    # report) with its own back-compat handling, not a local edit here.
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
