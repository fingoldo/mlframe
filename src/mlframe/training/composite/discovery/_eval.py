"""Per-transform candidate evaluator lifted out of ``_composite_discovery_fit.fit``.

``eval_one_transform`` was the ``_eval_one_transform`` nested closure inside ``fit``; it is lifted to module level so the parallel-dispatch wrapper can call it without re-capturing the (large) per-base arrays via closure cells. Every variable the closure captured from the enclosing ``fit`` scope is now an explicit parameter: the discovery instance (``self``, for ``self.config`` + ``self._reject``), the per-base context map (``base_contexts``), and the shared ``y_train`` / ``y_screen`` / ``target_col``. The function is pure w.r.t. shared state: it reads ``base_contexts[base]`` (read-only after setup) and returns a fresh list, so it is safe to call concurrently from the threading pool.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

from ..spec import CompositeSpec
from ._eval_stats import bootstrap_gain_p_value
from ._fold_refit import refit_transform_on_fold  # noqa: F401  (re-exported)
from .screening import (
    _aggregate_mi_per_feature,
    _mi_to_target,
    _mi_to_target_prebinned,
)
from ..transforms import compose_target_name
from mlframe.training.composite.transforms._call_gateway import call_transform

logger = logging.getLogger(__name__)


def build_unary_base_context(
    *,
    full_x_matrix: np.ndarray,
    full_x_prebinned: np.ndarray | None,
    per_feat_y_full: np.ndarray | None,
    y_screen: np.ndarray,
    n_train: int,
    sample_idx: np.ndarray,
    mi_aggregation: str,
    mi_nbins: int,
    mi_n_neighbors: int,
    random_state: int,
    mi_estimator: str,
) -> dict[str, Any] | None:
    """Build the dedicated UNARY (``requires_base=False``) evaluation context.

    Unary transforms ignore the base column entirely, so they are scored ONCE
    against the FULL feature matrix (NO base column dropped -- a unary drops no
    signal-carrying base, unlike a residual whose base IS removed from
    ``x_remaining`` to isolate the transform effect). Computing this context
    once makes a unary spec's ``mi_gain`` invariant to auto-base ranking order
    (the bug: pre-fix a unary was bound to the first base and scored against
    that base's ``x_remaining``, so its gain shifted with irrelevant base
    reordering).

    ``base_train`` / ``base_screen`` are a zeros placeholder of the right length
    -- ``requires_base=False`` transforms never read ``base`` in
    fit/forward/domain_check, so the values are immaterial; a real array (not
    ``None``) keeps the shared :func:`eval_one_transform` body's
    ``base_train[valid]`` gathers from needing a special case. ``mi_y_for_base``
    over the full matrix is the honest baseline: ``MI(T_unary, X)`` vs
    ``MI(y, X)`` on the SAME full feature set.

    Returns the context dict, or ``None`` when ``full_x_matrix`` has zero
    columns (degenerate -- the caller falls back to the per-base path so the
    unary still gets evaluated).
    """
    if full_x_matrix.shape[1] == 0:
        return None
    mi_kwargs: dict[str, Any] = dict(nbins=int(mi_nbins), aggregation=mi_aggregation)
    base_train = np.zeros(n_train, dtype=np.float32)
    base_screen = base_train[sample_idx]
    if full_x_prebinned is not None:
        if per_feat_y_full is not None:
            mi_y = _aggregate_mi_per_feature(per_feat_y_full, mi_aggregation)
        else:
            mi_y = _mi_to_target_prebinned(full_x_prebinned, y_screen, **mi_kwargs)
    else:
        mi_y = _mi_to_target(
            full_x_matrix, y_screen,
            n_neighbors=mi_n_neighbors,
            random_state=random_state,
            estimator=mi_estimator,
            **mi_kwargs,
        )
    return dict(
        base_train=base_train,
        base_screen=base_screen,
        x_remaining_matrix=full_x_matrix,
        _x_prebinned=full_x_prebinned,
        mi_y_for_base=mi_y,
        _mi_kwargs=mi_kwargs,
        # Shared shrunk-domain ``mi_y_compare`` memo (see the per-base context
        # build in ``_fit.py``); unary specs share this single sentinel context.
        _mi_y_compare_memo={},
        _mi_y_compare_memo_lock=threading.Lock(),
        # Per-unary-spec result memo. A ``requires_base=False`` transform's whole
        # evaluation (fit, forward, ``MI(T_unary, X_full)``, ``mi_gain``) is
        # base-independent, so its candidate result depends ONLY on the transform
        # (its fit is deterministic on the fixed full-train rows). Memoise the
        # finished candidate keyed by ``transform_name`` so any second call for
        # the SAME unary -- the per-base fallback when the sentinel context is
        # unavailable, or a re-dispatch -- reuses the bit-identical result rather
        # than recomputing the (per-feature, full-X) MI from scratch. Guarded by a
        # lock because ``eval_one_transform`` runs concurrently from the pool.
        _unary_result_memo={},
        _unary_result_memo_lock=threading.Lock(),
    )


def _boot_mi(x_rows, target, prebinned: bool, mi_kwargs, config) -> float:
    """MI of ``target`` against resampled feature rows, on whichever representation the base uses."""
    if prebinned:
        return _mi_to_target_prebinned(x_rows, target, **mi_kwargs)
    return _mi_to_target(
        x_rows, target,
        n_neighbors=config.mi_n_neighbors,
        random_state=config.random_state,
        estimator=config.mi_estimator,
        **mi_kwargs,
    )


def _bootstrap_mi_y_replicates(bootstrap_n, seed, n_screen, y_valid, x_pb_valid, x_screen_valid, mi_kwargs, config):
    """``(values, failure_messages)`` of ``MI(y, X)`` on each bootstrap draw of a fixed seed.

    Every transform on a base with the same valid-row mask draws exactly these rows (the generator is re-seeded per
    candidate from the same seed), so this is computed once per (base, mask, seed) and shared.
    """
    rng = np.random.default_rng(seed)
    values = np.full(bootstrap_n, np.nan)
    fails: dict = {}
    for b in range(bootstrap_n):
        idx_b = rng.integers(0, n_screen, size=n_screen)
        x_rows = x_pb_valid[idx_b] if x_pb_valid is not None else x_screen_valid[idx_b]
        try:
            values[b] = _boot_mi(x_rows, y_valid[idx_b], x_pb_valid is not None, mi_kwargs, config)
        except Exception as e:
            fails[b] = f"{type(e).__name__}: {e}"
    return values, fails


def _shared_bootstrap_inputs(ctx, valid_screen, bootstrap_n, seed, n_screen, y_valid, x_prebinned, x_screen_valid, mi_kwargs, config):
    """The row-major prebinned block and the ``MI(y, X)`` replicates for this base, mask and seed, built once and shared.

    The prebinned matrix is stored column-major for the per-column MI walk, but the replicate loop gathers ROWS once per
    replicate (102 ms against 14 ms at 100k x 100), so one row-major copy is taken. Every candidate on this base with the
    same mask and seed draws the same replicates, so the copy and the ``MI(y, X)`` vector are shared through the base
    context. Candidates run concurrently: whichever takes the key's lock first computes them, the others wait and reuse.
    """
    key = (hash(valid_screen.tobytes()), bootstrap_n, seed)
    with ctx["_mi_y_compare_memo_lock"]:
        entry = ctx.setdefault("_bootstrap_memo", {}).setdefault(key, {"lock": threading.Lock(), "value": None})
    with entry["lock"]:
        if entry["value"] is None:
            x_pb = np.ascontiguousarray(x_prebinned[valid_screen]) if x_prebinned is not None else None
            entry["value"] = (x_pb, _bootstrap_mi_y_replicates(bootstrap_n, seed, n_screen, y_valid, x_pb, x_screen_valid, mi_kwargs, config))
    return entry["value"]


class _ReplayedBootstrapError(Exception):
    """A bootstrap replicate whose ``MI(y, X)`` failed for an earlier candidate on the same draws; carries its message."""


def _bootstrap_gain_replicates(boot_gains, bootstrap_n, boot_rng, n_screen, t_screen, _y_screen_valid, _x_pb_valid_const, x_screen_valid, _mi_kwargs, config, failures, mi_y_reps=None):
    """Fill ``boot_gains`` with one resampled ``MI(T, X) - MI(y, X)`` per replicate, recording any that fail.

    A replicate that raises leaves NaN and appends its message to ``failures``; the caller decides whether enough
    survived to report a confidence bound. ``mi_y_reps`` is a ``(values, failure_messages)`` pair from an earlier
    candidate on the same base, mask and seed: the draws are the same, so each ``MI(y, X)`` replicate is reused and a
    replicate whose ``MI(y, X)`` failed there fails here with the same message. ``MI(y, X)`` is computed before
    ``MI(T, X)`` so a candidate-specific ``MI(T, X)`` failure never leaves a replicate unrecorded for the next candidate.
    Returns ``(failure_count, (values, failure_messages))`` for the caller's memo.
    """
    _boot_fail_count = 0
    reuse = mi_y_reps is not None
    mi_y_vals = mi_y_reps[0] if reuse else np.full(bootstrap_n, np.nan)
    mi_y_fail: dict = mi_y_reps[1] if reuse else {}

    def _mi(x_rows, target):
        """MI of ``target`` against the resampled feature rows."""
        return _boot_mi(x_rows, target, _x_pb_valid_const is not None, _mi_kwargs, config)

    for b in range(bootstrap_n):
        idx_b = boot_rng.integers(0, n_screen, size=n_screen)
        t_boot = t_screen[idx_b]
        y_boot = _y_screen_valid[idx_b]
        try:
            if _x_pb_valid_const is not None:
                x_rows = _x_pb_valid_const[idx_b]
            else:
                # Non-prebinned path is the only consumer of the float slice.
                assert x_screen_valid is not None
                x_rows = x_screen_valid[idx_b]
            if reuse:
                if b in mi_y_fail:
                    raise _ReplayedBootstrapError(mi_y_fail[b])
                mi_y_b = float(mi_y_vals[b])
            else:
                try:
                    mi_y_b = _mi(x_rows, y_boot)
                except Exception as _e_y:
                    mi_y_fail[b] = f"{type(_e_y).__name__}: {_e_y}"
                    raise
                mi_y_vals[b] = mi_y_b
            mi_t_b = _mi(x_rows, t_boot)
            boot_gains[b] = mi_t_b - mi_y_b
        except Exception as _e_boot:
            # Silent NaN on failure shifts the CI toward well-behaved bootstraps; warn on the FIRST failure (any replicate, not just b==0)
            # so operators see when the CI is computed over a reduced bootstrap sample. The `>= bootstrap_n // 2` guard below only
            # protects against extreme under-sampling, not the partial-bias case.
            _boot_fail_count += 1
            _msg = str(_e_boot) if isinstance(_e_boot, _ReplayedBootstrapError) else f"{type(_e_boot).__name__}: {_e_boot}"
            failures.append(f"replicate {b}: {_msg}")
            if _boot_fail_count == 1:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "composite_discovery: MI-bootstrap iteration "
                    "failed (%s); per-bootstrap result reported "
                    "as NaN. Bootstrap CI will use surviving "
                    "samples; with sparse failures the LCB is "
                    "biased toward well-behaved bootstraps "
                    "(failures so far: %d).",
                    _msg, _boot_fail_count,
                )
            boot_gains[b] = float("nan")
    return _boot_fail_count, (mi_y_vals, mi_y_fail)


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

    Unary (``requires_base=False``) fast path: a unary transform's entire result
    is base-independent (its fit + ``MI(T_unary, X_full)`` never read ``base``),
    so the FIRST evaluation is memoised on its context keyed by ``transform_name``
    and any later call for the same unary returns the bit-identical cached result
    WITHOUT recomputing the (per-feature, full-X) MI. This is what makes a unary's
    ``MI(T_unary, X)`` cost O(1 spec) rather than O(bases) even on the per-base
    fallback path (the normal sentinel routing already dedups via the work-list,
    but the memo also guards the fallback + any re-dispatch and pins the win).

    A grouped (``requires_groups``) transform is rejected here with the reason: the per-(base, transform) screen carries no
    group labels, so its fit raised and took down every composite of the target. Any other evaluation error likewise
    becomes this candidate's rejection (logged at WARNING and in the ledger) instead of aborting the whole fit.
    """
    _mi_y = base_contexts.get(base, {}).get("mi_y_for_base", float("nan"))
    if getattr(transform, "requires_groups", False):
        return [self._reject(base, transform_name, _mi_y, float("nan"), reason=(
            "grouped transform: discovery screens without group labels, so it cannot fit one; "
            "fit it directly with CompositeTargetEstimator(group_column=...)"))]
    if not transform.requires_base:
        _uctx = base_contexts[base]
        _memo = _uctx.get("_unary_result_memo")
        if _memo is not None:
            _memo_lock = _uctx.get("_unary_result_memo_lock")
            with _memo_lock:
                _cached = _memo.get(transform_name)
            if _cached is not None:
                # Return a fresh shallow copy of the cached candidate list so the
                # caller's downstream in-place mutations (e.g. the FDR ``kept`` /
                # ``fdr_dropped`` flags stamped in ``_fit.py``) on one call never
                # leak into another call's view of the same memoised entry.
                return [dict(_c) for _c in _cached]
            _result = _eval_one_transform_impl(
                self, base, transform_name, transform,
                base_contexts=base_contexts, y_train=y_train,
                y_screen=y_screen, target_col=target_col,
            )
            with _memo_lock:
                # First writer wins; a concurrent second compute produced the
                # bit-identical result, so either entry is equivalent.
                _memo.setdefault(transform_name, [dict(_c) for _c in _result])
            return _result
    try:
        return _eval_one_transform_impl(
            self, base, transform_name, transform,
            base_contexts=base_contexts, y_train=y_train,
            y_screen=y_screen, target_col=target_col,
        )
    except Exception as err:  # nosec B110 -- converted into this candidate's rejection, logged and ledgered, never silent
        logger.warning("[CompositeTargetDiscovery] candidate (%s, %s) failed to evaluate (%s: %s); rejected, discovery continues.",
                       base, transform_name, type(err).__name__, err)
        return [self._reject(base, transform_name, _mi_y, float("nan"), reason=f"evaluation failed: {type(err).__name__}: {err}")]


def _eval_one_transform_impl(
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
    """Core per-(base, transform) evaluation body (see :func:`eval_one_transform`)."""
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

    # Gather ``y_train[valid]`` /
    # ``base_train[valid]`` ONCE here and reuse the same arrays for both the
    # transform fit AND (below) the residual-std probe, instead of fancy-index
    # gathering the same rows a second time at the probe. ``transform.fit`` /
    # ``forward`` only READ these arrays (verified across linear/nonlinear
    # transforms), so sharing one gather is bit-identical. ``_valid_stale``
    # tracks whether the fitted-domain block below shrinks ``valid`` -- if it does, the
    # reused gather is re-taken on the narrowed mask (still bit-identical,
    # just not shared). The dominant probe cost (``transform.forward`` over all
    # train rows) is unchanged here; moving it onto the smaller screen sample
    # is NOT bit-identical (different rows -> different T_std/y_std ratio vs the
    # 0.001 gate) and is left as a measured FUTURE (perf-measure-first).
    _y_train_valid = y_train[valid]
    _base_train_valid = base_train[valid]
    _valid_stale = False
    fitted_params = call_transform(transform, "fit", _y_train_valid, _base_train_valid)
    # Fitted-params-aware domain refinement. The pre-fit
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
            _valid_stale = True  # the cached gather no longer matches valid.
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
    # Reject identity / near-identity transforms early.
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
    # linres_robust dedup. When the MAD-trim step in
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
    # Upper-bound degeneracy check. The pre-fix
    # ``is_degenerate`` flag in transform.fit only catches the
    # LOWER bound (transform explains <5% of y variance -- T ~= y).
    # The OPPOSITE pathology also exists: transform absorbs SO
    # much of y that the residual T is at or below the noise
    # floor (observed in prod on a logr spec: y_std=644,
    # T_std=0.001 -- ratio 644000:1). Even a tiny fitting error
    # on T compounds via inverse_transform into significant
    # y-scale error, AND downstream models train on essentially
    # white noise. Reject when T_std / y_std < 0.001 (T is below
    # 0.1% of y scale -- below typical noise floor for f32 tabular
    # targets).
    #
    # Probed on the SCREEN sample (``y_screen``/``base_screen``, already a random
    # subset of train), not the full train set -- moving off full-train was
    # measured NOT bit-identical (different rows -> a slightly different T_std/
    # y_std ratio) but a 600-trial sweep (10 true ratios from 5x below to 50x
    # above the 0.001 threshold, 20 seeds, 3 screen-sample sizes 5k/20k/50k)
    # found the accept/reject DECISION flips only when the true ratio sits
    # exactly ON the 0.001 boundary (an inherently unstable case where even two
    # full-train measurements taken a seed apart would disagree) -- zero flips
    # away from that knife-edge. Saves a full ``transform.forward`` pass over
    # ALL train rows on every (base, transform) candidate; the screen sample is
    # already gathered for the MI-gain scoring right below this block.
    try:
        _valid_screen_probe = np.asarray(transform.domain_check(y_screen, base_screen), dtype=bool)
        # Mirror the train-side fitted-domain refinement (see ``_dcf`` above): the pre-fit
        # ``domain_check`` cannot see learned params, so a transform whose fitted-domain hook
        # narrows further (not just NaN-producing rows -- any refined validity rule) must have
        # that SAME narrowing applied to the screen sample the probe reads, or the T_std/y_std
        # ratio is computed over rows the fit itself does not consider valid.
        if _dcf is not None and isinstance(fitted_params, dict):
            _valid_screen_fitted = np.asarray(_dcf(y_screen, base_screen, fitted_params), dtype=bool)
            if _valid_screen_fitted.shape == _valid_screen_probe.shape:
                _valid_screen_probe = _valid_screen_probe & _valid_screen_fitted
        _y_screen_valid = y_screen[_valid_screen_probe].astype(np.float64)
        _base_screen_valid = base_screen[_valid_screen_probe].astype(np.float64)
        _t_screen_full = call_transform(transform, "forward", _y_screen_valid, _base_screen_valid, fitted_params)
        _t_train_finite = _t_screen_full[np.isfinite(_t_screen_full)]
        _y_train_finite = _y_screen_valid[np.isfinite(_y_screen_valid)]
        if _t_train_finite.size > 1 and _y_train_finite.size > 1:
            _y_std = float(np.std(_y_train_finite))
            _t_std = float(np.std(_t_train_finite))
            _residual_ratio = _t_std / _y_std if _y_std > 0 else 1.0
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
    # Apply the same fitted-domain refinement to the screening mask so
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
    t_screen = call_transform(transform, "forward", y_screen[valid_screen], base_screen[valid_screen], fitted_params)

    # MI(T, X_remaining) on the same valid rows -- comparable
    # to mi_y_for_base computed on the same x_remaining.
    # x_screen_valid (the full-precision float slice) is consumed ONLY on the
    # non-prebinned MI path (mi_t else, mi_y_compare else, and the bootstrap
    # else); the prebinned path -- the default config (mi_estimator='bin') --
    # uses _x_prebinned slices instead, so this was a dead ~80-200 MB copy per
    # work item. Gate it, and gate the prebinned slice on valid_screen.all().
    x_screen_valid = x_remaining_matrix[valid_screen] if _x_prebinned is None else None
    if _x_prebinned is not None:
        _x_pb_valid = _x_prebinned if bool(valid_screen.all()) else _x_prebinned[valid_screen]
        mi_t = _mi_to_target_prebinned(
            _x_pb_valid, t_screen, **_mi_kwargs,
        )
    else:
        assert x_screen_valid is not None  # _x_prebinned is None in this branch, so x_screen_valid was built above
        mi_t = _mi_to_target(
            x_screen_valid, t_screen,
            n_neighbors=self.config.mi_n_neighbors,
            random_state=self.config.random_state,
            estimator=self.config.mi_estimator,
            **_mi_kwargs,
        )
    # When the screening sample shrunk after domain filtering (logratio with
    # negative rows in train), the mi_y baseline for THIS base must also be
    # recomputed on the same valid_screen subset to keep the comparison fair.
    # Many transforms that share the SAME base produce the SAME valid_screen mask
    # (every non-domain-shrinking bivariate residual on one base keeps the full
    # screen), so this baseline MI is otherwise recomputed identically per
    # transform. Memoise it on ``hash(valid_screen.tobytes())`` within the base
    # context so N transforms on one base compute it ONCE, bit-identical -- the
    # cached value is the exact scalar the recompute would return. The memo lives
    # on the base context (not ``self``), guarded by a per-base lock because
    # ``eval_one_transform`` runs concurrently from the discovery threading pool.
    if valid_screen.sum() < y_screen.size:
        _memo = _ctx.get("_mi_y_compare_memo")
        _memo_lock = _ctx.get("_mi_y_compare_memo_lock")
        _memo_key = hash(valid_screen.tobytes()) if _memo is not None else None
        mi_y_compare = None
        if _memo is not None:
            with _memo_lock:
                mi_y_compare = _memo.get(_memo_key)
        if mi_y_compare is None:
            if _x_prebinned is not None:
                mi_y_compare = _mi_to_target_prebinned(
                    _x_pb_valid, y_screen[valid_screen], **_mi_kwargs,
                )
            else:
                assert x_screen_valid is not None  # _x_prebinned is None in this branch, so x_screen_valid was built above
                mi_y_compare = _mi_to_target(
                    x_screen_valid, y_screen[valid_screen],
                    n_neighbors=self.config.mi_n_neighbors,
                    random_state=self.config.random_state,
                    estimator=self.config.mi_estimator,
                    **_mi_kwargs,
                )
            if _memo is not None:
                with _memo_lock:
                    _memo[_memo_key] = mi_y_compare
    else:
        mi_y_compare = mi_y_for_base
    mi_gain = mi_t - mi_y_compare

    # Bootstrap CI on mi_gain. The point-estimate has a noise floor that scales
    # with screening-sample size and y-tail heaviness; the absolute eps_mi_gain
    # threshold misses this. Bootstrap produces a 95% CI; the gate compares
    # against the LOWER CI bound (LCB), not the point estimate. Spec is rejected
    # if LCB <= eps_mi_gain. The same bootstrap replicates also feed a one-sided
    # p-value for H0 ``mi_gain <= 0`` (``bootstrap_p_value`` in the returned
    # entry), which ``_fit.py`` collects across the whole candidate family and
    # corrects with Benjamini-Hochberg FDR control: a per-spec CI controls only
    # its OWN error rate, so testing dozens of specs in one sweep inflates the
    # chance that a noise spec spuriously "beats baseline". The family-wise
    # correction MUST be a post-collection pass (BH needs the full p-value
    # vector), so it lives at the gate in ``_fit.py``, not here.
    bootstrap_n = int(getattr(
        self.config, "mi_gain_bootstrap_n", 0,
    ))
    mi_gain_lcb = mi_gain  # default: point estimate.
    bootstrap_p_value = float("nan")  # NaN until bootstrap replicates exist.
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
        _x_pb_valid_const, _mi_y_reps = _shared_bootstrap_inputs(
            _ctx, valid_screen, bootstrap_n, int(getattr(self.config, "mi_gain_bootstrap_random_state", 12345)),
            n_screen, _y_screen_valid, _x_prebinned, x_screen_valid, _mi_kwargs, self.config,
        )
        failures: list = []  # per-replicate failure messages, surfaced in the returned entry below
        _boot_fail_count, _ = _bootstrap_gain_replicates(
            boot_gains, bootstrap_n, boot_rng, n_screen, t_screen, _y_screen_valid,
            _x_pb_valid_const, x_screen_valid, _mi_kwargs, self.config, failures, mi_y_reps=_mi_y_reps,
        )
        boot_finite = boot_gains[np.isfinite(boot_gains)]
        # `boot_finite.size >= bootstrap_n // 2` is trivially true (0 >= 0) whenever
        # bootstrap_n <= 1 and that lone replicate failed -- np.percentile on an empty
        # array raises an uncaught IndexError, crashing the whole fit() call instead of
        # leaving mi_gain_lcb at its no-CI default. Require a non-empty array explicitly.
        if boot_finite.size > 0 and boot_finite.size >= bootstrap_n // 2:
            mi_gain_lcb = float(np.percentile(boot_finite, 2.5))
        # One-sided bootstrap p-value for H0 ``mi_gain <= 0`` from the same
        # replicates, fed to the family-wise FDR correction in ``_fit.py``. Use
        # ALL finite replicates (not gated on the >= n/2 floor the LCB uses) so a
        # sparsely-failing bootstrap still yields a usable, conservative p-value.
        bootstrap_p_value = bootstrap_gain_p_value(boot_gains)

    # Unary specs are base-free. When
    # ``transform.requires_base`` is False (cbrt_y / log_y / yeo_johnson_y /
    # quantile_normal_y / y_quantile_clip), the transform ignores ``base``
    # entirely. ``_fit.py`` routes these through the dedicated full-X sentinel
    # context (``base`` == ``""``), so their ``mi_gain`` no longer depends on
    # auto-base ranking order. The spec must NOT claim a base dependence: we
    # stamp an empty ``base_column`` and the base-free 2-segment name
    # ``y-cbrtY`` (``compose_target_name`` renders the 2-segment form when the
    # base is empty). Keying off ``transform.requires_base`` -- not the incoming
    # ``base`` string -- makes this authoritative even if a caller passes a
    # real base for a unary (the fallback path in ``_fit.py`` when the sentinel
    # context is unavailable). ``CompositeTargetEstimator`` already tolerates an
    # empty ``base_column`` for unary specs (it skips base extraction when
    # ``requires_base`` is False), and ``is_composite_target_name`` recognises
    # the 2-segment unary form so downstream metric labels stay MTRESID.
    if not transform.requires_base:
        _spec_base_column = ""
        _spec_name = compose_target_name(target_col, transform_name, "")
    else:
        _spec_base_column = base
        _spec_name = compose_target_name(target_col, transform_name, base)
    spec = CompositeSpec(
        name=_spec_name,
        target_col=target_col,
        transform_name=transform_name,
        base_column=_spec_base_column,
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
        "bootstrap_p_value": float(bootstrap_p_value),
        "bootstrap_failure_count": int(_boot_fail_count) if bootstrap_n > 0 else 0,
        "failures": failures if bootstrap_n > 0 else [],
    })
    return _local
