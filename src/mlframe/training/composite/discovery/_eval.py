"""Per-transform candidate evaluator lifted out of ``_composite_discovery_fit.fit``.

``eval_one_transform`` was the ``_eval_one_transform`` nested closure inside ``fit``; it is lifted to module level so the parallel-dispatch wrapper can call it without re-capturing the (large) per-base arrays via closure cells. Every variable the closure captured from the enclosing ``fit`` scope is now an explicit parameter: the discovery instance (``self``, for ``self.config`` + ``self._reject``), the per-base context map (``base_contexts``), and the shared ``y_train`` / ``y_screen`` / ``target_col``. The function is pure w.r.t. shared state: it reads ``base_contexts[base]`` (read-only after setup) and returns a fresh list, so it is safe to call concurrently from the threading pool.
"""
from __future__ import annotations

import inspect
import logging
import threading
from typing import Any

import numpy as np

from ..spec import CompositeSpec
from ._eval_stats import bootstrap_gain_p_value
from .screening import (
    _aggregate_mi_per_feature,
    _mi_to_target,
    _mi_to_target_prebinned,
)
from ..transforms import compose_target_name

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


def _fit_accepts_groups(fit_fn) -> bool:
    """True when ``transform.fit`` declares a ``groups`` parameter or ``**kwargs``.

    A local signature gate (mirrors ``estimator._callable_accepts_param`` but
    kept self-contained to avoid the ``estimator -> discovery`` import cycle).
    Used so the per-fold refit only threads ``groups`` into a fit that actually
    accepts it (``requires_groups=True`` transforms); permissive on builtins.
    """
    try:
        sig = inspect.signature(fit_fn)
    except (ValueError, TypeError):
        return True
    params = sig.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return True
    return "groups" in params


def refit_transform_on_fold(
    transform,
    y_fold: np.ndarray,
    base_fold: np.ndarray,
    *,
    groups_fold: np.ndarray | None = None,
    min_valid_rows: int = 2,
) -> tuple[dict[str, Any], np.ndarray] | None:
    """Re-fit a transform's params on ONE CV fold's TRAIN rows only.

    This is the ``_eval``-side contract for **per-fold transform refit**, the
    cure for the in-fold leakage: in :func:`eval_one_transform`
    the transform is fit ONCE on every valid train row (line ~71) and those
    GLOBAL ``fitted_params`` (e.g. ``linear_residual``'s alpha/beta) are then
    reused for every inner tiny-CV fold downstream
    (``_screening_tiny._tiny_cv_rmse_y_scale``). Because the global fit saw the
    rows that later become each fold's HELD-OUT validation set, the recovered
    ``T = forward(y, base, params)`` on the held-out fold is partly explained by
    parameters that already peeked at those very rows -> the held-out RMSE the
    tiny-CV reports is optimistic (the alpha/beta absorbed val-fold structure).

    The honest path is to re-fit the transform params on each fold's TRAIN rows
    ONLY, then ``forward``/``inverse`` the held-out fold with those fold-local
    params. This helper performs exactly that single-fold refit, reusing the
    SAME fit + fitted-domain-refinement logic ``eval_one_transform`` runs on the
    global sample, so the per-fold params are produced identically to how the
    shipped spec's params are produced -- just on a row subset.

    Parameters
    ----------
    transform
        A registry ``Transform`` (reads ``.fit``, ``.domain_check``,
        ``.domain_check_fitted``).
    y_fold, base_fold
        Raw (un-transformed) target / base columns for THIS fold's train rows.
        These are the exact arrays the caller must expose per fold -- the raw
        ``y`` and ``base``, NOT the globally-computed ``T``.
    groups_fold
        Group labels for the fold's train rows (grouped transforms only). Passed
        through to ``transform.fit`` only when the fit signature accepts it.
    min_valid_rows
        Minimum surviving (domain-valid) fold rows required to attempt a refit.
        Below this the fold is too small to re-estimate params reliably and the
        caller should keep the global params for this fold (we return ``None``).

    Returns
    -------
    ``(fold_params, valid_fold_mask)`` on success, where ``fold_params`` is the
    transform's fitted-params dict for THIS fold's train rows and
    ``valid_fold_mask`` is the boolean mask (aligned to ``y_fold``) of rows that
    survived the (fitted-)domain filter; or ``None`` when the fold is degenerate
    (too few valid rows, an empty mask, or a fit that flags
    ``is_degenerate`` / non-dict params) -- in which case the caller falls back
    to the global params so the fold still scores rather than dropping out.

    Notes
    -----
    * **No leakage by construction**: only ``y_fold`` / ``base_fold`` rows enter
      the fit, so a held-out fold scored with these params is honest.
    * **Bit-stable fallback**: returning ``None`` (not raising) lets the caller
      preserve today's global-fit numerics on degenerate folds, so enabling the
      per-fold path never crashes a previously-scoring spec.
    * The caller (``_screening_tiny._one_fold``) still owns the fold split and
      the ``forward``/``inverse`` calls; this helper only produces the params.
    """
    y_fold = np.asarray(y_fold)
    base_fold = np.asarray(base_fold)
    # Pre-fit domain filter (same gate eval_one_transform applies before fit).
    valid = np.asarray(transform.domain_check(y_fold, base_fold), dtype=bool)
    if valid.shape != y_fold.shape:
        # Defensive: a domain_check that returns a mis-shaped mask cannot be
        # trusted to subset rows; signal the caller to keep global params.
        return None
    if int(valid.sum()) < min_valid_rows:
        return None
    y_v = y_fold[valid]
    base_v = base_fold[valid]
    fit_kwargs: dict[str, Any] = {}
    if groups_fold is not None and _fit_accepts_groups(transform.fit):
        g_arr = np.asarray(groups_fold)
        if g_arr.shape[0] == y_fold.shape[0]:
            fit_kwargs["groups"] = g_arr[valid]
    try:
        fold_params = transform.fit(y_v, base_v, **fit_kwargs)
    except Exception as _fit_err:  # noqa: BLE001 -- degenerate fold, keep global
        logger.debug(
            "refit_transform_on_fold: per-fold fit failed (%s); caller should "
            "fall back to global params for this fold.", _fit_err,
        )
        return None
    if not isinstance(fold_params, dict):
        return None
    # A fold whose fit collapses to a near-identity / degenerate function is no
    # better than the global params (and downstream forward on it can NaN); let
    # the caller keep global params rather than score on a degenerate refit.
    if fold_params.get("is_degenerate"):
        return None
    # Fitted-params-aware domain refinement: drop rows that are
    # only out-of-domain once the fold's params exist, so the mask the caller
    # uses to subset the fold matches the params it was fit on.
    _dcf = getattr(transform, "domain_check_fitted", None)
    if _dcf is not None:
        try:
            valid_fitted = np.asarray(
                _dcf(y_fold, base_fold, fold_params), dtype=bool,
            )
        except Exception:  # noqa: BLE001 -- treat as no refinement
            valid_fitted = None
        if valid_fitted is not None and valid_fitted.shape == valid.shape:
            refined = valid & valid_fitted
            if int(refined.sum()) < min_valid_rows:
                return None
            valid = refined
    return fold_params, valid


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
    """
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
    return _eval_one_transform_impl(
        self, base, transform_name, transform,
        base_contexts=base_contexts, y_train=y_train,
        y_screen=y_screen, target_col=target_col,
    )


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
    fitted_params = transform.fit(_y_train_valid, _base_train_valid)
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
    # white noise. Compute residual std on full train sample
    # (cheap: one transform.forward call) and reject when
    # T_std / y_std < 0.001 (T is below 0.1% of y scale -- below
    # typical noise floor for f32 tabular targets).
    try:
        # Reuse the gather hoisted above. Re-take it only when the fitted-domain block
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
    })
    return _local
