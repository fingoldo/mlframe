"""Sibling of ``_fit_impl_core.py`` (part of the sub-split that brings the parent below
the project's 1k-LOC module-size gate).

Holds ``_fe_stage_cascade_early_a``: hybrid orthogonal-poly + hinge/tri-product basis FE (Layers 23/56), generic MI-greedy FE (Layer 26), CMI-greedy FE (Layer 60). Every FE family stage here reads the (possibly
already-augmented) ``X`` and appends its own winning engineered columns via
``fe_append_columns``/``fe_extract_columns`` -- mirrors the ``_hybrid_orth_family_variants``
siblings' own ``X``-in-``X``-out contract, confirmed the same way (grepping every
``X = fe_append_columns(X, ...)`` reassignment in range).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ._helpers import _dispatch_default_scorer, fe_decide_on_subsample
from .._fe_frame_ops import fe_append_columns, fe_extract_columns, fe_is_numeric_col

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")


def _fe_stage_cascade_early_a(
    self, *, X, y, verbose, fe_max_steps, _y_np, _fe_family_on, _fe_budget_ok,
    _hybrid_orth_pre_recipes, _mi_greedy_pre_recipes,
):
    """Run the hybrid orthogonal-poly + hinge/tri-product basis FE (Layers 23/56), generic MI-greedy FE (Layer 26), CMI-greedy FE (Layer 60) FE family stage(s).

    See the module docstring for the full section this carves out. ``_hybrid_orth_pre_recipes`` /
    ``_mi_greedy_pre_recipes`` are caller-owned dicts mutated in place (never reassigned here, so
    no return needed for them -- confirmed via a systematic reassignment-vs-mutation check, since
    a reassignment inside this function would NOT propagate back like the mutation does).

    Returns ``(X, _raw_input_cols_pre_fe, _hinge_deferred_values, _hinge_deferred_recipes)``:
    unlike the two dicts above, these three ARE reassigned (not just mutated) within this function,
    so they must be threaded back out explicitly.
    """
    # 2026-05-31 Layer 23 — hybrid orthogonal-polynomial + MI-greedy FE.
    # When ``fe_hybrid_orth_enable=True``, generate basis_n(z) columns for each
    # numeric input column and MI-rank against y; append the top-K winners
    # before screening. The hybrid pipeline lives in ``_orthogonal_univariate_fe``
    # and returns EngineeredRecipe objects so transform() can replay each
    # appended column without re-running the MI ranking (deterministic in X,
    # never references y at replay time).
    #
    # The injection happens BEFORE feature_names_in_ is set so the engineered
    # columns are NOT recorded as raw input features; instead they're
    # pre-registered in ``engineered_recipes`` dict (the same dict the FE-step
    # would populate) and the end-of-fit remap routes them through
    # ``self._engineered_recipes_`` automatically.
    self.hybrid_orth_features_ = []
    # Every column the hybrid-orth family APPENDED to the candidate pool, whether or not it later survived
    # selection. ``hybrid_orth_features_`` is intersected with ``support_`` (survivor-only), so when a sibling
    # FE family emits an equivalent column and wins the greedy, the survivor roster goes empty and there is no
    # way left to tell "the stage never fired" from "it fired and lost". This roster answers that directly.
    self.hybrid_orth_candidates_ = []
    # ADAPTIVE-FREQUENCY Fourier: names of the held-out-validated
    # adaptive sin/cos columns the extra-basis stage emitted. Used by the
    # support-finalisation ADAPTIVE-PROTECTION block to re-add any the MRMR
    # screen dropped. Always present (empty when no adaptive freq detected) so
    # transform / pickle / clone never trip on a missing attribute.
    self._adaptive_fourier_features_ = []
    # HINGE / change-point: names of the held-out-tau-validated
    # hinge legs the change-point stage emitted. Used by the support-
    # finalisation HINGE-PROTECTION block to re-add any the MRMR screen dropped
    # (a single relu leg is MONOTONE -> MI-INVARIANT by the DPI, so the greedy
    # MI screen drops it as redundant with raw x exactly as it drops the adaptive
    # Fourier legs - its value is downstream linear usability, not MI). Always
    # present (empty when hinge off / no kink) so transform / pickle / clone
    # never trip on a missing attribute.
    self._hinge_features_ = []
    # SUFFICIENT-SUMMARY EARLY-STOP verdict. The fitted-attribute mirror of
    # the last sufficient-summary check in the greedy FE loop (a SufficientSummaryVerdict,
    # or None when the early-stop never ran / was disabled). Surfaced so callers can inspect
    # WHY the FE search stopped (residual fraction, max raw MI, maxT floor). Always present
    # so transform / pickle / clone never trip on a missing attribute.
    self.sufficient_summary_ = None
    # Count of FE operator-search iterations actually executed (``_run_fe_step`` calls). The
    # sufficient-summary early-stop reduces this by skipping provably-pointless steps; the
    # biz_value test asserts on it as a DETERMINISTIC work-saved proxy (timing on a contended
    # box is jittery). Always present for transform / pickle / clone.
    self._fe_steps_executed_ = 0
    # PER-GATE FE REJECTION LEDGER (additive): reset the per-fit raw-record list HERE, before
    # ANY FE stage runs (recipe-FE families at L33/L34/L37/L38/L104 + cluster-basis all record
    # via their reject_sink BEFORE the pair-search loop). A later reset would clobber those
    # families' unified-gate abs-MAD floor kills; fe_rejection_ledger_ is built from it at fit-end.
    self._fe_rejection_records_ = []
    # Deferred hinge-leg buffer: the hinge stage detects + held-out-validates the
    # legs early (it needs the raw source columns before pair-FE rewrites them) but
    # DEFERS materialising them into the candidate matrix until support finalisation,
    # so the legs never perturb pair-composite recovery. {name: float64 values} and
    # {name: EngineeredRecipe}. Empty when the operator is off / detects nothing.
    _hinge_deferred_values: dict = {}
    _hinge_deferred_recipes: dict = {}
    # Format-agnostic FE seam primitives. CLOSED-FORM families route their DECISION through fe_decide_on_subsample with the
    # NATIVE frame (subsample gather is a small native copy, winners replay on native columns), so a 100+ GB polars frame is
    # never whole-copied. The few OOF / cross-row families that need the full frame gate their pandas materialisation on
    # fe_polars_exceeds (~2 GB, CLAUDE.md eager-conversion rule) and skip above it. Engineered columns append via fe_append_columns.
    # Snapshot the raw input columns BEFORE any FE stage appends engineered
    # intermediates. The cat_pair / cat_triple auto-detect paths restrict their
    # candidate members to this set so a cross is never built on an engineered
    # column (which cannot be replayed at transform time -> KeyError).
    _raw_input_cols_pre_fe = list(X.columns) if hasattr(X, "columns") else []
    # 2026-06-02 UNIVARIATE-BASIS FE — DEFAULT ON (closes the univariate-
    # nonlinearity gap). The pair-FE path (always on) recovers pair interactions
    # (a*b, a/b, |a-b|) but CANNOT express a single-variable nonlinearity (no
    # pairing makes a clean a**2 / a**3 / |a| out of one column); on a symmetric
    # domain raw ``a`` is uninformative about ``a**2`` (corr ~0), so a univariate
    # quadratic signal was silently MISSED (measured: a**2 corr 0.016, zero
    # engineered features). The orthogonal-basis univariate stage (``a__T2`` ~
    # a**2 etc.) closes that - ``fe_univariate_basis_enable`` (default True)
    # runs JUST the univariate basis FE, uplift-gated via ``min_uplift`` in
    # ``hybrid_orth_mi_fe_with_recipes`` so it is near-no-op when there is no
    # univariate nonlinearity, independent of the heavier pair-CROSS-basis stage
    # which stays behind ``fe_hybrid_orth_enable``. Recovery pinned in
    # ``test_biz_value_mrmr_univariate_basis_fe.py``.
    # fe_max_steps==0 is the documented "no FE at all" contract (see e.g. test_group_aware_mi_mrmr.py's
    # fe_max_steps=0 fixtures): both default-ON families must not fire just because the user never
    # explicitly touched their own enable flag - gate on fe_max_steps>0 too, matching the analogous
    # discrete-structural-operators precedent above (which DOES allow fe_max_steps=0 firing, but only
    # for an operator the caller explicitly opted into via its own flag - neither family here has that
    # explicit-opt-in carve-out, so fe_max_steps=0 disables both unconditionally).
    _hybrid_on = _fe_family_on("fe_hybrid_orth_enable", False) and fe_max_steps > 0
    _univ_basis_on = _fe_family_on("fe_univariate_basis_enable", True) and fe_max_steps > 0
    if (_hybrid_on or _univ_basis_on) and _fe_budget_ok():
        # Polars frames: skip with a warning - hybrid FE pipeline operates on
        # pandas. Native polars support would require a separate code path;
        # not in Layer 23 MVP scope.
        # Format-agnostic since the matrix-native FE seam (see triplet stage): skip-guard removed, runs on polars/pandas.
        try:
            from .._orthogonal_univariate_fe import (
                hybrid_orth_mi_fe_with_recipes,
                hybrid_orth_mi_pair_fe_with_recipes,
            )

            _y_for_hybrid = _y_np
            # Hybrid MI scoring expects discrete y. Two cases:
            #   (a) Float-encoded discrete labels (0.0/1.0) - safe to cast to int64.
            #   (b) Continuous regression target - truncating to int destroys the
            #       signal (e.g. y in [-2.5, 3.1] all collapses to {-2,-1,0,1,2,3},
            #       6 quasi-balanced bins, MI to any continuous predictor ~0).
            #       Quantile-bin instead so MI scoring sees a meaningful discrete y.
            if _y_for_hybrid.dtype.kind in "fc":
                _n_unique = int(np.unique(_y_for_hybrid).size)
                if _n_unique <= 32:
                    _y_for_hybrid = _y_for_hybrid.astype(np.int64)
                else:
                    try:
                        _y_for_hybrid = pd.qcut(
                            _y_for_hybrid, q=10, labels=False, duplicates="drop",
                        ).astype(np.int64)
                    except Exception as exc:
                        logger.debug("mrmr: qcut-based y discretisation failed (heavy ties/NaN); falling back to int-cast: %r", exc, exc_info=True)
                        # qcut can fail when y has heavy ties or NaN. Fall back to
                        # int-cast so the pipeline still runs (signal may degrade
                        # but does not crash the fit).
                        _y_for_hybrid = _y_for_hybrid.astype(np.int64)
            _h_degrees = tuple(int(d) for d in self.fe_hybrid_orth_degrees)
            _h_basis = str(self.fe_hybrid_orth_basis)
            _h_top_k = int(self.fe_hybrid_orth_top_k)
            # The pair-CROSS-basis stage is heavier and only runs under the
            # explicit ``fe_hybrid_orth_enable`` opt-in; the default-on
            # univariate-basis path (``fe_univariate_basis_enable`` only) is
            # univariate-only so it stays cheap + near-no-op (uplift-gated).
            _h_pair_enable = bool(self.fe_hybrid_orth_pair_enable) and _hybrid_on
            _h_pair_max_degree = int(self.fe_hybrid_orth_pair_max_degree)
            # Restrict the source pool to numeric columns the caller passed
            # via factors_names_to_use (when set); otherwise the hybrid
            # pipeline auto-routes to all numeric columns of X.
            _h_cols = None
            if getattr(self, "factors_names_to_use", None):
                _h_cols = [c for c in self.factors_names_to_use if c in X.columns]
            _X_before_hybrid_cols = list(X.columns)
            # 2026-06-01 Layer 85 — default-scorer routing for the L21
            # univariate basis-selection stage. Non-"plug_in" values
            # route the univariate dispatch through one of the alternate
            # scorers (CMIM, JMIM, KSG, copula, dCor, HSIC, TC, lasso,
            # elasticnet, auto, ensemble, meta). Recipes still emit as
            # ``orth_univariate``; only the SELECTION differs. The pair
            # stage (L22) is skipped under non-default routing because
            # the alternate scorers operate on univariate columns only.
            # "plug_in" preserves the master-branch byte-identical
            # behaviour: pair stage runs IFF ``pair_enable=True``.
            _default_scorer = str(getattr(
                self, "fe_hybrid_orth_default_scorer", "plug_in",
            ))
            if _default_scorer == "plug_in":
                if _h_pair_enable:
                    # Decide on the shared FE subsample; winners replayed at full n.
                    X_h, _uni_sc, _cross_sc, _recipes = fe_decide_on_subsample(
                        hybrid_orth_mi_pair_fe_with_recipes,
                        X, _y_for_hybrid,
                        subsample_n=int(getattr(self, "fe_check_pairs_subsample_n", 0) or 0),
                        subsample_seed=int(getattr(self, "random_seed", 0) or 0),
                        shared_subsample_idx=getattr(self, "_fe_shared_subsample_idx", None),
                        cols=_h_cols,
                        degrees=_h_degrees,
                        basis=_h_basis,
                        top_k=_h_top_k,
                        top_pair_count=_h_top_k,
                        pair_max_degree=_h_pair_max_degree,
                    )
                else:
                    # Decide on the shared FE subsample (native gather, no whole-frame copy); winners replay at full n.
                    X_h, _uni_sc, _recipes = fe_decide_on_subsample(
                        hybrid_orth_mi_fe_with_recipes,
                        X, _y_for_hybrid,
                        subsample_n=int(getattr(self, "fe_check_pairs_subsample_n", 0) or 0),
                        subsample_seed=int(getattr(self, "random_seed", 0) or 0),
                        shared_subsample_idx=getattr(self, "_fe_shared_subsample_idx", None),
                        cols=_h_cols,
                        degrees=_h_degrees,
                        basis=_h_basis,
                        top_k=_h_top_k,
                    )
            else:
                def _default_scorer_run(_Xs, _ys, **_kw):
                    """Adapt ``_default_scorer`` to the ``fe_decide_on_subsample`` callable signature (subsample-scorer callback)."""
                    return _dispatch_default_scorer(_default_scorer, X=_Xs, y=_ys, **_kw)
                X_h, _uni_sc, _recipes = fe_decide_on_subsample(
                    _default_scorer_run,
                    X, _y_for_hybrid,
                    subsample_n=int(getattr(self, "fe_check_pairs_subsample_n", 0) or 0),
                    subsample_seed=int(getattr(self, "random_seed", 0) or 0),
                    shared_subsample_idx=getattr(self, "_fe_shared_subsample_idx", None),
                    cols=_h_cols,
                    degrees=_h_degrees,
                    basis=_h_basis,
                    top_k=_h_top_k,
                )
            # Identify appended columns vs the pre-hybrid X.
            _appended = [c for c in X_h.columns if c not in _X_before_hybrid_cols]
            if _appended:
                X = fe_append_columns(X, fe_extract_columns(X_h, _appended))
                self.hybrid_orth_features_ = list(_appended)
                for _r in _recipes:
                    _hybrid_orth_pre_recipes[_r.name] = _r
                if verbose:
                    logger.info(
                        "MRMR.fit hybrid_orth: appended %d engineered " "column(s) (univariate + pair): %s",
                        len(_appended),
                        _appended[:8],
                    )
        except Exception as _h_exc:
            logger.warning(
                "MRMR.fit hybrid_orth FE raised %s: %s; continuing " "without hybrid-FE columns.",
                type(_h_exc).__name__,
                _h_exc,
            )
        # 2026-05-31 Layer 32 — extra-basis (B-spline / Fourier) FE stage.
        # Runs only when the master hybrid switch is on AND the user
        # opted in via a non-empty ``fe_hybrid_orth_extra_bases`` tuple.
        # Complementary to the polynomial path: spline catches threshold
        # rules, Fourier catches periodic patterns. Recipes are
        # closed-form (no y), replay safe.
        _extra_bases_cfg = tuple(getattr(self, "fe_hybrid_orth_extra_bases", ()) or ())
        # Defensive guard: the polynomial-stage ``try:`` may have raised
        # before defining ``_y_for_hybrid`` / ``_h_top_k``. Bind safe
        # defaults so the extra-basis stage can still run.
        try:
            _y_for_extra = _y_for_hybrid
        except NameError:
            _y_for_extra = _y_np
            if _y_for_extra.dtype.kind in "fc":
                if int(np.unique(_y_for_extra).size) <= 32:
                    _y_for_extra = _y_for_extra.astype(np.int64)
                else:
                    try:
                        _y_for_extra = pd.qcut(
                            _y_for_extra, q=10, labels=False, duplicates="drop",
                        ).astype(np.int64)
                    except Exception as exc:
                        logger.debug("mrmr: y densification failed for the hybrid-orth extra-basis FE seed pool; falling back to truncating int64 cast: %r", exc, exc_info=True)
                        _y_for_extra = _y_for_extra.astype(np.int64)
        _top_k_for_extra = int(getattr(self, "fe_hybrid_orth_top_k", 5))
        # Effective extra-basis set. Two independent contributors:
        #   * the EXPLICIT ``fe_hybrid_orth_extra_bases`` config, but only under
        #     the heavy ``fe_hybrid_orth_enable`` master switch (legacy gate - a
        #     user who set the config but not the master expected a no-op);
        #   * the DEFAULT-ON Fourier univariate basis (``fe_univariate_fourier_enable``),
        #     which runs in the univariate path WITHOUT the master switch so a
        #     pure oscillatory signal (sin/cos) is recovered by default. The
        #     extra-basis stage is uplift + multiple-comparison gated downstream,
        #     so adding Fourier is near-no-op when there is no oscillation.
        _univ_fourier_on = _fe_family_on("fe_univariate_fourier_enable", True)
        _eff_extra_bases = tuple(_extra_bases_cfg) if (_extra_bases_cfg and _hybrid_on) else ()
        # The default-on Fourier univariate basis is part of the plug-in univariate dispatch. Under an alternate ``fe_hybrid_orth_default_scorer`` (cmim / jmim / ksg / ...) the routing
        # runs ONLY the univariate basis-selection for that scorer (the pair stage is likewise skipped above); the Fourier extra basis is a plug-in-path addition, so adding it under
        # alternate routing would emit columns the routed scorer never selected and diverge from a direct call to that scorer. Gate it to plug-in routing.
        try:
            _extra_basis_scorer_ok = _default_scorer == "plug_in"
        except NameError:
            _extra_basis_scorer_ok = True
        if _univ_fourier_on and _univ_basis_on and _extra_basis_scorer_ok and "fourier" not in _eff_extra_bases:
            _eff_extra_bases = (*_eff_extra_bases, "fourier")
        if _eff_extra_bases:
            try:
                from .._orthogonal_univariate_fe import (
                    hybrid_orth_extra_basis_fe_with_recipes,
                )

                _fourier_freqs = tuple(float(f) for f in getattr(self, "fe_hybrid_orth_fourier_freqs", (1.0, 2.0)))
                _spline_knots = int(getattr(self, "fe_hybrid_orth_spline_knots", 5))
                _fourier_powers = tuple(int(p) for p in getattr(self, "fe_hybrid_orth_fourier_powers", (1, 2)))
                _X_before_extra_cols = list(X.columns)
                # Build the extra basis (Fourier/spline) on RAW columns only -
                # EXCLUDE the already-appended poly-basis columns (``a__T2`` ...).
                # Running Fourier on an engineered column would produce a NESTED
                # recipe (``a__T2__sin1``) whose transform-replay needs ``a__T2``
                # materialised first; the 1-deep replay path can't order that and
                # raises KeyError('a__T2') at transform time. Keeping the source
                # scope to raw columns keeps every extra-basis recipe 1-deep and
                # replayable (and honours factors_names_to_use when set).
                _already_eng_for_extra = set(self.hybrid_orth_features_ or [])
                if getattr(self, "factors_names_to_use", None):
                    _e_cols = [c for c in self.factors_names_to_use if c in X.columns and c not in _already_eng_for_extra]
                else:
                    _e_cols = [c for c in X.columns if c not in _already_eng_for_extra]
                # ADAPTIVE-FREQUENCY Fourier: default ON. The
                # fixed grid {1, 2} misses arbitrary-period oscillations
                # (sin(3.7*x), sin(5.3*x)); the adaptive detector sweeps a
                # coarse z-space grid + local-refines + held-out-validates
                # the dominant frequency per column, n-gated at >= 800 rows
                # (smaller n false-positives a chance frequency). The
                # emitted adaptive sin/cos recipes are tagged adaptive=True
                # and PROTECTED past screening below (a single leg has low
                # marginal MI - phase - so the screen would drop the
                # held-out-validated pair otherwise).
                _fourier_adaptive = bool(getattr(self, "fe_univariate_fourier_adaptive", True))
                _fourier_adaptive_mvc = float(
                    getattr(
                        self,
                        "fe_univariate_fourier_adaptive_min_val_corr",
                        0.15,
                    )
                )
                # ADAPTIVE-CHIRP: second argument-warp path. Runs
                # the same held-out detector on u = sign(z)*z**2 so a growing-
                # frequency chirp (sin(2*pi*f*z**2)) the linear-argument
                # Fourier cannot express is recovered. Emits __qsin/__qcos
                # legs tagged adaptive=True -> captured below + protected past
                # the screen + dedup-exempt exactly like the linear legs.
                _fourier_chirp = bool(getattr(self, "fe_univariate_fourier_chirp", True))
                _fourier_chirp_mvc = float(
                    getattr(
                        self,
                        "fe_univariate_fourier_chirp_min_val_corr",
                        0.15,
                    )
                )
                # Detect frequencies + rank MI on the shared subsample (native gather, no whole-frame copy - the
                # periodogram detector is the dominant orth-FE CPU cost); winners replay at full n via apply_recipe.
                X_e, _e_scores, _e_recipes = fe_decide_on_subsample(
                    hybrid_orth_extra_basis_fe_with_recipes,
                    X, _y_for_extra,
                    subsample_n=int(getattr(self, "fe_check_pairs_subsample_n", 0) or 0),
                    subsample_seed=int(getattr(self, "random_seed", 0) or 0),
                    shared_subsample_idx=getattr(self, "_fe_shared_subsample_idx", None),
                    cols=_e_cols,
                    extra_bases=_eff_extra_bases,
                    fourier_freqs=_fourier_freqs,
                    fourier_powers=_fourier_powers,
                    spline_knots=_spline_knots,
                    top_k=_top_k_for_extra,
                    fourier_adaptive=_fourier_adaptive,
                    fourier_adaptive_min_val_corr=_fourier_adaptive_mvc,
                    fourier_chirp=_fourier_chirp,
                    fourier_chirp_min_val_corr=_fourier_chirp_mvc,
                    max_adaptive_cols=getattr(self, "fe_univariate_fourier_adaptive_max_cols", None),
                )
                _e_appended = [c for c in X_e.columns if c not in _X_before_extra_cols]
                if _e_appended:
                    X = fe_append_columns(X, fe_extract_columns(X_e, _e_appended))
                    # Extend hybrid_orth_features_ with the extra-basis winners
                    # so the downstream remap / transform pipeline handles them
                    # exactly like the polynomial winners.
                    self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_e_appended)
                    for _r in _e_recipes:
                        _hybrid_orth_pre_recipes[_r.name] = _r
                    # Capture ADAPTIVE-tagged Fourier feature names so the
                    # support-finalisation block can re-add any the MRMR
                    # screen dropped (held-out-validated, must survive).
                    _adaptive_names = [
                        _r.name for _r in _e_recipes
                        if getattr(_r, "kind", None) == "orth_fourier"
                        and bool(dict(getattr(_r, "extra", {})).get("adaptive", False))
                        and _r.name in set(_e_appended)
                    ]
                    if _adaptive_names:
                        _prev_adaptive = list(getattr(self, "_adaptive_fourier_features_", None) or [])
                        self._adaptive_fourier_features_ = _prev_adaptive + _adaptive_names
                    if verbose:
                        logger.info(
                            "MRMR.fit hybrid_orth extra-basis: appended %d " "engineered column(s) (spline/fourier): %s",
                            len(_e_appended),
                            _e_appended[:8],
                        )
            except Exception as _e_exc:
                logger.warning(
                    "MRMR.fit hybrid_orth extra-basis FE raised %s: %s; " "continuing without extra-basis columns.",
                    type(_e_exc).__name__,
                    _e_exc,
                )
    # 2026-06-09 — HINGE / piecewise-linear change-point basis stage.
    # Independent opt-in via ``fe_hinge_enable`` (does NOT require
    # ``fe_hybrid_orth_enable``): captures a SLOPE CHANGE at a data-dependent
    # threshold ``y = a*x + b*max(x-tau,0)`` (pricing tiers / dose-response /
    # saturation) that the catalog cannot - ``numeric_rounding`` is piecewise-
    # CONSTANT, the cubic B-spline rounds off a sharp kink at its fixed quantile
    # knots, and orth-poly needs a high degree + rings (Gibbs) around the kink.
    # The breakpoint ``tau`` is detected by scanning inner-quantile cuts for the
    # max 2-segment-SSE drop, HELD-OUT-validated on the ``%3`` stride slice (the
    # 2-segment fit must beat plain linear OOS) so a chance kink / pure noise
    # admits no hinge. Emitted ``relu(x-tau)`` / ``relu(tau-x)`` legs carry a
    # genuinely different LINEAR shape from raw x, so they clear the standard
    # MI-uplift gate (unlike the MI-invariant isotonic / RankGauss). Recipes
    # (``hinge_basis``) store only ``{tau, side}`` - no y - so replay is the
    # pure function ``np.maximum(x-tau,0)``, leak-free. On a monotone target a
    # hinge can be near-collinear with raw x -> the downstream cross-stage
    # Spearman dedup drops it (no duplicate columns survive).
    if _fe_family_on("fe_hinge_enable", False) and _fe_budget_ok():
        # Format-agnostic since the matrix-native FE seam (see triplet stage): skip-guard removed, runs on polars/pandas.
        try:
            from .._hinge_basis_fe import hybrid_hinge_fe_with_recipes
            # The hinge detector + admission are REGRESSION-style (2-segment
            # SSE breakpoint search + held-out incremental linear-R^2 gate),
            # so they want the RAW continuous y - NOT the qcut-to-10-bins
            # coercion the MI-based FE stages use. Quantile-binning a
            # monotone slope-change target (y = a*x + b*relu(x-tau)) collapses
            # the saturating top tier into one bin and DESTROYS the very slope
            # change the hinge detects (measured: qcut y -> 0 breakpoints
            # found; raw y -> tau recovered). Raw class codes work for a
            # discrete classification y too (the linear-fit slope detection is
            # scale/shift invariant). y carries no leak: the recipe stores only
            # {tau, side}, never y.
            _y_for_hinge = _y_np
            _y_for_hinge = np.asarray(_y_for_hinge, dtype=np.float64).ravel()
            # Seed pool restricted to RAW source columns: a hinge built on a
            # prior-stage engineered column would create a recipe whose
            # src_name references an engineered column absent at transform
            # time (KeyError on replay). Honour factors_names_to_use.
            _hinge_already_appended = set(getattr(self, "hybrid_orth_features_", None) or [])
            if getattr(self, "factors_names_to_use", None):
                _hinge_cols = [c for c in self.factors_names_to_use if c in X.columns and c not in _hinge_already_appended and fe_is_numeric_col(X, c)]
            else:
                _hinge_cols = [c for c in X.columns if c not in _hinge_already_appended and fe_is_numeric_col(X, c)]
            _hinge_top_k = int(getattr(self, "fe_hinge_top_k", 5))
            _hinge_max_bp = int(getattr(self, "fe_hinge_max_breakpoints", 2))
            _hinge_emit_ind = bool(getattr(self, "fe_hinge_emit_indicator", False))
            _hinge_mvu = float(getattr(self, "fe_hinge_min_heldout_r2_uplift", 0.02))
            _X_before_hinge_cols = list(X.columns)
            X_h, _h_scores, _h_recipes = hybrid_hinge_fe_with_recipes(
                X, _y_for_hinge,
                cols=_hinge_cols,
                max_breakpoints=_hinge_max_bp,
                emit_indicator=_hinge_emit_ind,
                min_heldout_r2_uplift=_hinge_mvu,
                top_k=_hinge_top_k,
            )
            _h_appended = [c for c in X_h.columns if c not in _X_before_hinge_cols]
            if _h_appended:
                # DEFERRED MATERIALISATION: the hinge legs are a
                # TERMINAL univariate linear-usability stage - they must NOT
                # enter the pair-FE / screening candidate matrix, or (a) the
                # pair search consumes a leg as an operand (replacing a clean
                # raw operand with a hinge-transformed one) and (b) a leg's
                # high marginal MI crowds the genuine pair composites out of
                # selection (measured on y=a**2/b+log(c)*sin(d): the legs on
                # b/d displaced div(sqr(a),abs(b)) / mul(log(c),sin(d))). So
                # we do NOT append the legs to X here; we BUFFER the leg values
                # + recipes and materialise + protect them only at support
                # finalisation (after the FE loop has recovered the composites
                # untouched). This keeps the hidden-champion win (a pure
                # slope-change column with no competing composite still gets
                # its leg) without regressing multi-signal pair recovery.
                _hinge_deferred_values = {c: np.asarray(X_h[c].to_numpy(), dtype=np.float64) for c in _h_appended}
                _hinge_deferred_recipes = {_r.name: _r for _r in _h_recipes if _r.name in set(_h_appended)}
                if verbose:
                    logger.info(
                        "MRMR.fit hinge change-point FE: detected %d held-out-" "validated leg(s) (deferred to support finalisation): %s",
                        len(_h_appended),
                        _h_appended[:8],
                    )
        except Exception as _h_exc:
            logger.warning(
                "MRMR.fit hinge change-point FE raised %s: %s; " "continuing without hinge columns.",
                type(_h_exc).__name__,
                _h_exc,
            )
    # 2026-05-31 Layer 56 — TRI-PRODUCT cross-basis FE stage.
    # Independent opt-in (does NOT require fe_hybrid_orth_enable): captures
    # genuine 3-way interactions like 3-way XOR and price*quantity*count
    # that the pair stage cannot. O(seed_k^3 * deg^3) candidate count is
    # bounded by seed_k=4 default. Recipes (``orth_triplet_cross``) replay
    # from X only, no y, leakage-free by construction.
    # The GBM seeder (#6) opens 3-way generation via order-3-floored explicit triples
    # (``_seeded_triplets_names_``); run the triplet stage for those even when the legacy
    # univariate-seeded triplet path (``fe_hybrid_orth_triplet_enable``) is OFF.
    _gbm_seeded_triplet_names = list(getattr(self, "_seeded_triplets_names_", []) or [])
    from ._hybrid_orth_family_variants import _hybrid_orth_family_variants

    X = _hybrid_orth_family_variants(
        self,
        X=X,
        y=y,
        verbose=verbose,
        _y_np=_y_np,
        _hybrid_orth_pre_recipes=_hybrid_orth_pre_recipes,
        _gbm_seeded_triplet_names=_gbm_seeded_triplet_names,
        _fe_family_on=_fe_family_on,
    )
    # 2026-05-21 revert of Wave 29 P1 polars->pandas coercion. That
    # coercion was added on the premise that downstream ``X[target_name]
    # = y`` mutation assumed pandas and would raise on polars; but the
    # ``_is_polars_input`` branch immediately below (line ~1326) ALREADY
    # handles polars via ``X.with_columns(target_series)``. The Wave 29
    # coercion was a false-positive fix that killed the zero-copy
    # polars promise (test_mrmr_fe_zero_copy_polars regressed -
    # ``pl.DataFrame.to_pandas()`` was called 1x per fit on 100+ GB
    # production frames). Leaving polars frames untouched so the
    # native branch fires.

    # 2026-05-31 Layer 26 — generic MI-greedy FE constructor (sibling to the
    # hybrid orthogonal-polynomial stage above). Same wiring pattern: opt-in
    # via ``fe_mi_greedy_enable=True``, default OFF preserves byte-identical
    # behaviour. The seed pool is the RAW columns of X (NOT the post-hybrid
    # augmented frame) so the two stages can't compound transforms (e.g.
    # ``log(x__He2)``); each constructor explores its own design space and
    # the union of winners is screened by MRMR.
    self.mi_greedy_features_ = []
    if _fe_family_on("fe_mi_greedy_enable", False):
        # Format-agnostic since the matrix-native FE seam (see triplet stage): skip-guard removed, runs on polars/pandas.
        try:
            from .._mi_greedy_fe import greedy_mi_fe_construct_with_recipes
            from .._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection
            # W6: record abs-MAD floor kills in the mi_greedy stage into the
            # FE rejection ledger (pure-record; selection unchanged).
            _mig_step = int(getattr(self, "_fe_steps_executed_", -1))

            def _mig_reject_sink(**_kw):
                """Reject-sink callback for the greedy MI-construction FE stage; records abs-MAD floor kills into the FE rejection ledger (pure-record, does not affect selection)."""
                _record_fe_rejection(self, step=_mig_step, **_kw)

            _y_for_mig = _y_np
            if _y_for_mig.dtype.kind in "fc":
                _n_unique = int(np.unique(_y_for_mig).size)
                if _n_unique <= 32:
                    _y_for_mig = _y_for_mig.astype(np.int64)
                else:
                    try:
                        _y_for_mig = pd.qcut(
                            _y_for_mig, q=10, labels=False, duplicates="drop",
                        ).astype(np.int64)
                    except Exception as exc:
                        logger.debug("mrmr: y densification failed for the MI-greedy FE seed pool; falling back to truncating int64 cast: %r", exc, exc_info=True)
                        _y_for_mig = _y_for_mig.astype(np.int64)
            # Restrict the MI-greedy seed pool to RAW source columns only
            # (i.e. exclude hybrid-orth-appended columns from the prior
            # stage). Compound transforms like ``log(He2(x))`` would
            # create recipes whose ``src_names`` reference an engineered
            # column that does not exist at transform time - replay
            # would KeyError. Each constructor explores its OWN design
            # space; the union of winners is screened by MRMR.
            _hybrid_already_appended = set(getattr(self, "hybrid_orth_features_", None) or [])
            _mig_cols = None
            if getattr(self, "factors_names_to_use", None):
                _mig_cols = [c for c in self.factors_names_to_use if c in X.columns and c not in _hybrid_already_appended]
            else:
                _mig_cols = [c for c in X.columns if c not in _hybrid_already_appended]
            _X_before_mig_cols = list(X.columns)
            X_mg, _mig_scores, _mig_recipes = fe_decide_on_subsample(
                greedy_mi_fe_construct_with_recipes,
                X, _y_for_mig,
                subsample_n=int(getattr(self, "fe_check_pairs_subsample_n", 0) or 0),
                subsample_seed=int(getattr(self, "random_seed", 0) or 0),
                shared_subsample_idx=getattr(self, "_fe_shared_subsample_idx", None),
                cols=_mig_cols,
                seed_cols_count=int(self.fe_mi_greedy_seed_cols_count),
                top_k=int(self.fe_mi_greedy_top_k),
                include_unary=bool(self.fe_mi_greedy_include_unary),
                include_binary=bool(self.fe_mi_greedy_include_binary),
                reject_sink=_mig_reject_sink,
            )
            _mig_appended = [c for c in X_mg.columns if c not in _X_before_mig_cols]
            if _mig_appended:
                X = fe_append_columns(X, fe_extract_columns(X_mg, _mig_appended))
                self.mi_greedy_features_ = list(_mig_appended)
                for _r in _mig_recipes:
                    _mi_greedy_pre_recipes[_r.name] = _r
                if verbose:
                    logger.info(
                        "MRMR.fit mi_greedy: appended %d engineered " "column(s): %s",
                        len(_mig_appended),
                        _mig_appended[:8],
                    )
        except Exception as _mig_exc:
            logger.warning(
                "MRMR.fit mi_greedy FE raised %s: %s; continuing " "without MI-greedy columns.",
                type(_mig_exc).__name__,
                _mig_exc,
            )

    # 2026-05-31 Layer 60 — CMI-greedy FE constructor (sibling to Layer 26).
    # Ranks the same candidate library by ``CMI(candidate; y | support)``
    # instead of marginal ``MI(candidate; y)`` so duplicate-signal transforms
    # (``log_abs(x)`` + ``square(x)`` both monotone in |x|) cannot all be
    # picked: once one is in the support, the others' CMI collapses near
    # zero. Winners are MERGED into ``mi_greedy_features_`` (same recipe
    # kind ``mi_greedy_transform``) so downstream end-of-fit remap and
    # transform-time replay are shared infrastructure. Seed pool excludes
    # both prior hybrid-orth and prior marginal-MI-greedy engineered cols
    # (same rationale: replay must not reference engineered sources).
    if _fe_family_on("fe_mi_greedy_cmi_enable", False):
        # Format-agnostic since the matrix-native FE seam (see triplet stage): skip-guard removed, runs on polars/pandas.
        try:
            from .._mi_greedy_cmi_fe import greedy_cmi_fe_construct_with_recipes

            _y_for_cmi = _y_np
            if _y_for_cmi.dtype.kind in "fc":
                _n_unique_cmi = int(np.unique(_y_for_cmi).size)
                if _n_unique_cmi <= 32:
                    _y_for_cmi = _y_for_cmi.astype(np.int64)
                else:
                    try:
                        _y_for_cmi = pd.qcut(
                            _y_for_cmi, q=10, labels=False, duplicates="drop",
                        ).astype(np.int64)
                    except Exception as exc:
                        logger.debug("mrmr: y densification failed for the CMI-greedy FE seed pool; falling back to truncating int64 cast: %r", exc, exc_info=True)
                        _y_for_cmi = _y_for_cmi.astype(np.int64)
            _eng_already_appended = set(getattr(self, "hybrid_orth_features_", None) or []) | set(self.mi_greedy_features_ or [])
            if getattr(self, "factors_names_to_use", None):
                _cmi_cols = [c for c in self.factors_names_to_use if c in X.columns and c not in _eng_already_appended]
            else:
                _cmi_cols = [c for c in X.columns if c not in _eng_already_appended]
            _X_before_cmi_cols = list(X.columns)
            X_cmi, _cmi_scores, _cmi_recipes = fe_decide_on_subsample(
                greedy_cmi_fe_construct_with_recipes,
                X, _y_for_cmi,
                subsample_n=int(getattr(self, "fe_check_pairs_subsample_n", 0) or 0),
                subsample_seed=int(getattr(self, "random_seed", 0) or 0),
                shared_subsample_idx=getattr(self, "_fe_shared_subsample_idx", None),
                cols=_cmi_cols,
                seed_cols_count=int(self.fe_mi_greedy_cmi_seed_cols_count),
                top_k=int(self.fe_mi_greedy_cmi_top_k),
                include_unary=bool(getattr(self, "fe_mi_greedy_include_unary", True)),
                include_binary=bool(getattr(self, "fe_mi_greedy_include_binary", True)),
                min_cmi_gain=float(self.fe_mi_greedy_cmi_min_gain),
                # Was hardcoded to 0xC011 inside
                # greedy_cmi_fe_construct regardless of random_state, correlating the CMI noise-floor
                # permutation across nominally-independent bootstrap/multi-seed replicates.
                seed=int(getattr(self, "random_seed", 0) or 0),
            )
            _cmi_appended = [c for c in X_cmi.columns if c not in _X_before_cmi_cols]
            if _cmi_appended:
                X = fe_append_columns(X, fe_extract_columns(X_cmi, _cmi_appended))
                # Merge into the existing mi_greedy_features_ list so
                # end-of-fit dedup / remap / pickle treat both stages
                # uniformly. Skip names already present (the two stages
                # share the engineered-column namespace; CMI ones that
                # happen to collide with Layer-26 picks are dropped by
                # name-equality here).
                _existing = set(self.mi_greedy_features_ or [])
                for _c in _cmi_appended:
                    if _c not in _existing:
                        self.mi_greedy_features_.append(_c)
                        _existing.add(_c)
                for _r in _cmi_recipes:
                    if _r.name not in _mi_greedy_pre_recipes:
                        _mi_greedy_pre_recipes[_r.name] = _r
                if verbose:
                    logger.info(
                        "MRMR.fit mi_greedy_cmi: appended %d engineered " "column(s): %s",
                        len(_cmi_appended),
                        _cmi_appended[:8],
                    )
        except Exception as _cmi_exc:
            logger.warning(
                "MRMR.fit mi_greedy_cmi FE raised %s: %s; continuing " "without CMI-greedy columns.",
                type(_cmi_exc).__name__,
                _cmi_exc,
            )

    return X, _raw_input_cols_pre_fe, _hinge_deferred_values, _hinge_deferred_recipes
