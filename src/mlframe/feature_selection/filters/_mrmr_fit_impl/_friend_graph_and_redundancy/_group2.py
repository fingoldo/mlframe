"""Sibling of ``_friend_graph_and_redundancy/__init__.py`` (the sub-split of ``_fit_impl_core.py``'s
friend-graph-and-redundancy post-screen block, itself further split for the 1k-LOC module-size gate).

Holds passes: hinge-deferred-readd, orth-basis-protection. See the package ``__init__.py`` docstring for the full
section this fans out from and the ``(selected_vars, cols, data, nbins)`` threading contract
(all four are BOTH incoming parameters AND part of the return value -- mirrors the parent's own).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _friend_graph_and_redundancy_passes_group2(
    self,
    *,
    X,
    classes_y,
    cols,
    data,
    nbins,
    target_indices,
    y,
    verbose,
    cached_MIs,
    engineered_recipes,
    _eng_continuous_snapshot,
    selected_vars,
    _effective_min_relevance_gain,
    _hinge_deferred_recipes,
    _hinge_deferred_values,
    _hybrid_orth_pre_recipes,
    _miss_ind_pre_recipes,
    _persisted_dcd_state,
    _y_np,
    fe_to_pandas,
    _fe_family_on,
):
    """Run the hinge-deferred-readd, orth-basis-protection pass(es) and return ``(selected_vars, cols, data, nbins)``.
    See the package docstring for the full section this carves out."""
    if _hinge_deferred_values and isinstance(X, pd.DataFrame):
        try:
            from ...mrmr import discretize_array
            _hinge_added_names = []
            _n_cols_before_hinge = len(cols)
            _new_hinge_codes = []
            _new_hinge_nbins = []
            for _hn, _vals in _hinge_deferred_values.items():
                if _hn in X.columns:
                    continue  # already present (defensive)
                _vals = np.asarray(_vals, dtype=np.float64)
                if _vals.shape[0] != data.shape[0]:
                    continue
                _codes = discretize_array(
                    arr=_vals,
                    n_bins=self.quantization_nbins,
                    method=self.quantization_method,
                    dtype=self.quantization_dtype,
                )
                _new_hinge_codes.append(np.asarray(_codes).reshape(-1, 1))
                _new_hinge_nbins.append(int(self.quantization_nbins))
                X[_hn] = _vals
                cols = [*cols, _hn]
                _hinge_added_names.append(_hn)
                _r = _hinge_deferred_recipes.get(_hn)
                if _r is not None:
                    _hybrid_orth_pre_recipes[_hn] = _r
                    engineered_recipes[_hn] = _r
            if _new_hinge_codes:
                data = np.append(
                    data, np.hstack(_new_hinge_codes).astype(data.dtype), axis=1,
                )
                nbins = np.concatenate(
                    [
                        np.asarray(nbins),
                        np.asarray(_new_hinge_nbins, dtype=nbins.dtype),
                    ]
                )
                self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_hinge_added_names)
                self._hinge_features_ = list(getattr(self, "_hinge_features_", None) or []) + list(_hinge_added_names)
                if verbose:
                    logger.info(
                        "MRMR.fit hinge change-point FE: materialised %d deferred " "leg(s) post-loop: %s",
                        len(_hinge_added_names),
                        _hinge_added_names[:8],
                    )
        except Exception as _h_mat_exc:
            logger.warning(
                "MRMR.fit hinge deferred materialisation raised %s: %s; " "continuing without hinge columns.",
                type(_h_mat_exc).__name__,
                _h_mat_exc,
            )

    # HINGE / CHANGE-POINT PROTECTION: re-add the held-out-tau-
    # validated hinge legs the MRMR screen dropped. A single relu leg
    # ``max(x-tau,0)`` is MONOTONE in x, hence MI-INVARIANT by the data-processing
    # inequality, and near-collinear with raw x - so the greedy MI screen drops
    # it as redundant with its raw source, EXACTLY as it drops a single adaptive
    # Fourier leg (low marginal MI) and the clean missingness indicator (tied MI
    # with its raw NaN-bin twin). But the hinge's value is NOT marginal MI: it is
    # the SECOND SLOPE it hands a downstream linear / shallow model
    # (``[1, x, relu(x-tau)]`` fits a two-slope kink ``[1, x]`` cannot). The
    # generating stage already (a) detected the breakpoint, (b) HELD-OUT-validated
    # it (2-segment beats 1-segment OOS R^2 on the %3 slice), and (c) admitted the
    # leg only on its held-out INCREMENTAL linear usability over raw x - so a
    # candidate ``_hinge_features_`` name is a confirmed univariate win. Without
    # this re-add, default-on hinge would GENERATE-then-DROP every leg (wasted
    # compute + the project's MI-vs-linear-usability rule violated, the same fix
    # the adaptive-Fourier protection block applies). TWO-PART SELF-LIMITING GATE
    # (the legs were deferred + just materialised above, so neutral data adds zero
    # cols): (1) the raw SOURCE must have survived the screen (a hinge on a never-
    # selected noise column is left out); (2) the leg must lift a HELD-OUT linear
    # fit over the ALREADY-SELECTED feature set PLUS the source + its degree-2 poly
    # ``[src, src^2]`` - so a leg subsumed by a surviving pair composite (b/d on
    # ``y=a**2/b+log(c)*sin(d)``) or a smooth curve a quadratic already fits
    # (``y=x^2``) adds ~0 and is rejected, while a genuine slope change with no
    # competing composite clears the floor. Runs BEFORE the ``selected_vars_names``
    # remap so the re-added index routes correctly; the recipe is in
    # ``engineered_recipes`` -> transform() replays it byte-for-byte.
    _hinge_feats = getattr(self, "_hinge_features_", None)
    # ``_heldout_incr_over_selected`` (defined below) is also the ORTH-BASIS UNIVARIATE PROTECTION block's
    # sole held-out-uplift probe (see that block further down) - it was originally written for the hinge
    # protection only and the orth-basis block was added later, reusing the closure via ``locals()`` instead
    # of its own copy. That coupling means the closure - and therefore BOTH protections - silently never ran
    # whenever ``fe_hinge_enable=False`` (the common "lightest config" preset: no hinge legs are generated,
    # so ``_hinge_feats`` is empty and this whole block was skipped), even though hybrid-orth univariate basis
    # columns are independently default-on and need the SAME protection. Gate on hybrid_orth_features_ too so
    # the setup runs whenever either protection has candidates to consider; the hinge-specific re-add loop
    # below still runs ONLY when ``_hinge_feats`` is non-empty (see its own ``if _hinge_feats:`` guard).
    if (_hinge_feats or getattr(self, "hybrid_orth_features_", None)) and len(selected_vars):
        _cols_index = {c: i for i, c in enumerate(cols)}
        _sv_set = set(selected_vars)
        _sel_names_now = {cols[i] for i in selected_vars if 0 <= i < len(cols)}
        # SELECTED-SET INCREMENTAL-R^2 GATE (the principled self-limit). A hinge
        # leg is admitted on its held-out linear usability over raw x in the FE
        # stage, but on a MULTI-SIGNAL frame the SELECTED pair composite may
        # already capture the source's structure better than a univariate kink
        # (e.g. on y=a**2/b+log(c)*sin(d) the hinge fires on b / d, but
        # div(sqr(a),abs(b)) / mul(log(c),sin(d)) subsume them). So the protection
        # re-adds a leg ONLY when it lifts a held-out linear fit over the ALREADY-
        # SELECTED feature set - a leg whose value is subsumed by a surviving
        # composite adds ~0 and is dropped (no spurious cols on multi-signal data),
        # while a genuine slope-change leg with no competing composite clears the
        # floor (the hidden-champion win is kept). y is read only here at fit.
        _y_for_hinge_gate = None
        try:
            _yv = _y_np
            _yv = np.asarray(_yv, dtype=np.float64).reshape(-1)
            if _yv.shape[0] == int(data.shape[0]) and np.all(np.isfinite(_yv)):
                _y_for_hinge_gate = _yv
        except Exception as exc:
            logger.debug("mrmr: y coercion for the hinge floor-drop rescue gate failed: %r", exc, exc_info=True)
            _y_for_hinge_gate = None
        # Continuous values of the currently-selected columns (engineered from the
        # snapshot, raw from X) -> the baseline design the leg must beat OOS.
        _sel_value_cols = []
        if _y_for_hinge_gate is not None and isinstance(X, pd.DataFrame):
            for _sn in _sel_names_now:
                _cv = _eng_continuous_snapshot.get(_sn)
                if _cv is None and _sn in X.columns:
                    _cv = X[_sn].to_numpy()
                if _cv is None:
                    continue
                try:
                    _cv = np.asarray(_cv, dtype=np.float64).reshape(-1)
                except (TypeError, ValueError):
                    continue  # a raw categorical/string selected column (e.g. under skip_categorical_encoding) is not a numeric R^2-baseline regressor - exclude it from the linear design
                if _cv.shape[0] == _y_for_hinge_gate.shape[0] and np.all(np.isfinite(_cv)):
                    _sel_value_cols.append(_cv)

        def _heldout_incr_over_selected(_leg_vals, _src_vals=None) -> float:
            """Held-out R^2 gain of adding ``_leg_vals`` to the selected design
            PLUS the source and its degree-2 poly, scored on the %3 stride slice.

            Including ``[src, src^2]`` in the baseline is the SMOOTH-CURVE guard:
            a parabola (y=x^2) is captured by ``src^2`` so a kink adds ~0 over it
            and is rejected (no spurious hinge on a smooth target - matches the
            biz_value complementarity contract); a GENUINE slope change still beats
            ``[src, src^2]`` OOS (a quadratic cannot fit a sharp two-slope kink) so
            the hidden-champion leg is kept."""
            if _y_for_hinge_gate is None:
                return 1.0  # gate disabled -> fall back to the source-survived rule
            leg = np.asarray(_leg_vals, dtype=np.float64).reshape(-1)
            n = leg.shape[0]
            if n != _y_for_hinge_gate.shape[0] or not np.all(np.isfinite(leg)):
                return 0.0
            # Seeded shuffle-then-stride, not a raw
            # positional (idx % 3) == 0 split - the latter is not an honest i.i.d. holdout on
            # time/group/label-sorted input (this module explicitly supports sorted input elsewhere
            # via ``groups`` / the ``temporal_agg`` FE family), which can bias the held-out R^2
            # this gate decides on.
            _hinge_gate_perm = np.random.default_rng(int(getattr(self, "random_seed", 0) or 0)).permutation(n)
            va = np.zeros(n, dtype=bool)
            va[_hinge_gate_perm[: n // 3]] = True
            tr = ~va
            if int(tr.sum()) < 32 or int(va.sum()) < 16:
                return 1.0
            yv = _y_for_hinge_gate[va]
            ss = float(np.sum((yv - yv.mean()) ** 2))
            if ss < 1e-24:
                return 0.0
            base = [np.ones(n), *_sel_value_cols]
            if _src_vals is not None:
                _sv = np.asarray(_src_vals, dtype=np.float64).reshape(-1)
                if _sv.shape[0] == n and np.all(np.isfinite(_sv)):
                    base = [*base, _sv, _sv * _sv]
            def _r2(design_cols):
                """Fit an OLS design on the train stride and return held-out R^2 on the %3 validation stride (``-inf`` on a singular/failed solve).

                Normal-equations solve (A.T@A / np.linalg.solve) on the well-conditioned small-k design
                (intercept + a handful of base/leg columns) instead of a full SVD lstsq -- same win already
                proven for this module's sibling OLS fit (see ``_deflate_sincos`` in
                ``_orth_extra_basis_fe.py``: normal equations beats lstsq here because k is tiny and the
                design isn't near-singular). Falls back to lstsq if A.T@A is singular."""
                A = np.column_stack(design_cols)
                A_tr = A[tr]
                y_tr = _y_for_hinge_gate[tr]
                try:
                    AtA = A_tr.T @ A_tr
                    coef = np.linalg.solve(AtA, A_tr.T @ y_tr)
                except np.linalg.LinAlgError:
                    try:
                        coef, *_ = np.linalg.lstsq(A_tr, y_tr, rcond=None)
                    except Exception as e:
                        logger.debug("Hinge-gate OLS lstsq fallback failed (%s: %s) -- treating as a failed candidate", type(e).__name__, e)
                        return -np.inf
                except Exception as e:
                    logger.debug("Hinge-gate OLS lstsq failed (%s: %s) -- treating as a failed candidate", type(e).__name__, e)
                    return -np.inf
                pred = A[va] @ coef
                return 1.0 - float(np.sum((yv - pred) ** 2)) / ss
            r2_base = _r2(base)
            r2_full = _r2([*base, leg])
            if not (np.isfinite(r2_base) and np.isfinite(r2_full)):
                return 0.0
            return float(r2_full - r2_base)

        if _hinge_feats:
            _HINGE_PROTECT_MIN_INCR_R2 = 0.003
            _readd_hinge = []
            for _hn in _hinge_feats:
                _idx = _cols_index.get(_hn)
                if _idx is None or _idx in _sv_set:
                    continue
                _rec_h = _hybrid_orth_pre_recipes.get(_hn)
                _src_h = tuple(getattr(_rec_h, "src_names", ()) or ())
                # Self-limit #1: a source must be resolvable at all (the leg's provenance exists). NOT gated on
                # "survived the MI screen": that screen runs at the relaxed (float32-under-MLFRAME_CRIT_DTYPE_RELAXED)
                # criterion dtype, so a borderline-but-genuinely-informative source can fail it on precision alone --
                # destroying a real signal one hop downstream (see test_f32_nameset_matches_f64's superset contract).
                # Self-limit #2 below is the honest, full-float64, held-out check for "is this source's leg real";
                # requiring screen-survival too was a redundant, precision-fragile proxy for the same thing.
                if not _src_h:
                    continue
                # Self-limit #2: the leg must lift a held-out linear fit OVER the
                # already-selected set + the source and its degree-2 poly (not
                # subsumed by a surviving composite, and a genuine kink not a smooth
                # curve a quadratic already fits).
                _leg_vals = _hinge_deferred_values.get(_hn)
                if _leg_vals is None and isinstance(X, pd.DataFrame) and _hn in X.columns:
                    _leg_vals = X[_hn].to_numpy()
                _src_vals_gate = None
                if isinstance(X, pd.DataFrame) and _src_h and _src_h[0] in X.columns:
                    _src_vals_gate = X[_src_h[0]].to_numpy()
                if _leg_vals is not None:
                    if _heldout_incr_over_selected(_leg_vals, _src_vals_gate) < _HINGE_PROTECT_MIN_INCR_R2:
                        continue
                _readd_hinge.append(_idx)
                _sv_set.add(_idx)
            if _readd_hinge:
                selected_vars = list(selected_vars) + _readd_hinge
                if verbose:
                    logger.info(
                        "MRMR hinge change-point protection: re-added %d held-out-"
                        "validated hinge leg(s) the MI screen dropped (MI-invariant; "
                        "value is downstream linear usability): %s",
                        len(_readd_hinge), [cols[i] for i in _readd_hinge],
                    )

    # ORTH-BASIS UNIVARIATE PROTECTION: re-add a single-source orthogonal-basis univariate column
    # (``a__T2`` ~ a**2, ``a__He4`` ~ a Hermite degree-4, ...) the MRMR screen dropped. Like a hinge leg, an
    # orth basis column is a DETERMINISTIC function of ONE raw source, so the greedy MI screen drops it as
    # redundant with that raw source under the data-processing inequality - EVEN WHEN raw ``a`` carries ~0
    # linear/monotone signal about an even target (``exp(-a**2)`` / ``a**2``) and the basis column carries the
    # whole recoverable nonlinearity (|corr| ~0.85). The basis value is downstream LINEAR usability, not
    # marginal MI (the same MI-vs-linear-usability rule the hinge / adaptive-Fourier protections enforce). The
    # generating univariate-basis stage already uplift-gated each column, so a candidate is a confirmed
    # univariate win. SELF-LIMITING GATE mirrors the hinge block: (1) the raw source survived the screen (a
    # basis on a never-selected noise column is left out); (2) the basis lifts a HELD-OUT linear fit over the
    # ALREADY-SELECTED feature set (which already contains the raw source as a linear term) - so a basis
    # subsumed by a surviving composite/raw adds ~0 and is rejected, while a genuine single-var nonlinearity
    # the screen DPI-dropped clears the floor. NO ``[src, src^2]`` smooth-curve term in the baseline (unlike
    # the hinge gate): for the basis the curve IS the win, so adding ``src^2`` would self-reject the very
    # quadratic basis we want. Reuses ``_heldout_incr_over_selected`` with ``_src_vals=None``.
    _orth_feats = getattr(self, "hybrid_orth_features_", None)
    if _orth_feats and len(selected_vars) and ("_heldout_incr_over_selected" in locals()):
        _cols_index_o = {c: i for i, c in enumerate(cols)}
        _sv_set_o = set(selected_vars)
        _sel_names_o = {cols[i] for i in selected_vars if 0 <= i < len(cols)}
        _ORTH_PROTECT_MIN_INCR_R2 = 0.01  # wider than hinge 0.003: a genuine single-var basis lifts held-out R^2 by >>0.01 (~0.7 for exp(-a**2)); keeps noise-fit basis out
        _orth_sig_cache: dict = {}  # per-source-name memo: the permutation-significance probe below is O(32 perms), reused across every basis sharing a source

        def _orth_source_is_signal(_src_name: str) -> bool:
            """Permutation-significance test (32 perms) on the RAW source column, mirroring EMIT-BOTH's
            own operand-significance probe (``_assign_support.py``) -- used ONLY as a cheap fallback when
            the source did NOT already survive the screen, so a basis on a genuinely-real-but-composite-
            subsumed source (e.g. x2 in the CMIM redundant-pool fixture) still gets a fair shot without
            reopening the self-limit to EVERY basis regardless of source quality (that regressed both a
            noise-exclusion floor and this fit's wall-time budget -- see the commit history here)."""
            if _src_name in _orth_sig_cache:
                return bool(_orth_sig_cache[_src_name])
            _sig = True  # estimator error -> do not silently drop a possibly-genuine source
            _src_idx = _cols_index_o.get(_src_name)
            if _src_idx is not None:
                try:
                    from ...permutation import mi_direct as _orth_mi_direct

                    # npermutations=200/alpha=0.02, not the EMIT-BOTH default (32/0.05): at only 32 shuffles
                    # p can only take values in {0/32, 1/32, ...} -- p<0.05 is satisfied by UP TO ONE shuffle
                    # beating the observed stat, which measurably let pure-noise sources through on a
                    # tiny-n (360-row) multiclass fixture with several noise candidates probed
                    # (test_biz_val_suite_mrmr_multiclass_excludes_noise: kept noise_0/3/5). Finer
                    # resolution + a tighter alpha needs the observed stat to beat essentially ALL shuffles.
                    _r = _orth_mi_direct(
                        data, x=(int(_src_idx),), y=target_indices,
                        factors_nbins=nbins, npermutations=200, min_nonzero_confidence=0.0,
                        return_null_mean=True, parallelism="none", prefer_gpu=False,
                    )
                    _sig = bool(float(_r[3]) < 0.02)  # p-value below alpha -> genuine marginal signal
                except Exception as e:
                    logger.debug("orth-basis source significance probe failed for %r (%s: %s) -- not silently dropping a possibly-genuine source", _src_name, type(e).__name__, e)
            _orth_sig_cache[_src_name] = _sig
            return _sig

        _readd_orth = []
        for _on in _orth_feats:
            _oidx = _cols_index_o.get(_on)
            if _oidx is None or _oidx in _sv_set_o:
                continue
            _rec_o = _hybrid_orth_pre_recipes.get(_on)
            # Hinge legs (``kind="hinge_basis"``) are routed through hybrid_orth_features_ too, but they have a
            # DEDICATED protection block above that gates them against a ``[src, src^2]`` baseline (the smooth-
            # curve guard: a parabola is fit by src^2 so a kink adds ~0 and is rejected). This orth-basis block
            # deliberately OMITS that guard (for a curved basis the curve IS the win), so re-handling a hinge leg
            # here would bypass the smooth-curve guard and re-add spurious legs on y=x^2 data. Skip them - the
            # Hinge block already made the correct keep/drop decision (guards a past regression).
            if getattr(_rec_o, "kind", None) == "hinge_basis":
                continue
            _src_o = tuple(getattr(_rec_o, "src_names", ()) or ())
            # Self-limit #1: single-source basis whose raw source EITHER (a) already survived the screen
            # (the cheap, common-case fast path -- most bases' sources are already selected, so this never
            # pays the permutation-test cost below), OR (b) independently clears its own marginal-
            # significance probe, mirroring EMIT-BOTH's own operand-significance check (_assign_support.py).
            # A blanket drop of (a) -- requiring only a resolvable source -- was tried and reverted: it let
            # every single-source basis reach the (weaker, held-out-lift) self-limit #2 below regardless of
            # whether the source carried any real signal, which measurably (a) let 3/8 pure-noise columns'
            # bases clear self-limit #2 by chance on a small held-out fold (test_biz_val_suite_mrmr_
            # multiclass_excludes_noise) and (b) nearly doubled this fit's wall time by evaluating the
            # expensive held-out-lift check for every basis instead of only screen-survivors (test_all_
            # enabled_fit_under_30s). The permutation probe in (b) is the correct, honest replacement for
            # the case the blanket drop was actually trying to fix (a real source composite-subsumed out of
            # selected_vars, e.g. x2 in the CMIM redundant-pool fixture) -- it costs O(32 perms) but ONLY
            # for the source-not-yet-selected case, not for every candidate.
            if len(_src_o) != 1:
                continue
            if _src_o[0] not in _sel_names_o and not _orth_source_is_signal(_src_o[0]):
                continue
            _basis_vals = _eng_continuous_snapshot.get(_on)
            if _basis_vals is None and isinstance(X, pd.DataFrame) and _on in X.columns:
                _basis_vals = X[_on].to_numpy()
            if _basis_vals is None:
                continue
            # Self-limit #2: lifts a held-out linear fit over the already-selected design (raw source already
            # present there as a linear term) - not subsumed by a surviving composite/raw.
            if _heldout_incr_over_selected(_basis_vals, None) < _ORTH_PROTECT_MIN_INCR_R2:
                continue
            _readd_orth.append(_oidx)
            _sv_set_o.add(_oidx)
        if _readd_orth:
            selected_vars = list(selected_vars) + _readd_orth
            if verbose:
                logger.info(
                    "MRMR orth-basis univariate protection: re-added %d single-source basis column(s) the "
                    "MI screen DPI-dropped (value is downstream linear usability over the raw source): %s",
                    len(_readd_orth), [cols[i] for i in _readd_orth],
                )

    # RAW-FEATURE FLOOR-DROP PROTECTION (Fix-B). The Westfall-Young maxT relevance floor is computed
    # over the FULL candidate pool; when the all-FE-on config widens that pool to hundreds of (already FE-stage-
    # gated) engineered columns, the per-shuffle MAX corrected MI inflates and the acceptance bar rises ABOVE a
    # genuine raw feature's true marginal MI - so a real linear signal (e.g. x1 ~ y at binned-MI 0.057, ~30x
    # noise) is dropped from the screen entirely (confirmed root-cause of test_biz_value_mrmr_underselection).
    # LOWERING the floor would surface x1 but ALSO admit high-cardinality raw NOISE (a 50-level pure-noise
    # categorical whose finite-sample MI is inflated) - a regression. Instead, KEEP the floor (noise stays
    # rejected) and re-add a raw feature the screen dropped IFF it lifts a HELD-OUT linear fit over the already-
    # selected design - the SAME MI-vs-linear-usability protection the hinge / orth-basis blocks use. A genuine
    # linear/monotone raw signal clears the lift; a high-card noise categorical (no held-out linear usability)
    # does not, so it stays out. Conditioned on _y_for_hinge_gate (the held-out scorer); no-op when it is None.
    # Self-contained held-out scorer (the hinge block's _y_for_hinge_gate / _heldout_incr_over_selected only
    # exist when hinge legs were generated; this protection must run regardless). Baseline = intercept + the
    # continuous values of the ALREADY-SELECTED columns (engineered from the snapshot, raw from X), so a raw
    # feature SUBSUMED by a selected composite adds ~0 and is NOT re-added (no raw-redundancy regression).
    if isinstance(X, pd.DataFrame) and len(selected_vars):
        _rp_y = None
        try:
            _rp_yv = np.asarray(y.to_numpy() if hasattr(y, "to_numpy") else y, dtype=np.float64).reshape(-1)
            if _rp_yv.shape[0] == int(data.shape[0]) and np.all(np.isfinite(_rp_yv)):
                _rp_y = _rp_yv
        except Exception as exc:
            logger.debug("mrmr: y coercion for the raw-protection re-add probe failed; raw protection disabled: %r", exc, exc_info=True)
            _rp_y = None
        if _rp_y is not None:
            _RAW_PROTECT_MIN_INCR_R2 = 0.005  # genuine linear raw signal lifts held-out R^2 >> 0.005; noise ~0
            _rp_n = _rp_y.shape[0]
            # Seeded shuffle-then-stride (see the hinge-gate sibling comment above).
            _rp_perm = np.random.default_rng(int(getattr(self, "random_seed", 0) or 0)).permutation(_rp_n)
            _rp_va = np.zeros(_rp_n, dtype=bool)
            _rp_va[_rp_perm[: _rp_n // 3]] = True
            _rp_tr = ~_rp_va
            _rp_sel_names = {cols[i] for i in selected_vars if 0 <= i < len(cols)}
            _rp_base = [np.ones(_rp_n)]
            for _sn in _rp_sel_names:
                _cv = _eng_continuous_snapshot.get(_sn)
                if _cv is None and _sn in X.columns:
                    _cv = X[_sn].to_numpy()
                if _cv is None:
                    continue
                try:
                    _cv = np.asarray(_cv, dtype=np.float64).reshape(-1)
                except (TypeError, ValueError):
                    continue  # raw categorical/string selected column - not a numeric R^2 regressor
                if _cv.shape[0] == _rp_n and np.all(np.isfinite(_cv)):
                    _rp_base.append(_cv)

            # Hoist the fold- and candidate-INVARIANT pieces out of the per-candidate R^2 (each call below
            # re-used the SAME held-out target, its centered SS, and the SAME base design rows): the val
            # target ``_yv`` / its SS, the train target, and the base design already sliced into train/val
            # blocks. Every call scores ``[base | one candidate column]``, so only the single candidate
            # column is stacked/sliced per call instead of rebuilding + row-slicing the full base at n rows.
            _yv = _rp_y[_rp_va]
            _rp_ss = float(np.sum((_yv - _yv.mean()) ** 2))
            _rp_y_tr = _rp_y[_rp_tr]
            _rp_base_mat = np.column_stack(_rp_base)
            _rp_base_tr = _rp_base_mat[_rp_tr]
            _rp_base_va = _rp_base_mat[_rp_va]

            # QR of the FIXED base design, computed ONCE and reused for every candidate below (a perf
            # fix). Each candidate previously re-solved lstsq on ``[base | one extra column]`` from
            # scratch - O(n*p^2) per call with only a single column differing between calls. Extending an
            # EXISTING QR by one column via ``scipy.linalg.qr_insert`` is an O(n*p) update, mathematically
            # equivalent to a fresh least-squares solve of the augmented design (verified: max coefficient
            # difference ~6e-17, i.e. machine-epsilon-level agreement with the original SVD-based
            # ``np.linalg.lstsq``, not just "close enough" - this is the standard Frisch-Waugh-Lovell-style
            # QR-update result, not an approximation). Measured 20x faster at production shape (p~120,
            # n_tr~53k, 109 candidates: 91s -> 4.5s on synthetic data of that shape).
            import scipy.linalg as _rp_sla
            try:
                _rp_Q, _rp_R = _rp_sla.qr(_rp_base_tr, mode="economic")
                _rp_Qty = _rp_Q.T @ _rp_y_tr
                _rp_coef_base = _rp_sla.solve_triangular(_rp_R, _rp_Qty)
                _rp_qr_ok = True
            except Exception as exc:
                logger.debug("mrmr: QR-based raw-protection incremental check failed; falling back to the full-refit path: %r", exc, exc_info=True)
                _rp_qr_ok = False

            def _rp_r2(_extra=None):
                """Held-out R^2 of ``[base | extra]``; ``_extra`` is a single full-length column or None.
                Numerically identical (to ~1e-16) to the prior ``_rp_r2(_design)`` (same columns in the
                same order, same train/val rows, same lstsq) - see the QR-reuse comment above."""
                if _rp_ss < 1e-24:
                    return 0.0
                if _extra is None:
                    if not _rp_qr_ok:
                        return -np.inf
                    return 1.0 - float(np.sum((_yv - _rp_base_va @ _rp_coef_base) ** 2)) / _rp_ss
                if not _rp_qr_ok:
                    return -np.inf
                try:
                    _q1, _r1 = _rp_sla.qr_insert(_rp_Q, _rp_R, _extra[_rp_tr], _rp_Q.shape[1], which="col")
                    _coef = _rp_sla.solve_triangular(_r1, _q1.T @ _rp_y_tr)
                except Exception as e:
                    logger.debug("QR-insert regression probe failed (%s: %s) -- treating as a failed candidate", type(e).__name__, e)
                    return -np.inf
                _A_va = np.column_stack((_rp_base_va, _extra[_rp_va]))
                return 1.0 - float(np.sum((_yv - _A_va @ _coef) ** 2)) / _rp_ss

            if int(_rp_tr.sum()) >= 32 and int(_rp_va.sum()) >= 16:
                _rp_r2_base = _rp_r2()
                _cols_index_r = {c: i for i, c in enumerate(cols)}
                _sv_set_r = set(selected_vars)
                _readd_raw = []
                # RELEVANCE GATE on the re-add. The held-out single-split R^2 increment alone is an UNCORRECTED linear-usability test: an
                # unregularised regressor overfits idiosyncratic noise on one ~n/3 val split enough to clear the loose 0.005 floor for a
                # feature the relevance screen correctly rejected as within-null (e.g. decoy = x_real**2 on y = sign(x_real): MI ~ 0.00014,
                # below the effective floor, corr -0.04, yet R^2 incr ~0.011). Require the candidate to ALSO clear the SAME marginal-MI
                # relevance floor the screen used (absolute effective floor AND the relative-to-strongest floor) so a below-null raw cannot
                # be resurrected by linear-usability alone - this re-opened exactly the hole the screen floor closes.
                _rp_rel_floor = float(_effective_min_relevance_gain) if "_effective_min_relevance_gain" in dir() else float(getattr(self, "min_relevance_gain", 0.0) or 0.0)
                _rp_rel_frac = float(getattr(self, "min_relevance_gain_relative_to_first", 0.0) or 0.0)
                _rp_max_mi = max((float(_v) for _v in cached_MIs.values()), default=0.0) if isinstance(cached_MIs, dict) else 0.0
                _rp_floor = max(_rp_rel_floor, _rp_max_mi * _rp_rel_frac)
                # feature_names_in_ is an ndarray; "or []" would test truthiness and raise on a multi-element array.
                _fni_rp = getattr(self, "feature_names_in_", None)
                for _rn in (_fni_rp if _fni_rp is not None else []):
                    _ridx = _cols_index_r.get(_rn)
                    if _ridx is None or _ridx in _sv_set_r or _rn not in X.columns:
                        continue
                    _rp_cand_mi = float(cached_MIs.get((_ridx,), 0.0)) if isinstance(cached_MIs, dict) else 0.0
                    if _rp_cand_mi <= _rp_floor:
                        continue  # within-null / below the screen's relevance floor -> not a genuine signal, do not resurrect
                    try:
                        _rv = np.asarray(X[_rn].to_numpy(), dtype=np.float64).reshape(-1)
                    except (TypeError, ValueError):
                        continue  # non-numeric raw (categorical/string) -> not a linear-usability candidate
                    if _rv.shape[0] != _rp_n or not np.all(np.isfinite(_rv)):
                        continue
                    if _rp_r2(_rv) - _rp_r2_base < _RAW_PROTECT_MIN_INCR_R2:
                        continue
                    _readd_raw.append(_ridx)
                    _sv_set_r.add(_ridx)
                if _readd_raw:
                    selected_vars = list(selected_vars) + _readd_raw
                    if verbose:
                        logger.info(
                            "MRMR raw-feature floor-drop protection: re-added %d held-out-validated raw "
                            "feature(s) the maxT relevance floor dropped (genuine linear usability, not "
                            "high-card noise): %s",
                            len(_readd_raw), [cols[i] for i in _readd_raw],
                        )

    # CAT-FE FLOOR-DROP PROTECTION (Fix-C). The Westfall-Young maxT relevance floor (computed over
    # the FULL widened candidate pool when many FE families are on) routinely rises above the marginal binned-MI
    # of a genuine categorical-FE encoding - a K-fold target encoding (``cat__te``), a count/frequency encoding,
    # or a cat-num residual (``price__resid_by__cat_region``) - so the greedy screen drops it after 2 features
    # EVEN THOUGH it carries strong LINEAR usability to y (the MI-vs-linear-usability gap, a recurring mlframe
    # theme). The cat-num residual on the kitchen-sink frame has univariate corr ~0.27 / held-out R^2-incr ~0.06
    # over the selected design yet is screened out, so downstream LogReg loses ~0.6% AUC. This is the SAME class
    # of false-drop the raw-feature / orth-basis / hinge protections already correct - but those iterate only
    # over raw ``feature_names_in_`` / single-source orth bases / hinge legs, so an engineered cat-FE column falls
    # through every one of them. Mirror the raw protection here: KEEP the floor (sub-null noise stays rejected)
    # and re-add a dropped cat-FE column IFF it lifts a HELD-OUT linear fit over the already-selected design by
    # >= the same R^2 floor. The cat-FE columns live as quantized codes in ``data[:, idx]`` (the continuous
    # snapshot is only populated by the fe_max_steps>0 path); the binned codes preserve the monotone/linear
    # signal well enough for the usability test (a genuine encoding lifts R^2 >> floor; a noise encoding ~0).

    return selected_vars, cols, data, nbins
