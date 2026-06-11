"""``ShapProxiedFitMixin`` -- the search + fit machinery for :class:`ShapProxiedFS`.

Carved out of ``shap_proxied_fs.py`` to keep both files under the 1k LOC ceiling. The mixin holds
the optimizer dispatch (``_run_search``) and the full ``fit`` pipeline; ``ShapProxiedFS`` inherits
it so ``self`` (all constructor state + the resolver methods on the concrete class) stays intact.
Heavy dependencies are lazy-imported in-body as in the original, so this module imports only the
few module-scope names the two method bodies reference directly.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from mlframe.feature_selection.shap_proxied_fs._shap_proxied_resolvers import _resolve_adaptive_prescreen_width
from mlframe.utils.misc import rng_hygienic_fit

logger = logging.getLogger(__name__)


class ShapProxiedFitMixin:
    """Search-dispatch + fit pipeline for :class:`ShapProxiedFS` (see module docstring)."""

    def _run_search(self, optimizer, phi, base, y):
        """Dispatch to the chosen optimizer; returns list of (proxy_loss, feature_idx tuple)."""
        kw = dict(classification=self.classification, metric=self.metric,
                  max_card=self.max_features, top_n=self.top_n)
        if optimizer == "bruteforce":
            from mlframe.feature_selection._shap_proxy_search import brute_force_top_n

            return brute_force_top_n(phi, base, y, min_card=self.min_features,
                                     parallel=(phi.shape[1] >= 14), **kw)
        if optimizer == "bruteforce_gpu":
            from mlframe.feature_selection._shap_proxy_gpu import brute_force_top_n_gpu, gpu_available

            if gpu_available():
                return brute_force_top_n_gpu(phi, base, y, min_card=self.min_features, **kw)
            logger.warning("ShapProxiedFS: use_gpu requested but no CUDA device; falling back to numba brute force.")
            from mlframe.feature_selection._shap_proxy_search import brute_force_top_n

            return brute_force_top_n(phi, base, y, min_card=self.min_features, parallel=True, **kw)
        from mlframe.feature_selection import _shap_proxy_heuristics as H

        if optimizer == "beam":
            return H.beam_search(phi, base, y, beam_width=self.beam_width, min_card=self.min_features, **kw)
        if optimizer == "greedy_forward":
            return H.greedy_forward(phi, base, y, classification=self.classification, metric=self.metric,
                                    max_card=self.max_features, top_n=self.top_n)
        if optimizer == "greedy_backward":
            return H.greedy_backward(phi, base, y, classification=self.classification, metric=self.metric,
                                     min_card=self.min_features, top_n=self.top_n)
        if optimizer == "multistart":
            return H.multistart_local(phi, base, y, rng=self._rng, **kw)
        if optimizer == "genetic":
            return H.genetic(phi, base, y, rng=self._rng, **kw)
        if optimizer == "annealing":
            return H.simulated_annealing(phi, base, y, rng=self._rng, **kw)
        if optimizer == "gradient":
            from mlframe.feature_selection._shap_proxy_gradient import gradient_top_n

            return gradient_top_n(phi, base, y, classification=self.classification, metric=self.metric,
                                  random_state=int(self.random_state), top_n=self.top_n)
        raise ValueError(f"Unknown optimizer={optimizer!r}")

    # ------------------------------------------------------------------ fit
    @rng_hygienic_fit
    def fit(self, X, y):
        import time
        from contextlib import contextmanager

        from mlframe.feature_selection._shap_proxy_explain import compute_shap_matrix, make_default_estimator

        # Optional per-stage wall-clock instrumentation for the scaling benchmark / profiling. Set
        # ``self._stage_timings`` to a dict before calling fit and each stage's seconds land in it; a
        # no-op otherwise (zero overhead beyond a dict lookup), so production fits are unaffected.
        _timings = getattr(self, "_stage_timings", None)

        @contextmanager
        def _stage(name):
            if _timings is None:
                yield
                return
            t0 = time.perf_counter()
            try:
                yield
            finally:
                _timings[name] = _timings.get(name, 0.0) + (time.perf_counter() - t0)

        # Control/safety budget (parity with MRMR / RFECV). The OOF-SHAP + proxy-search core always
        # runs (it produces the candidate subsets), but the OPTIONAL expensive refinement phases
        # below -- honest revalidation, importance ablation, within-cluster refine -- are each gated
        # on this budget. When the wall-clock budget is exceeded OR ``stop_file`` exists, they are
        # skipped and fit() finalises with the proxy-best subset (a valid selection). ``getattr``
        # keeps old pickled instances (without the new attrs) working.
        from os.path import exists as _stop_file_exists
        _budget_t0 = time.perf_counter()
        _budget_max_mins = getattr(self, "max_runtime_mins", None)
        _budget_stop_file = getattr(self, "stop_file", None)

        def _budget_exhausted() -> bool:
            if _budget_max_mins and (time.perf_counter() - _budget_t0) > _budget_max_mins * 60.0:
                return True
            return bool(_budget_stop_file) and _stop_file_exists(_budget_stop_file)

        X = self._to_pandas(X).reset_index(drop=True)
        X.columns = [str(c) for c in X.columns]
        self.feature_names_in_ = np.asarray(list(X.columns))
        self.n_features_in_ = int(X.shape[1])
        n_features = self.n_features_in_
        y = self._coerce_target(y)
        # Reset per-fit su_seeded_interactions scratch so a re-fit never reuses a prior fit's pairs.
        self._su_seeded_pairs_orig = None
        self._su_seeded_screen_info = {}

        if self.model is not None:
            model_template = self.model
        else:
            # Resolve + validate booster_kind FIRST so a typo (e.g. "bogus") raises ValueError before
            # the expensive prefilter / OOF-SHAP stages start. Auto-detect when None.
            _booster_kind = self._resolve_booster_kind()
            # The catboost ``cat_features`` template (iter78) is forwarded to the booster, but the
            # surrounding pipeline is NOT categorical-aware: the prefilter densifies ``X.values``
            # to float64 (crashes on string/categorical columns inside ``f_classif_chunked``), the
            # clustering treats columns as numeric, and column-slicing between stages does not
            # remap name-based ``cat_features``. Fail fast with an actionable message here instead
            # of the obscure downstream ValueError. Numeric-only fits (cat_features unset) are fine.
            if _booster_kind == "catboost" and self.cat_features:
                raise ValueError(
                    "ShapProxiedFS(booster_kind='catboost', cat_features=...) is not yet supported: "
                    "the prefilter / clustering / column-slicing stages are not categorical-aware and "
                    "would crash on densification or mis-route the cat-feature indices. Pre-encode the "
                    "categorical columns numerically and pass cat_features=None, or supply an explicit "
                    "fitted-pipeline ``model=`` template that handles categoricals upstream."
                )
            model_template = make_default_estimator(
                self.classification, random_state=int(self.random_state),
                booster_kind=_booster_kind, cat_features=self.cat_features,
            )

        # Disjoint holdout for honest re-validation + trust guard (avoids winner's curse).
        stratify = y if self.classification else None
        idx_all = np.arange(len(X))
        idx_search, idx_hold = train_test_split(
            idx_all, test_size=self.holdout_size, random_state=int(self.random_state),
            shuffle=True, stratify=stratify)
        # Wide-frame split with deferred holdout materialisation. At C4 (width=20000, n_rows=10000)
        # the original frame is 1.49 GiB, the search slice (75% rows) is 1.12 GiB, and the holdout
        # slice (25% rows) is 381 MiB; the legacy back-to-back
        # `X.iloc[idx_search].reset_index(drop=True)` + `X.iloc[idx_hold].reset_index(drop=True)`
        # held all three simultaneously plus reset_index transient buffers and OOM'd on a
        # 17 GB / 6.4 GB-free host (iter46). The prefilter only needs `X_search`; once it returns
        # `working_cols` (typically <=704 entries -- effective_prefilter_top is bounded by the
        # SHAP-prefilter cap), the holdout can be built directly at that narrow column count
        # (~5 MiB instead of 381 MiB). Keep `X_vals_full` alive (a view into the original block,
        # zero extra alloc on a single-dtype input) so the deferred holdout materialisation has a
        # source to slice from. `reset_index(drop=True)` is dropped throughout: downstream consumers
        # all read via `.values` / `.iloc[:, cols]` / positional row access, none depend on the row
        # index being 0..n-1 RangeIndex (and `compute_shap_matrix` does its own
        # `reset_index(drop=True)` on the narrow post-prefilter frame anyway).
        X_cols = X.columns
        n_cols = X.shape[1]
        X_vals_full = X.values  # single-block float64 view on bench / homogeneous input
        # Build wide X_search via column-batched copy from the parent block. One batch's worth of
        # transient memory (~80 MiB at default 1024-col batch), not a full extra split copy.
        X_search_arr = np.empty((idx_search.shape[0], n_cols), dtype=X_vals_full.dtype)
        col_batch = self._split_col_batch
        for c0 in range(0, n_cols, col_batch):
            c1 = min(c0 + col_batch, n_cols)
            X_search_arr[:, c0:c1] = X_vals_full[idx_search, c0:c1]
        X_search = pd.DataFrame(X_search_arr, columns=X_cols, copy=False)
        del X, X_search_arr
        y_search = y[idx_search]
        # Defer X_hold materialisation: store the inputs and let the post-prefilter step build it
        # at the narrow working-column count.
        self._deferred_holdout = (X_vals_full, idx_hold, X_cols)
        X_hold = None  # built post-prefilter (or pre-clustering if prefilter is skipped)
        y_hold = y[idx_hold]

        report: dict = {}

        # iter66: validate + align the precomputed cross-selector artifacts
        # (canonically from ``MRMR(retain_artifacts=True).export_artifacts()``)
        # against X.columns. On any mismatch ``align_precomputed_to_X`` logs a
        # warning and returns None so the prefilter falls back to legacy
        # behaviour. The diagnostic block is always surfaced under
        # ``report['precomputed_used']`` so callers can confirm which path ran.
        from mlframe.feature_selection._shap_proxy_precomputed import (
            align_precomputed_to_X, su_to_prefilter_keep,
        )
        _precomputed_aligned, _precomputed_report = align_precomputed_to_X(
            self.precomputed, X_search,
        )
        report["precomputed_used"] = _precomputed_report
        report["precomputed_bins_available"] = bool(
            isinstance(_precomputed_aligned, dict) and _precomputed_aligned.get("bins")
        )

        # Cheap native-importance pre-filter BEFORE the expensive OOF-SHAP. SHAP cost scales with the
        # column count, and clustering only compresses CORRELATED features (independent noise stays as
        # singletons), so on wide data SHAP would otherwise run on ~all columns. Rank all features and
        # keep the top-K; ``working_cols`` maps the surviving working columns back to original indices
        # for the final selector output. ``prefilter_method`` trades speed against interaction-awareness
        # (model / univariate / fast_model / gpu_model / two_stage); "auto" stays quality-safe for
        # moderate widths and routes very wide data (n_features >= 8000) to the cheap-funnel +
        # capped-booster two_stage path -- see ``_shap_proxy_prefilter``.
        #
        # SHAP-pre-prefilter (iter31): when enabled (default), tighten the effective ``prefilter_top``
        # to ``shap_prefilter_top = max(brute_force_max_features * safety_factor,
        # shap_prefilter_min_features)`` (default 88) so the post-prefilter cohort that feeds OOF-SHAP
        # is sized to the downstream search budget plus a 4x cushion, instead of the default 2000.
        # The downstream search only consumes top-``brute_force_max_features`` by mean |phi| anyway,
        # so noise-tail columns between the search cap and the loose default were paying full TreeSHAP
        # cost for no contribution. Realized by REUSING the existing prefilter's booster ranking (a
        # separate post-clustering booster fit was bench-attempt-rejected 2026-05-28: the extra fit
        # cost ~1.2s while saving ~1.3s on OOF-SHAP for a +0.1s wash at width=1000/rows=5000/seed=1,
        # despite a +17% gain at the cold-start seed=0). Tightening at the prefilter step avoids the
        # double-booster work AND keeps the lever's win on warm runs.
        effective_prefilter_top = self.prefilter_top
        if self.shap_prefilter_enabled and self.prefilter_top is not None:
            from mlframe.feature_selection._shap_proxy_shap_prefilter import (
                resolve_shap_prefilter_top)

            sp_top = (self.shap_prefilter_top if self.shap_prefilter_top is not None
                      else resolve_shap_prefilter_top(
                          brute_force_max_features=self.brute_force_max_features,
                          safety_factor=self.shap_prefilter_safety_factor,
                          min_features=self.shap_prefilter_min_features))
            # Tighten only -- never expand the user's prefilter budget.
            effective_prefilter_top = min(int(self.prefilter_top), int(sp_top))
            report["shap_prefilter"] = dict(
                requested_top=int(sp_top),
                effective_prefilter_top=int(effective_prefilter_top),
                user_prefilter_top=int(self.prefilter_top))
        working_cols = np.arange(n_features)
        if effective_prefilter_top is not None and n_features > effective_prefilter_top:
            from mlframe.feature_selection._shap_proxy_prefilter import (
                _default_stage1_keep, prefilter_columns, resolve_prefilter_method)

            # iter33 SHAP-aware stage-A tightening: when ``shap_prefilter`` shrinks
            # ``effective_prefilter_top`` far below the legacy 2000 default, the two_stage prefilter's
            # stage-B booster fit on 2000 columns is the dominant wall-clock cost. Pre-resolve the
            # stage-A survivor count to ``max(floor, effective_prefilter_top * cushion)`` so the
            # booster fits on ~3x fewer columns at the same tree budget. Two protections:
            #   1) User-pinned ``prefilter_stage1_keep`` always wins (lever is a default-only tighten).
            #   2) Lever is gated on (a) ``shap_aware_stage1_keep=True``, (b)
            #      ``shap_prefilter_enabled=True``, (c) the resolved prefilter method == two_stage
            #      (only two_stage reads ``stage1_keep``; other paths ignore it).
            effective_stage1_keep = self.prefilter_stage1_keep
            if (effective_stage1_keep is None and self.shap_aware_stage1_keep
                    and self.shap_prefilter_enabled):
                _resolved = resolve_prefilter_method(
                    self.prefilter_method, n_features=n_features,
                    n_rows=int(X_search.shape[0]))
                if _resolved == "two_stage":
                    from mlframe.feature_selection._shap_proxy_shap_prefilter import (
                        resolve_shap_aware_stage1_keep)

                    effective_stage1_keep = resolve_shap_aware_stage1_keep(
                        effective_prefilter_top=int(effective_prefilter_top),
                        stage1_cushion=self.shap_aware_stage1_cushion,
                        stage1_floor=self.shap_aware_stage1_floor,
                        default_stage1_keep=_default_stage1_keep(n_features))
                    if "shap_prefilter" in report and isinstance(report["shap_prefilter"], dict):
                        report["shap_prefilter"]["stage1_keep_tightened"] = int(effective_stage1_keep)
                        report["shap_prefilter"]["stage1_keep_default"] = int(
                            _default_stage1_keep(n_features))

            with _stage("prefilter"):
                if (_precomputed_aligned is not None
                        and "su_to_target" in _precomputed_aligned):
                    # iter66: replace the booster / F-statistic prefilter with
                    # the SU(X_j, y) ranking the MRMR screen already computed.
                    # Skips the cloned-booster fit / chunked f_classif pass
                    # entirely; the ordering is more cardinality-honest than
                    # the F-statistic for mixed-cardinality features
                    # (Witten-Frank-Hall 2011).
                    working_cols = su_to_prefilter_keep(
                        _precomputed_aligned, keep_top=int(effective_prefilter_top),
                    )
                    pf_info = {
                        "method": "precomputed_su",
                        "kept": int(working_cols.size),
                        "source": "MRMR.export_artifacts",
                    }
                else:
                    working_cols, pf_info = prefilter_columns(
                        model_template, X_search, y_search, method=self.prefilter_method,
                        prefilter_top=effective_prefilter_top, classification=self.classification,
                        n_features=n_features, n_estimators_cap=self.prefilter_n_estimators,
                        stage1_keep=effective_stage1_keep,
                        univariate_batch_size=self.prefilter_univariate_batch_size)
                # su_seeded_interactions RESCUE (#5b, OPT-IN): the native-importance / F-statistic
                # prefilter ranks by MARGINAL signal, so a PURE-interaction operand pair (op_a, op_b
                # with ~0 marginal -- the exact regime the additive proxy is blind to) sits in the
                # tail the prefilter drops, and the downstream search can never pair them. Run the
                # CHEAP pairwise-SU synergy screen on the FULL pre-prefilter X_search HERE (operands
                # still present), SNR-gate it, and RESCUE the surviving operand columns into
                # working_cols so they flow through clustering / SHAP / search as normal columns. The
                # screen is O(P)+O(K) (no O(P^2) tensor) and NO-OPs cleanly (rescues nothing) when the
                # gate clears no pair. The kept pairs (by original column name) drive the post-search
                # sparse interaction objective. Skipped when the prefilter kept everything already.
                if self.su_seeded_interactions and len(working_cols) < n_features:
                    from mlframe.feature_selection._shap_proxy_interactions import su_synergy_screen

                    with _stage("su_seeded_interactions"):
                        # ISOLATED rng (NOT self._rng): the screen's permutation-null shuffles must not
                        # advance the selector's shared RNG stream, or a no-op (gate clears nothing)
                        # would still perturb the downstream stochastic revalidation and silently
                        # change the additive default's selection. A fixed offset keeps it
                        # deterministic + reproducible while leaving self._rng byte-untouched.
                        _su_rng = np.random.default_rng(int(self.random_state) + 7919)
                        _kept_pairs_orig, _su_prefilter_info = su_synergy_screen(
                            X_search, y_search,
                            n_bins=self.su_seeded_n_bins,
                            top_k=self.su_seeded_top_k,
                            max_screen_cols=self.su_seeded_max_screen_cols,
                            snr_z=self.su_seeded_snr_z,
                            snr_null_quantile=self.su_seeded_snr_null_quantile,
                            snr_abs_floor=self.su_seeded_snr_abs_floor,
                            n_permutations=self.su_seeded_n_permutations,
                            importance=None, rng=_su_rng)
                        # X_search columns at this point are the FULL original names (prefilter slice
                        # below has not run yet); map operand names -> original feature indices.
                        _name_to_orig = {str(c): i for i, c in enumerate(X_cols)}
                        _rescue_orig: set[int] = set()
                        self._su_seeded_pairs_orig = []
                        for _syn, _jsu, _ca, _cb in _kept_pairs_orig:
                            if str(_ca) in _name_to_orig and str(_cb) in _name_to_orig:
                                _rescue_orig.add(_name_to_orig[str(_ca)])
                                _rescue_orig.add(_name_to_orig[str(_cb)])
                                self._su_seeded_pairs_orig.append(
                                    (float(_syn), str(_ca), str(_cb)))
                        if _rescue_orig:
                            _wc_set = set(int(c) for c in working_cols)
                            _added = sorted(_rescue_orig - _wc_set)
                            if _added:
                                working_cols = np.sort(np.concatenate(
                                    [np.asarray(working_cols, dtype=np.int64),
                                     np.asarray(_added, dtype=np.int64)]))
                        self._su_seeded_screen_info = dict(_su_prefilter_info)
                        self._su_seeded_screen_info["n_rescued_orig"] = len(_rescue_orig)
                if len(working_cols) < n_features:
                    X_search = X_search.iloc[:, working_cols]
                report["prefilter"] = pf_info
        # Materialise X_hold at the narrow post-prefilter column count (or the full width if no
        # prefilter ran). Deferred from the row-split above so the C4 peak holds X + X_search and
        # NOT also a wide X_hold; with prefilter on, working_cols is typically <=704 entries so
        # this slice is ~5 MiB instead of 381 MiB.
        X_vals_full, idx_hold_saved, X_cols_full = self._deferred_holdout
        if len(working_cols) < n_features:
            hold_cols = [X_cols_full[c] for c in working_cols]
            X_hold = pd.DataFrame(
                X_vals_full[np.ix_(idx_hold_saved, working_cols)],
                columns=hold_cols, copy=False)
        else:
            X_hold = pd.DataFrame(X_vals_full[idx_hold_saved], columns=X_cols_full, copy=False)
        self._deferred_holdout = None
        del X_vals_full, X_cols_full

        # Optional correlated-feature clustering: collapse to denoised UNITS so SHAP + search run on
        # hundreds of columns, not tens of thousands. unit_to_members maps proxy(unit) index ->
        # original feature columns; None means proxy index == feature column (identity).
        do_cluster = self.cluster_features is True or (
            self.cluster_features == "auto" and n_features > self.cluster_auto_threshold)
        if do_cluster:
            from mlframe.feature_selection._shap_proxy_cluster import (
                build_unit_matrix, cluster_correlated_features, cluster_summary)

            with _stage("clustering"):
                # iter75: SU is now the UNCONDITIONAL DEFAULT clustering backend whenever auto-mode
                # picks it (precomputed bins available OR width <= ``cluster_su_auto_max_features``).
                # Pearson is opt-in via ``cluster_backend="pearson"`` and the wide-width auto fallback.
                # Bins are reused from MRMR.export_artifacts when present, else computed on the fly via
                # ``categorize_dataset`` on ``X_search`` so every default user gets non-linear
                # redundancy detection (XOR / saddle / sinusoidal) that Pearson |corr| misses.
                _bins = (
                    _precomputed_aligned.get("bins")
                    if isinstance(_precomputed_aligned, dict) else None
                )
                _have_precomputed_bins = (
                    self.cluster_use_precomputed_bins
                    and isinstance(_bins, dict)
                    and len(_bins) > 0
                )
                # ``cluster_use_precomputed_bins=False`` is a per-instance opt-out from the
                # SU-with-precomputed-bins fast path; we honour it as a forced Pearson dispatch
                # (legacy semantics) regardless of cluster_backend, so existing callers keep their
                # opt-out switch.
                _backend_norm = str(self.cluster_backend).lower()
                if not bool(self.cluster_use_precomputed_bins):
                    effective_backend = "pearson"
                elif _backend_norm == "pearson":
                    effective_backend = "pearson"
                elif _backend_norm == "su":
                    effective_backend = "su"
                else:
                    # auto: prefer SU whenever bins exist (cheap reuse) or width is below the
                    # on-the-fly-binning cost gate; above the cap Pearson's vectorised correlation
                    # matrix still beats pairwise SU on wall-clock.
                    n_search_cols = int(X_search.shape[1])
                    if _have_precomputed_bins or n_search_cols <= int(self.cluster_su_auto_max_features):
                        effective_backend = "su"
                    else:
                        effective_backend = "pearson"

                _cluster_bins_source = "n/a"
                if effective_backend == "su":
                    from mlframe.feature_selection._shap_proxy_cluster_su import (
                        cluster_correlated_features_su,
                    )

                    # Reuse precomputed bins when keyed to every working column; else bin
                    # X_search on the fly via MRMR's discretiser. Keeping the precomputed path
                    # avoids the second binning pass when MRMR already produced an aligned view.
                    _cluster_names = (
                        [c for c in X_search.columns.tolist() if c in _bins]
                        if _have_precomputed_bins else []
                    )
                    if _have_precomputed_bins and len(_cluster_names) == X_search.shape[1]:
                        _nbins_pf = (
                            _precomputed_aligned.get("nbins_per_feature")
                            if isinstance(_precomputed_aligned, dict) else None
                        )
                        _bins_for_su = _bins
                        _cluster_bins_source = "precomputed"
                    else:
                        # On-the-fly binning: categorize_dataset returns (data_array, column_names,
                        # nbins_per_col). The dict layout matches what MRMR.export_artifacts produces
                        # so cluster_correlated_features_su consumes it unchanged.
                        from mlframe.feature_selection.filters.discretization import (
                            categorize_dataset,
                        )
                        _bin_data, _bin_cols, _bin_nbins = categorize_dataset(
                            X_search, n_bins=int(self.cluster_su_n_bins),
                        )
                        _bins_for_su = {
                            name: _bin_data[:, i] for i, name in enumerate(_bin_cols)
                        }
                        _nbins_pf = {
                            name: int(_bin_nbins[i]) for i, name in enumerate(_bin_cols)
                        }
                        _cluster_names = list(X_search.columns)
                        _cluster_bins_source = "on_the_fly"

                    if len(_cluster_names) == X_search.shape[1]:
                        labels = cluster_correlated_features_su(
                            _bins_for_su,
                            threshold=self.cluster_su_threshold,
                            feature_names=_cluster_names,
                            nbins_per_feature=_nbins_pf,
                        )
                        _cluster_backend = "su"
                        _cluster_threshold = float(self.cluster_su_threshold)
                    else:
                        # Defensive: categorize_dataset may drop unsupported dtypes (rare in numeric
                        # X_search); fall back to Pearson rather than partial coverage.
                        labels = cluster_correlated_features(
                            X_search.values, threshold=self.cluster_corr_threshold,
                            use_gpu=self.cluster_use_gpu)
                        _cluster_backend = "pearson_fallback_bins_incomplete"
                        _cluster_threshold = float(self.cluster_corr_threshold)
                        _cluster_bins_source = "n/a"
                else:
                    labels = cluster_correlated_features(
                        X_search.values, threshold=self.cluster_corr_threshold,
                        use_gpu=self.cluster_use_gpu)
                    _cluster_backend = "pearson"
                    _cluster_threshold = float(self.cluster_corr_threshold)
                units, unit_to_members, _kind = build_unit_matrix(
                    X_search.values, labels, weighting=self.cluster_weighting)
                X_proxy = pd.DataFrame(units, columns=[f"unit{i}" for i in range(units.shape[1])])
                _cluster_block = cluster_summary(unit_to_members)
                _cluster_block["backend"] = _cluster_backend
                _cluster_block["threshold"] = _cluster_threshold
                _cluster_block["bins_source"] = _cluster_bins_source
                report["clustering"] = _cluster_block
        else:
            X_proxy = X_search
            unit_to_members = None

        # SHAP attribution on the proxy (unit or raw) columns. Request per-model attribution variance
        # only when the uncertainty lever is active AND we actually have multiple models to vary.
        want_var = self.uncertainty_penalty > 0 and self.n_models > 1
        want_per_fold_phi = bool(self.adaptive_prescreen_by_stability) and bool(self.out_of_fold)
        with _stage("oof_shap"):
            shap_out = compute_shap_matrix(
                model_template, X_proxy, y_search, classification=self.classification,
                out_of_fold=self.out_of_fold, n_splits=self.n_splits, n_models=self.n_models,
                config_jitter=self.config_jitter, return_variance=want_var,
                rng=self._rng, tqdm_desc=("shap-oof" if self.tqdm else None), n_jobs=self.n_jobs,
                n_estimators_cap=self.oof_shap_n_estimators,
                inner_n_jobs_cap=self.inner_n_jobs_cap,
                return_per_fold_phi_mean=want_per_fold_phi,
                cache_dir=self.cache_dir)
        if want_var and want_per_fold_phi:
            phi, base, y_phi, phi_var, per_fold_phi_mean = shap_out
        elif want_var:
            phi, base, y_phi, phi_var = shap_out
            per_fold_phi_mean = None
        elif want_per_fold_phi:
            phi, base, y_phi, per_fold_phi_mean = shap_out
            phi_var = None
        else:
            phi, base, y_phi = shap_out
            phi_var = None
            per_fold_phi_mean = None

        # Adaptive prescreen narrowing (iter59): when SHAP per-fold ranks are unstable, NARROW the
        # cap so noisy mid-rank features don't get injected into beam's candidate pool. The lever is
        # measurement-driven (median pairwise Spearman of per-fold mean |phi| feature ranks) and only
        # ever narrows; high-stability regimes keep the current cap. Computed BEFORE the prescreen
        # block below so the resolved cap drives that block's keep count.
        effective_brute_force_cap = self.brute_force_max_features
        adaptive_info: Optional[dict] = None
        if want_per_fold_phi and per_fold_phi_mean is not None and per_fold_phi_mean.shape[0] >= 2:
            from mlframe.feature_selection._shap_proxy_explain import compute_phi_rank_stability

            stability = compute_phi_rank_stability(
                per_fold_phi_mean, top_k=2 * max(self.brute_force_max_features, 40))
            effective_brute_force_cap = _resolve_adaptive_prescreen_width(
                stability, default_cap=self.brute_force_max_features)
            adaptive_info = dict(
                stability=float(stability),
                default_cap=int(self.brute_force_max_features),
                effective_cap=int(effective_brute_force_cap),
            )
            report["adaptive_prescreen"] = adaptive_info

        # su_seeded_interactions screen (#5b, OPT-IN) -- resolve the synergistic operand pairs in
        # PROXY-column space BEFORE the importance pre-screen, so their operand proxy-columns can be
        # RESCUED past the prescreen (pure-interaction operands have ~0 mean|phi|, the regime the
        # additive proxy is blind to, so they sit in the tail the prescreen drops). The prefilter
        # stage already ran the cheap screen + rescued the operands past the PREFILTER when the
        # prefilter narrowed the frame (storing ``self._su_seeded_pairs_orig``); if it did NOT run
        # (narrow data, no prefilter) we run the screen on X_proxy here. Either way the cost is the
        # O(P)+O(K) screen, never the O(P^2) tensor, and it NO-OPs cleanly when the SNR gate clears
        # nothing. Operand columns are keyed by ORIGINAL feature name and mapped to proxy columns via
        # unit_to_members so clustering's unit-rename does not break the pairing.
        _su_kept_pairs: list = []  # list of (synergy, proxy_col_name_a, proxy_col_name_b)
        _su_screen_info: dict = {}
        _su_rescue_proxy_idx: set[int] = set()
        if self.su_seeded_interactions and phi.shape[1] >= 2:
            from mlframe.feature_selection._shap_proxy_interactions import su_synergy_screen

            with _stage("su_seeded_interactions"):
                proxy_names = [str(c) for c in X_proxy.columns]
                # Build ORIGINAL-feature-name -> proxy-column index. Clustering renames proxy columns
                # to ``unitK`` whose members are WORKING-positions (unit_to_members); map each member
                # working-position -> original index -> name. When clustering is off X_proxy columns
                # ARE the post-prefilter original names, so the identity map applies.
                orig_name_to_proxy: dict[str, int] = {}
                if unit_to_members is not None:
                    for _u, _members in enumerate(unit_to_members):
                        for _m in _members:
                            _orig_idx = int(working_cols[int(_m)])
                            orig_name_to_proxy[str(X_cols[_orig_idx])] = _u
                else:
                    for _u, _nm in enumerate(proxy_names):
                        orig_name_to_proxy[_nm] = _u

                pairs_orig = getattr(self, "_su_seeded_pairs_orig", None)
                if pairs_orig is None:
                    # Prefilter screen did not run (no prefilter narrowing) -> run it on X_proxy now.
                    _kp, _su_screen_info = su_synergy_screen(
                        X_proxy, y_phi,
                        n_bins=self.su_seeded_n_bins, top_k=self.su_seeded_top_k,
                        max_screen_cols=self.su_seeded_max_screen_cols,
                        snr_z=self.su_seeded_snr_z,
                        snr_null_quantile=self.su_seeded_snr_null_quantile,
                        snr_abs_floor=self.su_seeded_snr_abs_floor,
                        n_permutations=self.su_seeded_n_permutations,
                        importance=np.abs(phi).mean(axis=0),
                        rng=np.random.default_rng(int(self.random_state) + 7919))
                    # X_proxy column names are the proxy-space keys directly here.
                    for _syn, _jsu, _ca, _cb in _kp:
                        _ia = proxy_names.index(str(_ca)) if str(_ca) in proxy_names else None
                        _ib = proxy_names.index(str(_cb)) if str(_cb) in proxy_names else None
                        if _ia is not None and _ib is not None and _ia != _ib:
                            _su_kept_pairs.append((float(_syn), str(_ca), str(_cb)))
                            _su_rescue_proxy_idx.add(_ia)
                            _su_rescue_proxy_idx.add(_ib)
                else:
                    # Use the prefilter-stage screen result (operand ORIGINAL names) mapped to proxy.
                    _su_screen_info = dict(getattr(self, "_su_seeded_screen_info", {}))
                    for _syn, _ca, _cb in pairs_orig:
                        _ia = orig_name_to_proxy.get(str(_ca))
                        _ib = orig_name_to_proxy.get(str(_cb))
                        if _ia is not None and _ib is not None and _ia != _ib:
                            _su_kept_pairs.append((float(_syn), proxy_names[_ia], proxy_names[_ib]))
                            _su_rescue_proxy_idx.add(_ia)
                            _su_rescue_proxy_idx.add(_ib)

        # Importance pre-screen: when the proxy still has more columns than the exact-search budget,
        # keep the top-K by SHAP importance (mean |phi|) so exhaustive-approx stays feasible.
        n_proxy = phi.shape[1]
        proxy_cols_kept = np.arange(n_proxy)  # proxy(unit) columns behind the current phi columns
        prescreen_top = self.prescreen_top
        if prescreen_top is None and n_proxy > effective_brute_force_cap and self.optimizer in ("auto", "bruteforce", "bruteforce_gpu"):
            prescreen_top = effective_brute_force_cap
        if prescreen_top is not None and prescreen_top < n_proxy:
            with _stage("prescreen"):
                importance = np.abs(phi).mean(axis=0)
                keep_set = set(int(i) for i in np.argsort(-importance)[:prescreen_top])
                # Rescue su_seeded synergistic operands the marginal-|phi| ranking would discard.
                keep_set |= {int(i) for i in _su_rescue_proxy_idx if 0 <= int(i) < n_proxy}
                keep = np.sort(np.fromiter(keep_set, dtype=np.int64))
                phi = np.ascontiguousarray(phi[:, keep])
                proxy_cols_kept = keep
                if phi_var is not None:
                    phi_var = np.ascontiguousarray(phi_var[:, keep])
                if unit_to_members is not None:
                    unit_to_members = [unit_to_members[i] for i in keep]
                else:
                    unit_to_members = [np.array([int(i)], dtype=np.int64) for i in keep]
                report["prescreen"] = dict(
                    kept=int(len(keep)), of=int(n_proxy),
                    su_rescued=int(len(_su_rescue_proxy_idx)))

        optimizer = self._resolve_optimizer(phi.shape[1])
        with _stage("search"):
            candidates = self._run_search(optimizer, phi, base, y_phi)

        # Interaction-aware coalition (#5): for interaction-heavy targets the main-effect sum can't
        # see a pair's joint signal (XOR partners have ~0 main effect). Add candidates ranked by the
        # SHAP-interaction coalition value and let honest re-validation arbitrate. Bounded to a small
        # proxy width (post pre-screen); tensor is O(P^2).
        if self.interaction_aware and phi.shape[1] <= self.max_interaction_features:
            from mlframe.feature_selection._shap_proxy_interactions import (
                compute_interaction_tensor, interaction_top_n)

            X_proxy_kept = X_proxy.iloc[:, list(proxy_cols_kept)]
            Phi, ibase = compute_interaction_tensor(
                model_template, X_proxy_kept, y_search, classification=self.classification, rng=self._rng)
            icands = interaction_top_n(
                Phi, ibase, y_phi, classification=self.classification, metric=self.metric,
                min_card=self.min_features, max_card=self.max_features, top_n=self.top_n,
                exhaustive_max=self.max_interaction_features)
            merged = {tuple(sorted(c)): l for l, c in candidates}
            for l, c in icands:
                merged.setdefault(tuple(sorted(c)), l)
            candidates = sorted(((l, c) for c, l in merged.items()), key=lambda t: t[0])
            report["interaction_aware"] = dict(applied=True, n_proxy=int(phi.shape[1]), n_interaction_candidates=len(icands))

        # su_seeded_interactions merge (#5b, OPT-IN): the CHEAP sparse alternative to
        # ``interaction_aware``'s O(P^2) tensor. The synergy screen + SNR gate ran above (operands
        # already rescued past the prescreen, so they are present in ``phi``). Here the interaction
        # objective runs on ONLY the surviving K pairs -- a sparse product-column augmentation, never
        # the dense P x P tensor. Each surviving product candidate is expanded into its two OPERAND
        # proxy-columns so the merged coalition lives in plain phi-column space (selecting the product
        # == recovering both operands); honest re-validation downstream still retrains only on real
        # columns. Runs at ANY phi width (no max_interaction_features gate). When the screen kept no
        # pair this whole block is skipped (the SNR-gate no-op).
        if self.su_seeded_interactions:
            from mlframe.feature_selection._shap_proxy_interactions import (
                sparse_interaction_candidates)

            with _stage("su_seeded_interactions"):
                # Map proxy-column NAMES -> phi-column indices (X_proxy_kept aligns 1:1 with phi).
                X_proxy_kept = X_proxy.iloc[:, list(proxy_cols_kept)]
                name_to_phi_idx = {str(c): i for i, c in enumerate(X_proxy_kept.columns)}
                # Keep only pairs whose BOTH operand proxy-columns survived into phi (the prescreen
                # rescue should guarantee this for every kept pair; the filter is defensive).
                usable_pairs = [
                    (s, a, b) for s, a, b in _su_kept_pairs
                    if str(a) in name_to_phi_idx and str(b) in name_to_phi_idx
                ]
                n_seed_cands = 0
                if usable_pairs:
                    # ISOLATED rng (NOT self._rng): keep the sparse-interaction refit off the shared
                    # RNG stream so the additive default stays byte-identical in the no-op case and the
                    # win-path stays reproducible.
                    icands_sparse, prod_to_operands = sparse_interaction_candidates(
                        model_template, X_proxy_kept, y_search, usable_pairs,
                        classification=self.classification, metric=self.metric,
                        min_card=self.min_features, max_card=self.max_features,
                        top_n=self.top_n,
                        rng=np.random.default_rng(int(self.random_state) + 7919))
                    # Expand augmented-index candidates back into phi-column space: a product index is
                    # replaced by its two operand phi-columns (selecting the product recovers both).
                    merged = {tuple(sorted(c)): l for l, c in candidates}
                    for l, c in icands_sparse:
                        expanded: set[int] = set()
                        for idx in c:
                            idx = int(idx)
                            if idx in prod_to_operands:
                                a_i, b_i = prod_to_operands[idx]
                                expanded.add(int(a_i))
                                expanded.add(int(b_i))
                            elif idx < phi.shape[1]:
                                expanded.add(idx)
                        if not expanded:
                            continue
                        key = tuple(sorted(expanded))
                        if key not in merged or l < merged[key]:
                            merged[key] = l
                            n_seed_cands += 1
                    # Always inject the bare operand pair of EVERY surviving synergistic pair as a
                    # candidate coalition (so even when the augmented search prefers larger subsets,
                    # the minimal interacting pair is still revalidated honestly on real columns).
                    for syn, col_a, col_b in usable_pairs:
                        key = tuple(sorted((name_to_phi_idx[str(col_a)], name_to_phi_idx[str(col_b)])))
                        merged.setdefault(key, float(candidates[0][0]) if candidates else 0.0)
                    candidates = sorted(((l, c) for c, l in merged.items()), key=lambda t: t[0])
                report["su_seeded_interactions"] = dict(
                    applied=True,
                    n_proxy=int(phi.shape[1]),
                    n_kept_pairs=len(_su_kept_pairs),
                    n_usable_pairs=len(usable_pairs),
                    n_seed_candidates=int(n_seed_cands),
                    kept_pairs=[(round(float(s), 6), str(a), str(b))
                                for s, a, b in _su_kept_pairs],
                    snr_gate=round(float(_su_screen_info.get("gate", float("nan"))), 6),
                    best_synergy=round(float(_su_screen_info.get("best_synergy", float("nan"))), 6),
                    n_screened_cols=int(_su_screen_info.get("n_screened_cols", 0)),
                    n_pairs=int(_su_screen_info.get("n_pairs", 0)))

        # min_selected_ratio guard: the proxy degrades for small subsets (the <50% wall). Ratio is in
        # proxy-column space (units/pre-screened columns).
        n_proxy_cols = phi.shape[1]
        if self.min_selected_ratio > 0:
            filtered = [(l, c) for l, c in candidates if len(c) / n_proxy_cols >= self.min_selected_ratio]
            candidates = filtered or candidates  # never return empty
        if not candidates:
            raise RuntimeError("ShapProxiedFS: search produced no candidate subsets.")

        report.update(optimizer=optimizer, n_candidates=len(candidates),
                      proxy_best=dict(features=tuple(candidates[0][1]), proxy_loss=candidates[0][0]))

        # One honest-retrain memo shared across trust guard, re-validation, ablation, and within-cluster
        # refine: within this fit the train/holdout split + model + metric are fixed, so a retrain's
        # loss is determined by the (column subset, seed). seed=None fits (trust anchors, ablation,
        # refine) frequently repeat the SAME large subset (e.g. the chosen winner is retrained in BOTH
        # the ablation and as refine's starting base) -- the cache returns those identical floats
        # without a duplicate fit. Random-seeded re-validation fits get distinct seeds, never wrongly
        # merged. Numerically identical to the uncached path (deterministic model on fixed data).
        from mlframe.feature_selection._shap_proxy_revalidate import HonestLossCache

        honest_cache = HonestLossCache()
        # iter80: extend iter79's disk-cache wiring through the honest-retrain stages
        # (proxy_trust_guard + revalidate_top_n + active_learning_revalidate). The disk cache adds a
        # cross-process layer underneath the in-memory ``honest_cache``: within-fit reuse stays in
        # RAM (HonestLossCache), repeat-fit reuse hits disk (DiskCache keyed on (X_search, y_search,
        # X_holdout, y_holdout, cols, template, seed, cap)). ``cache_dir=None`` (default) keeps the
        # legacy in-memory-only contract bit-identical. See ``_shap_proxy_revalidate.disk_cache_dir``
        # docstrings for the cache-key composition + best-effort failure policy.
        rv = dict(classification=self.classification, metric=self.metric, n_jobs=self.n_jobs,
                  unit_to_members=unit_to_members, cache=honest_cache,
                  inner_n_jobs_cap=self.inner_n_jobs_cap,
                  disk_cache_dir=self.cache_dir)

        # Proxy-trust diagnostic (proxy ranks units; honest retrains on member columns).
        if self.trust_guard:
            from mlframe.feature_selection._shap_proxy_revalidate import proxy_trust_guard

            # Stratified-anchor prior (opt-in via ``trust_guard_stratified_anchors``): when the
            # prefilter cached an F-score vector (two_stage / univariate paths), aggregate it into
            # UNIT space so the trust-guard sampler over-samples quality columns instead of drowning
            # in the noise tail. F-scores live in ORIGINAL column space (length n_features);
            # unit_to_members[u] -> WORKING-frame positions (post-prefilter); working_cols maps
            # working -> original. Per unit, take the MEAN F across its members (proxy for "is this
            # unit anchored by informative columns?"); singletons reduce to the member's own F.
            # Falls through to None (uniform sampler) when the prefilter didn't cache F-scores
            # (model / fast_model / gpu_model paths) OR the opt-in is OFF (the default; see
            # iter14-bench-attempt-rejected note in ``__init__``: the lever didn't pay at width=6000
            # because the post-two_stage cohort is already noise-filtered, so concentrating anchors
            # further compresses the spread the Spearman signal needs). Always-safe: misalignment is
            # detected inside ``proxy_trust_guard`` and degrades to uniform with a warning.
            unit_f_scores = None
            if self.trust_guard_stratified_anchors:
                from mlframe.feature_selection._shap_proxy_prefilter import get_cached_f_scores

                f_scores_orig = get_cached_f_scores(report.get("prefilter"))
                if f_scores_orig is not None:
                    try:
                        f_working = np.asarray(f_scores_orig, dtype=np.float64)[np.asarray(working_cols)]
                    except (IndexError, TypeError):
                        f_working = None
                    if f_working is not None:
                        n_units = phi.shape[1]
                        if unit_to_members is None:
                            if f_working.shape[0] == n_units:
                                unit_f_scores = f_working
                        else:
                            if all(int(m) < f_working.shape[0] for u in unit_to_members for m in u):
                                unit_f_scores = np.array(
                                    [float(np.mean(f_working[np.asarray(u, dtype=np.int64)]))
                                     for u in unit_to_members], dtype=np.float64)
            # iter18: resolve fidelity_floor / spearman_floor (deprecated alias). Supplying both
            # at the facade level is an error; supplying only the legacy name emits a
            # DeprecationWarning and copies the value through. ``None`` is the "unset" sentinel for
            # BOTH floors, so the both-set conflict is detected by ``fidelity_floor is not None``
            # (an explicit ``fidelity_floor=0.5`` is no longer mistaken for the default). The unset
            # ``fidelity_floor`` resolves to the effective 0.5 default.
            effective_floor = self.fidelity_floor if self.fidelity_floor is not None else 0.5
            if self.spearman_floor is not None:
                import warnings
                if self.fidelity_floor is not None:
                    raise ValueError(
                        "ShapProxiedFS: set either `fidelity_floor` (new name) or `spearman_floor` "
                        "(deprecated alias), not both.")
                warnings.warn(
                    "`ShapProxiedFS(spearman_floor=...)` is deprecated since iter18; use "
                    "`fidelity_floor=...` (same semantics). The kwarg name was inherited from the "
                    "iter15 raw-Spearman gate but the gate has been the composite "
                    "`proxy_fidelity_score` since iter16.",
                    DeprecationWarning, stacklevel=2,
                )
                effective_floor = self.spearman_floor
            with _stage("trust_guard"):
                report["trust"] = proxy_trust_guard(
                    phi, base, y_phi, model_template, X_search, X_hold, y_hold,
                    n_anchors=self.n_anchors, rng=self._rng, min_card=self.min_features,
                    max_card=self.max_features, fidelity_floor=effective_floor,
                    n_estimators_cap=self.trust_guard_n_estimators,
                    unit_f_scores=unit_f_scores,
                    anchor_uniform_tail_frac=self.trust_guard_uniform_tail_frac,
                    cardinality_dist=str(self.trust_guard_cardinality_dist).lower(),
                    zipf_alpha=self.trust_guard_zipf_alpha,
                    fidelity_weights=(float(self.trust_guard_fidelity_weights[0]),
                                       float(self.trust_guard_fidelity_weights[1])),
                    trustworthy_metric=str(self.trust_guard_metric).lower(), **rv)

        # Unified candidate re-ranking before the expensive top-N honest retrains: order by the
        # corrector's predicted honest loss (#3/#6, falls back to raw proxy) PLUS an uncertainty
        # penalty (#7). Focuses the retrain budget on subsets that are honestly-best AND stable.
        score = np.array([c[0] for c in candidates], dtype=np.float64)  # raw proxy loss
        if self.use_bias_corrector and self.trust_guard and report.get("trust", {}).get("_corrector_data"):
            from mlframe.feature_selection._shap_proxy_calibrate import fit_proxy_corrector, subset_redundancy_many

            cd = report["trust"]["_corrector_data"]
            corrector = fit_proxy_corrector(cd["proxy"], cd["honest"], cd["cards"], cd["redund"])
            if not corrector.fallback:
                cards = np.array([len(c[1]) for c in candidates], dtype=np.float64)
                redund = subset_redundancy_many(phi, [c[1] for c in candidates])
                score = corrector.predict(score, cards, redund)
                report["bias_corrector"] = dict(applied=True, n_anchors=len(cd["proxy"]))
        if self.uncertainty_penalty > 0 and phi_var is not None:
            from mlframe.feature_selection._shap_proxy_objective import subset_uncertainty_many

            unc = subset_uncertainty_many(phi_var, [c[1] for c in candidates])
            score = score + self.uncertainty_penalty * unc
            report["uncertainty"] = dict(applied=True, penalty=float(self.uncertainty_penalty))
        order = np.argsort(score, kind="stable")
        candidates = [candidates[i] for i in order]
        score = score[order]  # keep aligned with candidates for downstream UCB consumption

        # MMR de-duplication of the corrector-sorted candidate list BEFORE revalidate (iter50).
        # At wide regimes (n_features>=20000) top_n=20 candidates are near-duplicate unions of the
        # same SHAP-aware stage-B picks; UCB short-circuits the proxy-loss tail but still pays
        # per-batch dispatch on the redundant subsets. Greedy keep-if-Jaccard-distance>tau in
        # corrector-sorted order; dropped candidates are corrector-equivalent to a retained one
        # and would not pass the parsimony band as a meaningful improvement. Disabled by default
        # below width 20000.
        mmr_tau = self._resolve_revalidation_mmr_jaccard_threshold(n_features)
        if mmr_tau is not None and self.revalidate and len(candidates) > 1:
            kept_idx = self._mmr_filter_by_jaccard(candidates, float(mmr_tau))
            n_total = len(candidates)
            if len(kept_idx) < n_total:
                candidates = [candidates[i] for i in kept_idx]
                score = score[np.asarray(kept_idx, dtype=np.int64)]
                report["revalidation_mmr"] = dict(applied=True, tau=float(mmr_tau),
                                                  n_kept=len(kept_idx), n_total=int(n_total))
            else:
                report["revalidation_mmr"] = dict(applied=True, tau=float(mmr_tau),
                                                  n_kept=int(n_total), n_total=int(n_total))

        # Expose the ranked candidate subsets (expanded to feature names) so downstream patterns
        # (e.g. proposal-generator seeding RFECV/genetic honest search) can consume them.
        def _cand_names(idx):
            if unit_to_members is not None:
                cols = sorted({int(c) for u in idx for c in unit_to_members[int(u)]})
            else:
                cols = sorted(int(i) for i in idx)
            return [str(self.feature_names_in_[i]) for i in cols]

        report["candidates"] = [dict(proxy_loss=float(l), features=_cand_names(c))
                                for l, c in candidates[: self.top_n]]

        # Honest re-validation of the top-N on the disjoint holdout (active-learning variant when the
        # corrector anchors are available, else the static top-N retrain).
        _reval_budget_skip = self.revalidate and _budget_exhausted()
        if _reval_budget_skip:
            report["budget_skipped"] = dict(phase="revalidation", max_runtime_mins=_budget_max_mins, stop_file=_budget_stop_file)
            if self.verbose:
                logger.info("ShapProxiedFS: budget/stop reached; skipping honest revalidation, finalising with proxy-best subset.")
        if self.revalidate and not _reval_budget_skip:
            cdata = report.get("trust", {}).get("_corrector_data")
            with _stage("revalidation"):
                if self.active_learning and cdata:
                    from mlframe.feature_selection._shap_proxy_revalidate import active_learning_revalidate

                    budget = self.active_learning_budget or self.top_n
                    best_idx, ranked, n_eval = active_learning_revalidate(
                        candidates, model_template, X_search, y_search, X_hold, y_hold,
                        corrector_data=cdata, phi=phi, budget=budget, n_models=self.n_revalidation_models,
                        parsimony_tol=self.parsimony_tol, rng=self._rng,
                        revalidation_n_estimators=self.revalidation_n_estimators, **rv)
                    report["revalidation"] = dict(ranked=ranked[: self.top_n],
                                                  active_learning=dict(n_evaluated=n_eval, budget=budget))
                else:
                    from mlframe.feature_selection._shap_proxy_revalidate import revalidate_top_n

                    best_idx, ranked, baseline = revalidate_top_n(
                        candidates, model_template, X_search, y_search, X_hold, y_hold,
                        n_models=self.n_revalidation_models, lambda_stab=self.lambda_stab,
                        parsimony_tol=self.parsimony_tol, rng=self._rng,
                        revalidation_n_estimators=self.revalidation_n_estimators,
                        ucb_enabled=self.revalidation_ucb_enabled,
                        ucb_min_eval_size=self.revalidation_ucb_min_eval_size,
                        ucb_slack=self.revalidation_ucb_slack,
                        ucb_stdev_multiplier=self._resolve_revalidation_ucb_stdev_multiplier(
                            n_features),
                        adaptive_n_models=self.revalidation_adaptive_n_models,
                        candidate_score=score, **rv)
                    report["revalidation"] = dict(ranked=ranked[: self.top_n], random_baseline=baseline)
        else:
            best_idx = tuple(candidates[0][1])

        # Importance-top-k ablation (unique-value gate vs plain SHAP importance).
        if self.run_importance_ablation and best_idx and not _budget_exhausted():
            from mlframe.feature_selection._shap_proxy_revalidate import importance_topk_ablation

            with _stage("importance_ablation"):
                report["importance_ablation"] = importance_topk_ablation(
                    phi, best_idx, model_template, X_search, y_search, X_hold, y_hold,
                    classification=self.classification, metric=self.metric, unit_to_members=unit_to_members,
                    cache=honest_cache, disk_cache_dir=self.cache_dir)

        # Expand best proxy subset -> original member columns, then optionally prune redundant members.
        if unit_to_members is not None:
            member_cols = sorted({int(c) for u in best_idx for c in unit_to_members[int(u)]})
        else:
            member_cols = sorted(int(i) for i in best_idx)
        if self.within_cluster_refine and unit_to_members is not None and len(member_cols) > 1 and not _budget_exhausted():
            from mlframe.feature_selection._shap_proxy_objective import resolve_metric
            from mlframe.feature_selection._shap_proxy_revalidate import (
                _honest_loss, _open_disk_cache, within_cluster_refine,
            )

            with _stage("within_cluster_refine"):
                # Pass the per-unit member lists so refine can collapse each cluster to a single
                # representative in ONE parallel batch (O(sum k_c) trials) instead of legacy
                # O(k^2) greedy drops. unit_to_members is in proxy-unit space; each chosen unit
                # contributes one group of member columns.
                member_groups = [
                    [int(c) for c in unit_to_members[int(u)]] for u in best_idx
                ]
                refined = within_cluster_refine(
                    member_cols, model_template, X_search, y_search, X_hold, y_hold,
                    classification=self.classification, metric=self.metric,
                    parsimony_tol=self.parsimony_tol, n_jobs=self.n_jobs, cache=honest_cache,
                    member_groups=member_groups, refine_n_estimators=self.refine_n_estimators,
                    ucb_enabled=self.refine_ucb_enabled,
                    ucb_min_eval_size=self.refine_ucb_min_eval_size,
                    ucb_slack=self.refine_ucb_slack,
                    ucb_stdev_multiplier=self.refine_ucb_stdev_multiplier,
                    inner_n_jobs_cap=self.inner_n_jobs_cap,
                    disk_cache_dir=self.cache_dir)
                # Final full-template re-evaluation of the ONE chosen subset (uncapped n_estimators).
                # Refine's ranking trials use a cheaper capped booster (~100 trees) to decide WHICH
                # members to drop; the user-visible quality bar (and any downstream report consumer)
                # should see this subset's loss at the SAME booster size the other guards used, so the
                # values are apples-to-apples. The cache lookup is the full-template namespace (no
                # template_id), so this hits any prior pipeline retrain of the same subset (e.g. when
                # refine made no drops, this is a cache hit of the union retrain done elsewhere).
                refine_info = dict(before=len(member_cols), after=len(refined))
                if refined:
                    # iter81: the full-template re-eval of the refined subset frequently hits the
                    # disk cache too -- the same (cols, template, cap=None) tuple was retrained as
                    # the revalidation winner upstream, so a warm-cache lookup avoids an extra fit.
                    refine_info["honest_loss_full"] = float(_honest_loss(
                        model_template, X_search, y_search, X_hold, y_hold, list(refined),
                        self.classification, resolve_metric(self.classification, self.metric),
                        cache=honest_cache, disk_cache=_open_disk_cache(self.cache_dir)))
                report["within_cluster_refine"] = refine_info
                member_cols = refined

        # Expose sklearn contract: map working-space member columns back to ORIGINAL indices (the
        # pre-filter may have restricted the working set), names in INPUT column order.
        best_set = {int(working_cols[i]) for i in member_cols}
        self.selected_features_ = [c for i, c in enumerate(self.feature_names_in_) if i in best_set]
        self.support_ = np.array([i in best_set for i in range(n_features)], dtype=bool)
        self.shap_proxy_report_ = report
        if self.verbose:
            logger.info("ShapProxiedFS: optimizer=%s selected %d/%d features: %s",
                        optimizer, len(self.selected_features_), n_features, self.selected_features_)
        return self
