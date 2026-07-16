"""Fit-time helpers (stability outer loop, multioutput, identity shortcut, resample, setstate, summary, export).

Pure move from ``_mrmr_class`` into a mixin. ``fit`` itself stays on the ``MRMR`` class body; these helpers it
calls resolve through the MRO on the concrete ``MRMR`` instance. Class attributes they read
(``_SETSTATE_LEGACY_OVERRIDES``) and sibling classmethods (``_ctor_defaults`` / ``_effective_random_seed``)
stay reachable via the MRO.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pandas as pd

from sklearn.base import clone

from ._mrmr_class_shared import _mrmr_y_columns

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")


class _MRMRFitHelpersMixin:
    """Fit-time helpers for :class:`MRMR` (see module docstring).

    ``get_params`` / ``fit`` resolve via the ``BaseEstimator`` / ``MRMR`` MRO; ``random_seed`` /
    ``_effective_random_seed`` / ``get_feature_names_out`` resolve via the concrete ``MRMR`` /
    ``_MRMRConfigMixin`` MRO. Declared here type-checking-only (not assigned/defined at runtime, so the
    MRO's real implementations are never shadowed) so mypy resolves them on ``self``.
    """

    if TYPE_CHECKING:
        get_params: Any
        fit: Any
        random_seed: Any
        get_feature_names_out: Any

        def _effective_random_seed(self) -> Any:
            """Type-checking-only stub; resolves at runtime via the concrete ``MRMR`` MRO."""
            ...

    # opt-in stability-selection outer-loop wrapper.
    # Routes to Faletto-Bien 2022 Cluster Stability Selection or
    # Shah-Samworth 2013 Complementary Pairs Stability when
    # ``stability_selection_method != 'classic'``. The classic path falls
    # through to the legacy ``self.fit`` body.
    def _stability_outer_fit(self, X, y, **fit_kwargs):
        """Run the opt-in stability-selection outer loop (cluster or complementary-pairs) when configured, persisting
        ``support_``/``stability_freq_``/``stability_info_`` and returning ``self``; returns ``None`` for the
        ``'classic'`` method so the caller falls through to the legacy ``fit`` body."""
        method = getattr(self, "stability_selection_method", "classic")
        if method == "classic":
            return None  # fall through to legacy fit
        from .._stability_cluster import (
            cluster_stability_selection,
            complementary_pairs_stability,
        )
        import pandas as pd
        # Pass the FRAME (not to_numpy()) so per-column dtypes survive: a mixed numeric+categorical frame -> to_numpy() is an object array that
        # (a) crashed the cluster correlation's float64 coercion and (b) would feed the bootstrap sub-MRMR all-object columns. The stability helpers
        # cluster only the numeric columns (categoricals -> singletons) and hand the sub-selector dtype-preserved rows.
        X_df = X if hasattr(X, "iloc") else pd.DataFrame(np.asarray(X))
        y_arr = (y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)).ravel()
        feature_names = list(X_df.columns)
        # Computed ONCE outside the bootstrap/complementary-pairs loop (perf audit finding #9,
        # 2026-07-17): every replicate previously re-ran ``self.get_params()`` (a full ~300-key
        # dict) plus a fresh filter comprehension, even though the filtered base params are
        # identical across all ``stability_n_bootstrap`` (default 50) replicates -- only the
        # subsampled (X_sub, y_sub) differ per replicate, not the params.
        _sub_base_params = {
            k: v for k, v in self.get_params().items()
            if k not in (
                "stability_selection_method",
                "stability_selection_corr_threshold",
                "uaed_auto_size",
                "cmi_perm_stop",
                "cpt_test",
            )
        }
        _sub_base_params["stability_selection_method"] = "classic"
        _sub_base_params["verbose"] = 0

        def _inner_selector(X_sub, y_sub):
            """Fit a fresh classic-mode sibling MRMR on a bootstrap/pair row-subset and return its selected column indices (empty array if the sub-fit produced no support)."""
            # X_sub is a dtype-preserved frame row-subset (from .iloc) -> reset its index so it aligns with the default-indexed y_sub below; the
            # no-frame fallback (ndarray subset) wraps as before. (Wrapping a frame via pd.DataFrame(frame) would keep its non-default index and
            # mis-align against y_sub.)
            X_sub_df = X_sub.reset_index(drop=True) if hasattr(X_sub, "iloc") else pd.DataFrame(X_sub, columns=feature_names)
            y_sub_s = pd.Series(np.asarray(y_sub), name="y")
            # Use a fresh sibling instance with classic method to avoid
            # recursion AND drop bootstrap-incompatible settings.
            sub = type(self)(**_sub_base_params)
            sub.fit(X_sub_df, y_sub_s)
            if not hasattr(sub, "support_") or sub.support_ is None:
                return np.asarray([], dtype=np.int64)
            return np.asarray(sub.support_, dtype=np.int64)

        if method == "cluster":
            corr_thr = float(getattr(self, "stability_selection_corr_threshold", 0.8))
            sel, freq, info = cluster_stability_selection(
                X_df, y_arr, _inner_selector,
                n_bootstrap=int(getattr(self, "stability_n_bootstrap", 50)),
                pi_threshold=float(getattr(self, "stability_pi_threshold", 0.6)),
                corr_threshold=corr_thr,
                rng_seed=int(self.random_seed or 0),
            )
        elif method == "complementary_pairs":
            sel, freq, info = complementary_pairs_stability(
                X_df, y_arr, _inner_selector,
                n_pairs=int(getattr(self, "stability_n_bootstrap", 50)),
                pi_threshold=float(getattr(self, "stability_pi_threshold", 0.6)),
                rng_seed=int(self.random_seed or 0),
            )
        else:
            raise ValueError(f"unknown stability_selection_method={method!r}")

        # Persist the standard MRMR public-API attributes from the chosen set.
        self.support_ = np.asarray(sel, dtype=np.int64)
        self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        self.n_features_in_ = len(feature_names)
        self.n_features_ = int(self.support_.size)
        self.stability_freq_ = freq
        self.stability_info_ = info
        return self

    # Pickle BC: old MRMR pickles lacking newer attributes resurface with the legacy defaults injected.
    def _maybe_resample_for_sample_weight(self, X, y, sample_weight: np.ndarray | None):
        """When ``sample_weight`` is provided AND not effectively uniform, draw n=len(X) rows with replacement
        using probabilities w_i / sum(w). The resampled empirical bincount approximates the weighted bincount
        (np.bincount(x, weights=w) up to MC noise), so MI relevance / redundancy estimated downstream from
        binned joint histograms becomes weight-aware without touching info_theory / screen internals.
        Returns (X, y) unchanged when sample_weight is None / all-equal (preserves byte-for-byte legacy path
        and lets the FS cache reuse a single fit across uniform-weight callers)."""
        if sample_weight is None:
            return X, y
        sw = np.asarray(sample_weight, dtype=np.float64)
        if sw.ndim != 1:
            raise ValueError(f"MRMR.fit sample_weight must be 1-D, got shape {sw.shape}")
        n_rows = X.shape[0]
        if sw.shape[0] != n_rows:
            raise ValueError(f"MRMR.fit sample_weight length {sw.shape[0]} != n_rows {n_rows}")
        if not np.all(np.isfinite(sw)) or (sw < 0).any():
            raise ValueError("MRMR.fit sample_weight must be finite and non-negative")
        total = float(sw.sum())
        if total <= 0:
            raise ValueError("MRMR.fit sample_weight sums to zero")
        # Uniform -> nothing to do (preserves bit-exact legacy + cache reuse).
        if float(sw.max() - sw.min()) <= 1e-12 * max(1.0, abs(float(sw.mean()))):
            return X, y
        # Reproducible by default: follow the module-wide convention (``int(seed or 0)`` -> None maps to a
        # DETERMINISTIC 0, not OS entropy) so that two fits with the same (default) seed and the same
        # sample_weight draw the SAME resample. ``_effective_random_seed`` honours the ``random_state``
        # fallback alias. A caller wanting an independent draw passes a distinct ``random_seed``.
        rng = np.random.default_rng(int(self._effective_random_seed() or 0))
        probs = sw / total
        idx = rng.choice(n_rows, size=n_rows, replace=True, p=probs)
        # iloc preserves dtypes / category metadata; works on pandas + polars + numpy. polars DataFrame has no
        # ``.take()`` (removed upstream); row-select via ``__getitem__``, which every polars version supports.
        try:
            import polars as _pl
            if isinstance(X, _pl.DataFrame):
                X_rs = X[idx.tolist()]
            elif isinstance(X, pd.DataFrame):
                X_rs = X.iloc[idx]
            else:
                X_rs = np.asarray(X)[idx]
        except ImportError:
            if isinstance(X, pd.DataFrame):
                X_rs = X.iloc[idx]
            else:
                X_rs = np.asarray(X)[idx]
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_rs = y.iloc[idx]
        else:
            y_rs = np.asarray(y)[idx]
        return X_rs, y_rs

    def _print_fit_summary(self) -> None:
        """Human-readable end-of-fit summary, printed to STDOUT when ``verbose>=1``.

        Why ``print`` and not ``logger``: MRMR's informative ``logger.info`` calls
        are swallowed in any script that never configures the ``logging`` module
        (the common case), while the tqdm progress bars write straight to stderr.
        The net effect was that a user running ``MRMR(verbose=1).fit(...)`` saw a
        wall of progress bars and NO statement of what was selected / engineered --
        the run looked like it did nothing even when directed FE recovered the
        signal. This summary is the one guaranteed-visible line of truth.

        Pure reporting: never mutates state, never raises (a summary bug must not
        fail a fit). Built from the already-populated ``fe_provenance_`` /
        ``support_`` / ``get_feature_names_out`` fitted attributes.
        """
        try:
            if not getattr(self, "verbose", 0):
                return
            names = [str(n) for n in self.get_feature_names_out()]
            eng = [n for n in names if "(" in n]
            n_raw = len(names) - len(eng)
            n_in = getattr(self, "n_features_in_", "?")
            print(f"\n[MRMR] selected {len(names)} feature(s) " f"({n_raw} raw + {len(eng)} engineered) from {n_in} input(s)")
            prov = getattr(self, "fe_provenance_", None)
            if prov is not None and hasattr(prov, "empty") and not prov.empty:
                disp_cols = [c for c in ("support_rank", "feature_name", "origin", "mrmr_gain") if c in prov.columns]
                disp = prov[disp_cols]
                if "support_rank" in disp.columns:
                    # Show in greedy selection order (rank 0 first), not the raw
                    # provenance-frame row order. sort_values always returns a NEW
                    # frame, so no separate .copy() is needed on this branch.
                    disp = disp.sort_values("support_rank", kind="stable")
                else:
                    # No reordering happens on this branch, so disp_cols'
                    # column-list selection is still possibly a pandas view of
                    # ``prov`` -- copy once here (only branch that needs it)
                    # before the in-place mrmr_gain formatting below.
                    disp = disp.copy()
                if "mrmr_gain" in disp.columns:
                    disp["mrmr_gain"] = disp["mrmr_gain"].map(lambda v: f"{float(v):.4f}" if pd.notna(v) else "")
                print(disp.to_string(index=False))
            else:
                print("  " + ", ".join(names))
            if eng:
                print(f"[MRMR] {len(eng)} engineered feature(s) discovered: " + ", ".join(eng))
            else:
                print("[MRMR] no engineered features survived the MI-prevalence gate " "(fe_min_engineered_mi_prevalence); selection is raw-only")
        except Exception as exc:
            logger.debug("mrmr: selection-summary print failed (diagnostic only): %r", exc, exc_info=True)

    def export_artifacts(self) -> dict:
        """Return the in-fit reusable intermediates as a dict for downstream selectors.

        Requires the estimator to have been constructed with
        ``retain_artifacts=True`` (off by default to preserve the legacy memory
        footprint) and to have been fitted. The returned dict carries
        Symmetric Uncertainty + direct MI vectors against y, plus -- when
        ``retain_bins=True`` -- the per-column binned arrays. Schema is defined
        in ``_mrmr_artifacts._ARTIFACT_SCHEMA``; consumers MUST tolerate missing
        optional keys for forward compat.

        The canonical consumer is ``ShapProxiedFS(precomputed=...)``: passing
        ``mrmr.export_artifacts()`` lets that selector skip its own univariate
        F-statistic pre-screen and rank by MRMR's SU(X_j, y) instead. The
        selected subset is unchanged for SU-vs-F-ranking-equivalent regimes;
        the win is wall-clock + a more cardinality-honest ranking on mixed-
        cardinality data.

        Raises
        ------
        ValueError
            If ``self.retain_artifacts`` is False (artifacts were not captured).
        sklearn.exceptions.NotFittedError
            If the estimator has not been fitted yet.
        """
        if not getattr(self, "retain_artifacts", False):
            raise ValueError(
                "MRMR.export_artifacts() requires retain_artifacts=True at construction time. "
                "Re-construct as MRMR(retain_artifacts=True, ...) and fit before exporting."
            )
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, ["support_"])
        artifacts = getattr(self, "_artifacts_", None)
        if not artifacts:
            # retain_artifacts=True was set but the in-fit capture did not
            # populate the dict -- likely a fit() path that bypassed
            # _fit_impl (identity shortcut, FIT_CACHE hit on a legacy
            # cached instance, stability-selection outer loop). Surface a
            # clear error so the caller can adjust the pipeline rather than
            # silently consuming an empty dict.
            raise ValueError(
                "MRMR.export_artifacts(): retain_artifacts=True but the in-fit capture did "
                "not populate self._artifacts_. The fit may have hit the identity-shortcut "
                "cache or a pre-retain_artifacts cached instance; refit with "
                "MRMR._FIT_CACHE.clear() and mrmr_skip_when_prior_was_identity=False."
            )
        return dict(artifacts)

    def _fit_identity_shortcut(self, X) -> None:
        """Populate the fit-result attributes as if MRMR returned the input X unchanged.

        Used by the cross-target identity cache: when a previous fit on the SAME X returned identity (all input columns selected, zero engineered features), subsequent calls with a different y can skip the entire FE pipeline since the only y-dependent thing -- the selected feature subset -- is forced to "all input columns".
        """
        n_cols = X.shape[1] if X.ndim > 1 else 1
        self.support_ = np.arange(n_cols, dtype=np.int64)
        # 1 fix (loop iter 35): the prior expression
        # ``X.columns.tolist() if hasattr(X.columns, "tolist") else
        # list(X.columns) if hasattr(X, "columns") else [...]`` was a
        # mis-parenthesised ternary. Python parses it as
        # ``A if B1 else (C if B2 else E)``, evaluating ``B1 =
        # hasattr(X.columns, "tolist")`` BEFORE the outer ``B2 =
        # hasattr(X, "columns")`` guard. The inner ``X.columns`` access
        # raised AttributeError on ndarray X, so the identity-shortcut
        # cache-hit path (opt-in via ``mrmr_skip_when_prior_was_identity``)
        # crashed on every ndarray fit instead of short-circuiting.
        if hasattr(X, "columns"):
            _cols = X.columns
            _names = _cols.tolist() if hasattr(_cols, "tolist") else list(_cols)
        else:
            _names = [f"f{i}" for i in range(n_cols)]
        self.feature_names_in_ = np.asarray(_names, dtype=object)
        self._engineered_features_: list = []
        self._engineered_recipes_: list = []  # list invariant (matches the full-fit paths); consumers iterate it as a list
        self.n_features_in_ = int(n_cols)
        self.n_features_ = int(n_cols)
        self.fallback_used_ = False
        # 1: set DCD/diagnostic fitted attrs to safe
        # defaults so the identity shortcut produces a
        # fitted-state-complete estimator (matches full-fit attribute
        # surface). Without these the cache-replay tests and
        # downstream consumers that introspect ``sel.dcd_`` /
        # ``sel.mrmr_gains_`` /``sel.friend_graph_`` /
        # ``sel.cluster_aggregate_`` blow up on the shortcut path.
        self.dcd_ = None
        # identity-shortcut path must also expose the
        # ``cluster_members_`` attribute (None when DCD was disabled or did
        # not run) so introspection code paths don't AttributeError.
        self.cluster_members_ = None
        # hierarchical post-hoc cluster map. Empty
        # dict default (matches "DCD ran but found no super-structure" --
        # meaningfully different from None, which would mean DCD disabled).
        # Identity shortcut bypasses DCD entirely, so the empty default is
        # the correct attribute-complete marker.
        self.cluster_hierarchy_: dict = {}
        self.mrmr_gains_ = np.array([], dtype=np.float64)
        self.friend_graph_ = None
        self.cluster_aggregate_ = None
        self.ran_out_of_time_ = False
        self.provenance_: Optional[dict] = None
        self._feature_names_in_synthesized_ = not hasattr(X, "columns")
        # Mark for transform() to know we're in shortcut state. Some downstream code looks at .signature; safe-default to a stable string.
        self.signature: Optional[str] = f"_mrmr_identity_shortcut_n{n_cols}"

    def _fit_multioutput(self, X, y, groups, sample_weight, strategy: str, fit_params):
        """Fit one single-target MRMR per output column of a 2D ``y`` and aggregate the selected RAW columns via ``strategy`` ('union'/'intersect').

        Sets the standard fitted attributes (``support_`` as integer column indices, ``feature_names_in_``, ``n_features_in_``, ``n_features_``)
        plus ``multioutput_supports_`` (per-column selected raw-feature-name lists) and ``multioutput_strategy_``. Engineered features are not
        unioned -- their per-column recipes differ; this path recovers the RAW genuine features that the merged-target greedy under-selected.
        Memory: each sub-fit receives the SAME X (no copy) with a single 1D target column, mirroring RFECV's multioutput union.
        """
        feature_names = list(X.columns) if hasattr(X, "columns") else [f"f{i}" for i in range(X.shape[1])]
        name_to_idx = {str(n): i for i, n in enumerate(feature_names)}
        n_features = len(feature_names)

        per_column_selected: dict[str, list] = {}
        for label, y_col in _mrmr_y_columns(y):
            sub = clone(self)
            sub.multioutput_strategy = None  # the per-column sub-fit is single-target; force the legacy path so it does not recurse.
            sub.fit(X, y_col, groups=groups, sample_weight=sample_weight, **(fit_params or {}))
            sub_names = [str(feature_names[i]) for i in np.asarray(sub.support_, dtype=np.intp)]
            per_column_selected[label] = sub_names

        if not per_column_selected:
            raise ValueError("MRMR multioutput: y has no output columns to fit.")

        sets = [set(v) for v in per_column_selected.values()]
        aggregated = set().union(*sets) if strategy == "union" else set.intersection(*sets)
        selected_in_order = [str(n) for n in feature_names if str(n) in aggregated]

        self.support_ = np.asarray([name_to_idx[n] for n in selected_in_order], dtype=np.int64)
        self.feature_names_in_ = np.asarray([str(n) for n in feature_names], dtype=object)
        self.n_features_in_ = n_features
        self.n_features_ = int(self.support_.size)
        self._engineered_features_ = []
        self._engineered_recipes_ = []
        self.multioutput_supports_ = per_column_selected
        self.multioutput_strategy_ = strategy
        # No in-object skip signature for the multioutput path: this method always re-runs the per-column sub-fits (it never consults a content
        # signature), and the single-target skip check compares a 6-tuple ``(shape, shape, y_hash, x_hash, cols, params)`` against ``self.signature``.
        # ``None`` makes that comparison False BY CONSTRUCTION (a later single-target fit on this instance always refits), rather than relying on
        # the str-vs-tuple type mismatch of a content-free ``f"_mrmr_multioutput_..."`` string to never collide.
        self.signature = None
        self._fit_sample_weight_ = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        # This path returns before the legacy single-fit body runs, so the standard
        # single-target diagnostic surface (degenerate-column audit, provenance_,
        # fe_provenance_, fe_rejection_ledger_) is otherwise never populated -- these
        # documented public attributes would simply be absent, raising AttributeError
        # only in multi-output mode.
        try:
            from .._mrmr_degenerate import audit_degenerate_columns
            self.degenerate_columns_ = audit_degenerate_columns(X)
        except Exception as exc:
            logger.debug("mrmr multioutput: degenerate-column audit failed (diagnostic only): %r", exc, exc_info=True)
            self.degenerate_columns_ = {}
        _seed_resolved = getattr(self, "random_seed", None)
        if _seed_resolved is None:
            _seed_resolved = getattr(self, "random_state", None)
        self.provenance_ = {
            "step": "mrmr_multioutput",
            "source": "train_only",
            "n_rows": int(X.shape[0]) if hasattr(X, "shape") else None,
            "seed": int(_seed_resolved) if _seed_resolved is not None else None,
        }
        from .._mrmr_fe_provenance import populate_fe_provenance
        from .._fe_rejection_ledger import populate_fe_rejection_ledger
        populate_fe_provenance(self)
        populate_fe_rejection_ledger(self)
        return self
