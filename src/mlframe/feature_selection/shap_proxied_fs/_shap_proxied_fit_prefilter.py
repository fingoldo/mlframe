"""Post-prefilter holdout-materialisation + correlated-feature clustering block for
:meth:`ShapProxiedFitMixin.fit`.

Carved out of ``_shap_proxied_fit.py`` (Tier E) to keep that module under the 1k LOC ceiling.
This is a VERBATIM lift of the post-prefilter holdout-materialisation + clustering block: it
materialises the deferred disjoint holdout at the narrow post-prefilter column count, then runs the
optional correlated-feature clustering (SU / Pearson dispatch + unit-matrix build). It reads the
fit-local state it needs as explicit keyword params (no closure capture; the ``_stage`` timing
context manager is passed in) and returns ``(X_hold, X_proxy, unit_to_members)`` -- the values the
fit continuation consumes. ``self`` carries the constructor knobs + the ``_deferred_holdout`` scratch
exactly as before, and ``report`` is mutated in place, so behaviour is byte-for-byte identical.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def materialise_holdout_and_cluster(*, self, working_cols, n_features, _precomputed_aligned,
                                    X_search, report, _stage):
    """Materialise the deferred holdout at the working-column count, then optionally cluster.

    Returns ``(X_hold, X_proxy, unit_to_members)``. ``report`` is mutated in place and
    ``self._deferred_holdout`` is cleared, matching the original inline behaviour.
    """
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
        from mlframe.feature_selection.shap_proxied_fs._shap_proxy_cluster import (
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
                from mlframe.feature_selection.shap_proxied_fs._shap_proxy_cluster_su import (
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

    return X_hold, X_proxy, unit_to_members
