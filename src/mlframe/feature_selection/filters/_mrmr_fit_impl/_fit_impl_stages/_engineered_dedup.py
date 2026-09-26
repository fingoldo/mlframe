"""Cross-stage deduplication of engineered columns in ``_fit_impl``."""

from __future__ import annotations

# --- imports (managed) ---
import numpy as np
import pandas as pd
from mlframe.feature_selection.filters._mrmr_fit_impl._fe_roster_attrs import FE_ROSTER_ATTRS
from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_core import logger
from mlframe.feature_selection.filters._y_encoding import encode_y_for_classif_mi

# --- end imports ---

_DEPENDENCY_FAMILIES: tuple[str, ...] = (
    "hybrid_orth",
    "mi_greedy",
    "kfold_te",
    "binned_agg",
    "count_enc",
    "freq_enc",
    "cat_num",
    "miss_ind",
    "miss_cnt",
    "miss_pat",
    "ratio",
    "log_ratio",
    "grouped_delta",
    "lagged_diff",
    "grouped_agg",
    "composite_group_agg",
    "grouped_quantile",
    "cat_pair",
    "cat_triple",
    "numeric_decompose",
    "modular",
    "pairwise_modular",
    "integer_lattice",
    "row_argmax",
    "conditional_gate",
    "group_distance",
    "rare_category",
    "conditional_residual",
    "conditional_dispersion",
    "wavelet",
    "rankgauss",
    "temporal_agg",
    "conditional_quantile_rank",
    "ordinal_pattern",
    "random_fourier",
    "sir_direction",
    "lof",
    "mahalanobis_density",
)
"""Recipe families whose surviving recipes' ``src_names`` protect a column from the cross-stage dedup."""

_PRUNED_FAMILIES: tuple[str, ...] = (
    "hybrid_orth",
    "mi_greedy",
    "kfold_te",
    "count_enc",
    "freq_enc",
    "cat_num",
    "miss_ind",
    "miss_cnt",
    "miss_pat",
    "ratio",
    "log_ratio",
    "grouped_delta",
    "lagged_diff",
    "grouped_agg",
    "composite_group_agg",
    "grouped_quantile",
    "cat_pair",
    "cat_triple",
    "numeric_decompose",
    "modular",
    "pairwise_modular",
    "integer_lattice",
    "row_argmax",
    "conditional_gate",
    "group_distance",
    "rare_category",
    "conditional_residual",
    "conditional_dispersion",
    "conditional_quantile_rank",
    "ordinal_pattern",
    "random_fourier",
    "sir_direction",
    "lof",
    "mahalanobis_density",
    "wavelet",
    "rankgauss",
    "temporal_agg",
)
"""Recipe registries a pruned column is removed from (``binned_agg`` is not pruned here)."""



def _dedup_engineered_across_stages(self, _y_np, X, recipes, verbose):
    """Cross-stage dedup of engineered columns, keeping the member of a near-duplicate cluster with the most information about y."""
    _eng_cols_appended_raw = list(self.hybrid_orth_features_ or []) + list(self.mi_greedy_features_ or [])
    _eng_seen: set[str] = set()
    _eng_cols_appended = [_c for _c in _eng_cols_appended_raw if not (_c in _eng_seen or _eng_seen.add(_c))]  # type: ignore[func-returns-value]  # intentional order-preserving-dedup idiom: set.add()'s None return is used as the falsy side of `or`
    # ADAPTIVE-FOURIER columns are NEVER pruned by the cross-stage dedup: the held-out detector already validated the frequency, and a sin/cos pair at one
    # frequency is not monotone-equivalent to a fixed-grid twin, so the Spearman gate would only ever drop them on a spurious near-tie. Keeping them here
    # guarantees they remain in ``cols`` for the protection block.
    _adaptive_fourier_keep = set(getattr(self, "_adaptive_fourier_features_", None) or [])
    # Keep-higher-MI dedup policy: when a near-duplicate cluster spans stages, the survivor must be the column carrying the MOST information about y, NOT merely
    # the first-appended one. The default-on univariate-basis stage writes into ``hybrid_orth_features_`` and is appended BEFORE ``mi_greedy_features_``, so a
    # first-appended policy silently sacrifices a genuine mi_greedy ``|x|``-family signal (``log_abs(x)`` / ``sqrt_abs(x)`` / ``square(x)`` / ``abs(x)``) to a
    # monotone-equivalent basis twin (``x__L2`` / ``x__cos1`` / ...). We score every appended engineered column once with the SAME plug-in MI scorer + quantile
    # binning the FE stages used, then break dedup ties by higher MI, with the mi_greedy / constructor-requested column winning exact MI ties (a monotone twin
    # bins identically, so MI is numerically equal - prefer the explicitly-requested constructor output). MI scoring is best-effort: any failure falls back to
    # the order-preserving first-appended policy so the dedup never crashes a fit.
    _mig_set = set(self.mi_greedy_features_ or [])
    _eng_mi: dict[str, float] = {}
    try:
        from mlframe.feature_selection.filters._orthogonal_univariate_fe import _mi_classif_batch

        _y_for_eng_mi = _y_np
        _y_for_eng_mi = encode_y_for_classif_mi(_y_for_eng_mi)
        if isinstance(X, pd.DataFrame) and len(_eng_cols_appended) >= 2:
            _mi_cols = [_c for _c in _eng_cols_appended if _c in X.columns]
            if _mi_cols:
                _mi_mat = X[_mi_cols].to_numpy(dtype=np.float64)
                _mi_vals = _mi_classif_batch(_mi_mat, _y_for_eng_mi, nbins=10)
                _eng_mi = {_name: float(_v) for _name, _v in zip(_mi_cols, _mi_vals)}
    except Exception as exc:
        logger.debug("mrmr: engineered-MI dict computation failed; treating as empty (no engineered candidates this round): %r", exc, exc_info=True)
        _eng_mi = {}

    def _eng_dedup_prefer(cand: str, kept: str) -> bool:
        """Return True when ``cand`` should DISPLACE the already-kept ``kept`` on a near-duplicate collision.

        Only CROSS-STAGE collisions (exactly one of the pair is an mi_greedy / constructor-requested column) ever flip the survivor: within a single stage we preserve the original
        first-appended policy byte-for-byte, so the dedup stays deterministic on the monotone-twin families a single basis stage emits (a quantile-binned MI tie between ``x__He2`` /
        ``x__cos1`` / ``x__L2`` would otherwise reshuffle non-deterministically). Across stages we keep the column carrying more MI about y, and the explicitly-requested mi_greedy column
        wins an exact MI tie (a monotone twin bins identically, so its MI is numerically equal - without this the default-on basis twin would silently evict the genuine ``|x|``-family signal).
        """
        _cand_mig = cand in _mig_set
        _kept_mig = kept in _mig_set
        if _cand_mig == _kept_mig:
            return False
        _mi_cand = _eng_mi.get(cand)
        _mi_kept = _eng_mi.get(kept)
        if _mi_cand is None or _mi_kept is None:
            return False
        if _mi_cand > _mi_kept + 1e-12:
            return True
        if _mi_cand >= _mi_kept - 1e-12:
            return _cand_mig and not _kept_mig
        return False

    if len(_eng_cols_appended) >= 2 and isinstance(X, pd.DataFrame):
        from mlframe.feature_selection.filters._mrmr_fit_impl._eng_dedup_scan import scan_engineered_duplicates

        _eng_keep, _eng_drop, _eng_arrs, _eng_ranks = scan_engineered_duplicates(X, _eng_cols_appended, _adaptive_fourier_keep, _eng_dedup_prefer)
        if _eng_drop:
            # Dependency-closure guard: never drop an engineered column / recipe that a SURVIVING recipe consumes via src_names (e.g. a cat_pair_cross producer
            # feeding a modular / numeric_decompose recipe). Dropping the producer while keeping the consumer orphans the consumer's source -> KeyError at
            # transform replay. Fixpoint over all recipe dicts so multi-level chains stay intact.
            _all_pre_recipe_dicts = tuple(getattr(recipes, _f) for _f in _DEPENDENCY_FAMILIES)
            while True:
                _protected = {
                    _s for _d in _all_pre_recipe_dicts for _r in _d.values() if _r.name not in _eng_drop for _s in (getattr(_r, "src_names", ()) or ())
                }
                _newly = _eng_drop & _protected
                if not _newly:
                    break
                _eng_drop -= _newly
            X = X.drop(columns=list(_eng_drop))
            self.hybrid_orth_features_ = [c for c in (self.hybrid_orth_features_ or []) if c not in _eng_drop]
            # Mirror cleanup for hinge legs (a hinge near-duplicate of another engineered column the Spearman dedup removed must not be re-added by the
            # HINGE-PROTECTION block).
            self._hinge_features_ = [c for c in (getattr(self, "_hinge_features_", None) or []) if c not in _eng_drop]
            self.mi_greedy_features_ = [c for c in (self.mi_greedy_features_ or []) if c not in _eng_drop]
            # Layer 33: mirror the same cleanup for TE-encoded columns. Every engineered roster, from the one shared tuple. Two hand-maintained copies of this
            # list had already drifted: this pass filtered 18 rosters while the unified-gate pass below filtered 27, so a column dropped by the Spearman dedup
            # stayed in the other nine until a later reconciliation happened to catch it.
            for _roster_attr in FE_ROSTER_ATTRS:
                setattr(self, _roster_attr, [c for c in (getattr(self, _roster_attr, []) or []) if c not in _eng_drop])
            for _family in _PRUNED_FAMILIES:
                _registry = getattr(recipes, _family)
                for _c in _eng_drop.intersection(_registry):
                    _registry.pop(_c, None)
            if verbose:
                logger.info(
                    "MRMR.fit engineered-FE dedup: pruned %d near-duplicate engineered column(s) at Spearman |rho| >= 0.99: %s",
                    len(_eng_drop),
                    sorted(_eng_drop),
                )
    return X
