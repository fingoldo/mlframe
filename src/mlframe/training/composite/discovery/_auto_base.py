"""Auto-base candidate ranking via residualised MI for ``CompositeTargetDiscovery``.

Split out of ``composite_discovery.py`` to keep the parent below the 1k-line
monolith threshold. ``_auto_base`` is bound back onto the
``CompositeTargetDiscovery`` class at the parent's module bottom, so call
sites that invoke ``self._auto_base(...)`` continue to work unchanged.

Near-copy increment-learnability precheck cost: it fires only for the (few) candidates whose
``|corr(base, y)| > base_max_abs_corr_with_y`` and reuses the already-imported ``_mi_pair_bin`` kernel on a
row-capped screening subsample (``near_copy_precheck_max_sample``, default 5000), one linear residual +
per-other-feature bin-MI. cProfile on a 5k*8 screening frame: the added work is a small constant per near-copy
candidate (bounded by the cap and feature count), dwarfed by the surrounding matrix build / MI ranking -- no
actionable speedup, so no dispatcher/ladder is warranted for it.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Sequence

import numpy as np

# ``rankdata`` is an optional scipy dependency; if unavailable, the fallback
# argsort-of-argsort path is wrong on ties and the auto-base ranker would
# emit incorrect orders. Same graceful-fallback contract as the parent
# module preserves the rankdata-based Spearman demoter wiring.
try:
    from scipy.stats import rankdata
except ImportError:
    rankdata = None  # type: ignore[assignment]

from ._causal_lag import is_causal_base_name
from ._collinear_numba import block_shuffle_gather
from ._structural_hints import boost_for_features
from .screening import (
    _mi_from_binned_pair,
    _mi_pair_bin,
    _mi_per_feature_y_fixed,
    _mi_per_feature_y_fixed_per_col,
    _safe_corr,
    _sample_indices,
)

logger = logging.getLogger(__name__)


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
    gain": the residualised metric
    ranks candidates by how predictable ``y - alpha*x - beta``
    is from the remaining features. On a feature whose linear
    contribution is small, the residual still contains the
    dominant feature itself (we did not subtract it), so the
    remaining feature set predicts the residual perfectly --
    which inverts the ranking versus what we want. Pairwise
    MI(y, x) directly measures "how much information about y
    does this single feature carry" and surfaces a lag feature at
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
    # Adaptive hint cap. Previous fixed cap of
    # ``max(1, top_k // 2)`` was too aggressive when BD ablation
    # confidently identified the dominant base (e.g. delta% > 100%
    # for the top-1 feature in a prod incident). Now: if the user
    # supplied a hint AND we have ablation strengths in metadata,
    # check the strength signal. Strong hint (top-1 delta_pct >
    # ``hint_strength_threshold_pct``, default 50%) -> use FULL
    # hint (no cap). Weak/absent strength info -> fall back to the
    # half-slot cap so MI-leaders still get evaluated.
    #
    # Rationale: BD ablation directly measures "drop feature -> RMSE
    # delta%" which is a high-quality signal. When it screams +501%
    # for a lag feature (real production case), trust it; don't dilute
    # with MI-leaders that may be lower-quality features.
    strong_hint_threshold = float(getattr(
        self.config, "hint_strength_threshold_pct", 50.0,
    ))
    # Strength info is plumbed via the suite-level hint precompute
    # at core.py and stored on the discovery instance for this fit.
    # Absent = treat as unknown strength -> use half-slot cap.
    hint_strengths = getattr(self, "_hint_strengths_pct", None)
    # hint_raw and hint_strengths are positionally aligned; hint_kept drops filtered entries, so realign strengths to the surviving hints before taking the max -- otherwise a dropped strong hint's strength leaks onto a surviving weak one.
    _kept_set = set(hint_kept)
    _aligned = [s for f, s in zip(hint_raw, (hint_strengths or [])) if f in _kept_set]
    is_strong_hint = (
        hint_strengths is not None
        and len(_aligned) > 0
        and max(_aligned) >= strong_hint_threshold
    )
    if is_strong_hint:
        # Full hint -- no cap. Log so it's auditable.
        logger.info(
            "[CompositeTargetDiscovery] auto-base using FULL hint "
            "(%d candidates, max ablation delta%% = %.1f%% >= %.1f%% "
            "threshold; trusting BD over MI ranking).",
            len(hint_kept), max(_aligned),
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
    # Drop columns with ZERO observed values in the screening sample
    # BEFORE the all-row finite-mask. A single fully-NaN column made
    # the AND-mask return zero rows (observed in prod: 'auto-base:
    # only 0 finite rows in screening sample') AND every downstream
    # sklearn.SimpleImputer call sprayed UserWarnings about feature
    # index N. Per-column NaN tolerance: keep features with at least
    # ``_AUTO_BASE_MIN_FRACTION_FINITE`` non-NaN cells in the
    # screening sample; impute the rest per-column with the column
    # mean before the all-row finite mask.
    _MIN_FRAC_FINITE = 0.10  # at least 10% non-NaN cells
    _col_finite_frac = np.isfinite(x_matrix).mean(axis=0)
    _keep_cols = _col_finite_frac >= _MIN_FRAC_FINITE
    if not _keep_cols.all():
        _dropped = [
            usable_features[i] for i, k in enumerate(_keep_cols.tolist())
            if not k
        ]
        logger.info(
            "[CompositeTargetDiscovery] auto-base: dropping %d feature(s) "
            "with <%.0f%% finite cells in screening sample: %s",
            len(_dropped), _MIN_FRAC_FINITE * 100, _dropped[:10],
        )
        x_matrix = x_matrix[:, _keep_cols]
        usable_features = [
            f for f, k in zip(usable_features, _keep_cols.tolist()) if k
        ]
    # The MI RANKING must be estimated with
    # PER-PAIR (per-column) finite masking, NOT the global all-column
    # ``np.all(isfinite(x_matrix), axis=1)`` intersection. For mid-range-NaN
    # columns the intersection is a non-random (MNAR) subset -- the rows
    # where EVERY feature happens to be observed -- so MI(y, x_j) estimated
    # on it is biased by the joint-observability pattern and silently shifts
    # which base wins. Per-pair masking estimates each column's MI on its
    # own observed rows (mirroring ``_mi_to_target`` / the prebinned
    # ``-1``-sentinel path) and is bit-identical when nothing is NaN.
    #
    # Capture the pristine (pre-impute) screening matrix for the MI ranking:
    # the legacy ``< 50``-global-finite fallback below imputes NaNs in
    # ``x_matrix`` (used by the demoters / dedup, which need a SHARED row
    # basis for pairwise correlations), but the MI ranking must see the real
    # NaNs so per-pair masking can drop only the truly-missing rows. The
    # impute reassigns ``x_matrix`` to a fresh ``np.where`` array, so this
    # reference keeps pointing at the original NaN-bearing screening sample.
    x_matrix_for_mi = x_matrix
    use_per_pair = bool(getattr(self.config, "auto_base_mi_per_pair_mask", True))
    finite = np.isfinite(y_screen) & np.all(np.isfinite(x_matrix), axis=1)
    # Audit the MNAR severity once: how much of the per-pair-available row
    # mass the global intersection discards. Below the configured fraction
    # the global mask would meaningfully under-sample some column, which is
    # exactly the per-pair masking case; log it so the divergence is auditable.
    _mnar_threshold = float(getattr(
        self.config, "auto_base_mnar_per_pair_threshold", 0.5,
    ))
    _n_global = int(finite.sum())
    _per_col_finite = np.isfinite(x_matrix) & np.isfinite(y_screen)[:, None]
    _per_col_counts = _per_col_finite.sum(axis=0)
    _max_per_col = int(_per_col_counts.max()) if _per_col_counts.size else 0
    if (
        use_per_pair
        and _max_per_col > 0
        and _n_global < _mnar_threshold * _max_per_col
    ):
        logger.info(
            "[CompositeTargetDiscovery] auto-base: global all-column finite "
            "mask keeps %d row(s) vs %d per-pair-available (%.0f%%, below "
            "%.0f%% MNAR threshold); ranking by PER-PAIR MI to avoid "
            "selection bias.",
            _n_global, _max_per_col,
            100.0 * _n_global / _max_per_col, 100.0 * _mnar_threshold,
        )
    if finite.sum() < 50:
        # Even after the per-column drop, the all-row finite mask can
        # still be too tight when many features have sparse but
        # non-zero NaN density. Impute remaining NaNs with per-column
        # mean and proceed (correlation-quality features survive; truly
        # bad columns were already dropped above). NOTE: this only feeds
        # the demoters / dedup correlations; the MI ranking already uses
        # ``x_matrix_for_mi`` (pristine + per-pair masked) above.
        _col_means = np.nanmean(x_matrix, axis=0)
        _col_means = np.where(np.isfinite(_col_means), _col_means, 0.0)
        _nan_mask = ~np.isfinite(x_matrix)
        if _nan_mask.any():
            x_matrix = np.where(_nan_mask, _col_means[None, :], x_matrix)
        finite = np.isfinite(y_screen) & np.all(np.isfinite(x_matrix), axis=1)
        if finite.sum() < 50:
            logger.warning(
                "[CompositeTargetDiscovery] auto-base: only %d finite rows "
                "even after per-column NaN drop + mean impute; falling back "
                "to feature-list order.", int(finite.sum()),
            )
            # Keep hint features at the front: BD ablation is the one signal not derived from the broken screening sample, and a hint may not fall within the first top_k of the arbitrary column order.
            return (hint_kept + [c for c in usable_features if c not in hint_kept])[: self.config.auto_base_top_k]
    # Per-feature MI honours config.mi_estimator: bin-based when
    # the screening pipeline opted for the fast estimator. Hoist the
    # y-binning out of the per-feature loop -- y is fixed across all
    # candidate columns, so re-quantiling it inside ``_mi_pair_bin``
    # is wasted work.  See ``_mi_per_feature_y_fixed`` docstring for
    # the 1.67x bit-exact benchmark.
    if self.config.mi_estimator == "bin":
        if use_per_pair:
            # Per-pair NaN masking; bit-identical to the global path
            # on an all-finite screening sample.
            mi_per_feature = _mi_per_feature_y_fixed_per_col(
                x_matrix_for_mi, y_screen,
                nbins=self.config.mi_nbins,
            )
        else:
            mi_per_feature = _mi_per_feature_y_fixed(
                x_matrix[finite], y_screen[finite],
                nbins=self.config.mi_nbins,
            )
    else:
        from sklearn.feature_selection import mutual_info_regression
        if use_per_pair:
            # Kraskov kNN cannot ingest NaN, so mask per pair here and run
            # the single-column estimator on each column's surviving rows
            # (mirrors ``_mi_to_target``'s knn branch). A mostly-NaN column
            # only zeros its own MI instead of the whole sweep.
            _n_cols_mi = x_matrix_for_mi.shape[1]
            mi_per_feature = np.zeros(_n_cols_mi, dtype=np.float64)
            _y_fin_mi = np.isfinite(y_screen)
            for _jc in range(_n_cols_mi):
                _col = x_matrix_for_mi[:, _jc]
                _pair = _y_fin_mi & np.isfinite(_col)
                if int(_pair.sum()) < 50:
                    mi_per_feature[_jc] = 0.0
                    continue
                mi_per_feature[_jc] = float(mutual_info_regression(
                    _col[_pair].reshape(-1, 1), y_screen[_pair],
                    n_neighbors=self.config.mi_n_neighbors,
                    random_state=self.config.random_state,
                )[0])
        else:
            mi_per_feature = mutual_info_regression(
                x_matrix[finite], y_screen[finite],
                n_neighbors=self.config.mi_n_neighbors,
                random_state=self.config.random_state,
            )
    # Structural detectors for time-index
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
        # Use the module-level binding so the rankdata fix is detectable from
        # outside (see regression sensor test_m3_spearman_demoter_uses_rankdata).
        _rankdata = rankdata
        n_screen = int(finite.sum())
        row_idx = np.arange(n_screen, dtype=np.float64)
        # Hint features are IMMUNE from
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
        # Spatial-coord block detector tightened
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

    # Permutation-MI null filter. Catches
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
            # Permute whole blocks, then gather each permuted block's real (in-bounds) elements in
            # one fused njit pass (``block_shuffle_gather``): same ``rng.permutation`` draw, same
            # element order as the prior numpy broadcast+mask+fancy-index path (bit-identical), but
            # it skips the O(n_blocks*block_len) int64 index template + boolean ``idx < m`` mask.
            # ~2.6-3.0x per call at the 20k-100k null-screen sizes; the short trailing block's
            # padding is dropped, so that block contributes just its real elements wherever placed.
            perm = rng.permutation(n_blocks)
            return block_shuffle_gather(arr, perm, block_len)

        rng_perm = np.random.default_rng(
            int(self.config.random_state) + 7919
        )
        y_finite = y_screen[finite]
        # Under per-pair masking the null distribution must be estimated
        # on the SAME rows as the per-pair MI it is compared against (line
        # ``passes_null = mi_per_feature > null_threshold``); otherwise a
        # per-pair MI (its own observed rows) is gated against a null built on
        # the global intersection rows -- apples to oranges. We therefore mask
        # each column on its own ``isfinite(col) & isfinite(y_screen)`` rows
        # (from the pristine pre-impute matrix) when per-pair is active. The
        # fast hoisted-y-codes path stays for columns whose per-pair rows equal
        # the global rows (no extra NaN), which keeps the all-finite case
        # bit-identical and as fast as before.
        _y_fin_null = np.isfinite(y_screen)
        # Fast null path: np.quantile is shuffle-invariant and np.searchsorted commutes
        # with the permutation, so binning a *shuffled* NaN-free column is identical to
        # shuffling that column's bin codes. For the "bin" estimator we therefore bin y +
        # each clean column ONCE and shuffle the integer codes per permutation
        # (_mi_from_binned_pair), hoisting the per-perm quantile + searchsorted out of the
        # inner loop. Bit-identical to the per-call _mi_pair_bin (verified 0.0 MI diff;
        # _block_shuffle consumes the same RNG draws regardless of dtype, so the null
        # distribution is unchanged), ~5x faster on n=20k. Columns containing NaN fall back
        # to _mi_pair_bin, whose internal finite mask makes y-binning shuffle-dependent
        # (prebinning would not be bit-identical there).
        _nbins = self.config.mi_nbins
        _bin_estimator = (self.config.mi_estimator == "bin")
        _y_codes_null = None
        if _bin_estimator and n_screen >= 5 * _nbins and np.isfinite(y_finite).all():
            _qs_null = np.linspace(0.0, 1.0, _nbins + 1)[1:-1]
            _y_edges_null = np.quantile(y_finite, _qs_null)
            _y_codes_null = np.searchsorted(
                _y_edges_null, y_finite, side="right",
            ).astype(np.int64)
            np.clip(_y_codes_null, 0, _nbins - 1, out=_y_codes_null)
        null_means = np.zeros(x_matrix.shape[1])
        null_stds = np.zeros(x_matrix.shape[1])
        for j in range(x_matrix.shape[1]):
            if use_per_pair:
                # Per-pair rows for this column (matches the per-pair MI).
                _col_raw = x_matrix_for_mi[:, j]
                _pair_j = _y_fin_null & np.isfinite(_col_raw)
                if int(_pair_j.sum()) < 50:
                    null_means[j] = 0.0
                    null_stds[j] = 0.0
                    continue
                col = _col_raw[_pair_j]
                y_col = y_screen[_pair_j]
            else:
                col = x_matrix[finite, j]
                y_col = y_finite
            null_mis = np.empty(n_perms)
            # Prebin clean columns once, then shuffle codes instead of values (see note above).
            col_codes = None
            # Reuse the hoisted y-codes only when this column's per-pair rows
            # are exactly the global rows (same length AND same y); otherwise
            # bin y on the column's own rows so the null is self-consistent.
            _y_codes_for_col = None
            if _y_codes_null is not None and y_col.shape[0] == y_finite.shape[0]:
                _y_codes_for_col = _y_codes_null
            elif _bin_estimator and y_col.shape[0] >= 5 * _nbins and np.isfinite(y_col).all():
                _qs_col = np.linspace(0.0, 1.0, _nbins + 1)[1:-1]
                _ye = np.quantile(y_col, _qs_col)
                _y_codes_for_col = np.searchsorted(_ye, y_col, side="right").astype(np.int64)
                np.clip(_y_codes_for_col, 0, _nbins - 1, out=_y_codes_for_col)
            if _y_codes_for_col is not None and np.isfinite(col).all():
                _qs_col = np.linspace(0.0, 1.0, _nbins + 1)[1:-1]
                _x_edges_null = np.quantile(col, _qs_col)
                col_codes = np.searchsorted(
                    _x_edges_null, col, side="right",
                ).astype(np.int64)
                np.clip(col_codes, 0, _nbins - 1, out=col_codes)
            for p in range(n_perms):
                if col_codes is not None:
                    shuffled_codes = _block_shuffle(col_codes, rng_perm)
                    null_mis[p] = _mi_from_binned_pair(
                        shuffled_codes, _y_codes_for_col, nbins=_nbins,
                    )
                elif _bin_estimator:
                    shuffled = _block_shuffle(col, rng_perm)
                    null_mis[p] = _mi_pair_bin(
                        shuffled, y_col, nbins=_nbins,
                    )
                else:
                    shuffled = _block_shuffle(col, rng_perm)
                    from sklearn.feature_selection import mutual_info_regression
                    null_mis[p] = float(mutual_info_regression(
                        shuffled.reshape(-1, 1), y_col,
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
    # Structural-affinity boost. Surfaces OBVIOUS base columns from data
    # shape / correlation that the MI ranking alone can miss when a noisier
    # competitor's pairwise MI lands a hair higher: a near-affine predictor of
    # y (prime ``linear_residual`` base), a low-cardinality integer grouping
    # column (prime ``grouped`` base), a monotone/timestamp column (prime
    # ``time`` base). The boost is a BOUNDED nudge scaled to the MI spread --
    # it augments the MI ranking, never replaces it: a clearly larger MI gap
    # still wins. Applied to non-hint candidates only (hints already lead the
    # slots) and BEFORE the time/spatial demotion so a demoted column cannot
    # be re-promoted by the time detector here. A monotone column the
    # time-index demoter would sink is not boosted as a ``time`` base when the
    # demoter is active, so the two stay consistent.
    # Per-column |corr(col, y)| on the screen sample. Used to (a) gate the structural boost away from
    # near-copies of y and (b) exclude near-copy bases after ranking (a base ~= y makes the residual
    # inverse fragile under group shift). Computed once, vectorised.
    _abs_corr_to_y: dict[str, float] = {}
    try:
        _xm_c = np.asarray(x_matrix[finite], dtype=np.float64)
        _ys_c = np.asarray(y_screen[finite], dtype=np.float64).ravel()
        if _xm_c.shape[0] > 2 and _ys_c.size == _xm_c.shape[0]:
            _yc = _ys_c - _ys_c.mean()
            _yden = float(np.sqrt(np.dot(_yc, _yc)))
            if _yden > 0:
                _xc = _xm_c - _xm_c.mean(axis=0, keepdims=True)
                _xden = np.sqrt(np.einsum("ij,ij->j", _xc, _xc))
                _num = _xc.T @ _yc
                with np.errstate(invalid="ignore", divide="ignore"):
                    _corr = np.where(_xden > 0, _num / (_xden * _yden), 0.0)
                _abs_corr_to_y = {c: abs(float(_corr[j])) for j, c in enumerate(usable_features)}
    except Exception:  # noqa: BLE001 -- corr is a heuristic gate; never abort discovery on it
        _abs_corr_to_y = {}

    _boost_corr_gate = float(getattr(self.config, "auto_base_structural_boost_corr_gate", 0.98))
    if getattr(self.config, "auto_base_structural_boost", True):
        finite_mi = [m for m in mi_for_ranking.tolist() if math.isfinite(m)]
        mi_spread = (max(finite_mi) - min(finite_mi)) if len(finite_mi) >= 2 else 0.0
        boost_fraction = float(getattr(
            self.config, "auto_base_structural_boost_fraction", 0.25,
        ))
        boost, kinds = boost_for_features(
            x_matrix[finite], y_screen[finite], list(usable_features),
            mi_spread=mi_spread, max_boost_fraction=boost_fraction,
        )
        if kinds:
            hint_set_boost = set(hint_kept)
            applied: list[tuple[str, str, float]] = []
            _boost_corr_skipped = 0
            for j, col_name in enumerate(usable_features):
                if boost[j] <= 0.0:
                    continue
                # Never boost a hint (already leading) or a demoted feature
                # (the demoter's verdict is authoritative for that column).
                if col_name in hint_set_boost or col_name in demote_set:
                    continue
                # Don't boost a near-copy of y: boosting promotes exactly the fragile leaked columns.
                if _boost_corr_gate < 1.0 and _abs_corr_to_y.get(col_name, 0.0) > _boost_corr_gate:
                    _boost_corr_skipped += 1
                    continue
                if not math.isfinite(mi_for_ranking[j]):
                    continue
                mi_for_ranking[j] += boost[j]
                applied.append((col_name, kinds.get(col_name, "?"), float(boost[j])))
            if _boost_corr_skipped:
                logger.info(
                    "[CompositeTargetDiscovery] auto-base structural boost withheld from %d "
                    "near-copy-of-y candidate(s) (|corr(col,y)| > %.4g).",
                    _boost_corr_skipped, _boost_corr_gate,
                )
            if applied:
                preview = ", ".join(
                    f"{n}({k},+{b:.4g})" for n, k, b in
                    sorted(applied, key=lambda t: -t[2])[:5]
                )
                logger.info(
                    "[CompositeTargetDiscovery] auto-base structural boost "
                    "applied to %d candidate(s) (mi_spread=%.4g): %s",
                    len(applied), mi_spread, preview,
                )
    # Apply demotion to time-index / spatial-
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
    # MRMR base-candidate ranking (opt-in via base_ranking_criterion="mrmr").
    # Pure relevance ordering fills the shortlist with near-duplicate strong
    # bases (y_prev / y_prev_smoothed / y_lag2); MRMR trades a little relevance
    # for diversity so the top-K spans distinct signal, cutting ensemble
    # correlation. Redundancy reuses the bin-MI kernel on the shared finite
    # screening rows; relevance is the (post-demote/boost/null) MI. Default
    # "mi" leaves this path dormant and byte-identical to the legacy ranking.
    if (
        getattr(self.config, "base_ranking_criterion", "mi") == "mrmr"
        and len(ranked) > 1
    ):
        from ._mrmr_base_rank import mrmr_rank_bases
        _mrmr_names = [c for _m, c in ranked]
        _mrmr_rel = [m for m, _c in ranked]
        _mrmr_col = {name: i for i, name in enumerate(usable_features)}
        _mrmr_X = x_matrix[np.ix_(finite, [_mrmr_col[c] for c in _mrmr_names])]
        _mrmr_beta = float(getattr(self.config, "base_ranking_mrmr_beta", 1.0))
        _mrmr_nbins = int(self.config.mi_nbins)
        _mrmr_cache: dict[tuple[int, int], float] = {}

        def _mrmr_redundancy(i: int, j: int) -> float:
            key = (i, j) if i < j else (j, i)
            val = _mrmr_cache.get(key)
            if val is None:
                val = _mi_pair_bin(_mrmr_X[:, i], _mrmr_X[:, j], nbins=_mrmr_nbins)
                _mrmr_cache[key] = val
            return val

        _mrmr_order = mrmr_rank_bases(
            _mrmr_names, _mrmr_rel, _mrmr_redundancy, len(_mrmr_names),
            beta=_mrmr_beta,
        )
        _rel_lookup = dict(zip(_mrmr_names, _mrmr_rel))
        ranked = [(_rel_lookup[c], c) for c in _mrmr_order]
        logger.info(
            "[CompositeTargetDiscovery] auto-base MRMR reranked %d candidate(s) "
            "(beta=%.3g); top: %s",
            len(ranked), _mrmr_beta,
            ", ".join(c for _m, c in ranked[:5]),
        )
    # Near-copy-of-y exclusion. A base whose |corr(base, y)| is ~1.0 is y itself up to noise; the
    # residual inverse y = T_hat + alpha*base is then carried entirely by base and blows up on any
    # group/feature shift. Drop such bases (hint or not -- a literal copy is never a safe base), but
    # log loudly when a hint candidate is removed so the operator sees why the hint didn't lead.
    _copy_thresh = getattr(self.config, "base_max_abs_corr_with_y", 0.9995)
    if _copy_thresh is not None and float(_copy_thresh) < 1.0 and _abs_corr_to_y:
        _ct = float(_copy_thresh)
        # Provenance exemption: a strictly-causal base (grouped-causal engineered ``__gcausal_*`` or a named ``{y}_prev``
        # lag) is NOT a fragile near-copy of y even at |corr|~1 -- its additive inverse ``y = T_hat + y_prev`` uses a REAL
        # per-row previous value, so it stays in-range on unseen groups (the exact opposite of the gate's failure mode).
        # On a strong-AR target (lag-1 autocorr ~1) the causal lag is the single best base, so excluding it kills the win.
        # Exempt by PROVENANCE only (never a marginal corr match), so a contemporaneous near-copy of y is still dropped.
        _tcol = getattr(self, "_target_col", None)
        _causal_exempt = bool(getattr(self.config, "causal_base_gate_exempt", True))
        # Increment-learnability precheck. A non-provenance near-copy earns its keep as a COMPOSITE (y_hat = T_hat + base)
        # only when the residual ``y - linfit(base)`` still carries LEARNABLE signal from the OTHER features -- otherwise the
        # composite adds nothing over feeding ``base`` as a plain feature and the drop is correct. Measured cheaply on the
        # (capped) screening sample via the bin-MI kernel: exempt when max_j MI(x_j, residual) exceeds a small threshold.
        _precheck_on = bool(getattr(self.config, "near_copy_increment_learnability_precheck", True))
        _precheck_mi_thresh = float(getattr(self.config, "near_copy_increment_learnability_mi_threshold", 0.05))
        _precheck_cap = int(getattr(self.config, "near_copy_precheck_max_sample", 5000))
        _nbins = int(self.config.mi_nbins)
        _nc_idx = {name: i for i, name in enumerate(usable_features)}
        # Aligned, all-finite screening rows (capped) shared across the precheck's base + other-feature MI estimates.
        _pc_rows = np.flatnonzero(finite)
        if _pc_rows.size > _precheck_cap:
            _pc_rows = _pc_rows[np.linspace(0, _pc_rows.size - 1, _precheck_cap).astype(np.int64)]
        _pc_X = x_matrix[_pc_rows]
        _pc_y = y_screen[_pc_rows]

        # Bin-MI has a positive finite-sample bias (~(nbins-1)^2/(2N)) that makes a pure-noise residual look
        # faintly informative, so compare against a PERMUTATION null: max_j MI(x_j, resid) minus the same over a
        # shuffled residual removes the bias floor. A genuine learnable residual clears the null by a wide margin.
        _pc_perm = np.random.default_rng(int(getattr(self.config, "random_state", 0) or 0)).permutation(_pc_rows.size)

        def _near_copy_residual_is_learnable(_c: str) -> bool:
            _j = _nc_idx.get(_c)
            if _j is None or _pc_rows.size < 5 * _nbins or _pc_X.shape[1] < 2:
                return False
            _base = _pc_X[:, _j]
            _bvar = float(_base.var())
            if not math.isfinite(_bvar) or _bvar <= 0.0:
                return False
            _a = float(np.cov(_base, _pc_y)[0, 1] / _bvar)
            _resid = _pc_y - (_a * _base + float(_pc_y.mean() - _a * _base.mean()))
            _resid_null = _resid[_pc_perm]
            _best_real = 0.0
            _best_null = 0.0
            for _oj in range(_pc_X.shape[1]):
                if _oj == _j:
                    continue
                _col = _pc_X[:, _oj]
                _best_real = max(_best_real, _mi_pair_bin(_col, _resid, nbins=_nbins))
                _best_null = max(_best_null, _mi_pair_bin(_col, _resid_null, nbins=_nbins))
            return (_best_real - _best_null) >= _precheck_mi_thresh

        def _is_near_copy(_c: str) -> bool:
            if _abs_corr_to_y.get(_c, 0.0) <= _ct:
                return False
            if _causal_exempt and is_causal_base_name(_c, _tcol):
                return False
            if _precheck_on and _near_copy_residual_is_learnable(_c):
                return False
            return True

        _excluded = [(m, c) for m, c in ranked if _is_near_copy(c)]
        if _excluded:
            ranked = [(m, c) for m, c in ranked if not _is_near_copy(c)]
            _hint_excl = [c for _m, c in _excluded if c in set(hint_kept)]
            logger.info(
                "[CompositeTargetDiscovery] auto-base excluded %d near-copy-of-y base candidate(s) "
                "(|corr(base,y)| > %.4g; residual inverse would be carried entirely by base and is "
                "fragile under group shift): %s%s",
                len(_excluded), _ct,
                ", ".join(f"{c}({_abs_corr_to_y.get(c, 0.0):.4f})" for _m, c in _excluded[:5]),
                f" -- INCLUDING hint(s): {_hint_excl}" if _hint_excl else "",
            )
    # Cross-base correlation dedup. Two highly-correlated bases
    # (typical: ``y_prev``, ``y_prev_lag2``, ``y_smooth_3``)
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
        # Hint features are IMMUNE from dedup.
        # Otherwise on geological data with high feature
        # cross-correlation (e.g. Z ~ y_prev at |corr|=0.974),
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
