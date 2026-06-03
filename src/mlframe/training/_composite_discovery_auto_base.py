"""Auto-base candidate ranking via residualised MI for ``CompositeTargetDiscovery``.

Split out of ``composite_discovery.py`` to keep the parent below the 1k-line
monolith threshold. ``_auto_base`` is bound back onto the
``CompositeTargetDiscovery`` class at the parent's module bottom, so call
sites that invoke ``self._auto_base(...)`` continue to work unchanged.
"""
from __future__ import annotations

import logging
import math
import re

import numpy as np

# ``rankdata`` is an optional scipy dependency; if unavailable, the fallback
# argsort-of-argsort path is wrong on ties and the auto-base ranker would
# emit incorrect orders. Same graceful-fallback contract as the parent
# module preserves the M3 fix sensor wiring.
try:
    from scipy.stats import rankdata
except ImportError:
    rankdata = None  # type: ignore[assignment]

from .composite_screening import (
    _mi_from_binned_pair,
    _mi_pair_bin,
    _mi_per_feature_y_fixed,
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
    gain" of round-2 critique R2.27: the residualised metric
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
    # R10c bug #5 fix: adaptive hint cap. Previous fixed cap of
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
    is_strong_hint = (
        hint_strengths is not None
        and len(hint_strengths) > 0
        and max(hint_strengths[:len(hint_kept)]) >= strong_hint_threshold
    )
    if is_strong_hint:
        # Full hint -- no cap. Log so it's auditable.
        logger.info(
            "[CompositeTargetDiscovery] auto-base using FULL hint "
            "(%d candidates, max ablation delta%% = %.1f%% >= %.1f%% "
            "threshold; trusting BD over MI ranking).",
            len(hint_kept), max(hint_strengths[:len(hint_kept)]),
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
    finite = np.isfinite(y_screen) & np.all(np.isfinite(x_matrix), axis=1)
    if finite.sum() < 50:
        # Even after the per-column drop, the all-row finite mask can
        # still be too tight when many features have sparse but
        # non-zero NaN density. Impute remaining NaNs with per-column
        # mean and proceed (correlation-quality features survive; truly
        # bad columns were already dropped above).
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
            return list(usable_features)[: self.config.auto_base_top_k]
    # Per-feature MI honours config.mi_estimator: bin-based when
    # the screening pipeline opted for the fast estimator. Hoist the
    # y-binning out of the per-feature loop -- y is fixed across all
    # candidate columns, so re-quantiling it inside ``_mi_pair_bin``
    # is wasted work.  See ``_mi_per_feature_y_fixed`` docstring for
    # the 1.67x bit-exact benchmark.
    if self.config.mi_estimator == "bin":
        mi_per_feature = _mi_per_feature_y_fixed(
            x_matrix[finite], y_screen[finite],
            nbins=self.config.mi_nbins,
        )
    else:
        from sklearn.feature_selection import mutual_info_regression
        mi_per_feature = mutual_info_regression(
            x_matrix[finite], y_screen[finite],
            n_neighbors=self.config.mi_n_neighbors,
            random_state=self.config.random_state,
        )
    # R10b improvement #7: structural detectors for time-index
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
        # Use the module-level binding so the M3 fix is detectable from
        # outside (see regression sensor test_m3_spearman_demoter_uses_rankdata).
        _rankdata = rankdata
        n_screen = int(finite.sum())
        row_idx = np.arange(n_screen, dtype=np.float64)
        # R10c bug #2 extension: hint features are IMMUNE from
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
        # R10c bug #1 fix: spatial-coord block detector tightened
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

    # R10b improvement #2: permutation-MI null filter. Catches
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
            blocks = [arr[i * block_len:(i + 1) * block_len]
                      for i in range(n_blocks)]
            perm = rng.permutation(len(blocks))
            shuffled = np.concatenate([blocks[p] for p in perm])
            return shuffled[:m]

        rng_perm = np.random.default_rng(
            int(self.config.random_state) + 7919
        )
        y_finite = y_screen[finite]
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
            col = x_matrix[finite, j]
            null_mis = np.empty(n_perms)
            # Prebin clean columns once, then shuffle codes instead of values (see note above).
            col_codes = None
            if _y_codes_null is not None and np.isfinite(col).all():
                _x_edges_null = np.quantile(col, _qs_null)
                col_codes = np.searchsorted(
                    _x_edges_null, col, side="right",
                ).astype(np.int64)
                np.clip(col_codes, 0, _nbins - 1, out=col_codes)
            for p in range(n_perms):
                if col_codes is not None:
                    shuffled_codes = _block_shuffle(col_codes, rng_perm)
                    null_mis[p] = _mi_from_binned_pair(
                        shuffled_codes, _y_codes_null, nbins=_nbins,
                    )
                elif _bin_estimator:
                    shuffled = _block_shuffle(col, rng_perm)
                    null_mis[p] = _mi_pair_bin(
                        shuffled, y_finite, nbins=_nbins,
                    )
                else:
                    shuffled = _block_shuffle(col, rng_perm)
                    from sklearn.feature_selection import mutual_info_regression
                    null_mis[p] = float(mutual_info_regression(
                        shuffled.reshape(-1, 1), y_finite,
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
    # R10b improvement #7: apply demotion to time-index / spatial-
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
        # R10c bug #2 fix: hint features are IMMUNE from dedup.
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
