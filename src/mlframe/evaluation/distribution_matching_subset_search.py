"""``distribution_matching_subset_search``: pick the block subset whose feature distribution best matches a target.

Source: 1st_lanl-earthquake-prediction.md -- "sampling 10 full earthquakes multiple times (up to 10k times) on
train, comparing the average KS statistic of all selected features on the sampled earthquakes to the feature
distributions in full test." Given train data grouped into blocks (per-entity/per-time-series -- so a subset
of BLOCKS is sampled, not individual rows, preserving each block's internal structure), randomly search
subsets of blocks whose aggregate feature distribution best matches a reference (target/test) distribution by
mean KS statistic across features -- useful whenever train and test come from different regimes and simple
row-level reweighting isn't enough, since row-level resampling would break within-block temporal/entity
structure that block-level resampling preserves.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from numba import njit

# Combined-size threshold below which the njit two-pointer scan beats the numpy-vectorized path.
# Measured (scratch A/B, best-of-5): n=200/m=500 (combined 700) -> njit ~3.4x faster (0.0087s vs 0.0295s
# per 500 calls); n=2000/m=5000 (combined 7000) -> njit ~1.19x SLOWER (0.142s vs 0.119s per 200 calls).
# numba's internal np.sort has more fixed per-call overhead than numpy's C sort, so it only wins while
# sort cost is small relative to that overhead -- a genuine crossover, not a strict win either way. This
# search's per-trial sample size is block_count * rows_per_block, typically in the hundreds (LANL-style
# "10 earthquakes" scale), so njit is the common-case default; 3000 sits between the two measured points.
_KS_NJIT_COMBINED_SIZE_THRESHOLD = 3000


@njit(cache=True, fastmath=True)
def _ks_statistic_njit(sample_sorted: np.ndarray, target_sorted: np.ndarray) -> float:
    na = len(sample_sorted)
    nb = len(target_sorted)
    i = j = 0
    d = 0.0
    while i < na and j < nb:
        if sample_sorted[i] <= target_sorted[j]:
            i += 1
        else:
            j += 1
        diff = abs(i / na - j / nb)
        if diff > d:
            d = diff
    return d


def _ks_statistic(sample_vals: np.ndarray, target_vals: np.ndarray) -> float:
    """Two-sample KS statistic only (no p-value) -- dispatched by combined array size.

    ``scipy.stats.ks_2samp`` computes the same statistic but pays real per-call dispatch overhead (axis/nan-
    policy wrapping, broadcast-shape validation, exact-vs-asymptotic p-value machinery neither of which this
    caller needs) -- measured as the dominant cost of a many-trials search (2000 calls: 0.45s scipy vs 0.04s
    here, ~11x, bit-identical statistic). This search only ever needs the statistic to rank subsets, never a
    p-value, so bypassing scipy's wrapper entirely is safe.

    Below ``_KS_NJIT_COMBINED_SIZE_THRESHOLD`` combined elements, a compiled two-pointer merge-scan
    (``_ks_statistic_njit``) is dispatched instead of the numpy-vectorized concatenate+searchsorted path --
    measured faster at small sizes but slower once sort cost dominates numba's per-call overhead (see the
    threshold comment above). CUDA/cupy were not tried: these array sizes (tens to low-thousands of rows per
    block-subset) are far below mlframe's own documented CUDA-wins threshold (n>=500k).
    """
    sample_sorted = np.sort(sample_vals)
    target_sorted = np.sort(target_vals)
    if len(sample_sorted) + len(target_sorted) < _KS_NJIT_COMBINED_SIZE_THRESHOLD:
        return float(_ks_statistic_njit(sample_sorted, target_sorted))
    all_vals = np.concatenate([sample_sorted, target_sorted])
    all_vals.sort()
    cdf_sample = np.searchsorted(sample_sorted, all_vals, side="right") / len(sample_sorted)
    cdf_target = np.searchsorted(target_sorted, all_vals, side="right") / len(target_sorted)
    return float(np.max(np.abs(cdf_sample - cdf_target)))


def _mean_ks_statistic(sample_df: pd.DataFrame, target_df: pd.DataFrame, feature_cols: Sequence[str]) -> float:
    stats = []
    for col in feature_cols:
        sample_vals = sample_df[col].to_numpy()
        target_vals = target_df[col].to_numpy()
        sample_vals = sample_vals[np.isfinite(sample_vals)]
        target_vals = target_vals[np.isfinite(target_vals)]
        if len(sample_vals) == 0 or len(target_vals) == 0:
            stats.append(1.0)  # maximally dissimilar when a column has no comparable data.
            continue
        stats.append(_ks_statistic(sample_vals, target_vals))
    return float(np.mean(stats))


# Fraction of the total trial budget spent on the pure-random init phase of "greedy_swap" before switching to
# hill-climbing -- half-and-half is a neutral default (no prior on whether random exploration or local swap
# refinement is more valuable for a given synthetic); see the biz_value A/B test for the measured outcome.
_GREEDY_SWAP_INIT_FRACTION = 0.5


def _energy_distance(sample_mat: np.ndarray, target_mat: np.ndarray, target_target_dist: float) -> float:
    """Multivariate (joint) energy distance between two point clouds -- E(X,Y) = 2*E|X-Y| - E|X-X'| - E|Y-Y'|.

    Unlike per-feature KS (which only compares marginals independently, one column at a time), this operates
    on the FULL feature vector at once and is sensitive to joint/correlation-structure mismatches that are
    invisible when every marginal matches (e.g. two features individually normal(0,1) on both sides, but
    correlated in one sample and independent in the other -- per-feature KS sees ~0 in both cases, energy
    distance does not). ``target_target_dist`` (the target-vs-itself term) is passed in precomputed since the
    target point cloud is invariant across trials -- recomputing it every call would be pure wasted O(m^2) work.
    """
    d_xy = float(np.mean(np.linalg.norm(sample_mat[:, None, :] - target_mat[None, :, :], axis=2)))
    d_xx = float(np.mean(np.linalg.norm(sample_mat[:, None, :] - sample_mat[None, :, :], axis=2)))
    return 2.0 * d_xy - d_xx - target_target_dist


def distribution_matching_subset_search(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    block_col: str,
    feature_cols: Optional[Sequence[str]] = None,
    n_blocks: int = 10,
    n_trials: int = 1000,
    random_state: int = 0,
    search_strategy: str = "greedy_swap",
    joint_distance_mode: Optional[str] = None,
    joint_weight: float = 1.0,
) -> dict:
    """Search subsets of ``n_blocks`` train blocks for the one best matching ``target_df``'s distribution.

    Parameters
    ----------
    train_df
        Train frame, containing a ``block_col`` grouping key (e.g. an entity/time-series id).
    target_df
        Reference frame (e.g. unlabeled test data) whose feature distribution the selected train subset
        should match.
    block_col
        Column identifying blocks; a WHOLE block is included or excluded together (never split), preserving
        within-block structure.
    feature_cols
        Numeric columns to score; defaults to every numeric column present in both frames (excluding
        ``block_col``).
    n_blocks
        Number of blocks to sample per trial.
    n_trials
        Total evaluation budget: for ``search_strategy="random"`` this is the number of random subsets tried;
        for ``"greedy_swap"`` this is the TOTAL number of KS-statistic evaluations spent across the random
        init phase, per-block scoring, and swap trials combined -- never more evaluations than ``"random"``
        would spend at the same ``n_trials``.
    random_state
        Seed for the block sampler.
    search_strategy
        ``"greedy_swap"`` (default): spend ``_GREEDY_SWAP_INIT_FRACTION`` of the budget on random sampling
        (same mechanics as ``"random"`` below), then hill-climb from the best subset found: each remaining
        budget unit either (a) scores one currently-selected block against ``target_df`` in isolation
        (lazily, cached, only when that block's own score is unknown) to identify the weakest block in the
        current subset, or (b) swaps the identified weakest block for one random unused block and evaluates
        the resulting subset, keeping the swap only if it improves the mean KS statistic (else reverting).
        Made the default after a multi-seed (20 seeds x 4 budgets) A/B at equal ``n_trials`` showed a
        decisive, consistent win over pure random: mean best-found KS statistic 11.7-31.1% lower and
        downstream RMSE 11.1-36.1% lower across ``n_trials`` in {50, 100, 300, 1000} (see the module's
        biz_value test for the pinned regression numbers).
        ``"random"``: pure random sampling, kept as an explicit opt-out for legacy callers/reproducibility.
    joint_distance_mode
        ``None`` (default): score subsets by mean per-feature KS statistic only, exactly as before -- this
        argument existing at all does not change default-path behavior (bit-identical, see the module's
        biz_value test). ``"energy"``: add a multivariate energy-distance term (see ``_energy_distance``)
        computed over the FULL standardized feature vector, catching joint/correlation-structure mismatches
        that per-feature KS cannot see because it only ever compares one marginal at a time -- e.g. a subset
        whose every feature matches the target's marginal distribution individually but whose features
        co-vary differently (different correlation matrix) than the target's. The combined score is
        ``mean_ks + joint_weight * energy_distance``, still lower-is-better, still used for ranking/selection
        throughout (init phase, block-isolation scoring, and swap acceptance).
    joint_weight
        Weight of the energy-distance term when ``joint_distance_mode="energy"``; unused otherwise.

    Returns
    -------
    dict
        ``{"best_blocks": list, "best_score": float, "all_scores": np.ndarray}`` -- ``best_score`` is the
        LOWEST mean KS statistic found (0 = perfect distributional match). For ``"random"``, ``all_scores``
        has exactly one entry per trial. For ``"greedy_swap"``, ``all_scores`` has one entry per full-subset
        evaluation performed (init-phase trials plus accepted/rejected swap trials) -- its length can be
        shorter than ``n_trials`` because some budget units are spent on single-block scoring, not full-subset
        evaluation.
    """
    cols = list(feature_cols) if feature_cols is not None else [c for c in train_df.select_dtypes(include=[np.number]).columns if c != block_col]

    unique_blocks = train_df[block_col].unique()
    if n_blocks > len(unique_blocks):
        raise ValueError(f"distribution_matching_subset_search: n_blocks ({n_blocks}) exceeds available blocks ({len(unique_blocks)}).")

    if joint_distance_mode is not None and joint_distance_mode != "energy":
        raise ValueError(f"distribution_matching_subset_search: unknown joint_distance_mode {joint_distance_mode!r}, expected None or 'energy'.")

    # Precomputed once (target point cloud is invariant across trials) so the opt-in joint mode adds no
    # per-trial setup cost beyond the energy-distance evaluation itself; None keeps the default path from
    # doing ANY extra work, guaranteeing bit-identical scores to before this feature existed.
    _joint_ctx: Optional[dict] = None
    if joint_distance_mode == "energy":
        target_mat = target_df[cols].to_numpy(dtype=float)
        target_mat = target_mat[np.all(np.isfinite(target_mat), axis=1)]
        target_mean = target_mat.mean(axis=0)
        target_std = target_mat.std(axis=0)
        target_std = np.where(target_std > 1e-12, target_std, 1.0)
        target_std_mat = (target_mat - target_mean) / target_std
        target_target_dist = float(np.mean(np.linalg.norm(target_std_mat[:, None, :] - target_std_mat[None, :, :], axis=2)))
        _joint_ctx = {
            "mean": target_mean,
            "std": target_std,
            "target_std_mat": target_std_mat,
            "target_target_dist": target_target_dist,
        }

    def _score(sample_df: pd.DataFrame) -> float:
        base = _mean_ks_statistic(sample_df, target_df, cols)
        if _joint_ctx is None:
            return base
        sample_mat = sample_df[cols].to_numpy(dtype=float)
        sample_mat = sample_mat[np.all(np.isfinite(sample_mat), axis=1)]
        if len(sample_mat) < 2:
            return base + joint_weight  # degenerate/empty joint sample -- maximally penalize, no crash.
        sample_std_mat = (sample_mat - _joint_ctx["mean"]) / _joint_ctx["std"]
        energy = _energy_distance(sample_std_mat, _joint_ctx["target_std_mat"], _joint_ctx["target_target_dist"])
        return base + joint_weight * energy

    rng = np.random.default_rng(random_state)

    if search_strategy == "random":
        all_scores = np.empty(n_trials, dtype=np.float64)
        best_score = np.inf
        best_blocks: List = []

        for trial in range(n_trials):
            candidate_blocks = rng.choice(unique_blocks, size=n_blocks, replace=False)
            sample_df = train_df[train_df[block_col].isin(candidate_blocks)]
            score = _score(sample_df)
            all_scores[trial] = score
            if score < best_score:
                best_score = score
                best_blocks = list(candidate_blocks)

        return {"best_blocks": best_blocks, "best_score": best_score, "all_scores": all_scores, "joint_distance_mode": joint_distance_mode}

    if search_strategy != "greedy_swap":
        raise ValueError(f"distribution_matching_subset_search: unknown search_strategy {search_strategy!r}, expected 'random' or 'greedy_swap'.")

    # --- greedy_swap: phase 1, pure random init (same mechanics as "random"), spending half the budget. ---
    init_trials = max(1, min(n_trials, int(n_trials * _GREEDY_SWAP_INIT_FRACTION)))
    scores_list: List[float] = []
    best_score = np.inf
    best_blocks = []
    for _ in range(init_trials):
        candidate_blocks = rng.choice(unique_blocks, size=n_blocks, replace=False)
        sample_df = train_df[train_df[block_col].isin(candidate_blocks)]
        score = _score(sample_df)
        scores_list.append(score)
        if score < best_score:
            best_score = score
            best_blocks = list(candidate_blocks)

    # --- phase 2: hill-climb from the best random subset for the remaining budget. ---
    remaining = n_trials - init_trials
    current_blocks: List = list(best_blocks)
    current_score = best_score
    used_blocks = set(current_blocks)
    unused_blocks = [b for b in unique_blocks if b not in used_blocks]
    block_scores: dict = {}  # per-block mean KS vs target_df in isolation, computed lazily and cached.

    spent = 0
    while spent < remaining:
        missing = [b for b in current_blocks if b not in block_scores]
        if missing:
            b = missing[0]
            block_df = train_df[train_df[block_col] == b]
            block_scores[b] = _score(block_df)
            spent += 1
            continue
        if not unused_blocks:
            break  # no unused blocks left to swap in -- stop early, remaining budget goes unused.
        worst_block = max(current_blocks, key=lambda b: block_scores[b])
        replacement = unused_blocks[int(rng.integers(len(unused_blocks)))]
        swapped_blocks: List = [replacement if b == worst_block else b for b in current_blocks]
        candidate_df = train_df[train_df[block_col].isin(swapped_blocks)]
        score = _score(candidate_df)
        scores_list.append(score)
        spent += 1
        if score < current_score:
            current_score = score
            current_blocks = swapped_blocks
            used_blocks.discard(worst_block)
            used_blocks.add(replacement)
            unused_blocks.remove(replacement)
            unused_blocks.append(worst_block)
            del block_scores[worst_block]  # left the subset; rescored lazily if it ever re-enters.
            if score < best_score:
                best_score = score
                best_blocks = list(swapped_blocks)
        # else: swap rejected, swapped_blocks discarded -- current_blocks/current_score stay as-is (revert).

    return {
        "best_blocks": best_blocks,
        "best_score": best_score,
        "all_scores": np.array(scores_list, dtype=np.float64),
        "joint_distance_mode": joint_distance_mode,
    }


__all__ = ["distribution_matching_subset_search"]
