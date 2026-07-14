"""``CompositeTargetDiscovery.fit_with_stability_check`` -- carved out of ``discovery/__init__.py``
to keep the facade file under its 750-LOC budget (see ``test_composite_discovery_facade.py``).
Bound onto the class at the bottom of ``discovery/__init__.py`` (same pattern as ``_auto_base`` /
``_tiny_model_rerank`` / ``fit`` / ``_filter_features`` / ``fit_stacked*``).
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from ..ensemble import derive_seeds
from ..spec import CompositeSpec

if TYPE_CHECKING:
    from . import CompositeTargetDiscovery

logger = logging.getLogger(__name__)


def fit_with_stability_check(
    self: "CompositeTargetDiscovery",
    df: Any,
    target_col: str,
    feature_cols: Sequence[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray | None = None,
    test_idx: np.ndarray | None = None,
    *,
    n_bootstrap_runs: int = 5,
    min_keep_fraction: float = 0.6,
    subsample_fraction: float = 0.5,
) -> "CompositeTargetDiscovery":
    """Run :meth:`fit` ``n_bootstrap_runs`` times on DECORRELATED reseeds and per-run row subsamples, keeping only specs that survive in at least ``min_keep_fraction * n_bootstrap_runs`` runs.

    Filters "lucky split" wins where a single seed happens to find a spec that does not generalise. Default thresholds (5 runs, 60% majority, 50% subsample) match the standard stability-selection literature (Meinshausen-Buhlmann), whose procedure draws each replicate on a *random half* of the rows -- not merely a reseed of the same sample.

    Returns ``self``. After the call, ``self.specs_`` is the stable subset and ``self.stability_counts_`` maps each name to its survival count.

    Decorrelation rationale
    -----------------------
    Two defects made the pre-fix "bootstrap" runs near-duplicates rather
    than independent replicates, so the gate barely filtered anything:

    1. **Seed-stride collision.** The per-run stride was
       ``base_seed + i*7919``. The inner multi-seed sweep
       (``_screening_tiny._tiny_cv_rmse_*_multiseed``) strides the SAME
       7919 as ``base_random_state + s_idx*7919``. So run ``i``'s reseed
       landed exactly on run ``i-1``'s second inner seed -> the
       "independent" runs shared their CV draws on a 7919-aligned ladder,
       correlating the very replicates the gate assumes are independent.
       Fixed by deriving each run's master seed via the sha256-based
       :func:`derive_seeds` (no arithmetic relationship to the inner
       ``*7919`` ladder), which cannot collide with the multi-seed stride.
    2. **No row subsample.** Every run reused the *identical* ``train_idx``,
       so the only variation was the seed -- a spec found on one sample was
       almost always re-found on the same sample. Meinshausen-Buhlmann
       stability selection draws each replicate on a random subsample of
       the rows; we now draw a ``subsample_fraction`` (default 0.5) slice of
       ``train_idx`` per run with a per-run-seeded RNG. ``val_idx`` /
       ``test_idx`` are passed through untouched (never resampled -- fit
       only ever reads ``train_idx`` rows). Set ``subsample_fraction=1.0``
       to recover the legacy reseed-only behaviour.

    Perf note: the decorrelation additions are a per-call ``derive_seeds``
    (n_runs sha256 hashes) plus one ``np.random.choice(replace=False)+sort``
    per run. Measured ~400 us total for 5 runs at n_train=400 and ~16 ms for
    a single 50% draw at n_train=400k -- negligible vs one ``fit()`` (MI
    screening + tiny-model CV over the whole sample, seconds). No actionable
    speedup; the draw is intrinsically O(n_train) and is the cheapest part of
    each replicate.
    """
    if n_bootstrap_runs <= 1:
        return self.fit(df, target_col, feature_cols, train_idx, val_idx, test_idx)

    # Per-run reseeding (and any mid-fit heavy-tail mi_n_strata boost that swaps self.config for a model_copy) must mutate only a config we own:
    # otherwise the final restore would write back the swapped copy and leave the caller's shared config permanently reseeded, poisoning later targets.
    _saved_cfg = self.config
    self.config = self.config.model_copy()
    base_seed = int(self.config.random_state)
    # Decorrelate run seeds from the inner multi-seed ``*7919`` ladder
    # (defect 1 above): sha256-derive one master seed per run keyed on the
    # base seed + run index, so no run's reseed can land on another run's
    # inner CV seed. Masked to int32 to stay a valid numpy/config seed.
    _run_seeds = derive_seeds(base_seed, [f"stability_run_{i}" for i in range(int(n_bootstrap_runs))])
    train_idx = np.asarray(train_idx)
    _frac = float(subsample_fraction)
    keep_counter: Counter = Counter()
    spec_by_name: dict[str, CompositeSpec] = {}
    # Group-aware resampling: on grouped data resample whole GROUPS (leave-wells-out), not rows -- a row draw puts a
    # group's rows in both the replicate and its complement, so a spec that only memorised per-group levels looks
    # stable and the stability check leaks the very overfit it exists to catch. No group key -> the row draw below,
    # bit-identical to the prior behaviour. Reuses the same helpers as ``stability_select_specs``.
    from ._stability import _align_group_labels, _resolve_group_ids, _subsample_groups
    _stab_group_aware = bool(getattr(self.config, "stability_group_aware", True))
    _group_labels_train = None
    if _stab_group_aware:
        _grp_full = _resolve_group_ids(None, getattr(self.config, "group_column", None), df, self)
        _aligned = _align_group_labels(_grp_full, train_idx) if _grp_full is not None else None
        if _aligned is not None and np.unique(_aligned).size >= 2:
            _group_labels_train = _aligned
    # Carve the honest holdout ONCE for the whole sweep (shared across replicates). Per-replicate
    # carves each drew a fresh 20% of that run's subsample, so every replicate's holdout landed
    # inside other replicates' screening pools -- no row set stayed "never touched" sweep-wide,
    # and each run paid a redundant carve. Now the holdout is fixed up front, every replicate
    # subsamples the SCREEN pool only, and ``carve_screening_holdout`` reuses the shared indices
    # (see ``_stability_shared_holdout_idx`` in ``_honest_holdout``). Seeded by the base seed,
    # group-aware when a key resolves -- identical carve semantics to a single fit.
    from ._honest_holdout import split_screening_holdout
    _screen_pool, _shared_holdout = split_screening_holdout(
        train_idx,
        getattr(self.config, "honest_holdout_frac", 0.2),
        base_seed,
        group_ids=_group_labels_train,
    )
    self._stability_shared_holdout_idx = _shared_holdout if _shared_holdout is not None else np.empty(0, dtype=train_idx.dtype)
    _pool_n = int(_screen_pool.size)
    # M-B subsample size is ``subsample_fraction`` of the CALLER's original ``train_idx`` (the
    # documented public contract: "a subsample_fraction slice of train_idx"), capped at the
    # screen-pool size (the shared holdout carve above removed ``honest_holdout_frac`` of
    # train_idx up front, so the pool can never supply more rows than it holds). A prior version
    # computed this fraction against the ALREADY-reduced pool instead of the original train_idx,
    # silently shrinking every replicate by an extra ~honest_holdout_frac (e.g. frac=0.5 on 400
    # rows drew 160, not the documented round(0.5*400)=200) -- fixed here; see
    # test_m3_runs_use_distinct_row_subsamples. frac=1.0 ("legacy full-sample opt-out") is capped
    # to the full screen pool (NOT bit-identical to the original train_idx): a real subsample of
    # 100% cannot also include the shared honest holdout without leaking it into every replicate's
    # training data, so the opt-out's true ceiling is "everything except the shared holdout", not
    # "everything" -- see test_m3_legacy_full_sample_opt_out's updated assertion.
    _pool_sub_n = min(max(2, round(_frac * train_idx.size)), _pool_n)
    _group_labels_pool = None
    if _group_labels_train is not None:
        _pool_pos = np.isin(train_idx, _screen_pool)
        _group_labels_pool = _group_labels_train[_pool_pos]
    try:
        for i in range(int(n_bootstrap_runs)):
            _run_seed = int(_run_seeds[f"stability_run_{i}"]) & 0x7FFFFFFF
            self.config.random_state = _run_seed
            # Per-run subsample (defect 2 above) drawn from the SCREEN POOL (never the shared
            # holdout): a dedicated RNG seeded from the decorrelated run seed keeps the draw
            # reproducible per base seed while making each run a genuinely different population.
            # Sorted to preserve any time/order semantics train_idx carried.
            if _pool_sub_n < _pool_n:
                _run_rng = np.random.default_rng(_run_seed)
                if _group_labels_pool is not None:
                    _run_train_idx = _subsample_groups(_screen_pool, _group_labels_pool, _frac, _run_rng)
                else:
                    _run_train_idx = np.sort(_run_rng.choice(_screen_pool, size=_pool_sub_n, replace=False))
            else:
                _run_train_idx = _screen_pool
            try:
                self.fit(df, target_col, feature_cols, _run_train_idx, val_idx, test_idx)
            except Exception as _exc:
                logger.warning(
                    "[CompositeTargetDiscovery.stability] bootstrap run %d failed: %s",
                    i, _exc,
                )
                continue
            for spec in self.specs_:
                keep_counter[spec.name] += 1
                spec_by_name.setdefault(spec.name, spec)
    finally:
        # Restore the caller's original config and drop the shared-holdout marker so later
        # standalone ``fit`` calls carve normally.
        self.config = _saved_cfg
        self._stability_shared_holdout_idx = None
    threshold = max(1, int(min_keep_fraction * n_bootstrap_runs))
    stable_names = [n for n, c in keep_counter.items() if c >= threshold]
    self.specs_ = [spec_by_name[n] for n in stable_names if n in spec_by_name]
    self.stability_counts_ = dict(keep_counter)
    logger.info(
        "[CompositeTargetDiscovery.stability] n_runs=%d, threshold=%d/%d, "
        "subsample=%d/%d rows (frac=%.2f). Kept %d spec(s); counts: %s",
        n_bootstrap_runs, threshold, n_bootstrap_runs,
        _pool_sub_n, _pool_n, _frac,
        len(self.specs_), dict(keep_counter),
    )
    return self
