"""Stability selection for discovered composite-target specs.

A single discovery run can pick "lucky" specs that owe their selection to one
particular train sample / seed and do not reproduce. Stability selection (the
Meinshausen-Buhlmann procedure) re-runs the SAME discovery on ``n_replicates``
random row-subsamples of ``train_idx`` and keeps only the specs SELECTED in a
high fraction of replicates -- the robust, reproducible subset.

This module is a *generic* driver: it takes a ``discovery_factory`` callable
(zero-arg, returns a fresh discovery object exposing
``fit(df, target, feature_cols, train_idx).specs_``) and replays it per
replicate on a deterministic subsample. It does NOT depend on
:class:`CompositeTargetDiscovery` directly, so any selector with the same tiny
contract can be stability-screened.

Leakage discipline (CRITICAL)
-----------------------------
Every replicate's ``fit`` is called with a SUBSET of ``train_idx`` only. Val /
test rows are never resampled or passed in -- the factory's ``fit`` reads only
the train rows it is handed. No frame copy is made: each replicate gets a row
INDEX subsample (``np.ndarray`` of positions), never a sliced frame, so a
100+ GB ``df`` is never duplicated.

Group-aware resampling (leave-wells-out)
----------------------------------------
On grouped / panel data (per-entity target -- e.g. 773 wellbore wells) a
ROW-level subsample puts rows of the SAME group in both a replicate and its
complement, so a spec that only "works" by memorising per-group levels is
re-found in every replicate and looks STABLE -- the exact overfit stability
selection is meant to catch leaks through. When a group key is available (an
explicit ``group_ids`` / ``group_column`` arg, or the discovery instance's
``_group_ids_for_rerank``) each replicate instead resamples WHOLE GROUPS: a
``subsample_frac`` fraction of the distinct groups drawn WITHOUT replacement,
keeping every row of the drawn groups. No group is split across a replicate and
its complement, so a spec is only counted stable if it survives on DISJOINT
group subsets -- i.e. it generalises to UNSEEN groups. Leave-a-fraction-of-
groups-out (not bootstrap-of-groups-with-replacement) is chosen because it is
the direct group analog of the Meinshausen-Buhlmann half-subsample the row path
already uses, yields genuinely disjoint group subsets, and avoids the row-
multiplicity distortion a with-replacement cluster bootstrap injects into the
per-replicate transform-param fits and MI estimates. The default is group-aware
whenever a key resolves (gated by ``getattr(config, "stability_group_aware",
True)`` read off the discovery instance); the ROW path is the fallback when no
key is available and stays BIT-IDENTICAL to the pre-feature behaviour.

Profiling
---------
cProfile of the whole sweep (``n_replicates=5``, n_train=40k, 200 groups, a
trivial factory) is 27 ms total; with a REAL factory ~100% of wall is the
per-replicate ``fit`` (MI screening + tiny-model CV, seconds each). The group
draw itself -- ``np.unique`` + ``rng.choice`` + ``np.isin`` + a final
``np.sort`` of the drawn row indices -- is ~1.4 ms/replicate at 40k rows (warm
median), only ~0.3 ms more than the row-level draw (~1.1 ms); both are O(n_train)
and dominated by the sort + the ``np.isin`` membership mask. That is <0.1% of
any real fit, so there is no actionable speedup: the driver is intrinsically
``fit``-bound and the group draw is the cheapest part of each replicate (the
``np.sort`` mirrors the row path's order-preservation and is kept for parity).
See ``_profile_stability_sweep`` in the test module for the runnable harness.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# Default fraction of train rows drawn (without replacement) per replicate.
# Meinshausen-Buhlmann draw a random HALF of the rows; 0.5 is the canonical
# default and decorrelates replicates far better than a reseed of the full
# sample (a spec found on a sample is almost always re-found on that same
# sample, so full-sample reseeds barely filter anything).
_DEFAULT_SUBSAMPLE_FRAC = 0.5

# Default selection-frequency threshold for the stable subset. A spec must be
# picked in >= this fraction of replicates to survive. 0.6 = a 3/5 majority at
# the default replicate count -- standard in the stability-selection literature.
_DEFAULT_FREQ_THRESHOLD = 0.6


@dataclass
class StabilityResult:
    """Outcome of :func:`stability_select_specs`.

    Attributes
    ----------
    frequencies
        ``{spec_name -> selection_frequency}`` over all SUCCESSFUL replicates,
        each value in ``[0, 1]``.
    stable_specs
        Spec names with ``frequency >= freq_threshold``, sorted by frequency
        descending then name (deterministic order).
    spec_by_name
        One representative spec OBJECT per surviving name (the first replicate
        that produced it), so callers can hand the stable subset straight back
        to a downstream consumer without re-running discovery.
    n_replicates
        Number of replicates actually run.
    n_successful
        Number of replicates whose ``fit`` did not raise (the denominator of
        every frequency).
    freq_threshold
        The threshold applied to derive ``stable_specs``.
    group_aware
        ``True`` when replicates resampled whole GROUPS (leave-wells-out),
        ``False`` for the ROW-level fallback (no group key / single group).
    """

    frequencies: Dict[str, float]
    stable_specs: List[str]
    spec_by_name: Dict[str, Any] = field(default_factory=dict)
    n_replicates: int = 0
    n_successful: int = 0
    freq_threshold: float = _DEFAULT_FREQ_THRESHOLD
    group_aware: bool = False

    @property
    def stable_spec_objects(self) -> List[Any]:
        """Representative spec objects for the stable subset, same order as
        :attr:`stable_specs`."""
        return [self.spec_by_name[n] for n in self.stable_specs if n in self.spec_by_name]


def _subsample_indices(
    train_idx: np.ndarray, frac: float, rng: np.random.Generator,
) -> np.ndarray:
    """Draw a ``frac`` slice of ``train_idx`` WITHOUT replacement.

    Sorted to preserve any time / order semantics the caller's ``train_idx``
    carried (downstream ``fit`` reads rows positionally). Clamped to ``[2, n]``:
    a <2-row subsample cannot fit transform params, so we fall back to the full
    train index in that pathological case.
    """
    n = int(train_idx.size)
    if frac >= 1.0:
        return train_idx
    sub_n = max(2, round(frac * n))
    sub_n = min(sub_n, n)
    if sub_n >= n:
        return train_idx
    return np.sort(rng.choice(train_idx, size=sub_n, replace=False))


def _pull_group_column(df: Any, col: str) -> np.ndarray | None:
    """Pull ``col`` from ``df`` as a RAW 1-D ndarray (no float32 cast).

    Group labels may be strings / large ints; unlike the numeric feature pull
    (``_extract_column_array``) we keep the native dtype so ``np.unique`` /
    ``np.isin`` compare true labels. Duck-typed (polars ``get_column`` else
    pandas ``[]``) to avoid a hard import here; returns ``None`` on any failure
    so a bad ``group_column`` degrades to the row-level path, never raises.
    """
    getcol = getattr(df, "get_column", None)  # polars DataFrame
    try:
        if callable(getcol):
            return np.asarray(getcol(col).to_numpy())
        return np.asarray(df[col].to_numpy())
    except Exception as exc:
        logger.warning("[stability_select_specs] group_column %r unreadable: %s", col, exc)
        return None


def _align_group_labels(group_ids: Any, train_idx: np.ndarray) -> np.ndarray | None:
    """One group label per ``train_idx`` row, or ``None`` if unalignable.

    Accepts either a per-train-row vector (length == ``train_idx.size``, used
    as-is) or a full-frame vector (indexed positionally by ``train_idx``) --
    the same convention as ``split_screening_holdout`` so the two group-aware
    paths agree on how a caller's ``group_ids`` maps to rows.
    """
    if group_ids is None:
        return None
    g = np.asarray(group_ids)
    if g.ndim != 1:
        return None
    n = int(train_idx.size)
    if g.shape[0] == n:
        return g
    try:
        return np.asarray(g[train_idx])
    except (IndexError, TypeError):
        return None


def _subsample_groups(
    train_idx: np.ndarray, group_labels: np.ndarray, frac: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a ``frac`` fraction of whole GROUPS (leave-wells-out) and return
    the row indices of the drawn groups.

    ``group_labels`` is aligned to ``train_idx`` (one label per train row). A
    ``frac`` fraction of the DISTINCT groups is drawn WITHOUT replacement and
    every row of the drawn groups is kept, so no group is split across a
    replicate and its complement -- the group analog of the M-B half-subsample.
    Sorted like the row path to preserve any time / order semantics. Falls back
    to the full train index when ``frac >= 1``, when there are <2 distinct
    groups (nothing to leave out), or when the draw would yield <2 rows (a
    degenerate replicate cannot fit transform params).
    """
    if frac >= 1.0:
        return train_idx
    uniq = np.unique(group_labels)
    n_groups = int(uniq.size)
    if n_groups < 2:
        return train_idx
    sub_g = max(1, round(frac * n_groups))
    sub_g = min(sub_g, n_groups)
    if sub_g >= n_groups:
        return train_idx
    drawn = rng.choice(uniq, size=sub_g, replace=False)
    sub_pos = np.nonzero(np.isin(group_labels, drawn))[0]
    if sub_pos.size < 2:
        return train_idx
    return np.sort(train_idx[sub_pos])


def _resolve_group_ids(
    group_ids: Any, group_column: str | None, df: Any, probe: Any,
) -> np.ndarray | None:
    """Resolve the group key from the explicit args first, then the discovery
    instance's ``_group_ids_for_rerank``.

    Precedence: an explicit ``group_ids`` array wins; else a ``group_column``
    pulled from ``df``; else the probe discovery object's
    ``_group_ids_for_rerank`` (the same attr the honest-holdout / tiny-rerank
    group-aware paths read). ``None`` when no key is available -> row fallback.
    """
    if group_ids is not None:
        return np.asarray(group_ids)
    if group_column:
        pulled = _pull_group_column(df, group_column)
        if pulled is not None:
            return pulled
    if probe is not None:
        g = getattr(probe, "_group_ids_for_rerank", None)
        if g is not None:
            return np.asarray(g)
    return None


def stability_select_specs(
    discovery_factory: Callable[[], Any],
    df: Any,
    target: str,
    feature_cols: Sequence[str],
    train_idx: np.ndarray,
    *,
    n_replicates: int = 5,
    subsample_frac: float = _DEFAULT_SUBSAMPLE_FRAC,
    freq_threshold: float = _DEFAULT_FREQ_THRESHOLD,
    random_state: int = 0,
    group_ids: Any = None,
    group_column: str | None = None,
    group_aware: bool | None = None,
) -> StabilityResult:
    """Stability-select discovered specs across bootstrap row-subsample replicates.

    Re-runs ``discovery_factory().fit(df, target, feature_cols, sub_idx)`` for
    ``n_replicates`` deterministic subsamples of ``train_idx`` and returns each
    spec's selection FREQUENCY plus the stable subset (frequency >=
    ``freq_threshold``).

    Parameters
    ----------
    discovery_factory
        Zero-arg callable returning a FRESH discovery object per call. The
        object must expose ``fit(df, target, feature_cols, train_idx)`` and,
        post-fit, a ``specs_`` iterable of objects each with a ``.name``
        attribute (e.g. :class:`CompositeTargetDiscovery`). A fresh object per
        replicate guarantees no fitted state bleeds across replicates.
    df
        Frame containing ``target`` + ``feature_cols``. Never copied -- only
        row-index subsamples are passed to ``fit``.
    target, feature_cols, train_idx
        Forwarded to each replicate's ``fit``. ``train_idx`` is the population
        the per-replicate subsample is drawn from; val/test rows are out of
        scope here (fit reads train rows only -- no leakage).
    n_replicates
        Number of subsample replicates to run (>= 1).
    subsample_frac
        Fraction of ``train_idx`` drawn (without replacement) per replicate.
        ``1.0`` recovers a reseed-only (no-subsample) behaviour.
    freq_threshold
        Minimum selection frequency for a spec to enter ``stable_specs``.
    random_state
        Seeds the per-replicate subsample RNGs deterministically: same inputs +
        same ``random_state`` -> identical frequencies.
    group_ids
        Optional group key: either a per-``train_idx``-row vector or a full-frame
        vector (indexed positionally by ``train_idx``). When present (and group-
        aware is on) each replicate resamples WHOLE GROUPS instead of rows.
    group_column
        Optional name of a ``df`` column to read the group key from (used only
        when ``group_ids`` is not given). Unreadable column -> row fallback.
    group_aware
        Tri-state. ``None`` (default) reads ``getattr(config,
        "stability_group_aware", True)`` off a probe discovery instance and
        engages group resampling whenever a key resolves. ``True`` / ``False``
        force / disable it. Group resampling only actually engages when a key
        resolves to >= 2 distinct groups; otherwise the ROW path runs (and is
        bit-identical to the no-group behaviour).

    Returns
    -------
    StabilityResult
        ``.frequencies`` (all in ``[0, 1]``), ``.stable_specs``, plus the
        representative spec objects + run bookkeeping.
    """
    train_idx = np.asarray(train_idx)
    n_replicates = max(1, int(n_replicates))
    frac = float(subsample_frac)
    # Independent, reproducible per-replicate RNGs via SeedSequence spawning --
    # no arithmetic stride that could collide with an inner multi-seed ladder.
    seed_seqs = np.random.SeedSequence(int(random_state)).spawn(n_replicates)

    # A single probe instance (discarded) supplies the config toggle + the
    # discovery-side ``_group_ids_for_rerank`` fallback. Built only when the
    # group decision is not already fully pinned by the explicit args, so a
    # caller passing ``group_aware=False`` pays nothing. Never touches the RNG
    # sequence -> the row path stays bit-identical regardless.
    probe = None
    if group_aware is None or (group_aware and group_ids is None and not group_column):
        try:
            probe = discovery_factory()
        except Exception as exc:
            logger.warning("[stability_select_specs] probe factory() failed: %s", exc)
    if group_aware is None:
        group_aware = bool(getattr(getattr(probe, "config", None), "stability_group_aware", True))

    group_labels = None
    if group_aware:
        resolved = _resolve_group_ids(group_ids, group_column, df, probe)
        aligned = _align_group_labels(resolved, train_idx)
        # Leave-groups-out needs >= 2 distinct groups; a single (or absent) group
        # cannot leave a well out, so fall back to the row-level path there.
        if aligned is not None and np.unique(aligned).size >= 2:
            group_labels = aligned
    use_group = group_labels is not None

    keep_counter: Counter = Counter()
    spec_by_name: Dict[str, Any] = {}
    n_successful = 0
    for i in range(n_replicates):
        rng = np.random.default_rng(seed_seqs[i])
        if use_group:
            sub_idx = _subsample_groups(train_idx, group_labels, frac, rng)  # type: ignore[arg-type]  # group_labels is not None per use_group guard, mypy can't narrow the bool flag
        else:
            sub_idx = _subsample_indices(train_idx, frac, rng)
        try:
            discovery = discovery_factory()
            discovery.fit(df, target, list(feature_cols), sub_idx)
        except Exception as exc:
            logger.warning(
                "[stability_select_specs] replicate %d/%d failed: %s",
                i + 1, n_replicates, exc,
            )
            continue
        n_successful += 1
        seen_this_run: set[str] = set()
        for spec in getattr(discovery, "specs_", None) or []:
            name = spec.name
            # Count each spec name at most once per replicate (a replicate is a
            # Bernoulli selection trial -- duplicates within one fit would skew
            # the frequency above 1.0).
            if name in seen_this_run:
                continue
            seen_this_run.add(name)
            keep_counter[name] += 1
            spec_by_name.setdefault(name, spec)

    denom = max(1, n_successful)
    frequencies = {name: count / denom for name, count in keep_counter.items()}
    stable_specs = sorted(
        (n for n, f in frequencies.items() if f >= freq_threshold),
        key=lambda n: (-frequencies[n], n),
    )
    logger.info(
        "[stability_select_specs] n_replicates=%d (successful=%d), frac=%.2f, "
        "threshold=%.2f, group_aware=%s. Stable: %d/%d spec(s). Frequencies: %s",
        n_replicates, n_successful, frac, freq_threshold, use_group,
        len(stable_specs), len(frequencies),
        {k: round(v, 3) for k, v in frequencies.items()},
    )
    return StabilityResult(
        frequencies=frequencies,
        stable_specs=stable_specs,
        spec_by_name=spec_by_name,
        n_replicates=n_replicates,
        n_successful=n_successful,
        freq_threshold=freq_threshold,
        group_aware=use_group,
    )
