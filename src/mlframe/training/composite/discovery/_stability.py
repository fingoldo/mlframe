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
    """

    frequencies: Dict[str, float]
    stable_specs: List[str]
    spec_by_name: Dict[str, Any] = field(default_factory=dict)
    n_replicates: int = 0
    n_successful: int = 0
    freq_threshold: float = _DEFAULT_FREQ_THRESHOLD

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
    sub_n = max(2, int(round(frac * n)))
    sub_n = min(sub_n, n)
    if sub_n >= n:
        return train_idx
    return np.sort(rng.choice(train_idx, size=sub_n, replace=False))


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

    keep_counter: Counter = Counter()
    spec_by_name: Dict[str, Any] = {}
    n_successful = 0
    for i in range(n_replicates):
        rng = np.random.default_rng(seed_seqs[i])
        sub_idx = _subsample_indices(train_idx, frac, rng)
        try:
            discovery = discovery_factory()
            discovery.fit(df, target, list(feature_cols), sub_idx)
        except Exception as exc:  # noqa: BLE001 -- one bad replicate must not abort the screen
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
        "threshold=%.2f. Stable: %d/%d spec(s). Frequencies: %s",
        n_replicates, n_successful, frac, freq_threshold,
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
    )
