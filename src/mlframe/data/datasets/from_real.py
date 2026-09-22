"""Partial ground truth on real data, by injecting probes drawn to look exactly like the real columns.

Real data has no answer key. That is the whole difficulty with the real leg of this benchmark: it can say
whether selection pays, and it cannot say what a method recovered, because nobody knows what the right
answer is. Synthetic beds have an answer key and nobody believes they resemble the problems people have.

There is one construction that gets part of both, and it is the design behind the NIPS 2003 feature
selection challenge: take a real dataset and ADD columns that are known to be irrelevant. Nothing is known
about the real columns -- some are informative, some are not, and that stays unknown -- but the injected
ones are irrelevant by construction, because they were drawn without ever looking at the target.

That gives a one-sided answer key, and one side is enough for the question that matters most:

* **false discovery is measurable.** Every injected column a method selects is a false positive, known to
  be one. A method selecting them at a rate above its own declared FDR is caught, on real data.
* **recall is not measurable and is not reported.** How much of the real signal a method found remains
  unknown; claiming otherwise from this construction would be the error it exists to avoid.

The probes have to be indistinguishable from the real columns on everything except their relationship to
the target. A probe drawn from a standard normal beside real columns with heavy tails and point masses is
findable by its marginal alone -- the selector never needs to look at the target, the false-positive rate
comes out flattering, and the bed has measured nothing. So each probe is drawn by RESAMPLING a real
column's own values, which reproduces its marginal exactly (every point mass, every tail, every repeated
value) and destroys its dependence with the target and with the other columns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from mlframe.data.datasets._rng import stream_for

logger = logging.getLogger(__name__)

__all__ = ["PROBE_PREFIX", "RealBedWithProbes", "inject_matched_probes", "probe_false_discovery_rate"]

#: Prefix every injected column carries. Read by the scorer, so it is a constant rather than a convention:
#: a bed whose probes were named by a caller could collide with a real column name and make a real column
#: count as a known false positive.
PROBE_PREFIX = "__probe_"


@dataclass(frozen=True)
class RealBedWithProbes:
    """A real frame with known-irrelevant columns added, and the one-sided truth that buys."""

    frame: pd.DataFrame
    target: np.ndarray
    probe_columns: Tuple[str, ...]
    real_columns: Tuple[str, ...]
    source: str

    def truth(self) -> Dict[str, Any]:
        """Return the truth dictionary the harness reads, stating plainly what is and is not known.

        There is no ``base`` key and that is deliberate. The harness treats ``base`` as the answer key and
        computes recall against it; a key listing the real columns would claim they are all relevant, and
        a key listing none would claim none are. Neither is known. What IS known travels under its own
        names, and every consumer has to ask for it explicitly.
        """
        return {
            "known_irrelevant": list(self.probe_columns),
            "unknown_relevance": list(self.real_columns),
            "declared_target_size": None,
            "source": self.source,
            "recall_is_unmeasurable": True,
            "note": "injected probes are irrelevant by construction; the real columns' relevance is unknown, so precision against the probes is measurable and recall is not",
        }


def _resample_column(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return a column with the same marginal and no dependence on anything.

    Sampling WITH replacement from the column's own values, which reproduces its empirical distribution
    exactly: every point mass, every tail, every repeated value, and the same dtype range. A parametric
    fit would smooth all of that away and leave a probe a marginal test could separate from the real
    columns without ever consulting the target.
    """
    finite = values[np.isfinite(values)] if np.issubdtype(values.dtype, np.floating) else values
    if finite.size == 0:
        return np.zeros_like(values)
    return np.asarray(rng.choice(finite, size=values.shape[0], replace=True), dtype=values.dtype)


def inject_matched_probes(
    frame: pd.DataFrame,
    target: np.ndarray,
    n_probes: int,
    seed: int = 0,
    source: str = "unnamed",
    columns_to_match: Optional[List[str]] = None,
) -> RealBedWithProbes:
    """Add ``n_probes`` known-irrelevant columns whose marginals match the real ones.

    Args:
        frame: The real feature frame; not modified.
        target: The real labels; used only for their length, never for drawing a probe.
        n_probes: How many probes to add. They cycle through the real columns, so the injected set carries
            the same MIX of marginal shapes as the bed does rather than repeating one shape.
        seed: Root seed; every probe draws from its own name-addressed stream, so adding a probe leaves the
            others bit-identical.
        source: Name of the real dataset, recorded in the truth.
        columns_to_match: Restrict the marginals probes are drawn from; defaults to every numeric column.

    Returns:
        A :class:`RealBedWithProbes`.

    Raises:
        ValueError: If the frame has no numeric column to draw a marginal from, since a probe with no
            marginal to match would have to be invented and would be findable by its shape alone.
    """
    numeric = list(columns_to_match) if columns_to_match is not None else [str(column) for column in frame.select_dtypes(include=["number"]).columns]
    if not numeric:
        raise ValueError(f"{source!r} has no numeric column whose marginal a probe could match")

    out = frame.copy()
    probe_names: List[str] = []
    for index in range(max(0, int(n_probes))):
        donor = numeric[index % len(numeric)]
        name = f"{PROBE_PREFIX}{index:04d}"
        stream = stream_for(seed, source, "matched_probe", name)
        out[name] = _resample_column(np.asarray(frame[donor].to_numpy()), stream)
        probe_names.append(name)

    logger.info("injected %d matched probes into %r (%d real columns)", len(probe_names), source, frame.shape[1])
    return RealBedWithProbes(
        frame=out,
        target=np.asarray(target),
        probe_columns=tuple(probe_names),
        real_columns=tuple(str(column) for column in frame.columns),
        source=source,
    )


def probe_false_discovery_rate(selected: List[str], probe_columns: Tuple[str, ...]) -> Optional[float]:
    """Return the share of a selection that is known to be wrong, or ``None`` when nothing was selected.

    This is a LOWER bound on the false discovery rate, not an estimate of it: some of the real columns a
    method selected are probably wrong too, and nothing here can say which. Reported as a bound and named
    as one, because a bound that gets quoted as an estimate is worse than no number.
    """
    if not selected:
        return None
    known_bad = sum(1 for name in selected if str(name) in set(probe_columns))
    return float(known_bad) / float(len(selected))
