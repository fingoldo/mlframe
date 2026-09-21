"""Masking observed values after the link has consumed the complete ones.

Missingness is applied to the emitted frame and to nothing else. The link reads complete values, the
calibration bisects on complete values, and ``true_prob`` is the law relating complete values to the
target. Masking afterwards leaves that law untouched and changes only what a method can see.

That ordering is a decision, not an implementation detail, and it is the one that keeps the ceiling
honest. If the mask were applied before the link, the recorded ``true_prob`` would be the law of a
half-observed world, and the generator would have to say what a missing value contributes to a score --
a question with no answer that is not itself a modelling choice. With the mask last, the recorded ceiling
is the COMPLETE-data ceiling, which is exactly defined, and the observed-data ceiling is lower by an
amount the generator does not claim to know. The truth record says so in a caveat instead of reporting a
number nobody computed.

Sentinel values are never used. A masked cell is ``NaN``, and integer and categorical columns are widened
to hold it rather than being given an out-of-range code: a ``-999`` in an integer column is a value some
method will happily treat as a very small number, which turns a missingness bed into an outlier bed.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np

from mlframe.data.datasets._rng import stream_for
from mlframe.data.datasets.spec import MissingnessSpec, resolve_knob

logger = logging.getLogger(__name__)

__all__ = ["missing_mask", "apply_missingness"]


def missing_mask(spec: MissingnessSpec, columns: Dict[str, np.ndarray], rate: float, rng: np.random.Generator) -> np.ndarray:
    """Return the boolean mask of rows whose value of ``spec.column`` is hidden.

    Every mechanism hides the same EXPECTED share of rows, so a comparison between them is a comparison of
    which rows went missing and never of how many. Without that, a bed would confound the mechanism with
    the amount, and the amount is by far the stronger effect.

    Args:
        spec: The declaration.
        columns: Realised complete columns.
        rate: The resolved share of rows to hide.
        rng: Stream for the draw.

    Returns:
        A boolean array, one entry per row, true where the value is to be hidden.

    Raises:
        KeyError: If the declaration names a column the dataset does not contain.
    """
    if spec.column not in columns:
        raise KeyError(f"missingness references column {spec.column!r}, which the dataset does not contain")
    n = columns[spec.column].shape[0]
    share = float(np.clip(rate, 0.0, 1.0))
    if share <= 0.0:
        return np.zeros(n, dtype=bool)

    if spec.mechanism == "mcar":
        return np.asarray(rng.random(n) < share, dtype=bool)

    if spec.mechanism == "mar":
        driver = spec.driver or ""
        if driver not in columns:
            raise KeyError(f"missingness on {spec.column!r} is driven by {driver!r}, which the dataset does not contain")
        ordering = columns[driver]
    else:
        # MNAR: the masked column's own magnitude decides. The largest values go missing BECAUSE they are
        # large, which is what makes the bias unidentifiable from the observed data.
        ordering = columns[spec.column]

    # Deterministic top-share selection rather than a probability increasing in the driver. Both are
    # legitimate MAR/MNAR mechanisms, but the deterministic one gives the stated rate exactly and makes the
    # mechanism's shape unambiguous: everything above a quantile is gone, nothing below it is.
    cut = float(np.quantile(ordering, 1.0 - share))
    return np.asarray(ordering >= cut, dtype=bool)


def apply_missingness(
    declarations: Sequence[MissingnessSpec],
    columns: Dict[str, np.ndarray],
    root_seed: int,
    spec_name: str,
    knob_rng: np.random.Generator,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Mask the declared columns in place-ish, returning the new columns and the caveats they earn.

    Args:
        declarations: What to mask and how.
        columns: Realised complete columns; not mutated, a copy is returned.
        root_seed: Root seed for the name-addressed stream.
        spec_name: Dataset name, part of the stream path.
        knob_rng: Stream for resolving a prior-valued rate.

    Returns:
        ``(columns_with_holes, caveats)``. The caveats are the sentences the truth record has to carry:
        each one names a column, its mechanism and its realised rate, and says that the recorded ceiling
        is the complete-data one.
    """
    if not declarations:
        return columns, []

    out = dict(columns)
    caveats: List[str] = []
    for declaration in declarations:
        rate = float(resolve_knob(declaration.rate, knob_rng))
        stream = stream_for(root_seed, spec_name, "missingness", declaration.column)
        mask = missing_mask(declaration, columns, rate, stream)
        values = np.asarray(out[declaration.column], dtype=np.float64).copy()
        values[mask] = np.nan
        out[declaration.column] = values
        realised = float(np.mean(mask))
        caveats.append(
            f"{declaration.column!r} is {realised:.1%} missing under {declaration.mechanism.upper()}"
            + (f" driven by {declaration.driver!r}" if declaration.mechanism == "mar" else "")
            + "; the recorded ceiling is the COMPLETE-data ceiling and the observed-data ceiling is lower by an amount this generator does not compute"
        )
        logger.debug("masked %.1f%% of %s under %s", realised * 100.0, declaration.column, declaration.mechanism)
    return out, caveats
