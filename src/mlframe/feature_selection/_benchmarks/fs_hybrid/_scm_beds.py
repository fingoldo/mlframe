"""The SCM scenario library, adapted to the benchmark harness's bed interface.

The adversarial beds in this package build their data and hand-write their truth dictionary alongside it.
That works, and it is how every existing bed here was written, but it puts the answer key and the data
generator in two places that a later edit can silently separate -- which is exactly the failure that left
one bed's "collider" a plain noise column while its truth dict still called it a collider.

`mlframe.data.datasets` derives the answer key FROM the graph that generated the data, so the two cannot
drift. This module exposes those scenarios in the shape the runner already consumes, which means they get
the whole existing protocol for free: matched-K cuts, the all-features null, the panel, the paired
statistics, the reliability accounting.

Three details of the translation are worth stating, because each is a decision rather than plumbing:

* the answer key is the **Markov blanket**, matching the pre-registered primary target set. Truth's `base`
  is the blanket, which is also what fixes the matched-K grid at 1x, 2x and 5x the blanket's size.
* `expected_to_break` travels with the bed, so the coverage meta-test sees these beds the same way it sees
  the hand-written ones.
* the calibrated ceiling is carried in `notes`, because a downstream AUC only means something against the
  AUC that was achievable -- 0.72 is a poor result on a bed calibrated to 0.90 and an excellent one on a
  bed calibrated to 0.75.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["SCM_BED_ROWS", "scm_bed_scenarios", "build_scm_bed"]

# A uniform override a caller may pass; NOT a default. Each scenario declares the size its structure needs,
# and a bed named for its row count is the clearest case of a size that is part of the design.
SCM_BED_ROWS = 4000


def build_scm_bed(name: str, seed: int, n_samples: Optional[int] = None) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    """Build one SCM scenario in the harness's `(X, y, truth)` shape.

    Args:
        name: A scenario registered in `mlframe.data.datasets.scenarios`.
        seed: Dataset seed; the development and reserved ranges are the caller's discipline.
        n_samples: Rows to generate, or ``None`` to keep the size the scenario itself declares. A bed whose
            whole point is its size -- `linear_gaussian_lowdim_n200` exists to be 200 rows, where a
            t-statistic beats a binned mutual-information estimate -- must not be silently resized by a
            caller that wanted a uniform grid.

    Returns:
        The frame, the labels, and the truth dictionary the runner and the analysis layer read.
    """
    from mlframe.data.datasets import scenarios
    from mlframe.data.datasets.generator import generate

    scenario = scenarios.get(name)
    spec = scenario.build(seed=seed)
    if n_samples is not None and int(n_samples) != spec.n_samples:
        spec = spec.model_copy(update={"n_samples": int(n_samples)})
    dataset = generate(spec)

    blanket = list(dataset.truth.primary_target_set().members)
    columns = [str(column) for column in dataset.frame.columns]
    noise = [column for column in columns if column not in set(blanket)]

    ceiling = dataset.calibration.get("bayes_auc")
    truth: Dict[str, Any] = {
        # `base` is the answer key AND the matched-K anchor, so it must be the pre-registered primary set.
        "base": blanket,
        "relevant": blanket,
        "noise": noise,
        "interaction_operands": [],
        "quadratic_operands": [],
        "expected_to_break": tuple(scenario.expected_to_break),
        "metric": "recall",
        "notes": f"{scenario.purpose} (calibrated Bayes AUC {ceiling:.3f})" if ceiling is not None else scenario.purpose,
        "bayes_auc": ceiling,
        "spec_hash": spec.content_hash(),
        # Recorded so a leg cannot claim a bed it did not run: the lock hashes the SPEC, and an adapter
        # that resized the bed afterwards would leave the hash intact while changing what was measured.
        "n_samples": int(spec.n_samples),
    }
    return dataset.frame, np.asarray(dataset.target), truth


def scm_bed_scenarios(include_null: bool = False, n_samples: Optional[int] = None) -> List[Tuple[str, Callable[[int], Any]]]:
    """Return the SCM beds as `(name, generator)` pairs for `run_grid`.

    The null beds are excluded by default for the same reason the real leg excludes them from its kill
    criterion: with no relevant column, "did any arm beat all-features" has no meaning. They remain
    available because the false-discovery curve is measured on exactly those beds.
    """
    from mlframe.data.datasets import scenarios

    names = [name for name in scenarios.names() if include_null or not name.startswith("null_")]
    return [(name, (lambda bed: (lambda seed: build_scm_bed(bed, seed, n_samples)))(name)) for name in names]
