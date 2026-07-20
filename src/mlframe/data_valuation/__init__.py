"""Per-row data valuation (gt_04): cooperative-game credit assignment along the ROW axis.

Training examples are players, ``v(S)`` is validation performance of a model trained on subset ``S``,
and a row's Shapley/Banzhaf value measures its marginal contribution -- negative for mislabeled or
harmful rows, near-zero for redundant ones, high for genuinely informative ones. Uses: label-noise
detection/cleaning, per-row ``sample_weight`` for training (down-weight harmful rows instead of
deleting them outright), data-acquisition prioritization.

Engines:
    :func:`knn_shapley` (RECOMMENDED DEFAULT) -- exact closed-form KNN-Shapley (Jia et al., VLDB 2019),
    ``O(n log n)`` per validation point, no retraining. Classification only in v1; continuous targets
    raise ``NotImplementedError`` pointing at :func:`tmc_shapley`.
    :func:`tmc_shapley` -- Truncated Monte Carlo Shapley (Ghorbani & Zou, ICML 2019), model-agnostic
    but each marginal contribution is a retrain: cost is ``O(n_permutations * n_rows)`` retrains in the
    worst case (fewer once truncation fires). Only practical for small/medium ``n_rows`` or a
    stratified subsample (see :func:`propagate_subsample_values`).
    :func:`data_banzhaf` -- MSR-Banzhaf semivalue (Wang & Jia, AISTATS 2023) over the same
    caller-supplied utility, sharing TMC's cost model per coalition.

:func:`valuation_sample_weight` turns any of the above into a ``(n,)`` non-negative training weight.

gt_06 extends this package with a non-cooperative-game construction: :func:`dro_reweight_fit`
(distributionally robust optimization as a two-player zero-sum game -- the model minimizes weighted
loss, an adversary reweights within a chi-square uncertainty ball to maximize it) and
:func:`adversarial_validation` (train-vs-test shift diagnostic with a discriminator-game reading). This
is NOT a GAN, NOT multi-agent RL, NOT mechanism design -- see ``_adversarial_reweighting.py``'s module
docstring for the explicit scope statement.

Related work NOT implemented here (out of scope, gradient-based/model-specific): influence functions,
TracIn.  # codespell:ignore tracin

Valuation-driven ``sample_weight`` is NOT wired into any mlframe training config by default and never
will be without explicit opt-in evidence -- weighting training data changes model behaviour globally;
this package ships the facade + engines + weight transform, ready for a caller to wire in manually via
the existing ``_setup_sample_weight`` choke point in ``training/_data_helpers.py``.
"""

from __future__ import annotations

from mlframe.data_valuation._adversarial_reweighting import dro_reweight_fit, project_chi2_ball
from mlframe.data_valuation._adversarial_validation import adversarial_validation
from mlframe.data_valuation._knn_shapley import knn_shapley
from mlframe.data_valuation._mc_sampling import data_banzhaf, propagate_subsample_values, tmc_shapley
from mlframe.data_valuation._training_weight_adapter import training_sample_weight_from_valuation
from mlframe.data_valuation._weights import valuation_sample_weight

__all__ = [
    "knn_shapley",
    "tmc_shapley",
    "data_banzhaf",
    "propagate_subsample_values",
    "valuation_sample_weight",
    "training_sample_weight_from_valuation",
    "dro_reweight_fit",
    "project_chi2_ball",
    "adversarial_validation",
]
