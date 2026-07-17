"""Regression: the disk-loaded (suite) and in-memory (from-models) predict paths must replay
ensembles identically.

Two pre-fix divergences:
  (a) the suite path called ``_resolve_quantile_alphas`` WITHOUT the model-object arg, so disk-loaded
      quantile bundles could not recover alphas via model introspection -> fix_quantile_crossing was
      skipped -> crossed quantiles leaked through.
  (b) the from-models prob-combine omitted ``rrf_k``, replaying RRF with a hardcoded k=60 even when the
      train side stamped a different k.
"""

from __future__ import annotations

import numpy as np

from mlframe.training.core.predict import (
    _combine_probs,
    _resolve_chosen_ensemble_params,
    _resolve_quantile_alphas,
)


class _FakeQuantileModel:
    """Mimics a CB MultiQuantile / XGB quantile member carrying its alpha list."""

    def __init__(self, alphas):
        self.quantile_alpha = list(alphas)


def test_quantile_alphas_recovered_only_when_model_arg_passed():
    """The suite path omitted the model arg; metadata-less alphas were then unrecoverable."""
    metadata = {}  # no metadata-stored alphas: forces model introspection
    tt, tn = "quantile_regression", "y"
    model = _FakeQuantileModel([0.1, 0.5, 0.9])

    # Pre-fix suite call shape (no model) -> None -> fix_quantile_crossing skipped.
    assert _resolve_quantile_alphas(metadata, tt, tn) is None
    # Post-fix suite call shape (model threaded, matching from-models) -> alphas recovered.
    assert _resolve_quantile_alphas(metadata, tt, tn, model) == [0.1, 0.5, 0.9]


def test_rrf_k_threaded_and_actually_changes_combine_output():
    metadata = {"ensembles_chosen_params": {"binary": {"y": {"rrf_k": 7}}}}
    params = _resolve_chosen_ensemble_params(metadata, "binary", "y")
    assert int(params.get("rrf_k", 60)) == 7

    probs = [
        np.array([[0.2, 0.8], [0.6, 0.4], [0.1, 0.9]]),
        np.array([[0.7, 0.3], [0.3, 0.7], [0.4, 0.6]]),
    ]
    out_default = _combine_probs(probs, "rrf", rrf_k=60)
    out_stamped = _combine_probs(probs, "rrf", rrf_k=7)
    # If rrf_k is silently dropped (hardcoded 60) the stamped replay would mismatch the train side.
    assert not np.allclose(out_default, out_stamped), "rrf_k must affect RRF combine output"
