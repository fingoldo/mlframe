"""Artifact export and repr contracts.

* An out-of-range SU (cached MI inconsistent with the marginal entropies) is clamped but no longer silent.
* The artifact dict records which target its SU / MI vectors were computed against, and a multi-target fit says it used only the first.
* The get_feature_names_out memo is not pickled.
* A truncated repr does not get an exact ``n_workers=`` spliced after its elided parameter list.
"""

from __future__ import annotations

import logging
import pickle  # nosec B403 - round-trip of an estimator this test just fitted

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._mrmr_artifacts import compute_mrmr_artifacts
from mlframe.feature_selection.filters.mrmr import MRMR


def _binned(n=400, seed=0):
    """Two binned features and a binary target column at index 2."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    data = np.column_stack([y ^ (rng.random(n) < 0.1), rng.integers(0, 3, size=n), y]).astype(np.int32)
    return data, np.array([2, 3, 2], dtype=np.int32)


def _artifacts(cached, target_indices=(2,)):
    """Artifacts for the fixture with the given MI cache."""
    data, nbins = _binned()
    return compute_mrmr_artifacts(
        data=data, cols=["a", "b", "y"], nbins=nbins, target_indices=np.array(target_indices), cached_MIs=cached,
        feature_names_in=["a", "b"], support_original=np.array([0]), retain_bins=False,
    )


def test_su_clamp_warns_on_out_of_range(caplog):
    """A cached MI far above (H(X)+H(y))/2 clamps SU to 1.0 and names the column in a warning."""
    with caplog.at_level(logging.WARNING):
        art = _artifacts({(0,): 5.0, (1,): 0.001})
    assert art["su_to_target"][0] == 1.0
    assert any("'a'" in r.getMessage() and "outside [0, 1]" in r.getMessage() for r in caplog.records)
    assert not any("'b'" in r.getMessage() for r in caplog.records), "an in-range SU must not warn"


def test_artifacts_record_the_target_used(caplog):
    """The export names the target column; a two-target call warns that only the first was used."""
    assert _artifacts({(0,): 0.1})["target_index_used"] == 2
    with caplog.at_level(logging.WARNING):
        art = _artifacts({(0,): 0.1}, target_indices=(2, 1))
    assert art["target_index_used"] == 2
    assert any("first target only" in r.getMessage() for r in caplog.records)


def test_names_memo_is_not_pickled_and_names_survive():
    """get_feature_names_out's memo stays out of the pickle; the names are unchanged after a round-trip."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(300, 3)), columns=["a", "b", "c"])
    y = (X["a"] > 0).astype(np.int32).to_numpy()
    m = MRMR(full_npermutations=3, baseline_npermutations=2, n_jobs=1, verbose=0, fe_max_steps=0, random_seed=0).fit(X, y)
    names = list(m.get_feature_names_out())
    assert "_engineered_names_cache_" not in m.__getstate__()
    assert list(pickle.loads(pickle.dumps(m)).get_feature_names_out()) == names  # nosec B301


def test_repr_annotation_skipped_when_truncated():
    """Under a small N_CHAR_MAX the elided repr carries no n_workers= claim; the full repr still does."""
    m = MRMR(verbose=1, fe_max_steps=3, full_npermutations=7)
    short = m.__repr__(N_CHAR_MAX=40)
    assert "..." in short, "fixture precondition: the repr must be truncated"
    assert "n_workers=" not in short
    assert "n_workers=" in m.__repr__(N_CHAR_MAX=100000)
