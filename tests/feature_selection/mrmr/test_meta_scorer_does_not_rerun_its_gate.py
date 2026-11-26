"""The meta scorer does not run the cmim scorer again when cmim's own gate already ran in the same lane with the same arguments: that run is the same
computation, and its columns would be dropped as duplicates of the gate's. The fit selects and engineers the same columns either way."""

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._mrmr_fit_impl._hybrid_orth_family_variants import _group4 as G4
from mlframe.feature_selection.filters._orthogonal_meta_scorer_fe import hybrid_orth_mi_meta_fe_with_recipes
from tests.feature_selection.conftest import make_fast_mrmr


def _quadratic(seed, n=2000):
    """``y`` from two squared columns among three noise columns."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x1": rng.standard_normal(n), "x2": rng.standard_normal(n), **{f"noise_{i}": rng.standard_normal(n) for i in range(3)}})
    s = X["x1"] ** 2 + 0.6 * X["x2"] ** 2
    return X, pd.Series((s + 0.05 * rng.standard_normal(n) > float(np.median(s))).astype(int), name="y")


def test_the_meta_scorer_returns_the_frame_unchanged_for_a_scorer_already_run():
    """A chosen scorer in ``already_run`` is not dispatched: X unchanged, empty scores, no recipes, the choice and fingerprint still reported."""
    X, y = _quadratic(0)
    X_aug, scores, recipes, chosen, fp = hybrid_orth_mi_meta_fe_with_recipes(X, y.to_numpy(), force_scorer="cmim", already_run={"cmim"})
    assert X_aug is X and scores.empty and recipes == [] and chosen == "cmim" and fp


@pytest.mark.parametrize("seed", [0, 1])
def test_a_fit_with_cmim_and_meta_selects_the_same_with_and_without_the_rerun(monkeypatch, seed):
    """cmim's gate and the meta scorer forced to cmim: the selected and the engineered columns equal those of a fit in which meta re-runs cmim."""
    X, y = _quadratic(seed)
    kw = dict(fe_hybrid_orth_cmim_enable=True, fe_hybrid_orth_meta_enable=True, fe_hybrid_orth_meta_force_scorer="cmim", fe_hybrid_orth_degrees=(2,), fe_hybrid_orth_basis="hermite", fe_hybrid_orth_top_k=2)
    calls = []
    real = G4._meta_already_run

    def spy(*a, **k):
        """Record what the fit would skip."""
        out = real(*a, **k)
        calls.append(set(out))
        return out

    monkeypatch.setattr(G4, "_meta_already_run", spy)
    skipped = make_fast_mrmr(**kw).fit(X, y)
    assert calls and calls[-1] == {"cmim"}, "the fit did not see cmim's gate run"
    monkeypatch.setattr(G4, "_meta_already_run", lambda *a, **k: set())
    rerun = make_fast_mrmr(**kw).fit(X, y)
    assert list(skipped.get_feature_names_out()) == list(rerun.get_feature_names_out())
    assert list(skipped.hybrid_orth_features_ or []) == list(rerun.hybrid_orth_features_ or [])
