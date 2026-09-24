"""The honest RMSE gate must reuse honest-OOF's predictions when both measured the same rows, and only then.

With group ids, honest-OOF fits a tiny model per spec on the screen rows and predicts the honest holdout; the gate then
fitted the same models on the same rows again whenever both samples were under their caps, producing numbers identical
to full precision. It now takes the cached prediction when the fit rows, eval rows and spec mask all match, and refits
otherwise, so its verdicts cannot change.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite import CompositeTargetDiscovery
from mlframe.training.composite.discovery._honest_oof_select import cached_honest_prediction, prediction_key
from mlframe.training.configs import CompositeTargetDiscoveryConfig

from tests.training.composite.discovery.test_biz_val_discovery_honest_oof_select import _frame, _split_upper_tail

_FEATS = ["base_full", "base_partial", "x1"]


def _grouped_fit(monkeypatch=None, **cfg):
    """A grouped discovery where honest-OOF runs, optionally counting LightGBM trainings.

    ``lightgbm.train`` is counted: the sklearn wrapper's fit goes through it, and so do the gates' shared-fold fits, which
    call it directly.
    """
    counter = {"n": 0}
    if monkeypatch is not None:
        import lightgbm as lgb

        real_train = lgb.train

        def counting_train(*args, **kwargs):
            """Count every LightGBM training."""
            counter["n"] += 1
            return real_train(*args, **kwargs)

        monkeypatch.setattr(lgb, "train", counting_train)
        import lightgbm.sklearn as lgb_sklearn

        monkeypatch.setattr(lgb_sklearn, "train", counting_train)
    df, groups, _y, levels = _frame(n_groups=20, per=120)
    train_idx, _holdout = _split_upper_tail(groups, levels, 4)
    disc = CompositeTargetDiscovery(CompositeTargetDiscoveryConfig(
        enabled=True, random_state=0, base_candidates=["base_full", "base_partial"], tiny_model_n_estimators=20, **cfg,
    ))
    disc._group_ids_for_rerank = groups
    disc.fit(df, "y", _FEATS, train_idx)
    return disc, counter["n"]


def test_the_gate_verdicts_match_a_run_that_refits_everything(monkeypatch):
    """Reuse is a speed change only: specs and their honest numbers equal those of a gate forced to refit, the selection
    pass is served honest-OOF's predictions, and reuse never costs a fit.

    The total fit count need not drop: the record-only report pass reads other rows, so it fits the models the selection
    pass took from honest-OOF, where a forced refit fits them in the selection pass and the report pass reuses them.
    """
    from mlframe.training.composite.discovery import _honest_rmse_gate

    served = {"n": 0}
    real_cached = _honest_rmse_gate.cached_honest_prediction

    def counting_cached(*a, **k):
        """Count the predictions the gate actually borrows from honest-OOF."""
        out = real_cached(*a, **k)
        served["n"] += out is not None
        return out

    monkeypatch.setattr(_honest_rmse_gate, "cached_honest_prediction", counting_cached)
    reused, n_reused = _grouped_fit(monkeypatch)
    assert served["n"] > 0, "the selection pass must be served honest-OOF's predictions"

    monkeypatch.setattr(_honest_rmse_gate, "cached_honest_prediction", lambda *a, **k: None)
    refit, n_refit = _grouped_fit(monkeypatch)
    summary = [(s.name, s.honest_holdout_rmse, s.honest_holdout_raw_rmse) for s in reused.specs_]
    assert summary == [(s.name, s.honest_holdout_rmse, s.honest_holdout_raw_rmse) for s in refit.specs_]
    assert n_reused <= n_refit, f"reuse cost fits ({n_reused} vs {n_refit})"


def test_a_prediction_is_withheld_when_the_rows_differ():
    """A gate that drew a different sample must refit, never borrow a prediction made on other rows."""
    disc, _ = _grouped_fit()
    cache = disc._honest_oof_predictions
    assert cache and cache["specs"], "honest-OOF must have cached predictions on this grouped fixture"
    fit_idx, eval_idx = cache["fit_idx"], cache["eval_idx"]
    assert cached_honest_prediction(disc, fit_idx, eval_idx) is not None
    assert cached_honest_prediction(disc, fit_idx[:-1], eval_idx) is None
    assert cached_honest_prediction(disc, fit_idx, eval_idx[1:]) is None


def test_a_prediction_is_withheld_when_the_fit_mask_differs():
    """The gate refines the domain with the fitted-domain check; a different mask means a different model."""
    disc, _ = _grouped_fit()
    cache = disc._honest_oof_predictions
    fit_idx, eval_idx = cache["fit_idx"], cache["eval_idx"]
    full_mask = np.ones(fit_idx.size, dtype=bool)
    unmasked = [name for name, (key, _pred) in cache["specs"].items() if key == prediction_key(full_mask)]
    assert unmasked, "at least one spec on this fixture trains on every fit row"
    name = unmasked[0]
    assert cached_honest_prediction(disc, fit_idx, eval_idx, name, full_mask) is not None
    other_mask = full_mask.copy()
    other_mask[0] = False
    assert cached_honest_prediction(disc, fit_idx, eval_idx, name, other_mask) is None
    assert cached_honest_prediction(disc, fit_idx, eval_idx, "no-such-spec", full_mask) is None


@pytest.mark.parametrize("gate_cap, expect_reuse", [(None, True), (20_000, False)])
def test_above_the_old_cap_the_gate_still_reuses_honest_oof(monkeypatch, gate_cap, expect_reuse):
    """With the gate's default cap equal to honest-OOF's, a frame whose screen exceeds 20k rows still shares its predictions;
    the old 20k gate cap drew a different sample there and refit every spec."""
    from mlframe.training.composite import CompositeTargetDiscovery
    from mlframe.training.composite.discovery import _honest_rmse_gate
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    served = {"n": 0}
    real_cached = _honest_rmse_gate.cached_honest_prediction

    def counting_cached(*a, **k):
        """Count the predictions the gate actually borrows from honest-OOF."""
        out = real_cached(*a, **k)
        served["n"] += out is not None
        return out

    monkeypatch.setattr(_honest_rmse_gate, "cached_honest_prediction", counting_cached)
    df, groups, _y, levels = _frame(n_groups=40, per=1500)  # 60k rows: the screen sample exceeds both caps
    train_idx, _holdout = _split_upper_tail(groups, levels, 4)
    kw = {} if gate_cap is None else {"honest_rmse_gate_sample_n": gate_cap}
    disc = CompositeTargetDiscovery(CompositeTargetDiscoveryConfig(
        enabled=True, random_state=0, base_candidates=["base_full", "base_partial"], tiny_model_n_estimators=20, mi_sample_n=45_000, **kw,
    ))
    disc._group_ids_for_rerank = groups
    disc.fit(df, "y", _FEATS, train_idx)
    assert disc.specs_, "the fixture must leave specs for the gate to score"
    assert (served["n"] > 0) == expect_reuse, served
