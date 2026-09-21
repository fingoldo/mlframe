"""The honest RMSE gate must reuse honest-OOF's predictions when both measured the same rows, and only then.

With group ids, honest-OOF fits a tiny model per spec on the screen rows and predicts the honest holdout; the gate then
fitted the same models on the same rows again whenever both samples were under their caps, producing numbers identical
to full precision. It now takes the cached prediction when the fit rows, eval rows and spec mask all match, and refits
otherwise, so its verdicts cannot change.
"""

from __future__ import annotations

import numpy as np

from mlframe.training.composite import CompositeTargetDiscovery
from mlframe.training.composite.discovery._honest_oof_select import cached_honest_prediction, prediction_key
from mlframe.training.configs import CompositeTargetDiscoveryConfig

from tests.training.composite.discovery.test_biz_val_discovery_honest_oof_select import _frame, _split_upper_tail

_FEATS = ["base_full", "base_partial", "x1"]


def _grouped_fit(monkeypatch=None, **cfg):
    """A grouped discovery where honest-OOF runs, optionally counting sklearn LightGBM fits."""
    counter = {"n": 0}
    if monkeypatch is not None:
        import lightgbm as lgb

        real_fit = lgb.LGBMRegressor.fit

        def counting_fit(self, *args, **kwargs):
            """Count every sklearn LightGBM fit."""
            counter["n"] += 1
            return real_fit(self, *args, **kwargs)

        monkeypatch.setattr(lgb.LGBMRegressor, "fit", counting_fit)
    df, groups, _y, levels = _frame(n_groups=20, per=120)
    train_idx, _holdout = _split_upper_tail(groups, levels, 4)
    disc = CompositeTargetDiscovery(CompositeTargetDiscoveryConfig(
        enabled=True, random_state=0, base_candidates=["base_full", "base_partial"], tiny_model_n_estimators=20, **cfg,
    ))
    disc._group_ids_for_rerank = groups
    disc.fit(df, "y", _FEATS, train_idx)
    return disc, counter["n"]


def test_the_gate_verdicts_match_a_run_that_refits_everything(monkeypatch):
    """Reuse is a speed change only: specs and their honest numbers equal those of a gate forced to refit."""
    reused, n_reused = _grouped_fit(monkeypatch)

    from mlframe.training.composite.discovery import _honest_rmse_gate

    monkeypatch.setattr(_honest_rmse_gate, "cached_honest_prediction", lambda *a, **k: None)
    refit, n_refit = _grouped_fit(monkeypatch)
    summary = [(s.name, s.honest_holdout_rmse, s.honest_holdout_raw_rmse) for s in reused.specs_]
    assert summary == [(s.name, s.honest_holdout_rmse, s.honest_holdout_raw_rmse) for s in refit.specs_]
    assert n_reused < n_refit, f"reuse saved no fits ({n_reused} vs {n_refit})"


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
