"""Regression: the AR(1) lag-failsafe val cross-check reproduces + fixes the prod "ship lag over the better model" bug.

Prod (TV_training.log): lag_predict and the trained LGBM tied on group-K-fold OOF (~13.64 each), so the failsafe shipped
zero-param lag_predict (test RMSE 12.29) even though the trained model scored 9.31 on BOTH the group-disjoint val and test
splits -- a 32% worse deployment. ``decide_ar1_failsafe_val_veto`` catches this: on the val split (same honest regime as
test) the trained component beats lag by far more than the failsafe tolerance, so the OOF tie is confirmed spurious and
the veto returns the trained index to deploy instead of lag.
"""

from __future__ import annotations


import numpy as np

from mlframe.training.core._ar1_failsafe_veto import compute_val_veto, decide_ar1_failsafe_val_veto


class _Comp:
    """Minimal component whose predict returns a fixed y-scale vector (indexed by the passed positional frame)."""

    def __init__(self, preds):
        self._preds = np.asarray(preds, dtype=np.float64)

    def predict(self, _frame):
        """Predict."""
        return self._preds


class _Cfg:
    """Groups tests covering cfg."""
    def __init__(self, on=True):
        self.ar1_failsafe_val_crosscheck = on


class TestComputeValVeto:
    """Groups tests covering compute val veto."""
    def test_prod_scenario_wrapper_vetoes(self):
        # OOF tie (both ~13.64); on val the trained component is near-perfect, lag is far off -> veto -> trained idx 0.
        """Prod scenario wrapper vetoes."""
        n = 400
        y = np.linspace(0.0, 100.0, n)
        trained = _Comp(y + np.resize([0.3, -0.3], n))  # val RMSE ~0.3
        lag = _Comp(y + 13.0)  # val RMSE ~13
        idx = compute_val_veto(
            ["raw#lgb", "lag_predict"],
            [13.64, 13.64],
            [trained, lag],
            filtered_val_df=object(),
            filtered_val_idx=np.arange(n),
            oof_y_full=y,
            lag_failsafe_tol=0.10,
            config=_Cfg(on=True),
        )
        assert idx == 0

    def test_wrapper_disabled_returns_none(self):
        """Wrapper disabled returns none."""
        n = 400
        y = np.linspace(0.0, 100.0, n)
        trained, lag = _Comp(y), _Comp(y + 13.0)
        idx = compute_val_veto(
            ["raw#lgb", "lag_predict"],
            [13.64, 13.64],
            [trained, lag],
            filtered_val_df=object(),
            filtered_val_idx=np.arange(n),
            oof_y_full=y,
            lag_failsafe_tol=0.10,
            config=_Cfg(on=False),
        )
        assert idx is None

    def test_wrapper_no_val_data_returns_none(self):
        """Wrapper no val data returns none."""
        assert (
            compute_val_veto(
                ["raw#lgb", "lag_predict"],
                [13.64, 13.64],
                [_Comp([0.0]), _Comp([0.0])],
                filtered_val_df=None,
                filtered_val_idx=None,
                oof_y_full=None,
                lag_failsafe_tol=0.10,
                config=_Cfg(on=True),
            )
            is None
        )


def _val_map(d):
    """Val map."""
    return lambda i: d.get(i, float("nan"))


class TestAr1FailsafeValVeto:
    """Groups tests covering ar1 failsafe val veto."""
    def test_prod_scenario_oof_tie_val_prefers_trained(self):
        # names[0]=trained, names[1]=lag. OOF tie (both 13.64). Val: trained 9.31 << lag 13.37 -> veto -> deploy trained.
        """Prod scenario oof tie val prefers trained."""
        names = ["raw#lgb", "lag_predict"]
        oof = [13.64, 13.64]
        veto = decide_ar1_failsafe_val_veto(names, oof, 0.10, _val_map({0: 9.31, 1: 13.37}))
        assert veto == 0, "trained must be deployed when it beats lag on val despite the OOF tie"

    def test_no_veto_when_val_also_ties(self):
        # If val ALSO shows lag ~ trained, the failsafe is legitimate -> no veto (deploy lag downstream).
        """No veto when val also ties."""
        names = ["raw#lgb", "lag_predict"]
        oof = [13.64, 13.64]
        assert decide_ar1_failsafe_val_veto(names, oof, 0.10, _val_map({0: 13.2, 1: 13.37})) is None

    def test_no_veto_when_failsafe_would_not_fire(self):
        # lag OOF far worse than best trained -> failsafe never fires -> nothing to veto.
        """No veto when failsafe would not fire."""
        names = ["raw#lgb", "lag_predict"]
        oof = [9.0, 13.64]
        assert decide_ar1_failsafe_val_veto(names, oof, 0.10, _val_map({0: 9.31, 1: 13.37})) is None

    def test_veto_requires_margin_beyond_tolerance(self):
        # trained beats lag on val but only slightly (< tolerance) -> not a confident veto.
        """Veto requires margin beyond tolerance."""
        names = ["raw#lgb", "lag_predict"]
        oof = [13.64, 13.64]
        # 13.0 vs 13.37: ratio 0.972 > 1/1.10=0.909 -> within tolerance -> no veto.
        assert decide_ar1_failsafe_val_veto(names, oof, 0.10, _val_map({0: 13.0, 1: 13.37})) is None

    def test_no_trained_component(self):
        """No trained component."""
        assert decide_ar1_failsafe_val_veto(["lag_predict"], [13.64], 0.10, _val_map({0: 13.6})) is None

    def test_no_lag_in_pool(self):
        """No lag in pool."""
        assert decide_ar1_failsafe_val_veto(["raw#a", "raw#b"], [9.0, 9.1], 0.10, _val_map({0: 9.0, 1: 9.1})) is None

    def test_tol_zero_disables(self):
        """Tol zero disables."""
        assert decide_ar1_failsafe_val_veto(["raw#lgb", "lag_predict"], [13.64, 13.64], 0.0, _val_map({0: 9.31, 1: 13.37})) is None

    def test_nan_val_no_veto(self):
        """Nan val no veto."""
        names = ["raw#lgb", "lag_predict"]
        assert decide_ar1_failsafe_val_veto(names, [13.64, 13.64], 0.10, _val_map({0: float("nan"), 1: 13.37})) is None

    def test_picks_best_trained_by_oof_among_several(self):
        # two trained + lag; the best-by-OOF trained (idx1, oof 13.5) is the candidate; it beats lag on val -> veto idx1.
        """Picks best trained by oof among several."""
        names = ["raw#a", "raw#b", "lag_predict"]
        oof = [13.9, 13.5, 13.6]
        veto = decide_ar1_failsafe_val_veto(names, oof, 0.10, _val_map({1: 9.4, 2: 13.4}))
        assert veto == 1

    def test_all_nan_oof_no_veto(self):
        """All nan oof no veto."""
        assert decide_ar1_failsafe_val_veto(["raw#lgb", "lag_predict"], [float("nan"), float("nan")], 0.10, _val_map({0: 9.0, 1: 13.0})) is None
