"""Unit tests for estimator-type-aware RFECV importance aggregation (importance_agg='dispatched').

Covers each family path: tree (variance down-weight), linear (sign-harmony), kernel (legacy defer),
plus family detection and the dispatcher's fallback contract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR

from mlframe.feature_selection.wrappers._enums import VotesAggregation
from mlframe.feature_selection.wrappers._helpers_importance_agg import (
    aggregate_importances_dispatched,
    aggregate_linear,
    aggregate_tree,
    detect_estimator_family,
    get_signed_linear_coef,
)


def test_detect_family_linear():
    """Detect family linear."""
    assert detect_estimator_family(LogisticRegression()) == "linear"
    assert detect_estimator_family(Ridge()) == "linear"


def test_detect_family_tree():
    """Detect family tree."""
    assert detect_estimator_family(RandomForestClassifier()) == "tree"


def test_detect_family_kernel():
    """Detect family kernel."""
    assert detect_estimator_family(SVC()) == "kernel"
    assert detect_estimator_family(SVR()) == "kernel"


def test_aggregate_tree_downweights_high_variance():
    # A and B have identical mean (1.0); B has high fold-to-fold variance -> ranked lower.
    """Aggregate tree downweights high variance."""
    fi = {
        "r0": {"A": 1.0, "B": 2.0},
        "r1": {"A": 1.0, "B": 0.0},
        "r2": {"A": 1.0, "B": 0.0},
        "r3": {"A": 1.0, "B": 2.0},
    }
    scores = aggregate_tree(fi, k_cv=1.0)
    assert scores["A"] > scores["B"], "steady feature must beat high-variance same-mean feature"
    assert scores["A"] == pytest.approx(1.0)
    assert scores["B"] < 0.6


def test_aggregate_tree_single_run_is_raw_mean():
    """Aggregate tree single run is raw mean."""
    fi = {"r0": {"A": 0.7, "B": 0.3}}
    scores = aggregate_tree(fi, k_cv=1.0)
    assert scores["A"] == pytest.approx(0.7)
    assert scores["B"] == pytest.approx(0.3)


def test_aggregate_linear_sign_harmony_demotes_flipper():
    # A consistently positive; B flips sign (3 pos, 2 neg) -> heavily demoted.
    """Aggregate linear sign harmony demotes flipper."""
    sg = {
        "r0": {"A": 1.0, "B": 1.0},
        "r1": {"A": 1.0, "B": 1.0},
        "r2": {"A": 1.0, "B": 1.0},
        "r3": {"A": 1.0, "B": -1.0},
        "r4": {"A": 1.0, "B": -1.0},
    }
    scores = aggregate_linear(sg)
    assert scores["A"] == pytest.approx(1.0)
    # |mean(1,1,1,-1,-1)| * max(0.6,0.4) = 0.2 * 0.6 = 0.12
    assert scores["B"] == pytest.approx(0.12, abs=1e-9)
    assert scores["A"] > scores["B"] * 5


def test_aggregate_linear_consistent_sign_keeps_magnitude():
    """Aggregate linear consistent sign keeps magnitude."""
    sg = {"r0": {"A": -2.0}, "r1": {"A": -2.0}, "r2": {"A": -2.0}}
    scores = aggregate_linear(sg)
    assert scores["A"] == pytest.approx(2.0)  # all negative -> agreement 1.0, magnitude 2.0


def test_aggregate_linear_vectorized_matches_per_row_reference():
    # Pins the vectorised numpy aggregation bit-identical to the original per-row loop, including
    # the NaN-finite, all-zero (agreement 1.0), and all-non-finite (score 0.0) edge rows.
    """Aggregate linear vectorized matches per row reference."""
    import math
    import numpy as np
    import pandas as pd

    def _reference(signed_importances, eps=1e-12):
        """Helper that reference."""
        table = pd.DataFrame(signed_importances)
        if table.empty:
            return {}
        out = {}
        for feat in table.index:
            row = table.loc[feat].to_numpy(dtype=float)
            row = row[np.isfinite(row)]
            if row.size == 0:
                out[feat] = 0.0
                continue
            mean_signed = float(np.mean(row))
            nz = row[np.abs(row) > eps]
            agreement = 1.0 if nz.size == 0 else max(float(np.mean(nz > 0)), 1.0 - float(np.mean(nz > 0)))
            out[feat] = abs(mean_signed) * agreement
        return out

    rng = np.random.default_rng(0)
    for F, R, nanfrac, zerofrac in [(40, 5, 0.0, 0.0), (25, 3, 0.3, 0.2), (8, 4, 0.5, 0.4), (1, 1, 0.0, 0.0), (12, 2, 1.0, 0.0)]:
        sg = {}
        for r in range(R):
            vals = rng.standard_normal(F)
            vals[rng.random(F) < nanfrac] = np.nan
            vals[rng.random(F) < zerofrac] = 0.0
            sg[f"r{r}"] = {f"f{i}": float(vals[i]) for i in range(F)}
        ref = _reference(sg)
        got = aggregate_linear(sg)
        assert set(ref) == set(got)
        for k in ref:
            assert ref[k] == got[k] or (math.isnan(ref[k]) and math.isnan(got[k])), (k, ref[k], got[k])
    assert aggregate_linear({}) == {}


def test_dispatcher_tree_path_ranks_by_downweighted_mean():
    """Dispatcher tree path ranks by downweighted mean."""
    fi = {
        "r0": {"A": 1.0, "B": 2.0},
        "r1": {"A": 1.0, "B": 0.0},
        "r2": {"A": 1.0, "B": 0.0},
        "r3": {"A": 1.0, "B": 2.0},
    }
    ranks = aggregate_importances_dispatched(fi, family="tree", votes_aggregation_method=VotesAggregation.Borda)
    assert ranks[0] == "A"


def test_dispatcher_linear_uses_signed_when_present():
    """Dispatcher linear uses signed when present."""
    fi = {"r0": {"A": 1.0, "B": 1.0}}  # abs'd values (B looks equal to A)
    sg = {
        "r0": {"A": 1.0, "B": 1.0},
        "r1": {"A": 1.0, "B": -1.0},
        "r2": {"A": 1.0, "B": -1.0},
    }
    ranks = aggregate_importances_dispatched(
        fi,
        family="linear",
        votes_aggregation_method=VotesAggregation.Borda,
        signed_importances=sg,
    )
    assert ranks[0] == "A"


def test_dispatcher_linear_falls_back_to_legacy_without_signed():
    # No signed_importances -> cannot do sign-harmony -> must defer to legacy vote (not crash).
    """Dispatcher linear falls back to legacy without signed."""
    fi = {"r0": {"A": 0.9, "B": 0.1}, "r1": {"A": 0.8, "B": 0.2}}
    ranks = aggregate_importances_dispatched(
        fi,
        family="linear",
        votes_aggregation_method=VotesAggregation.Borda,
        signed_importances=None,
    )
    assert set(ranks) == {"A", "B"}
    assert ranks[0] == "A"


def test_dispatcher_kernel_defers_to_legacy():
    """Dispatcher kernel defers to legacy."""
    fi = {"r0": {"A": 0.9, "B": 0.1}, "r1": {"A": 0.8, "B": 0.2}}
    ranks = aggregate_importances_dispatched(fi, family="kernel", votes_aggregation_method=VotesAggregation.Borda)
    assert ranks[0] == "A"


def test_get_signed_linear_coef_preserves_sign():
    """Get signed linear coef preserves sign."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    y = (X[:, 0] * 2 - X[:, 1] * 2 > 0).astype(int)
    m = LogisticRegression(max_iter=500).fit(X, y)
    signed = get_signed_linear_coef(m, current_features=["f0", "f1", "f2"], train_data=X, coef_scale_source="none")
    assert signed is not None
    assert signed["f0"] > 0 and signed["f1"] < 0  # opposite-sign signal recovered


def test_get_signed_linear_coef_none_for_tree():
    """Get signed linear coef none for tree."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    y = (X[:, 0] > 0).astype(int)
    m = RandomForestClassifier(n_estimators=10).fit(X, y)
    assert get_signed_linear_coef(m, current_features=["a", "b", "c"], train_data=X) is None


def test_rfecv_constructor_validates_importance_agg():
    """Rfecv constructor validates importance agg."""
    from mlframe.feature_selection.wrappers.rfecv import RFECV

    with pytest.raises(ValueError):
        RFECV(estimator=LogisticRegression(), importance_agg="bogus")
    r = RFECV(estimator=LogisticRegression(), importance_agg="dispatched")
    assert r.importance_agg == "dispatched"


def test_rfecv_default_is_dispatched():
    """Rfecv default is dispatched."""
    from mlframe.feature_selection.wrappers.rfecv import RFECV

    r = RFECV(estimator=LogisticRegression())
    assert r.importance_agg == "dispatched", "default must be the flipped dispatched aggregation"


def test_rfecv_end_to_end_dispatched_tree_runs():
    """Rfecv end to end dispatched tree runs."""
    from mlframe.feature_selection.wrappers.rfecv import RFECV

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(400, 8)), columns=[f"c{i}" for i in range(8)])
    y = (X["c0"] * 1.5 + X["c1"] - X["c2"] + rng.normal(scale=0.5, size=400) > 0).astype(int)
    sel = RFECV(
        estimator=RandomForestClassifier(n_estimators=30, random_state=0),
        cv=3,
        max_refits=6,
        importance_agg="dispatched",
        early_stopping_val_nsplits=None,
        random_state=0,
    )
    sel.fit(X, y)
    assert sel._fi_family == "tree"
    assert sel.support_.sum() >= 1


def test_rfecv_end_to_end_dispatched_linear_collects_signed():
    """Rfecv end to end dispatched linear collects signed."""
    from mlframe.feature_selection.wrappers.rfecv import RFECV

    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(400, 8)), columns=[f"c{i}" for i in range(8)])
    y = (X["c0"] * 2 - X["c1"] * 2 + rng.normal(scale=0.7, size=400) > 0).astype(int)
    sel = RFECV(estimator=LogisticRegression(max_iter=500), cv=3, max_refits=6, importance_agg="dispatched", early_stopping_val_nsplits=None, random_state=1)
    sel.fit(X, y)
    assert sel._fi_family == "linear"
    # The fold loop must have stashed at least one signed-coef run.
    assert len(sel._signed_importances) >= 1
