"""A raw operand is not dropped as "subsumed" by a child that merely re-encodes it.

The raw-redundancy drop used to remove ``region`` when the one surviving child ``max(qubed(region), qubed(x))`` had a binned code that determined ``region`` (conditional MI exactly
0 against a floor of exactly 0), although the child carried no more information about y than ``region`` itself, and a linear model that receives only the child cannot use the
non-monotone relation of ``region`` to y. The drop now needs a consumer that carries more about y than the raw, by more than that consumer's own permutation floor.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _fe_raw_redundancy_drop as R


def _stand_in_state(raw_code, child_codes, y):
    """The few members of the redundancy state the informativeness probe reads."""
    seed = 0
    raw_marginal = lambda _name: R._excess_and_floor(raw_code, y, None, seed=seed, kx=int(raw_code.max()) + 1)
    return SimpleNamespace(_raw_marginal=raw_marginal, eng_bin=dict(enumerate(child_codes)), y_arr=y)


def test_a_recoding_of_the_raw_adds_no_information():
    """A child whose code is a relabelling of the raw's own code carries exactly the raw's information: the probe says no consumer adds information, so the raw is kept."""
    rng = np.random.default_rng(0)
    n = 6000
    a = rng.integers(0, 8, n)
    y = (rng.normal(size=n) + np.sin(a * 1.7) * 1.5 > 0).astype(np.int64)  # non-monotone in a
    relabel = rng.permutation(8)
    st = _stand_in_state(a.astype(np.int64), [relabel[a].astype(np.int64)], y)
    assert R._some_consumer_adds_information(st, [0], "a", seed=0) is False


def test_a_child_with_a_second_informative_source_adds_information():
    """A child that also holds a second operand that moves y carries more than the raw alone: the probe says yes, so the existing subsumption verdict decides as before."""
    rng = np.random.default_rng(1)
    n = 6000
    a = rng.integers(0, 6, n)
    b = rng.integers(0, 6, n)
    y = ((a - 2.5) + (b - 2.5) + rng.normal(size=n) * 0.5 > 0).astype(np.int64)
    joint = (a * 6 + b).astype(np.int64)
    st = _stand_in_state(a.astype(np.int64), [joint], y)
    assert R._some_consumer_adds_information(st, [0], "a", seed=0) is True


def test_a_failed_probe_leaves_the_decision_to_the_existing_verdict():
    """A state that cannot answer (a missing member) is not read as 'no information'; the probe returns True so the old verdict still decides."""
    assert R._some_consumer_adds_information(SimpleNamespace(), [0], "a", seed=0) is True


def test_region_survives_a_recoding_compound_on_the_seed_that_lost_it():
    """On the grouped-key fixture, seed 2 used to select two columns (region and x dropped, downstream AUC 0.808); the raw key now survives and the LogReg AUC is above 0.9."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(1)
    n = 4000
    region = rng.integers(0, 10, n)
    group_mean = rng.uniform(-3.0, 3.0, 10)
    x = group_mean[region] + rng.normal(0.0, 3.0, n)
    y = pd.Series((group_mean[region] + 0.1 * rng.normal(0.0, 1.0, n) > 0.0).astype(int), name="y")
    X = pd.DataFrame({"region": region.astype(np.int64), "x": x, "noise_0": rng.normal(size=n), "noise_1": rng.normal(size=n)})
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    m = MRMR(
        max_runtime_mins=1.0,
        random_seed=2,
        fe_grouped_agg_enable=True,
        fe_grouped_agg_group_cols=("region",),
        fe_grouped_agg_num_cols=("x",),
        fe_grouped_agg_top_k=5,
        fe_grouped_quantile_enable=True,
    ).fit(Xtr, ytr)
    f_tr = np.nan_to_num(np.asarray(m.transform(Xtr), dtype=np.float64))
    f_te = np.nan_to_num(np.asarray(m.transform(Xte), dtype=np.float64))
    auc = roc_auc_score(yte.to_numpy(), LogisticRegression(max_iter=2000).fit(f_tr, ytr.to_numpy()).predict_proba(f_te)[:, 1])
    assert auc > 0.9, f"the raw key was lost again: AUC {auc:.4f}, selected {[str(c) for c in m.get_feature_names_out()]}"
