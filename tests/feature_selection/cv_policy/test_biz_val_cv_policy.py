"""Business value of the shared split policy: on regime-drifting data a selector that follows it stops selecting a feature that only memorises time."""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from mlframe.feature_selection.cv_policy import CVPolicy, wire_selector_policy
from mlframe.feature_selection.functional_adapters import ForwardSelectSelector


def _regime_frame(seed: int, n: int = 600, block: int = 20):
    """Rows in time order; the label is one i.i.d. coin per time block (so time index memorises it inside a shuffled fold, never across time) plus a weak stable signal."""
    rng = np.random.default_rng(seed)
    regime = np.repeat(rng.integers(0, 2, n // block), block)
    signal = rng.normal(size=n)
    y = ((0.5 * (2 * regime - 1) + 0.3 * signal + 0.3 * rng.normal(size=n)) > 0).astype(int)
    X = pd.DataFrame({"time_index": np.arange(n, dtype=float), "signal": signal, "noise": rng.normal(size=n)})
    perm = rng.permutation(n)
    return X.iloc[perm].reset_index(drop=True), y[perm], np.arange(n)[perm]


def _select(seed: int, follow_policy: bool) -> set:
    X, y, ts = _regime_frame(seed)
    sel = ForwardSelectSelector(DecisionTreeClassifier(max_depth=6, random_state=0), cv=4, max_features=1, scoring="roc_auc", random_state=seed)
    if follow_policy:
        wire_selector_policy(sel, CVPolicy("temporal", "biz_val", ts), classification=True)
    return set(sel.fit(X, y).selected_features_)


def test_biz_val_cv_policy_forward_select_temporal_avoids_time_memorising_feature():
    seeds = range(4)
    shuffled = [_select(s, follow_policy=False) for s in seeds]
    temporal = [_select(s, follow_policy=True) for s in seeds]
    shuffled_time_hits = sum("time_index" in s for s in shuffled)
    temporal_time_hits = sum("time_index" in s for s in temporal)
    assert shuffled_time_hits >= 3, f"control: shuffled CV should keep picking the time-memorising column, got {shuffled}"
    assert temporal_time_hits <= 1, f"temporal policy still selected the time-memorising column: {temporal}"
    assert sum("signal" in s for s in temporal) >= 3, f"temporal policy should keep the stable signal: {temporal}"
