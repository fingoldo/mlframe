"""Two properties the whole protocol rests on: no arm can see the holdout, and the panel repeats exactly.

Both are assumptions every number in the report inherits, and both fail silently when broken. A leak makes
every arm look better on the beds where it exists, in proportion to how greedily the arm selects. A
non-reproducible downstream fit puts noise into every paired difference, because the statistics above
difference an arm against the null hypothesis on the same holdout and assume that pairing is exact.

The leakage test is written the way the pre-registration asks for it: a planted column correlated with the
target ONLY on the holdout rows. An arm fitted on train cannot see that correlation, so recovering the
column is proof the split leaked -- and unlike a generic "does it overfit" check, this one has a specific,
named answer key.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from mlframe.feature_selection._benchmarks.fs_hybrid._arms import build_arm_roster
from mlframe.feature_selection._benchmarks.fs_hybrid._panel import PANEL_RANDOM_STATE, PANEL_THREADS, fit_and_score_panel, panel_factories
from mlframe.feature_selection._benchmarks.fs_hybrid.run_experiment import HOLDOUT_FRACTION

ROWS = 2000
PLANTED = "holdout_only_leak"

#: Arms cheap enough to run the whole planted-leak battery on. The expensive wrappers are covered by the
#: protocol-level argument -- they see the same training frame as everybody else -- and running them here
#: would turn a fast guard into one nobody runs.
CHEAP_ARMS = ("univariate-mi", "skb-f", "skb-mi", "select-fdr", "lars-order", "variance-sort")


def _bed_with_a_holdout_only_leak(seed: int = 0) -> Any:
    """Return `(x_train, y_train, x_test, y_test)` where one column is informative ONLY on the test rows.

    The bed carries GENUINE signal as well as the plant, and that is not decoration. A first version made
    every column noise, so the planted one topped the ranking about one seed in thirteen purely by being
    the luckiest noise column, and three arms "failed" a leak test on a bed with no leak. Real competitors
    give the assertion something to mean: the plant carries nothing on the training rows, so it must rank
    BELOW columns that carry something.
    """
    rng = np.random.default_rng(seed)
    signals = {f"s{i}": rng.normal(size=ROWS) for i in range(3)}
    frame = pd.DataFrame({**signals, **{f"n{i:02d}": rng.normal(size=ROWS) for i in range(9)}})
    score = 1.6 * signals["s0"] + 1.2 * signals["s1"] + 0.9 * signals["s2"]
    labels = (rng.random(ROWS) < 1.0 / (1.0 + np.exp(-score))).astype(np.int64)
    frame[PLANTED] = rng.normal(size=ROWS)

    x_train, x_test, y_train, y_test = train_test_split(frame, labels, test_size=HOLDOUT_FRACTION, random_state=seed, stratify=labels)
    # Planted after the split, into the holdout half only.
    x_test = x_test.copy()
    x_test[PLANTED] = np.asarray(y_test, dtype=np.float64) * 4.0 + rng.normal(scale=0.1, size=len(y_test))
    return x_train, np.asarray(y_train), x_test, np.asarray(y_test)


def test_the_planted_column_really_is_informative_on_the_holdout() -> None:
    """A teeth check: a leak nobody could detect would make the guard below pass for the wrong reason."""
    _x_train, _y_train, x_test, y_test = _bed_with_a_holdout_only_leak()

    assert abs(float(np.corrcoef(x_test[PLANTED], y_test)[0, 1])) > 0.9


def test_the_planted_column_carries_nothing_on_the_training_rows() -> None:
    """The other half of the teeth check: on the data an arm is given, the column must be noise."""
    x_train, y_train, _x_test, _y_test = _bed_with_a_holdout_only_leak()

    assert abs(float(np.corrcoef(x_train[PLANTED], y_train)[0, 1])) < 0.1


@pytest.mark.parametrize("arm_name", CHEAP_ARMS)
def test_no_arm_recovers_a_column_that_is_informative_only_on_the_holdout(arm_name: str) -> None:
    """Recovering it would prove the arm was fitted on rows it must never see."""
    x_train, y_train, _x_test, _y_test = _bed_with_a_holdout_only_leak()
    roster = build_arm_roster(int(x_train.shape[1]), k=3, random_state=0)

    result = roster[arm_name]().run(x_train, y_train)
    names = [str(column) for column in x_train.columns]
    selected = {names[index] for index, keep in enumerate(np.asarray(result.support, dtype=bool)) if keep}

    genuine = {f"s{i}" for i in range(3)}
    if result.score is not None:
        order = [names[index] for index in np.argsort(-np.asarray(result.score, dtype=np.float64))]
        rank = order.index(PLANTED)
        # It must sit below every column that actually carries signal on the training rows. Beating one of
        # the nine probes is luck; beating the signals is only possible by having seen the holdout.
        assert rank >= len(genuine), f"{arm_name} ranked the holdout-only column {rank + 1} of {len(names)}, above genuine signal it was given"
    else:
        assert not (selected & {PLANTED}) or genuine <= selected, f"{arm_name} selected the holdout-only column ahead of genuine signal"


def test_the_panel_scores_the_same_inputs_identically_twice() -> None:
    """Both statistical layers assume the pairing is exact; a panel that drifted would put noise in every delta."""
    x_train, y_train, x_test, y_test = _bed_with_a_holdout_only_leak()
    columns = [column for column in x_train.columns if column != PLANTED][:8]

    first = fit_and_score_panel(x_train, y_train, x_test, y_test, columns)
    second = fit_and_score_panel(x_train, y_train, x_test, y_test, columns)

    for member in first["models"]:
        for metric, value in first["models"][member].items():
            other = second["models"][member][metric]
            if value is None or other is None:
                assert value is other, f"{member}/{metric} was {value} once and {other} the next time"
                continue
            assert float(value) == pytest.approx(float(other), abs=1e-12), f"{member}/{metric} drifted between two identical runs: {value} vs {other}"


def test_the_gradient_boosted_member_is_pinned_for_reproducibility() -> None:
    """LightGBM is not bit-deterministic across thread counts unless it is told to be.

    Checked on the constructed estimator rather than by reading the source: the parameters are what the
    library acts on, and a refactor that dropped one would otherwise pass a source-text check.
    """
    model = panel_factories()["lightgbm"]()
    params = model.get_params()

    assert params["deterministic"] is True
    assert params["force_row_wise"] is True
    assert params["random_state"] == PANEL_RANDOM_STATE
    assert params["n_jobs"] == PANEL_THREADS, "a variable thread count is exactly what LightGBM's determinism guarantee excludes"


def test_the_linear_member_is_deterministic_by_construction() -> None:
    """The logistic member has no randomness to pin, and that must stay true rather than be assumed."""
    model = panel_factories()["logistic"]()
    params = model.get_params()

    assert params["logisticregression__max_iter"] >= 1000, "an unconverged solver is a nondeterminism of its own"


def test_two_panels_on_different_column_orders_score_the_same_set_identically() -> None:
    """Column ORDER is not part of the selection, so a panel sensitive to it would rank arms by accident."""
    x_train, y_train, x_test, y_test = _bed_with_a_holdout_only_leak()
    columns: List[str] = [column for column in x_train.columns if column != PLANTED][:6]

    forward = fit_and_score_panel(x_train, y_train, x_test, y_test, columns)
    backward = fit_and_score_panel(x_train, y_train, x_test, y_test, list(reversed(columns)))

    metrics: Dict[str, Any] = forward["models"]["logistic"]
    for metric, value in metrics.items():
        other = backward["models"]["logistic"][metric]
        if value is None or other is None:
            continue
        assert float(value) == pytest.approx(float(other), abs=1e-6), f"logistic/{metric} depends on column order: {value} vs {other}"
