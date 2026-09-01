"""Two feature-engineering defects that both make a check pass for the wrong reason.

`two_step_target_encode` forwarded `order=None` into step 1, and `ordered_target_encode` then falls back to
input ROW order. Step 2 in the same function already defaulted to the time column, so the two steps disagreed
about what "causal" meant. On a frame stored entity-major -- all of card A's events, then all of card B's, which
is how a transactions table normally arrives -- step 1's expanding mean was built from rows chronologically in
the future of the row it encodes, and the module's "leak-free" claim did not hold for the default call.

`multi_window_aggregate`'s opt-in horizon selector hard-coded `DummyClassifier(strategy="prior")` as the
no-feature baseline even on the documented regression path. Scored against a continuous target with `r2` that
does not raise -- it returns a large negative number -- so every candidate horizon cleared `min_lift` by roughly
that margin whether or not it carried any signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_engineering.multi_window_aggregate import _select_predictive_horizons
from mlframe.feature_engineering.two_step_target_encode import two_step_recency_weighted_target_encode


def _encode(df, y, order=None):
    """The encoder under test, with the fixture's column names bound."""
    return two_step_recency_weighted_target_encode(df, "ent", ["f"], y, "t", decay_half_life=5.0, order=order, causal=True)


class TestStepOneIsOrderedByTime:
    """The leakage half."""

    @pytest.fixture
    def entity_major(self):
        """Rows grouped by entity, so row order and time order disagree."""
        rng = np.random.default_rng(0)
        rows = []
        for ent in ("A", "B", "C"):
            for t in rng.permutation(20):
                rows.append({"ent": ent, "t": float(t), "f": f"{ent}{int(t) % 3}"})
        df = pd.DataFrame(rows)
        y = (df["t"].to_numpy() > 10).astype(float)
        return df, y

    def test_the_default_encoding_matches_an_explicit_time_order(self, entity_major):
        """`order=None` is documented as defaulting to `time_col`; assert it actually does."""
        df, y = entity_major
        default = _encode(df, y)
        explicit = _encode(df, y, order=df["t"].to_numpy(dtype=float))
        assert np.allclose(default, explicit, equal_nan=True)

    def test_it_differs_from_a_row_ordered_encoding(self, entity_major):
        """The discriminating half: row order and time order must NOT give the same answer on this frame."""
        df, y = entity_major
        default = _encode(df, y)
        row_ordered = _encode(df, y, order=np.arange(len(df), dtype=float))
        assert not np.allclose(default, row_ordered, equal_nan=True), "row order and time order agree; the fixture proves nothing"

    def test_shuffling_the_rows_does_not_change_the_encoding(self, entity_major):
        """Causality is a property of time, so a permutation of the frame must carry the values with it."""
        df, y = entity_major
        base = _encode(df, y)
        perm = np.random.default_rng(1).permutation(len(df))
        shuffled = _encode(df.iloc[perm].reset_index(drop=True), y[perm])
        assert np.allclose(base[perm], shuffled, equal_nan=True)

    def test_an_explicit_order_is_still_honoured(self, entity_major):
        """The parameter must keep overriding the default."""
        df, y = entity_major
        custom = _encode(df, y, order=(-df["t"]).to_numpy(dtype=float))
        assert not np.allclose(custom, _encode(df, y), equal_nan=True)


class TestTheBaselineMatchesTheTask:
    """The unconditional-pass half."""

    @pytest.fixture
    def noise(self):
        """A continuous target and two horizons' worth of pure-noise columns."""
        rng = np.random.default_rng(0)
        n = 300
        out = pd.DataFrame({f"h{h}_c{j}": rng.normal(0, 1, n) for h in (2, 5) for j in range(2)})
        return out, {2.0: ["h2_c0", "h2_c1"], 5.0: ["h5_c0", "h5_c1"]}, rng.normal(0, 1, n)

    def test_a_pure_noise_horizon_is_not_kept_on_a_regression_target(self, noise):
        """The concrete failure: the first horizon passed unconditionally, by about +8.6 of fake lift."""
        from sklearn.linear_model import Ridge

        out, cols, y = noise
        kept, lifts = _select_predictive_horizons(out, [2.0, 5.0], cols, y, 5, "r2", 0.005, Ridge())
        assert not kept, f"a pure-noise horizon was kept with lifts {lifts}"

    def test_the_reported_lift_is_not_an_offset(self, noise):
        """A lift near +8.6 on pure noise is the signature of the mismatched baseline."""
        from sklearn.linear_model import Ridge

        out, cols, y = noise
        _, lifts = _select_predictive_horizons(out, [2.0, 5.0], cols, y, 5, "r2", 0.005, Ridge())
        assert max(abs(v) for v in lifts.values()) < 1.0, lifts

    def test_a_real_regression_signal_is_still_kept(self):
        """Fixing the baseline must not make the selector reject everything."""
        from sklearn.linear_model import Ridge

        rng = np.random.default_rng(1)
        n = 400
        signal = rng.normal(0, 1, n)
        out = pd.DataFrame({"h2_c0": signal, "h5_c0": rng.normal(0, 1, n)})
        y = 3.0 * signal + rng.normal(0, 0.2, n)
        kept, _ = _select_predictive_horizons(out, [2.0, 5.0], {2.0: ["h2_c0"], 5.0: ["h5_c0"]}, y, 5, "r2", 0.005, Ridge())
        assert 2.0 in kept

    def test_the_classification_path_still_uses_a_classifier_baseline(self):
        """Choosing by task must not break the path that was already right."""
        rng = np.random.default_rng(2)
        n = 300
        out = pd.DataFrame({"h2_c0": rng.normal(0, 1, n)})
        y = rng.integers(0, 2, n)
        _, lifts = _select_predictive_horizons(out, [2.0], {2.0: ["h2_c0"]}, y, 5, "accuracy", 0.005, None)
        assert np.isfinite(list(lifts.values())).all() and max(abs(v) for v in lifts.values()) < 1.0
