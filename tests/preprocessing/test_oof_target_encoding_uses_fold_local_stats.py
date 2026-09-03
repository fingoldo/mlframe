"""The out-of-fold target encoder blended in a prior computed over the FULL training labels.

`global_mean = y_train.mean()` was computed once, outside the fold loop, and then used inside it twice: as the
shrinkage prior and as the unseen-category fallback. So a held-out row's own label informed its own encoded
value through both. For a singleton category -- no entry in the fold's stats -- the row simply received the mean
of every train label including its own, making the encoding 100% own-label contaminated.

The contamination weight is only `1/n_train`, but it is systematically aligned with the row's own label, so it
does not average out: it shows up as a positive correlation between `train_encoded` and `y_train` that a model
fitted on those rows will read as signal. That is precisely what the out-of-fold path exists to prevent, and the
docstring states outright that "a row's own label never informs its own encoded value".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.preprocessing.category_support import smoothed_target_encode_column

# `random_state` defaults to None, so KFold reshuffles on every call and two runs on identical input encode
# differently. Every test here pins it: without that the leak measurement is swamped by fold-assignment noise.

N = 400


@pytest.fixture
def singletons():
    """Every category occurs exactly once, so an honest encoding can carry no information about y at all."""
    rng = np.random.default_rng(0)
    cats = pd.Series([f"c{i}" for i in range(N)])
    y = pd.Series(rng.integers(0, 2, N).astype(float))
    return cats, y


class TestNoOwnLabelLeaksIn:
    """The defect, measured rather than argued."""

    def test_a_singleton_category_encoding_is_uncorrelated_with_its_own_label(self, singletons):
        """The sharpest case: nothing but the row's own label could produce any correlation here."""
        cats, y = singletons
        train_encoded, _ = smoothed_target_encode_column(cats, cats.iloc[:5], y, oof=True, random_state=0)
        assert abs(float(np.corrcoef(train_encoded, y)[0, 1])) < 0.10

    def test_the_leak_is_smaller_than_the_in_sample_path(self, singletons):
        """`oof=False` is documented as the leaky one; the default must be materially better, not equal."""
        cats, y = singletons
        oof_enc, _ = smoothed_target_encode_column(cats, cats.iloc[:5], y, oof=True, random_state=0)
        insample_enc, _ = smoothed_target_encode_column(cats, cats.iloc[:5], y, oof=False, random_state=0)
        assert abs(np.corrcoef(oof_enc, y)[0, 1]) < abs(np.corrcoef(insample_enc, y)[0, 1])

    def test_a_rows_encoding_does_not_move_when_only_its_own_label_flips(self, singletons):
        """The definition of "own label never informs its own value", stated directly."""
        cats, y = singletons
        base, _ = smoothed_target_encode_column(cats, cats.iloc[:5], y, oof=True, random_state=0)
        flipped_y = y.copy()
        flipped_y.iloc[7] = 1.0 - flipped_y.iloc[7]
        flipped, _ = smoothed_target_encode_column(cats, cats.iloc[:5], flipped_y, oof=True, random_state=0)
        assert base.iloc[7] == pytest.approx(flipped.iloc[7]), "row 7's encoding moved when only row 7's label changed"

    def test_the_shrinkage_prior_is_fold_local(self, singletons):
        """Not just the fallback: the `smoothing * prior` term carried the same contamination."""
        rng = np.random.default_rng(1)
        cats = pd.Series(np.repeat([f"c{i}" for i in range(20)], 20))
        y = pd.Series(rng.integers(0, 2, 400).astype(float))
        base, _ = smoothed_target_encode_column(cats, cats.iloc[:5], y, smoothing=50.0, oof=True, random_state=0)
        flipped_y = y.copy()
        flipped_y.iloc[3] = 1.0 - flipped_y.iloc[3]
        flipped, _ = smoothed_target_encode_column(cats, cats.iloc[:5], flipped_y, smoothing=50.0, oof=True, random_state=0)
        assert base.iloc[3] == pytest.approx(flipped.iloc[3])


class TestTheRestOfTheContractHolds:
    """Fold-local statistics must not break what already worked."""

    def test_the_test_encoding_still_uses_the_full_train_prior(self, singletons):
        """Correct there: test rows never inform their own encoding, so the full-train mean is the right prior."""
        cats, y = singletons
        _, test_encoded = smoothed_target_encode_column(cats, pd.Series(["unseen_a", "unseen_b"]), y, oof=True, random_state=0)
        assert test_encoded.tolist() == pytest.approx([float(y.mean())] * 2)

    def test_every_train_row_is_encoded(self, singletons):
        """No fold may be left as NaN."""
        cats, y = singletons
        train_encoded, _ = smoothed_target_encode_column(cats, cats.iloc[:5], y, oof=True, random_state=0)
        assert train_encoded.notna().all() and len(train_encoded) == N

    def test_a_real_signal_still_comes_through(self, ):
        """The encoder must still encode: a category that genuinely predicts y has to separate."""
        rng = np.random.default_rng(2)
        cats = pd.Series(np.repeat(["hot", "cold"], 200))
        y = pd.Series(np.concatenate([rng.random(200) * 0.2 + 0.8, rng.random(200) * 0.2]))
        train_encoded, _ = smoothed_target_encode_column(cats, cats.iloc[:5], y, oof=True, random_state=0)
        assert train_encoded[:200].mean() - train_encoded[200:].mean() > 0.4

    def test_the_in_sample_path_is_untouched(self, singletons):
        """`oof=False` is explicitly the legacy in-sample behaviour and must not change."""
        cats, y = singletons
        enc, _ = smoothed_target_encode_column(cats, cats.iloc[:5], y, oof=False, random_state=0)
        gm = float(y.mean())
        expected = (1.0 * y + 10.0 * gm) / 11.0  # smoothing=10.0 default, count=1 per singleton
        assert enc.to_numpy() == pytest.approx(expected.to_numpy())

    def test_the_index_is_preserved(self):
        """A non-default index must survive the fold round-trip."""
        idx = pd.Index([f"r{i}" for i in range(100)])
        cats = pd.Series(np.repeat(["a", "b"], 50), index=idx)
        y = pd.Series(np.tile([0.0, 1.0], 50), index=idx)
        train_encoded, _ = smoothed_target_encode_column(cats, cats.iloc[:5], y, oof=True, random_state=0)
        assert train_encoded.index.equals(idx)
