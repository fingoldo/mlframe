"""The default fingerprint materialized the caller's whole frame, eight times over.

`default_fingerprint` returns eight scalars, and bills itself as "stat-only". It computed them over the entire
input: `to_numpy()`, then `astype(np.float64)`, then `finite`, `finite_mask` (the identical `np.isfinite(a)`
computed a second time), `a0`, `dev`, `z`, `z**3`, `z**4`, `filled` and `inds` -- roughly two float64 copies,
two boolean masks and six float64 temporaries of the full frame live at once, with no size cap anywhere. On the
100GB frames this project sizes for, the result is an OOM inside a helper that returns eight numbers.

The statistics now come from a bounded, deterministic row sample. `n` and `p` still report the TRUE shape,
because they are part of the fingerprint's identity.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.utils import _param_oracle as po
from mlframe.utils._param_oracle import default_fingerprint

KEYS = {"n", "p", "dtype_kind", "sparsity", "mean_abs_skew", "mean_kurtosis", "cardinality_mean", "mean_abs_corr"}


class TestTheWorkIsBounded:
    """The OOM, addressed at its cause."""

    def test_a_tall_frame_is_subsampled(self):
        """The row cap, stated directly."""
        arr = np.zeros((5_000_000, 4))
        assert po._subsample_rows(arr, 5_000_000, 4).shape[0] <= po._FINGERPRINT_MAX_CELLS // 4

    def test_a_wide_frame_is_subsampled_harder(self):
        """Cells, not rows: 2000 columns must take far fewer rows than 4 columns do."""
        narrow = po._subsample_rows(np.zeros((1_000_000, 4)), 1_000_000, 4).shape[0]
        wide = po._subsample_rows(np.zeros((1_000_000, 2000)), 1_000_000, 2000).shape[0]
        assert wide < narrow

    def test_a_small_frame_is_untouched(self):
        """Sampling must not kick in where it would cost accuracy for nothing."""
        arr = np.zeros((500, 6))
        assert po._subsample_rows(arr, 500, 6) is arr

    def test_a_very_wide_frame_still_keeps_a_floor_of_rows(self):
        """A cell cap alone would take 2 rows from a 2M-column frame, and no statistic survives that."""
        # The shape arguments carry the size; the array itself only needs enough rows to slice.
        assert po._subsample_rows(np.zeros((50_000, 2)), 50_000, 100_000).shape[0] >= po._FINGERPRINT_MIN_ROWS

    def test_the_correlation_matrix_is_capped(self):
        """`np.corrcoef` builds a (p, p) matrix, which on a wide frame dwarfs everything else here."""
        rng = np.random.default_rng(0)
        fp = default_fingerprint([rng.normal(0, 1, (2000, 600))], {})
        assert np.isfinite(fp["mean_abs_corr"])


class TestTheAnswerIsStillRight:
    """A bounded fingerprint that reports different numbers would be a different bug."""

    @pytest.fixture
    def frame(self):
        """Mixed content: NaNs, a constant column, ordinary noise."""
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (500, 6))
        X[::37, 2] = np.nan
        X[:, 4] = 0.0
        return X

    def test_the_moments_match_scipy(self, frame):
        """The skew/kurtosis refactor removed three temporaries; the values must be unchanged."""
        from scipy import stats

        cols = [frame[np.isfinite(frame[:, j]), j] for j in range(frame.shape[1])]
        valid = [c for c in cols if c.size >= 3 and c.std() > 1e-12]
        fp = default_fingerprint([frame], {})
        assert fp["mean_abs_skew"] == pytest.approx(float(np.mean([abs(stats.skew(c)) for c in valid])))
        assert fp["mean_kurtosis"] == pytest.approx(float(np.mean([stats.kurtosis(c) for c in valid])))

    def test_the_true_shape_is_reported(self):
        """`n` is part of the fingerprint's identity; sampling must not shrink it."""
        rng = np.random.default_rng(1)
        fp = default_fingerprint([rng.normal(0, 1, (2_000_000, 3))], {})
        assert fp["n"] == 2_000_000 and fp["p"] == 3

    def test_it_is_deterministic(self):
        """This value is a cache key: a random sample would make every lookup miss."""
        rng = np.random.default_rng(2)
        big = rng.normal(0, 1, (600_000, 5))
        assert default_fingerprint([big], {}) == default_fingerprint([big], {})

    def test_the_key_set_is_unchanged(self, frame):
        """Consumers read these by name."""
        assert set(default_fingerprint([frame], {})) == KEYS

    def test_an_object_frame_still_returns_the_categorical_shape(self):
        """The object branch also walked every row and column with `np.unique(...astype(str))`."""
        arr = np.array([["a", "b"], ["c", "d"], ["a", "b"]] * 100_000, dtype=object)
        fp = default_fingerprint([arr], {})
        assert fp["dtype_kind"] == "O" and fp["n"] == 300_000

    def test_a_non_array_argument_still_returns_the_empty_fingerprint(self):
        """Unchanged contract."""
        assert default_fingerprint(["not an array"], {})["dtype_kind"] in {"?", "O"}
