"""MAR and MNAR hide exactly the requested share even when the driver is tied, and a NaN in the ordering column does not switch the mask off."""
import numpy as np

from mlframe.data.datasets._missing import missing_mask
from mlframe.data.datasets.spec import MissingnessSpec


def _mask(mechanism, columns, rate, seed=0):
    """Mask of column 'x' under the given mechanism."""
    spec = MissingnessSpec(column="x", mechanism=mechanism, driver="d" if mechanism == "mar" else None)
    return missing_mask(spec, columns, rate, np.random.default_rng(seed))


def test_binary_driver_hides_the_requested_share_not_every_tied_row():
    """A 0/1 driver that is half ones and a rate of 0.1: the quantile cut was 1.0 and ALL ones were masked (50%)."""
    n = 1000
    cols = {"x": np.arange(n, dtype=float), "d": (np.arange(n) % 2).astype(float)}
    m = _mask("mar", cols, 0.1)
    assert m.sum() == 100
    assert cols["d"][m].min() == 1.0  # still driven by the top of d


def test_nan_in_the_ordering_column_does_not_disable_the_mask():
    """np.quantile returned NaN and nothing was masked."""
    n = 500
    x = np.arange(n, dtype=float)
    x[::50] = np.nan
    m = _mask("mnar", {"x": x}, 0.2)
    assert m.sum() == 100 and not m[::50].any()
