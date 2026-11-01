"""``get_sapp_dataset`` only draws from scipy.stats distribution objects (EXE-19); the staleness check only passes hex object names to git (CMD-02)."""

import pytest

from mlframe.data.datasets._sapp import get_sapp_dataset


@pytest.mark.parametrize("name", ["_rvs_method", "describe", "os", "norm.rvs", "", "tstd"])
def test_a_scipy_stats_name_that_is_not_a_distribution_is_refused(name):
    """Functions, private names and dotted paths in ``scipy.stats`` are not distributions and are refused before any call."""
    with pytest.raises(ValueError, match="not a scipy.stats distribution"):
        get_sapp_dataset(distr_name=name, N=20)


def test_a_distribution_name_still_draws():
    """The default and a named continuous distribution still produce a frame of the requested size."""
    frame = get_sapp_dataset(distr_name="uniform", N=30)[0]
    assert len(frame) == 30
