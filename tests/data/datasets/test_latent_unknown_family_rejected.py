"""An unknown latent family is refused instead of silently drawn as Student-t (audit 2026-11-20, finding 04-F7)."""
import pytest

from mlframe.data.datasets._latent import realize_latent
from mlframe.data.datasets.spec import LatentSpec


def test_unknown_family_raises():
    """A misspelt or unsupported family used to fall through to standard_t(4) without a word."""
    latent = LatentSpec(name="z", family="student_t", reflections=("r1",), loadings=(1.0,))
    with pytest.raises(ValueError, match="unknown family"):
        realize_latent(latent, n=50, root_seed=0, spec_name="probe")


def test_known_normal_family_still_draws():
    """The default normal family keeps working."""
    latent = LatentSpec(name="z", family="normal", reflections=("r1",), loadings=(1.0,))
    out = realize_latent(latent, n=50, root_seed=0, spec_name="probe")
    assert out is not None
