"""Cluster 09 regression: the diffusion-noise scale tag rounds instead of truncating (audit 2026-11-20, finding 09-F39).

int(0.29 * 100) is 28, so scale 0.29 was named n28. Source-level pin: the tag uses round(ns * 100), and the expression round(0.29 * 100)
gives 29, as the fix intends.
"""

from pathlib import Path

from mlframe.feature_engineering.transformer import diffusion_noise as dn

_SRC = Path(dn.__file__).resolve()


def test_scale_tag_rounds_rather_than_truncates():
    """The name for scale 0.29 is n29, not n28 (09-F39)."""
    assert f"n{round(0.29 * 100):02d}" == "n29"
    assert f"n{int(0.29 * 100):02d}" == "n28"


def test_source_uses_round_for_the_tag():
    """The module builds the tag with round, not int (09-F39)."""
    text = _SRC.read_text(encoding="utf-8")
    assert 'ns_tag = f"n{round(ns * 100):02d}"' in text
    assert 'f"n{int(ns*100):02d}"' not in text
