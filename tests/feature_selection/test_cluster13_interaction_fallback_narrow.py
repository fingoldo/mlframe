"""Cluster 13 regression: the interaction proxy's additive fallback catches only the expected failures (audit 2026-11-20, finding 13-F26).

Source-level pin: both fallback handlers name the narrow set (ValueError, TypeError, RuntimeError, LinAlgError) rather than Exception, so
a programmer error such as NameError surfaces instead of silently degrading the selection to additive.
"""

from pathlib import Path

from mlframe.feature_selection import shap_proxied_fs as pkg

_MOD = Path(pkg.__file__).resolve().parent / "_shap_proxied_fit_interactions.py"


def test_fallback_handlers_are_narrow():
    """No broad Exception handler remains in the interaction module (13-F26)."""
    text = _MOD.read_text(encoding="utf-8")
    assert "except Exception as exc:" not in text
    assert text.count("except (ValueError, TypeError, RuntimeError, np.linalg.LinAlgError) as exc:") == 2
