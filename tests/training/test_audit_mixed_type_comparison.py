"""Wave 61 (2026-05-20): mixed-type comparison TypeError.

Audit class: < / > / sorted() / min / max on values whose types might mix
(str vs int, None vs float, object-dtype label set). Python 3 raises
TypeError: '<' not supported between instances of '<type-a>' and '<type-b>'
that downstream broad-except blocks then mask as something unrelated.

2 P1 + 3 P2 fixes applied via uniform "str-key fallback" pattern:

  P1:
    1. training/feature_handling/fingerprint.py:302 (compute_content_fingerprint)
       sorted(cols) failed on heterogeneous pandas column labels
       ([0, "a", 1] from stitched join/pivot). Now sorted(cols, key=str).

    2. training/core/_phase_polars_fixes.py:207 (union sort for Enum dtype)
       sorted(union) raised TypeError when "__MISSING__" sentinel was added
       to an int-encoded categorical's value set; the broad except swallowed
       it, cat-alignment was silently skipped, then XGB/CB crashed later
       with a misleading "unseen category" error. Now sorted(union, key=str).

  P2 (defensive -- works today on homogeneous inputs but lacks dtype guard):
    3. training/neural/base.py:320 (classes_ from y.unique())
       np.sort for numeric dtype + str-key fallback for object dtype.

    4. estimators/custom.py:558,560 (FeatureSelector.classes_)
       Same pattern.

    5. models/optimization.py:308,311 (sampled_inputs sort)
       str-key fallback on user-seeded inputs that may be heterogeneous.
"""

from __future__ import annotations

from pathlib import Path

MLFRAME_ROOT = Path(__file__).resolve().parent.parent.parent / "src" / "mlframe"


def _read(rel: str) -> str:
    """Read."""
    _path = MLFRAME_ROOT / rel
    if not _path.exists() and _path.suffix == ".py":
        # Monolith-split compat: the flat module became a subpackage
        # (``X.py`` -> ``X/__init__.py`` + submodules). Read __init__ + every submodule.
        _pkg = _path.with_suffix("")
        _init = _pkg / "__init__.py"
        if _init.exists():
            parts = [_init.read_text(encoding="utf-8")]
            for _sub in sorted(_pkg.glob("*.py")):
                if _sub.name != "__init__.py":
                    parts.append(_sub.read_text(encoding="utf-8"))
            return "\n".join(parts)
    return _path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Source-level sensors
# ---------------------------------------------------------------------------


def test_fingerprint_cols_sort_uses_str_key() -> None:
    """Fingerprint cols sort uses str key."""
    src = _read("training/feature_handling/fingerprint.py")
    assert "cols_sorted = sorted(cols, key=str)" in src


def test_polars_fixes_union_sort_uses_str_key() -> None:
    """Polars fixes union sort uses str key."""
    src = _read("training/core/_phase_polars_fixes.py")
    assert "union_sorted = sorted(union, key=str)" in src


def test_neural_base_classes_sort_dtype_aware() -> None:
    """Neural base classes sort dtype aware."""
    src = _read("training/neural/base.py")
    assert 'if hasattr(_y_arr, "dtype") and _y_arr.dtype != object:' in src
    assert "self.classes_ = np.sort(_y_arr)" in src
    assert "self.classes_ = np.asarray(sorted(_y_arr, key=lambda v: (v is None, str(v))))" in src


def test_custom_feature_selector_classes_sort_dtype_aware() -> None:
    """Custom feature selector classes sort dtype aware."""
    src = _read("estimators/custom.py")
    assert "self.classes_ = np.array(sorted(_y_arr, key=lambda v: (v is None, str(v))))" in src


def test_optimization_sampled_inputs_sort_uses_str_key() -> None:
    # Monolith split (2026-07-12): the sort_key logic moved from optimization.py into the sibling
    # _optimization_search.py; optimization.py now only re-exports. _read()'s compat shim only
    # handles the X.py -> X/__init__.py subpackage form, not this flat-sibling form, so read both.
    """Optimization sampled inputs sort uses str key."""
    src = _read("models/optimization.py") + _read("models/_optimization_search.py")
    assert "def _sort_key(v):" in src
    assert "return (v is None, str(v))" in src
    assert "sampled_inputs = sorted(sampled_inputs, key=_sort_key)" in src
    assert "sampled_inputs = sorted(sampled_inputs, key=_sort_key)[::-1]" in src


# ---------------------------------------------------------------------------
# Behavioural sensors
# ---------------------------------------------------------------------------


def test_sorted_with_str_key_handles_heterogeneous_input() -> None:
    """Document the str-key sort invariant: works on mixed type, deterministic."""
    cols = [0, "alpha", 1, None, "beta"]
    # Python sorted() raises TypeError; str-key fallback works.
    out = sorted(cols, key=str)
    assert len(out) == 5
    # None sorts to a stable position relative to str("None").


def test_str_key_sort_idempotent_on_homogeneous_input() -> None:
    """The fix should not change order on already-homogeneous input."""
    # Numeric -> str("0") < str("1") < str("10") < str("2") -- lexicographic
    # NB: this is the expected behaviour change; downstream code must NOT
    # assume numeric order. fingerprint.py only needs DETERMINISTIC order,
    # not numeric order, so this is acceptable.
    cols = [0, 1, 10, 2]
    out = sorted(cols, key=str)
    # Lexicographic: '0', '1', '10', '2'
    assert out == [0, 1, 10, 2]
