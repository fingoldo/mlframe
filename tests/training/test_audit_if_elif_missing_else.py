"""Wave 55 (2026-05-20): if/elif chains without else / silent fallthrough audit.

Audit class: dispatcher chains where falling off the end silently returns None,
leaves a variable uninitialised (UnboundLocalError on next use), or skips an
intended action -- code rots when a new upstream enum value is added without
updating the dispatcher.

1 P1 + 2 P2 fixes applied:

  P1:
    1. feature_selection/filters/discretization.py:178 (categorize_1d_array)
       The `else:` branch under `method != "discretizer"` handled only "numpy"
       and "astropy"; any other value left bin_edges undefined, raising
       UnboundLocalError on line 188. Now raises a typed ValueError.

  P2:
    2. training/extractors.py:329 (show_target_diagnostics)
       isinstance dispatch over (pl.Series, pd.Series, np.ndarray) for the
       histogram path; unknown target type (LazyFrame / torch tensor / list)
       left desc_data undefined -> NameError. Initialise desc_data=None and
       skip the display when no branch matched.

    3. training/pipeline.py:1102 (_select_scalable_numeric_columns)
       Inner if/elif chains over method in {robust, standard, min_max} had no
       else branch; an unknown method produced zero method-specific stats and
       skipped the zero-spread check entirely, then propagated to
       polars_ds.scale where it crashed cryptically. Validate method at entry.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


MLFRAME_ROOT = Path(__file__).resolve().parent.parent.parent / "src" / "mlframe"


def _read(rel: str) -> str:
    """Read a source file under src/mlframe.

    Monolith-split compat: when a parent was carved into themed sibling
    files (``_extractors_showcase.py`` for the show_target_diagnostics
    body, ``_pipeline_extensions.py`` for the ``_select_scalable_numeric_columns``
    body), concat parent + siblings so source-grep sensors still match
    after the splits.
    """
    src = (MLFRAME_ROOT / rel).read_text(encoding="utf-8")
    if rel == "training/extractors.py":
        for sib_name in (
            "_extractors_showcase.py",
            "_extractors_simple.py",
            "_extractors_dtype_helpers.py",
        ):
            sib = MLFRAME_ROOT / "training" / sib_name
            if sib.exists():
                src += "\n" + sib.read_text(encoding="utf-8")
    elif rel == "training/pipeline.py":
        for sib_name in ("_pipeline_extensions.py", "_pipeline_fit_transform.py"):
            sib = MLFRAME_ROOT / "training" / sib_name
            if sib.exists():
                src += "\n" + sib.read_text(encoding="utf-8")
    return src


# ---------------------------------------------------------------------------
# Source-level sensors
# ---------------------------------------------------------------------------


def test_categorize_1d_array_rejects_unknown_method() -> None:
    src = _read("feature_selection/filters/discretization.py")
    # The fix adds an explicit else: raise ValueError with the method name.
    assert "categorize_1d_array: unknown method=" in src
    assert "expected one of " in src
    assert "'discretizer', 'numpy', 'astropy'" in src


def test_extractors_display_diagnostic_initialises_desc_data() -> None:
    src = _read("training/extractors.py")
    # The fix initialises desc_data = None before the isinstance dispatch
    # and guards the display with `if desc_data is not None`.
    assert "desc_data = None" in src
    assert "if desc_data is not None:" in src


def test_pipeline_select_scalable_validates_method() -> None:
    src = _read("training/pipeline.py")
    assert "_select_scalable_numeric_columns: unknown method=" in src
    assert "expected one of 'robust', 'standard', 'min_max'" in src


# ---------------------------------------------------------------------------
# Behavioural sensors
# ---------------------------------------------------------------------------


def test_categorize_1d_array_raises_typed_on_unknown_method() -> None:
    """An unknown method must raise ValueError, not UnboundLocalError."""
    from mlframe.feature_selection.filters import discretization as disc_mod

    if "src" + "\\" + "mlframe" not in disc_mod.__file__ and "src/mlframe" not in disc_mod.__file__:
        pytest.skip(f"discretization loaded from stale build path {disc_mod.__file__}")

    # min_ncats and bins chosen so we enter the nuniques > min_ncats branch.
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    with pytest.raises(ValueError, match="unknown method='banana'"):
        disc_mod.categorize_1d_array(
            vals=vals,
            min_ncats=2,
            method="banana",
            astropy_sample_size=1000,
            method_kwargs={"bins": 3},
            dtype=np.int16,
            nan_filler=0.0,
        )


def test_select_scalable_numeric_columns_raises_typed_on_unknown_method() -> None:
    pl = pytest.importorskip("polars")
    from mlframe.training import pipeline as pipe_mod

    if "src" + "\\" + "mlframe" not in pipe_mod.__file__ and "src/mlframe" not in pipe_mod.__file__:
        pytest.skip(f"pipeline loaded from stale build path {pipe_mod.__file__}")

    df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="unknown method='banana'"):
        pipe_mod._select_scalable_numeric_columns(df, method="banana", verbose=False)
