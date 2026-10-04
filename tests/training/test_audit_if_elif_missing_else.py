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

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Dispatcher fallthrough guards
# ---------------------------------------------------------------------------


def test_categorize_1d_array_rejects_unknown_method() -> None:
    """An unknown discretisation method is refused with a ValueError naming it and listing the supported ones."""
    from mlframe.feature_selection.filters import discretization as disc_mod

    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    with pytest.raises(ValueError, match=r"categorize_1d_array: unknown method='banana'.*expected one of .*'discretizer', 'numpy', 'astropy'"):
        disc_mod.categorize_1d_array(
            vals=vals,
            min_ncats=2,
            method="banana",
            astropy_sample_size=1000,
            method_kwargs={"bins": 3},
            dtype=np.int16,
            nan_filler=0.0,
        )


def test_extractors_display_diagnostic_initialises_desc_data(capsys) -> None:
    """A target of an unrecognised container type is skipped by the distribution display instead of raising NameError."""
    from mlframe.training.configs import TargetTypes
    from mlframe.training.extractors._extractors_showcase import _showcase_target_distributions

    kind = TargetTypes.BINARY_CLASSIFICATION
    _showcase_target_distributions({kind: {"t": [0, 1, 1, 0]}}, in_jupyter=False, random_seed=0, max_hist_samples=100)
    unknown_out = capsys.readouterr().out
    assert unknown_out.strip() == f"{kind} t"

    _showcase_target_distributions({kind: {"t": np.array([0, 1, 1, 0])}}, in_jupyter=False, random_seed=0, max_hist_samples=100)
    known_out = capsys.readouterr().out
    assert known_out.startswith(f"{kind} t")
    assert len(known_out.strip().splitlines()) > 1


# ---------------------------------------------------------------------------
# Behavioural sensors
# ---------------------------------------------------------------------------


def test_categorize_1d_array_raises_typed_on_unknown_method() -> None:
    """An unknown method must raise ValueError, not UnboundLocalError."""
    from mlframe.feature_selection.filters import discretization as disc_mod

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
    """Select scalable numeric columns raises typed on unknown method."""
    pl = pytest.importorskip("polars")
    from mlframe.training import pipeline as pipe_mod

    df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="unknown method='banana'"):
        pipe_mod._select_scalable_numeric_columns(df, method="banana", verbose=False)
