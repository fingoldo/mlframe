"""The PSI heatmap computed every column exactly, then drew 40 of them.

At 1M rows x 200 columns that was 18.6 s of per-bucket histogram sweeps to draw a 40x10 grid, and the
frames this repo targets are wider still. The ranking now runs on a bounded stratified row sample and only
the survivors are recomputed on all rows -- so the DRAWN numbers must stay exact, and the drawn feature SET
must not change. Both are asserted here, along with the trap a naive screen falls into: sampling the head
of a time-sorted frame shows the screen only the baseline period, where there is no drift to rank on.
"""

from __future__ import annotations

import numpy as np

from mlframe.reporting.charts.drift import PSI_SCREEN_OVERSAMPLE, compute_psi_matrix

MAXF = 8
NCOLS = MAXF * PSI_SCREEN_OVERSAMPLE + 20


def _frame(kind: str, n: int = 6000, ncols: int = NCOLS, n_drift: int = 5):
    """A frame whose drift sits in a chosen place, plus the timestamps and the drifting column indices."""
    rng = np.random.default_rng(11)
    x = rng.standard_normal((n, ncols))
    ts = np.sort(rng.random(n))
    frac = np.linspace(0.0, 1.0, n)
    drifted = np.arange(n_drift)
    if kind == "broad":
        x[:, drifted] += frac[:, None] * 6.0
    elif kind == "late":
        x[:, drifted] += (frac > 0.85)[:, None] * 8.0
    elif kind == "spike":
        x[:, drifted] += ((frac > 0.45) & (frac < 0.55))[:, None] * 8.0
    elif kind == "variance":
        x[:, drifted] *= 1.0 + frac[:, None] * 8.0
    return x, ts, {f"f{i}" for i in drifted}


def _run(x, ts, **kw):
    """``(names, matrix)`` for a frame with generated column names, so a set comparison is readable."""
    names = [f"f{i}" for i in range(x.shape[1])]
    m, kept, _ = compute_psi_matrix(x, ts, feature_names=names, max_features=MAXF, **kw)
    return list(kept), m


def test_the_screen_draws_the_same_features_as_the_exact_ranking():
    """Selection equivalence is the whole bar: the screen may only choose, never change, what is drawn."""
    for kind in ("broad", "late", "spike", "variance"):
        x, ts, _ = _frame(kind)
        exact, _ = _run(x, ts, screen_rows=0)
        screened, _ = _run(x, ts, screen_rows=1500)
        assert screened == exact, f"{kind}: the screen drew {set(screened) ^ set(exact)} differently"


def test_the_drawn_values_are_computed_on_all_rows_not_on_the_sample():
    """A screened cell must be bit-identical to the exact pass; only the candidate set is sampled."""
    x, ts, _ = _frame("broad")
    exact_names, exact_m = _run(x, ts, screen_rows=0)
    names, m = _run(x, ts, screen_rows=1500)
    assert names == exact_names
    assert np.array_equal(m, exact_m), "the drawn PSI values came from the sample, not from all rows"


def test_a_drift_confined_to_the_last_bucket_still_survives_the_screen():
    """The trap: a head sample of a time-sorted frame is all baseline, where nothing has drifted yet."""
    x, ts, drifted = _frame("late")
    names, _ = _run(x, ts, screen_rows=1500)
    assert drifted <= set(names), f"late-drifting {drifted - set(names)} were screened out"


def test_a_drift_confined_to_one_middle_bucket_still_survives_the_screen():
    """A one-bucket spike is the narrowest thing the chart exists to catch, and the easiest to sample past."""
    x, ts, drifted = _frame("spike")
    names, _ = _run(x, ts, screen_rows=1500)
    assert drifted <= set(names), f"spike-drifting {drifted - set(names)} were screened out"


def test_a_narrow_frame_never_screens():
    """Below the oversample width there is nothing to save, so the exact pass must run unchanged."""
    x, ts, _ = _frame("broad", ncols=MAXF * PSI_SCREEN_OVERSAMPLE)
    assert _run(x, ts, screen_rows=10)[0] == _run(x, ts, screen_rows=0)[0]


def test_the_screen_is_deterministic():
    """A random screen would redraw a different feature set on the same data run to run."""
    x, ts, _ = _frame("broad")
    assert _run(x, ts, screen_rows=1500)[0] == _run(x, ts, screen_rows=1500)[0]


def test_screen_rows_zero_disables_the_screen_entirely():
    """Explicit opt-out for a caller that would rather pay the exact ranking."""
    x, ts, _ = _frame("broad")
    names, m = _run(x, ts, screen_rows=0)
    assert len(names) == MAXF and m.shape[0] == MAXF


def test_non_numeric_columns_are_still_skipped_under_the_screen():
    """The string-column guard sits before the screen now; a categorical column must not reach either pass."""
    n = 3000
    rng = np.random.default_rng(4)
    ts = np.sort(rng.random(n))
    import pandas as pd

    frame = pd.DataFrame({f"f{i}": rng.standard_normal(n) for i in range(NCOLS)})
    frame["cat"] = "FIXED"
    m, names, _ = compute_psi_matrix(frame, ts, max_features=MAXF, screen_rows=800)
    assert "cat" not in names
    assert m.shape[0] == MAXF


def test_within_bucket_row_order_does_not_change_any_cell():
    """Rows are gathered in sorted order inside each bucket now; a bucket is a histogram, so that is free.

    If a cell ever became order-sensitive the sort would silently change every drawn number, which is the
    one thing the speedup is not allowed to do.
    """
    x, ts, _ = _frame("broad", n=4000)
    base_names, base_m = _run(x, ts, screen_rows=0)
    shuffle = np.random.default_rng(5).permutation(x.shape[0])
    names, m = _run(x[shuffle], ts[shuffle], screen_rows=0)
    assert names == base_names
    assert np.allclose(m, base_m, equal_nan=True)


def test_the_screen_does_not_engage_when_it_would_not_be_a_real_reduction():
    """A sample nearly as large as the frame costs a gather per column and saves nothing; measured a net loss."""
    x, ts, _ = _frame("broad", n=4000)
    names, m = _run(x, ts, screen_rows=3000)
    exact_names, exact_m = _run(x, ts, screen_rows=0)
    assert names == exact_names
    assert np.array_equal(m, exact_m)


def test_the_drift_verdict_is_unchanged_on_a_frame_with_no_drift_at_all():
    """Under pure noise the screen and the exact ranking rank noise, so the drawn SET may legitimately differ.

    What must not differ is what the chart SAYS. Ranking arbitrary among indistinguishable features is not a
    defect; reporting drift that is not there, or missing drift that is, would be.
    """
    from mlframe.reporting.charts.drift import PSI_SIGNIFICANT

    rng = np.random.default_rng(21)
    n = 6000
    x = rng.standard_normal((n, NCOLS))
    ts = np.sort(rng.random(n))
    _, exact_m = _run(x, ts, screen_rows=0)
    _, screen_m = _run(x, ts, screen_rows=1200)
    drifting = [int(np.sum(np.nanmax(mat, axis=1) > PSI_SIGNIFICANT)) for mat in (exact_m, screen_m)]
    assert drifting == [0, 0], f"a noise-only frame reported {drifting} drifting features"


def test_bin_counting_matches_numpy_histogram_including_values_sitting_on_an_edge():
    """The counts are taken with searchsorted now; numpy's own histogram is the reference they must match.

    ``np.histogram`` sorts the whole slice when the bin edges are not uniform, and quantile edges never are,
    so a full sort ran per (feature, bucket) cell. The replacement is only allowed if it counts identically:
    histogram's bins are half-open with the last one closed, which is what makes a value landing exactly on
    an interior edge the case worth pinning.
    """
    from mlframe.reporting.charts.drift import _binned_proportions, _quantile_edges

    rng = np.random.default_rng(7)
    for values in (
        rng.standard_normal(2000),
        np.round(rng.standard_normal(2000), 1),
        np.full(500, 2.5),
        np.where(rng.random(2000) < 0.5, np.nan, rng.standard_normal(2000)),
        np.full(300, np.nan),
        rng.integers(0, 3, 2000).astype(float),
        np.concatenate([rng.standard_normal(1000), [np.inf, -np.inf]]),
        rng.standard_normal(2000) * 1e-9 + 1e9,
    ):
        edges = _quantile_edges(rng.standard_normal(3000), 10)
        probe = np.concatenate([values, edges[np.isfinite(edges)]])  # values sitting exactly on an edge
        for data in (values, probe):
            finite = data[np.isfinite(data)]
            counts = np.histogram(finite, bins=edges)[0].astype(np.float64)
            expected = counts / counts.sum() if counts.sum() > 0 else np.zeros(len(edges) - 1)
            assert np.array_equal(_binned_proportions(data, edges), expected)
