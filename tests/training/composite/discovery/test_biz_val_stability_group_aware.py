"""Unit + biz_value + cProfile tests for GROUP-AWARE (leave-wells-out) spec
stability selection in ``_stability.stability_select_specs``.

On grouped / panel data a ROW-level bootstrap keeps rows of the SAME group in
both a replicate and its complement, so a spec that only "works" by memorising
per-group levels is re-found in every replicate and looks STABLE. Group-aware
resampling draws WHOLE groups per replicate (a fraction of the distinct groups,
without replacement) so a spec is only counted stable if it survives on
DISJOINT group subsets -- i.e. generalises to UNSEEN groups.

Covered here:
* no-group path is BIT-IDENTICAL to the pre-feature row-level draw (pinned
  against an INDEPENDENT re-implementation of the draw, so a change to
  ``_subsample_indices`` / seed derivation trips it);
* group replicates contain only WHOLE groups (no group split across a replicate
  and its complement);
* selection-frequency threshold semantics preserved;
* degenerate keys (1 group / all-identical / fewer groups than replicates);
* biz_value: a fragile per-group-memorisation spec is DEMOTED by group-aware
  resampling while a genuinely group-robust spec is retained, whereas the
  row-level path wrongly keeps the fragile spec.
"""

from __future__ import annotations

import cProfile
import io
import pstats

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite.discovery._stability import (
    _subsample_groups,
    stability_select_specs,
)


class _Spec:
    def __init__(self, name: str) -> None:
        self.name = name


class _Cfg:
    """Stand-in for the discovery config; only ``stability_group_aware`` is read."""

    def __init__(self, stability_group_aware: bool = True) -> None:
        self.stability_group_aware = stability_group_aware


class _RecordingDiscovery:
    """Records every ``fit`` subsample into a shared list and selects specs from
    an optional ``selector(sub_idx) -> list[str]`` (default: single ``g``)."""

    def __init__(self, recorder, *, group_ids=None, config=None, selector=None) -> None:
        self._recorder = recorder
        self.config = config
        if group_ids is not None:
            self._group_ids_for_rerank = np.asarray(group_ids)
        self._selector = selector
        self.specs_ = []

    def fit(self, df, target, feature_cols, train_idx):
        sub = np.asarray(train_idx).copy()
        self._recorder.append(sub)
        names = self._selector(sub) if self._selector is not None else ["g"]
        self.specs_ = [_Spec(n) for n in names]
        return self


def _factory(recorder, **kw):
    return lambda: _RecordingDiscovery(recorder, **kw)


def _row_reference(train_idx, frac, n_replicates, random_state):
    """Independent re-implementation of the ROW-level draw (NOT calling
    ``_subsample_indices``), so the bit-identity pin fails if the production
    draw / seed derivation changes."""
    train_idx = np.asarray(train_idx)
    n = int(train_idx.size)
    seed_seqs = np.random.SeedSequence(int(random_state)).spawn(int(n_replicates))
    out = []
    for ss in seed_seqs:
        rng = np.random.default_rng(ss)
        if frac >= 1.0:
            out.append(train_idx)
            continue
        sub_n = min(max(2, round(frac * n)), n)
        if sub_n >= n:
            out.append(train_idx)
        else:
            out.append(np.sort(rng.choice(train_idx, size=sub_n, replace=False)))
    return out


def _grouped_df(n_groups=20, per=100, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), per)
    n = groups.size
    df = pd.DataFrame({"well": groups, "y": rng.standard_normal(n), "x": rng.standard_normal(n)})
    return df, np.arange(n), groups


# ---------------------------------------------------------------------------
# Unit: no-group path is bit-identical to the row-level draw (the pin)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("frac", [0.5, 0.3, 1.0])
def test_no_group_path_bit_identical_to_row_reference(frac):
    df, idx, _ = _grouped_df()
    rec = []
    res = stability_select_specs(
        _factory(rec),
        df,
        "y",
        ["x"],
        idx,
        n_replicates=5,
        subsample_frac=frac,
        random_state=7,
        group_aware=False,
    )
    assert res.group_aware is False
    expected = _row_reference(idx, frac, 5, 7)
    assert len(rec) == len(expected) == 5
    for got, exp in zip(rec, expected):
        np.testing.assert_array_equal(got, exp)


def test_row_pin_would_fail_if_draw_changed():
    """Sensor for the pin itself: a perturbed reference (different seed) must NOT
    match the production draw, proving the equality assertion has teeth."""
    df, idx, _ = _grouped_df()
    rec = []
    stability_select_specs(
        _factory(rec),
        df,
        "y",
        ["x"],
        idx,
        n_replicates=4,
        subsample_frac=0.5,
        random_state=7,
        group_aware=False,
    )
    wrong = _row_reference(idx, 0.5, 4, 999)  # different seed -> different draws
    mismatched = any(got.shape != exp.shape or not np.array_equal(got, exp) for got, exp in zip(rec, wrong))
    assert mismatched, "row draw must depend on random_state; pin has no teeth otherwise"


def test_group_aware_with_no_key_falls_back_bit_identical():
    """group_aware left at default (None) but NO group key available -> row path,
    identical arrays to the explicit row reference."""
    df, idx, _ = _grouped_df()
    rec = []
    res = stability_select_specs(
        _factory(rec, config=_Cfg(True)),
        df,
        "y",
        ["x"],
        idx,
        n_replicates=5,
        subsample_frac=0.5,
        random_state=7,
    )
    assert res.group_aware is False
    for got, exp in zip(rec, _row_reference(idx, 0.5, 5, 7)):
        np.testing.assert_array_equal(got, exp)


# ---------------------------------------------------------------------------
# Unit: group replicates contain only WHOLE groups
# ---------------------------------------------------------------------------


def _assert_whole_groups(recorded, train_idx, groups):
    train_idx = np.asarray(train_idx)
    for sub in recorded:
        subset = set(sub.tolist())
        for g in np.unique(groups[train_idx]):
            rows_g = set(train_idx[groups[train_idx] == g].tolist())
            inter = rows_g & subset
            assert inter == rows_g or inter == set(), f"group {g} split across replicate/complement: {len(inter)}/{len(rows_g)} rows in the replicate"


def test_group_replicates_are_whole_groups():
    df, idx, groups = _grouped_df(n_groups=20, per=100)
    rec = []
    res = stability_select_specs(
        _factory(rec, group_ids=groups, config=_Cfg(True)),
        df,
        "y",
        ["x"],
        idx,
        n_replicates=6,
        subsample_frac=0.5,
        random_state=3,
    )
    assert res.group_aware is True
    assert len(rec) == 6
    _assert_whole_groups(rec, idx, groups)
    # Each replicate holds ~half the groups (10 of 20), never all/none.
    for sub in rec:
        n_g = np.unique(groups[sub]).size
        assert 1 <= n_g < 20


def test_group_ids_via_explicit_arg_and_column_agree():
    df, idx, groups = _grouped_df(n_groups=12, per=80)
    rec_arg, rec_col = [], []
    stability_select_specs(
        _factory(rec_arg, config=_Cfg(True)),
        df,
        "y",
        ["x"],
        idx,
        n_replicates=5,
        subsample_frac=0.5,
        random_state=11,
        group_ids=groups,
    )
    stability_select_specs(
        _factory(rec_col, config=_Cfg(True)),
        df,
        "y",
        ["x"],
        idx,
        n_replicates=5,
        subsample_frac=0.5,
        random_state=11,
        group_column="well",
    )
    for a, b in zip(rec_arg, rec_col):
        np.testing.assert_array_equal(a, b)
    _assert_whole_groups(rec_arg, idx, groups)


def test_full_length_group_ids_indexed_by_train_idx():
    """group_ids full-frame length, train_idx a strict subset -> labels indexed
    positionally by train_idx; whole-group property still holds on the subset."""
    df, _, groups = _grouped_df(n_groups=15, per=60)
    train_idx = np.sort(np.random.default_rng(0).choice(groups.size, size=600, replace=False))
    rec = []
    res = stability_select_specs(
        _factory(rec, config=_Cfg(True)),
        df,
        "y",
        ["x"],
        train_idx,
        n_replicates=4,
        subsample_frac=0.5,
        random_state=5,
        group_ids=groups,
    )
    assert res.group_aware is True
    _assert_whole_groups(rec, train_idx, groups)


# ---------------------------------------------------------------------------
# Unit: selection-frequency threshold still applies (group path)
# ---------------------------------------------------------------------------


def test_threshold_semantics_group_path():
    df, idx, groups = _grouped_df(n_groups=20, per=50)
    calls = {"i": 0}  # shared across the per-replicate fresh instances.

    def selector(sub):
        names = ["always"]
        # "sometimes" selected in only 3 of 8 replicates -> freq 0.375 < 0.6.
        if calls["i"] < 3:
            names.append("sometimes")
        calls["i"] += 1
        return names

    res = stability_select_specs(
        _factory([], group_ids=groups, config=_Cfg(True), selector=selector),
        df,
        "y",
        ["x"],
        idx,
        n_replicates=8,
        subsample_frac=0.5,
        freq_threshold=0.6,
        random_state=1,
    )
    assert res.group_aware is True
    assert res.frequencies["always"] == 1.0
    assert res.stable_specs == ["always"]
    assert res.frequencies.get("sometimes", 0.0) == pytest.approx(3 / 8)
    assert res.frequencies["sometimes"] < 0.6
    for f in res.frequencies.values():
        assert 0.0 <= f <= 1.0


# ---------------------------------------------------------------------------
# Unit: degenerate keys
# ---------------------------------------------------------------------------


def test_single_group_falls_back_to_row_path():
    df, idx, _ = _grouped_df(n_groups=1, per=300)
    one = np.zeros(idx.size, dtype=int)
    rec = []
    res = stability_select_specs(
        _factory(rec, group_ids=one, config=_Cfg(True)),
        df,
        "y",
        ["x"],
        idx,
        n_replicates=5,
        subsample_frac=0.5,
        random_state=7,
    )
    # <2 distinct groups: cannot leave a well out -> row path, bit-identical.
    assert res.group_aware is False
    for got, exp in zip(rec, _row_reference(idx, 0.5, 5, 7)):
        np.testing.assert_array_equal(got, exp)


def test_all_groups_identical_label_falls_back():
    df, idx, _ = _grouped_df(n_groups=1, per=200)
    same = np.full(idx.size, 42)
    res = stability_select_specs(
        _factory([], group_ids=same, config=_Cfg(True)),
        df,
        "y",
        ["x"],
        idx,
        n_replicates=4,
        subsample_frac=0.5,
        random_state=2,
    )
    assert res.group_aware is False


def test_fewer_groups_than_replicates_ok():
    df, idx, groups = _grouped_df(n_groups=3, per=100)
    rec = []
    res = stability_select_specs(
        _factory(rec, group_ids=groups, config=_Cfg(True)),
        df,
        "y",
        ["x"],
        idx,
        n_replicates=6,
        subsample_frac=0.5,
        random_state=4,
    )
    assert res.group_aware is True
    assert len(rec) == 6
    _assert_whole_groups(rec, idx, groups)
    for sub in rec:  # round(0.5*3)=2 groups per replicate
        assert np.unique(groups[sub]).size == 2


def test_config_toggle_off_disables_group_path():
    df, idx, groups = _grouped_df(n_groups=10, per=60)
    rec = []
    res = stability_select_specs(
        _factory(rec, group_ids=groups, config=_Cfg(stability_group_aware=False)),
        df,
        "y",
        ["x"],
        idx,
        n_replicates=5,
        subsample_frac=0.5,
        random_state=7,
    )
    # config says off -> row path even though a valid key is present.
    assert res.group_aware is False
    for got, exp in zip(rec, _row_reference(idx, 0.5, 5, 7)):
        np.testing.assert_array_equal(got, exp)


def test_subsample_groups_helper_degenerate_and_frac_one():
    groups = np.array([0, 0, 1, 1, 2, 2])
    idx = np.arange(6)
    rng = np.random.default_rng(0)
    # frac>=1 returns the full index unchanged.
    np.testing.assert_array_equal(_subsample_groups(idx, groups, 1.0, rng), idx)
    # single group -> full fallback.
    np.testing.assert_array_equal(_subsample_groups(idx, np.zeros(6, int), 0.5, rng), idx)


# ---------------------------------------------------------------------------
# biz_value: group-aware DEMOTES the fragile spec; row path wrongly keeps it
# ---------------------------------------------------------------------------


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _gain(y_comp, pred, base):
    denom = _rmse(y_comp, base)
    if denom < 1e-12:
        return 0.0
    return 1.0 - _rmse(y_comp, pred) / denom


def _robust_gain(y, x0, sub, comp):
    coef = np.polyfit(x0[sub], y[sub], 1)
    pred = np.polyval(coef, x0[comp])
    base = np.full(comp.shape, float(y[sub].mean()))
    return _gain(y[comp], pred, base)


def _fragile_gain(y, groups, sub, comp):
    glob = float(y[sub].mean())
    means = {int(g): float(y[sub][groups[sub] == g].mean()) for g in np.unique(groups[sub])}
    pred = np.array([means.get(int(g), glob) for g in groups[comp]])
    base = np.full(comp.shape, glob)
    return _gain(y[comp], pred, base)


class _CompDiscovery:
    """Faithful tiny discovery: a spec is selected iff its predictor, fit on the
    replicate rows, generalises (RMSE gain over the global-mean baseline) to the
    replicate's COMPLEMENT. ``robust_x0`` = global linear signal (generalises
    always); ``fragile_group_mean`` = per-group mean encoding (generalises only
    when the complement shares the replicate's groups -> true under ROW
    resampling, false under leave-groups-out)."""

    def __init__(self, full_train_idx, group_ids, *, gate, config=None) -> None:
        self.full_train_idx = np.asarray(full_train_idx)
        self._group_ids_for_rerank = np.asarray(group_ids)
        self.gate = gate
        self.config = config
        self.specs_ = []

    def fit(self, df, target, feature_cols, train_idx):
        y = df["y"].to_numpy().astype(float)
        x0 = df["x0"].to_numpy().astype(float)
        g = self._group_ids_for_rerank
        sub = np.asarray(train_idx)
        comp = np.setdiff1d(self.full_train_idx, sub)
        names = []
        if comp.size >= 2:
            if _robust_gain(y, x0, sub, comp) >= self.gate:
                names.append("robust_x0")
            if _fragile_gain(y, g, sub, comp) >= self.gate:
                names.append("fragile_group_mean")
        self.specs_ = [_Spec(n) for n in names]
        return self


def _make_grouped_biz(n_groups=20, per=100, seed=0):
    rng = np.random.default_rng(seed)
    well_level = rng.normal(0.0, 3.0, n_groups)  # large per-well offsets (fragile signal)
    groups = np.repeat(np.arange(n_groups), per)
    n = groups.size
    x0 = rng.standard_normal(n)  # global signal (robust)
    y = well_level[groups] + 3.0 * x0 + rng.normal(0.0, 0.5, n)
    df = pd.DataFrame({"well": groups, "y": y, "x0": x0})
    return df, np.arange(n), groups


def test_biz_val_group_aware_demotes_fragile_keeps_robust():
    """Group-aware resampling must DROP a per-well-memorisation spec (selection
    frequency well below threshold) while KEEPING the genuinely group-robust
    spec; the row-level path wrongly keeps the fragile spec. Measured: group
    fragile freq ~0.0 vs robust ~1.0; row fragile freq ~1.0. Floors below use
    generous margins."""
    df, idx, groups = _make_grouped_biz()
    gate = 0.15
    kw = dict(n_replicates=6, subsample_frac=0.5, freq_threshold=0.6, random_state=13)

    def factory():
        return _CompDiscovery(idx, groups, gate=gate, config=_Cfg(True))

    group_res = stability_select_specs(factory, df, "y", ["x0"], idx, group_ids=groups, **kw)
    row_res = stability_select_specs(factory, df, "y", ["x0"], idx, group_aware=False, **kw)

    assert group_res.group_aware is True
    assert row_res.group_aware is False

    # Group-aware: fragile demoted, robust retained.
    g_fragile = group_res.frequencies.get("fragile_group_mean", 0.0)
    g_robust = group_res.frequencies.get("robust_x0", 0.0)
    assert g_robust >= 0.8, f"robust spec should survive group resampling; freq={g_robust:.2f}"
    assert g_fragile <= 0.25, f"fragile spec should be demoted; freq={g_fragile:.2f}"
    assert "robust_x0" in group_res.stable_specs
    assert "fragile_group_mean" not in group_res.stable_specs

    # Row-level: the leak -- fragile wrongly looks stable.
    r_fragile = row_res.frequencies.get("fragile_group_mean", 0.0)
    assert r_fragile >= 0.8, f"row path should (wrongly) keep fragile; freq={r_fragile:.2f}"
    assert "fragile_group_mean" in row_res.stable_specs

    # The whole point: group-aware strictly separates the two schemes on fragile.
    assert g_fragile + 0.4 < r_fragile


# ---------------------------------------------------------------------------
# cProfile harness (documented in _stability.py; runs fast here as a guard)
# ---------------------------------------------------------------------------


def _profile_stability_sweep(n_groups=200, per=200, n_replicates=5):
    """Runnable cProfile harness for the group-aware sweep. Uses a trivial
    factory so the profile isolates the DRIVER (resampling + bookkeeping) rather
    than a real fit. Verdict (see ``_stability`` docstring): the sweep is
    ``fit``-bound in production; the group draw is ~0.3 ms/replicate at 40k rows
    and never actionable."""
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(n_groups), per)
    n = groups.size
    df = pd.DataFrame({"well": groups, "y": rng.standard_normal(n), "x": rng.standard_normal(n)})
    idx = np.arange(n)
    rec = []
    pr = cProfile.Profile()
    pr.enable()
    stability_select_specs(
        _factory(rec, group_ids=groups, config=_Cfg(True)),
        df,
        "y",
        ["x"],
        idx,
        n_replicates=n_replicates,
        subsample_frac=0.5,
        random_state=0,
    )
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(15)
    return s.getvalue()


def test_profile_harness_runs_fast():
    out = _profile_stability_sweep(n_groups=40, per=100, n_replicates=5)
    assert "stability_select_specs" in out


if __name__ == "__main__":
    print(_profile_stability_sweep())
