"""biz_value: MRMR ``sample_weight`` x FE-mechanism axis (finding param_axes-06).

The selector-level ``sample_weight`` and the feature-engineering mechanisms are tested in
isolation across the suite (test_mrmr_sample_weight_unit.py covers none/uniform/shape/nonuniform
on the RAW screen; the 97 layer files exercise FE flags with NO weights), but the CROSS of the
two -- ``MRMR(fe_<mech>_enable=True).fit(X, y, sample_weight=w)`` -- is witnessed nowhere with a
behavioral assertion. This file pins that cross.

The canonical "integer weight == row duplication" equivalence is the right lens: weighting a row
by integer ``w`` SHOULD be indistinguishable from physically duplicating it ``w`` times. We assert
it on two axes and report where each holds:

* RAW name-set equivalence (the SELECTION the user gets): holds. On a strong-signal target the
  selected raw feature set is identical between duplication and ``sample_weight`` across seeds, so
  the weighted MI screen recovers the same signal a duplicated frame would. Hard-asserted.

* Engineered-recipe equivalence (the VALUES the engineered columns replay to on a probe frame):
  does NOT hold, and this is a real production gap, not a tolerance to widen.
  - ``MRMR``'s ``sample_weight`` is consumed by ``_maybe_resample_for_sample_weight`` as a
    FIXED-SIZE (n=len(X)) Monte-Carlo resample WITH REPLACEMENT at probability proportional to
    weight -- NOT a length-expanding row duplication. So the data the FE stage sees under
    ``sample_weight=w`` is a different empirical sample than the duplicated frame, and any
    data-dependent recipe parameter (relu split thresholds, per-category target-encoding means)
    differs.
  - ``filters/_target_encoding_fe.py`` has ZERO ``sample_weight`` references, so K-fold target
    encoding cannot weight its per-cell ``mean(y)`` at all; the encoded ``{col}__te`` column
    diverges from the duplication baseline even before the resample noise is considered.
  The engineered-recipe sub-cases are therefore ``xfail(strict=False)`` with a PROD-GAP reason --
  written to the CORRECT expected behavior (recipe values allclose), so the day weights flow into
  the FE recipes the xfail flips to an XPASS and the gap closes loudly.

groups smoke (part b): MRMR accepts ``groups=`` for sklearn-Pipeline API symmetry but does NOT
consume them. The documented contract is: default path WARNS (UserWarning) and stamps
``groups_ignored_=True``; ``strict_groups=True`` RAISES ``NotImplementedError``. Asserted per
mechanism on the stamped attribute -- never "groups forwarded".

Quantitative floors are calibrated from a measured development run (see each test's docstring) and
set below the measured value to absorb seed noise without losing regression detection.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from tests.feature_selection.conftest import is_fast_mode, fast_subset

warnings.filterwarnings("ignore")


# Isolated base config: the redundancy mechanisms (dcd / cluster_aggregate) and the heavier Fourier
# univariate stage are pinned OFF so the only varying axis is the FE mechanism under test. cv=3 +
# full_npermutations=3 keep each fit at ~4 s. random_seed=0 makes the resample draw deterministic.
_BASE_KW = dict(
    verbose=0,
    random_seed=0,
    max_runtime_mins=0.7,
    cv=3,
    run_additional_rfecv_minutes=False,
    full_npermutations=3,
    min_features_fallback=1,
    dcd_enable=False,
    cluster_aggregate_enable=False,
    fe_univariate_fourier_enable=False,
)

# Three FE mechanisms, each a real MRMR ctor flag (verified against filters/mrmr.py):
#   clean_orth -- fe_univariate_basis_enable (the production-DEFAULT orthogonal univariate FE)
#   mi_greedy  -- fe_mi_greedy_enable (generic unary/binary MI-greedy transform constructor)
#   kfold_te   -- fe_kfold_te_enable (K-fold target encoding of a categorical column)
# Each entry: (id, extra_ctor_kwargs, needs_categorical).
_MECHANISMS = [
    ("clean_orth", dict(fe_univariate_basis_enable=True), False),
    (
        "mi_greedy",
        dict(
            fe_univariate_basis_enable=False,
            fe_mi_greedy_enable=True,
            fe_mi_greedy_top_k=4,
            fe_mi_greedy_seed_cols_count=4,
        ),
        False,
    ),
    (
        "kfold_te",
        dict(
            fe_univariate_basis_enable=False,
            fe_kfold_te_enable=True,
            fe_kfold_te_cols=("cat",),
        ),
        True,
    ),
]

_MECH_PARAMS = [pytest.param(kw, cat, id=mid) for (mid, kw, cat) in _MECHANISMS]


def _mech_subset():
    """Under MLFRAME_FAST=1 keep one representative mechanism (mi_greedy: it exercises the FE
    transform-replay path that surfaces the prod gap). Full mode runs all three."""
    return fast_subset(_MECH_PARAMS, n=1)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def _make_numeric(seed: int, n: int = 1500):
    """Strong linear-signal binary target ``y = sign(x0 + 0.6*x1 + noise)`` with 6 noise columns.
    The dominant x0/x1 signal makes the SELECTED raw set robust to the resample noise so the
    weight-vs-duplication RAW-set equivalence holds; the engineered recipes still diverge."""
    rng = np.random.default_rng(seed)
    p = 8
    X = rng.normal(size=(n, p))
    y = (X[:, 0] + 0.6 * X[:, 1] + 0.3 * rng.normal(size=n) > 0).astype(np.int64)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(p)])
    return df, pd.Series(y, name="y")


def _make_categorical(seed: int, n: int = 1500):
    """Target driven by a per-category mean shift on an 8-level categorical ``cat`` plus a linear
    numeric ``x_num`` -- the regime where K-fold target encoding of ``cat`` is the useful FE."""
    rng = np.random.default_rng(seed)
    cat = rng.integers(0, 8, size=n)
    cat_effect = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])[cat]
    x_num = rng.normal(size=n)
    noise = rng.normal(size=(n, 3))
    score = cat_effect + 0.8 * x_num + 0.3 * rng.normal(size=n)
    y = (score > np.median(score)).astype(np.int64)
    df = pd.DataFrame(np.column_stack([x_num, noise]), columns=["x_num", "n0", "n1", "n2"])
    df["cat"] = pd.Categorical(cat)
    return df, pd.Series(y, name="y")


def _make_data(needs_categorical: bool, seed: int, n: int = 1500):
    return _make_categorical(seed, n) if needs_categorical else _make_numeric(seed, n)


# ---------------------------------------------------------------------------
# Name-set helpers (engineered tail excluded for the RAW set)
# ---------------------------------------------------------------------------


def _raw_names(sel) -> list[str]:
    names_in = set(str(c) for c in getattr(sel, "feature_names_in_", []))
    return sorted(n for n in (str(x) for x in sel.get_feature_names_out()) if n in names_in)


def _engineered_names(sel) -> list[str]:
    names_in = set(str(c) for c in getattr(sel, "feature_names_in_", []))
    return [n for n in (str(x) for x in sel.get_feature_names_out()) if n not in names_in]


def _fit_pair(ctor_kw: dict, needs_categorical: bool, seed: int, n: int = 1500):
    """Fit A on the row-duplicated frame (integer weight w in {1,2} -> physical duplication) and
    B on the original frame with ``sample_weight=w``. Returns ``(selA, selB, df, w)`` where ``df``
    is the ORIGINAL (un-duplicated) frame so a probe / transform comparison is on common columns."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    df, ys = _make_data(needs_categorical, seed, n)
    w = np.ones(n)
    w[: n // 2] = 2.0  # integer weights in {1, 2}

    dup_idx = np.repeat(np.arange(n), w.astype(int))
    df_dup = df.iloc[dup_idx].reset_index(drop=True)
    ys_dup = ys.iloc[dup_idx].reset_index(drop=True)

    kw = dict(_BASE_KW)
    kw.update(ctor_kw)
    sel_a = MRMR(**kw).fit(df_dup, ys_dup)
    sel_b = MRMR(**kw).fit(df, ys, sample_weight=w)
    return sel_a, sel_b, df, w


# ---------------------------------------------------------------------------
# (a) weight-vs-duplication equivalence
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("ctor_kw, needs_categorical", _mech_subset())
def test_biz_val_mrmr_int_weight_matches_row_duplication_raw_set(ctor_kw, needs_categorical):
    """RAW selected name-set is identical between integer-weight and row-duplication, across each
    FE mechanism. Measured: 3/3 seeds identical for clean_orth and mi_greedy; the strong x0/x1
    (or cat/x_num) signal dominates the resample noise. We require a multi-seed MAJORITY (>=2/3)
    so a single unlucky resample draw cannot trip the test, while a real regression that lets the
    weighted screen pick a different raw set on the typical seed still fails.

    This is the SELECTION the user receives -- the part of the weight contract that genuinely holds
    on the FE-enabled path. The engineered-VALUE part is the separate prod-gap test below."""
    seeds = (0, 1, 2)
    equal = 0
    detail = []
    for seed in seeds:
        sel_a, sel_b, _df, _w = _fit_pair(ctor_kw, needs_categorical, seed)
        ra, rb = _raw_names(sel_a), _raw_names(sel_b)
        # Each mechanism's signal columns must actually be recovered (not an empty-vs-empty match).
        assert len(ra) >= 1, f"seed {seed}: duplication path selected no raw features"
        if ra == rb:
            equal += 1
        detail.append(f"seed{seed}: dup={ra} sw={rb}")
    assert equal >= 2, (
        "weight-vs-duplication RAW-set equivalence failed majority: "
        f"{equal}/3 seeds equal. " + " | ".join(detail)
    )


@pytest.mark.slow
@pytest.mark.parametrize("ctor_kw, needs_categorical", _mech_subset())
@pytest.mark.xfail(
    reason="PROD GAP: MRMR sample_weight is a fixed-size MC resample (not row duplication) and "
    "_target_encoding_fe.py ignores sample_weight (0 refs), so engineered-recipe replay values "
    "diverge from the duplication baseline -- param_axes-06",
    strict=False,
)
@pytest.mark.parametrize("seed", [0, 1])
def test_biz_val_mrmr_int_weight_matches_row_duplication_engineered_values(ctor_kw, needs_categorical, seed):
    """The engineered columns common to BOTH fits must replay to allclose values on a shared probe
    frame -- the property that holds iff weights are honored as duplication inside the FE recipes.

    Currently they do NOT (measured kfold_te ``cat__te`` maxdiff ~0.033; mi_greedy relu split
    thresholds differ outright), because the resample is fixed-size MC and TE is weight-blind. The
    assertion is written to the CORRECT target (allclose, rtol=1e-6) and xfailed strict=False, so
    when weights are plumbed into the FE recipes this flips to XPASS and the gap closes visibly.

    Skips cleanly (no false XPASS) when the two fits share no engineered column name -- the
    divergence then shows up structurally in the name set rather than in values, which the raw-set
    test already covers."""
    sel_a, sel_b, df, _w = _fit_pair(ctor_kw, needs_categorical, seed)

    common_eng = sorted(set(_engineered_names(sel_a)) & set(_engineered_names(sel_b)))
    if not common_eng:
        pytest.skip(
            "no engineered column name common to both fits "
            f"(dup={_engineered_names(sel_a)} sw={_engineered_names(sel_b)})"
        )

    probe = df.iloc[:30].copy()
    out_a = sel_a.transform(probe)
    out_b = sel_b.transform(probe)
    cols_a = {str(c): c for c in getattr(out_a, "columns", [])}
    cols_b = {str(c): c for c in getattr(out_b, "columns", [])}

    checked = 0
    for name in common_eng:
        if name not in cols_a or name not in cols_b:
            continue
        va = np.asarray(out_a[cols_a[name]], dtype=float)
        vb = np.asarray(out_b[cols_b[name]], dtype=float)
        checked += 1
        assert np.allclose(va, vb, rtol=1e-6, atol=1e-8), (
            f"engineered column {name!r} replay diverges between duplication and sample_weight: "
            f"maxdiff={np.max(np.abs(va - vb)):.6g}"
        )
    if checked == 0:
        pytest.skip("common engineered names not present as transform output columns")


# ---------------------------------------------------------------------------
# (b) groups smoke per mechanism: documented strict_groups contract
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("ctor_kw, needs_categorical", _mech_subset())
def test_biz_val_mrmr_groups_warns_and_stamps_ignored_per_mechanism(ctor_kw, needs_categorical):
    """Default path: a non-None ``groups`` argument is accepted (fit completes) but NOT consumed --
    MRMR WARNS (UserWarning mentioning groups) and stamps ``groups_ignored_=True``. Asserted on the
    stamped attribute + the warning, per the documented contract; never 'groups forwarded'.

    Run once per FE mechanism so the FE pipeline does not swallow / drop the wrapper-level groups
    handling. groups_ are a coarse block structure (150 groups of 10 rows)."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    df, ys = _make_data(needs_categorical, seed=0)
    n = len(df)
    groups = np.repeat(np.arange(n // 10), 10)[:n]

    kw = dict(_BASE_KW)
    kw.update(ctor_kw)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sel = MRMR(**kw).fit(df, ys, groups=groups)

    assert getattr(sel, "groups_ignored_", None) is True, (
        "default-path groups must stamp groups_ignored_=True; "
        f"got {getattr(sel, 'groups_ignored_', 'MISSING')!r}"
    )
    assert len(_raw_names(sel)) >= 1, "fit with groups produced no selected raw features"
    group_warns = [w for w in caught if "groups" in str(w.message).lower()]
    assert group_warns, "default-path groups must emit a UserWarning mentioning groups"


@pytest.mark.slow
@pytest.mark.parametrize("ctor_kw, needs_categorical", _mech_subset())
def test_biz_val_mrmr_strict_groups_raises_per_mechanism(ctor_kw, needs_categorical):
    """strict_groups=True turns the warn-and-stamp fallback into a hard ``NotImplementedError`` so a
    group-aware caller cannot silently get group-naive MI. Verified per FE mechanism."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    df, ys = _make_data(needs_categorical, seed=0)
    n = len(df)
    groups = np.repeat(np.arange(n // 10), 10)[:n]

    kw = dict(_BASE_KW)
    kw.update(ctor_kw)
    sel = MRMR(strict_groups=True, **kw)
    with pytest.raises(NotImplementedError, match="strict_groups"):
        sel.fit(df, ys, groups=groups)


# ---------------------------------------------------------------------------
# Fast-mode representative (always runs, even under MLFRAME_FAST=1): one quick numeric fit
# exercising the weighted FE path + groups stamp, so the suite keeps a path without the slow sweep.
# ---------------------------------------------------------------------------


def test_biz_val_mrmr_sample_weight_fe_fast_representative():
    """Lightweight always-on path: a single small (n=600) mi_greedy fit under sample_weight, plus a
    groups-stamp check. Keeps coverage of the cross axis when the slow-marked sweep is skipped.

    Not a quantitative-win test (those are the slow majority tests above) -- a smoke that the
    weighted FE-enabled fit completes, selects the dominant signal, and honors the groups contract."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    n = 600 if is_fast_mode() else 800
    df, ys = _make_numeric(seed=0, n=n)
    w = np.ones(n)
    w[: n // 2] = 2.0

    kw = dict(_BASE_KW)
    kw.update(
        fe_univariate_basis_enable=False,
        fe_mi_greedy_enable=True,
        fe_mi_greedy_top_k=3,
        fe_mi_greedy_seed_cols_count=3,
        max_runtime_mins=0.4,
    )
    sel = MRMR(**kw).fit(df, ys, sample_weight=w)
    raw = _raw_names(sel)
    assert "x0" in raw or "x1" in raw, f"weighted FE fit missed the dominant signal: {raw}"

    groups = np.repeat(np.arange(n // 10), 10)[:n]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sel_g = MRMR(**kw).fit(df, ys, groups=groups)
    assert getattr(sel_g, "groups_ignored_", None) is True
    assert any("groups" in str(w.message).lower() for w in caught)
