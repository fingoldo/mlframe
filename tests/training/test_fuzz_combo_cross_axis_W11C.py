"""Cross-axis fuzz blind-spot sensors (W11C extension of the F1-F7 series).

Each sensor pins a cross-axis combo that the pairwise sampler may not always reach but is known to stress a
distinct production code path. Sensors are intentionally narrow: build a FuzzCombo with the axis overrides and
assert it constructs + the fields survive canonicalisation. Full-suite execution stays under ``--run-fuzz``.

Findings:

- C1: ``inject_all_nan_col=True`` x ``use_mrmr_fs=True`` x ``mrmr_nan_strategy_cfg`` -- an all-NaN column under
  MRMR previously crashed the MI estimator on divide-by-zero when fillna_zero produced a constant-0 post-fill
  column (B1 F4 tests-expand.md). Pin a 3-axis combo so the canonicaliser cannot collapse this away.
- C2: ``recurrent_model_cfg in {lstm, gru, transformer}`` x ``weight_schemas=("recency",)`` -- recurrent models
  have their own sample-weight handling and recency-only schemas (no uniform fallback) are a likely null-weight
  trap on empty mini-batches (B1 F3 tests-expand.md).
- C3: ``composite_discovery_enabled_cfg=True`` x ``outlier_detection in {lof, ocsvm}`` x ``imbalance_ratio !=
  balanced`` -- the F6 2-axis combo extended with a 3rd axis to catch the documented "four layered 0-row val
  tolerances" cluster (CLAUDE.md) where outlier removal + imbalance shift + composite-target discovery stack to
  collapse val to 0 rows silently.
"""
from __future__ import annotations

import pytest

from tests.training._fuzz_combo import AXES, _build_combo


def _make_combo(**overrides):
    axes = {name: values[0] for name, values in AXES.items()}
    axes.update(overrides)
    return _build_combo(models=("cb",), axes=axes, seed=0)


# ---------------------------------------------------------------------------
# C1: inject_all_nan_col x use_mrmr_fs x mrmr_nan_strategy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nan_strategy", ["separate_bin", "fillna_zero"])
def test_C1_all_nan_col_x_mrmr_x_nan_strategy_reachable(nan_strategy):
    """3-axis cross-combo must construct without canon-collapsing inject_all_nan_col or use_mrmr_fs to False.

    The MRMR NaN-handling strategy axis was added 2026-04 (line 766 _fuzz_combo) specifically because fillna_zero
    on an all-NaN column produces a constant-0 column and the MI estimator can divide-by-zero. Sensor pins the
    reachability of the bug-surface combo so future canon edits cannot quietly hide it."""
    combo = _make_combo(
        inject_all_nan_col=True,
        use_mrmr_fs=True,
        mrmr_nan_strategy_cfg=nan_strategy,
        target_type="regression",
    )
    assert combo.inject_all_nan_col is True
    assert combo.use_mrmr_fs is True
    # Distinct canonical key vs the use_mrmr_fs=False variant (canon should NOT collapse this 3-axis bug-surface).
    combo_no_mrmr = _make_combo(
        inject_all_nan_col=True,
        use_mrmr_fs=False,
        mrmr_nan_strategy_cfg=nan_strategy,
        target_type="regression",
    )
    assert combo.canonical_key() != combo_no_mrmr.canonical_key(), (
        "C1: use_mrmr_fs True/False must produce distinct canonical keys under inject_all_nan_col"
    )


# ---------------------------------------------------------------------------
# C2: recurrent_model_cfg x weight_schemas=("recency",)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rec", ["lstm", "gru", "transformer"])
def test_C2_recurrent_x_recency_only_weights_reachable(rec):
    """``recurrent_model_cfg in {lstm,gru,transformer}`` x ``weight_schemas=("recency",)`` must construct and
    keep both axis values intact.

    Recency-only schemas (no uniform fallback) are iter150-new and recurrent models handle sample weights through
    a distinct PyTorch-Lightning path. Sensor pins that the bug-surface combo is reachable; the actual fit
    behaviour is asserted by the suite when the combo runs under --run-fuzz."""
    combo = _make_combo(
        recurrent_model_cfg=rec,
        weight_schemas=("recency",),
        target_type="regression",
        n_rows=1000,
    )
    assert combo.recurrent_model_cfg == rec
    assert tuple(combo.weight_schemas) == ("recency",)
    # Distinct vs the uniform-only baseline (catches accidental canon collapse of the recency-only schema).
    combo_uniform = _make_combo(
        recurrent_model_cfg=rec,
        weight_schemas=("uniform",),
        target_type="regression",
        n_rows=1000,
    )
    assert combo.canonical_key() != combo_uniform.canonical_key(), (
        "C2: weight_schemas=(recency,) must NOT canonicalise to (uniform,) under recurrent_model_cfg"
    )


# ---------------------------------------------------------------------------
# C3: composite_discovery x outlier_detection x imbalance_ratio
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("od", ["lof", "ocsvm"])
@pytest.mark.parametrize("imb", ["rare_5pct", "rare_1pct"])
def test_C3_composite_discovery_x_outlier_x_imbalance_reachable(od, imb):
    """3-axis combo extends F6 (composite_discovery x outlier_detection) with imbalance_ratio so the documented
    "four layered 0-row val tolerances" cluster (CLAUDE.md) is fuzz-reachable: outlier removal + imbalance shift
    + composite-target discovery stack to collapse val to 0 rows in the trainer.

    Sensor only verifies combo construction; the suite-level run under --run-fuzz asserts the trainer raises a
    clear OutlierEmptyValError instead of silently producing an empty val frame."""
    combo = _make_combo(
        composite_discovery_enabled_cfg=True,
        outlier_detection=od,
        imbalance_ratio=imb,
        target_type="regression",
    )
    assert combo.composite_discovery_enabled_cfg is True
    assert combo.outlier_detection == od
    assert combo.imbalance_ratio == imb


# ---------------------------------------------------------------------------
# iter373: LTR + no native ranker = unrunnable combo filter
# ---------------------------------------------------------------------------


def test_iter373_no_ltr_combos_without_native_ranker():
    """iter373 regression: enumerator must never emit LTR combos whose model
    subset has zero native rankers (cb/xgb/lgb/mlp). Pre-fix the random pick
    surfaced c0120_0a4f6506 with models=('linear',) target=LTR which crashed
    at fit-time with NotImplementedError. The filter rejects such subsets
    in all three enumeration phases (initial powerset, pairwise greedy,
    random fill)."""
    from tests.training._fuzz_combo import (
        _LTR_NATIVE_RANKERS, enumerate_combos,
    )
    combos = enumerate_combos(target=150, master_seed=20260422)
    unrunnable = [
        c for c in combos
        if c.target_type == "learning_to_rank"
        and not any(m in _LTR_NATIVE_RANKERS for m in c.models)
    ]
    assert not unrunnable, (
        f"enumerator emitted {len(unrunnable)} LTR combos with no native ranker: "
        f"{[c.short_id() for c in unrunnable[:5]]}"
    )


# ---------------------------------------------------------------------------
# iter466: MRMR friend-graph + cluster-aggregate axes
# ---------------------------------------------------------------------------


def test_iter466_mrmr_friend_graph_cluster_axes_flow_to_kwargs():
    """iter466: the recent mrmr.py friend-graph + cluster-aggregate features
    must be (a) present as fuzz axes, (b) varied across MRMR-on combos, and
    (c) threaded into the mrmr_kwargs dict the suite passes to the MRMR
    constructor. Non-MRMR combos canonicalise the 4 axes to their mrmr.py
    defaults so they don't gain phantom variation."""
    from tests.training._fuzz_combo import (
        AXES, enumerate_combos, build_mrmr_kwargs,
    )
    for ax in (
        "mrmr_build_friend_graph_cfg", "mrmr_friend_graph_prune_cfg",
        "mrmr_cluster_aggregate_enable_cfg", "mrmr_cluster_aggregate_mode_cfg",
    ):
        assert ax in AXES, f"missing fuzz axis {ax}"

    combos = enumerate_combos(target=150, master_seed=20260422)
    mrmr_combos = [c for c in combos if c.use_mrmr_fs]
    assert mrmr_combos, "expected at least one MRMR-on combo in the suite"

    # kwargs carry all 4 keys (names match the MRMR constructor params).
    kw = build_mrmr_kwargs(mrmr_combos[0])
    for k in (
        "build_friend_graph", "friend_graph_prune",
        "cluster_aggregate_enable", "cluster_aggregate_mode",
    ):
        assert k in kw, f"mrmr_kwargs missing {k}"

    # The mode axis must actually vary across MRMR combos (both augment+replace
    # reachable) so the dedup pass didn't collapse the new variation away.
    modes = {c.mrmr_cluster_aggregate_mode_cfg for c in mrmr_combos}
    assert modes == {"augment", "replace"}, f"mode variation lost: {modes}"

    # Non-MRMR combos must canonicalise prune to its default False (gated on
    # both use_mrmr_fs AND build_friend_graph) so they can't gain variation.
    non_mrmr = [c for c in combos if not c.use_mrmr_fs]
    if non_mrmr:
        kw_off = build_mrmr_kwargs(non_mrmr[0])
        assert kw_off is None, "use_mrmr_fs=False must yield None mrmr_kwargs"
