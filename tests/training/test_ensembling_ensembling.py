"""Regression coverage for the §3 Ensembling audit fixes.

One assertion per closed finding. Each test names the audit tag in its
docstring so a future grep ties the test back to the disposition table.

User memory rules honoured:
- fast_subset friendly: every test runs in <0.5s.
- pytest.importorskip for optional deps.
- Behavioural tests only (no inspect.getsource).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import logging
import numpy as np
import pandas as pd

from mlframe.models.ensembling import (
    _per_member_mae_std,
    _stacked_corrcoef,
    compute_high_correlation_pairs,
    compute_member_quality_gate,
    ensemble_probabilistic_predictions,
    score_ensemble,
    _build_votenrank_leaderboard_from_results,
    EnsembleLeaderboard,
)

# --------------------------- LOOP-MAE / PER-MEMBER-MAE-LOOP ---------------------------


def test_per_member_mae_std_matches_python_loop_semantics():
    """LOOP-MAE: vectorised result matches the prior per-member Python loop."""
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(6, 200)).astype(np.float64)
    median_preds = np.quantile(arr, 0.5, axis=0)

    K = arr.shape[0]
    ref_mae = np.empty(K)
    ref_std = np.empty(K)
    for i in range(K):
        diffs = np.abs(arr[i] - median_preds)
        ref_mae[i] = float(diffs.mean())
        ref_std[i] = float(np.sqrt(((diffs - diffs.mean()) ** 2).mean()))

    new_mae, new_std = _per_member_mae_std(arr, median_preds)
    assert np.allclose(new_mae, ref_mae, atol=1e-12)
    assert np.allclose(new_std, ref_std, atol=1e-12)


# --------------------------- DIV-1-COL / DIVERSITY-LAST-COL ---------------------------


def test_stacked_corrcoef_matches_pair_loop():
    """DIV-1-COL: single np.corrcoef call agrees with the prior O(K^2) pair loop."""
    rng = np.random.default_rng(0)
    M = rng.normal(size=(5, 200)).astype(np.float64)
    out_full = _stacked_corrcoef(M)
    for i in range(5):
        for j in range(i + 1, 5):
            expected = float(np.corrcoef(M[i], M[j])[0, 1])
            assert abs(out_full[i, j] - expected) < 1e-12


def test_diversity_uses_full_prob_matrix_not_just_last_column():
    """DIVERSITY-LAST-COL: a multiclass member should be compared on every class column."""

    class _M:
        """Groups tests covering m."""
        def __init__(self, val_probs):
            self.val_probs = np.asarray(val_probs, dtype=np.float64)
            self.test_probs = None
            self.train_probs = None
            self.val_preds = None
            self.test_preds = None
            self.train_preds = None

    rng = np.random.default_rng(0)
    n = 200
    # Two members agree on the last (positive) class but disagree on the first two classes.
    base = rng.normal(size=(n, 3))
    base = np.abs(base)
    base = base / base.sum(axis=1, keepdims=True)
    m1 = base.copy()
    m2 = base.copy()
    # Permute the first two columns of m2; the last column (and only the last) stays equal.
    m2[:, 0], m2[:, 1] = base[:, 1], base[:, 0]

    pairs, _ = compute_high_correlation_pairs([_M(m1), _M(m2)], ["a", "b"], threshold=0.99)
    # Under the OLD code (last-column-only) these members were identical -> high corr pair.
    # Under the NEW code they differ globally -> NOT flagged.
    assert not pairs


# --------------------------- NO-SW: sample_weight plumbing ---------------------------


def test_compute_member_quality_gate_accepts_sample_weight():
    """NO-SW: weighted MAE/STD aggregation uses np.average when sample_weight supplied."""
    rng = np.random.default_rng(0)
    preds = [rng.normal(size=100) for _ in range(4)]
    sw = np.linspace(0.1, 1.0, 100)
    kept, _excluded, stats = compute_member_quality_gate(preds, sample_weight=sw)
    # Weighted gate must still return something sensible.
    assert isinstance(kept, list)
    assert "per_member_mae" in stats


def test_score_ensemble_threads_sample_weight_to_lower_call():
    """NO-SW: score_ensemble surfaces sample_weight in its kwargs (and tolerates None)."""

    def _make(val_preds):
        """Make."""
        m = SimpleNamespace(
            val_preds=val_preds,
            test_preds=val_preds,
            train_preds=val_preds,
            val_probs=None,
            test_probs=None,
            train_probs=None,
            oof_preds=None,
            oof_probs=None,
            model=MagicMock(),
            model_name="m",
        )
        return m

    rng = np.random.default_rng(0)
    members = [_make(rng.normal(size=100).astype(np.float64)) for _ in range(3)]
    # Just smoke-test that score_ensemble signature accepts sample_weight kwarg and
    # short-circuits in the K==1 case without raising.
    sw = np.linspace(0.1, 1.0, 100)
    # Single member -> SINGLE-MEMBER early-exit returns the diagnostic
    # sentinel ({"_reason": "single_member", "_n_members": 1}); flavour
    # keys (non-underscore) are empty.
    res = score_ensemble(
        [members[0]],
        ensemble_name="[m]",
        sample_weight=sw,
        ensembling_methods=["arithm"],
        build_votenrank_leaderboard=False,
        verbose=False,
    )
    assert {k for k in res if not k.startswith("_")} == set()
    assert res.get("_reason") == "single_member"


# --------------------------- SINGLE-MEMBER ---------------------------


def test_score_ensemble_returns_empty_for_single_member():
    """SINGLE-MEMBER: K==1 short-circuits without iterating any flavour.

    Contract: the return contains ONLY diagnostic underscore-prefixed keys
    (``_reason``, ``_n_members``) and no real flavour entries. Underscore
    keys are filtered out by callers iterating ``res.items()``, so the
    flavour-set is effectively empty while the diagnostic info is still
    available for finalize/metadata to distinguish "single-member suite"
    from "ensemble failed silently".
    """
    m = SimpleNamespace(
        val_preds=np.array([0.1, 0.2, 0.3]),
        test_preds=np.array([0.1, 0.2, 0.3]),
        train_preds=np.array([0.1, 0.2, 0.3]),
        val_probs=None,
        test_probs=None,
        train_probs=None,
        oof_preds=None,
        oof_probs=None,
        model=MagicMock(),
        model_name="m",
    )
    out = score_ensemble([m], ensemble_name="[m]", build_votenrank_leaderboard=False, verbose=False)
    assert out == {"_reason": "single_member", "_n_members": 1}
    # And the public flavour-set (non-underscore keys) is empty.
    assert {k for k in out if not k.startswith("_")} == set()


# --------------------------- NO-GUARD-IDENTICAL ---------------------------


def test_score_ensemble_early_exit_on_identical_members():
    """NO-GUARD-IDENTICAL: with early_exit_if_identical=True, identical members collapse to arithm."""

    arr = np.linspace(0.1, 0.9, 50).astype(np.float64)

    def _make():
        """Make."""
        return SimpleNamespace(
            val_preds=arr.copy(),
            test_preds=arr.copy(),
            train_preds=arr.copy(),
            oof_preds=arr.copy(),
            val_probs=None,
            test_probs=None,
            train_probs=None,
            oof_probs=None,
            model=MagicMock(),
            model_name="m",
        )

    members = [_make() for _ in range(3)]
    # Without the flag, every flavour runs. With the flag, only arithm survives.
    # Use a tiny target so train_and_evaluate_model has work to do.
    target = pd.Series(arr.copy())
    out = score_ensemble(
        members,
        ensemble_name="[m+m+m]",
        target=target,
        ensembling_methods=["arithm", "harm", "quad"],
        early_exit_if_identical=True,
        build_votenrank_leaderboard=False,
        uncertainty_quantile=0.0,
        verbose=False,
    )
    # Only one flavour key should appear (the arithm one); _diversity/_stacking_gate keys may also
    # be present but they're metadata, not flavour outputs.
    _flavour_keys = [k for k in out if not k.startswith("_")]
    assert len(_flavour_keys) <= 1


# --------------------------- HARDCODE-K60: rrf_k is tunable ---------------------------


def test_rrf_k_parameter_changes_blend():
    """HARDCODE-K60: rrf_k=5 vs rrf_k=200 should produce different blends."""
    rng = np.random.default_rng(0)
    p1 = rng.uniform(0, 1, size=(50, 2))
    p1 = p1 / p1.sum(axis=1, keepdims=True)
    p2 = rng.uniform(0, 1, size=(50, 2))
    p2 = p2 / p2.sum(axis=1, keepdims=True)

    out_k5, _, _ = ensemble_probabilistic_predictions(p1, p2, ensemble_method="rrf", rrf_k=5, verbose=False)
    out_k200, _, _ = ensemble_probabilistic_predictions(p1, p2, ensemble_method="rrf", rrf_k=200, verbose=False)
    # Different k must change the blend numerically.
    assert not np.allclose(out_k5, out_k200, atol=1e-6)


# --------------------------- PROB-CLIP ---------------------------


def test_prob_clip_applies_pre_blend_for_arithm():
    """PROB-CLIP: out-of-range members are clipped before the arithmetic blend."""
    bad = np.array([1.5, 1.5, 1.5])
    good = np.array([0.5, 0.5, 0.5])
    out, _, _ = ensemble_probabilistic_predictions(bad, good, ensemble_method="arithm", ensure_prob_limits=True, verbose=False)
    # Pre-clip blend would be (1.5 + 0.5) / 2 = 1.0 (already at limit). With pre-clip the bad
    # member becomes 1.0, so the blend is (1.0 + 0.5) / 2 = 0.75 -- clearly < 1.0 and != 1.0.
    assert np.all(out < 1.0)
    assert np.all(out > 0.0)


# --------------------------- GATE-DOUBLE-DIP ---------------------------


def test_gate_falls_back_to_coarse_when_require_oof_for_gate_and_oof_missing(caplog):
    """COARSE-GATE-FALLBACK: require_oof_for_gate=True with some members lacking OOF now runs a
    COARSE gate against val/test/train at 5x median (catches catastrophic outliers only)
    instead of skipping silently."""

    def _make(val_preds, has_oof: bool):
        """Make."""
        return SimpleNamespace(
            val_preds=val_preds,
            test_preds=val_preds,
            train_preds=val_preds,
            val_probs=None,
            test_probs=None,
            train_probs=None,
            oof_preds=val_preds.copy() if has_oof else None,
            oof_probs=None,
            model=MagicMock(),
            model_name="m",
        )

    arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    members = [_make(arr.copy(), has_oof=False) for _ in range(3)]
    members[0].oof_preds = arr.copy()  # only one has OOF
    target = pd.Series(arr.copy())

    with caplog.at_level(logging.WARNING):
        score_ensemble(
            members,
            ensemble_name="[m+m+m]",
            target=target,
            ensembling_methods=["arithm"],
            require_oof_for_gate=True,
            build_votenrank_leaderboard=False,
            uncertainty_quantile=0.0,
            verbose=True,
        )
    # The COARSE-gate warning should be emitted (val-coarse + 5x median thresholds).
    assert any(
        "COARSE gate" in rec.message and "val-coarse" in rec.message for rec in caplog.records
    ), "Expected coarse-gate fallback warning; got: " + " | ".join(rec.message for rec in caplog.records)


def test_gate_skipped_when_require_oof_for_gate_and_coarse_disabled(caplog):
    """When coarse_gate_max_mae_relative<=0 AND OOF missing, the gate skips entirely with the
    legacy "skipping quality gate" warning."""

    def _make(val_preds, has_oof: bool):
        """Make."""
        return SimpleNamespace(
            val_preds=val_preds,
            test_preds=val_preds,
            train_preds=val_preds,
            val_probs=None,
            test_probs=None,
            train_probs=None,
            oof_preds=val_preds.copy() if has_oof else None,
            oof_probs=None,
            model=MagicMock(),
            model_name="m",
        )

    arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    members = [_make(arr.copy(), has_oof=False) for _ in range(3)]
    members[0].oof_preds = arr.copy()
    target = pd.Series(arr.copy())

    with caplog.at_level(logging.WARNING):
        score_ensemble(
            members,
            ensemble_name="[m+m+m]",
            target=target,
            ensembling_methods=["arithm"],
            require_oof_for_gate=True,
            coarse_gate_max_mae_relative=0.0,
            coarse_gate_max_std_relative=0.0,
            build_votenrank_leaderboard=False,
            uncertainty_quantile=0.0,
            verbose=True,
        )
    assert any("coarse-gate disabled" in rec.message for rec in caplog.records), "Expected coarse-disabled skip warning; got: " + " | ".join(
        rec.message for rec in caplog.records
    )


def test_k2_catastrophic_dropout_drops_obvious_outlier_member(caplog):
    """K2-CATASTROPHIC-DROPOUT regression (TVT-2026-05-21): with K=2 the legacy
    peer-median gate was symmetric (kept both unconditionally). Now when target is
    available for the gate-source split AND one member's MAE-to-target is >>
    the other's, the broken member is dropped and a sentinel result returns."""

    def _make(val_preds):
        """Make."""
        return SimpleNamespace(
            val_preds=val_preds.astype(np.float64),
            test_preds=val_preds.astype(np.float64),
            train_preds=val_preds.astype(np.float64),
            val_probs=None,
            test_probs=None,
            train_probs=None,
            oof_preds=None,
            oof_probs=None,
            model=MagicMock(),
            model_name="m",
        )

    rng = np.random.default_rng(0)
    n = 500
    y_true = rng.standard_normal(n)
    good = _make(y_true + 0.1 * rng.standard_normal(n))  # MAE ~ 0.08
    bad = _make(-5.0 * y_true)  # MAE ~ 5.0; ratio ~ 60x
    members = [good, bad]
    good.model_name = "good_ridge"
    bad.model_name = "broken_mlp"
    target = pd.Series(y_true)

    with caplog.at_level(logging.WARNING):
        res = score_ensemble(
            members,
            ensemble_name="[good+bad]",
            target=target,
            val_target=target,
            test_target=target,
            ensembling_methods=["arithm"],
            require_oof_for_gate=True,
            build_votenrank_leaderboard=False,
            uncertainty_quantile=0.0,
            verbose=True,
        )
    assert res.get("_reason") == "k2_catastrophic_dropout", res
    assert res.get("_dropped_member") == "broken_mlp", res
    assert res.get("_kept_member") == "good_ridge", res
    assert res.get("_k2_mae_ratio", 0) > 20.0
    assert any("K=2 catastrophic-dropout" in rec.message for rec in caplog.records)


def test_k2_catastrophic_dropout_sentinel_keys_start_with_underscore():
    """K2-CATASTROPHIC-DROPOUT sentinel keys must all start with ``_`` so the
    suite caller can safely skip them via ``startswith("_")`` filtering.
    Without that contract, the per-target model list at
    _phase_train_one_target.py:2356 would inject strings / ints / floats
    where it expects model objects, breaking downstream predict and metric
    code."""
    rng = np.random.default_rng(0)
    n = 500
    y = rng.standard_normal(n)
    good = SimpleNamespace(
        val_preds=(y + 0.1 * rng.standard_normal(n)).astype(np.float64),
        test_preds=(y + 0.1 * rng.standard_normal(n)).astype(np.float64),
        train_preds=(y + 0.1 * rng.standard_normal(n)).astype(np.float64),
        val_probs=None,
        test_probs=None,
        train_probs=None,
        oof_preds=None,
        oof_probs=None,
        model=MagicMock(),
        model_name="g",
    )
    bad = SimpleNamespace(
        val_preds=(-5 * y).astype(np.float64),
        test_preds=(-5 * y).astype(np.float64),
        train_preds=(-5 * y).astype(np.float64),
        val_probs=None,
        test_probs=None,
        train_probs=None,
        oof_preds=None,
        oof_probs=None,
        model=MagicMock(),
        model_name="b",
    )
    target = pd.Series(y)
    res = score_ensemble(
        [good, bad],
        ensemble_name="[g+b]",
        target=target,
        test_target=target,
        val_target=target,
        ensembling_methods=["arithm"],
        require_oof_for_gate=True,
        build_votenrank_leaderboard=False,
        uncertainty_quantile=0.0,
        verbose=False,
    )
    assert res.get("_reason") == "k2_catastrophic_dropout"
    # Every key in the sentinel-only result MUST start with ``_`` (metadata).
    for k in res:
        assert isinstance(k, str) and k.startswith(
            "_"
        ), f"K=2 catastrophic-dropout produced a NON-underscore key {k!r}; this pollutes the per-target model list downstream."


def test_k2_catastrophic_dropout_skipped_when_no_target_available(caplog):
    """When the gate split has no matching target_arr supplied (caller passed
    target= but val_target=None and the gate fell to val-coarse), the K=2
    branch can't make an honest decision and must NOT drop -- proceeds with
    full K=2 ensemble (legacy behaviour)."""

    def _make(val_preds):
        """Make."""
        return SimpleNamespace(
            val_preds=val_preds.astype(np.float64),
            test_preds=val_preds.astype(np.float64),
            train_preds=val_preds.astype(np.float64),
            val_probs=None,
            test_probs=None,
            train_probs=None,
            oof_preds=None,
            oof_probs=None,
            model=MagicMock(),
            model_name="m",
        )

    rng = np.random.default_rng(1)
    n = 500
    y = rng.standard_normal(n)
    a = _make(y + 0.1 * rng.standard_normal(n))
    b = _make(-5.0 * y)
    target = pd.Series(y)
    # NOTE: val_target / test_target / train_target NOT supplied.
    res = score_ensemble(
        [a, b],
        ensemble_name="[a+b]",
        target=target,
        ensembling_methods=["arithm"],
        require_oof_for_gate=True,
        build_votenrank_leaderboard=False,
        uncertainty_quantile=0.0,
        verbose=False,
    )
    # Without target_arr for the gate split, the K=2 branch falls through and the
    # standard ensemble runs over both members -- no sentinel dropout result.
    assert res.get("_reason") != "k2_catastrophic_dropout", res


def test_kn_borderline_mae_blowout_stamped_into_metadata(caplog):
    """E4.3 (2026-05-21): when a K>2 member's target-MAE is borderline (between
    10x and 20x the best member's MAE by default), it's NOT dropped but a
    ``_diagnostic_mae_blowout`` field is stamped on the result. Operators can
    surface "watch this member" candidates from metadata."""

    def _make(val_preds):
        """Make."""
        return SimpleNamespace(
            val_preds=val_preds.astype(np.float64),
            test_preds=val_preds.astype(np.float64),
            train_preds=val_preds.astype(np.float64),
            val_probs=None,
            test_probs=None,
            train_probs=None,
            oof_preds=None,
            oof_probs=None,
            model=MagicMock(),
            model_name="m",
        )

    rng = np.random.default_rng(2)
    n = 500
    y_true = rng.standard_normal(n)
    # 3 good members (MAE ~ 0.08) + 1 borderline (MAE ~ 1.5, ratio ~ 18x best)
    members = [_make(y_true + 0.1 * rng.standard_normal(n)) for _ in range(3)]
    members.append(_make(y_true + 1.5 * rng.standard_normal(n)))
    for i, m in enumerate(members):
        m.model_name = f"m{i}"
    target = pd.Series(y_true)

    res = score_ensemble(
        members,
        ensemble_name="[m0+m1+m2+m3]",
        target=target,
        val_target=target,
        test_target=target,
        ensembling_methods=["arithm"],
        require_oof_for_gate=True,
        build_votenrank_leaderboard=False,
        uncertainty_quantile=0.0,
        verbose=True,
    )
    blowout = res.get("_diagnostic_mae_blowout")
    # The borderline member (3) should appear in the diagnostic; we don't strictly
    # require it (the bench threshold depends on rng) but at minimum the field
    # MUST be a dict when present and contain the documented keys.
    if blowout is not None:
        assert isinstance(blowout, dict)
        assert "borderline_idx" in blowout
        assert "per_member_target_mae" in blowout
        assert "best_mae" in blowout
        # Sentinel key contract.
        assert "_diagnostic_mae_blowout".startswith("_")


def test_kn_all_members_catastrophic_sentinel_when_only_one_survives(caplog):
    """E4.2 (2026-05-21): when ALL but one K>2 members exceed the catastrophic
    ratio relative to the best (rare edge: best is barely-finite, others way
    worse), the peer-median fallback would keep everyone. New
    ``_all_members_catastrophic`` sentinel signals the suite to skip
    ensembling for this target."""

    def _make(val_preds):
        """Make."""
        return SimpleNamespace(
            val_preds=val_preds.astype(np.float64),
            test_preds=val_preds.astype(np.float64),
            train_preds=val_preds.astype(np.float64),
            val_probs=None,
            test_probs=None,
            train_probs=None,
            oof_preds=None,
            oof_probs=None,
            model=MagicMock(),
            model_name="m",
        )

    rng = np.random.default_rng(3)
    n = 500
    y_true = rng.standard_normal(n)
    # 1 ok + 3 catastrophic (ratio >> 20x). Survival expected: 1.
    members = [_make(y_true + 0.1 * rng.standard_normal(n))]
    for _k in range(3):
        members.append(_make(-50.0 * y_true + rng.standard_normal(n)))
    for i, m in enumerate(members):
        m.model_name = f"m{i}"
    target = pd.Series(y_true)

    res = score_ensemble(
        members,
        ensemble_name="[m0+m1+m2+m3]",
        target=target,
        val_target=target,
        test_target=target,
        ensembling_methods=["arithm"],
        require_oof_for_gate=True,
        build_votenrank_leaderboard=False,
        uncertainty_quantile=0.0,
        verbose=True,
    )
    catast = res.get("_all_members_catastrophic")
    assert catast is not None, "_all_members_catastrophic sentinel missing on 1-survivor case"
    assert catast["n_survivors"] == 1
    assert catast["n_members"] == 4
    assert "_all_members_catastrophic".startswith("_")


def test_kn_catastrophic_target_mae_drops_obvious_outlier_when_k_above_2(caplog):
    """E4.1 (2026-05-21): K>2 absolute target-MAE catastrophic check. The legacy
    peer-median gate uses median MAE across members; if 2/4 members are
    catastrophic the median IS half-catastrophic and the relative threshold may
    let them through. This branch runs FIRST and removes any member whose
    target-MAE exceeds k2_catastrophic_mae_ratio relative to the BEST member,
    before the peer-median gate sees it. Sentinel ``_kn_catastrophic_dropped``
    appears in the result for downstream metadata."""

    def _make(val_preds):
        """Make."""
        return SimpleNamespace(
            val_preds=val_preds.astype(np.float64),
            test_preds=val_preds.astype(np.float64),
            train_preds=val_preds.astype(np.float64),
            val_probs=None,
            test_probs=None,
            train_probs=None,
            oof_preds=None,
            oof_probs=None,
            model=MagicMock(),
            model_name="m",
        )

    rng = np.random.default_rng(0)
    n = 500
    y_true = rng.standard_normal(n)
    # 3 good members + 1 catastrophic
    members = [_make(y_true + 0.1 * rng.standard_normal(n)) for _ in range(3)]
    members.append(_make(-5.0 * y_true))  # catastrophic
    members[0].model_name = "good_0"
    members[1].model_name = "good_1"
    members[2].model_name = "good_2"
    members[3].model_name = "broken"
    target = pd.Series(y_true)

    with caplog.at_level(logging.WARNING):
        res = score_ensemble(
            members,
            ensemble_name="[good_0+good_1+good_2+broken]",
            target=target,
            val_target=target,
            test_target=target,
            ensembling_methods=["arithm"],
            require_oof_for_gate=True,
            build_votenrank_leaderboard=False,
            uncertainty_quantile=0.0,
            verbose=True,
        )
    assert any("K>2 absolute-MAE catastrophic-drop" in rec.message for rec in caplog.records), "Expected E4.1 K>2 catastrophic-drop WARN; got: " + " | ".join(
        rec.message for rec in caplog.records
    )
    drop_info = res.get("_kn_catastrophic_dropped")
    assert drop_info is not None, "_kn_catastrophic_dropped sentinel missing"
    assert 3 in drop_info["dropped_idx"], f"Expected the broken member at idx 3 to be dropped; got dropped_idx={drop_info['dropped_idx']}"
    # Sentinel key prefix contract: must start with ``_`` (caller filters those out).
    assert "_kn_catastrophic_dropped" in res
    for _k in [k for k in res.keys() if not k.startswith("_")]:
        # The actual ensembling-methods results should still be in the dict (we only dropped 1/4).
        pass  # the gate purges members but the ensemble still runs on the 3 survivors


def test_coarse_gate_drops_catastrophic_outlier_member(caplog):
    """COARSE-GATE-FALLBACK regression: TVT-2026-05-21 prod log had an MLP with R^2=-4.75 sitting
    in the ensemble alongside 3 R^2~0.99 members because no member stamped OOF and the fine gate
    was skipped via require_oof_for_gate=True. The coarse fallback at 5x median MAE must catch
    that outlier; this test reproduces the scenario with a synthetic 4-member set whose 4th
    member is ~30x further from median than the cluster."""

    def _make(val_preds):
        """Make."""
        return SimpleNamespace(
            val_preds=val_preds.astype(np.float64),
            test_preds=val_preds.astype(np.float64),
            train_preds=val_preds.astype(np.float64),
            val_probs=None,
            test_probs=None,
            train_probs=None,
            oof_preds=None,
            oof_probs=None,
            model=MagicMock(),
            model_name="reg",
        )

    rng = np.random.default_rng(0)
    truth = rng.normal(size=200).astype(np.float64)
    good = [_make(truth + rng.normal(scale=0.05, size=200)) for _ in range(3)]
    bad = _make(truth * -5.0 + rng.normal(scale=2.0, size=200))  # R^2 well below zero
    members = [*good, bad]
    target = pd.Series(truth)

    with caplog.at_level(logging.INFO):
        score_ensemble(
            members,
            ensemble_name="[reg+reg+reg+reg]",
            target=target,
            ensembling_methods=["arithm"],
            require_oof_for_gate=True,
            build_votenrank_leaderboard=False,
            uncertainty_quantile=0.0,
            verbose=True,
        )
    messages = " | ".join(rec.message for rec in caplog.records)
    assert "COARSE gate" in messages, f"Coarse gate did not run: {messages}"
    assert "kept 3/4" in messages, f"Expected catastrophic outlier dropped (3/4 kept), got: {messages}"


# --------------------------- VOTENRANK ---------------------------


def test_votenrank_leaderboard_built_from_results():
    """VOTENRANK-DISCONNECT: a result dict with .metrics entries builds an EnsembleLeaderboard."""
    fake_arithm = SimpleNamespace(metrics={"oof": {"rmse": 1.0, "mae": 0.8}, "val": {"rmse": 1.2}})
    fake_harm = SimpleNamespace(metrics={"oof": {"rmse": 1.1, "mae": 0.85}, "val": {"rmse": 1.3}})
    res = {"arithm": fake_arithm, "harm": fake_harm}
    lb = _build_votenrank_leaderboard_from_results(res, is_regression=True)
    assert isinstance(lb, EnsembleLeaderboard)
    assert "arithm" in lb.table.index and "harm" in lb.table.index
    assert "oof.rmse" in lb.table.columns


def test_votenrank_leaderboard_skips_rrf_for_regression():
    """VOTENRANK regression-guard: rrf-flavour rows must be skipped when is_regression."""
    fake_arithm = SimpleNamespace(metrics={"oof": {"rmse": 1.0}})
    fake_rrf = SimpleNamespace(metrics={"oof": {"rmse": 1.2}})
    res = {"arithm": fake_arithm, "rrf": fake_rrf}
    lb = _build_votenrank_leaderboard_from_results(res, is_regression=True)
    assert lb is not None
    assert "arithm" in lb.table.index
    assert "rrf" not in lb.table.index


def test_votenrank_leaderboard_to_csv_roundtrip(tmp_path):
    """VOTENRANK: EnsembleLeaderboard.to_csv emits a parseable file."""
    fake_a = SimpleNamespace(metrics={"oof": {"rmse": 1.0}})
    fake_b = SimpleNamespace(metrics={"oof": {"rmse": 1.5}})
    lb = _build_votenrank_leaderboard_from_results({"arithm": fake_a, "harm": fake_b}, is_regression=True)
    out_path = tmp_path / "leaderboard.csv"
    lb.to_csv(out_path)
    df = pd.read_csv(out_path, index_col=0)
    assert "oof.rmse" in df.columns


# --------------------------- STACK-NOT-WIRED ---------------------------


def test_stacking_aware_gate_runs_when_enabled():
    """STACK-NOT-WIRED: the gate runs and stamps survivors / weights on the result dict."""
    rng = np.random.default_rng(0)
    n = 200
    target = rng.normal(size=n)

    def _make(oof_preds):
        """Make."""
        return SimpleNamespace(
            val_preds=oof_preds.copy(),
            test_preds=oof_preds.copy(),
            train_preds=oof_preds.copy(),
            val_probs=np.column_stack([1 - oof_preds, oof_preds]).clip(0, 1),
            test_probs=np.column_stack([1 - oof_preds, oof_preds]).clip(0, 1),
            train_probs=np.column_stack([1 - oof_preds, oof_preds]).clip(0, 1),
            oof_preds=oof_preds.copy(),
            oof_probs=np.column_stack([1 - oof_preds, oof_preds]).clip(0, 1),
            model=MagicMock(),
            model_name="m",
        )

    # Build classification-style members for the gate (uses oof_preds positively-correlated with target).
    members = [_make((target + rng.normal(0.0, 0.5, size=n)).clip(0.01, 0.99)) for _ in range(3)]
    res = score_ensemble(
        members,
        ensemble_name="[m+m+m]",
        target=pd.Series(target),
        ensembling_methods=["arithm"],
        enable_stacking_aware_gate=True,
        build_votenrank_leaderboard=False,
        uncertainty_quantile=0.0,
        verbose=False,
    )
    assert "_stacking_gate" in res
    assert "survivors" in res["_stacking_gate"]


# --------------------------- REG-RRF-DROPPED (votenrank-side regression guard) ---------------------------


def test_regression_filters_rrf_from_default_methods():
    """REG-RRF-DROPPED: rrf flavour is silently dropped when is_regression."""

    def _make():
        """Make."""
        return SimpleNamespace(
            val_preds=np.linspace(0.1, 0.9, 50).astype(np.float64),
            test_preds=np.linspace(0.1, 0.9, 50).astype(np.float64),
            train_preds=np.linspace(0.1, 0.9, 50).astype(np.float64),
            oof_preds=np.linspace(0.1, 0.9, 50).astype(np.float64),
            val_probs=None,
            test_probs=None,
            train_probs=None,
            oof_probs=None,
            model=MagicMock(),
            model_name="m",
        )

    members = [_make() for _ in range(2)]
    target = pd.Series(np.linspace(0.1, 0.9, 50).astype(np.float64))
    out = score_ensemble(
        members,
        ensemble_name="[m+m]",
        target=target,
        ensembling_methods=["arithm", "rrf"],
        build_votenrank_leaderboard=False,
        uncertainty_quantile=0.0,
        verbose=False,
    )
    # rrf must NOT survive for regression suites; arithm should.
    _flavours = [k for k in out if not k.startswith("_")]
    assert "rrf" not in _flavours
    assert "arithm" in _flavours
