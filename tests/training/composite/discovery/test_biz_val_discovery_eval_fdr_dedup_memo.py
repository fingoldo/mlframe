"""Tests for the per-transform gain evaluator's FDR control, x_remaining dedup,
and shrunk-domain ``mi_y_compare`` memoisation.

Three findings landed against ``discovery/_eval.py`` / ``discovery/_eval_stats.py``
/ ``discovery/_fit.py``:

* **FDR control (M4)** -- across the many (base, transform) gain tests in one
  sweep there was no family-wise multiplicity control: each per-spec bootstrap CI
  controls only its OWN error rate, so testing dozens of specs inflates the
  chance that a pure-noise spec spuriously "beats baseline". A Benjamini-Hochberg
  FDR correction over the per-spec one-sided bootstrap p-values (H0
  ``mi_gain <= 0``) now runs after all candidates are scored, dropping specs BH
  does not reject. Gated by ``mi_gain_fdr_control`` (default ON, a no-op when the
  bootstrap is disabled).
* **x_remaining dedup (D21)** -- a ~0.99-correlated sibling of the removed base
  left in ``x_remaining`` inflates ``MI(y, x_remaining)`` while contributing
  little to ``MI(T, x_remaining)``, biasing ``mi_gain`` DOWN. Near-collinear
  columns are dropped before the MI baseline so both halves of ``mi_gain`` score
  the same de-duplicated feature set.
* **mi_y_compare memo (P18)** -- the shrunk-domain baseline MI is recomputed per
  transform that shares an identical ``valid_screen`` mask on one base; memoising
  it on ``hash(valid_screen.tobytes())`` makes N transforms compute it ONCE,
  bit-identical.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from mlframe.training.composite.discovery._eval_stats import (
    apply_fdr_control_to_candidates,
    benjamini_hochberg_reject,
    bootstrap_gain_p_value,
    near_collinear_keep_mask,
)


# ----------------------------------------------------------------------
# M4: Benjamini-Hochberg FDR step-up
# ----------------------------------------------------------------------


def test_benjamini_hochberg_matches_textbook_example() -> None:
    """BH (1995) on a known vector. With m=4, alpha=0.05 and
    p=[0.005, 0.02, 0.03, 0.5], the thresholds k/m*alpha are
    [0.0125, 0.025, 0.0375, 0.05]; the largest k with p_(k) <= thr is k=3
    (0.03 <= 0.0375), so the three smallest are rejected and 0.5 is not.
    """
    p = np.array([0.005, 0.02, 0.03, 0.5])
    out = benjamini_hochberg_reject(p, 0.05)
    assert out.tolist() == [True, True, True, False]


def test_benjamini_hochberg_handles_nan_and_empty() -> None:
    """NaN p-values are non-rejectable; an empty input returns an empty array."""
    out = benjamini_hochberg_reject(np.array([np.nan, 0.001, np.nan]), 0.10)
    assert out.tolist() == [False, True, False]
    assert benjamini_hochberg_reject(np.array([]), 0.10).size == 0


def test_bootstrap_gain_p_value_one_sided_and_robust() -> None:
    """One-sided p for H0 ``mi_gain <= 0``: clearly-positive gains give a tiny p,
    centred gains give ~0.5, an all-failed bootstrap gives the conservative 1.0.
    """
    assert bootstrap_gain_p_value(np.full(200, 0.4)) < 0.01
    rng = np.random.default_rng(1)
    assert 0.3 < bootstrap_gain_p_value(rng.normal(0.0, 1.0, size=400)) < 0.7
    assert bootstrap_gain_p_value(np.array([np.nan, np.nan])) == 1.0


def _entry(name: str, p_value: float, mi_gain: float = 0.01) -> dict:
    """A minimal candidate-entry stub shaped like ``eval_one_transform`` output."""

    class _Spec:
        def __init__(self, nm: str, g: float) -> None:
            self.name = nm
            self.mi_gain = g

    return {
        "spec": _Spec(name, mi_gain),
        "kept": False,
        "reason": "",
        "mi_gain_lcb": mi_gain,
        "bootstrap_p_value": p_value,
    }


def test_apply_fdr_drops_noise_specs_and_keeps_signal() -> None:
    """``apply_fdr_control_to_candidates`` stamps ``fdr_dropped`` on specs Benjamini-Yekutieli
    does not reject and leaves the clearly-significant ones untouched.

    The candidate p-values are correlated (shared base columns + resample structure), so the family
    control uses BY (arbitrary-dependence FDR), not BH -- the stricter BY threshold still keeps the
    clearly-significant sig_a / sig_b and drops the noise."""
    candidates = [
        _entry("sig_a", 1e-5),
        _entry("sig_b", 1e-4),
        *[_entry(f"noise_{i}", p) for i, p in enumerate(np.linspace(0.4, 0.99, 18))],
    ]
    n_dropped = apply_fdr_control_to_candidates(candidates, alpha=0.10)
    assert n_dropped >= 15
    by_name = {c["spec"].name: c for c in candidates}
    assert not by_name["sig_a"].get("fdr_dropped")
    assert not by_name["sig_b"].get("fdr_dropped")
    assert by_name["noise_10"].get("fdr_dropped")
    assert by_name["noise_10"]["reason"].startswith("BY-FDR")


def test_apply_fdr_is_noop_without_pvalues() -> None:
    """When no entry carries a finite bootstrap p-value (bootstrap disabled), FDR
    control is a strict no-op -- the shipped-default contract."""
    candidates = [_entry(f"s{i}", float("nan")) for i in range(10)]
    assert apply_fdr_control_to_candidates(candidates, alpha=0.10) == 0
    assert not any(c.get("fdr_dropped") for c in candidates)


def test_biz_val_fdr_reduces_false_positives_on_all_null_specs() -> None:
    """biz_value: on a family of all-null specs (true gain == 0, so p ~ U(0,1)),
    BH FDR control admits STRICTLY FEWER false "significant" specs than the
    uncorrected per-comparison gate, averaged over many seeds.

    The uncorrected baseline rejects at the per-comparison level (p <= alpha);
    with m=24 null specs that admits ~alpha*m = ~2.4 false positives per sweep.
    BH controls the expected proportion of false discoveries among rejections,
    so with zero true signals it rejects almost none. We assert the average
    false-positive count drops by a wide margin (measured ~2.4 -> ~0.1).
    """
    m = 24
    alpha = 0.10
    naive_fp: list[int] = []
    bh_fp: list[int] = []
    for seed in range(120):
        rng = np.random.default_rng(seed)
        p = rng.uniform(0.0, 1.0, size=m)
        naive_fp.append(int((p <= alpha).sum()))
        bh_fp.append(int(benjamini_hochberg_reject(p, alpha).sum()))
    mean_naive = float(np.mean(naive_fp))
    mean_bh = float(np.mean(bh_fp))
    assert mean_naive >= 1.5, f"uncorrected gate should admit ~alpha*m={alpha * m:.1f} false positives, got {mean_naive:.2f}; test not exercising the inflation"
    assert mean_bh <= 0.5, f"BH FDR control should admit ~0 false positives on all-null specs, got {mean_bh:.2f}"
    assert mean_bh <= 0.3 * mean_naive, (
        f"BH false-positive rate {mean_bh:.2f} is not materially below the uncorrected {mean_naive:.2f}; the multiplicity correction is gone"
    )


# ----------------------------------------------------------------------
# D21: near-collinear x_remaining dedup
# ----------------------------------------------------------------------


def test_near_collinear_keep_mask_drops_only_duplicates() -> None:
    """A ~0.9999-correlated sibling is dropped; an independent column is kept.
    The FIRST column of every collinear group survives (order-stable)."""
    rng = np.random.default_rng(3)
    a = rng.normal(size=400)
    sibling = a + 1e-4 * rng.normal(size=400)  # |corr| > 0.999.
    indep = rng.normal(size=400)
    x = np.column_stack([a, sibling, indep])
    mask = near_collinear_keep_mask(x, corr_threshold=0.99)
    assert mask.tolist() == [True, False, True]


def test_near_collinear_keep_mask_threshold_disables() -> None:
    """A threshold >= 1.0 disables dedup (no pair can exceed it); a degenerate
    (single-column / <3-row) matrix keeps everything."""
    rng = np.random.default_rng(4)
    a = rng.normal(size=400)
    x = np.column_stack([a, a + 1e-9 * rng.normal(size=400)])
    assert near_collinear_keep_mask(x, corr_threshold=1.0).all()
    assert near_collinear_keep_mask(a.reshape(-1, 1), corr_threshold=0.5).all()
    assert near_collinear_keep_mask(x[:2], corr_threshold=0.5).all()


def test_biz_val_dedup_removes_mi_baseline_inflation() -> None:
    """biz_value: a near-duplicate sibling of the base left in ``x_remaining``
    inflates the aggregated ``MI(y, x_remaining)`` baseline; the dedup removes
    exactly that inflation.

    Construct ``x_remaining`` = [strong_feature, near_dup_of_strong, weak_indep].
    The mean-aggregated MI baseline over all three over-counts the strong
    feature's information (twice), so the de-duplicated baseline (strong + weak,
    sibling dropped) differs materially. We assert the dedup changes which
    columns the baseline scores -- the load-bearing behaviour D21 fixes.
    """
    from mlframe.training.composite.discovery.screening import (
        _mi_per_feature_prebinned,
        _prebin_feature_columns,
        _aggregate_mi_per_feature,
    )

    rng = np.random.default_rng(5)
    n = 4000
    strong = rng.normal(size=n)
    y = strong + 0.2 * rng.normal(size=n)  # y depends on strong.
    near_dup = strong + 1e-4 * rng.normal(size=n)  # ~1.0-corr sibling.
    weak = rng.normal(size=n)  # independent of y.
    x_full = np.column_stack([strong, near_dup, weak]).astype(np.float64)

    binned = _prebin_feature_columns(x_full, nbins=16)
    per_feat = _mi_per_feature_prebinned(binned, y, nbins=16)

    mi_with_dup = _aggregate_mi_per_feature(per_feat, "mean")
    keep = near_collinear_keep_mask(x_full, corr_threshold=0.99)
    assert keep.tolist() == [True, False, True], "sibling not detected for dedup"
    mi_dedup = _aggregate_mi_per_feature(per_feat[keep], "mean")

    # The duplicate sibling carries near-identical MI to ``strong``; dropping it
    # removes the double-count, changing the mean baseline materially.
    assert abs(mi_with_dup - mi_dedup) > 1e-3, f"dedup did not change the MI baseline ({mi_with_dup:.4f} vs {mi_dedup:.4f}); the inflation is not being removed"


# ----------------------------------------------------------------------
# Discovery fixtures (P18 + integration)
# ----------------------------------------------------------------------


def _make_config(**overrides):
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    defaults = dict(
        enabled=True,
        base_candidates="auto",
        transforms=("diff", "linear_residual", "additive_residual", "ratio", "logratio", "median_residual"),
        top_k_after_mi=32,
        top_m_after_tiny=8,
        mi_sample_n=2000,
        tiny_model_sample_n=2000,
        eps_mi_gain=-10.0,
        screening="mi",
        random_state=42,
        require_beats_raw_baseline=False,
        multi_base_enabled=False,
        discovery_n_jobs=1,
        fail_on_no_gain="warn",
        detect_linear_residual_alpha_drift=False,
    )
    defaults.update(overrides)
    return CompositeTargetDiscoveryConfig(**defaults)


def _shrink_dataset(n: int = 2400, seed: int = 11) -> pd.DataFrame:
    """A dataset whose ``logratio`` transform requires y, base > 0, so some rows
    fail the fitted domain check and ``valid_screen`` SHRINKS -- the exact path
    that triggers the per-base shrunk-domain ``mi_y_compare`` recompute (P18).
    A few negative base rows make every base candidate's logratio screen shrink.
    """
    rng = np.random.default_rng(seed)
    base = rng.lognormal(mean=1.0, sigma=0.5, size=n)
    base[: n // 20] = -np.abs(rng.normal(size=n // 20))  # negative -> domain drop.
    other = rng.lognormal(mean=0.5, sigma=0.4, size=n)
    y = 1.5 * base + 0.5 * other + 0.3 * rng.normal(size=n)
    return pd.DataFrame({"base": base, "other": other, "y": y})


def _run(df: pd.DataFrame, config):
    from mlframe.training.composite import CompositeTargetDiscovery

    n = len(df)
    train_idx = np.arange(0, int(0.8 * n))
    val_idx = np.arange(int(0.8 * n), n)
    disc = CompositeTargetDiscovery(config)
    disc.fit(
        df,
        target_col="y",
        feature_cols=["base", "other"],
        train_idx=train_idx,
        val_idx=val_idx,
    )
    return disc


# ----------------------------------------------------------------------
# P18: memoised shrunk-domain mi_y_compare is bit-identical
# ----------------------------------------------------------------------


def test_p18_memo_is_bit_identical_to_unmemoised_recompute() -> None:
    """P18 memoises the shrunk-domain ``mi_y_compare`` on the ``valid_screen``
    mask hash. Running discovery WITH the shared memo must be BIT-IDENTICAL to a
    run whose memo is disabled (no shared dict): same specs, same ``mi_y`` /
    ``mi_gain`` per spec to the last bit.

    We disable the memo by stripping the ``_mi_y_compare_memo`` keys from the
    base contexts after they are built, forcing the recompute branch every time;
    the memoised run keeps them. The two must agree exactly.
    """
    cfg = _make_config()
    df = _shrink_dataset()

    disc_memo = _run(df, cfg)

    # Disabled-memo run: monkeypatch the context builder so contexts carry no
    # memo dict, forcing eval_one_transform down the recompute branch each call.
    import mlframe.training.composite.discovery._fit as fit_mod

    orig_lock_attr = "_mi_y_compare_memo"
    disc_nomemo = _StripMemoRunner(df, _make_config(), fit_mod, orig_lock_attr)

    memo_by_name = {s.name: s for s in disc_memo.specs_}
    nomemo_by_name = {s.name: s for s in disc_nomemo.specs_}
    assert set(memo_by_name) == set(nomemo_by_name), "memo vs no-memo produced different spec sets"
    assert memo_by_name, "discovery produced no specs; test is vacuous"
    for name, s_memo in memo_by_name.items():
        s_no = nomemo_by_name[name]
        assert s_memo.mi_y == s_no.mi_y, f"{name}: mi_y diverged {s_memo.mi_y!r} != {s_no.mi_y!r}"
        assert s_memo.mi_gain == s_no.mi_gain, f"{name}: mi_gain diverged {s_memo.mi_gain!r} != {s_no.mi_gain!r}"


class _StripMemoRunner:
    """Runs discovery with the shrunk-domain memo disabled by replacing the
    base-context memo dict with ``None`` so ``eval_one_transform`` always takes
    the recompute branch (used to prove the memo is bit-identical to recompute).
    """

    def __new__(cls, df, config, fit_mod, memo_key):
        import mlframe.training.composite.discovery._eval as eval_mod

        orig_eval = eval_mod.eval_one_transform

        def _patched(self, base, transform_name, transform, *, base_contexts, **kw):
            ctx = base_contexts.get(base)
            if ctx is not None and ctx.get(memo_key) is not None:
                ctx = dict(ctx)
                ctx[memo_key] = None
                ctx["_mi_y_compare_memo_lock"] = None
                base_contexts = dict(base_contexts)
                base_contexts[base] = ctx
            return orig_eval(self, base, transform_name, transform, base_contexts=base_contexts, **kw)

        eval_mod.eval_one_transform = _patched
        fit_mod.eval_one_transform = _patched
        try:
            return _run(df, config)
        finally:
            eval_mod.eval_one_transform = orig_eval
            fit_mod.eval_one_transform = orig_eval


def test_p18_memo_present_on_base_contexts_and_used() -> None:
    """The memo dict is populated when a base's screen shrinks: after a discovery
    fit over a domain-shrinking dataset, at least one base context's memo holds a
    cached scalar (proving the recompute path ran and stored its result)."""
    from mlframe.training.composite.discovery._eval_stats import (
        bootstrap_gain_p_value,  # noqa: F401  (import-resolution smoke)
    )

    # Drive eval_one_transform once directly to assert the memo is hit twice
    # bit-identically for two transforms sharing a base + valid_screen mask.
    from mlframe.training.composite.discovery import _eval as eval_mod
    from mlframe.training.composite.discovery.screening import (
        _prebin_feature_columns,
    )
    from mlframe.training.composite.transforms import get_transform
    import threading

    rng = np.random.default_rng(7)
    n = 3000
    # ``base`` is the auto-base column (all positive); ``y`` is the TARGET. For
    # the train rows both must be positive so logratio's residual is non-trivial;
    # the SCREEN target carries a few negative rows so its domain mask shrinks --
    # that shrink is exactly the per-base ``mi_y_compare`` recompute P18 memoises.
    base = rng.lognormal(mean=0.8, sigma=0.4, size=n).astype(np.float32)
    y_train = (2.0 * base + rng.lognormal(mean=0.5, sigma=0.3, size=n)).astype(np.float32)
    y_screen = y_train.copy()
    y_screen[: n // 25] = -np.abs(rng.normal(size=n // 25)).astype(np.float32)  # neg rows.
    x_remaining = rng.normal(size=(n, 3)).astype(np.float64)
    x_prebinned = _prebin_feature_columns(x_remaining, nbins=16)

    class _Cfg:
        mi_n_neighbors = 3
        random_state = 42
        mi_estimator = "bin"
        min_valid_domain_frac = 0.5
        mi_gain_bootstrap_n = 0
        mi_gain_bootstrap_random_state = 12345

    class _Disc:
        config = _Cfg()

        def _reject(self, *a, **k):
            return {"spec": None, "kept": False, "reason": k.get("reason", "rej")}

    memo: dict = {}
    ctx = dict(
        base_train=base,
        base_screen=base,
        x_remaining_matrix=x_remaining,
        _x_prebinned=x_prebinned,
        mi_y_for_base=0.0,
        _mi_kwargs=dict(nbins=16, aggregation="mean"),
        _mi_y_compare_memo=memo,
        _mi_y_compare_memo_lock=threading.Lock(),
    )
    base_contexts = {"base": ctx}
    disc = _Disc()

    # ``logratio`` requires y, base > 0 -> the negative screen-target rows shrink
    # ``valid_screen`` and trigger the memoised baseline recompute.
    tr = get_transform("logratio")
    out1 = eval_mod.eval_one_transform(
        disc,
        "base",
        "logratio",
        tr,
        base_contexts=base_contexts,
        y_train=y_train,
        y_screen=y_screen,
        target_col="y",
    )
    assert memo, "shrunk-domain mi_y_compare memo was never populated"
    cached_keys = set(memo)
    cached_vals = dict(memo)

    # A second transform on the SAME base + same valid_screen mask must HIT the
    # memo (no new key) and read back the identical cached scalar.
    tr2 = get_transform("logratio")
    out2 = eval_mod.eval_one_transform(
        disc,
        "base",
        "logratio",
        tr2,
        base_contexts=base_contexts,
        y_train=y_train,
        y_screen=y_screen,
        target_col="y",
    )
    assert set(memo) == cached_keys, "second call added a memo key (not a hit)"
    for k, v in cached_vals.items():
        assert memo[k] == v, "memo value changed between bit-identical calls"
    # Both calls produced the same spec mi_y (the memoised baseline).
    assert out1 and out2 and out1[0].get("spec") is not None
    assert out1[0]["spec"].mi_y == out2[0]["spec"].mi_y


# ----------------------------------------------------------------------
# Integration: FDR wiring through CompositeTargetDiscovery.fit
# ----------------------------------------------------------------------


def test_fdr_control_default_on_is_noop_without_bootstrap() -> None:
    """With the shipped default (bootstrap disabled) ``mi_gain_fdr_control=True``
    must NOT change the discovered spec set vs explicitly off -- the no-op
    contract that keeps recovery on default configs unchanged."""
    df = _shrink_dataset()
    disc_on = _run(df, _make_config(mi_gain_fdr_control=True, mi_gain_bootstrap_n=0))
    disc_off = _run(df, _make_config(mi_gain_fdr_control=False, mi_gain_bootstrap_n=0))
    names_on = sorted(s.name for s in disc_on.specs_)
    names_off = sorted(s.name for s in disc_off.specs_)
    assert names_on == names_off, "FDR control changed the spec set with the bootstrap disabled; it must be a strict no-op there"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
