"""biz_val test for ``BorutaShap``: on a synthetic 10-feature dataset where 2 features are genuinely informative (``y`` derived from ``0.7 * x_inf + 0.3 * x_inf2 + noise``) and the other 8 are pure standard-normal noise, the selector must recover BOTH informative features AND admit at most a handful of noise features.

Locks in the SHAP-driven Boruta core invariant: rejecting noise-only columns while keeping signal-bearing ones. A regression that either rejects the informative pair (overly-strict pvalue / percentile drift) OR accepts all 10 (broken shadow-comparison) would fail this test.

Per CLAUDE.md (biz_value floor = measured-value minus 5-15% headroom):
  - measured dev run (seed=0, n=600, n_trials=30): informative_kept=2/2, noise_kept=0/8
  - asserted floors: informative_kept >= 2, noise_kept <= 5 (well below the 8-noise-everything regression watermark while leaving room for noise correlation luck at non-fixed seeds)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_biz_val_boruta_shap_filters_noise_keeps_informative():
    pytest.importorskip("shap")
    from mlframe.feature_selection.boruta_shap import BorutaShap

    rng = np.random.default_rng(0)
    n = 600
    x_inf = rng.normal(size=n)
    x_inf2 = rng.normal(size=n)
    noise_cols = rng.normal(size=(n, 8))

    # Linear combo with small Gaussian residual; threshold on its median to balance the binary target ~50/50.
    linear = 0.7 * x_inf + 0.3 * x_inf2 + 0.10 * rng.normal(size=n)
    y = (linear > np.median(linear)).astype(np.int64)

    cols = ["inf1", "inf2"] + [f"noise_{i}" for i in range(8)]
    X = pd.DataFrame(np.column_stack([x_inf, x_inf2, noise_cols]), columns=cols)

    sel = BorutaShap(
        importance_measure="gini",
        classification=True,
        n_trials=30,
        random_state=0,
        verbose=False,
        optimistic=True,
    )
    sel.fit(X, pd.Series(y))

    selected = set(sel.selected_features_)
    informative_kept = selected & {"inf1", "inf2"}
    noise_kept = [c for c in selected if c.startswith("noise_")]

    # Recovery: BOTH informative features must survive. Measured 2/2; floor 2/2 (no headroom -- losing one is a real regression we want to detect).
    assert informative_kept == {"inf1", "inf2"}, (
        f"BorutaShap must retain both informative features; got informative_kept={informative_kept}, "
        f"full selected={sorted(selected)}"
    )

    # Discrimination: at most 5 of 8 noise columns admitted. Measured 0/8; floor 5 keeps room for seed-to-seed correlation luck while still catching the "all 10 admitted" failure mode.
    assert len(noise_kept) <= 5, (
        f"BorutaShap admitted too many noise columns ({len(noise_kept)} of 8); noise_kept={noise_kept}"
    )

    # Sanity: support_ shape and dtype match the sklearn-style contract every other selector in the suite exposes.
    assert sel.support_.shape == (10,)
    assert sel.support_.dtype == bool
    # support_ must be consistent with selected_features_.
    support_named = {c for c, m in zip(cols, sel.support_) if m}
    assert support_named == selected, f"support_ disagrees with selected_features_: {support_named} vs {selected}"


def test_biz_val_boruta_early_stop_tentative_saves_wall_at_accepted_equivalence():
    """biz_value for the opt-in ``early_stop_tentative`` margin-gated trial-stop: on a dataset with a residual
    tentative tail it must (a) reclaim a large fraction of trials/wall vs the full ``n_trials`` cap and (b) keep the
    ACCEPTED (confirmed) set IDENTICAL -- the load-bearing decision. Floors set below the committed bench
    (bench_boruta_early_stop_tentative.py: mean 47.9% wall saved, accepted-Jaccard mean 0.982): require >= 30% trials
    saved AND identical accepted set. This pins the measured win even though the default stays OFF (the bench's
    per-scenario accepted-Jaccard dips to 0.944 on noise-heavy beds -> kept opt-in, REJECTED-not-DELETED)."""
    pytest.importorskip("shap")
    from sklearn.ensemble import RandomForestClassifier
    from mlframe.feature_selection.boruta_shap import BorutaShap

    rng = np.random.default_rng(0)
    n = 3000
    z = rng.standard_normal((n, 8))
    logit = (1.4 * z[:, 0] + 1.1 * z[:, 1] - 1.0 * z[:, 2] + 0.9 * z[:, 3]
             + 1.6 * z[:, 4] * z[:, 5] + 1.3 * (z[:, 6] ** 2 - 1.0) + 0.8 * z[:, 7]) / 1.6
    y = pd.Series((rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int))
    cols = {f"inf_{i}": z[:, i] for i in range(8)}
    for parent in (0, 4, 6):
        for j in range(4):
            cols[f"red_{parent}_{j}"] = z[:, parent] + 0.30 * rng.standard_normal(n)
    for i in range(28):
        cols[f"noise_{i}"] = rng.standard_normal(n)
    X = pd.DataFrame(cols)

    def _mk(es):
        return BorutaShap(
            model=RandomForestClassifier(n_estimators=50, n_jobs=4, random_state=0),
            importance_measure="gini", classification=True, n_trials=70, percentile=95,
            pvalue=0.05, verbose=False, random_state=0,
            early_stop_tentative=es, early_stop_patience=20, early_stop_margin=0.15,
        )

    off = _mk(False); off.fit(X, y)
    on = _mk(True); on.fit(X, y)

    assert off.n_trials_run_ == off.n_trials, "OFF run must burn the cap (else no tail to reclaim)"
    trials_saved = (off.n_trials_run_ - on.n_trials_run_) / off.n_trials_run_
    assert trials_saved >= 0.30, f"early-stop reclaimed only {trials_saved:.1%} of trials (floor 30%)"
    assert set(on.accepted) == set(off.accepted), (
        f"early-stop accepted set diverged: on={sorted(on.accepted)} off={sorted(off.accepted)}"
    )


def test_biz_val_boruta_shadow_min_pad_reduces_noise_false_accept_on_narrow_frame():
    """biz_value for the canonical >=5 shadow pad (shadow_min_pad, B7): on a NARROW 2-feature frame (1 informative +
    1 pure-noise column) the thin one-shadow-per-column null (pad=0) lets the noise column clear the gate more often
    than the padded null (pad=5). Floor below the committed bench (bench_shadow_min_pad_narrow_frames.py: shape (2,1)
    noise false-accept 0.1667 -> 0.0833 over 12 seeds): require the padded default to admit the noise column on
    STRICTLY FEWER seeds than pad=0, with the informative column retained by BOTH. Catches a regression that disables
    the pad (default flip back to 0) or breaks the recycled-column null."""
    import numpy as np
    import pandas as pd
    from mlframe.feature_selection.boruta_shap import BorutaShap

    seeds = range(8)
    noise_fa_pad, noise_fa_nopad, inf_kept_pad = 0, 0, 0
    for seed in seeds:
        rng = np.random.default_rng(seed)
        n = 400
        x_inf = rng.standard_normal(n)
        x_noise = rng.standard_normal(n)
        y = (x_inf + 0.25 * rng.standard_normal(n) > 0).astype(int)
        X = pd.DataFrame({"inf": x_inf, "noise": x_noise})
        for pad, is_pad in ((0, False), (5, True)):
            sel = BorutaShap(importance_measure="gini", classification=True, n_trials=25,
                             random_state=seed, verbose=False, shadow_min_pad=pad)
            sel.fit(X, pd.Series(y))
            acc = set(sel.accepted)
            if is_pad:
                inf_kept_pad += "inf" in acc
                noise_fa_pad += "noise" in acc
            else:
                noise_fa_nopad += "noise" in acc

    assert inf_kept_pad == len(list(seeds)), f"pad5 must retain the informative column on all seeds; got {inf_kept_pad}"  # noqa: E501
    assert noise_fa_pad < noise_fa_nopad, (
        f"shadow_min_pad=5 must admit the noise column on fewer seeds than pad=0 (the thin-null false accept); "
        f"got pad5={noise_fa_pad} vs nopad={noise_fa_nopad}"
    )
