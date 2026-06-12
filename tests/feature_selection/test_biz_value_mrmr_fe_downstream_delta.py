"""BIZ-VALUE: end-to-end proof that the engineered composites MRMR recovers
IMPROVE downstream MODEL ACCURACY versus a raw-features-only baseline.

This is the value proof of the whole FE pipeline. The earlier biz_value tests
(``test_biz_value_mrmr_fe_canonical.py``) only assert that the RIGHT engineered
columns are *recovered* (names cover the signal pairs) and that ``transform``
*replays* them leak-safely. They never check that a downstream model trained on
the FE-augmented selection actually PREDICTS better than one trained on the raw
columns alone. That is what this file pins, with a clean leak-safe train/test
split (FE fit on train, replayed on test) and a held-out metric delta.

THE CLAIM (both directions, the load-bearing contract):

  (1) FE HELPS when the signal is genuinely NON-ADDITIVE. A composite such as
      ``mul(log(c),sin(d))`` / ``div(sqr(a),abs(b))`` makes a signal that a
      LINEAR model cannot represent from the raw inputs linearly usable, so a
      Ridge / LogisticRegression on the FE-augmented selection beats the same
      model on raw-only by a MEANINGFUL positive margin. (A gradient-boosted
      tree already models interactions natively, so its delta is ~0 -- the
      composites are not redundant bloat, their value is specifically LINEAR
      usability. We therefore pin the delta on the LINEAR downstream model.)

  (2) FE is HARMLESS when the signal is already raw-LINEAR. On a target that is a
      plain linear combination of the raw inputs the recovered composites are
      additive re-expressions that add no linear information, so the linear
      downstream delta is ~0 (non-negative within tolerance) -- FE must not crowd
      out the raws and HURT.

THE BASELINE. ``raw-only`` disables every default-ON FE generator (the pair /
unary-binary composite step AND the univariate basis / fourier / hinge / wavelet
/ conditional-dispersion / orth paths) so the selection is RAW columns only,
selected by the SAME MRMR relevance/redundancy machinery. ``FE-on`` enables only
the single-step pair composite FE (``fe_max_steps=1`` -> always-replayable
recipes) so the comparison isolates the engineered composites this coverage item
targets, not the univariate-basis features.

NEVER xfail / weaken. If a delta regresses, the FE pipeline lost downstream
value -- fix prod, not the assertion.
"""
from __future__ import annotations

import sys
import os

import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, roc_auc_score

from mlframe.feature_selection.filters.mrmr import MRMR

# Reusable synthetic generators live beside this file.
sys.path.insert(0, os.path.dirname(__file__))
from _biz_val_synth import (  # noqa: E402
    make_polynomial_target, make_heavy_tail_skewed, make_signal_plus_noise, as_df,
)

# Every default-ON FE generator OFF -> MRMR selects RAW columns only.
_RAW_ONLY = dict(
    fe_max_steps=0,
    fe_univariate_basis_enable=False, fe_univariate_fourier_enable=False,
    fe_hinge_enable=False, fe_conditional_dispersion_enable=False,
    fe_wavelet_enable=False, fe_hybrid_orth_pair_enable=False,
    fe_auto_escalation_enable=False, fe_pair_prewarp_enable=False,
    fe_rung_schedule_enable=False, fe_stability_vote_enable=False,
    cluster_aggregate_enable=False, dcd_enable=False,
)
# Single-step pair composites only (always replayable); univariate paths OFF so
# the delta is attributable to the engineered COMPOSITES, not basis features.
_FE_ON = dict(
    fe_max_steps=1,
    fe_univariate_basis_enable=False, fe_univariate_fourier_enable=False,
    fe_hinge_enable=False, fe_conditional_dispersion_enable=False,
    fe_wavelet_enable=False, fe_hybrid_orth_pair_enable=False,
    fe_auto_escalation_enable=False, fe_pair_prewarp_enable=False,
    fe_rung_schedule_enable=False, fe_stability_vote_enable=False,
    cluster_aggregate_enable=False, dcd_enable=False,
)


def _golden(seed: int = 42, n: int = 6000, scale: float = 3.0):
    """The canonical golden composite ``y = a**2/b + f/5 + scale*log(c)*sin(d)``
    (the F1 golden from ``test_biz_value_mrmr_fe_canonical``). Genuinely
    non-additive: a LINEAR model cannot represent ``a**2/b`` or ``log(c)*sin(d)``
    from the raw inputs, so the recovered composites are what make it usable."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(1.0, 5.0, n); b = rng.uniform(1.0, 5.0, n)
    c = rng.uniform(1.0, 5.0, n); d = rng.uniform(0.0, 2.0 * np.pi, n)
    e = rng.normal(0.0, 1.0, n); f = rng.normal(0.0, 1.0, n)
    y = a ** 2 / b + f / 5.0 + scale * np.log(c) * np.sin(d)
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e}), pd.Series(y, name="y")


def _split(df, y, frac: float = 0.7):
    """Leak-safe shuffle split: FE is fit on train, replayed on test."""
    n = len(df)
    idx = np.arange(n)
    np.random.default_rng(0).shuffle(idx)
    k = int(n * frac)
    tr, te = idx[:k], idx[k:]
    return (df.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True),
            df.iloc[te].reset_index(drop=True), y.iloc[te].reset_index(drop=True))


def _linear_model(classification: bool):
    if classification:
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def _select_transform(kwargs, df_tr, y_tr, df_te):
    fs = MRMR(verbose=0, random_seed=42, **kwargs)
    fs.fit(df_tr, y_tr)
    Xtr = np.nan_to_num(np.asarray(fs.transform(df_tr), float), nan=0.0, posinf=0.0, neginf=0.0)
    Xte = np.nan_to_num(np.asarray(fs.transform(df_te), float), nan=0.0, posinf=0.0, neginf=0.0)
    return Xtr, Xte, list(fs.get_feature_names_out())


def _linear_downstream_delta(df, y, classification: bool):
    """Return ``(delta, s_fe, s_raw, names_fe, names_raw)`` for the LINEAR
    downstream model: held-out metric on the FE-augmented selection minus the
    same metric on the raw-only selection. Metric is AUC (classification) or
    R2 (regression)."""
    df_tr, y_tr, df_te, y_te = _split(df, y)
    Xtr_fe, Xte_fe, n_fe = _select_transform(_FE_ON, df_tr, y_tr, df_te)
    Xtr_raw, Xte_raw, n_raw = _select_transform(_RAW_ONLY, df_tr, y_tr, df_te)
    assert Xtr_fe.shape[1] > 0, f"FE-on selection empty -> 0 features downstream; names={n_fe}"
    assert Xtr_raw.shape[1] > 0, f"raw-only selection empty -> 0 features downstream; names={n_raw}"

    def _fit_score(Xtr, Xte):
        m = _linear_model(classification)
        m.fit(Xtr, y_tr.values)
        if classification:
            return float(roc_auc_score(y_te.values, m.predict_proba(Xte)[:, 1]))
        return float(r2_score(y_te.values, m.predict(Xte)))

    s_fe = _fit_score(Xtr_fe, Xte_fe)
    s_raw = _fit_score(Xtr_raw, Xte_raw)
    return s_fe - s_raw, s_fe, s_raw, n_fe, n_raw


def _has_engineered(names):
    return [n for n in names if ("(" in n) or ("__" in n)]


# ---------------------------------------------------------------------------
# Direction 1: FE HELPS on genuinely non-additive signal.
# ---------------------------------------------------------------------------
@pytest.mark.timeout(300)
def test_golden_composite_fe_improves_linear_downstream_r2():
    """GOLDEN ``y=a**2/b + log(c)*sin(d)``: the recovered composites lift held-out
    Ridge R2 by a MEANINGFUL margin over raw-only (measured ~ +0.12). A LINEAR
    model cannot represent the non-additive terms from raws, so the composites
    are the load-bearing carriers of the signal."""
    df, y = _golden(n=6000)
    delta, s_fe, s_raw, n_fe, n_raw = _linear_downstream_delta(df, y, classification=False)
    assert _has_engineered(n_fe), f"FE-on recovered NO engineered composite: {n_fe}"
    assert delta > 0.05, (
        f"engineered composites did NOT meaningfully improve linear downstream R2: "
        f"FE-on={s_fe:.4f} raw-only={s_raw:.4f} delta={delta:+.4f} (want > +0.05); "
        f"FE names={n_fe}, raw names={n_raw}. Either the recovered feature is not "
        f"usable downstream (transform/replay bug) or the composite was not selected."
    )


@pytest.mark.timeout(300)
def test_polynomial_interaction_fe_improves_linear_downstream_auc():
    """NON-ADDITIVE ``y=sign(0.7*x0^2 - 0.5*x1^2 + 0.3*x0*x1)``: the recovered
    (x0,x1) composite lifts held-out LogReg AUC over raw-only (measured ~ +0.13).
    The pair carries quadratic + cross-term structure a linear model cannot reach
    from the raws."""
    X, yy, _sig = make_polynomial_target(n=4000, degree=2)
    df, y = as_df(X, yy)
    delta, s_fe, s_raw, n_fe, n_raw = _linear_downstream_delta(df, y, classification=True)
    assert _has_engineered(n_fe), f"FE-on recovered NO engineered composite: {n_fe}"
    assert delta > 0.04, (
        f"polynomial-interaction composite did NOT improve linear downstream AUC: "
        f"FE-on={s_fe:.4f} raw-only={s_raw:.4f} delta={delta:+.4f} (want > +0.04); "
        f"FE names={n_fe}, raw names={n_raw}."
    )


@pytest.mark.timeout(300)
def test_heavy_tail_logmultiplicative_fe_improves_linear_downstream_auc():
    """HEAVY-TAIL log-multiplicative ``y=sign(log(base)+log(other))`` on lognormal
    inputs: the recovered ratio/log composite lifts held-out LogReg AUC over the
    raw-only baseline (measured ~ +0.04). A linear model on the raw heavy-tailed
    inputs under-fits the multiplicative relationship; the composite recovers it."""
    X, yy, _sig = make_heavy_tail_skewed(n=4000)
    df, y = as_df(X, yy)
    delta, s_fe, s_raw, n_fe, n_raw = _linear_downstream_delta(df, y, classification=True)
    assert _has_engineered(n_fe), f"FE-on recovered NO engineered composite: {n_fe}"
    assert delta > 0.015, (
        f"heavy-tail composite did NOT improve linear downstream AUC: "
        f"FE-on={s_fe:.4f} raw-only={s_raw:.4f} delta={delta:+.4f} (want > +0.015); "
        f"FE names={n_fe}, raw names={n_raw}."
    )


# ---------------------------------------------------------------------------
# Direction 2: FE is HARMLESS on raw-linear signal (must not HURT).
# ---------------------------------------------------------------------------
@pytest.mark.timeout(300)
def test_raw_linear_signal_fe_does_not_hurt_linear_downstream_auc():
    """RAW-LINEAR ``y=sign(x0+x1+x2 + noise)``: the signal is already linearly
    usable from the raws, so the recovered (additive) composites add no linear
    information and the held-out LogReg AUC delta is ~0 (non-negative within
    tolerance). FE must NOT crowd out the raws and HURT here."""
    X, yy, _sig = make_signal_plus_noise(n=4000, p_signal=3, p_noise=8, linear_only=True)
    df, y = as_df(X, yy)
    delta, s_fe, s_raw, n_fe, n_raw = _linear_downstream_delta(df, y, classification=True)
    # ~0 delta: FE neither meaningfully helps nor hurts on an already-linear target.
    assert delta > -0.01, (
        f"FE HURT the linear downstream on a raw-linear target (engineered noise "
        f"crowding out raws): FE-on={s_fe:.4f} raw-only={s_raw:.4f} delta={delta:+.4f} "
        f"(want >= ~0); FE names={n_fe}, raw names={n_raw}."
    )
