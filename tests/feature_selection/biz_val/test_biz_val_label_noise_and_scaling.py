"""biz_val proofs: label-noise graceful degradation + RFECV adversarial feature scaling.

Two showcase contracts the rest of the suite does not pin (audit findings
bizvalue_value_proofs-10 and gaps_selection_masking-10):

A. Label-noise graceful degradation (``bizvalue_value_proofs-10``). MRMR's
   permutation-confirmed gating should degrade MORE gracefully under label
   flips than the naive ``SelectKBest(mutual_info_classif)`` filter a user
   reaches for first: MRMR holds FULL signal recall as the flip rate climbs
   to 30% while plain top-K MI loses half the signal, and MRMR's false-
   positive (noise) count stays bounded rather than blowing up.

B. RFECV adversarial feature scaling (``gaps_selection_masking-10``). A
   strongly-informative feature multiplied by 1e9 gets a |coef| ~ 1e-9 and
   would be eliminated FIRST under a coef-importance ranking unless the
   train-std rescale (``coef_scale_source='train'``, the default) restores
   its true importance. The negative control ``coef_scale_source='none'``
   pins that the rescale is what keeps it -- with the rescale OFF the mis-
   scaled feature IS dropped, so the default is doing real work.

All floors are calibrated from a measured development run, then set with
headroom per CLAUDE.md "pin a quantitative threshold 5-15% below measured".
ASCII-only; fixed seeds; majority-of-seeds for any cross-seed win.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import Ridge

from mlframe.feature_selection.filters.mrmr import MRMR
from mlframe.feature_selection.wrappers import RFECV
from tests.feature_selection._biz_val_synth import (
    as_df,
    make_signal_plus_noise,
    signal_recovery_count,
)
from tests.feature_selection.conftest import is_fast_mode

warnings.filterwarnings("ignore")

_XREF = re.compile(r"x(\d+)")

# Calibrated on a measured dev run (n=2000, p_signal=4, p_noise=16, 5 seeds):
#   MRMR signal recall == 4 at EVERY flip rate {0.0, 0.1, 0.2, 0.3} and EVERY seed.
#   SelectKBest-MI recall collapses 4 -> 4 -> {3..4} -> {1..2} as the rate climbs.
#   MRMR noise count: 0 at rate 0.0 (all seeds); {0,1,2} at rate 0.3 (median 2, max 2).
# Rate 0.1 is dropped from the swept set (its cell is identical to 0.0 for both methods: rec=4, noise=0);
# keeping {0.0, 0.2, 0.3} captures clean baseline, degradation onset, and hardest case while staying under
# the per-test 60s timeout (9 MRMR fits, not 12).
_FLIP_RATES = (0.0, 0.2, 0.3)
_LN_SEEDS = (0, 1, 2)
_K = 4  # match SelectKBest's K to the true signal width


def _flip_labels(y: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Flip a fixed ``rate`` fraction of binary labels (deterministic given ``rng``)."""
    y = y.copy()
    n_flip = round(rate * len(y))
    if n_flip:
        idx = rng.choice(len(y), size=n_flip, replace=False)
        y[idx] = 1 - y[idx]
    return y


def _make_mrmr() -> MRMR:
    """Default confirmation gates; ``full_npermutations=10`` for a stable permutation null on n=2000."""
    return MRMR(
        min_relevance_gain=0.0,
        cv=3,
        run_additional_rfecv_minutes=False,
        full_npermutations=10,
        random_seed=0,
        min_features_fallback=1,
        verbose=False,
    )


def _mrmr_noise_count(selected_names: list[str], signal: list[int]) -> int:
    """Count selected feature names (raw or engineered) that reference ONLY noise columns."""
    sig = set(int(i) for i in signal)
    noise = 0
    for nm in selected_names:
        refs = set(int(m) for m in _XREF.findall(nm))
        if refs and not (refs & sig):
            noise += 1
    return noise


def _run_label_noise_cell(rate: float, seed: int):
    """Fit MRMR + SelectKBest-MI on a single (rate, seed) cell. Returns
    ``(recall_mrmr, noise_mrmr, recall_mi, noise_mi)``."""
    X, y, signal = make_signal_plus_noise(n=2000, p_signal=_K, p_noise=16, seed=seed)
    # Fixed rng per rate (shared across seeds) so the flipped-row SET is rate-determined, per the recipe.
    rng = np.random.default_rng(1000 + round(rate * 100))
    y_noisy = _flip_labels(y, rate, rng)
    df, ys = as_df(X, y_noisy)

    mrmr = _make_mrmr()
    mrmr.fit(df, ys)
    names = list(mrmr.get_feature_names_out())
    recall_mrmr = signal_recovery_count(mrmr, signal)
    noise_mrmr = _mrmr_noise_count(names, signal)

    skb = SelectKBest(mutual_info_classif, k=_K).fit(df.values, ys.values)
    sel_mi = [i for i, keep in enumerate(skb.get_support()) if keep]
    recall_mi = len(set(sel_mi) & set(signal))
    noise_mi = len([i for i in sel_mi if i not in signal])
    return recall_mrmr, noise_mrmr, recall_mi, noise_mi


@pytest.mark.slow
def test_biz_val_label_noise_mrmr_dominates_selectkbest_across_flip_rate_sweep():
    """MRMR degrades more gracefully than top-K MI under label noise.

    Measured (5 seeds): MRMR recall == 4 at every rate; MI recall drops to
    1-2 at rate 0.3. We pin three measured wins:
      (i)   dominance-or-tie at EVERY rate (recall_mrmr >= recall_mi - 0.1),
            majority of the 3 seeds;
      (ii)  graceful floor at rate 0.3: recall_mrmr >= 0.5 (measured 4/4);
      (iii) FDR stays roughly flat: MRMR noise count at rate 0.3 stays
            bounded (<= rate-0.0 count + 3; measured worst rate-0.3 noise
            was 2 with rate-0.0 noise 0), and crucially MRMR keeps strictly
            MORE signal than MI does at rate 0.3 on a majority of seeds --
            the showcase claim plain MI cannot match.
    """
    rates = (0.0, 0.3) if is_fast_mode() else _FLIP_RATES
    seeds = (0,) if is_fast_mode() else _LN_SEEDS

    cells = {(rate, seed): _run_label_noise_cell(rate, seed) for rate in rates for seed in seeds}

    # (i) dominance-or-tie at every rate, majority of seeds.
    for rate in rates:
        wins = sum(1 for seed in seeds if cells[(rate, seed)][0] >= cells[(rate, seed)][2] - 0.1)
        assert wins > len(seeds) // 2, (
            f"rate={rate}: MRMR recall failed dominance-or-tie vs SelectKBest-MI on a majority of seeds "
            f"(wins={wins}/{len(seeds)}); per-seed (rec_mrmr,rec_mi)="
            f"{[(cells[(rate, s)][0], cells[(rate, s)][2]) for s in seeds]}"
        )

    # (ii) graceful floor at the hardest rate: MRMR keeps at least half the signal on every seed.
    hardest = max(rates)
    recalls_hard = [cells[(hardest, s)][0] for s in seeds]
    assert min(recalls_hard) >= 0.5 * _K, f"rate={hardest}: MRMR recall floor breached; per-seed recall={recalls_hard} (need >= {0.5 * _K})"

    # (iii-a) FDR stays roughly flat for MRMR: noise count at the hardest rate stays bounded
    # relative to the clean-label noise count. Measured worst was +2; bound at +3 for headroom.
    if 0.0 in rates:
        base_noise = max(cells[(0.0, s)][1] for s in seeds)
        hard_noise = max(cells[(hardest, s)][1] for s in seeds)
        assert hard_noise <= base_noise + 3, f"MRMR FDR not held flat: noise@{hardest} (max {hard_noise}) exceeds noise@0.0 (max {base_noise}) + 3"

    # (iii-b) the headline: at the hardest rate MRMR recovers strictly more signal than MI on a
    # majority of seeds (measured MRMR=4 vs MI in {1,2} -> 3/3). This is the graceful-degradation win.
    strictly_more = sum(1 for s in seeds if cells[(hardest, s)][0] > cells[(hardest, s)][2])
    assert strictly_more > len(seeds) // 2, (
        f"rate={hardest}: MRMR did not recover strictly more signal than SelectKBest-MI on a majority of seeds "
        f"(strictly_more={strictly_more}/{len(seeds)}); per-seed (rec_mrmr,rec_mi)="
        f"{[(cells[(hardest, s)][0], cells[(hardest, s)][2]) for s in seeds]}"
    )


def test_biz_val_label_noise_and_scaling_fast_representative():
    """Non-slow fast representative so ``MLFRAME_FAST=1`` (which skips the
    @slow sweep above and the @slow RFECV pair below) still exercises BOTH
    showcase paths end to end on a minimal config.

    One label-noise cell at the hardest rate (0.3, seed 0): MRMR must keep
    strictly more signal than SelectKBest-MI (measured MRMR=4 vs MI in
    {1,2}). One RFECV scaling pair: the train-std rescale keeps the 1e9-
    scaled x0 that the 'none' control drops.
    """
    recall_mrmr, _noise_mrmr, recall_mi, _noise_mi = _run_label_noise_cell(0.3, 0)
    assert recall_mrmr >= 0.5 * _K, f"MRMR recall floor breached at rate 0.3: {recall_mrmr}"
    assert recall_mrmr > recall_mi, f"MRMR should recover strictly more signal than SelectKBest-MI at rate 0.3 (rec_mrmr={recall_mrmr}, rec_mi={recall_mi})"

    df, ys = _make_scaling_data(seed=0)
    sup_train = list(_make_scaling_rfecv("train").fit(df, ys).get_feature_names_out())
    sup_none = list(_make_scaling_rfecv("none").fit(df, ys).get_feature_names_out())
    assert "x0" in sup_train, f"coef_scale_source='train' should keep x0; support={sup_train}"
    assert "x0" not in sup_none, f"coef_scale_source='none' should drop x0; support={sup_none}"


# --------------------------------------------------------------------------
# Part B: RFECV adversarial feature scaling (gaps_selection_masking-10)
# --------------------------------------------------------------------------


def _make_scaling_data(seed: int = 0):
    """n=400, p=6; x0 is the strongest informative; ``X[:,0] *= 1e9``.

    Regression target ``y = (raw_x0 + 0.5*x1 > 0)`` (float) so a Ridge
    estimator's coef on x0 scales as ``effect / 1e9`` (~1e-10) -- the LOWEST
    raw |coef| of any column -- while the train-std rescale (* std_x0 ~ 1e9)
    restores x0 to the HIGHEST importance. This is exactly the case
    ``coef_scale_source`` exists to fix.
    """
    rng = np.random.default_rng(seed)
    n, p = 400, 6
    X = rng.normal(size=(n, p))
    raw_x0 = X[:, 0].copy()
    y = (raw_x0 + 0.5 * X[:, 1] > 0).astype(np.float64)
    X[:, 0] *= 1e9
    cols = [f"x{i}" for i in range(p)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="y")


def _make_scaling_rfecv(coef_scale_source: str) -> RFECV:
    """Ridge + coef importance + a parsimonious cap so the elimination order
    actually decides whether the mis-scaled x0 survives. ``importance_getter='coef_'``
    is REQUIRED: 'auto' resolves to permutation importance on small data and
    never exercises the coef rescale branch."""
    return RFECV(
        estimator=Ridge(),
        cv=3,
        max_refits=6,
        random_state=0,
        leakage_corr_threshold=None,
        importance_getter="coef_",
        coef_scale_source=coef_scale_source,
        n_features_selection_rule="argmax",
        max_nfeatures=2,
        verbose=0,
    )


@pytest.mark.slow
def test_biz_val_rfecv_coef_scale_source_train_rescues_mis_scaled_informative():
    """Default ``coef_scale_source='train'`` keeps a 1e9-scaled informative
    feature that a raw-|coef| ranking would eliminate first.

    Measured: with the rescale ON, x0's importance |coef|*std_x0 ~ 0.37 is
    the LARGEST, so RFECV keeps x0 (support ~ {x0, x1}). The companion
    negative control below pins that the rescale is what does it.
    """
    df, ys = _make_scaling_data(seed=0)
    rfecv = _make_scaling_rfecv("train")
    rfecv.fit(df, ys)
    support = list(rfecv.get_feature_names_out())
    assert "x0" in support, f"coef_scale_source='train' (default) should rescue the 1e9-scaled informative x0, but support={support}"


@pytest.mark.slow
def test_biz_val_rfecv_coef_scale_source_none_drops_mis_scaled_informative_negative_control():
    """Negative control: with the rescale OFF (``coef_scale_source='none'``)
    the SAME data drops x0 -- proving the rescale is load-bearing, not a no-op.

    Per the regression-test rule, this verifies the (train) assertion would
    FAIL here: under 'none' x0 has the LOWEST raw |coef| (~1e-10) and is
    eliminated first, so the parsimonious support excludes it.
    """
    df, ys = _make_scaling_data(seed=0)
    rfecv = _make_scaling_rfecv("none")
    rfecv.fit(df, ys)
    support = list(rfecv.get_feature_names_out())
    assert "x0" not in support, (
        f"coef_scale_source='none' should leave the mis-scaled x0 eliminated (raw |coef| ~ 1e-10 ranks last), "
        f"but support={support}; if this fails the rescale no longer discriminates and the positive test is vacuous"
    )
    # The discriminating feature x1 (normal scale, real signal) must survive in its place.
    assert "x1" in support, f"coef_scale_source='none': the normal-scale informative x1 should survive when x0 is dropped, support={support}"
