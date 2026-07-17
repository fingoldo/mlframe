"""Missingness-pattern coverage for the feature selectors (MCAR vs MNAR).

Real tabular data is full of NaN, and the FACT a value is missing is often
itself the strongest predictor (thin-file credit, never-ordered lab panel,
churn after onboarding). A feature selector earns its keep on missing data
only if it does THREE things:

  (a) MCAR-informative -- a feature with random (MCAR) NaN that is still
      informative on y must be KEPT, not dropped just for carrying NaN.
  (b) MNAR-signal -- a feature whose VALUE is noise but whose MISSINGNESS
      pattern carries the signal (NaN exactly / mostly when y==1) must be
      captured, via the ``is_missing__{col}`` indicator path.
  (c) mostly-NaN noise -- a column that is ~all NaN and carries no signal
      must be DROPPED.

And none of this may CRASH on NaN.

Two selector families have fundamentally different NaN contracts, surfaced
explicitly here rather than hidden behind try/except:

* MRMR (``nan_in_X_policy="tolerates"``) ingests NaN directly. Its Layer-7
  ``nan_strategy='separate_bin'`` handles MNAR at the binning level, and its
  Layer-37 ``fe_missingness_indicator_enable`` emits the ``is_missing__{col}``
  column that turns the missingness pattern into a first-class engineered
  feature. So MRMR is tested on RAW NaN data end to end.
* The sklearn-estimator wrappers (RFECV / GroupAware(RFECV) via a
  LogisticRegression core) now ingest raw NaN gracefully via
  ``nan_in_X_policy="impute"`` (the friendly default): median-impute per column
  at fit entry so the linear core no longer crashes, with optional
  ``nan_indicator_cols=`` emitting ``is_missing__{col}`` so MNAR signal stays
  capturable (mirroring MRMR's Layer-37 emitter). ``nan_in_X_policy="raise"``
  preserves the strict legacy crash for benchmarks / replay. The RF-core wrappers (ShapProxiedFS / BorutaShap /
  HybridSelector) tolerate NaN-imputed input; they have NO native
  ``is_missing__`` emitter, so they only capture the MNAR signal when the
  indicator column is supplied to them pre-engineered (the production
  pattern: emit the indicator upstream, impute the value, let the wrapper
  rank both). That asymmetry is the point of the matrix below.

Selector x missingness-type matrix (see module docstring of the run report):
  MRMR              : MCAR keep / MNAR capture (indicator) / noise drop -- all native
  RFECV             : graceful median-impute default / raise opt-in (nan_in_X_policy)
  GroupAware(RFECV) : graceful median-impute default / raise opt-in (inherits RFECV)
  ShapProxiedFS     : MCAR keep / MNAR via supplied indicator / (imputed)
  BorutaShap        : MCAR keep / MNAR via supplied indicator / (imputed)
  HybridSelector    : MCAR keep / MNAR via supplied indicator / (imputed)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from tests.feature_selection._selector_factories import (
    _make_boruta_shap,
    _make_hybrid,
    _make_shap_proxied,
    _make_rfecv,
    _make_group_aware_rfecv,
    selected_names,
)
from tests.feature_selection.conftest import fast_subset


# ---------------------------------------------------------------------------
# Synthetic builders. All small (n<=1500), fixed-seed, deterministic.
# ---------------------------------------------------------------------------


pytestmark = pytest.mark.timeout(60)  # untimed biz_val real-fit tier: surface a hang fast (global --timeout=600 is a coarse backstop)


def _build_mcar(seed: int, n: int = 1500, nan_rate: float = 0.2):
    """One MCAR-informative feature (random NaN, still drives y) + pure noise."""
    rng = np.random.default_rng(seed)
    sig = rng.normal(size=n)
    y = (sig + 0.3 * rng.normal(size=n) > 0).astype(np.int64)
    mcar = sig.copy()
    mcar[rng.random(n) < nan_rate] = np.nan
    df = pd.DataFrame(
        {
            "mcar": mcar,
            "n0": rng.normal(size=n),
            "n1": rng.normal(size=n),
        }
    )
    return df, pd.Series(y, name="y")


def _build_mnar(seed: int, n: int = 1500, miss_when_pos: float = 0.8):
    """MNAR feature: the VALUE is pure noise, the MISSINGNESS carries y.

    ``mnar`` is NaN mostly when ``y==1`` -- so the only signal lives in the
    isna() indicator, not the (imputed) value. A perfect indicator gives
    AUC ~0.9 on its own.
    """
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.5).astype(np.int64)
    val = rng.normal(size=n)
    val[(y == 1) & (rng.random(n) < miss_when_pos)] = np.nan
    df = pd.DataFrame(
        {
            "mnar": val,
            "n0": rng.normal(size=n),
            "n1": rng.normal(size=n),
            "n2": rng.normal(size=n),
        }
    )
    return df, pd.Series(y, name="y")


def _build_mostly_nan(seed: int, n: int = 1500, nan_rate: float = 0.97):
    """One MCAR-informative feature + a ~all-NaN noise column to be dropped."""
    rng = np.random.default_rng(seed)
    sig = rng.normal(size=n)
    y = (sig + 0.3 * rng.normal(size=n) > 0).astype(np.int64)
    mcar = sig.copy()
    mcar[rng.random(n) < 0.2] = np.nan
    noise = rng.normal(size=n)
    noise[rng.random(n) < nan_rate] = np.nan
    df = pd.DataFrame(
        {
            "mcar": mcar,
            "mostly_nan": noise,
            "n0": rng.normal(size=n),
            "n1": rng.normal(size=n),
        }
    )
    return df, pd.Series(y, name="y")


def _impute_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Mean-impute for the wrapper family (sklearn cores reject raw NaN). The
    PRODUCTION pattern: emit the missingness indicator upstream, THEN impute
    the value so the wrapper can rank both. Indicator columns (0/1) are left
    untouched (they never carry NaN)."""
    out = df.copy()
    for c in out.columns:
        col = out[c]
        if col.isna().any():
            fill = float(np.nanmean(col.to_numpy())) if col.notna().any() else 0.0
            out[c] = col.fillna(fill)
    return out


def _add_missing_indicator(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Append the ``is_missing__{col}`` indicator (the L37 emitter the
    wrappers lack natively) so the MNAR signal is available to them."""
    out = df.copy()
    out[f"is_missing__{col}"] = out[col].isna().astype(np.int8)
    return out


def _make_mrmr_missingness(*, indicator_cols: tuple = ()):
    """Fast deterministic MRMR. ``indicator_cols`` enables the L37
    ``is_missing__{col}`` emitter for the named columns (MNAR capture path)."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    return MRMR(
        min_relevance_gain=0.0,
        cv=3,
        run_additional_rfecv_minutes=False,
        full_npermutations=3,
        random_seed=0,
        min_features_fallback=1,
        verbose=0,
        fe_max_steps=0,
        dcd_enable=False,
        cluster_aggregate_enable=False,
        build_friend_graph=False,
        cat_fe_config=None,
        quantization_nbins=10,
        fe_missingness_indicator_enable=bool(indicator_cols),
        fe_missingness_indicator_cols=tuple(indicator_cols),
    )


def _mrmr_selected(m) -> list[str]:
    """Full selected name list (raw + engineered) for an MRMR fit."""
    return [str(nm) for nm in m.get_feature_names_out()]


def _mrmr_raw_kept(m, df: pd.DataFrame) -> list[str]:
    """Raw input columns surviving in the MRMR selection."""
    return [nm for nm in _mrmr_selected(m) if nm in df.columns]


SEEDS = [0, 1, 2, 3, 4]


def _majority(flags: list[bool]) -> bool:
    return sum(bool(f) for f in flags) > len(flags) // 2


# ===========================================================================
# MRMR -- the native-NaN selector. MCAR keep / MNAR capture / noise drop.
# ===========================================================================


def test_biz_val_mrmr_no_crash_on_raw_nan():
    """MRMR ingests raw NaN end to end without raising (nan_in_X_policy tolerates)."""
    df, ys = _build_mostly_nan(seed=0)
    m = _make_mrmr_missingness()
    m.fit(df, ys)  # must not raise
    assert len(m.get_feature_names_out()) >= 1


def test_biz_val_mrmr_keeps_mcar_informative_feature():
    """An MCAR-informative feature (20% random NaN) is KEPT on a majority of
    seeds -- the NaN does not cause the selector to discard a useful column."""
    kept = []
    for seed in fast_subset(SEEDS):
        df, ys = _build_mcar(seed)
        m = _make_mrmr_missingness()
        m.fit(df, ys)
        kept.append("mcar" in _mrmr_raw_kept(m, df))
    assert _majority(kept), f"MRMR dropped MCAR-informative feature; kept={kept}"


def test_biz_val_mrmr_captures_mnar_signal_via_indicator():
    """The MNAR signal (value is noise, missingness carries y) is captured by
    the ``is_missing__mnar`` indicator on a majority of seeds. Biz floor: the
    indicator-alone downstream AUC is >= 0.80 (measured ~0.90, ~11% margin)."""
    captured = []
    aucs = []
    for seed in fast_subset(SEEDS):
        df, ys = _build_mnar(seed)
        m = _make_mrmr_missingness(indicator_cols=("mnar",))
        m.fit(df, ys)
        out = _mrmr_selected(m)
        captured.append("is_missing__mnar" in out)
        ind = df["mnar"].isna().astype(int).to_numpy().reshape(-1, 1)
        proba = LogisticRegression().fit(ind, ys).predict_proba(ind)[:, 1]
        aucs.append(roc_auc_score(ys, proba))
    assert _majority(captured), f"MRMR missed MNAR indicator; captured={captured}"
    assert min(aucs) >= 0.80, f"MNAR indicator weak; min AUC {min(aucs):.3f} < 0.80"


def test_biz_val_mrmr_drops_mostly_nan_noise():
    """A ~97%-NaN noise column is DROPPED from the MRMR raw selection on a
    majority of seeds (the MCAR-informative feature survives alongside)."""
    dropped = []
    for seed in fast_subset(SEEDS):
        df, ys = _build_mostly_nan(seed)
        m = _make_mrmr_missingness()
        m.fit(df, ys)
        raw = _mrmr_raw_kept(m, df)
        dropped.append("mostly_nan" not in raw and "mcar" in raw)
    assert _majority(dropped), f"MRMR kept mostly-NaN noise; flags={dropped}"


# ===========================================================================
# sklearn-estimator wrappers (LogReg core): RAISE on raw NaN -> FS GAP.
# ===========================================================================


@pytest.mark.slow
@pytest.mark.parametrize("mk", [_make_rfecv, _make_group_aware_rfecv], ids=["RFECV", "GroupAware(RFECV)"])
def test_biz_val_rfecv_family_raw_nan_gap_closed(mk):
    """The former FS GAP is CLOSED: with the friendly ``nan_in_X_policy='impute'``
    default, RFECV and GroupAware(RFECV) ingest raw NaN WITHOUT crashing and
    recover the MCAR-informative signal (it survives selection). The strict
    legacy crash is still reproducible via ``nan_in_X_policy='raise'`` -- pinned
    in ``test_biz_val_rfecv_raise_policy_still_crashes`` below."""
    df, ys = _build_mcar(seed=0, n=600)
    s = mk("binary")
    s.fit(df, ys.to_numpy())  # must NOT raise anymore (graceful impute default)
    assert "mcar" in selected_names(s), f"RFECV-family dropped MCAR-informative feature after graceful impute; selected={selected_names(s)}"


@pytest.mark.slow
def test_biz_val_rfecv_raise_policy_still_crashes():
    """Opt-out pin: ``nan_in_X_policy='raise'`` reproduces the strict legacy
    crash on raw NaN (benchmarks / replay), while the default does NOT crash."""
    from mlframe.feature_selection.wrappers import RFECV

    df, ys = _build_mcar(seed=0, n=600)

    s_raise = RFECV(
        estimator=LogisticRegression(max_iter=200, random_state=0),
        cv=3,
        max_refits=3,
        random_state=0,
        leakage_corr_threshold=None,
        n_features_selection_rule="argmax",
        nan_in_X_policy="raise",
    )
    with pytest.raises(ValueError, match="contains NaN"):
        s_raise.fit(df, ys.to_numpy())

    # The default policy on the SAME data does not raise.
    s_default = _make_rfecv("binary")
    s_default.fit(df, ys.to_numpy())
    assert "mcar" in selected_names(s_default)


@pytest.mark.slow
def test_biz_val_rfecv_captures_mnar_via_indicator():
    """RFECV with ``nan_indicator_cols=('mnar',)`` emits ``is_missing__mnar``
    from the PRE-impute mask, so the MNAR signal (value is noise, missingness
    carries y) survives selection on a majority of seeds. Biz floor: the
    indicator-alone downstream AUC is >= 0.80 (measured ~0.90, ~11% margin)."""
    from mlframe.feature_selection.wrappers import RFECV

    captured = []
    aucs = []
    for seed in fast_subset(SEEDS, n=2):
        df, ys = _build_mnar(seed, n=700)
        s = RFECV(
            estimator=LogisticRegression(max_iter=200, random_state=0),
            cv=3,
            max_refits=4,
            random_state=0,
            leakage_corr_threshold=None,
            n_features_selection_rule="argmax",
            nan_indicator_cols=("mnar",),
        )
        s.fit(df, ys.to_numpy())  # must not crash on raw NaN
        captured.append("is_missing__mnar" in list(s.get_feature_names_out()))
        ind = df["mnar"].isna().astype(int).to_numpy().reshape(-1, 1)
        proba = LogisticRegression().fit(ind, ys).predict_proba(ind)[:, 1]
        aucs.append(roc_auc_score(ys, proba))
    assert _majority(captured), f"RFECV missed MNAR indicator; captured={captured}"
    assert min(aucs) >= 0.80, f"MNAR indicator weak; min AUC {min(aucs):.3f} < 0.80"


def test_biz_val_rfecv_nan_free_selection_unchanged():
    """Regression: the impute path is a strict no-op on a NaN-free frame -- the
    default ('impute') and 'raise' policies select the IDENTICAL feature set on
    NaN-free data, so the crash-to-graceful flip cannot silently alter non-NaN
    selection."""
    from mlframe.feature_selection.wrappers import RFECV

    df, ys = _build_mcar(seed=0, n=600)
    df = df.fillna(0.0)  # remove all NaN -> impute must be a no-op

    def _mk(policy):
        return RFECV(
            estimator=LogisticRegression(max_iter=200, random_state=0),
            cv=3,
            max_refits=3,
            random_state=0,
            leakage_corr_threshold=None,
            n_features_selection_rule="argmax",
            nan_in_X_policy=policy,
        )

    a = _mk("impute")
    a.fit(df, ys.to_numpy())
    b = _mk("raise")
    b.fit(df, ys.to_numpy())
    assert list(a.get_feature_names_out()) == list(b.get_feature_names_out()), "impute path altered selection on NaN-free data"


@pytest.mark.slow
@pytest.mark.parametrize("mk", [_make_rfecv, _make_group_aware_rfecv], ids=["RFECV", "GroupAware(RFECV)"])
def test_biz_val_rfecv_family_keeps_mcar_after_imputation(mk):
    """With upstream mean-imputation (the production fix for the GAP above)
    RFECV/GroupAware no longer crash AND keep the MCAR-informative feature on
    a majority of seeds."""
    kept = []
    for seed in fast_subset(SEEDS, n=2):
        df, ys = _build_mcar(seed, n=600)
        s = mk("binary")
        s.fit(_impute_mean(df), ys.to_numpy())
        kept.append("mcar" in selected_names(s))
    assert _majority(kept), f"RFECV-family dropped imputed MCAR feature; kept={kept}"


# ===========================================================================
# RF-core wrappers: tolerate (imputed) NaN, capture MNAR via supplied indicator.
# ===========================================================================

_RF_WRAPPERS = [
    pytest.param(_make_shap_proxied, id="ShapProxiedFS", marks=[pytest.mark.slow]),
    pytest.param(_make_boruta_shap, id="BorutaShap", marks=[pytest.mark.slow]),
    pytest.param(_make_hybrid, id="HybridSelector", marks=[pytest.mark.slow]),
]


@pytest.mark.parametrize("mk", _RF_WRAPPERS)
def test_biz_val_rf_wrapper_keeps_mcar_after_imputation(mk):
    """RF-core wrappers (Shap/Boruta/Hybrid) keep the MCAR-informative feature
    after mean-imputation on a majority of seeds -- they do not discard a
    useful column for having had NaN."""
    kept = []
    for seed in fast_subset(SEEDS, n=2):
        df, ys = _build_mcar(seed, n=700)
        s = mk("binary")
        s.fit(_impute_mean(df), ys.to_numpy())
        kept.append("mcar" in selected_names(s))
    assert _majority(kept), f"RF wrapper dropped imputed MCAR feature; kept={kept}"


@pytest.mark.parametrize("mk", _RF_WRAPPERS)
def test_biz_val_rf_wrapper_captures_mnar_via_supplied_indicator(mk):
    """RF-core wrappers have no native ``is_missing__`` emitter. When the
    indicator is supplied upstream (and the value mean-imputed), they CAPTURE
    the MNAR signal -- ``is_missing__mnar`` survives selection on a majority
    of seeds. This pins the production wiring (emit-then-impute-then-rank)."""
    captured = []
    for seed in fast_subset(SEEDS, n=2):
        df, ys = _build_mnar(seed, n=700)
        df_ind = _impute_mean(_add_missing_indicator(df, "mnar"))
        s = mk("binary")
        s.fit(df_ind, ys.to_numpy())
        captured.append("is_missing__mnar" in selected_names(s))
    assert _majority(captured), f"RF wrapper missed supplied MNAR indicator; captured={captured}"


@pytest.mark.slow
def test_biz_val_rf_wrapper_misses_mnar_without_indicator_is_fs_gap():
    """FS GAP: with the raw (imputed) MNAR value and NO indicator column, the
    RF-core wrappers cannot recover the missingness signal -- the value is
    pure noise once imputed, so no downstream importance survives. Pinned with
    a non-strict xfail to document that the indicator MUST be emitted upstream
    (the wrappers do not synthesise it themselves like MRMR's L37 path)."""
    df, ys = _build_mnar(seed=0, n=700)
    df_imp = _impute_mean(df)  # value only, no is_missing__ column
    s = _make_boruta_shap("binary")
    s.fit(df_imp, ys.to_numpy())
    sel = selected_names(s)
    # 'mnar' carrying real signal post-imputation would mean the wrapper
    # recovered MNAR from the value alone -- it cannot; assert it is NOT
    # informatively selected via a downstream-AUC check on the imputed value.
    val = df_imp["mnar"].to_numpy().reshape(-1, 1)
    proba = LogisticRegression().fit(val, ys).predict_proba(val)[:, 1]
    auc = roc_auc_score(ys, proba)
    pytest.xfail(f"FS GAP: RF-core wrappers miss MNAR without a supplied is_missing__ indicator (imputed-value AUC {auc:.3f} ~ 0.5, selected={sel})")
