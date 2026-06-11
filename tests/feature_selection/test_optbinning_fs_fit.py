"""Fit-level coverage for the optbinning IV feature-SELECTION pipelines.

``get_binningprocess_featureselectors`` returns four sklearn Pipelines; the
companion ``test_optbinning.py`` only ever fits the *no-feature-selection*
``bp_nocats_nofs`` variant, so the IV ``selection_criteria`` gate -- the whole
point of the module -- had zero executed validation. This file fits the
``_fs`` variants and pins, quantitatively, that the IV gate actually drops
noise columns while keeping a strong signal.

De-flaked IV biz_value (no defensive skip):

The companion test's biz_value assertion self-skips whenever optbinning
collapses the signal column to a single bin (IV==0). The proposal suggested
removing that skip by feeding a 2-valued ``np.sign(x)`` signal "so IV>0 is
guaranteed arithmetically". Empirically that premise is FALSE for optbinning:
a PERFECTLY separable 2-valued column (``y == (sign>0)``) produces
``splits == []`` -- the optimizer keeps one ``(-inf, inf)`` bin and reports
IV == 0 (verified: ``OptimalBinning`` on ``sign(x)`` vs ``x>0`` -> IV 0.0,
status OPTIMAL). A perfect separator is the WORST case here, not the safe one.

The construction that genuinely removes the need for a skip is a strong but
*noisy* logistic signal ``p = sigmoid(2.5*x); y ~ Bernoulli(p)``. The binner
then finds a clean monotone, multi-split WoE gradient: signal IV ~3.0-3.6
across seeds (measured 3.08/3.20/3.46/3.32/3.57 for seeds 0-4) vs max-noise IV
~0.07-0.16 -- a 23-47x margin that never collapses to zero. The floors below
are pinned 5-15% under the measured minima.

Requires optbinning >= 0.21 for sklearn 1.6+ compatibility (0.20.x calls the
removed ``check_array(force_all_finite=...)``; 0.21 uses ``ensure_all_finite``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("optbinning")
pytest.importorskip("category_encoders")


def _optbinning_sklearn_compatible() -> bool:
    """optbinning <= 0.20.x calls scikit-learn's ``check_array(force_all_finite=...)``,
    a kwarg REMOVED in scikit-learn 1.6 (renamed ``ensure_all_finite``). On such a
    combo every optbinning ``.fit`` raises ``TypeError: check_array() got an unexpected
    keyword argument 'force_all_finite'`` from deep inside optbinning's own
    ``metrics.jeffrey`` -- a THIRD-PARTY version mismatch, not an mlframe defect. The
    module docstring already pins the requirement (optbinning >= 0.21). ``importorskip``
    only checks PRESENCE, so guard the version combo here too so this fit-level suite
    skips cleanly on the incompatible pairing instead of failing in library internals."""
    try:
        from packaging.version import parse as _v
        import optbinning as _ob
        import sklearn as _sk
        if _v(_sk.__version__) >= _v("1.6") and _v(_ob.__version__) < _v("0.21"):
            return False
    except Exception:
        # If we cannot determine versions, let the tests run and surface any real error.
        return True
    return True


if not _optbinning_sklearn_compatible():
    import optbinning as _ob_mod
    import sklearn as _sk_mod
    pytest.skip(
        f"optbinning {_ob_mod.__version__} is incompatible with scikit-learn "
        f"{_sk_mod.__version__} (optbinning<0.21 calls the removed "
        f"check_array(force_all_finite=...) on sklearn>=1.6); the IV feature-selection "
        f"fit path raises in optbinning internals. Upgrade optbinning>=0.21 to exercise "
        f"this suite. See module docstring.",
        allow_module_level=True,
    )

from mlframe.feature_selection.optbinning import get_binningprocess_featureselectors  # noqa: E402

from tests.feature_selection.conftest import fast_subset  # noqa: E402


# A strong-noisy-logistic signal yields signal IV ~3.0-3.6 (never 0); a 2-valued
# perfect separator yields IV 0 (optbinning keeps one bin) -- see module docstring.
_SIGNAL_IV_FLOOR = 2.7          # 5-15% below the measured per-seed minimum 3.08
_IV_RATIO_FLOOR = 18.0          # measured min ratio 23.2; floor ~20% under
_N_NOISE = 6
_SEEDS = (0, 1, 2, 3)


def _make_noisy_logistic_df(n: int = 800, n_noise: int = _N_NOISE, seed: int = 0):
    """Numeric frame: one strong noisy-logistic ``signal_step`` column +
    ``n_noise`` pure-noise columns, binary target. The signal is informative
    but NOT perfectly separable, so optbinning produces a multi-split, high-IV
    binning (IV ~3.0-3.6) -- the construction that guarantees IV>0 in practice.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-2.5 * x))
    y = (rng.uniform(size=n) < p).astype(int)
    df = pd.DataFrame(
        {"signal_step": x, **{f"noise_{i}": rng.normal(size=n) for i in range(n_noise)}}
    )
    return df, pd.Series(y, name="y")


def _make_categorical_signal_df(n: int = 700, seed: int = 0):
    """A categorical + numeric frame where BOTH carry signal, for the
    CatBoostEncoder -> BinningProcess ``bp_withcats_fs`` chain. Category "A"
    pushes the target positive; the numeric column is a noisy logistic driver.
    """
    rng = np.random.default_rng(seed)
    cats = pd.Categorical(rng.choice(["A", "B", "C"], size=n))
    x_num = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-(1.8 * x_num + 2.0 * (cats.codes == 0).astype(float) - 0.7)))
    y = (rng.uniform(size=n) < p).astype(int)
    df = pd.DataFrame({"cat_feat": cats, "num_feat": x_num})
    return df, pd.Series(y, name="y")


def _fit_nocats_fs(df, y, iv_kwargs):
    _, _, bp_nocats_fs, _ = get_binningprocess_featureselectors(df, n_jobs=1, iv_kwargs=iv_kwargs)
    bp_nocats_fs.fit(df, y)
    return bp_nocats_fs


# ---------------------------------------------------------------------------
# (a) the FS pipeline actually selects: drops noise, keeps the signal
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("seed", _SEEDS)
def test_fs_pipeline_fit_drops_noise_keeps_signal(seed):
    """``bp_nocats_fs`` with an IV gate must SELECT: the fitted transform is
    narrower than the input frame, the strong signal survives ``get_support``,
    and at least half the noise columns are dropped.

    ``iv_kwargs`` min is 0.2 (not the proposal's 0.05): noise IVs hover at
    0.05-0.16 here, so at min=0.05 only 1-2 of 6 noise cols drop -- too few to
    pin ">=half noise dropped" robustly. At min=0.2 the signal (IV ~3.0+) clears
    easily while every noise col (IV <=0.16) falls below the gate across all
    seeds (measured 6/6 noise dropped, seeds 0-5).
    """
    df, y = _make_noisy_logistic_df(n=800, seed=seed)
    bp_pipe = _fit_nocats_fs(df, y, iv_kwargs={"min": 0.2, "strategy": "highest"})
    transformed = bp_pipe.transform(df)

    assert transformed.shape[0] == df.shape[0], "transform must preserve row count"
    assert transformed.shape[1] < df.shape[1], (
        f"FS transform must be narrower than input; got {transformed.shape[1]} >= {df.shape[1]}"
    )

    bp = dict(bp_pipe.steps)["BP"]
    support = set(str(c) for c in bp.get_support(names=True))
    assert "signal_step" in support, f"strong signal must survive selection; support={sorted(support)}"

    noise_cols = [c for c in df.columns if c.startswith("noise_")]
    dropped_noise = [c for c in noise_cols if c not in support]
    assert len(dropped_noise) >= len(noise_cols) / 2, (
        f"IV gate must drop >= half the noise; dropped {len(dropped_noise)}/{len(noise_cols)}; "
        f"support={sorted(support)}"
    )


def test_fs_pipeline_fit_drops_noise_keeps_signal_fast():
    """Fast-mode representative of the parametrized selection test (one seed)."""
    seed = fast_subset(_SEEDS, n=1)[0]
    df, y = _make_noisy_logistic_df(n=800, seed=seed)
    bp_pipe = _fit_nocats_fs(df, y, iv_kwargs={"min": 0.2, "strategy": "highest"})
    transformed = bp_pipe.transform(df)
    assert transformed.shape[1] < df.shape[1]
    bp = dict(bp_pipe.steps)["BP"]
    support = set(str(c) for c in bp.get_support(names=True))
    assert "signal_step" in support
    noise_cols = [c for c in df.columns if c.startswith("noise_")]
    dropped = [c for c in noise_cols if c not in support]
    assert len(dropped) >= len(noise_cols) / 2


# ---------------------------------------------------------------------------
# (b) de-flaked IV biz_value: signal IV >> noise IV, no defensive skip
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("seed", _SEEDS)
def test_biz_value_signal_iv_dominates_noise_no_skip(seed):
    """biz_value (de-flaked, no ``pytest.skip``): the fitted BinningProcess must
    assign the noisy-logistic ``signal_step`` an IV >= ``_SIGNAL_IV_FLOOR`` and
    at least ``_IV_RATIO_FLOOR``x the largest noise IV.

    Floors are pinned below MEASURED minima (signal IV 3.08, ratio 23.2 over
    seeds 0-3). This replaces the companion test's IV==0 skip branch: the noisy
    construction never collapses the column, so the assertion always executes.
    """
    df, y = _make_noisy_logistic_df(n=800, seed=seed)
    bp_pipe = _fit_nocats_fs(df, y, iv_kwargs={"min": 0.02, "strategy": "highest"})
    bp = dict(bp_pipe.steps)["BP"]

    iv_map = {var: float(bp.get_binned_variable(var).binning_table.iv) for var in bp.variable_names}
    signal_iv = iv_map["signal_step"]
    noise_ivs = [iv for name, iv in iv_map.items() if name.startswith("noise_")]
    max_noise_iv = max(noise_ivs)

    assert signal_iv >= _SIGNAL_IV_FLOOR, (
        f"signal IV {signal_iv:.3f} below floor {_SIGNAL_IV_FLOOR}; iv_map={iv_map}"
    )
    assert signal_iv > 0.0, "noisy-logistic signal must never collapse to IV==0 (the de-flake invariant)"
    assert signal_iv >= _IV_RATIO_FLOOR * max_noise_iv, (
        f"signal IV {signal_iv:.3f} must be >= {_IV_RATIO_FLOOR}x max noise IV "
        f"{max_noise_iv:.3f}; iv_map={iv_map}"
    )


def test_biz_value_signal_iv_dominates_noise_fast():
    """Fast-mode representative of the IV biz_value test (one seed)."""
    seed = fast_subset(_SEEDS, n=1)[0]
    df, y = _make_noisy_logistic_df(n=800, seed=seed)
    bp_pipe = _fit_nocats_fs(df, y, iv_kwargs={"min": 0.02, "strategy": "highest"})
    bp = dict(bp_pipe.steps)["BP"]
    iv_map = {var: float(bp.get_binned_variable(var).binning_table.iv) for var in bp.variable_names}
    signal_iv = iv_map["signal_step"]
    max_noise_iv = max(iv for name, iv in iv_map.items() if name.startswith("noise_"))
    assert signal_iv >= _SIGNAL_IV_FLOOR
    assert signal_iv >= _IV_RATIO_FLOOR * max_noise_iv


def test_perfect_separator_collapses_to_zero_iv_documents_skip_smell():
    """Pins the WHY behind the de-flake: a perfectly separable 2-valued column
    (the proposal's ``np.sign(x)`` recipe) makes optbinning keep ONE bin and
    report IV==0. This is exactly the case the companion test's defensive skip
    fires on -- so the sign-step recipe would NOT have removed the skip; the
    noisy-logistic construction is what does.
    """
    rng = np.random.default_rng(0)
    n = 800
    x = rng.normal(size=n)
    y = (x > 0).astype(int)
    df = pd.DataFrame({"signal_step": np.sign(x), "noise_0": rng.normal(size=n)})
    ys = pd.Series(y, name="y")
    _, _, bp_nocats_fs, _ = get_binningprocess_featureselectors(
        df, n_jobs=1, iv_kwargs={"min": 0.05, "strategy": "highest"}
    )
    bp_nocats_fs.fit(df, ys)
    bp = dict(bp_nocats_fs.steps)["BP"]
    sep_table = bp.get_binned_variable("signal_step").binning_table
    assert float(sep_table.iv) == 0.0, (
        "perfect 2-valued separator must collapse to IV==0 (no split) -- the de-flake rationale; "
        f"got IV={float(sep_table.iv):.4f}"
    )


# ---------------------------------------------------------------------------
# (c) bp_withcats_fs: the CatBoostEncoder -> BinningProcess chain fits
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("seed", _SEEDS)
def test_withcats_fs_fit_with_categorical(seed):
    """``bp_withcats_fs`` (CatBoostEncoder -> BinningProcess with an IV gate) must
    fit and transform a frame containing a categorical column. The chain is never
    fitted by the companion test. Assert the transform preserves row count and
    yields a non-empty, no-wider-than-input width, and that the signal-carrying
    categorical column survives the IV gate.
    """
    df, y = _make_categorical_signal_df(n=700, seed=seed)
    bp_withcats_fs, _, _, _ = get_binningprocess_featureselectors(df, n_jobs=1)
    bp_withcats_fs.fit(df, y)
    transformed = bp_withcats_fs.transform(df)

    assert transformed.shape[0] == df.shape[0], "transform must preserve row count"
    assert 0 < transformed.shape[1] <= df.shape[1], (
        f"withcats transform width must be in (0, {df.shape[1]}]; got {transformed.shape[1]}"
    )

    bp = dict(bp_withcats_fs.steps)["BP"]
    support = set(str(c) for c in bp.get_support(names=True))
    assert "cat_feat" in support, (
        f"signal-carrying categorical column must survive the IV gate; support={sorted(support)}"
    )


def test_withcats_fs_fit_with_categorical_fast():
    """Fast-mode representative of the withcats fit test (one seed)."""
    seed = fast_subset(_SEEDS, n=1)[0]
    df, y = _make_categorical_signal_df(n=700, seed=seed)
    bp_withcats_fs, _, _, _ = get_binningprocess_featureselectors(df, n_jobs=1)
    bp_withcats_fs.fit(df, y)
    transformed = bp_withcats_fs.transform(df)
    assert transformed.shape[0] == df.shape[0]
    assert 0 < transformed.shape[1] <= df.shape[1]


# ---------------------------------------------------------------------------
# (d) memory= (sklearn pipeline cache) smoke
# ---------------------------------------------------------------------------


def test_fs_pipeline_memory_cache_smoke(tmp_path):
    """``memory=str(tmp_path)`` is a real pipeline kwarg (untested). The cached
    pipeline must fit + transform without error and produce a selecting (narrower)
    transform identical in width to the uncached path.
    """
    df, y = _make_noisy_logistic_df(n=800, seed=0)
    iv_kwargs = {"min": 0.2, "strategy": "highest"}

    _, _, bp_cached, _ = get_binningprocess_featureselectors(
        df, n_jobs=1, memory=str(tmp_path), iv_kwargs=iv_kwargs
    )
    bp_cached.fit(df, y)
    cached_w = bp_cached.transform(df).shape[1]

    _, _, bp_uncached, _ = get_binningprocess_featureselectors(df, n_jobs=1, iv_kwargs=iv_kwargs)
    bp_uncached.fit(df, y)
    uncached_w = bp_uncached.transform(df).shape[1]

    assert cached_w < df.shape[1], f"cached FS pipeline must still select; width {cached_w} >= {df.shape[1]}"
    assert cached_w == uncached_w, (
        f"memory= must not change the selection; cached width {cached_w} != uncached {uncached_w}"
    )
    cache_files = list(tmp_path.rglob("*"))
    assert any(f.is_file() for f in cache_files), "memory= cache dir must hold at least one cached artefact"
