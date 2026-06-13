"""Unit + biz_value + cProfile triad for the Haar wavelet / localized
multiresolution basis (backlog #13, 2026-06-09).

A Haar wavelet captures a LOCALIZED bump / multiscale piecewise structure --
``y`` jumps only inside a narrow sub-window of x, or has step/contrast structure
at several scales at once -- a signal shape the catalog cannot: Fourier is GLOBAL
(Gibbs-rings a bump), the cubic B-spline's FIXED quantile knots smooth a narrow
bump away, and rounding is a global flat-step quantiser.

UNIT contracts:
* ``_dyadic_haar_leg`` is the closed-form +1/-1/0 indicator on the dyadic cell;
* recipe replay is a pure, leak-safe function of X (bit-exact + pickle roundtrip);
* pure noise admits no wavelet leg (held-out scale-selection rejects all);
* the dispatcher routes the ``orth_wavelet`` kind;
* held-out scale-selection bounds the candidate count (<= max_legs).

BIZ_VALUE contracts (the decisive ones -- a Haar leg is NON-monotone, hence
MI-VISIBLE, so the value IS measurable as held-out incremental information /
Ridge R^2 of the leg SET over raw x):
* LOCALIZED-STEP WIN: held-out Ridge R^2 of [x, Haar leg set] BEATS the best
  Fourier leg set AND the best B-spline leg set on a dyadic step fixture;
* MULTISCALE WIN: on a step + narrow bump the Haar set beats the global Fourier
  set (Gibbs) -- the multiresolution showcase;
* SMOOTH COMPLEMENTARITY: on y = sin(2*pi*x) Fourier WINS and the Haar set does
  NOT beat it (proves it is localized-complementary, not a Fourier-clone), and
  the admission gate admits ZERO legs on smooth data.

DEFAULT-PATH contracts (the operator is DEFAULT-ON, 2026-06-09): the win must
manifest with a plain ``MRMR()`` (no opt-in flag) and self-limit to zero columns
on data without localized structure:
* a localized-step target gets a Haar leg selected into support on the default
  transform output;
* on pure noise / a smooth sin / the canonical pair-FE fixture the default path
  retains ZERO wavelet legs (no spurious columns, canonical recovery intact).
"""
from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

N = 4000


def _heldout_r2(feat_cols, y):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    X = np.column_stack([np.asarray(c, float) for c in feat_cols])
    idx = np.arange(len(y))
    va = (idx % 3) == 0
    tr = ~va
    sc = StandardScaler().fit(X[tr])
    r = Ridge(alpha=1.0).fit(sc.transform(X[tr]), y[tr])
    pred = r.predict(sc.transform(X[va]))
    yv = y[va]
    sse = float(np.sum((yv - pred) ** 2))
    sst = float(np.sum((yv - yv.mean()) ** 2))
    return 1.0 - sse / sst if sst > 1e-12 else 0.0


def _best_set_r2(legs, x, y, k=4):
    """Held-out Ridge R^2 of [x, top-k legs by train-side MI] -- the real
    operator behaviour (small scale-selected SET, not a single leg)."""
    from mlframe.feature_selection.filters._wavelet_basis_fe import _binned_mi
    n = len(y)
    va = (np.arange(n) % 3) == 0
    tr = ~va
    scored = []
    for key, leg in legs.items():
        if np.std(leg[tr]) < 1e-9:
            continue
        scored.append((_binned_mi(leg[tr], y[tr]), key))
    scored.sort(reverse=True)
    sel = [key for _, key in scored[:k]]
    if not sel:
        return _heldout_r2([x], y)
    return _heldout_r2([x] + [legs[key] for key in sel], y)


def _haar_set(x, lo, span, max_scale=3):
    from mlframe.feature_selection.filters._wavelet_basis_fe import _dyadic_haar_leg
    z = np.clip((x - lo) / span, 0.0, 1.0)
    legs = {}
    for j in range(max_scale + 1):
        for k in range(2 ** j):
            legs[(j, k)] = _dyadic_haar_leg(z, j, k)
    return legs


def _fourier_set(x, lo, span, n_freq=8):
    z = (x - lo) / span
    legs = {}
    for f in range(1, n_freq + 1):
        legs[(f, "sin")] = np.sin(2 * np.pi * f * z)
        legs[(f, "cos")] = np.cos(2 * np.pi * f * z)
    return legs


def _spline_set(x, lo, hi, n_inner=8, degree=3):
    from scipy.interpolate import BSpline
    z = np.clip((x - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    inner = np.linspace(1.0 / (n_inner + 1), n_inner / (n_inner + 1), n_inner)
    knots = np.concatenate([np.zeros(degree + 1), inner, np.ones(degree + 1)])
    nb = len(knots) - degree - 1
    legs = {}
    for i in range(nb):
        c = np.zeros(nb)
        c[i] = 1.0
        legs[i] = np.nan_to_num(BSpline(knots, c, degree, extrapolate=False)(z), nan=0.0)
    return legs


# --------------------------------------------------------------------------- #
#                                   UNIT                                       #
# --------------------------------------------------------------------------- #
def test_dyadic_haar_leg_closed_form():
    """psi_{j,k} is +1 on the left half, -1 on the right half of the dyadic cell
    [k/2^j, (k+1)/2^j), 0 outside."""
    from mlframe.feature_selection.filters._wavelet_basis_fe import _dyadic_haar_leg
    z = np.linspace(0.0, 1.0, 1001)
    # j=0,k=0: +1 on [0,0.5), -1 on [0.5,1)
    leg = _dyadic_haar_leg(z, 0, 0)
    assert np.all(leg[(z >= 0.0) & (z < 0.5)] == 1.0)
    assert np.all(leg[(z >= 0.5) & (z < 1.0)] == -1.0)
    # j=1,k=1: cell [0.5,1.0) -> +1 on [0.5,0.75), -1 on [0.75,1.0), 0 on [0,0.5)
    leg = _dyadic_haar_leg(z, 1, 1)
    assert np.all(leg[(z >= 0.0) & (z < 0.5)] == 0.0)
    assert np.all(leg[(z >= 0.5) & (z < 0.75)] == 1.0)
    assert np.all(leg[(z >= 0.75) & (z < 1.0)] == -1.0)
    # zero-mean over its support (a wavelet integrates to 0)
    leg = _dyadic_haar_leg(z, 2, 1)
    assert abs(float(leg.sum())) <= 2.0  # only boundary-cell rounding


def test_recipe_replay_is_leak_safe_and_bit_exact():
    """The emitted column == apply_recipe replay byte-for-byte, and the recipe
    reads only X (no y)."""
    from mlframe.feature_selection.filters._wavelet_basis_fe import (
        generate_wavelet_features, build_orth_wavelet_recipe,
    )
    from mlframe.feature_selection.filters.engineered_recipes import apply_recipe
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, N)
    y = ((x > 0.5).astype(float) + rng.normal(0, 0.3, N) > 0.5).astype(int)
    X = pd.DataFrame({"a": x, "b": rng.normal(0, 1, N)})
    eng, meta = generate_wavelet_features(X, y=y)
    assert not eng.empty
    for nm in eng.columns:
        m = meta[nm]
        r = build_orth_wavelet_recipe(
            name=nm, src_name=m["src"], j=m["j"], k=m["k"],
            lo=m["lo"], span=m["span"],
        )
        replay = apply_recipe(r, X)
        assert np.array_equal(replay, eng[nm].to_numpy()), f"{nm} not bit-exact"
        # pickle round-trip
        r2 = pickle.loads(pickle.dumps(r))
        assert np.array_equal(apply_recipe(r2, X), eng[nm].to_numpy())


def test_dispatcher_routes_orth_wavelet():
    from mlframe.feature_selection.filters.engineered_recipes import (
        apply_recipe, build_orth_wavelet_recipe,
    )
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.uniform(0, 1, 500)})
    r = build_orth_wavelet_recipe(name="a__haar_j1k0", src_name="a", j=1, k=0, lo=0.0, span=1.0)
    out = apply_recipe(r, X)
    assert out.shape == (500,)
    assert set(np.unique(out)).issubset({-1.0, 0.0, 1.0})


def test_pure_noise_admits_no_wavelet():
    """Held-out scale-selection + the incremental gate reject every leg on pure
    noise -> 0 legs."""
    from mlframe.feature_selection.filters._wavelet_basis_fe import (
        hybrid_wavelet_fe_with_recipes,
    )
    n_admit = 0
    for seed in range(8):
        rng = np.random.default_rng(100 + seed)
        X = pd.DataFrame({
            "a": rng.uniform(0, 1, N), "b": rng.normal(0, 1, N),
            "c": rng.uniform(-1, 1, N), "d": rng.normal(0, 1, N),
        })
        yc = (rng.normal(0, 1, N) > 0).astype(int)
        _, keep, _, _ = hybrid_wavelet_fe_with_recipes(X, yc)
        n_admit += len(keep)
    assert n_admit == 0, f"noise admitted {n_admit} legs over 8 seeds (want 0)"


def test_scale_selection_bounds_candidate_count():
    """Even on a richly localized column the emitted leg count is capped at
    max_legs (candidate-explosion control)."""
    from mlframe.feature_selection.filters._wavelet_basis_fe import (
        generate_wavelet_features,
    )
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 1, N)
    # multi-scale structure to trip many candidate legs
    y = ((x > 0.5).astype(float) + 0.8 * ((x > 0.75) & (x < 0.875))
         + rng.normal(0, 0.2, N) > 0.5).astype(int)
    X = pd.DataFrame({"a": x})
    eng, meta = generate_wavelet_features(X, y=y, max_legs=6)
    per_src = {}
    for nm in eng.columns:
        per_src.setdefault(meta[nm]["src"], 0)
        per_src[meta[nm]["src"]] += 1
    for src, cnt in per_src.items():
        assert cnt <= 6, f"{src} emitted {cnt} legs > max_legs=6"


# --------------------------------------------------------------------------- #
#                                 BIZ_VALUE                                    #
# --------------------------------------------------------------------------- #
def test_bizval_localized_step_beats_fourier_and_spline():
    """Decisive WIN: on a dyadic step y=1[x>0.5] the Haar leg SET held-out Ridge
    R^2 beats the best Fourier set AND the best B-spline set."""
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 1, N)
    y = (x > 0.5).astype(float) + rng.normal(0, 0.3, N)
    lo, hi = float(x.min()), float(x.max())
    span = hi - lo
    r_haar = _best_set_r2(_haar_set(x, lo, span), x, y)
    r_four = _best_set_r2(_fourier_set(x, lo, span), x, y)
    r_spl = _best_set_r2(_spline_set(x, lo, hi), x, y)
    assert r_haar > r_four, f"Haar {r_haar:.4f} !> Fourier {r_four:.4f}"
    assert r_haar > r_spl, f"Haar {r_haar:.4f} !> spline {r_spl:.4f}"


def test_bizval_multiscale_beats_fourier():
    """MULTISCALE showcase: step + narrow bump -> Haar set beats the global
    Fourier set (Gibbs rings the discontinuities)."""
    rng = np.random.default_rng(4)
    x = rng.uniform(0, 1, N)
    y = ((x > 0.5).astype(float) + 0.8 * ((x > 0.75) & (x < 0.875))
         + rng.normal(0, 0.25, N))
    lo, hi = float(x.min()), float(x.max())
    span = hi - lo
    r_haar = _best_set_r2(_haar_set(x, lo, span), x, y)
    r_four = _best_set_r2(_fourier_set(x, lo, span), x, y)
    assert r_haar > r_four, f"Haar {r_haar:.4f} !> Fourier {r_four:.4f}"


def test_bizval_smooth_complementarity_fourier_wins():
    """COMPLEMENTARITY control: on y=sin(2*pi*x) Fourier WINS and the Haar set
    does NOT beat it -> Haar is localized-complementary, not a Fourier clone."""
    rng = np.random.default_rng(5)
    x = rng.uniform(0, 1, N)
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.3, N)
    lo, hi = float(x.min()), float(x.max())
    span = hi - lo
    r_haar = _best_set_r2(_haar_set(x, lo, span), x, y)
    r_four = _best_set_r2(_fourier_set(x, lo, span), x, y)
    assert r_four >= r_haar, f"smooth: Fourier {r_four:.4f} should >= Haar {r_haar:.4f}"


def test_bizval_smooth_admits_zero_legs():
    """The admission gate's complementarity guard admits ZERO wavelet legs on a
    smooth sin column (Fourier owns that regime)."""
    from mlframe.feature_selection.filters._wavelet_basis_fe import (
        hybrid_wavelet_fe_with_recipes,
    )
    n_admit = 0
    for seed in range(8):
        rng = np.random.default_rng(200 + seed)
        x = rng.uniform(0, 1, N)
        y = np.sin(2 * np.pi * x) + rng.normal(0, 0.3, N)
        yc = (y > np.median(y)).astype(int)
        X = pd.DataFrame({
            "a": x, "b": rng.normal(0, 1, N),
            "c": rng.uniform(-1, 1, N), "d": rng.normal(0, 1, N),
        })
        _, keep, _, _ = hybrid_wavelet_fe_with_recipes(X, yc)
        n_admit += len(keep)
    assert n_admit == 0, f"smooth admitted {n_admit} legs over 8 seeds (want 0)"


def test_bizval_localized_step_admits_legs():
    """The gate DOES admit legs on a localized step (positive control for the
    complementarity test above)."""
    from mlframe.feature_selection.filters._wavelet_basis_fe import (
        hybrid_wavelet_fe_with_recipes,
    )
    admits = 0
    for seed in range(8):
        rng = np.random.default_rng(300 + seed)
        x = rng.uniform(0, 1, N)
        y = (x > 0.5).astype(float) + rng.normal(0, 0.3, N)
        yc = (y > np.median(y)).astype(int)
        X = pd.DataFrame({"a": x, "b": rng.normal(0, 1, N)})
        _, keep, _, _ = hybrid_wavelet_fe_with_recipes(X, yc)
        if keep:
            admits += 1
    assert admits >= 6, f"step admitted legs in only {admits}/8 seeds (want >=6)"


# --------------------------------------------------------------------------- #
#                              DEFAULT-PATH                                    #
# --------------------------------------------------------------------------- #
def test_default_on_selects_wavelet_on_localized_step():
    """A plain MRMR() (no opt-in) selects a Haar leg into support on a localized
    step target, and transform() replays it leak-free."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    rng = np.random.default_rng(6)
    x = rng.uniform(0, 1, 3000)
    y = ((x > 0.5).astype(float) + rng.normal(0, 0.3, 3000) > 0.5).astype(int)
    X = pd.DataFrame({"loc": x, "n1": rng.normal(0, 1, 3000), "n2": rng.uniform(-1, 1, 3000)})
    m = MRMR(max_runtime_mins=2, verbose=0)
    m.fit(X, y)
    assert getattr(m, "fe_wavelet_enable", False) is True, "default should be ON"
    wv = list(getattr(m, "wavelet_features_", []) or [])
    assert any("haar" in str(c) for c in wv), f"no Haar leg selected: {wv}"
    Xt = m.transform(X)
    assert any("haar" in str(c) for c in Xt.columns)


def test_default_on_canonical_recovery_not_perturbed():
    """On the canonical pair-FE fixture y=a^2/b+log(c)*sin(d) the wavelet stage
    admits ZERO legs -> it does NOT perturb genuine pair-FE recovery."""
    from mlframe.feature_selection.filters._wavelet_basis_fe import (
        hybrid_wavelet_fe_with_recipes,
    )
    rng = np.random.default_rng(7)
    a = rng.uniform(0.5, 2, N); b = rng.uniform(0.5, 2, N)
    c = rng.uniform(0.5, 2, N); d = rng.uniform(0, 2 * np.pi, N)
    y = a ** 2 / b + np.log(c) * np.sin(d) + rng.normal(0, 0.05, N)
    yc = (y > np.median(y)).astype(int)
    X = pd.DataFrame({"a": a, "b": b, "c": c, "d": d})
    _, keep, _, _ = hybrid_wavelet_fe_with_recipes(X, yc)
    assert keep == [], f"canonical admitted spurious wavelet legs: {keep}"


# --------------------------------------------------------------------------- #
#                                 cPROFILE                                     #
# --------------------------------------------------------------------------- #
def test_cprofile_wavelet_stage_hotspot(capsys):
    """cProfile the generate+hybrid wavelet stage; assert it completes well under
    a wall-clock budget on a representative frame (n=4000, p=8). The stage hotspot
    is the per-candidate-leg binned-MI in scale-selection; the held-out MAD floor
    + max_legs cap bound the leg count so the stage stays cheap."""
    import cProfile
    import io
    import pstats
    import time
    from mlframe.feature_selection.filters._wavelet_basis_fe import (
        hybrid_wavelet_fe_with_recipes,
    )
    rng = np.random.default_rng(8)
    x = rng.uniform(0, 1, N)
    y = ((x > 0.5).astype(float) + rng.normal(0, 0.3, N) > 0.5).astype(int)
    cols = {"loc": x}
    for i in range(7):
        cols[f"n{i}"] = rng.normal(0, 1, N)
    X = pd.DataFrame(cols)
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    hybrid_wavelet_fe_with_recipes(X, y)
    pr.disable()
    elapsed = time.perf_counter() - t0
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(12)
    with capsys.disabled():
        print(f"\n[wavelet cProfile] p=8 n={N} wall={elapsed*1000:.1f} ms")
        print(s.getvalue()[:1400])
    # Generous budget; the stage is a handful of binned-MI passes per column.
    assert elapsed < 5.0, f"wavelet stage took {elapsed:.2f}s (>5s budget)"


def _binned_mi_legacy_reference(feat, y, nbins=10):
    """Pre-optimization double-loop reference for ``_binned_mi`` (perf iter47).

    The production kernel now builds the contingency table via a single bincount over the dense
    joint code; this reference keeps the original O(|fa|*|yb|*n) boolean-mask formulation so the
    bit-identity regression below cannot silently drift if the kernel changes."""
    feat = np.asarray(feat, dtype=np.float64).ravel()
    y = np.asarray(y).ravel()
    n = feat.size
    if n == 0 or n != y.size:
        return 0.0
    uniq_f = np.unique(feat)
    if uniq_f.size <= nbins:
        fb = np.searchsorted(uniq_f, feat)
    else:
        edges = np.quantile(feat, np.linspace(0.0, 1.0, nbins + 1)[1:-1])
        fb = np.digitize(feat, edges)
    if np.issubdtype(y.dtype, np.integer) and np.unique(y).size <= 20:
        yb = y.astype(np.int64)
    elif np.unique(y).size <= 20:
        uy = np.unique(y)
        yb = np.searchsorted(uy, y)
    else:
        edges_y = np.quantile(y, np.linspace(0.0, 1.0, nbins + 1)[1:-1])
        yb = np.digitize(y, edges_y)
    mi = 0.0
    for a in np.unique(fb):
        pa = np.mean(fb == a)
        if pa <= 0:
            continue
        mask_a = fb == a
        for b in np.unique(yb):
            pab = np.mean(mask_a & (yb == b))
            if pab > 0:
                pb = np.mean(yb == b)
                mi += pab * np.log(pab / (pa * pb))
    return float(max(mi, 0.0))


def test_binned_mi_histogram_bit_identical_to_legacy_double_loop():
    """perf iter47: the bincount joint-histogram rewrite of ``_binned_mi`` must be bit-identical
    to the prior double-loop across ternary-Haar-leg, discrete-class, and continuous-y inputs."""
    from mlframe.feature_selection.filters._wavelet_basis_fe import _binned_mi

    rng = np.random.default_rng(1234)
    for _ in range(300):
        n = int(rng.integers(60, 2000))
        if rng.random() < 0.5:
            feat = rng.choice([-1.0, 0.0, 1.0], size=n, p=[0.3, 0.4, 0.3])
        else:
            feat = rng.normal(size=n)
        if rng.random() < 0.6:
            y = rng.integers(0, int(rng.integers(2, 8)), size=n)
        else:
            y = rng.normal(size=n)
        assert _binned_mi(feat, y) == _binned_mi_legacy_reference(feat, y)
