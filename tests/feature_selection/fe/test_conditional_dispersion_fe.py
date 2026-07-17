"""Family D -- conditional DISPERSION / 2nd-moment FE (backlog #12, 2026-06-09).

The TRIAD for a new operator (unit + biz_value + cProfile), pinning real numbers.

Mechanism
---------
Bin ``x_j``; per bin store conditional ``(mu_hat, sigma_hat)`` of ``x_i``; emit the
conditional z-score ``|z| = |(x_i - mu_hat_bin) / sigma_hat_bin|`` and ``z**2``
(the dispersion anomaly). Models conditional SCALE -- the gap Family B's
conditional MEAN leaves.

Contracts pinned (never xfail):

UNIT
* the emitted ``|z|`` / ``z**2`` columns equal the closed-form bin-lookup formula;
* recipe replay (``apply_recipe`` / ``transform``) reproduces the fit column EXACTLY
  (leak-safe: no y reference, byte-stable);
* recipe pickle / round-trips.

BIZ_VALUE (the operator earns its keep)
* HETEROSCEDASTIC two-bin fixture (bin A ``x_i~N(0,1)``, bin B ``x_i~N(0,5)``,
  ``y = 1[|x_i| > 2*sigma_bin]``): ``MI(y; |z|)`` HIGH and STRICTLY BEATS both the
  Family-B mean-residual sibling MI and the raw ``x_i`` MI -> the dispersion signal
  the location feature misses; downstream AUC lift over raw >= +0.03.
* HOMOSCEDASTIC control (same spread both bins): ``|z|`` is rank-identical to the
  scaled ``|mean-residual|`` (Spearman ~1) -> the dual-uplift gate / dedup drops it
  (self-limiting), so the hybrid admits 0 genuine dispersion columns.
* NOISE control (pure noise x, random y): the hybrid admits 0 columns.
* CANONICAL pair-FE fixture (``y = a**2/b + log(c)*sin(d)``): the dispersion stage
  admits 0 columns and the engineered set is byte-identical to dispersion-OFF ->
  it does NOT perturb genuine-feature recovery.

CPROFILE
* the fit hotspot is exercised and stays within a generous wall budget on the
  n<=4000 fixture (regression guard against an accidental O(n^2)).
"""

from __future__ import annotations

import pickle
import time
import warnings

import numpy as np
import pandas as pd
import pytest

from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from mlframe.feature_selection.filters._extra_fe_families import (
    apply_conditional_dispersion,
    build_conditional_dispersion_recipe,
    engineered_name_conditional_dispersion,
    generate_conditional_dispersion_features,
    generate_conditional_residual_features,
    hybrid_conditional_dispersion_fe,
)
from mlframe.feature_selection.filters.engineered_recipes import apply_recipe


def _mi_one(col, y, nbins: int = 10) -> float:
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
        _mi_classif_batch,
    )

    arr = np.asarray(col, dtype=np.float64).reshape(-1, 1)
    return float(_mi_classif_batch(arr, np.asarray(y).astype(np.int64), nbins=nbins)[0])


def _hetero_fixture(seed: int = 0, n: int = 4000):
    """Two-bin heteroscedastic fixture: bin selected by x_j; bin A x_i~N(0,1),
    bin B x_i~N(0,5); target = anomalous FOR its conditional spread."""
    rng = np.random.default_rng(seed)
    xj = rng.random(n)
    binB = xj > 0.5
    sd = np.where(binB, 5.0, 1.0)
    xi = rng.standard_normal(n) * sd
    y = (np.abs(xi) > 2.0 * sd).astype(int)
    X = pd.DataFrame({"xi": xi, "xj": xj})
    return X, y, sd


# ===========================================================================
# UNIT
# ===========================================================================
class TestConditionalDispersionUnit:
    def test_zscore_matches_closed_form(self):
        X, _y, _sd = _hetero_fixture()
        enc, raw = generate_conditional_dispersion_features(
            X,
            ["xi", "xj"],
            n_bins=10,
            kinds=("absz", "z2"),
        )
        absz_name = engineered_name_conditional_dispersion("xi", "xj", "absz")
        assert absz_name in enc.columns
        rec = raw[absz_name]
        # Closed-form recompute via the stored bins. ``_digitize_with_edges`` is
        # owned by the parent ``_extra_fe_families`` (the dispersion sibling now
        # imports it lazily to avoid the parent<->sibling import cycle, so it is no
        # longer re-exposed at the sibling's module top); ``_zscore_from_bins`` is
        # the dispersion module's own helper.
        from mlframe.feature_selection.filters._extra_fe_families import _digitize_with_edges
        from mlframe.feature_selection.filters._extra_fe_families_dispersion import _zscore_from_bins

        codes = _digitize_with_edges(X["xj"].to_numpy(), rec["edges"])
        z = _zscore_from_bins(
            X["xi"].to_numpy(dtype=float),
            codes,
            rec["bin_mean"],
            rec["bin_std"],
        )
        np.testing.assert_allclose(enc[absz_name].to_numpy(), np.abs(z), rtol=0, atol=0)
        z2_name = engineered_name_conditional_dispersion("xi", "xj", "z2")
        np.testing.assert_allclose(enc[z2_name].to_numpy(), z * z, rtol=0, atol=0)

    def test_replay_is_leak_safe_and_exact(self):
        X, _y, _sd = _hetero_fixture()
        enc, raw = generate_conditional_dispersion_features(
            X,
            ["xi", "xj"],
            n_bins=10,
            kinds=("absz",),
        )
        name = engineered_name_conditional_dispersion("xi", "xj", "absz")
        recipe = build_conditional_dispersion_recipe(name=name, **raw[name])
        # apply_recipe (the dispatch path) reproduces the fit column exactly.
        replay = np.asarray(apply_recipe(recipe, X), dtype=float)
        np.testing.assert_allclose(replay, enc[name].to_numpy(), rtol=0, atol=0)
        # Direct apply on a fresh slice equals the dispatch path (no y anywhere).
        head = X.head(500)
        direct = apply_conditional_dispersion(head, {**raw[name]})
        via_dispatch = np.asarray(apply_recipe(recipe, head), dtype=float)
        np.testing.assert_allclose(direct, via_dispatch, rtol=0, atol=0)

    def test_recipe_pickle_round_trip(self):
        X, y, _sd = _hetero_fixture()
        # decoy noise cols give a non-degenerate raw noise floor (a 2-col raw
        # frame yields a degenerate MAD floor that gates everything out).
        rng = np.random.default_rng(99)
        X = X.assign(g1=rng.standard_normal(len(X)), g2=rng.standard_normal(len(X)))
        _, _appended, recipes, _ = hybrid_conditional_dispersion_fe(
            X,
            y,
            num_cols=["xi", "xj", "g1", "g2"],
            n_bins=10,
            top_k=5,
        )
        assert recipes, "expected at least one dispersion recipe on hetero fixture"
        r = recipes[0]
        r2 = pickle.loads(pickle.dumps(r))
        a = np.asarray(apply_recipe(r, X), dtype=float)
        b = np.asarray(apply_recipe(r2, X), dtype=float)
        np.testing.assert_allclose(a, b, rtol=0, atol=0)

    def test_degenerate_constant_xi_does_not_blow_up(self):
        # Constant x_i within a bin -> sigma_hat falls back to global std (no /0).
        n = 2000
        rng = np.random.default_rng(3)
        xj = rng.random(n)
        xi = np.full(n, 2.5)  # fully constant
        X = pd.DataFrame({"xi": xi, "xj": xj})
        enc, _ = generate_conditional_dispersion_features(X, ["xi", "xj"], n_bins=10)
        # A constant x_i yields all-zero residual -> the std-guarded emission is
        # dropped (no information), so no xi-by-xj dispersion column is emitted.
        assert all(np.isfinite(enc[c].to_numpy()).all() for c in enc.columns)


# ===========================================================================
# BIZ_VALUE
# ===========================================================================
class TestConditionalDispersionBizValue:
    @pytest.mark.parametrize("seed", [0, 7, 42])
    def test_hetero_mi_beats_mean_residual_and_raw(self, seed):
        X, y, _sd = _hetero_fixture(seed=seed)
        enc_d, _ = generate_conditional_dispersion_features(
            X,
            ["xi", "xj"],
            n_bins=10,
            kinds=("absz",),
        )
        enc_b, _ = generate_conditional_residual_features(X, ["xi", "xj"], n_bins=10)
        absz = engineered_name_conditional_dispersion("xi", "xj", "absz")
        resid = "xi__cond_resid_by__xj"
        mi_absz = _mi_one(enc_d[absz].to_numpy(), y)
        mi_resid = _mi_one(enc_b[resid].to_numpy(), y)
        mi_absresid = _mi_one(np.abs(enc_b[resid].to_numpy()), y)
        mi_raw = _mi_one(X["xi"].to_numpy(), y)
        # The dispersion |z| carries strictly MORE MI about the heteroscedastic
        # target than the Family-B mean-residual sibling AND the raw column.
        assert mi_absz > mi_resid + 0.01, f"[seed={seed}] |z| MI {mi_absz:.4f} !> mean-resid {mi_resid:.4f}"
        assert mi_absz > mi_absresid, f"[seed={seed}] |z| MI {mi_absz:.4f} !> |mean-resid| {mi_absresid:.4f}"
        assert mi_absz > mi_raw, f"[seed={seed}] |z| MI {mi_absz:.4f} !> raw xi {mi_raw:.4f}"

    def test_hetero_downstream_auc_lift(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split

        X, y, _sd = _hetero_fixture(seed=0, n=4000)
        enc_d, _ = generate_conditional_dispersion_features(
            X,
            ["xi", "xj"],
            n_bins=10,
            kinds=("absz", "z2"),
        )
        absz = engineered_name_conditional_dispersion("xi", "xj", "absz")
        raw = X["xi"].to_numpy().reshape(-1, 1)
        disp = enc_d[absz].to_numpy().reshape(-1, 1)
        Xtr_r, Xte_r, Xtr_d, Xte_d, ytr, yte = train_test_split(
            raw,
            disp,
            y,
            test_size=0.3,
            random_state=0,
            stratify=y,
        )
        auc_raw = roc_auc_score(yte, LogisticRegression(max_iter=200).fit(Xtr_r, ytr).predict_proba(Xte_r)[:, 1])
        auc_disp = roc_auc_score(yte, LogisticRegression(max_iter=200).fit(Xtr_d, ytr).predict_proba(Xte_d)[:, 1])
        assert auc_disp >= auc_raw + 0.03, f"dispersion AUC {auc_disp:.3f} not >= raw {auc_raw:.3f}+0.03"

    def test_hetero_hybrid_admits_genuine_dispersion(self):
        X, y, _sd = _hetero_fixture(seed=0)
        # add decoy noise cols for a realistic raw noise floor
        rng = np.random.default_rng(99)
        X = X.assign(g1=rng.standard_normal(len(X)), g2=rng.standard_normal(len(X)))
        _, appended, _, _ = hybrid_conditional_dispersion_fe(
            X,
            y,
            num_cols=["xi", "xj", "g1", "g2"],
            n_bins=10,
            top_k=10,
        )
        genuine = [a for a in appended if a.endswith("by__xj")]
        assert genuine, f"genuine xi-by-xj dispersion not admitted: {appended}"

    def test_homoscedastic_self_limits(self):
        # Same spread both bins; conditional MEAN shift only -> |z| ~ scaled
        # |mean-residual| -> the dual-uplift gate admits NO genuine dispersion col.
        rng = np.random.default_rng(11)
        n = 4000
        xj = rng.random(n)
        mu = np.where(xj > 0.5, 3.0, -3.0)
        xi = mu + rng.standard_normal(n) * 1.5  # constant spread
        resid = xi - mu
        y = (np.abs(resid) > 1.5).astype(int)
        X = pd.DataFrame(
            {
                "xi": xi,
                "xj": xj,
                "n1": rng.standard_normal(n),
                "n2": rng.standard_normal(n),
                "n3": rng.random(n),
                "n4": rng.standard_normal(n),
            }
        )
        _, appended, _, _ = hybrid_conditional_dispersion_fe(
            X,
            y,
            num_cols=["xi", "xj", "n1", "n2", "n3", "n4"],
            n_bins=10,
            top_k=10,
        )
        genuine = [a for a in appended if a.endswith("by__xj")]
        assert genuine == [], f"homoscedastic dispersion NOT self-limited (gate admitted {genuine})"
        # And the dedup-self-limit invariant holds: |z| is rank-identical to |resid|.
        enc_d, _ = generate_conditional_dispersion_features(
            X[["xi", "xj"]],
            ["xi", "xj"],
            n_bins=10,
            kinds=("absz",),
        )
        enc_b, _ = generate_conditional_residual_features(
            X[["xi", "xj"]],
            ["xi", "xj"],
            n_bins=10,
        )
        rho = spearmanr(
            enc_d["xi__absz_by__xj"].to_numpy(),
            np.abs(enc_b["xi__cond_resid_by__xj"].to_numpy()),
        ).correlation
        assert rho > 0.95, f"homoscedastic |z| not ~ |mean-resid| (Spearman {rho:.3f})"

    def test_pure_noise_admits_nothing(self):
        rng = np.random.default_rng(7)
        n = 4000
        X = pd.DataFrame(
            {
                "xi": rng.standard_normal(n),
                "xj": rng.random(n),
                "g1": rng.standard_normal(n),
                "g2": rng.standard_normal(n),
            }
        )
        y = rng.integers(0, 2, n)
        _, appended, _, _ = hybrid_conditional_dispersion_fe(
            X,
            y,
            num_cols=["xi", "xj", "g1", "g2"],
            n_bins=10,
            top_k=10,
        )
        assert appended == [], f"pure noise admitted dispersion columns: {appended}"


# ===========================================================================
# CANONICAL pair-FE recovery non-perturbation (the hinge-regression guard)
# ===========================================================================
class TestConditionalDispersionDoesNotPerturbCanonical:
    def test_dispersion_admits_zero_on_canonical_and_engineered_set_unchanged(self):
        from mlframe.feature_selection.filters.mrmr import MRMR

        rng = np.random.default_rng(42)
        n = 4000
        a = rng.uniform(1, 5, n)
        b = rng.uniform(1, 5, n)
        c = rng.uniform(1, 5, n)
        d = rng.uniform(0, 2 * np.pi, n)
        e = rng.normal(0, 1, n)
        f = rng.normal(0, 1, n)
        y = a**2 / b + f / 5.0 + 3.0 * np.log(c) * np.sin(d)
        df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e})

        RAW = {"a", "b", "c", "d", "e"}
        fs_on = MRMR(verbose=0)
        fs_on.fit(df.copy(), pd.Series(y, name="y"))
        fs_off = MRMR(verbose=0, fe_conditional_dispersion_enable=False)
        fs_off.fit(df.copy(), pd.Series(y, name="y"))

        # The dispersion stage admits NO column on the (homoscedastic-in-x) fixture.
        assert list(getattr(fs_on, "conditional_dispersion_features_", []) or []) == [], "dispersion crowded the canonical fixture (should self-limit to 0)"
        eng_on = sorted(n for n in fs_on.get_feature_names_out() if n not in RAW)
        eng_off = sorted(n for n in fs_off.get_feature_names_out() if n not in RAW)
        assert eng_on == eng_off, f"dispersion DEFAULT-ON perturbed the engineered set:\n  ON ={eng_on}\n  OFF={eng_off}"


# ===========================================================================
# MRMR integration: default-on, leak-safe transform, clone/pickle
# ===========================================================================
class TestConditionalDispersionMRMRIntegration:
    def test_default_on(self):
        from mlframe.feature_selection.filters.mrmr import MRMR

        assert MRMR(verbose=0).fe_conditional_dispersion_enable is True

    def test_end_to_end_selects_dispersion_on_hetero_regression(self):
        from mlframe.feature_selection.filters.mrmr import MRMR

        rng = np.random.default_rng(0)
        n = 4000
        xj = rng.random(n)
        binB = xj > 0.5
        sd = np.where(binB, 5.0, 1.0)
        xi = rng.standard_normal(n) * sd
        g1 = rng.standard_normal(n)
        y = np.abs(xi) / sd + 0.5 * g1 + rng.standard_normal(n) * 0.1
        X = pd.DataFrame({"xi": xi, "xj": xj, "g1": g1, "g2": rng.standard_normal(n)})
        fs = MRMR(verbose=0)
        fs.fit(X, pd.Series(y, name="y"))
        out = list(fs.get_feature_names_out())
        # A dispersion column (directly or as a pair-FE operand) reaches support.
        assert any("absz_by" in c or "z2_by" in c for c in out), f"no dispersion column selected on heteroscedastic regression: {out}"
        # transform replays leak-safe (no y), finite, right shape.
        Xt = np.asarray(fs.transform(X.head(200)))
        assert Xt.shape == (200, len(out))
        assert np.isfinite(Xt).all()

    def test_clone_and_pickle_preserve_param(self):
        from sklearn.base import clone
        from mlframe.feature_selection.filters.mrmr import MRMR

        m = MRMR(verbose=0, fe_conditional_dispersion_top_k=7)
        m2 = clone(m)
        assert m2.fe_conditional_dispersion_top_k == 7
        assert m2.fe_conditional_dispersion_enable is True
        m3 = pickle.loads(pickle.dumps(m))
        assert m3.fe_conditional_dispersion_enable is True


# ===========================================================================
# CPROFILE perf guard
# ===========================================================================
def test_cprofile_hotspot_within_budget():
    """cProfile the dispersion generation on the n=4000 fixture; assert the
    wall time stays within a generous budget (O(n^2) regression guard). Prints
    the top hotspots for the perf log."""
    import cProfile
    import io
    import pstats

    X, _y, _sd = _hetero_fixture(seed=0, n=4000)
    rng = np.random.default_rng(5)
    for k in range(4):
        X[f"g{k}"] = rng.standard_normal(len(X))

    def _run():
        for _ in range(5):
            generate_conditional_dispersion_features(
                X,
                list(X.columns),
                n_bins=10,
                kinds=("absz", "z2"),
            )

    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    _run()
    pr.disable()
    wall = time.perf_counter() - t0
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(8)
    # 5 reps over a 6-col O(p^2)=30-pair fixture at n=4000; generous 12s budget.
    assert wall < 12.0, f"dispersion generation too slow: {wall:.2f}s (>12s)\n{s.getvalue()}"
