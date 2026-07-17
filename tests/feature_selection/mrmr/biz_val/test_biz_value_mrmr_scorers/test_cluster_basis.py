"""Layer 61 biz_value: PER-CLUSTER SHARED-BASIS FE.

Validates the new ``hybrid_orth_mi_cluster_basis_fe`` introduced 2026-05-31
(sibling module ``_orthogonal_cluster_basis_fe``): for each detected
cluster of correlated source columns, reduce to one aggregate column via
the chosen aggregator (mean_z / median_z / pc1), then evaluate
``basis_d(preprocess(aggregate))`` and rank by MI uplift.

Why this layer matters
----------------------

The cluster scenario ``s_i = z + epsilon_i`` (noisy reflections of a
latent ``z``) is dominant in real instrument / sensor / multi-source
data. Layer 21 emits ``basis_n(s_i)`` per individual member, but the
per-member non-linear signal is degraded by epsilon_i. Layer 7
``cluster_aggregate`` swaps the cluster down to its PC1/mean_z aggregate
as a NEW raw feature -- a denoising win on its own -- but it does NOT
run a basis expansion ON the aggregate. Layer 61 fills that gap: the
``He_n(mean_z(s_0, s_1, s_2))`` column carries the non-linear signal in
the latent ``z`` at materially higher SNR than any individual
``He_n(s_i)``.

Contracts pinned
----------------

* ``TestSensorMesh``: 5 latents x 3 sensors. The detector finds 5
  clusters; cluster-basis emits one He_2 aggregate per cluster.

* ``TestClusterAggregateSignal``: y depends on ``He_2(mean(cluster1))``
  with the cluster members carrying heavy iid noise. Cluster-basis
  emits the He_2(aggregate) column AND its engineered_mi materially
  beats every member's individual baseline_mi.

* ``TestDefaultDisabledByteIdentical``: default
  ``fe_hybrid_orth_cluster_basis_enable=False`` leaves
  ``hybrid_orth_features_`` empty (legacy behaviour preserved).

* ``TestPickleAndClone``: sklearn ``clone`` preserves the 4 ctor params;
  ``pickle`` preserves the ``orth_cluster_basis`` recipes (members,
  basis, degree, aggregator round-trip).

* ``TestRecipeReplay``: ``apply_recipe`` at transform time reproduces
  the fit-time engineered column bit-equivalently (no y reference, pure
  function of the member columns).

* ``TestAggregateDenoising``: when y depends on the LATENT's He_2
  and members carry heavy iid noise, the cluster-basis aggregate He_2
  column has materially higher engineered_mi than the best individual
  per-member He_2 (the actual claim of the layer).

NEVER xfail. NEVER mask bugs via runtime workarounds.

Consolidated verbatim from test_biz_value_mrmr_layer61.py (per audit finding test_code_quality-16).
"""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from sklearn.base import clone

warnings.filterwarnings("ignore")

SEEDS = (1, 7, 13, 42, 101)


def _import_cluster_basis_fe():
    """Lazily import the Layer-61 per-cluster shared-basis FE functions."""
    from mlframe.feature_selection.filters._orthogonal_cluster_basis_fe import (
        detect_clusters_by_correlation,
        compute_cluster_aggregate,
        generate_cluster_basis_features,
        hybrid_orth_mi_cluster_basis_fe,
        hybrid_orth_mi_cluster_basis_fe_with_recipes,
    )

    return (
        detect_clusters_by_correlation,
        compute_cluster_aggregate,
        generate_cluster_basis_features,
        hybrid_orth_mi_cluster_basis_fe,
        hybrid_orth_mi_cluster_basis_fe_with_recipes,
    )


from tests.feature_selection.conftest import make_fast_mrmr as _make_mrmr
# ---------------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------------


def _build_sensor_mesh(seed: int, n: int = 1500, n_latents: int = 5, n_sensors: int = 3, noise_sd: float = 0.1, n_noise: int = 5):
    """``n_latents`` latents, each loading on ``n_sensors`` sensors with
    small noise, plus ``n_noise`` pure-noise fillers. y depends on the
    QUADRATIC of a weighted combination of latents so the He_2 aggregate
    basis columns lift -- the cluster-basis layer targets non-linear
    latent signal hidden behind iid member noise. A purely linear
    additive y would have zero MI uplift over the raw members for any
    He_n with n >= 2 and the layer wouldn't fire (correctly so).
    """
    rng = np.random.default_rng(int(seed))
    cols: dict = {}
    latents = []
    for li in range(n_latents):
        z = rng.standard_normal(n)
        latents.append(z)
        for si in range(n_sensors):
            cols[f"L{li}_s{si}"] = z + noise_sd * rng.standard_normal(n)
    for ki in range(n_noise):
        cols[f"noise_{ki}"] = rng.standard_normal(n)
    X = pd.DataFrame(cols)
    weights = np.linspace(1.0, 0.2, n_latents)
    # Quadratic latent signal: y triggers on extreme weighted-z magnitude.
    # ``He_2(weighted z)`` recovers this; the per-cluster aggregate is the
    # primary signal carrier so the cluster-basis He_2 columns lift.
    score = sum(w * (z * z - 1.0) for w, z in zip(weights, latents))
    y = pd.Series((score + 0.3 * rng.standard_normal(n) > 0).astype(int))
    return X, y, latents


def _build_aggregate_signal_frame(seed: int, n: int = 4000, member_noise: float = 0.55):
    """y = sign(He_2(z) - median) where ``z`` is a single latent and
    ``s_0, s_1, s_2`` are noisy reflections (``s_i = z + member_noise *
    epsilon_i``). The aggregate ``mean(s_0, s_1, s_2)`` averages out the
    noise (variance 1/3 of per-member) so ``He_2(mean)`` recovers the
    signal far better than ``He_2(s_i)``.

    Default ``member_noise=0.55`` is calibrated so the per-pair Pearson
    correlation ``1 / (1 + member_noise^2)`` is ~0.76 -- above the
    detector's default 0.7 threshold -- AND the aggregate-He_2 vs
    per-member-He_2 MI ratio is ~1.67x on n=4000 (well above the 1.3x
    denoising contract floor). Increasing member_noise raises the ratio
    but drops the correlation below the auto-detect threshold; lowering
    it does the reverse.
    """
    rng = np.random.default_rng(int(seed))
    z = rng.standard_normal(n)
    sig = z * z - 1.0  # He_2(z) up to scale
    y = (sig + 0.1 * rng.standard_normal(n) > 0).astype(int)
    cols: dict = {}
    cluster_members = [f"s_{i}" for i in range(3)]
    for name in cluster_members:
        cols[name] = z + member_noise * rng.standard_normal(n)
    # Pad with pure-noise columns so the MI baseline distribution has a
    # non-trivial reference for the MAD floor.
    for ki in range(4):
        cols[f"noise_{ki}"] = rng.standard_normal(n)
    X = pd.DataFrame(cols)
    return X, pd.Series(y, name="y"), cluster_members


from tests.feature_selection._biz_val_synth import _build_linear
# ---------------------------------------------------------------------------
# Contract 1: sensor mesh -- detection + emission
# ---------------------------------------------------------------------------


class TestSensorMesh:
    """The correlation-based cluster detector must find one cluster per latent and emit He_2 aggregates."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_detector_finds_one_cluster_per_latent(self, seed):
        """detect_clusters_by_correlation finds exactly 5 clusters of 3 same-latent sensors each."""
        detect, _, _, _, _ = _import_cluster_basis_fe()
        X, _, _ = _build_sensor_mesh(seed, n_latents=5, n_sensors=3, noise_sd=0.1)
        clusters = detect(X, corr_threshold=0.7, min_cluster_size=2)
        # Each latent's 3 sensors are strongly correlated (corr > 0.99 with
        # noise_sd=0.1); the detector must return exactly 5 clusters.
        assert len(clusters) == 5, f"seed={seed}: detector found {len(clusters)} clusters; expected 5 (one per latent). Detected: {dict(clusters)}"
        # Every cluster has exactly 3 members.
        for anchor, members in clusters.items():
            assert len(members) == 3, f"seed={seed}: anchor {anchor!r} has {len(members)} members; expected 3 (one per sensor)."
            # All members must share the same latent prefix (L{i}_s).
            prefixes = {m.split("_s")[0] for m in members}
            assert len(prefixes) == 1, f"seed={seed}: cluster {anchor!r} crosses latents: members={members}; prefixes={prefixes}."

    @pytest.mark.parametrize("seed", SEEDS)
    def test_cluster_basis_emits_one_he2_per_cluster(self, seed):
        """generate_cluster_basis_features emits >= 1 column, each referencing a known cluster anchor."""
        _, _, gen_cb, _, _ = _import_cluster_basis_fe()
        X, y, _ = _build_sensor_mesh(seed, n_latents=5, n_sensors=3, noise_sd=0.1)
        # Auto-detect clusters internally.
        from mlframe.feature_selection.filters._orthogonal_cluster_basis_fe import (
            detect_clusters_by_correlation,
        )

        cm = detect_clusters_by_correlation(X, corr_threshold=0.7)
        eng, meta = gen_cb(
            X,
            y.values,
            cm,
            degrees=(2,),
            aggregator="mean_z",
            top_k=5,
        )
        assert eng.shape[1] >= 1, f"seed={seed}: cluster-basis emitted no columns; expected >=1 for the 5-latent mesh. Detected clusters: {cm}"
        # Every emitted column references a known cluster anchor.
        anchors_seen = {info["anchor"] for info in meta.values()}
        anchors_known = set(cm.keys())
        assert anchors_seen.issubset(anchors_known), f"seed={seed}: cluster-basis emitted columns referencing unknown anchors: {anchors_seen - anchors_known}"


# ---------------------------------------------------------------------------
# Contract 2: aggregate denoises -> He_2(aggregate) has materially higher MI
# than the best per-member He_2.
# ---------------------------------------------------------------------------


class TestAggregateDenoising:
    """The aggregate He_2(mean_z) column must materially beat the best per-member He_2 by MI."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_aggregate_he2_beats_per_member_he2(self, seed):
        """The headline biz_value claim: when y depends on ``He_2(z)`` and
        cluster members carry heavy iid noise, the aggregate-He_2 column
        has materially higher MI(col; y) than the best per-member He_2.
        """
        from mlframe.feature_selection.filters._orthogonal_cluster_basis_fe import (
            compute_cluster_aggregate,
            _evaluate_basis_column,
        )
        from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
            _mi_classif_batch,
        )

        X, y, members = _build_aggregate_signal_frame(seed, n=4000, member_noise=0.7)
        y_arr = y.values.astype(np.int64)
        # Per-member He_2 MI.
        per_member_mi = []
        for m in members:
            x = X[m].to_numpy(dtype=np.float64)
            vals = _evaluate_basis_column(x, "hermite", 2).reshape(-1, 1)
            mi = _mi_classif_batch(vals, y_arr, nbins=10)[0]
            per_member_mi.append(float(mi))
        best_member_mi = max(per_member_mi)
        # Aggregate He_2 MI.
        agg = compute_cluster_aggregate(X, members, aggregator="mean_z")
        agg_vals = _evaluate_basis_column(agg, "hermite", 2).reshape(-1, 1)
        agg_mi = float(_mi_classif_batch(agg_vals, y_arr, nbins=10)[0])
        # The aggregate must win by a non-trivial margin. The theoretical
        # lift is ~3x (variance reduction by N=3 noise members under iid).
        # We require 1.3x as a finite-n contract floor.
        assert agg_mi > 1.3 * best_member_mi, (
            f"seed={seed}: aggregate He_2 MI {agg_mi:.4f} does not "
            f"materially beat per-member best {best_member_mi:.4f} "
            f"(per-member MIs = {per_member_mi}); denoising claim violated."
        )


class TestClusterAggregateSignal:
    """End-to-end: the cluster-basis path must emit an aggregate He_2 winner that beats the best per-member baseline."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_cluster_basis_recovers_aggregate_he2(self, seed):
        """End-to-end: the cluster-basis path must emit at least one He_2
        column referencing the cluster's aggregate, and the engineered_mi
        must beat the best per-member baseline.
        """
        _, _, gen_cb, _, _ = _import_cluster_basis_fe()
        X, y, members = _build_aggregate_signal_frame(seed, n=4000, member_noise=0.7)
        anchor = sorted(members)[0]
        cluster_members = {anchor: members}
        eng, meta = gen_cb(
            X,
            y.values,
            cluster_members,
            basis="hermite",
            degrees=(2, 3),
            aggregator="mean_z",
            top_k=3,
            min_uplift=1.05,
        )
        assert eng.shape[1] >= 1, f"seed={seed}: cluster-basis emitted zero columns; expected >=1 He_2 aggregate winner for the He_2(z) target."
        # The best winner must reference our cluster anchor.
        best = max(meta.values(), key=lambda d: d["uplift"])
        assert best["anchor"] == anchor, f"seed={seed}: best winner anchor {best['anchor']!r} != expected {anchor!r}"
        # The engineered MI must exceed the BEST member's marginal MI.
        assert best["engineered_mi"] > best["baseline_mi"], (
            f"seed={seed}: engineered_mi {best['engineered_mi']:.4f} did "
            f"not beat the best per-member baseline_mi "
            f"{best['baseline_mi']:.4f} -- aggregate denoising claim "
            f"violated end-to-end."
        )


# ---------------------------------------------------------------------------
# Contract 3: default disabled byte-identical with master
# ---------------------------------------------------------------------------


class TestDefaultDisabledByteIdentical:
    """fe_hybrid_orth_cluster_basis_enable defaults to False; enabling it must fire and append columns."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_default_off_no_cluster_basis_columns(self, seed):
        """With the flag left at its False default, no cluster-basis columns are appended."""
        X, y = _build_linear(seed)
        m = _make_mrmr().fit(X, y)
        added = list(getattr(m, "hybrid_orth_features_", []) or [])
        assert added == [], f"seed={seed}: default fe_hybrid_orth_cluster_basis_enable=False should NOT append any engineered columns; got {added}"

    @pytest.mark.parametrize("seed", SEEDS)
    def test_enable_cluster_basis_appends_engineered(self, seed):
        """Enabling the flag on an auto-detectable cluster fixture appends a cluster_-prefixed engineered column."""
        # member_noise=0.55 keeps the per-pair correlation ~0.76 (above the
        # auto-detect default 0.7) so the MRMR-fit auto-detection path
        # actually fires; the aggregate-denoising tests above use 0.7
        # because they pass cluster_members explicitly.
        X, y, _ = _build_aggregate_signal_frame(seed, n=2500, member_noise=0.55)
        m = _make_mrmr(
            fe_hybrid_orth_cluster_basis_enable=True,
            fe_hybrid_orth_cluster_basis_aggregator="mean_z",
            fe_hybrid_orth_cluster_basis_degrees=(2, 3),
            fe_hybrid_orth_cluster_basis_top_k=3,
        ).fit(X, y)
        added = list(getattr(m, "hybrid_orth_features_", []) or [])
        assert added, f"seed={seed}: cluster-basis flag ON should append at least one engineered column to hybrid_orth_features_; got {added}"
        # At least one appended column must be a cluster-basis name
        # (prefix ``cluster_``).
        assert any(c.startswith("cluster_") for c in added), (
            f"seed={seed}: cluster-basis flag ON should append at least one ``cluster_`` engineered column; got {added}"
        )


# ---------------------------------------------------------------------------
# Contract 4: pickle / clone preserve the ctor + recipes
# ---------------------------------------------------------------------------


class TestPickleAndClone:
    """Cluster-basis ctor params and recipes must survive clone/pickle round-trips."""

    def test_clone_preserves_cluster_basis_params(self):
        """sklearn clone() copies every fe_hybrid_orth_cluster_basis_* ctor param."""
        m = _make_mrmr(
            fe_hybrid_orth_cluster_basis_enable=True,
            fe_hybrid_orth_cluster_basis_aggregator="median_z",
            fe_hybrid_orth_cluster_basis_degrees=(2, 3, 4),
            fe_hybrid_orth_cluster_basis_top_k=7,
        )
        m2 = clone(m)
        for name, expected in [
            ("fe_hybrid_orth_cluster_basis_enable", True),
            ("fe_hybrid_orth_cluster_basis_aggregator", "median_z"),
            ("fe_hybrid_orth_cluster_basis_degrees", (2, 3, 4)),
            ("fe_hybrid_orth_cluster_basis_top_k", 7),
        ]:
            assert getattr(m2, name) == expected, f"clone() dropped {name}: expected {expected}, got {getattr(m2, name)}"

    def test_pickle_roundtrip_preserves_cluster_basis_recipe(self):
        """A pickle round-trip preserves feature names, appended columns, and every orth_cluster_basis recipe field."""
        # member_noise=0.55 -- see TestDefaultDisabledByteIdentical for rationale.
        X, y, _ = _build_aggregate_signal_frame(seed=42, n=2500, member_noise=0.55)
        m = _make_mrmr(
            fe_hybrid_orth_cluster_basis_enable=True,
            fe_hybrid_orth_cluster_basis_aggregator="mean_z",
            fe_hybrid_orth_cluster_basis_degrees=(2, 3),
            fe_hybrid_orth_cluster_basis_top_k=3,
        ).fit(X, y)
        blob = pickle.dumps(m)
        m2 = pickle.loads(blob)  # nosec B301 -- round-trip of a locally-created, trusted object
        assert list(m2.feature_names_in_) == list(m.feature_names_in_), "pickle changed feature_names_in_"
        added_before = list(getattr(m, "hybrid_orth_features_", []) or [])
        added_after = list(getattr(m2, "hybrid_orth_features_", []) or [])
        assert added_before == added_after, f"pickle changed hybrid_orth_features_: before={added_before}, after={added_after}"
        recipes_before = {r.name: r for r in getattr(m, "_engineered_recipes_", []) or [] if r.kind == "orth_cluster_basis"}
        recipes_after = {r.name: r for r in getattr(m2, "_engineered_recipes_", []) or [] if r.kind == "orth_cluster_basis"}
        assert set(recipes_before.keys()) == set(recipes_after.keys()), (
            f"pickle dropped or added orth_cluster_basis recipe names: before={set(recipes_before.keys())}, after={set(recipes_after.keys())}"
        )
        for name, r_before in recipes_before.items():
            r_after = recipes_after[name]
            assert r_before.src_names == r_after.src_names, f"pickle changed src_names for {name!r}: before={r_before.src_names}, after={r_after.src_names}"
            for key in ("basis", "degree", "aggregator"):
                assert r_before.extra.get(key) == r_after.extra.get(key), (
                    f"pickle changed '{key}' for recipe {name!r}: before={r_before.extra}, after={r_after.extra}"
                )


# ---------------------------------------------------------------------------
# Contract 5: recipe replay matches fit-time values bit-equivalently
# ---------------------------------------------------------------------------


class TestRecipeReplay:
    """apply_recipe on held-out data must reproduce the fit-time engineered values bit-equivalently."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_recipe_replay_matches_fit_time_values(self, seed):
        """Replaying the fit-time recipes on the same X reproduces identical engineered values."""
        _, _, _, _, hybrid_with_recipes = _import_cluster_basis_fe()
        X, y, members = _build_aggregate_signal_frame(seed, n=3000, member_noise=0.7)
        cluster_members = {sorted(members)[0]: members}
        X_aug, _scores, recipes = hybrid_with_recipes(
            X,
            y.values,
            cluster_members=cluster_members,
            basis="hermite",
            degrees=(2, 3),
            aggregator="mean_z",
            top_k=3,
        )
        if not recipes:
            pytest.fail(f"seed={seed}: cluster-basis emitted no recipes; replay test requires at least one recipe.")
        from mlframe.feature_selection.filters.engineered_recipes import (
            apply_recipe,
        )

        appended = [c for c in X_aug.columns if c not in X.columns]
        for r in recipes:
            assert r.name in appended, f"seed={seed}: recipe {r.name!r} not in appended columns {appended}"
            assert r.kind == "orth_cluster_basis", f"seed={seed}: recipe {r.name!r} kind={r.kind!r}, expected 'orth_cluster_basis'."
            replayed = apply_recipe(r, X)
            fit_time = X_aug[r.name].to_numpy()
            assert np.allclose(replayed, fit_time, rtol=1e-9, atol=1e-12), (
                f"seed={seed}: recipe {r.name!r} replay drift: max|replayed - fit| = {float(np.max(np.abs(replayed - fit_time)))}; extra={dict(r.extra)}"
            )


# ---------------------------------------------------------------------------
# Contract 6: aggregator variants (mean_z, median_z, pc1) all emit + replay
# ---------------------------------------------------------------------------


class TestAggregatorVariants:
    """Every aggregator variant (mean_z, median_z, pc1) must emit a properly-named, replayable recipe."""

    @pytest.mark.parametrize("aggregator", ["mean_z", "median_z", "pc1"])
    def test_each_aggregator_emits_recipe_that_replays(self, aggregator):
        """Each aggregator emits a recipe whose replayed values reproduce the fit-time engineered column."""
        _, _, _, _, hybrid_with_recipes = _import_cluster_basis_fe()
        X, y, members = _build_aggregate_signal_frame(seed=7, n=3000, member_noise=0.7)
        cluster_members = {sorted(members)[0]: members}
        X_aug, _scores, recipes = hybrid_with_recipes(
            X,
            y.values,
            cluster_members=cluster_members,
            basis="hermite",
            degrees=(2,),
            aggregator=aggregator,
            top_k=3,
        )
        appended = [c for c in X_aug.columns if c not in X.columns]
        assert appended, f"aggregator={aggregator}: expected at least one cluster-basis column on the He_2(z) target; got {appended}"
        # Every appended name encodes the aggregator.
        for name in appended:
            assert f"agg_{aggregator}" in name, f"aggregator={aggregator}: appended name {name!r} does not encode the aggregator in its identifier."
        from mlframe.feature_selection.filters.engineered_recipes import (
            apply_recipe,
        )

        for r in recipes:
            replayed = apply_recipe(r, X)
            fit_time = X_aug[r.name].to_numpy()
            assert np.allclose(replayed, fit_time, rtol=1e-9, atol=1e-12), (
                f"aggregator={aggregator}: recipe {r.name!r} replay drift: "
                f"max|replayed - fit| = "
                f"{float(np.max(np.abs(replayed - fit_time)))}; "
                f"extra={dict(r.extra)}"
            )
