"""biz_value tests for the Param-Oracle adaptive-dispatch / L2O cache.

These tests prove the quantitative wins claimed in
``mlframe.utils._param_oracle``:

* benchmark mode records every param combo with its objective,
* inference recommends the empirically-best combo on a fresh-but-similar
  fingerprint,
* the on-disk store is STAT-ONLY (no raw arrays ever persisted),
* cold-start never crashes and falls back to the caller default,
* log-scale fingerprint bucketing is size-stable (n=1000 ~ n=1050),
* k-NN fallback recommends a near-neighbour's best when no exact match,
* concurrent writers do not corrupt the store,
* hybrid epsilon-greedy mostly exploits but sometimes explores,
* a REAL consumer demo: oracle around the MI-scorer choice recommends
  ``cmim`` on a redundant-signal fingerprint (the L99 FE-recommender
  foundation).
"""

from __future__ import annotations

import os
import random

import orjson

import numpy as np

from mlframe.utils._param_oracle import (
    ParamOracle,
    bucketize_fingerprint,
    default_fingerprint,
    log_bucket,
)

FIXED_TS = "2026-01-01T00:00:00+00:00"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _store(tmp_path, name="oracle.parquet"):
    return os.path.join(str(tmp_path), name)


def _make_dataset(n, p, seed=0, redundant=False):
    rng = np.random.default_rng(seed)
    if redundant:
        base = rng.standard_normal((n, 1))
        # All columns are near-copies of one signal -> high mean_abs_corr.
        X = base + 0.01 * rng.standard_normal((n, p))
    else:
        X = rng.standard_normal((n, p))
    return X


# ---------------------------------------------------------------------------
# 1. benchmark mode records all combos x objective
# ---------------------------------------------------------------------------


def test_benchmark_mode_records_all_combos(tmp_path):
    X = _make_dataset(1000, 4, seed=1)

    def toy(x, alpha=1, beta=1):
        return float((x * alpha + beta).sum())

    space = {"alpha": [1, 2], "beta": [0, 10]}
    oracle = ParamOracle(_store(tmp_path), param_space=space, mode="benchmark", minimize="elapsed_s")
    res = oracle.benchmark(toy, (X,), {}, ts=FIXED_TS)

    assert len(res) == 4  # 2 x 2 combos
    rows = oracle.store.read_rows()
    combos = {r["param_combo_json"] for r in rows}
    # orjson with OPT_SORT_KEYS produces byte-identical output to
    # ``json.dumps(..., sort_keys=True, separators=(",", ":"))`` for
    # plain-dict primitives -- compact form + sorted keys is orjson's
    # default. Production storage in ``_param_oracle.py`` still uses
    # stdlib json (it's prod code, not test code, so the no-stdlib-json
    # meta-linter doesn't cover it); the test-side comparison is a
    # string equality check, hence byte-identity matters.
    expected = {orjson.dumps({"alpha": a, "beta": b}, option=orjson.OPT_SORT_KEYS).decode() for a in (1, 2) for b in (0, 10)}
    assert combos == expected
    # Every row carries an objective with the optimised metric.
    for r in rows:
        obj = orjson.loads(r["objective_json"])
        assert "elapsed_s" in obj


# ---------------------------------------------------------------------------
# 2. inference recommends the best combo on a similar fingerprint
# ---------------------------------------------------------------------------


def test_inference_recommends_best_on_similar_fingerprint(tmp_path):
    X = _make_dataset(1000, 4, seed=2)
    fp = default_fingerprint((X,), {})

    # Hand-record so combo "fast" is clearly cheaper than "slow", with
    # enough observations to clear the confidence gate.
    space = {"impl": ["fast", "slow"]}
    oracle = ParamOracle(_store(tmp_path), param_space=space, mode="inference", minimize="elapsed_s", min_observations=3)
    for _ in range(4):
        oracle.record(fp, {"impl": "fast"}, {"elapsed_s": 0.001}, ts=FIXED_TS, fn_name="job")
        oracle.record(fp, {"impl": "slow"}, {"elapsed_s": 0.100}, ts=FIXED_TS, fn_name="job")

    # Fresh-but-similar dataset (same shape, different seed -> same bucket).
    X2 = _make_dataset(1050, 4, seed=99)
    fp2 = default_fingerprint((X2,), {})
    rec = oracle.recommend(fp2, fn_name="job")
    assert rec == {"impl": "fast"}


# ---------------------------------------------------------------------------
# 3. stat-only persistence -- NO raw array / DataFrame on disk
# ---------------------------------------------------------------------------


def test_persistence_is_stat_only(tmp_path):
    X = _make_dataset(2000, 6, seed=3)

    def toy(x, k=1):
        return float((x**k).mean())

    oracle = ParamOracle(_store(tmp_path), param_space={"k": [1, 2]}, mode="benchmark", minimize="elapsed_s")
    oracle.benchmark(toy, (X,), {}, ts=FIXED_TS)

    rows = oracle.store.read_rows()
    # Every persisted value must be a scalar or a JSON string of scalars.
    for r in rows:
        for col, val in r.items():
            assert not isinstance(val, (list, tuple, dict, np.ndarray)), f"non-scalar persisted in {col}: {type(val)}"
        fp_bucket = orjson.loads(r["fp_bucket_json"])
        for v in fp_bucket.values():
            assert isinstance(v, (int, float, str)), f"non-scalar in fp_bucket: {v!r}"
    # The raw data magnitude must NOT be reconstructable: the store size is
    # tiny relative to the 2000x6 float64 array (96 KB).
    store_bytes = os.path.getsize(oracle.store._path)
    assert store_bytes < X.nbytes, "store larger than raw array -> likely leaking data"


# ---------------------------------------------------------------------------
# 4. cold-start fallback -> caller default, never crashes
# ---------------------------------------------------------------------------


def test_cold_start_returns_caller_default(tmp_path):
    space = {"impl": ["a", "b", "c"]}
    oracle = ParamOracle(_store(tmp_path), param_space=space, mode="inference", minimize="elapsed_s")
    fp = default_fingerprint((_make_dataset(500, 3),), {})
    rec = oracle.recommend(fp, fn_name="never_seen")
    assert rec == {"impl": "a"}  # first combo = caller default


def test_cold_start_empty_args_no_crash(tmp_path):
    oracle = ParamOracle(_store(tmp_path), param_space={"x": [1]}, mode="inference")
    # Fingerprint of nothing array-like must not crash.
    fp = default_fingerprint((), {})
    assert fp["n"] == 0
    assert oracle.recommend(fp, fn_name="z") == {"x": 1}


# ---------------------------------------------------------------------------
# 5. fingerprint bucketing is size-stable
# ---------------------------------------------------------------------------


def test_fingerprint_bucketing_size_stable(tmp_path):
    X1 = _make_dataset(1000, 5, seed=10)
    X2 = _make_dataset(1050, 5, seed=11)
    b1 = bucketize_fingerprint(default_fingerprint((X1,), {}))
    b2 = bucketize_fingerprint(default_fingerprint((X2,), {}))
    assert b1["n"] == b2["n"], "n=1000 and n=1050 must share a log-bucket"
    assert log_bucket(1000) == log_bucket(1050) == 3.0


# ---------------------------------------------------------------------------
# 6. k-NN fallback -> recommend the neighbour's best
# ---------------------------------------------------------------------------


def test_knn_fallback_recommends_neighbor_best(tmp_path):
    # Benchmark on a SMALL dataset (n~100 -> bucket 2.0).
    fp_small = default_fingerprint((_make_dataset(100, 4, seed=20),), {})
    oracle = ParamOracle(_store(tmp_path), param_space={"impl": ["x", "y"]}, mode="inference", minimize="elapsed_s", min_observations=3)
    for _ in range(4):
        oracle.record(fp_small, {"impl": "y"}, {"elapsed_s": 0.001}, ts=FIXED_TS, fn_name="job")
        oracle.record(fp_small, {"impl": "x"}, {"elapsed_s": 0.050}, ts=FIXED_TS, fn_name="job")

    # Query a MEDIUM dataset (n~300 -> bucket 2.5): no exact match, but the
    # nearest neighbour in fingerprint space is the small one -> recommend y.
    fp_med = default_fingerprint((_make_dataset(300, 4, seed=21),), {})
    assert bucketize_fingerprint(fp_med)["n"] != bucketize_fingerprint(fp_small)["n"]
    rec = oracle.recommend(fp_med, fn_name="job")
    assert rec == {"impl": "y"}


# ---------------------------------------------------------------------------
# 7. concurrency -- two oracles writing same store don't corrupt it
# ---------------------------------------------------------------------------


def test_concurrent_writers_do_not_corrupt(tmp_path):
    path = _store(tmp_path)
    fp = default_fingerprint((_make_dataset(800, 3, seed=30),), {})
    space = {"impl": ["p", "q"]}
    o1 = ParamOracle(path, param_space=space, mode="inference")
    o2 = ParamOracle(path, param_space=space, mode="inference")

    # Simulate interleaved parallel shards writing the same store.
    for _i in range(5):
        o1.record(fp, {"impl": "p"}, {"elapsed_s": 0.01}, ts=FIXED_TS, fn_name="job")
        o2.record(fp, {"impl": "q"}, {"elapsed_s": 0.02}, ts=FIXED_TS, fn_name="job")

    rows = o1.store.read_rows()
    # After aggregation: one row per (combo) -- both writers' combos survive.
    combos = {orjson.loads(r["param_combo_json"])["impl"] for r in rows}
    assert combos == {"p", "q"}
    # Observation counts preserved (5 each, median-aggregated into 1 row each).
    by_combo = {orjson.loads(r["param_combo_json"])["impl"]: r for r in rows}
    assert by_combo["p"]["n_obs"] == 5
    assert by_combo["q"]["n_obs"] == 5
    # Store still parses (not corrupt).
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# 8. hybrid epsilon-greedy -- mostly exploit, occasionally explore
# ---------------------------------------------------------------------------


def test_hybrid_epsilon_greedy_explore_vs_exploit(tmp_path):
    X = _make_dataset(1000, 4, seed=40)
    fp = default_fingerprint((X,), {})

    calls = {"count": 0}

    def toy(x, impl="a"):
        calls["count"] += 1
        return impl

    space = {"impl": ["a", "b"]}
    # Seed a clear best so exploit is well-defined.
    oracle = ParamOracle(_store(tmp_path), param_space=space, mode="hybrid", minimize="elapsed_s", epsilon=0.3, min_observations=1, rng=random.Random(12345))  # nosec B311 -- deterministic-seed test PRNG, not used for security/crypto purposes
    # Pre-seed: "a" is the cheap best.
    for _ in range(3):
        oracle.record(fp, {"impl": "a"}, {"elapsed_s": 0.001}, ts=FIXED_TS, fn_name="toy")
        oracle.record(fp, {"impl": "b"}, {"elapsed_s": 0.500}, ts=FIXED_TS, fn_name="toy")

    decorated = oracle(toy)
    actions = []
    for _ in range(200):
        decorated(X)
        actions.append(decorated._last_action)

    n_explore = actions.count("explore")
    n_exploit = actions.count("exploit")
    assert n_explore + n_exploit == 200
    # epsilon=0.3 -> expect ~60 explore, ~140 exploit. Loose bounds.
    assert n_exploit > n_explore, "hybrid must mostly exploit"
    assert n_explore > 0, "hybrid must sometimes explore"
    assert 0.15 < n_explore / 200 < 0.45


# ---------------------------------------------------------------------------
# 9. REAL consumer demo: MI-scorer selection (plug_in vs cmim)
# ---------------------------------------------------------------------------


def test_real_consumer_mi_scorer_selection(tmp_path):
    """Wire ParamOracle around the MI-scorer choice and prove it learns to
    recommend ``cmim`` on a redundant-signal fingerprint (per L83: CMIM is
    the conditional/redundancy-aware mechanism). This is the L99 Meta
    FE-recommender foundation in miniature.

    We define the objective as a *quality* score (maximize), where on a
    redundant dataset ``cmim`` (redundancy-aware) earns a higher quality
    than ``plug_in`` (marginal). The oracle should then recommend ``cmim``
    for a fresh redundant fingerprint, and ``plug_in`` for a clean one.
    """
    redundant = _make_dataset(2000, 8, seed=50, redundant=True)
    clean = _make_dataset(2000, 8, seed=51, redundant=False)

    fp_redundant = default_fingerprint((redundant,), {})
    fp_clean = default_fingerprint((clean,), {})

    # Sanity: the fingerprint actually distinguishes redundancy.
    assert fp_redundant["mean_abs_corr"] > 0.5
    assert fp_clean["mean_abs_corr"] < 0.2

    def quality_objective(output, elapsed_s, rss_delta_mb):
        # output is the (scorer_name, simulated_quality) tuple.
        _scorer, q = output
        return {"quality": float(q), "elapsed_s": float(elapsed_s)}

    space = {"scorer": ["plug_in", "cmim"]}
    oracle = ParamOracle(
        _store(tmp_path),
        objective_fn=quality_objective,
        param_space=space,
        mode="benchmark",
        maximize="quality",
        min_observations=1,
    )

    def select_features(x, scorer="plug_in"):
        # Simulate: cmim shines when redundancy is high (it conditions on
        # already-selected features); plug_in is fine on clean signals.
        fp = default_fingerprint((x,), {})
        redundancy = fp["mean_abs_corr"]
        if scorer == "cmim":
            q = 0.5 + 0.4 * redundancy  # rewarded for handling redundancy
        else:  # plug_in
            q = 0.7 - 0.3 * redundancy  # degrades as redundancy grows
        return (scorer, q)

    # Benchmark on BOTH dataset types so the oracle sees both fingerprints.
    oracle.benchmark(select_features, (redundant,), {}, ts=FIXED_TS)
    oracle.benchmark(select_features, (clean,), {}, ts=FIXED_TS)

    # Fresh redundant dataset -> recommend cmim.
    fresh_redundant = _make_dataset(2100, 8, seed=77, redundant=True)
    rec_r = oracle.recommend(default_fingerprint((fresh_redundant,), {}), fn_name="select_features")
    assert rec_r == {"scorer": "cmim"}, f"expected cmim on redundant, got {rec_r}"

    # Fresh clean dataset -> recommend plug_in.
    fresh_clean = _make_dataset(2100, 8, seed=78, redundant=False)
    rec_c = oracle.recommend(default_fingerprint((fresh_clean,), {}), fn_name="select_features")
    assert rec_c == {"scorer": "plug_in"}, f"expected plug_in on clean, got {rec_c}"
