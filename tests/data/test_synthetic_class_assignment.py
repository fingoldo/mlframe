import numpy as np

from mlframe.data.synthetic import assign_classes_from_probability, generate_modelling_data


def _assign_classes_python_reference(predictors, draw, n_classes, out):
    """The exact pre-njit per-row Python loop, kept here as the bit-identity oracle."""
    n_samples = predictors.shape[0]
    for i in range(n_samples):
        total = 0.0  # numpy promotion keeps `total` float32 as legacy did; matched by the njit kernel's float32 accumulator
        out[i] = n_classes - 1
        for j in range(n_classes):
            total += predictors[i, j]
            if draw[i] < total:
                out[i] = j
                break
    return out


def test_assign_classes_njit_bit_identical_to_python_loop():
    """The njit kernel must reproduce the legacy float32-accumulating Python loop EXACTLY (not approximately):
    a float64 accumulator flips the chosen class on draws landing within ~1e-8 of a cumulative boundary. Sweeps
    several shapes and includes the boundary-sensitive (n=1e6, 3-class) regime that surfaced the divergence."""
    for n_classes in (3, 8):
        for n_samples in (10_000, 1_000_000):
            rng = np.random.default_rng(n_samples + n_classes)
            predictors = rng.random((n_samples, n_classes)).astype(np.float32)
            predictors /= predictors.sum(axis=1, keepdims=True)
            draw = rng.random(n_samples)

            ref = _assign_classes_python_reference(predictors, draw, n_classes, np.empty(n_samples, dtype=np.int32))
            got = assign_classes_from_probability(predictors, draw, n_classes, out=np.empty(n_samples, dtype=np.int32))
            assert np.array_equal(ref, got), f"njit kernel diverged from Python loop at n={n_samples}, n_classes={n_classes}"


def test_assign_classes_fallthrough_row_gets_last_class_not_garbage():
    """A row whose cumulative predictor total never exceeds the draw (float32 rounding, or a degenerate
    all-zero row) must fall back to the last class. Pre-fix the loop left ``out[i]`` holding uninitialized
    np.empty garbage for exactly these rows. ``out`` is seeded with an out-of-range sentinel so a skipped
    assignment is caught instead of coincidentally matching a valid label."""
    n_classes = 3
    predictors = np.array(
        [
            [0.2, 0.3, 0.5],   # normal: total reaches 1.0
            [0.0, 0.0, 0.0],   # degenerate all-zero row -> never exceeds draw -> fallthrough
            [0.3, 0.3, 0.3],   # float32-style under-1.0 sum, draw above it -> fallthrough
        ],
        dtype=np.float32,
    )
    draw = np.array([0.99, 0.5, 0.95], dtype=np.float64)

    out = np.full(n_classes, -999, dtype=np.int32)
    assign_classes_from_probability(predictors, draw, n_classes, out=out)

    assert out.min() >= 0, f"label below 0 -> uninitialized garbage left in out: {out}"
    assert out.max() <= n_classes - 1, f"label above n_classes-1 -> garbage: {out}"
    assert out[1] == n_classes - 1, "degenerate all-zero row must fall back to the last class"
    assert out[2] == n_classes - 1, "under-1.0-sum fallthrough row must fall back to the last class"
    assert out[0] == 2, "draw 0.99 lands in the last bucket of [0.2,0.3,0.5]"


def test_generate_modelling_data_labels_in_range_smoke():
    """End-to-end smoke: labels stay in range. Fast distributions + minimal feature kinds keep it well
    under the time budget (some scipy continuous dists, e.g. noncentral-t, are pathologically slow)."""
    n_classes = 3
    fast_dists = {"norm", "uniform", "expon", "laplace", "logistic"}
    for seed in range(20):
        X, y, fnames = generate_modelling_data(
            n_samples=400,
            n_classes=n_classes,
            n_informative=n_classes,
            n_singly_correlated=0,
            n_mutually_correlated=0,
            n_unrelated_single=0,
            n_unrelated_intercorrelated=0,
            n_repeated=0,
            include_distributions=fast_dists,
            flip_y=0.0,
            shuffle=False,
            return_dataframe=False,
            random_state=seed,
        )
        assert y.min() >= 0 and y.max() <= n_classes - 1, f"seed {seed}: label out of range {y.min()}..{y.max()}"


def _generate(feature_noise=0.0, timeseries=False, max_cardinality=None, seed=0):
    fast_dists = {"norm", "uniform", "expon", "laplace", "logistic"}
    return generate_modelling_data(
        n_samples=2000,
        n_classes=3,
        n_informative=3,
        n_singly_correlated=2,
        n_mutually_correlated=1,
        n_unrelated_single=2,
        n_unrelated_intercorrelated=0,
        n_repeated=0,
        include_distributions=fast_dists,
        flip_y=0.0,
        shuffle=False,
        return_dataframe=False,
        random_state=seed,
        feature_noise=feature_noise,
        timeseries=timeseries,
        max_cardinality=max_cardinality,
    )


def test_feature_noise_perturbs_correlated_features():
    """feature_noise=0 vs feature_noise>0 must produce DIFFERENT correlated-feature values (same seed);
    pre-fix the parameter was accepted but silently ignored (bit-identical output regardless of its value)."""
    X_clean, _, _ = _generate(feature_noise=0.0)
    X_noisy, _, _ = _generate(feature_noise=0.5)
    assert not np.allclose(X_clean, X_noisy), "feature_noise had no effect on generated features"


def test_timeseries_drifts_correlated_feature_dependence():
    """timeseries=True must make correlated features differ from the stationary (timeseries=False) generation."""
    X_stationary, _, _ = _generate(timeseries=False)
    X_drifting, _, _ = _generate(timeseries=True)
    assert not np.allclose(X_stationary, X_drifting), "timeseries=True had no effect on generated features"


def test_max_cardinality_bounds_unrelated_single_feature_cardinality():
    """When max_cardinality is set, unrelated_single feature columns must be discretized to at most that many
    distinct values (pre-fix the raw continuous/discrete draw's natural, unbounded cardinality was kept)."""
    X, _, fnames = _generate(max_cardinality=4, seed=1)
    unrelated_cols = [i for i, n in enumerate(fnames) if n.startswith("unr_")]
    assert unrelated_cols, "no unrelated_single columns generated"
    for i in unrelated_cols:
        assert len(np.unique(X[:, i])) <= 4, f"column {fnames[i]} exceeds max_cardinality=4"
