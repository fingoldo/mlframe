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
