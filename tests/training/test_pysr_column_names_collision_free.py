"""Regression: PySR equation column names must be content-hashed so distinct seeds emit distinct columns, and the equation -> column-name mapping must round-trip via ``out_pysr_equations`` for predict-time replay.

Pre-fix the column name was ``pysr_eq{idx}`` where ``idx`` is the row position in ``model.equations_``. Two seeds discovering different equations both landed on ``pysr_eq0`` / ``pysr_eq1`` / ..., so ensembling or cross-run model loading silently overlaid different equations onto the same column slot.
"""

from __future__ import annotations

import hashlib
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from mlframe.training.configs import PreprocessingExtensionsConfig
from mlframe.training.pipeline import apply_preprocessing_extensions


def _make_frames(n: int = 200, seed: int = 0):
    """Make frames."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "x1": rng.standard_normal(n).astype(np.float32),
            "x2": rng.standard_normal(n).astype(np.float32),
        }
    )
    y = (X["x1"] * X["x2"]).astype(np.float32).values
    return X.copy(), X.iloc[: n // 4].copy(), X.iloc[n // 4 : n // 2].copy(), y


def _build_fake_model(equation_strings):
    """Build fake model."""
    fake_model = MagicMock()
    eq_df = pd.DataFrame(
        {
            "equation": list(equation_strings),
            "score": np.linspace(1.0, 0.1, len(equation_strings), dtype=np.float64),
            "complexity": [3] * len(equation_strings),
            "loss": [0.1] * len(equation_strings),
        }
    )
    eq_df.index = list(range(len(equation_strings)))
    fake_model.equations_ = eq_df
    fake_model.predict = MagicMock(side_effect=lambda df, index=0: np.zeros(len(df), dtype=np.float32))
    return fake_model


def _run(seed: int, equations: list[str]):
    """Runs the PySR preprocessing extension with the given seed and fake discovered equations, returning the transformed frame."""
    X_train, X_val, X_test, y = _make_frames()
    cfg = PreprocessingExtensionsConfig(
        pysr_enabled=True,
        pysr_params={"niterations": 1, "population_size": 5},
        random_seed=seed,
    )
    fake_model = _build_fake_model(equations)

    def fake_run_pysr(df, target_col, sample_size, encode_categoricals, verbose, pysr_params_override, **kwargs):
        """Fake run pysr."""
        return fake_model

    out_equations: dict[str, str] = {}
    with patch(
        "mlframe.feature_engineering.bruteforce.run_pysr_feature_engineering",
        side_effect=fake_run_pysr,
    ):
        train_out, *_ = apply_preprocessing_extensions(
            train_df=X_train,
            val_df=X_val,
            test_df=X_test,
            config=cfg,
            verbose=0,
            y_train=np.asarray(y),
            out_pysr_equations=out_equations,
        )
    return train_out, out_equations


def test_pysr_seeds_with_different_equations_produce_distinct_column_names():
    """Two seeds discovering DIFFERENT equations must NOT collide on a shared ``pysr_eq{idx}`` slot."""
    train_a, eq_map_a = _run(seed=42, equations=["x1 * x2", "x1 + x2"])
    train_b, eq_map_b = _run(seed=43, equations=["sin(x1)", "cos(x2)"])

    pysr_cols_a = [c for c in train_a.columns if c.startswith("pysr__")]
    pysr_cols_b = [c for c in train_b.columns if c.startswith("pysr__")]
    assert pysr_cols_a, "seed=42 produced no pysr columns"
    assert pysr_cols_b, "seed=43 produced no pysr columns"

    # Distinct equations across seeds -> NO overlap in column names. Pre-fix both would land on ``pysr_eq0`` / ``pysr_eq1``.
    assert set(pysr_cols_a).isdisjoint(pysr_cols_b), f"PySR column-name collision across seeds: {set(pysr_cols_a) & set(pysr_cols_b)}"

    # Equation map populated for both runs and keys match emitted column names.
    assert set(eq_map_a.keys()) == set(pysr_cols_a)
    assert set(eq_map_b.keys()) == set(pysr_cols_b)


def test_pysr_same_equation_different_seeds_emits_different_columns():
    """Even when the SAME equation is rediscovered under a different seed, the column name carries the seed suffix so two seeds in an ensemble keep separately addressable features."""
    train_a, _eq_map_a = _run(seed=42, equations=["x1 * x2"])
    train_b, _eq_map_b = _run(seed=43, equations=["x1 * x2"])

    cols_a = [c for c in train_a.columns if c.startswith("pysr__")]
    cols_b = [c for c in train_b.columns if c.startswith("pysr__")]
    assert len(cols_a) == 1 and len(cols_b) == 1
    assert cols_a != cols_b, "Same equation across two seeds collapsed to one column name; ensembling would alias features."
    # Hash part identical, seed suffix differs.
    assert cols_a[0].split("__")[1] == cols_b[0].split("__")[1]
    assert cols_a[0].split("__")[2] == "42"
    assert cols_b[0].split("__")[2] == "43"


def test_pysr_column_name_is_deterministic_blake2b_of_equation():
    """The hash component must be ``blake2b(equation_str, digest_size=4)`` so post-load callers can recover the column name from the persisted equation string alone."""
    _train, eq_map = _run(seed=7, equations=["x1 * x2 + 1"])
    (col_name,) = list(eq_map.keys())
    equation_str = eq_map[col_name]
    expected_hash8 = hashlib.blake2b(equation_str.encode("utf-8"), digest_size=4).hexdigest()
    assert col_name == f"pysr__{expected_hash8}__7"


def test_pysr_equation_map_round_trip_recovers_column_names():
    """Persisting eq_map and feeding it back must recover identical column names. This is the predict-time replay contract."""
    _train, eq_map = _run(seed=11, equations=["x1 + x2", "x1 * x2"])
    # Simulate a save/load roundtrip via dict copy.
    saved = dict(eq_map)
    for col_name, equation_str in saved.items():
        recomputed_hash = hashlib.blake2b(equation_str.encode("utf-8"), digest_size=4).hexdigest()
        assert col_name == f"pysr__{recomputed_hash}__11"
