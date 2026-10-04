"""Wave 61 (2026-05-20): mixed-type comparison TypeError.

Audit class: < / > / sorted() / min / max on values whose types might mix
(str vs int, None vs float, object-dtype label set). Python 3 raises
TypeError: '<' not supported between instances of '<type-a>' and '<type-b>'
that downstream broad-except blocks then mask as something unrelated.

2 P1 + 3 P2 fixes applied via uniform "str-key fallback" pattern:

  P1:
    1. training/feature_handling/fingerprint.py:302 (compute_content_fingerprint)
       sorted(cols) failed on heterogeneous pandas column labels
       ([0, "a", 1] from stitched join/pivot). Now sorted(cols, key=str).

    2. training/core/_phase_polars_fixes.py:207 (union sort for Enum dtype)
       sorted(union) raised TypeError when "__MISSING__" sentinel was added
       to an int-encoded categorical's value set; the broad except swallowed
       it, cat-alignment was silently skipped, then XGB/CB crashed later
       with a misleading "unseen category" error. Now sorted(union, key=str).

  P2 (defensive -- works today on homogeneous inputs but lacks dtype guard):
    3. training/neural/base.py:320 (classes_ from y.unique())
       np.sort for numeric dtype + str-key fallback for object dtype.

    4. estimators/custom.py:558,560 (FeatureSelector.classes_)
       Same pattern.

    5. models/optimization.py:308,311 (sampled_inputs sort)
       str-key fallback on user-seeded inputs that may be heterogeneous.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Source-level sensors
# ---------------------------------------------------------------------------


def test_fingerprint_cols_sort_uses_str_key() -> None:
    """A frame with int and str column labels fingerprints, and the fingerprint does not depend on the column order."""
    import numpy as np
    import pandas as pd

    from mlframe.training.feature_handling.fingerprint import fingerprint_df

    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(30, 4)))
    frame.columns = [0, "alpha", 1, "beta"]
    shuffled = frame[["beta", 1, "alpha", 0]].copy()
    first = fingerprint_df(frame)
    assert first.n_cols == 4
    assert fingerprint_df(shuffled) == first
    changed = frame.copy()
    changed["alpha"] = changed["alpha"] + 1.0
    assert fingerprint_df(changed) != first


def test_polars_fixes_union_sort_uses_str_key() -> None:
    """The Enum domain is the name-sorted union of train, val and the null-fill sentinel, whatever order the pre-computed union lists it in."""
    import polars as pl

    from mlframe.training.core._phase_polars_fixes import apply_polars_categorical_fixes

    train = pl.DataFrame({"c": ["b", "a", None, "b"]}, schema={"c": pl.Categorical})
    val = pl.DataFrame({"c": ["a", "z"]}, schema={"c": pl.Categorical})
    result = apply_polars_categorical_fixes(
        train_df_polars=train,
        val_df_polars=val,
        test_df_polars=None,
        train_df_pd=None,
        val_df_pd=None,
        test_df_pd=None,
        filtered_train_df=None,
        filtered_val_df=None,
        cat_features=["c"],
        align_polars_categorical_dicts=True,
        defer_pandas_conv=False,
        was_polars_input=True,
        verbose=False,
        precomputed_category_union={"c": ["z", "b", "a"]},
    )
    expected = ["__MISSING__", "a", "b", "z"]
    assert result.train_df_polars.schema["c"] == pl.Enum(expected)
    assert result.val_df_polars.schema["c"] == pl.Enum(expected)
    assert result.train_df_polars["c"].to_list() == ["b", "a", "__MISSING__", "b"]
    assert result.enum_domains["c"] == ["a", "b", "z"]


def test_neural_base_classes_sort_dtype_aware() -> None:
    """The classifier's classes_ is the sorted label set: numeric labels sort numerically, string labels by value."""
    import pytest

    torch = pytest.importorskip("torch")
    pytest.importorskip("lightning")
    import numpy as np

    from mlframe.training.neural import MLPTorchModel, PytorchLightningClassifier, TorchDataModule

    def make_classifier():
        """Tiny one-epoch CPU classifier."""
        return PytorchLightningClassifier(
            model_class=MLPTorchModel,
            model_params={"loss_fn": torch.nn.CrossEntropyLoss(), "learning_rate": 1e-3},
            network_params={"nlayers": 1, "first_layer_num_neurons": 8, "dropout_prob": 0.0, "activation_function": torch.nn.ReLU},
            datamodule_class=TorchDataModule,
            datamodule_params={
                "read_fcn": None,
                "data_placement_device": None,
                "features_dtype": torch.float32,
                "labels_dtype": torch.int64,
                "dataloader_params": {"batch_size": 32, "num_workers": 0},
            },
            trainer_params={
                "max_epochs": 1,
                "enable_model_summary": False,
                "default_root_dir": None,
                "log_every_n_steps": 1,
                "devices": 1,
                "logger": False,
                "accelerator": "cpu",
            },
        )

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4)).astype(np.float32)
    numeric = np.array([10, 2, 7] * 20)
    clf = make_classifier().fit(X, numeric)
    assert clf.classes_.tolist() == [2, 7, 10]
    text = np.array(["pear", "apple", "fig"] * 20, dtype=object)
    clf = make_classifier().fit(X, text)
    assert clf.classes_.tolist() == ["apple", "fig", "pear"]


def test_custom_feature_selector_classes_sort_dtype_aware() -> None:
    """IdentityClassifier.fit records sorted classes_, with None last and mixed-type labels ordered by their text."""
    import numpy as np
    import pandas as pd

    from mlframe.estimators.custom import IdentityClassifier

    X = np.zeros((6, 1))
    assert IdentityClassifier(feature_indices=[0]).fit(X, np.array([10, 2, 7, 2, 10, 7])).classes_.tolist() == [2, 7, 10]
    mixed = pd.Series(["pear", None, "apple", 3, "pear", None], dtype=object)
    assert IdentityClassifier(feature_indices=[0]).fit(X, mixed).classes_.tolist() == [3, "apple", "pear", None]


def test_optimization_sampled_inputs_sort_uses_str_key() -> None:
    """Initial samples of a mixed-type search space are evaluated in text order, None last, ascending or descending as asked."""
    from mlframe.models.optimization import MBHOptimizer

    space = [3, "b", None, "a", 10]
    for ascending, expected in ((True, [10, 3, "a", "b", None]), (False, [None, "b", "a", 3, 10])):
        optimizer = MBHOptimizer(
            search_space=space,
            init_num_samples=5,
            init_evaluate_ascending=ascending,
            init_evaluate_descending=not ascending,
            model_name="ETR",
            model_params={"n_estimators": 5},
            random_state=0,
        )
        assert optimizer.pre_seeded_candidates == expected


# ---------------------------------------------------------------------------
# Behavioural sensors
# ---------------------------------------------------------------------------


def test_sorted_with_str_key_handles_heterogeneous_input() -> None:
    """Document the str-key sort invariant: works on mixed type, deterministic."""
    cols = [0, "alpha", 1, None, "beta"]
    # Python sorted() raises TypeError; str-key fallback works.
    out = sorted(cols, key=str)
    assert len(out) == 5
    # None sorts to a stable position relative to str("None").


def test_str_key_sort_idempotent_on_homogeneous_input() -> None:
    """The fix should not change order on already-homogeneous input."""
    # Numeric -> str("0") < str("1") < str("10") < str("2") -- lexicographic
    # NB: this is the expected behaviour change; downstream code must NOT
    # assume numeric order. fingerprint.py only needs DETERMINISTIC order,
    # not numeric order, so this is acceptable.
    cols = [0, 1, 10, 2]
    out = sorted(cols, key=str)
    # Lexicographic: '0', '1', '10', '2'
    assert out == [0, 1, 10, 2]
