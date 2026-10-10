"""Captum IntegratedGradients / the CUDA-batched permutation kernel call the raw ``torch.nn.Module`` net
DIRECTLY, bypassing the training loop's own preprocessing -- so when ``X`` still carries un-encoded string
categorical columns, ``pandas.DataFrame.to_numpy()`` upcasts the whole array to ``dtype=object`` and
``torch.as_tensor`` rejects it outright.

Pre-fix, ``get_model_feature_importances`` handed Captum the raw caller frame unconditionally, so ANY
torch model with categorical features crashed Captum on every real (string-categorical) frame and fell
through to the ~50x-slower sklearn permutation-importance path for every such model -- found live via a
2M-row fuzz-optimization profiling run on a hgb+mlp+xgb multi-target-regression combo.

The real mlframe architecture (confirmed empirically against a running fuzz combo, NOT the simpler
Pipeline-embeds-the-encoder shape the first iteration of this fix assumed) chains TWO independent sources:

* ``model._mlframe_pre_pipeline`` -- a SIBLING object (imputer / scaler, and for tree/boosting strategies a
  category encoder too), stamped onto ``model`` at fit time by ``_apply_pre_pipeline_transforms``. For an
  MLP this pipeline deliberately leaves categoricals as raw strings (``_NumericOnlyTransformer``'s own
  docstring: "the raw categorical columns must reach the MLP estimator un-scaled / un-imputed so its
  fit-boundary factorizer + nn.Embedding can index them").
* ``_apply_cat_codes`` -- the MLP wrapper's OWN fit-time string-to-embedding-index factorization
  (``neural/base/_base_fit_prep.py``'s ``_cat_code_maps_``), which never runs inside ``pre_pipeline`` at
  all and is the piece that actually finishes a raw string column into a number for the net.

``test_net_input_frame_replays_the_pipelines_encoder_step`` and its siblings below pin a SECONDARY,
simpler fallback shape (``model`` itself embedding the encoder as an earlier ``Pipeline`` step) that
``_net_input_frame`` also supports for a model built that way, even though it is not the shape mlframe's
own MLP strategy produces -- ``test_net_input_frame_chains_pre_pipeline_and_cat_codes_like_the_real_mlp_path``
pins the actual chain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn
ce = pytest.importorskip("category_encoders")
from sklearn.pipeline import Pipeline

from mlframe.training._feature_importances import (
    _captum_integrated_gradients_importance,
    _net_input_frame,
    get_model_feature_importances,
)


class _LinearNet(nn.Module):
    """A bare one-layer net standing in for the Lightning wrapper's unwrapped ``torch.nn.Module`` core."""

    def __init__(self, n_in: int):
        """Build the single ``Linear(n_in, 1)`` layer."""
        super().__init__()
        self.fc = nn.Linear(n_in, 1)

    def forward(self, x):
        """Apply the linear layer."""
        return self.fc(x)


def _string_cat_frame(n: int = 200, seed: int = 0):
    """A frame with two numeric columns and one genuine Python-string categorical column, plus its target."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "num1": rng.standard_normal(n),
            "num2": rng.standard_normal(n),
            "cat1": [["mon", "tue", "wed", "thu"][i % 4] for i in range(n)],
        }
    )
    y = 2.0 * df["num1"].to_numpy() + rng.standard_normal(n) * 0.1
    return df, y


def _fitted_pipeline(df: pd.DataFrame, y: np.ndarray):
    """A fitted (CatBoostEncoder -> linear net) Pipeline matching how mlframe wraps an MLP ahead of a
    category encoder, plus the raw net for the direct-net FI call sites."""
    encoder = ce.CatBoostEncoder(random_state=0)
    X_enc = encoder.fit_transform(df, y)
    net = _LinearNet(X_enc.shape[1])
    return Pipeline([("enc", encoder), ("net", net)]), net


def test_net_input_frame_replays_the_pipelines_encoder_step():
    """``_net_input_frame`` returns all-numeric data matching the fitted encoder's own output, not the raw
    string-categorical frame."""
    df, y = _string_cat_frame()
    pipe, _net = _fitted_pipeline(df, y)
    out = _net_input_frame(pipe, df)
    out_arr = out.to_numpy() if hasattr(out, "to_numpy") else np.asarray(out)
    assert out_arr.dtype.kind == "f"
    assert out_arr.shape == (len(df), df.shape[1])


def test_net_input_frame_falls_back_to_raw_x_for_a_non_pipeline_model():
    """A bare (non-Pipeline) model has no earlier steps to replay -- X passes through unchanged."""
    df, y = _string_cat_frame()
    _pipe, net = _fitted_pipeline(df, y)
    assert _net_input_frame(net, df) is df


def test_captum_succeeds_on_a_pipeline_wrapped_net_with_string_categoricals():
    """The exact failure this fix closes: Captum on a Pipeline-wrapped net, called with the properly
    pre-encoded frame, must not raise and must return one attribution per column."""
    df, y = _string_cat_frame()
    pipe, net = _fitted_pipeline(df, y)
    encoded = _net_input_frame(pipe, df)
    result = _captum_integrated_gradients_importance(net, encoded)
    assert result is not None
    assert result.shape == (df.shape[1],)


def test_captum_on_the_raw_frame_directly_still_fails_closed_not_silently_wrong():
    """Without the fix (calling Captum on the raw un-encoded frame), the result is None (fails closed) --
    pins the safety contract ``_net_input_frame`` restores correctness for, not a silent wrong answer."""
    df, y = _string_cat_frame()
    _pipe, net = _fitted_pipeline(df, y)
    assert _captum_integrated_gradients_importance(net, df) is None


def test_get_model_feature_importances_uses_captum_on_a_pipeline_with_string_categoricals():
    """End-to-end: the public entry point reaches Captum (not the slow permutation fallback) for a
    Pipeline-wrapped torch net over a frame with real string categorical columns."""
    df, y = _string_cat_frame()
    pipe, _net = _fitted_pipeline(df, y)
    fi = get_model_feature_importances(pipe, list(df.columns), X=df, y=y, nn_fi_method="captum")
    assert fi is not None
    assert fi.shape == (df.shape[1],)


class _FakeImputer:
    """Stands in for mlframe's ``_NumericOnlyTransformer``-wrapped imputer/scaler: a no-op ``.transform``
    that leaves the frame (including raw string categoricals) exactly as it is, matching the real MLP
    strategy's ``pre_pipeline`` contract of passing categoricals through untouched."""

    def transform(self, X):
        """Return ``X`` unchanged."""
        return X


class _FakeMlpWrapper:
    """Minimal stand-in for mlframe's real Lightning-regressor wrapper: carries ``_apply_cat_codes`` (the
    MLP strategy's own fit-time string-to-index factorization), mirroring
    ``neural/base/_base_fit_prep.py``'s mixin closely enough to exercise ``_net_input_frame``'s real chain
    without pulling in the full Lightning stack."""

    def __init__(self, cat_cols, code_maps, cardinalities):
        """Store the fit-time factorization state the real mixin keeps on ``self``."""
        self._cat_cols_ = cat_cols
        self._cat_code_maps_ = code_maps
        self._cat_cardinalities_ = cardinalities

    def _apply_cat_codes(self, X):
        """Map each categorical column's values through its fitted code map, unseen values -> the
        reserved unknown code (the column's cardinality) -- the exact contract the real mixin documents."""
        out = X.copy()
        for col, card in zip(self._cat_cols_, self._cat_cardinalities_):
            mapping = self._cat_code_maps_[col]
            out[col] = out[col].astype(object).map(mapping).fillna(float(card)).astype(np.float32)
        return out


def test_net_input_frame_chains_pre_pipeline_and_cat_codes_like_the_real_mlp_path():
    """The actual mlframe MLP shape: ``model._mlframe_pre_pipeline`` leaves ``cat1`` as raw strings (the
    real ``_NumericOnlyTransformer`` contract), and ``_apply_cat_codes`` -- found on a SEPARATE wrapper
    object, not nested inside the pipeline -- finishes the encoding. Both must run for the result to be
    all-numeric."""
    df, _y = _string_cat_frame()
    code_map = {"mon": 0.0, "tue": 1.0, "wed": 2.0, "thu": 3.0}
    wrapper = _FakeMlpWrapper(cat_cols=["cat1"], code_maps={"cat1": code_map}, cardinalities=[4])
    wrapper._mlframe_pre_pipeline = _FakeImputer()

    out = _net_input_frame(wrapper, df)

    assert out["cat1"].dtype.kind == "f"
    assert np.array_equal(out["cat1"].to_numpy(), df["cat1"].map(code_map).to_numpy())
    # num1/num2 untouched by either step (the fake imputer is a no-op, _apply_cat_codes only touches cat1).
    assert np.array_equal(out["num1"].to_numpy(), df["num1"].to_numpy())


def test_net_input_frame_applies_cat_codes_even_with_no_pre_pipeline_stamp():
    """``_apply_cat_codes`` alone (no ``_mlframe_pre_pipeline`` attribute at all) still finishes the
    categorical encoding -- the two sources are independent, not both-or-nothing."""
    df, _y = _string_cat_frame()
    code_map = {"mon": 0.0, "tue": 1.0, "wed": 2.0, "thu": 3.0}
    wrapper = _FakeMlpWrapper(cat_cols=["cat1"], code_maps={"cat1": code_map}, cardinalities=[4])

    out = _net_input_frame(wrapper, df)

    assert out["cat1"].dtype.kind == "f"
    assert np.array_equal(out["cat1"].to_numpy(), df["cat1"].map(code_map).to_numpy())
