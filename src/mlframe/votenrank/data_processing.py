from __future__ import annotations

import pandas as pd
import numpy as np


def preprocess_glue(glue, head=None):
    for model, count in glue["Model"].value_counts().items():
        if count == 1:
            break
        to_add = pd.Series((glue["Model"][glue["Model"] == model].reset_index(drop=True).index + 1)).apply(lambda x: f"_{x}")
        mask = glue["Model"] == model
        glue.loc[mask, "Model"] = glue.loc[mask, "Model"] + to_add.values

    glue = glue.drop(columns=["Rank", "Name", "URL", "Score"]).set_index("Model")
    glue = glue.replace("-", np.nan)

    double_columns = ["MRPC", "STS-B", "QQP"]
    for column in double_columns:
        glue[column + "_n1"] = glue[column].apply(lambda x: x.split("/")[0])
        glue[column + "_n2"] = glue[column].apply(lambda x: x.split("/")[1])

    glue = glue.drop(columns=double_columns)
    glue = glue.astype(float)

    glue_weights = {col: 0.5 for col in glue.columns if col.endswith("n1") or col.endswith("n2")}
    glue_weights["AX"] = 0.0

    if head is not None:
        glue = glue.head(head)

    glue.rename(
        {"BERT: 24-layers, 16-heads, 1024-hidden": "BERT 24-layers, 16-heads, 1024-hidden"},
        inplace=True,
    )

    return glue, glue_weights


def preprocess_sglue(sglue):
    for model, count in sglue["Model"].value_counts().items():
        if count == 1:
            break
        to_add = pd.Series((sglue["Model"][sglue["Model"] == model].reset_index(drop=True).index + 1)).apply(lambda x: f"_{x}")
        mask = sglue["Model"] == model
        sglue.loc[mask, "Model"] = sglue.loc[mask, "Model"] + to_add.values

    sglue = sglue.drop(columns=["Rank", "Name", "URL", "Score"]).set_index("Model")
    sglue = sglue.replace("-", np.nan)

    double_columns = ["CB", "MultiRC", "ReCoRD", "AX-g"]
    for column in double_columns:
        sglue[column + "_n1"] = sglue[column].apply(lambda x: np.nan if x != x else x.split("/")[0])
        sglue[column + "_n2"] = sglue[column].apply(lambda x: np.nan if x != x else x.split("/")[1])

    sglue = sglue.drop(columns=double_columns)
    sglue = sglue.astype(float)

    sglue_weights = {col: 0.5 for col in sglue.columns if col.endswith("n1") or col.endswith("n2")}
    sglue_weights["AX-g_n1"] = 0.0
    sglue_weights["AX-g_n2"] = 0.0
    sglue_weights["AX-b"] = 0.0

    return sglue, sglue_weights


def preprocess_value(value):
    value = value.set_index("Model").drop(columns=["Mean-Rank", "Meta-Ave"])
    value = value.replace({"-": np.nan})
    value = value.astype(float)
    value.index = [
        "Human",
        "craig.starr",
        "DuKG",
        "HERO 1",
        "HERO 2",
        "HERO 3",
        "HERO 4",
    ]

    return value
