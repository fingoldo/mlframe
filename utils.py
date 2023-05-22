"""Auxiliary helpers not worth their own modules."""
import os
import random
import numpy as np


def set_ml_random_seed(seed: int = 42, set_hash_seed: bool = False, set_torch_seed: bool = False):
    """Seed everything ml-related."""
    random.seed(seed)
    np.random.seed(seed)

    if set_hash_seed:
        os.environ["PYTHONHASHSEED"] = str(seed)

    if set_torch_seed:
        import torch  # pylint: disable=import-outside-toplevel

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # type: ignore


def get_pipeline_last_element(clf) -> object:
    for elem_name, elem in clf.named_steps.items():
        pass
    return elem


def get_full_classifier_name(clf: object) -> str:
    clf_name = type(clf).__name__
    if clf_name == "TransformedTargetRegressor":
        regressor_name = get_full_classifier_name(clf.regressor)
        if clf.transformer:
            transformer_name = type(clf.transformer).__name__
            try:
                transformer_name += " " + clf.transformer.method
            except:
                pass
            try:
                transformer_name += " " + clf.transformer.output_distribution  # QuantileTransformer
            except:
                pass
        else:
            try:
                transformer_name = clf.func.__name__
            except:
                transformer_name = "func"

        full_clf_name = " -> ".join([regressor_name, transformer_name])
    elif clf_name == "Pipeline":
        elem = get_pipeline_last_element(clf)
        return f"pipe[{get_full_classifier_name(elem)}]"
    elif clf_name == "MultiOutputRegressor":
        return f"MultiOutputRegressor[{get_full_classifier_name(clf.estimator)}]"

    else:
        if "Dummy" in clf_name:
            full_clf_name = clf_name + "[" + clf.strategy + "]"
        else:
            full_clf_name = clf_name

    return full_clf_name
