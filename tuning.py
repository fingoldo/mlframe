# ****************************************************************************************************************************
# Imports
# ****************************************************************************************************************************

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from pyutilz.pythonlib import ensure_installed

ensure_installed("pandas numpy scipy")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *
from scipy.stats import uniform, loguniform, randint

from scipy.stats._distn_infrastructure import rv_continuous_frozen, rv_discrete_frozen
from sklearn.model_selection import ParameterSampler
from sklearn.model_selection import train_test_split

from pyutilz import db
import logging

from random import random
import pandas as pd, numpy as np
from catboost import CatBoostRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_validate

from enum import Enum, auto

# MLTaskType = Enum("MLTaskType", ["Regression", "Multiregression", "Classification", "Multiclassification", "MultilabelClassification", "Ranking"])
class MLTaskType(Enum):
    Regression = auto()
    Multiregression = auto()
    Classification = auto()
    Multiclassification = auto()
    MultilabelClassification = auto()
    Ranking = auto()


trained_models = {}


class hashabledict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))


def check_condition(condition, params: dict) -> bool:
    if isinstance(condition, (dict, hashabledict)):
        skipped = False
        # must hold on all the conditions!
        for cond_field, cond_value in condition.items():
            if isinstance(cond_value, list):
                # print(cond_field,cond_value)
                for sub_value in cond_value:
                    if value_by_key(dct=params, key=cond_field, expected_value=sub_value):
                        # print("True")
                        return True
                # print("False")
            else:
                if value_by_key(dct=params, key=cond_field, expected_value=cond_value):
                    return True
        return skipped
    else:
        return condition


def value_by_key(dct: dict, key, expected_value) -> bool:
    return dct.get(key) == expected_value


def check_rules(params, drop_if_rules=None, drop_if_not_rules=None, skip_if_values_or=None, allow_if_values_or=None, allow_if_values_and=None):
    n = 0
    if drop_if_rules:
        for rule in drop_if_rules:
            for condition in rule.get("conditions", []):
                if check_condition(condition, params):
                    for field in rule.get("fields"):
                        if field in params:
                            # print(f'deleted {field} field')
                            del params[field]
    if drop_if_not_rules:
        for rule in drop_if_not_rules:
            for condition in rule.get("conditions", []):
                if not check_condition(condition, params):
                    for field in rule.get("fields"):
                        if field in params:
                            # print(f'deleted {field} field')
                            del params[field]
    if skip_if_values_or:
        for conditions, fields in skip_if_values_or.items():
            skip = False
            for condition in conditions:
                if not check_condition(condition, params):
                    skip = True
                    break
            if not skip:
                for field_cond in fields:
                    if check_condition(field_cond, params):
                        # print(f"skip cond {condition} triggered for {field_cond}")
                        return False
                    else:
                        # print(f'skip cond {condition} not triggered for {field_cond}')
                        pass

    if allow_if_values_or:
        for conditions, fields in allow_if_values_or.items():
            skip = False
            for condition in conditions:
                if not check_condition(condition, params):
                    skip = True
                    break
            if not skip:
                if check_condition(condition, params):
                    any_triggered = False
                    for field_cond in fields:
                        if check_condition(field_cond, params):
                            # print(f"allow cond {condition} triggered for {field_cond}, {params}")
                            any_triggered = True
                            break
                    if not any_triggered:
                        # print(f"none of allow_if_values_or {conditions} {fields} triggered")
                        return False

    if allow_if_values_and:
        for conditions, fields in allow_if_values_and.items():
            skip = False
            # print("allow_if_values_and precheck: ", conditions)
            for condition in conditions:
                if not check_condition(condition, params):
                    skip = True
                    break
            if not skip:
                # print("allow_if_values_and check: ", conditions)
                for field_cond in fields:
                    if not check_condition(field_cond, params):
                        # print(f"allow_if_values_and cond {condition} NOT triggered for {field_cond}")
                        return False
                # (hashabledict({"posterior_sampling": True}),): [
                # {"model_shrink_mode": "Constant"},  # Posterior Sampling requires Сonstant Model Shrink Mode
                # {"langevin": True},  # Posterior Sampling requires Langevin boosting],
    return True


def double_check_dist_params(cand: dict, drop_none: bool = False) -> dict:
    """If some of params were not resolved by the first ParameterSampler call,
    try on a deeper level.
    """
    nchanged = 1
    while nchanged > 0:
        nchanged = 0
        for key, value in cand.copy().items():
            if isinstance(value, (rv_continuous_frozen, rv_discrete_frozen)):
                cand[key] = value.rvs()  # list(ParameterSampler({key: value}, n_iter=1))[0][key]
                nchanged += 1
    return cand


def generate_valid_candidates(
    params,
    drop_if_rules=None,
    drop_if_not_rules=None,
    skip_if_values_or=None,
    allow_if_values_or=None,
    allow_if_values_and=None,
    n: int = 1,
    max_iters: int = 1000,
):
    logging.info(f"Generating {n} valid candidates...")
    approved = []
    niters = 0
    while True:
        params_samples = list(ParameterSampler(params, n_iter=n))

        for sample in params_samples:
            double_check_dist_params(sample)
            if check_rules(
                sample,
                drop_if_rules=drop_if_rules,
                drop_if_not_rules=drop_if_not_rules,
                skip_if_values_or=skip_if_values_or,
                allow_if_values_or=allow_if_values_or,
                allow_if_values_and=allow_if_values_and,
            ):
                approved.append(sample)
                niters += 1
                if len(approved) == n or niters >= max_iters:
                    return approved


def preprocess_df(df, cat_features):

    for var in cat_features:
        df[var] = df[var].fillna("")


def prepare_trials_dataset(experiment_name: str, objective_name: str) -> pd.DataFrame:
    logging.info(f"Getting trials for experiment {experiment_name}...")
    res = []
    for id, node, params, results in db.safe_execute("select id,node,params,results from experiments where  project=%s", (experiment_name,)):
        if objective_name in results:
            params["target"] = results[objective_name]
            res.append(params)

    df = pd.DataFrame(res)
    cat_features = []

    if len(df) > 0:
        df = df.loc[:, (df != df.iloc[0]).any()]

        for var in "grow_policy model_shrink_mode verbose boost_from_average".split():
            if var in df:
                del df[var]

        cat_features = df.select_dtypes(include="object").columns.values.tolist()

        preprocess_df(df, cat_features)

    return df, cat_features


def normalize_probs(probs: np.ndarray):
    np.divide(probs, probs.sum(), out=probs)


def favorize_unexplored(candidates: list, probs: np.ndarray, trials: pd.DataFrame, cat_features: list, order: int = 1) -> None:
    """
    Assign higher select probabilities to combinations of order N that were never chosen yet in the trials object.
    """
    if len(cat_features) == 0 or len(trials) == 0:
        return

    logging.info(f"Favorizing unexplored trials...")

    already_sampled = {}
    for col in cat_features:
        already_sampled[col] = trials[col].unique().tolist()

    favorized_items = []
    for i in range(len(probs)):
        candidate = candidates[i]
        for param, value in candidate.items():
            if param in cat_features:
                known_values = already_sampled.get(param, [])
                if value not in known_values:
                    probs[i] *= 2
                    favorized_items.append({param: value})
                    known_values.append(value)
                    break

    if len(favorized_items) > 0:
        normalize_probs(probs)
        if len(favorized_items) < 5:
            logging.info(f"Favorized {len(favorized_items):_} previously unexplored candidates: {favorized_items}")
        else:
            logging.info(f"Favorized {len(favorized_items):_} previously unexplored candidates")


def get_model(experiment_name: str, trials: pd.DataFrame, cat_features: list, cv: int, scoring: str, min_score: float, max_new_trials: int = 10):
    # trained_models[experiment_name]=[fitted_model,len(trials),ml_scoring,expected_performance]
    should_retrain = True
    if experiment_name in trained_models:
        fitted_model, num_trials, prev_scoring, expected_score, model_columns = trained_models[experiment_name]
        if prev_scoring == scoring:

            if expected_score >= min_score:
                if len(trials) - num_trials > max_new_trials:
                    logging.info(f"Model for experiment {experiment_name} found, but it needs updating with new data")
                else:
                    logging.info(f"Passing model for experiment {experiment_name} found ;-)")
                    should_retrain = False
            else:
                if len(trials) == num_trials:
                    logging.info(f"Model for experiment {experiment_name} found, but score was bad, and since then no new trials data arrived (")
                    should_retrain = False
                else:
                    logging.info(f"Model for experiment {experiment_name} found, score was bad, but since then new trials data has arrived!")

    y = trials.pop("target").values
    if should_retrain:
        model = CatBoostRegressor(iterations=100, task_type="CPU", cat_features=cat_features, verbose=False)

        X = trials
        model_columns = X.columns

        fitted_model, expected_score = justify_estimator(model, X, y, refit=True, test_size=0.1, cv=cv, scoring=scoring, min_score=min_score)

        y_min, y_max = y.min, y.max()
        trained_models[experiment_name] = [fitted_model, len(trials), scoring, expected_score, model_columns]

    return fitted_model, model_columns, y


def justify_estimator(est, X, y, cv=3, refit: bool = True, scoring="r2", min_score: float = 0.6, test_size=0.1, plot=False):

    logging.info(f"Checking if ML gains some predictive power already on {len(y):_} samples...")

    cv_results = cross_validate(est, X, y, cv=cv, scoring=scoring)
    mean_score = np.mean(cv_results["test_score"])

    if mean_score >= min_score:
        logging.info(f"OOS mean {scoring}={mean_score}, so ML can be used.")
        if refit:
            logging.info(f"Fitting a model")

            if "cat" in str(est).lower():
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
                est.fit(X_train, y_train, eval_set=(X_test, y_test), plot=plot, verbose=False)
            else:
                est.fit(X, y)
        else:
            est = None
    else:
        logging.info(f"OOS mean {scoring}={mean_score}, so ML can't be used (yet).")
        est = None
    return est, mean_score


def create_ctr_params(GPU_ENABLED: bool = True, params: dict = {}) -> str:
    res = []
    for main_type in (
        "Borders Buckets BinarizedTargetMeanValue Counter".split() if not GPU_ENABLED else "Borders Buckets FeatureFreq FloatTargetMeanValue".split()
    ):  # The method for transforming categorical features to numerical features.
        if random() > 0.5:
            cands = generate_valid_candidates(
                params={
                    "CtrBorderCount": [1, randint(1, 255)],
                    "CtrBorderType": ["Uniform"] if not GPU_ENABLED else "Median Uniform".split(),
                    "TargetBorderCount": [1, randint(1, 255)],
                    "TargetBorderType": ["Uniform"] if not GPU_ENABLED else "Median Uniform UniformAndQuantiles MaxLogSum MinEntropy GreedyLogSum".split(),
                }
            )
            line = main_type
            for key, val in list(cands)[0].items():
                # "Counter:TargetBorderType=Uniform:TargetBorderCount=1: Target borders options are unsupported for counter ctr
                if main_type in ("Counter", "FeatureFreq"):
                    if "Target" in key:
                        continue
                if key == "TargetBorderCount":  # Setting TargetBorderCount is not supported for loss function CrossEntropy
                    if "CrossEntropy" in params.get("loss_function", []):
                        continue
                line += ":" + key + "=" + str(val)
            if line != main_type:
                res.append(line)
    if res == []:
        res = None
    return res


class ParamsOptimizer:
    def __init__(self):
        # ,db_name:str=None,db_host:str=None,db_port:int=None,db_username:str=None,db_pwd:str=None,db_schema:str="public"
        if False:
            db.connect_to_db(
                m_db_name=db_name,
                m_db_host=db_host,
                m_db_port=db_port,
                m_db_username=db_username,
                m_db_pwd=db_pwd,
                m_init_params_fn=None,
                m_db_schema=db_schema,
                m_db_sslmode="require",
            )

    def create_study(self, task_id: str, stydy_type: str = "ml_estimator_hyperparameters"):
        pass

    def suggest_trials(
        self,
        experiment_name: str,
        objective_name: str,
        minimize: bool = False,
        n: int = 1,
        sampler: str = "ml",
        search_space_multiplier: int = 2,
        search_space_minsize: int = 50,
        favor_unexplored: bool = True,
        max_attempts=10,
        min_samples_for_ml: int = 30,
        improving_by_atleast: float = 0.0,
        ml_cv: int = 3,
        ml_scoring: str = "r2",
        ml_min_score: float = 0.6,
    ):

        assert sampler in ("random", "ml")

        def get_n_cands(n):
            return generate_valid_candidates(
                params=self.params,
                drop_if_rules=self.drop_if_rules,
                drop_if_not_rules=self.drop_if_not_rules,
                skip_if_values_or=self.skip_if_values_or,
                allow_if_values_or=self.allow_if_values_or,
                allow_if_values_and=self.allow_if_values_and,
                n=n,
            )

        if sampler == "random":
            return get_n_cands(n)
        elif sampler == "ml":

            # ---------------------------------------------------------------------------------------------
            # get all known trials for the experiment
            # ---------------------------------------------------------------------------------------------

            trials, cat_features = prepare_trials_dataset(experiment_name=experiment_name, objective_name=objective_name)

            # if we have too few samples yet, return random, but favor yet unexplored invididual cat values

            if len(trials) == 0:
                return get_n_cands(n)

            for _ in range(max_attempts):

                # ---------------------------------------------------------------------------------------------
                # sample much more valid candidates
                # ---------------------------------------------------------------------------------------------

                candidates = get_n_cands(n=max(n * search_space_multiplier, search_space_minsize))

                if len(candidates) <= 1:
                    logging.warning(f"Nothing to sample for experiment {experiment_name}")
                    return candidates

                probs = np.ones(len(candidates), np.float32) / len(candidates)
                if len(candidates) > n:

                    if len(trials) >= min_samples_for_ml:

                        # ---------------------------------------------------------------------------------------------
                        # score them with the ML model
                        # ---------------------------------------------------------------------------------------------

                        fitted_model, model_columns, y = get_model(
                            experiment_name=experiment_name, trials=trials, cat_features=cat_features, cv=ml_cv, scoring=ml_scoring, min_score=ml_min_score
                        )

                        if fitted_model is not None:

                            # ---------------------------------------------------------------------------------------------
                            # score them with the ML model
                            # ---------------------------------------------------------------------------------------------

                            candidates_df = pd.DataFrame(candidates)
                            for col in model_columns:
                                if col not in candidates_df:
                                    candidates_df[col] = np.nan
                            candidates_df = candidates_df[model_columns]

                            preprocess_df(candidates_df, cat_features)
                            predictions = fitted_model.predict(candidates_df)

                            if predictions.min() < 0:
                                probs = predictions + predictions.min()
                            else:
                                probs = predictions

                            if minimize:
                                probs = 1 - probs

                            # ---------------------------------------------------------------------------------------------
                            # only samples with expected performance above currently highest one
                            # ---------------------------------------------------------------------------------------------

                            if improving_by_atleast:

                                if minimize:
                                    desired_objective = y.max() - (y.max() - y.min()) * improving_by_atleast
                                    bad_indices = np.where(predictions >= desired_objective)[0]

                                    logging.info(
                                        f"Best and worst observed trial's objectives: {y.min()}, {y.max()}. Skipping {len(bad_indices)} candidates with predicted objective over {desired_objective}"
                                    )

                                else:
                                    desired_objective = (y.max() - y.min()) * improving_by_atleast + y.min()
                                    bad_indices = np.where(predictions <= desired_objective)[0]

                                    logging.info(
                                        f"Best and worst observed trial's objectives: {y.max()}, {y.min()}. Leaving only {len(bad_indices)} candidates with predicted objective under {desired_objective}"
                                    )

                                probs[bad_indices] = 0.0
                            normalize_probs(probs)

                    # ---------------------------------------------------------------------------------------------
                    # give more weights to yet unexplored individual values of cat params
                    # ---------------------------------------------------------------------------------------------

                    if favor_unexplored:
                        favorize_unexplored(candidates=candidates, probs=probs, trials=trials, cat_features=cat_features, order=1)

                    # ---------------------------------------------------------------------------------------------
                    # return potentially most promising
                    # ---------------------------------------------------------------------------------------------

                    # Fewer non-zero entries in p than size
                    n = min(n, len(np.where(probs > 0)[0]))
                    return np.random.choice(candidates, n, replace=False, p=probs)

            return candidates[:n]

    def report_trial_results(self, objectives: dict):
        """
        Persists results
        """
        pass


class CatboostParamsOptimizer(ParamsOptimizer):
    def __init__(
        self,
        GPU_ENABLED: bool = False,
        groups: bool = False,
        need_training_continuation: bool = False,
        task: MLTaskType = MLTaskType.Regression,
        params_override: dict = {},
        delete_params: Sequence = [],
    ):

        # ,db_name:str=None,db_host:str=None,db_port:int=None,db_username:str=None,db_pwd:str=None,db_schema:str="public"
        # super().init(db_name=db_name,db_host=db_host,db_port=db_port,db_usernam=db_username,db_pwd=db_pwd,db_schema=db_schema)

        # --per-float-feature-quantization 0:border_count=1024

        self.params = {
            # Special params
            # monotone-constraints = "Feature2:1,Feature4:-1"
            # feature_weights = [0.1, 1, 3]
            # first_feature_use_penalties = "2:1.1,4:0.1"
            # per_object_feature_penalties = "2:1.1,4:0.1"
            # ----------------------------------------------------------------------------------------------------------------------------
            # Float params
            # ----------------------------------------------------------------------------------------------------------------------------
            "subsample": [None, uniform(0.5, 1.0 - 0.5)],
            "learning_rate": uniform(0.1, 0.4 - 0.1),  # Alias: eta
            "bagging_temperature": [
                0,
                1,
                loguniform(0.01, 2),
            ],  # Defines the settings of the Bayesian bootstrap. It is used by default in classification and regression modes. Possible values are in the range [0, +inf]. can be used if the selected bootstrap type is Bayesian.
            # "bayesian_matrix_reg": uniform(0.01, 0.9-0.01),
            "eval_fraction": loguniform(0.01, 0.3 - 0.01),
            "rsm": [1, uniform(0.8, 1.0 - 0.8)] if not GPU_ENABLED else [1],  # Alias:colsample_bylevel # rsm on GPU is supported for pairwise modes only
            "target_border": [
                None,
                uniform(0.35 - 0.15, 0.5 + 0.15 - 0.35),
            ],  # If set, defines the border for converting target values to 0 and 1. Depending on the specified value:target_value≤border_value the target is converted to 0; target_value>border_value the target is converted to 1.
            # "mvs_reg": [
            #    None,
            #    loguniform(0.01, 100),
            # ],  # Affects the weight of the denominator and can be used for balancing between the importance and Bernoulli sampling (setting it to 0 implies importance sampling and to ∞ - Bernoulli). This parameter is supported only for the MVS sampling method (the bootstrap_type parameter must be set to MVS).
            "fold_len_multiplier": [2, loguniform(1.1, 2.9)],
            "diffusion_temperature": [10000, loguniform(1_000, 100_000)],
            "penalties_coefficient": [1, loguniform(1, 3)],
            # ----------------------------------------------------------------------------------------------------------------------------
            # Int params
            # ----------------------------------------------------------------------------------------------------------------------------
            "depth": randint(1, 16),  # Maximum tree depth is 16
            "max_leaves": randint(2, 70),
            "l2_leaf_reg": randint(1, 10),  # Any positive value is allowed.
            "border_count": [None, randint(30, 300)],
            "penalties_coefficient": randint(1, 10),  # Any positive value is allowed.
            "model_size_reg": randint(1, 10),
            "one_hot_max_size": [None, randint(2, 300)],
            "ctr_leaf_count_limit": [
                None,
                randint(10, 100),
            ]
            if not GPU_ENABLED
            else [
                None
            ],  # The maximum number of leaves with categorical features. If the quantity exceeds the specified value a part of leaves is discarded. This option reduces the resulting model size and the amount of memory required for training. Note that the resulting quality of the model can be affected.
            "random_strength": [
                1,
                randint(1, 5),
            ],  # The amount of randomness to use for scoring splits when the tree structure is selected. Use this parameter to avoid overfitting the model. The value of this parameter is used when selecting splits. On every iteration each possible split gets a score (for example, the score indicates how much adding this split will improve the loss function for the training dataset). The split with the highest score is selected.
            "max_ctr_complexity": [None, randint(1, 10)],
            "min_data_in_leaf": [
                1,
                randint(1, 10),
            ],  # Alias: min_child_samples. The minimum number of training samples in a leaf. CatBoost does not search for new splits in leaves with samples count less than the specified value. Can be used only with the Lossguide and Depthwise growing policies.
            "leaf_estimation_iterations": [
                None,
                randint(1, 30),
            ],  # CatBoost might calculate leaf values using several gradient or newton steps instead of a single one. This parameter regulates how many steps are done in every tree when calculating leaf values.
            "iterations": randint(100, 1000),  # Aliases: num_boost_round, n_estimators, num_trees
            "fold_permutation_block": [1, randint(1, 256)],
            # ----------------------------------------------------------------------------------------------------------------------------
            # Cat params
            # ----------------------------------------------------------------------------------------------------------------------------
            "sampling_unit": ["Object", "Group"]
            if groups
            else ["Object"],  # The sampling scheme. #No groups in dataset. Please disable sampling or use per object sampling
            "boosting_type": [
                None,
                "Plain",
                "Ordered",
            ],  # ,  It is set to Ordered by default for datasets with less then 50 thousand objects. TheOrdered scheme requires a lot of memory.
            "sampling_frequency": ["PerTree", "PerTreeLevel"],  # Frequency to sample weights and objects when building trees.
            "leaf_estimation_method": ["Newton", "Gradient", "Exact"],  # The method used to calculate the values in leaves.
            "nan_mode": ["Min", "Max"],
            "counter_calc_method": ["SkipTest", "Full"],
            "feature_border_type": "Median Uniform UniformAndQuantiles MaxLogSum MinEntropy GreedyLogSum".split(),  # The quantization mode for numerical features.
            # ----------------------------------------------------------------------------------------------------------------------------
            # Bool params
            # ----------------------------------------------------------------------------------------------------------------------------
            "langevin": [False, True],
            "posterior_sampling": [False, True],
            "has_time": [
                False,
                True,
            ],  # Use this option if the objects in your dataset are given in the required order. In this case, random permutations are not performed during the Transforming categorical features to numerical features and Choosing the tree structure stages.
            "approx_on_full_history": [False, True] if not GPU_ENABLED else [False],
            "store_all_simple_ctr": [
                False,
                True,
            ]
            if not GPU_ENABLED
            else [False],  # Ignore categorical features, which are not used in feature combinations, when choosing candidates for exclusion.
            # Device specific params
            "leaf_estimation_backtracking": [
                "AnyImprovement",
                "No",
                "Armijo",
            ],  # Armijo -gpu only. When the value of the leaf_estimation_iterations parameter is greater than 1, CatBoost makes several gradient or newton steps when calculating the resulting leaf values of a tree.
            "score_function": [
                "Cosine",  # (do not use this score type with the Lossguide tree growing policy)
                "NewtonCosine",  # (do not use this score type with the Lossguide tree growing policy)
                "L2",
                "NewtonL2",
            ],  # GPU — All score types, CPU — Cosine, L2
            "bootstrap_type": [
                "Bayesian",
                "Bernoulli",
                "MVS",
                "No",
                "Poisson",  # (supported for GPU only)
            ],  # Bootstrap type. Defines the method for sampling the weights of objects.
            "task_type": ["GPU" if GPU_ENABLED else "CPU"],
        }
        if False:
            if task == MLTaskType.Regression:
                self.params["loss_function"] = "MAE MAPE Poisson Quantile RMSE LogLinQuantile LogCosh".split() + [
                    "Lq:q=" + str(loguniform(1, 100).rvs()),
                    "Expectile:alpha=" + str(loguniform(0.01, 1 - 0.01).rvs()),
                    "Tweedie:variance_power=" + str(uniform(1.01, 1.99 - 1.01).rvs()),
                    "Huber:delta=" + str(loguniform(0.1, 100).rvs()),
                    # "MultiQuantile:alpha=" + str(loguniform(0.01, 1 - 0.01).rvs()) + "," + str(loguniform(0.01, 1 - 0.01).rvs()), # if MultiQuantile is chosen, it hs to be both a loss and eval_metric!
                ]  # RMSEWithUncertainty needs double target #Alpha parameter for expectile metric should be in interval [0, 1]
                self.params["eval_metric"] = (
                    self.params["loss_function"]
                    + "FairLoss SMAPE R2 MSLE MedianAbsoluteError".split()
                    + ["NumErrors:greater_than=" + str(loguniform(1, 100).rvs())]
                )
                if not GPU_ENABLED:
                    self.params[
                        "feature_border_type"
                    ] = (
                        "Median Uniform UniformAndQuantiles MaxLogSum MinEntropy GreedyLogSum".split()
                    )  # The quantization type for the label value. Only used for regression problems.
            elif task == MLTaskType.Multiregression:
                self.params["loss_function"] = "MultiRMSE MultiRMSEWithMissingValues".split()
                self.params["eval_metric"] = self.params["loss_function"]
            elif task == MLTaskType.Classification:
                self.params["loss_function"] = "Logloss CrossEntropy".split()
                self.params["eval_metric"] = (
                    self.params["loss_function"]
                    + "Precision Recall F1 BalancedAccuracy BalancedErrorRate MCC Accuracy AUC NormalizedGini BrierScore HingeLoss HammingLoss ZeroOneLoss Kappa WKappa LogLikelihoodOfPrediction".split()
                    + ["F:beta=" + str(loguniform(0.01, 100).rvs())]
                )  # CtrFactor cannot be used for overfitting detection or selecting best iteration on validation
                # QueryAUC : Groupwise loss/metrics require nontrivial groups
            elif task == MLTaskType.Multiclassification:
                self.params["loss_function"] = "MultiClass MultiClassOneVsAll".split()
                self.params["eval_metric"] = (
                    self.params["loss_function"] + " Precision Recall F F1 TotalF1 MCC Accuracy HingeLoss HammingLoss ZeroOneLoss Kappa WKappa AUC".split()
                )
            elif task == MLTaskType.MultilabelClassification:
                self.params["loss_function"] = "MultiLogloss MultiCrossEntropy".split()
                self.params["eval_metric"] = self.params["loss_function"] + " Precision Recall F F1 Accuracy HammingLoss".split()
            elif task == MLTaskType.Ranking:
                self.params[
                    "loss_function"
                ] = "PairLogit PairLogitPairwise YetiRank YetiRankPairwise StochasticFilter StochasticRank QueryCrossEntropy QueryRMSE QuerySoftMax".split()
                self.params["eval_metric"] = (
                    self.params["loss_function"] + "PairAccuracy PFound NDCG DCG FilteredDCG AverageGain PrecisionAt RecallAt MAP ERR MRR AUC QueryAUC".split()
                )
                self.params["force_unit_auto_pair_weights"] = [False, True]
            else:
                raise ValueError("Unknown task %s", task)

            # all kinds of classification
            if task in (MLTaskType.Classification, MLTaskType.Multiclassification, MLTaskType.MultilabelClassification):
                self.params["auto_class_weights"] = [
                    None,
                    "Balanced",
                    "SqrtBalanced",
                ]  # The values are used as multipliers for the object weights. This parameter can be used for solving binary classification and multiclassification problems.

            if not need_training_continuation:
                self.params["model_shrink_mode"] = [
                    "Constant",
                    "Decreasing",
                ]  # Model shrinkage in combination with learning continuation is not implemented yet. Reset model_shrink_rate to 0.
                if not GPU_ENABLED:
                    self.params["model_shrink_rate"] = [
                        None,
                        loguniform(0.01, 1.0),
                    ]  # The constant used to calculate the coefficient for multiplying the model on each iteration.The actual model shrinkage coefficient calculated at each iteration depends on the value of the --model-shrink-mode.The resulting value of the coefficient should be always in the range (0, 1].
                self.params["boost_from_average"] = [False, True]  # You can't use boost_from_average with initial model now
                self.params["grow_policy"] = ["SymmetricTree", "Depthwise", "Lossguide"]  # Models summation supported only for symmetric trees

        if params_override:
            self.params.update(params_override)

        self.params["simple_ctr"] = [
            None,
            create_ctr_params(GPU_ENABLED=GPU_ENABLED, params=self.params),
        ]  # ['Borders:CtrBorderCount=15:CtrBorderType=Uniform:TargetBorderCount=1:TargetBorderType=MinEntropy:Prior=0/1:Prior=0.5/1:Prior=1/1','Counter:CtrBorderCount=15:CtrBorderType=Uniform:Prior=0/1'], # Quantization settings for simple categorical features. Use this parameter to specify the principles for defining the class of the object for regression tasks. By default, it is considered that an object belongs to the positive class if its' label value is greater than the median of all label values of the dataset.
        self.params["combinations_ctr"] = [
            None,
            create_ctr_params(GPU_ENABLED=GPU_ENABLED, params=self.params),
        ]  # Quantization settings for combinations of categorical features.

        for key in delete_params:
            if key in self.params:
                del self.params[key]
        self.drop_if_rules = [
            {
                "conditions": [GPU_ENABLED],
                "fields": ["sampling_frequency"],
            },  # Error: change of option sampling_frequency is unimplemented for task type GPU and was not default in previous run
            {"conditions": [{"bootstrap_type": "No"}], "fields": ["subsample"]},  # Error: you shoudn't provide bootstrap options if bootstrap is disabled
            {"conditions": [{"bootstrap_type": "Bayesian"}], "fields": ["subsample"]},  # Error: bayesian bootstrap doesn't support taken fraction option
            {
                "conditions": [GPU_ENABLED],
                "fields": ["model_shrink_mode"],
            },  # Error: change of option model_shrink_mode is unimplemented for task type GPU and was not default in previous run
            {"conditions": [{"posterior_sampling": True}], "fields": ["diffusion_temperature"]},  # Diffusion Temperature in Posterior Sampling is specified
        ]

        self.drop_if_not_rules = [
            {
                "conditions": [{"bootstrap_type": "Bayesian"}],
                "fields": ["bagging_temperature"],
            },  # Error: bagging temperature available for bayesian bootstrap only
            {"conditions": [{"grow_policy": "Lossguide"}], "fields": ["max_leaves"]},  # max_leaves option works only with lossguide tree growing
        ]

        # No groups in dataset. Please disable sampling or use per object sampling

        self.skip_if_values_or = {
            (hashabledict({"sampling_frequency": "PerTreeLevel"}),): [
                {"grow_policy": "Lossguide"}
            ],  # PerTreeLevel sampling is not supported for Lossguide grow policy.
            (hashabledict({"bootstrap_type": "Poisson"}),): [not GPU_ENABLED],  # Error: poisson bootstrap is not supported on CPU
            (not GPU_ENABLED,): [{"leaf_estimation_backtracking": "Armijo"}],  # Backtracking type Armijo is supported only on GPU
            (hashabledict({"approx_on_full_history": True}),): [
                {"boosting_type": [None, "Plain"]}
            ],  # Can't use approx-on-full-history with Plain boosting-type
            (hashabledict({"leaf_estimation_method": "Newton"}),): [
                {
                    "loss_function": ["MAE", "MAPE", "Quantile", "MultiQuantile", "LogLinQuantile"]
                    + [el for el in self.params.get("loss_function", []) if el.startswith("Lq")]
                }
            ],  # Newton leaves estimation method is not supoprted for MAPE loss function # Newton leaves estimation method is not supoprted for Lq loss function with q < 2 !TODO
            (hashabledict({"leaf_estimation_method": "Exact"}),): [
                {"approx_on_full_history": True}
            ],  # ApproxOnFullHistory option is not available within Exact method on CPU.
        }

        self.allow_if_values_or = {
            (hashabledict({"bootstrap_type": "MVS"}),): [{"sampling_unit": "Object"}],  # MVS bootstrap supports per object sampling only.
            (hashabledict({"boosting_type": "Ordered"}),): [{"grow_policy": "SymmetricTree"}],  # Ordered boosting is not supported for nonsymmetric trees.
            (not GPU_ENABLED,): [
                {"score_function": "Cosine"},
                {"score_function": "L2"},
                {"score_function": None},
            ],  # Only Cosine and L2 score functions are supported for CPU.
            (hashabledict({"leaf_estimation_method": "Exact"}),): [
                {"loss_function": "Quantile"},
                {"loss_function": "MAE"},
                {"loss_function": "MAPE"},
                {"loss_function": "LogCosh"},
            ],  # Exact method is only available for Quantile, MAE, MAPE and LogCosh loss functions.
            (hashabledict({"auto_class_weights": "Balanced"}),): [
                {"loss_function": "Logloss"},
                {"loss_function": "MultiClass"},
                {"loss_function": "MultiClassOneVsAll"},
            ],  # class weights takes effect only with Logloss, MultiClass, MultiClassOneVsAll and user-defined loss functions
            (hashabledict({"auto_class_weights": "SqrtBalanced"}),): [
                {"loss_function": "Logloss"},
                {"loss_function": "MultiClass"},
                {"loss_function": "MultiClassOneVsAll"},
            ],  # class weights takes effect only with Logloss, MultiClass, MultiClassOneVsAll and user-defined loss functions
            (hashabledict({"boost_from_average": True}),): [
                {"loss_function": "MAE MAPE Quantile MultiQuantile RMSE".split()}
            ],  #  You can use boost_from_average only for these loss functions now:
        }
        self.allow_if_values_and = {
            (hashabledict({"posterior_sampling": True}),): [
                {"model_shrink_mode": "Constant"},  # Posterior Sampling requires Сonstant Model Shrink Mode
                {"langevin": True},  # Posterior Sampling requires Langevin boosting
            ],
        }
