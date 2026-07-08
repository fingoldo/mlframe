from __future__ import annotations

# ****************************************************************************************************************************
# Imports
# ****************************************************************************************************************************

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import Optional, Sequence, Union
from scipy.stats import uniform, loguniform, randint

from scipy.stats._distn_infrastructure import rv_continuous_frozen, rv_discrete_frozen
from sklearn.model_selection import ParameterSampler
from sklearn.model_selection import train_test_split

from pyutilz import db

import random as _stdlib_random
import pandas as pd, numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

from enum import Enum, auto

# MLTaskType = Enum("MLTaskType", ["Regression", "Multiregression", "Classification", "Multiclassification", "MultilabelClassification", "Ranking"])
class MLTaskType(Enum):
    Regression = auto()
    Multiregression = auto()
    Classification = auto()
    Multiclassification = auto()
    MultilabelClassification = auto()
    Ranking = auto()


trained_models: dict = {}


class hashabledict(dict):
    def __hash__(self):  # type: ignore[override]  # intentional: dict.__hash__ is None (unhashable); this recipe makes it hashable
        return hash(tuple(sorted(self.items())))


def check_condition(condition, params: dict) -> bool:
    if isinstance(condition, (dict, hashabledict)):
        # must hold on all the conditions!
        for cond_field, cond_value in condition.items():
            if isinstance(cond_value, list):
                for sub_value in cond_value:
                    if value_by_key(dct=params, key=cond_field, expected_value=sub_value):
                        return True
            else:
                if value_by_key(dct=params, key=cond_field, expected_value=cond_value):
                    return True
        return False
    else:
        return bool(condition)


def value_by_key(dct: dict, key, expected_value) -> bool:
    return bool(dct.get(key) == expected_value)


def check_rules(params, drop_if_rules=None, drop_if_not_rules=None, skip_if_values_or=None, allow_if_values_or=None, allow_if_values_and=None):
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


def double_check_dist_params(cand: dict, rng: Optional[np.random.Generator] = None) -> dict:
    """If some of params were not resolved by the first ParameterSampler call,
    try on a deeper level.
    """
    if rng is None:
        rng = np.random.default_rng()
    nchanged = 1
    while nchanged > 0:
        nchanged = 0
        for key, value in cand.copy().items():
            if isinstance(value, (rv_continuous_frozen, rv_discrete_frozen)):
                cand[key] = value.rvs(random_state=rng)  # list(ParameterSampler({key: value}, n_iter=1))[0][key]
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
    random_state: Union[int, np.random.Generator, None] = None,
):
    rng = np.random.default_rng(random_state)
    logger.info("Generating %s valid candidates...", n)
    approved = []
    attempts = 0
    inner_n = n
    while attempts < max_iters:
        sklearn_seed = int(rng.integers(0, 2**32 - 1))
        params_samples = list(ParameterSampler(params, n_iter=inner_n, random_state=sklearn_seed))

        approvals_this_batch = 0
        for sample in params_samples:
            attempts += 1
            double_check_dist_params(sample, rng=rng)
            if check_rules(
                sample,
                drop_if_rules=drop_if_rules,
                drop_if_not_rules=drop_if_not_rules,
                skip_if_values_or=skip_if_values_or,
                allow_if_values_or=allow_if_values_or,
                allow_if_values_and=allow_if_values_and,
            ):
                approved.append(sample)
                approvals_this_batch += 1
                if len(approved) == n:
                    return approved
            if attempts >= max_iters:
                return approved
        if approvals_this_batch == 0:
            inner_n = min(inner_n * 2, max(n * 8, 64))
    return approved


def preprocess_df(df, cat_features):

    for var in cat_features:
        df[var] = df[var].fillna("")


def prepare_trials_dataset(experiment_name: str, objective_name: str) -> pd.DataFrame:
    logger.info("Getting trials for experiment %s...", experiment_name)
    res = []
    for _id, node, params, results in db.safe_execute("select id,node,params,results from experiments where  project=%s", (experiment_name,)):
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

        cat_features = df.select_dtypes(include=["object", "string"]).columns.values.tolist()

        preprocess_df(df, cat_features)

    return df, cat_features


def normalize_probs(probs: np.ndarray):
    total = probs.sum()
    if total <= 0 or not np.isfinite(total):
        probs[:] = 1.0 / len(probs)
        return
    np.divide(probs, total, out=probs)


def objective_to_sampling_weights(predictions: np.ndarray, y: np.ndarray, minimize: bool, improving_by_atleast: float) -> np.ndarray:
    """Turn model-predicted candidate objectives into non-negative sampling weights.

    Shifts predictions to be non-negative; when minimizing, rank-inverts so low-objective candidates carry the most weight.
    When ``improving_by_atleast`` is set, zeroes the weight of candidates not predicted to improve on the observed band:
    minimize keeps candidates at/below ``y.min() + range*frac``, maximize keeps those at/above ``y.max() - range*frac``.
    """
    predictions = np.asarray(predictions, dtype=np.float64)

    if predictions.min() < 0:
        probs = predictions - predictions.min()
    else:
        probs = predictions.copy()

    if minimize:
        probs = probs.max() - probs

    if improving_by_atleast:
        if minimize:
            desired_objective = y.min() + (y.max() - y.min()) * improving_by_atleast
            bad_indices = np.where(predictions >= desired_objective)[0]
            logger.info(
                "Best and worst observed trial's objectives: %s, %s. Skipping %s candidates with predicted objective over %s",
                y.min(), y.max(), len(bad_indices), desired_objective,
            )
        else:
            desired_objective = y.max() - (y.max() - y.min()) * improving_by_atleast
            bad_indices = np.where(predictions <= desired_objective)[0]
            logger.info(
                "Best and worst observed trial's objectives: %s, %s. Skipping %s candidates with predicted objective under %s",
                y.max(), y.min(), len(bad_indices), desired_objective,
            )
        probs[bad_indices] = 0.0

    return np.asarray(probs)


def favorize_unexplored(candidates: list, probs: np.ndarray, trials: pd.DataFrame, cat_features: list, order: int = 1) -> None:
    """
    Assign higher select probabilities to combinations of order N that were never chosen yet in the trials object.
    """
    if len(cat_features) == 0 or len(trials) == 0:
        return

    logger.info("Favorizing unexplored trials...")

    already_sampled = {col: set(trials[col].unique().tolist()) for col in cat_features}
    newly_seen: dict = {col: set() for col in cat_features}

    # Iterate cat_features directly instead of walking every candidate key and testing ``param in cat_features``
    # (a list -> O(len(cat_features)) membership per key). Candidates also carry many numeric params the
    # favorization ignores, so the old per-key scan was pure waste. Bit-identical: same first-occurrence-wins
    # order, same multiplicative factor, same favorized_items count gating normalize_probs.
    _MISS = object()
    favorized_items = []
    for i in range(len(probs)):
        candidate = candidates[i]
        novel_factor = 1.0
        for param in cat_features:
            value = candidate.get(param, _MISS)
            if value is _MISS:
                continue
            seen = newly_seen[param]
            if value not in already_sampled[param] and value not in seen:
                novel_factor *= 2.0
                favorized_items.append({param: value})
                seen.add(value)
        if novel_factor > 1.0:
            probs[i] *= novel_factor

    if len(favorized_items) > 0:
        normalize_probs(probs)
        if len(favorized_items) < 5:
            logger.info("Favorized %s previously unexplored candidates: %s", f"{len(favorized_items):_}", favorized_items)
        else:
            logger.info("Favorized %s previously unexplored candidates", f"{len(favorized_items):_}")


def get_model(experiment_name: str, trials: pd.DataFrame, cat_features: list, cv: int, scoring: str, min_score: float, max_new_trials: int = 10, random_state: Union[int, np.random.Generator, None] = None):
    # trained_models[cache_key]=[fitted_model,len(trials),ml_scoring,expected_performance]
    # Include a signature of the trial feature columns (the columns the surrogate is actually fit on) in the
    # cache key. Keying on experiment_name + cat_features alone let two callers that share an experiment name
    # but optimise DIFFERENT param spaces (different feature columns) silently hit / overwrite each other's
    # fitted model -- the cached model_columns would not match the new trials, producing wrong predictions.
    feature_cols = tuple(c for c in trials.columns if c != "target")
    cache_key = (experiment_name, tuple(cat_features), feature_cols)
    should_retrain = True
    if cache_key in trained_models:
        fitted_model, num_trials, prev_scoring, expected_score, model_columns = trained_models[cache_key]
        if prev_scoring == scoring:

            if expected_score >= min_score:
                if len(trials) - num_trials > max_new_trials:
                    logger.info("Model for experiment %s found, but it needs updating with new data", experiment_name)
                else:
                    logger.info("Passing model for experiment %s found ;-)", experiment_name)
                    should_retrain = False
            else:
                if len(trials) == num_trials:
                    logger.info("Model for experiment %s found, but score was bad, and since then no new trials data arrived (", experiment_name)
                    should_retrain = False
                else:
                    logger.info("Model for experiment %s found, score was bad, but since then new trials data has arrived!", experiment_name)

    # Do NOT mutate the caller's DataFrame: ``pop`` would drop the "target" column in place, corrupting
    # the frame for any later reuse. Read the target as a view and build X from the remaining columns.
    y = trials["target"].values
    if should_retrain:
        model = CatBoostRegressor(iterations=100, task_type="CPU", cat_features=cat_features, verbose=False)

        X = trials.drop(columns=["target"])
        model_columns = X.columns

        fitted_model, expected_score = justify_estimator(
            model, X, y, refit=True, test_size=0.1, cv=cv, scoring=scoring,
            min_score=min_score, random_state=random_state, early_stopping_rounds=50,
        )

        trained_models[cache_key] = [fitted_model, len(trials), scoring, expected_score, model_columns]

    return fitted_model, model_columns, y


def justify_estimator(
    est,
    X,
    y,
    cv=3,
    refit: bool = True,
    scoring="r2",
    min_score: float = 0.6,
    test_size=0.1,
    plot=False,
    random_state: Union[int, np.random.Generator, None] = None,
    early_stopping_rounds: Optional[int] = 50,
):

    logger.info("Checking if ML gains some predictive power already on %s samples...", f"{len(y):_}")

    # Seed the gate's CV splitter from the same RNG that seeds the refit split below, so the whole function is
    # reproducible under one ``random_state``. An int ``cv`` would otherwise build an unseeded (shuffle=False)
    # KFold -- deterministic only by accident of fold order, and inconsistent with the seeded refit split.
    cv_splitter = cv
    if isinstance(cv, int):
        _gate_rng = np.random.default_rng(random_state)
        cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=int(_gate_rng.integers(0, 2**32 - 1)))
    cv_results = cross_validate(est, X, y, cv=cv_splitter, scoring=scoring)
    # Wave 21 P1: use np.nanmean so a single degenerate fold (e.g. all-one-
    # class y, sklearn returns NaN) doesn't make mean_score == NaN; pre-fix
    # the >= gate then returned False and the function silently reported
    # "ML can't be used" + returned None, even when most folds were fine.
    _test_scores = np.asarray(cv_results["test_score"], dtype=float)
    _n_nan = int(np.sum(~np.isfinite(_test_scores)))
    if _n_nan > 0:
        logger.warning(
            "ml_check_min_dataset_size: %d/%d CV folds returned non-finite "
            "scores; using nanmean over surviving folds. If most folds are "
            "degenerate, the gate may still reject; check for class "
            "imbalance / leaky split.", _n_nan, len(_test_scores),
        )
    mean_score = float(np.nanmean(_test_scores))

    if mean_score >= min_score:
        logger.info("OOS mean %s=%s, so ML can be used.", scoring, mean_score)
        if refit:
            logger.info("Fitting a model")

            is_catboost = isinstance(est, (CatBoostRegressor,)) or "catboost" in est.__class__.__module__.lower()
            if is_catboost:
                rng = np.random.default_rng(random_state)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(rng.integers(0, 2**32 - 1)))
                fit_kwargs = {"eval_set": (X_test, y_test), "plot": plot, "verbose": False}
                if early_stopping_rounds is not None:
                    fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
                est.fit(X_train, y_train, **fit_kwargs)
            else:
                est.fit(X, y)
        else:
            est = None
    else:
        logger.info("OOS mean %s=%s, so ML can't be used (yet).", scoring, mean_score)
        est = None
    return est, mean_score


# GPU_ENABLED default is False here to match CatboostParamsOptimizer.__init__ (the primary public entry): a
# mismatched default silently emitted GPU-only ctr vocab (FeatureFreq / FloatTargetMeanValue / Median borders)
# into params destined for a CPU CatBoost fit, which CatBoost then rejects or silently reinterprets.
def create_ctr_params(GPU_ENABLED: bool = False, params: Optional[dict] = None, stdlib_rng: Optional[_stdlib_random.Random] = None, random_state: Union[int, np.random.Generator, None] = None) -> Optional[list]:
    if params is None:
        params = {}
    if stdlib_rng is None:
        _rng_tmp = np.random.default_rng(random_state)
        stdlib_rng = _stdlib_random.Random(int(_rng_tmp.integers(0, 2**32 - 1)))  # nosec B311 - non-crypto sampling/jitter, not used for tokens/secrets
    res: list = []
    for main_type in (
        "Borders Buckets BinarizedTargetMeanValue Counter".split() if not GPU_ENABLED else "Borders Buckets FeatureFreq FloatTargetMeanValue".split()
    ):  # The method for transforming categorical features to numerical features.
        if stdlib_rng.random() > 0.5:
            cands = generate_valid_candidates(
                params={
                    "CtrBorderCount": [1, randint(1, 255)],
                    "CtrBorderType": ["Uniform"] if not GPU_ENABLED else "Median Uniform".split(),
                    "TargetBorderCount": [1, randint(1, 255)],
                    "TargetBorderType": ["Uniform"] if not GPU_ENABLED else "Median Uniform UniformAndQuantiles MaxLogSum MinEntropy GreedyLogSum".split(),
                },
                random_state=random_state,
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
        return None
    return res


class ParamsOptimizer:
    # Set by subclasses (e.g. CatboostParamsOptimizer) in their __init__, not here.
    params: dict
    drop_if_rules: list
    drop_if_not_rules: list
    skip_if_values_or: dict
    allow_if_values_or: dict
    allow_if_values_and: dict

    def __init__(self, random_state: Union[int, np.random.Generator, None] = None):
        # ,db_name:str=None,db_host:str=None,db_port:int=None,db_username:str=None,db_pwd:str=None,db_schema:str="public"
        self._rng = np.random.default_rng(random_state)
        self._stdlib_rng = _stdlib_random.Random(int(self._rng.integers(0, 2**32 - 1)))  # nosec B311 - non-crypto sampling/jitter, not used for tokens/secrets
        self._random_state = random_state
        # (removed a dead ``if False:`` db.connect_to_db block that referenced
        # commented-out constructor params db_name/db_host/...; it never executed
        # and the names were undefined -- surfaced by the star-import removal.)

    def create_study(self, task_id: str, stydy_type: str = "ml_estimator_hyperparameters"):
        """STUB / no-op. Study persistence is not implemented; this method intentionally does nothing and
        returns None. It exists as an extension point for a future persistence backend. Callers must not rely
        on any study being created or stored."""
        logger.warning("ParamsOptimizer.create_study is a no-op stub; no study was persisted for task_id=%s.", task_id)
        return None

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

        # Wave 31 (2026-05-20): assert -> ValueError. Pre-fix typo
        # silently skipped both branches under -O.
        if sampler not in ("random", "ml"):
            raise ValueError(f"sampler must be 'random' or 'ml'; got {sampler!r}.")

        def get_n_cands(n):
            return generate_valid_candidates(
                params=self.params,
                drop_if_rules=self.drop_if_rules,
                drop_if_not_rules=self.drop_if_not_rules,
                skip_if_values_or=self.skip_if_values_or,
                allow_if_values_or=self.allow_if_values_or,
                allow_if_values_and=self.allow_if_values_and,
                n=n,
                random_state=self._rng,
            )

        if sampler == "random":
            return get_n_cands(n)
        elif sampler == "ml":

            # ---------------------------------------------------------------------------------------------
            # get all known trials for the experiment
            # ---------------------------------------------------------------------------------------------

            trials, cat_features = prepare_trials_dataset(experiment_name=experiment_name, objective_name=objective_name)

            # if we have too few samples yet, return random, but favor yet unexplored individual cat values

            if len(trials) == 0:
                return get_n_cands(n)

            for _ in range(max_attempts):

                # ---------------------------------------------------------------------------------------------
                # sample much more valid candidates
                # ---------------------------------------------------------------------------------------------

                candidates = get_n_cands(n=max(n * search_space_multiplier, search_space_minsize))

                if len(candidates) <= 1:
                    logger.warning("Nothing to sample for experiment %s", experiment_name)
                    return candidates

                probs = np.ones(len(candidates), np.float64) / len(candidates)
                if len(candidates) > n:

                    if len(trials) >= min_samples_for_ml:

                        # ---------------------------------------------------------------------------------------------
                        # score them with the ML model
                        # ---------------------------------------------------------------------------------------------

                        fitted_model, model_columns, y = get_model(
                            experiment_name=experiment_name, trials=trials, cat_features=cat_features, cv=ml_cv, scoring=ml_scoring, min_score=ml_min_score,
                            random_state=self._rng,
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

                            probs = objective_to_sampling_weights(predictions=predictions, y=y, minimize=minimize, improving_by_atleast=improving_by_atleast)
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
                    return self._rng.choice(candidates, n, replace=False, p=probs)

            return candidates[:n]

    def report_trial_results(self, objectives: dict):
        """STUB / no-op. Result persistence is NOT implemented: the passed ``objectives`` are silently dropped.
        This method exists as an extension point for a future persistence backend; do not rely on it to store
        anything. A warning is emitted so callers notice the results are not being saved."""
        logger.warning("ParamsOptimizer.report_trial_results is a no-op stub; %d objective(s) were NOT persisted.", len(objectives) if objectives else 0)
        return None


class CatboostParamsOptimizer(ParamsOptimizer):
    def __init__(
        self,
        GPU_ENABLED: bool = False,
        groups: bool = False,
        need_training_continuation: bool = False,
        task: MLTaskType = MLTaskType.Regression,
        params_override: Optional[dict] = None,
        delete_params: Optional[Sequence] = None,
        random_state: Union[int, np.random.Generator, None] = None,
    ):

        super().__init__(random_state=random_state)
        if params_override is None:
            params_override = {}
        if delete_params is None:
            delete_params = []
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
            "learning_rate": loguniform(1e-3, 0.3),  # Alias: eta
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
            "model_size_reg": randint(1, 10),
            "one_hot_max_size": [None, randint(2, 300)],
            "ctr_leaf_count_limit": (
                [
                    None,
                    randint(10, 100),
                ]
                if not GPU_ENABLED
                else [None]
            ),  # The maximum number of leaves with categorical features. If the quantity exceeds the specified value a part of leaves is discarded. This option reduces the resulting model size and the amount of memory required for training. Note that the resulting quality of the model can be affected.
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
            "sampling_unit": (
                ["Object", "Group"] if groups else ["Object"]
            ),  # The sampling scheme. #No groups in dataset. Please disable sampling or use per object sampling
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
        # CatBoost per-task loss_function/eval_metric options are not swept here; see CatBoost docs for the full menu per task type:
        #   Regression: MAE/MAPE/Poisson/Quantile/RMSE/LogLinQuantile/LogCosh/Lq/Expectile/Tweedie/Huber (eval also FairLoss/SMAPE/R2/MSLE/MedianAbsoluteError/NumErrors)
        #   Multiregression: MultiRMSE/MultiRMSEWithMissingValues
        #   Classification: Logloss/CrossEntropy (eval also Precision/Recall/F1/BalancedAccuracy/MCC/Accuracy/AUC/NormalizedGini/BrierScore/...)
        #   Multiclassification: MultiClass/MultiClassOneVsAll; MultilabelClassification: MultiLogloss/MultiCrossEntropy
        #   Ranking: PairLogit/YetiRank/YetiRankPairwise/StochasticRank/QueryCrossEntropy/QueryRMSE/QuerySoftMax (eval also PFound/NDCG/DCG/MAP/ERR/MRR/QueryAUC)

        if params_override:
            self.params.update(params_override)

        self.params["simple_ctr"] = [
            None,
            create_ctr_params(GPU_ENABLED=GPU_ENABLED, params=self.params, stdlib_rng=self._stdlib_rng, random_state=self._rng),
        ]  # ['Borders:CtrBorderCount=15:CtrBorderType=Uniform:TargetBorderCount=1:TargetBorderType=MinEntropy:Prior=0/1:Prior=0.5/1:Prior=1/1','Counter:CtrBorderCount=15:CtrBorderType=Uniform:Prior=0/1'], # Quantization settings for simple categorical features. Use this parameter to specify the principles for defining the class of the object for regression tasks. By default, it is considered that an object belongs to the positive class if its' label value is greater than the median of all label values of the dataset.
        self.params["combinations_ctr"] = [
            None,
            create_ctr_params(GPU_ENABLED=GPU_ENABLED, params=self.params, stdlib_rng=self._stdlib_rng, random_state=self._rng),
        ]  # Quantization settings for combinations of categorical features.

        for key in delete_params:
            if key in self.params:
                del self.params[key]
        self.drop_if_rules = [
            {
                "conditions": [GPU_ENABLED],
                "fields": ["sampling_frequency"],
            },  # Error: change of option sampling_frequency is unimplemented for task type GPU and was not default in previous run
            {"conditions": [{"bootstrap_type": "No"}], "fields": ["subsample"]},  # Error: you shouldn't provide bootstrap options if bootstrap is disabled
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
