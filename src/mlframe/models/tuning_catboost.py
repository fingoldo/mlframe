"""``CatboostParamsOptimizer``: the concrete ``ParamsOptimizer`` for CatBoost.

Carved out of ``tuning.py`` (monolith split, CLAUDE.md "sibling re-export" convention) to keep the
parent module under the 1000 LOC budget. Re-exported from ``tuning.py`` unchanged.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
from scipy.stats import loguniform, randint, uniform

from ._tuning_types import HashableDict, MLTaskType
from .tuning_rules import ParamsOptimizer, create_ctr_params

__all__ = ["CatboostParamsOptimizer"]


class CatboostParamsOptimizer(ParamsOptimizer):
    """Concrete ``ParamsOptimizer`` for CatBoost.

    Builds a large parameter-distribution search space spanning CatBoost's float/int/categorical/bool
    hyperparameters plus CTR (categorical-feature encoding) config, along with the full drop/skip/allow rule
    set encoding CatBoost's real parameter-compatibility constraints. See ``__init__`` for the details.
    """

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
        """Build the CatBoost hyperparameter search space + compatibility rule set.

        Populates ``self.params`` with distributions spanning CatBoost's float/int/categorical/bool
        hyperparameters (device-gated where CPU/GPU support differs) plus randomly-generated CTR encoding
        strings (``simple_ctr``/``combinations_ctr`` via ``create_ctr_params``), then ``params_override``
        (merged on top) and ``delete_params`` (removed) are applied. Also builds ``self.drop_if_rules``,
        ``self.drop_if_not_rules``, ``self.skip_if_values_or``, ``self.allow_if_values_or`` and
        ``self.allow_if_values_and``, encoding CatBoost's real parameter-compatibility constraints (e.g.
        ``posterior_sampling`` requires Constant Model Shrink Mode + Langevin boosting; MVS bootstrap supports
        only per-object sampling; Newton leaf estimation is unsupported for MAE/MAPE/Quantile losses).
        """

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
            "grow_policy": [
                "SymmetricTree",
                "Depthwise",
                "Lossguide",
            ],  # Tree growing policy. Required as a rule-DSL companion field: allow_if_values_or requires SymmetricTree when boosting_type=='Ordered', and drop_if_not_rules only keeps max_leaves when this is 'Lossguide'.
            "model_shrink_mode": [
                "Constant",
                "Decreasing",
            ],  # How the model shrink rate is applied. Required as a rule-DSL companion field: allow_if_values_and requires 'Constant' when posterior_sampling=True. Dropped again for GPU via drop_if_rules (unimplemented there).
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
            (HashableDict({"sampling_frequency": "PerTreeLevel"}),): [
                {"grow_policy": "Lossguide"}
            ],  # PerTreeLevel sampling is not supported for Lossguide grow policy.
            (HashableDict({"bootstrap_type": "Poisson"}),): [not GPU_ENABLED],  # Error: poisson bootstrap is not supported on CPU
            (not GPU_ENABLED,): [{"leaf_estimation_backtracking": "Armijo"}],  # Backtracking type Armijo is supported only on GPU
            (HashableDict({"approx_on_full_history": True}),): [
                {"boosting_type": [None, "Plain"]}
            ],  # Can't use approx-on-full-history with Plain boosting-type
            (HashableDict({"leaf_estimation_method": "Newton"}),): [
                {
                    "loss_function": ["MAE", "MAPE", "Quantile", "MultiQuantile", "LogLinQuantile"]
                    + [el for el in self.params.get("loss_function", []) if el.startswith("Lq")]
                }
            ],  # Newton leaves estimation method is not supoprted for MAPE loss function # Newton leaves estimation method is not supoprted for Lq loss function with q < 2 !TODO
            (HashableDict({"leaf_estimation_method": "Exact"}),): [
                {"approx_on_full_history": True}
            ],  # ApproxOnFullHistory option is not available within Exact method on CPU.
        }

        self.allow_if_values_or = {
            (HashableDict({"bootstrap_type": "MVS"}),): [{"sampling_unit": "Object"}],  # MVS bootstrap supports per object sampling only.
            (HashableDict({"boosting_type": "Ordered"}),): [{"grow_policy": "SymmetricTree"}],  # Ordered boosting is not supported for nonsymmetric trees.
            (not GPU_ENABLED,): [
                {"score_function": "Cosine"},
                {"score_function": "L2"},
                {"score_function": None},
            ],  # Only Cosine and L2 score functions are supported for CPU.
            (HashableDict({"leaf_estimation_method": "Exact"}),): [
                {"loss_function": "Quantile"},
                {"loss_function": "MAE"},
                {"loss_function": "MAPE"},
                {"loss_function": "LogCosh"},
            ],  # Exact method is only available for Quantile, MAE, MAPE and LogCosh loss functions.
            (HashableDict({"auto_class_weights": "Balanced"}),): [
                {"loss_function": "Logloss"},
                {"loss_function": "MultiClass"},
                {"loss_function": "MultiClassOneVsAll"},
            ],  # class weights takes effect only with Logloss, MultiClass, MultiClassOneVsAll and user-defined loss functions
            (HashableDict({"auto_class_weights": "SqrtBalanced"}),): [
                {"loss_function": "Logloss"},
                {"loss_function": "MultiClass"},
                {"loss_function": "MultiClassOneVsAll"},
            ],  # class weights takes effect only with Logloss, MultiClass, MultiClassOneVsAll and user-defined loss functions
            (HashableDict({"boost_from_average": True}),): [
                {"loss_function": "MAE MAPE Quantile MultiQuantile RMSE".split()}
            ],  #  You can use boost_from_average only for these loss functions now:
        }
        self.allow_if_values_and = {
            (HashableDict({"posterior_sampling": True}),): [
                {"model_shrink_mode": "Constant"},  # Posterior Sampling requires Сonstant Model Shrink Mode
                {"langevin": True},  # Posterior Sampling requires Langevin boosting
            ],
        }
