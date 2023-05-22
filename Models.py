
# Web imports
import time, datetime
import urllib, requests
from datetime import datetime

#Charting
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pandas.plotting import lag_plot
from pandas.plotting import scatter_matrix

# Perf & other Utils
import joblib

from copy import deepcopy

from sklearn.utils import shuffle
from IPython.display import display

from shutil import rmtree
from tempfile import mkdtemp
from sklearn.externals.joblib import Memory

# Data science libs
import scipy
import pandas as pd

import numpy as np
from numpy import *


# Dustributions
from scipy.stats import expon, uniform, norm
from scipy.stats import randint as sp_randint

# Make the graphs a bit prettier, and bigger
pd.set_option('display.max_columns', 100)

# Nonlinear feature generation
from sklearn.kernel_approximation import RBFSampler,Nystroem

# Dimensionality reduction
from sklearn.manifold import TSNE,Isomap
from sklearn.decomposition import PCA, KernelPCA, NMF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# mlxtender declares!
from mlxtend.feature_selection import ColumnSelector
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

# Preprocessing
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, \
    Normalizer, Binarizer

# Clustering
#from sklearn.mixture import GMM

# Metrics
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, roc_auc_score, brier_score_loss

####################################################################
#RMSLE can be used when you don’t want to penalize huge differences when both the values are huge numbers.
#Also, it can be used when you want to penalize under estimates more than over estimates.
def rmse_loss(y, y0):
    #assert len(y) == len(y0)
    return np.sqrt(np.mean((y-y0)**2))
rmse_score = make_scorer(rmse_loss, greater_is_better=False)
def rmsle_loss(y, y0):
    #assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(np.clip(y0,0,None)), 2)))
rmsle_score = make_scorer(rmsle_loss, greater_is_better=False)
def rmsle_K(y, y0):
    return K.sqrt(K.mean(K.square(tf.log1p(y) - tf.log1p(y0))))
    
class log_uniform():        
    def __init__(self, a=-1, b=0, base=10):
        self.loc = a
        self.scale = b - a
        self.base = base

    def rvs(self, size=None, random_state=None):
        myuniform = uniform(loc=self.loc, scale=self.scale)
        if size is None:
            return np.power(self.base, myuniform.rvs(random_state=random_state))
        else:
            return np.power(self.base, myuniform.rvs(size=size, random_state=random_state))
# Calibration
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# Model evaluation & tuning

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate

# learning Machines
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR, NuSVC, LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, ExtraTreeClassifier, export_graphviz
from sklearn.linear_model import SGDClassifier, SGDRegressor, LogisticRegression, RidgeClassifier, \
    PassiveAggressiveClassifier

from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, BaggingClassifier, \
    AdaBoostClassifier, VotingClassifier

#Regressors
from sklearn.linear_model import LinearRegression,ElasticNet

def ProbaScoreProxy(y_true, y_probs, class_idx, proxied_func, **kwargs):
    return proxied_func(y_true, y_probs[:, class_idx], **kwargs)

##########################################################################################################################################################################################################
# Dummy classifier
##########################################################################################################################################################################################################
def TrainDummyClassifier(x, y):
    the_pipe = DummyClassifier(strategy='stratified')
    the_pipe.fit(x, y)
    return the_pipe


##########################################################################################################################################################################################################
### TPOT?
##########################################################################################################################################################################################################
def TraintTPOT(cv):
    from tpot import TPOTClassifier
    from sklearn.model_selection import train_test_split

    tpot = TPOTClassifier(generations=5, population_size=10, scoring='f1', verbosity=2, n_jobs=31, cv=cv)
    tpot.fit(x, y_up)
    print(tpot.score(x, y_up))

    # ShowClassifierPerformance(tpot,"Tpot")
    print(metrics.classification_report(y_up, tpot.predict(x)))
    print(metrics.classification_report(
        np.where(BTC_XMR[BTC_XMR.index >= dtFirstTestDate].up_val > fCurChangePercent, 1, 0),
        tpot.predict(BTC_XMR[BTC_XMR.index >= dtFirstTestDate][sActiveInputs].values)))

    from tpot import TPOTClassifier
    from sklearn.model_selection import train_test_split
    tpot2 = TPOTClassifier(generations=7, population_size=50, scoring='f1', verbosity=2, n_jobs=30, cv=outer_tscv)
    tpot2.fit(x, y_up)
    print(tpot2.score(x, y_up))


def TrainPipelineNoOpt(x, y):
    # base_model= LogisticRegression(max_iter=10000,class_weight=my_class_weight)
    # base_model= SVC(kernel='rbf',degree=2,class_weight=my_class_weight,probability=True)
    # base_model= KNeighborsClassifier()
    base_model = DecisionTreeClassifier(class_weight=my_class_weight)
    the_pipe = Pipeline([('scaler', StandardScaler()), ('learner', base_model)])
    # CV Evaluation
    # scores_up = cross_val_score(the_pipe, x, y_up, cv=outer_tscv,scoring=my_metric,n_jobs=nJobs)
    # print("Simple pipeline crossvalidation Quality: %0.2f (+/- %0.2f): %0.2f" % (scores_up.mean(), scores_up.std() * 2,np.median(scores_up)))
    the_pipe.fit(x, y)
    # print(the_pipe.named_steps['linear_model'].oob_score_)
    return the_pipe



def ReplaceDictKeys(the_dict, replace_starting_with_what, replace_with):
    import copy
    for keys in copy.deepcopy(the_dict):
        if keys.startswith(replace_starting_with_what):
            new_key = keys.replace(replace_starting_with_what, replace_with)
            the_dict[new_key] = the_dict.pop(keys)
            print(keys + " replaced with " + new_key)
    return the_dict


def PlotMyCustomCalibrationPlot(clf, x, y):
    probs = np.floor(pipe_up.predict_proba(x)[:, 1] * 100).astype(int)
    i = 0
    counts = np.zeros(100, dtype=int)
    fires = np.zeros(100, dtype=int)
    for p in probs:
        counts[p] += 1
        fires[p] += y[i]
        i += 1
    probs = np.sort(np.unique(probs))
    freqs = np.zeros(len(probs))
    i = 0
    for p in probs:
        freqs[i] = fires[p] / counts[p]
        i += 1
    emp_inds = np.argwhere(freqs == 0)
    # Now lets delete cells with zero freqs
    probs = np.delete(probs, emp_inds)
    freqs = np.delete(freqs, emp_inds)
    plt.plot(probs, freqs);


def plot_roc(clf, X, y, title, show_calibration=True):
    if hasattr(clf, "predict_proba"):
        predictions = clf.predict_proba(X)[:, 1]
    elif hasattr(clf, "decision_function"):
        predictions = clf.decision_function(X)
    else:
        print("weird situation:" + str(clf))

    false_positive_rate, recall, thresholds = roc_curve(y, predictions)
    if show_calibration:
        fraction_of_positives, mean_predicted_value = calibration_curve(y, predictions, n_bins=50, normalize=True)
    roc_auc = auc(false_positive_rate, recall)
    plt.title('ROC for ' + title)
    plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.ylabel('Recall')
    plt.xlabel('Fall-out')
    if show_calibration:
        plt.plot(mean_predicted_value, fraction_of_positives);
    plt.show()
    print(metrics.classification_report(y, clf.predict(X)))


def ShowClassifierPerformance(clf, x_tr, y_tr, x_ts, y_ts, title):
    if len(x_tr)>2:
        plot_roc(clf, x_tr, y_tr, title + " train")
    if len(x_ts)>2:
        plot_roc(clf, x_ts, y_ts, title + " test")

def PlotTrueVsPredicted(y_tr, y_pr, title):
    a, = plt.plot(y_tr, marker='o', label='True');
    b, = plt.plot(y_pr, marker='*', label='Predicted');
    plt.title(title)
    plt.legend(handles=[a, b])
    plt.show()

def ShowRegressorPerformance(clf, x_tr, y_tr, x_ts, y_ts, title):
    if len(x_tr)>2:
        x_sm, _, y_sm, _ = train_test_split(x_tr, y_tr, train_size=min(len(y_tr),100), shuffle=True)
        y_pr=clf.predict(x_sm)
        PlotTrueVsPredicted(y_sm,y_pr, title + " train")
    if len(x_ts)>2:
        x_sm, _, y_sm, _ = train_test_split(x_tr, y_tr, train_size=min(len(y_tr),100), shuffle=True)
        y_pr=clf.predict(x_sm)
        PlotTrueVsPredicted(y_sm,y_pr, title + " test")

##########################################################################################################################################################################################################
# Stability island checking:
##########################################################################################################################################################################################################
def PlotOptimumIsland():
    scores = [x[1] for x in gs.grid_scores_]
    scores = np.array(scores).reshape(5, 4)

    plt.matshow(scores)
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar
    plt.xticks(np.arange(4), param_grid['gamma'])
    plt.yticks(np.arange(5), param_grid['C'])


##########################################################################################################################################################################################################
# Optional saving of data to csv
##########################################################################################################################################################################################################
def SavePredictionsToCsv():
    np.savetxt("pr.csv", predicted_up[:, 1], delimiter=",")
    np.savetxt("dt.csv", sample.index, delimiter=",")
    np.savetxt("x.csv", sample[['open', 'high', 'low', 'close']].values, delimiter=",")

    # When performing classification you often want to predict not only the class label, but also the associated probability.
    # This probability gives you some kind of confidence on the prediction. However, not all classifiers provide well-calibrated
    # probabilities, some being over-confident while others being under-confident. Thus, a separate calibration of predicted probabilities
    # is often desirable as a postprocessing.
    # http://scikit-learn.org/stable/auto_examples/calibration/plot_calibration.html#sphx-glr-auto-examples-calibration-plot-calibration-py


def TrainPipelineWithRandomizedOptimumSearch(x, y, x_t, y_t, modelName, optimizer_scoring, inner_cv, outer_tscv,
                                             active_inputs, my_days_without_retraining, my_most_important_recent_days,
                                             my_class_weight="balanced", optimizer_refit_by="roc_auc_weighted", groups=None,
                                             calibration_method=None, use_caching=False, measure_duration=True,
                                             num_iters=10, do_cross_validate=False, use_scalers=True,
                                             use_dimreducers=False, tsne_dim=0, use_polynoms=False,
                                             columnselector_n_features=0, nJobs=1, kernel_approx_dim=0,
                                             rfe_features_to_select=0,verbose=1):
    ##########################################################################################################################################################################################################
    # Optimized pipeline with caching
    ##########################################################################################################################################################################################################
    # Params:
    # calibration_method=None|'isotonic'|'sigmoid'
    ##########################################################################################################################################################################################################
    if measure_duration:
        startTime = datetime.now()
        if do_cross_validate:
            if verbose>0:
                print("Starting cross-validation of "+modelName+": " + str(startTime))
    if use_caching:
        pipe_cachedir = mkdtemp()
        pipe_mem = Memory(cachedir=pipe_cachedir, verbose=0)
    else:
        pipe_mem = None
        ##########################################################################################################################################################################################################
        # Feature selection
        ##########################################################################################################################################################################################################
        # combined_features = FeatureUnion([("original_features",original_features), ("pca", pca), ("nmf", nmf), ("lda", lda)])
    ##########################################################################################################################################################################################################
    # Data scaling
    ##########################################################################################################################################################################################################
    if use_scalers:
        common_scalers = [None, StandardScaler(), MinMaxScaler(feature_range=(1, 2))]  # ,RobustScaler()
    ##########################################################################################################################################################################################################
    # All kinds of machine learners
    ##########################################################################################################################################################################################################

    ##########################################################################################################################################################################################################
    # DO NOT USE!! SLOW TRAINING

    ##########################################################################################################################################################################################################
    if modelName == "LinearSVC":
        base_model = LinearSVC(class_weight=my_class_weight)  # !!!!'LinearSVC' object has no attribute 'predict_proba'
        resulting_param_grid = dict(m__C=loguniform(-6, 6, 50), m__loss=['squared_hinge'], m__penalty=['l1', 'l2'],
                                    m__dual=[True, False], m__fit_intercept=[True, False],
                                    m__intercept_scaling=norm(1, 1), m__tol=expon(scale=.1))
    elif modelName == "GaussianNB":
        base_model = GaussianNB()
        resulting_param_grid = dict()
    elif modelName == "KNeighborsClassifier":
        # !!! DO NOT USE: bad accuracy
        base_model = KNeighborsClassifier()
        resulting_param_grid = dict(m__n_neighbors=sp_randint(1, 5), m__weights=['uniform', 'distance'],
                                    m__algorithm=['ball_tree', 'kd_tree'], m__leaf_size=sp_randint(1, 50),
                                    m__p=[1, 2])
    elif modelName == "RadiusNeighborsClassifier":
        # !!! DO NOT USE: bad accuracy
        # !!! No neighbors found for test samples [0, 1], you can try using larger radius, give a label for outliers, or consider removing them from your dataset.
        # RadiusNeighborsClassifier' object has no attribute 'predict_proba'
        base_model = RadiusNeighborsClassifier()
        resulting_param_grid = dict(
            m__weights=['uniform', 'distance'], m__algorithm=['ball_tree', 'kd_tree', 'brute'],
            m__leaf_size=sp_randint(10, 50), m__p=[1, 2], m__radius=uniform(5, 9))
    elif modelName == "SVC":
        # !!! DO NOT USE: terribly slow
        base_model = SVC(kernel='rbf', degree=2, class_weight=my_class_weight, probability=False, cache_size=20000)
        resulting_param_grid = dict(m__C=loguniform(-6, 6, 50), m__gamma=loguniform(-6, 6, 50),
                                    m__shrinking=[True, False], m__tol=expon(scale=.001),
                                    m__kernel=['poly', 'rbf', 'sigmoid'])  # 'linear'
    elif modelName == "NuSVC":
        # !!! DO NOT USE: bad accuracy AND slow
        base_model = NuSVC(probability=False, class_weight=my_class_weight)
        resulting_param_grid = dict(
            m__nu=uniform(0.01, 0.09), m__gamma=loguniform(-6, 6, 50),
            m__shrinking=[True, False], m__tol=expon(scale=.1), m__kernel=['rbf'])
    elif modelName == "LogisticRegression":
        base_model = LogisticRegression(max_iter=150, class_weight=my_class_weight)  # possible solvers : {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’},,n_jobs=nJobs
        resulting_param_grid = dict(m__C=loguniform(-6, 6, 50), m__tol=expon(scale=.1), m__penalty=['l1',
                                                                                                    'l2'])  # lbfgs is the fastest solver with almost highest train score :-)
    elif modelName == "SGDClassifier":
        base_model = SGDClassifier(max_iter=1000, class_weight=my_class_weight,
                                   n_jobs=nJobs)  # probability estimates are not available for loss='perceptron',probability estimates are not available for loss='hinge'
        resulting_param_grid = dict(m__loss=['log', 'modified_huber', 'perceptron', 'hinge', 'squared_hinge'],
                                    m__penalty=['l1', 'l2', 'elasticnet'], m__alpha=expon(scale=0.01),
                                    m__l1_ratio=uniform(0, 1), m__eta0=uniform(0.01, 1), m__tol=expon(scale=0.01),
                                    m__learning_rate=['constant', 'invscaling', 'optimal'], m__power_t=norm(0.5))
    elif modelName == "RidgeClassifier":
        base_model = RidgeClassifier(
            class_weight=my_class_weight)  # 'RidgeClassifier' object has no attribute 'predict_proba'
        resulting_param_grid = dict(m__alpha=loguniform(-6, 6, 50), m__tol=expon(scale=0.01))
    elif modelName == "QuadraticDiscriminantAnalysis":
        base_model = QuadraticDiscriminantAnalysis()
        resulting_param_grid = dict(m__tol=expon(scale=0.01))
    elif modelName == "PassiveAggressiveClassifier":
        # !!! DO NOT USE: bad accuracy
        # 'PassiveAggressiveClassifier' object has no attribute 'predict_proba'
        base_model = PassiveAggressiveClassifier(class_weight=my_class_weight)
        resulting_param_grid = dict(m__C=loguniform(-6, 6, 50), m__tol=expon(scale=0.01))
    elif modelName == "MLPClassifier":
        base_model = MLPClassifier()
        resulting_param_grid = dict(
            m__activation=['identity', 'logistic', 'tanh', 'relu'],
            m__alpha=expon(scale=.0001), m__learning_rate=['constant', 'invscaling', 'adaptive'],
            m__learning_rate_init=uniform(0.001,0.1), m__power_t=norm(0.5), m__shuffle=[False, True],
            m__tol=expon(scale=.00001), m__momentum=norm(0.9),
            m__nesterovs_momentum=[True, False], m__early_stopping=[True, False]
        )
    elif modelName == "DecisionTreeClassifier":
        base_model = DecisionTreeClassifier(criterion='entropy', class_weight=my_class_weight)
        resulting_param_grid = dict(m__max_depth=sp_randint(1, 110), m__max_features=uniform(0, 1),
                                    m__min_samples_split=uniform(0, 1),
                                    m__min_samples_leaf=uniform(0, 0.5), m__splitter=['best', 'random'])
    elif modelName == "RandomForestClassifier":
        base_model = RandomForestClassifier(criterion='entropy', class_weight=my_class_weight, n_estimators=100,
                                            n_jobs=nJobs)
        resulting_param_grid = dict(m__bootstrap=[True, False],
                                    m__max_depth=sp_randint(1, 110), m__max_features=uniform(0, 1),
                                    m__min_samples_split=uniform(0, 1),
                                    m__min_samples_leaf=uniform(0, 0.5))
    elif modelName == "GradientBoostingClassifier":
        # !!! DO NOT USE: bad accuracy
        base_model = GradientBoostingClassifier(n_estimators=100)
        resulting_param_grid = dict(m__max_depth=sp_randint(1, 11), m__max_features=uniform(0, 1),
                                    m__min_samples_leaf=sp_randint(1, 11), m__loss=['deviance', 'exponential'],
                                    m__learning_rate=uniform(0, 1), m__subsample=uniform(0, 1),
                                    m__criterion=['friedman_mse', 'mse', 'mae'],
                                    m__min_weight_fraction_leaf=uniform(0, 0.5), m__min_samples_split=sp_randint(2, 11),
                                    m__min_impurity_decrease=uniform(0, 0.5),
                                    m__max_leaf_nodes=sp_randint(3, x.shape[1]))
    elif modelName == "ExtraTreeClassifier":
        base_model = ExtraTreeClassifier(class_weight=my_class_weight)
        resulting_param_grid = dict(m__max_depth=sp_randint(1, 11), m__max_features=uniform(0, 1),
                                    m__min_samples_split=sp_randint(2, 11),
                                    m__min_samples_leaf=uniform(0, 0.5), m__splitter=['best', 'random'],
                                    m__criterion=['gini', 'entropy'])
    elif modelName == "ExtraTreesClassifier":
        # !!! DO NOT USE: bad accuracy (but WHY??? single ExtraTreeClassifier performs well)
        base_model = ExtraTreesClassifier(class_weight=my_class_weight, n_estimators=100)
        resulting_param_grid = dict(m__bootstrap=[True, False], m__max_depth=sp_randint(1, 11),
                                    m__max_features=uniform(0, 1), m__min_samples_split=sp_randint(2, 11),
                                    m__min_samples_leaf=uniform(0, 0.5), m__criterion=['gini', 'entropy'])
    elif modelName =="LinearRegression":
        base_model = LinearRegression()
        resulting_param_grid=dict(m__fit_intercept=[True, False])
    elif modelName == "DecisionTreeRegressor":
        base_model = DecisionTreeRegressor()
        resulting_param_grid = dict(m__max_depth=sp_randint(1, 110), m__max_features=uniform(0, 1),
                                    m__min_samples_split=uniform(0, 1),
                                    m__min_samples_leaf=uniform(0, 0.5), m__splitter=['best', 'random'], m__criterion=['mse', 'mae','friedman_mse'])
    else:
        print("Unknown modelName")
        return
    if len(resulting_param_grid) == 0:
        if use_dimreducers == False:
            if use_scalers == True:
                num_iters = len(common_scalers)
            else:
                num_iters = 1
                ##########################################################################################################################################################################################################
    # Create pipeline itself based on passed options
    ##########################################################################################################################################################################################################
    if calibration_method:
        pipe_steps = [('m', CalibratedClassifierCV(base_model, cv=10, method=calibration_method))]
    else:
        pipe_steps = [('m', base_model)]

    if rfe_features_to_select > 0:
        from sklearn.feature_selection import RFE

        rf = RandomForestClassifier()
        rfe = RFE(estimator=rf, n_features_to_select=rfe_features_to_select, step=1)

        pipe_steps.insert(0, ('rfe', rfe))

    ##########################################################################################################################################################################################################
    # Last come feature-replacing transformers
    ##########################################################################################################################################################################################################
    if use_dimreducers:
        ##########################################################################################################################################################################################################
        ### Possible dimensionality reduction steps
        ##########################################################################################################################################################################################################
        # This dataset is way too high-dimensional. Maybe PCA can produce good features?

        common_dim_reducers = [PCA()]  # KernelPCA(kernel='rbf'), NMF(), LinearDiscriminantAnalysis()

        pipe_steps.insert(0, ('dr', PCA()))

        resulting_param_grid['dr'] = common_dim_reducers

        # PCA
        resulting_param_grid['dr__whiten'] = [True, False]
        resulting_param_grid['dr__n_components'] = uniform(0.2, 0.3)

        # KernelPCA
        # nJobs=max(nJobs,4)
        # resulting_param_grid['dr__n_components']=[2,3] #uniform(0.7,0.3)
        # resulting_param_grid['dr__kernel']=['rbf'] #'linear','poly','precomputed',,'sigmoid','cosine'
        # resulting_param_grid['dr__degree']=[2]

        # NMF
        # resulting_param_grid['dr__n_components']=sp_randint(1,x.shape[1])
        # resulting_param_grid['dr__l1_ratio']=uniform(0,1)
        # resulting_param_grid['dr__alpha']=loguniform(-6,3,50)
        # resulting_param_grid['dr__tol']=expon(scale=0.01)
        # resulting_param_grid['dr__beta_loss']=['frobenius', 'kullback-leibler']
        # resulting_param_grid['dr__solver']=['cd', 'mu']
    if use_scalers:
        pipe_steps.insert(0, ('sc', StandardScaler()))
        resulting_param_grid['sc'] = common_scalers
    ##########################################################################################################################################################################################################
    # Then feature-selectors
    ##########################################################################################################################################################################################################
    if columnselector_n_features > 0:
        pipe_steps.insert(0, ('cs', ColumnSelector()))
        from itertools import combinations
        all_comb = []
        for size in [columnselector_n_features]:  # range(1, columnselector_n_features):
            all_comb += list(combinations(range(x.shape[1]), r=size))
        resulting_param_grid['cs__cols'] = all_comb
        ##########################################################################################################################################################################################################
    # Additional feature-generation transformers come first
    ##########################################################################################################################################################################################################
    if use_polynoms:
        pipe_steps.insert(0, ('pl', PolynomialFeatures(degree=2, interaction_only=True)))
    if tsne_dim > 0:
        pipe_steps.insert(0, ('cl', Isomap(n_components=tsne_dim)))
    if kernel_approx_dim > 0:
        pipe_steps.insert(0, ('ka', Nystroem(n_components=kernel_approx_dim)))
        resulting_param_grid['ka__gamma'] = loguniform(-6, 6, 50)

    the_pipe = Pipeline(pipe_steps, memory=pipe_mem)

    # base_model= MLPClassifier(hidden_layer_sizes=(10,5))

    # base_model=GradientBoostingClassifier()
    # base_model=BaggingClassifier()
    # base_model=AdaBoostClassifier()

    BaggingClassifier_param_grid = dict(m__n_estimators=sp_randint(10, 1000), m__max_samples=uniform(0, 1),
                                        m__max_features=uniform(0, 1), m__bootstrap=[True, False],
                                        m__bootstrap_features=[False, True])

    AdaBoostClassifier_param_grid = dict(m__n_estimators=sp_randint(10, 1000), m__learning_rate=uniform(0, 1))

    ##########################################################################################################################################################################################################
    # If calibration was specified, need to replace simple m with base_estimator in param dict
    ##########################################################################################################################################################################################################
    if calibration_method:
        resulting_param_grid = ReplaceDictKeys(resulting_param_grid, "m__", "m__base_estimator__")
    ##########################################################################################################################################################################################################
    # print(resulting_param_grid)

    grid_search = RandomizedSearchCV(the_pipe, iid=False, return_train_score=False, refit=optimizer_refit_by,
                                     error_score=0.0,
                                     param_distributions=resulting_param_grid, cv=inner_cv, scoring=optimizer_scoring,
                                     verbose=3, n_jobs=nJobs, n_iter=num_iters)

    if do_cross_validate:
        cvl = cross_validate(grid_search, x, y, groups=groups, cv=outer_tscv,scoring=optimizer_refit_by, return_train_score=False) #n_jobs=nJobs НЕ СТАВИМ!!
    else:
        grid_search.fit(x, y, groups=groups)
        # print(grid_search.best_score_)
        print("best_estimator:" + str(grid_search.best_estimator_))

    # Controlling accuracy by varying the threshold
    # print(metrics.classification_report(np.where(BTC_XMR[BTC_XMR.index>=dtFirstTestDate].up_val>fCurChangePercent,1,0),np.where(grid_search.predict_proba(BTC_XMR[BTC_XMR.index>=dtFirstTestDate][sActiveInputs].values)[:,0]>0.4,0,1)))

    # cvres=pd.DataFrame (grid_search.cv_results_)
    # cvres.sort_values(by='mean_test_AUC') .tail()

    # scores_up = cross_validate(grid_search, x, y_up, cv=outer_tscv,scoring=my_metric,n_jobs=nJobs, verbose=0,return_train_score=True)
    # print("Tuned pipeline crossvalidation Quality: %0.2f (+/- %0.2f): %0.2f" % (scores_up.mean(), scores_up.std() * 2,np.median(scores_up)))
    if use_caching:
        rmtree(pipe_cachedir)

    # cv_scores_up=pd.DataFrame (scores_up)
    if measure_duration:
        if verbose > 0:
            timeElapsed = datetime.now() - startTime
            print('Time elapsed (hh:mm:ss) {}'.format(timeElapsed))
    if verbose > 0:
        print("NumIters: " + str(num_iters))
        print("Inputs: " + str(active_inputs))
    if do_cross_validate:
        if verbose>0:
            r = cvl['test_score']
            med = np.median(r)
            plt.plot(r, 'g^');
            plt.plot(np.ones(len(r)) * 0.5, 'r--');
            plt.plot(np.ones(len(r)) * med, 'b-.');
            the_title = "Cross-validated %s,\n retrain each %d days on last %d days\n scaled=%s, dimreduced=%s,\n pre_polynomed=%s,pre_clustered=%s,pre_kernelized=%s" % (
            modelName, my_days_without_retraining, my_most_important_recent_days, str(use_scalers), str(use_dimreducers),
            str(use_polynoms), str(tsne_dim), str(kernel_approx_dim))
            if columnselector_n_features > 0:
                the_title = the_title + "\n Try selection of " + str(columnselector_n_features) + " best features"
            plt.title(the_title)
            plt.ylabel(optimizer_refit_by, style='italic');
            plt.xlabel("months", style='italic')
            plt.text(len(r) - 1, 0.01, '%s: med=%.2f +- %0.2f' % (optimizer_refit_by, med, np.std(r)),
                     verticalalignment='bottom', horizontalalignment='right', color='blue', fontsize=12)
            plt.axis([-1, len(r), 0, 1])

        return cvl
    else:
        if verbose>0:
            if 'Classifier' in modelName:
                ShowClassifierPerformance(grid_search, x, y, x_t, y_t, modelName)
            else:
                ShowRegressorPerformance(grid_search, x, y, x_t, y_t, modelName)

        return grid_search


def loguniform(low=0, high=1, size=None):
    return np.exp(np.random.uniform(low, high, size))
