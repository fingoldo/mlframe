import numpy as np
import pandas as pd
##########################################################################################################################################################################################################
#Databases
##########################################################################################################################################################################################################
from sqlalchemy import create_engine
##########################################################################################################################################################################################################
#Plotting
##########################################################################################################################################################################################################
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
##########################################################################################################################################################################################################
#Helpers
##########################################################################################################################################################################################################
import json

import time
from datetime import datetime

import joblib

# winsound is Windows-only; use conditional import
try:
    import winsound
except ImportError:
    winsound = None  # type: ignore

from os import path
import os,platform,socket,sys
# Removed: from past.builtins import execfile (Python 2 deprecated)

import itertools
from collections import OrderedDict

from shutil import rmtree
from tempfile import mkdtemp
from joblib import Memory  # Changed from sklearn.externals.joblib (deprecated)

##########################################################################################################################################################################################################
#Sparse data
##########################################################################################################################################################################################################
from scipy import sparse
from scipy.sparse import csr_matrix,hstack
##########################################################################################################################################################################################################
# Dustributions
##########################################################################################################################################################################################################
from mlframe.Models import log_uniform
from scipy.stats import expon, uniform, norm
from scipy.stats import randint as sp_randint
##########################################################################################################################################################################################################
#Preprocessing
##########################################################################################################################################################################################################
from sklearn.impute import SimpleImputer as Imputer  # Changed from sklearn.preprocessing.Imputer (deprecated)
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import Normalizer,Binarizer,KBinsDiscretizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer
##########################################################################################################################################################################################################
#Feature engineering
##########################################################################################################################################################################################################
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_extraction.text import TfidfTransformer
from mlframe.FeaturesEngineering import TargetEncodingTransformer
# Nonlinear feature generation
from sklearn.kernel_approximation import RBFSampler,Nystroem,AdditiveChi2Sampler,SkewedChi2Sampler

##########################################################################################################################################################################################################
#Outliers detection    
from sklearn.ensemble import IsolationForest
##########################################################################################################################################################################################################
#Dimensionality Reduction
##########################################################################################################################################################################################################
from sklearn.neural_network import BernoulliRBM
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA,KernelPCA,NMF,TruncatedSVD,FastICA
from sklearn.random_projection import GaussianRandomProjection,SparseRandomProjection
##########################################################################################################################################################################################################
#Manifold learning
##########################################################################################################################################################################################################
from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
#from MulticoreTSNE import MulticoreTSNE
##########################################################################################################################################################################################################
#Clustering
##########################################################################################################################################################################################################
#from sklearn.mixture import GaussianMixture
##########################################################################################################################################################################################################
#Feature selection
##########################################################################################################################################################################################################
from sklearn.feature_selection import mutual_info_classif,mutual_info_regression
from sklearn.feature_selection import VarianceThreshold,SelectKBest,RFECV,SelectFromModel
##########################################################################################################################################################################################################
#Metrics
##########################################################################################################################################################################################################
from sklearn import metrics
from sklearn.metrics import make_scorer,get_scorer
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score,brier_score_loss

roc_auc_weighted = make_scorer(roc_auc_score, average='weighted')

def brier_and_precision_score(y_true, y_prob,max_loss,min_precision):
    brier=brier_score_loss(y_true, y_prob, sample_weight=None, pos_label=None)
    if brier>max_loss:
        return 0
    precision=precision_score(y_true, y_prob>0.5)
    if precision<min_precision:
        return 0
    return precision-brier

from mlframe.Models import rmsle_score,rmsle_loss,rmse_score,rmse_loss

brier_and_precision_scorer= make_scorer(brier_and_precision_score,needs_proba=True,greater_is_better=True)

####################################################################
def plot_pr(clf, X, y, title, show_calibration=True,saveAs=None,thresh=0.5):
    import warnings
    from scipy import stats
    from sklearn.metrics import brier_score_loss
    from sklearn.metrics import precision_recall_curve,average_precision_score
    warnings.filterwarnings(action='ignore')
    sns.set()
    if clf is None:
        predictions=X
    else:
        if hasattr(clf, "predict_proba"):
            output=clf.predict_proba(X)
            if output.shape[1]==2:
                predictions = output[:, 1]
            else:
                predictions = output[:, 0]
        elif hasattr(clf, "decision_function"):
            predictions = clf.decision_function(X)
        else:
            print("weird situation, classifier has no method for predictions: " + str(clf))
    #pr    
    step_kwargs = {'step': 'post'}
    precision, recall, _ = precision_recall_curve(y, predictions)
    dummy_predictions=np.array(list(stats.mode(y)[0])*len(y))
    dummy_precision, dummy_recall, _ = precision_recall_curve(y,dummy_predictions)
    pr_auc=average_precision_score(y, predictions)
    dummy_pr_auc=average_precision_score(y, dummy_predictions)
    #Plot
    plt.clf();
    plt.title('PRC/ROC for ' + title+', BrierLoss=%.3f' % brier_score_loss(y,predictions))
    
    plt.step(recall, precision, color='b', alpha=0.4,label='PR AUC=%.2f/%.2fR' % (pr_auc,dummy_pr_auc),where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    plt.step(dummy_recall, dummy_precision, color='r', alpha=0.1,where='post')
    plt.fill_between(dummy_recall, dummy_precision, alpha=0.1, color='r', **step_kwargs)  

    plt.xlabel('Recall/Fall-out')
    plt.ylabel('Precision/Recall')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    
    #roc 
    false_positive_rate, recall, thresholds = roc_curve(y, predictions)
    roc_auc = auc(false_positive_rate, recall)
    plt.plot(false_positive_rate, recall, 'b', label='ROC AUC=%0.3f' % roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')    
    plt.legend(loc='lower right')
    
    if show_calibration:
        fraction_of_positives, mean_predicted_value = calibration_curve(y, predictions, n_bins=50, normalize=True)
        plt.plot(mean_predicted_value, fraction_of_positives,'--g');
        
    if saveAs:plt.gcf().savefig(saveAs,bbox_inches='tight')
    plt.show(block = False);plt.pause(0.001)
    print(metrics.classification_report(y, predictions>thresh))

def plot_roc(clf, X, y, title, show_calibration=True,saveAs=None):
    if hasattr(clf, "predict_proba"):
        output=clf.predict_proba(X)
        if output.shape[1]==2:
            predictions = output[:, 1]
        else:
            predictions = output[:, 0]
    elif hasattr(clf, "decision_function"):
        predictions = clf.decision_function(X)
    else:
        print("weird situation, classifier has no method for predictions: " + str(clf))

    false_positive_rate, recall, thresholds = roc_curve(y, predictions)
    if show_calibration:
        fraction_of_positives, mean_predicted_value = calibration_curve(y, predictions, n_bins=50, normalize=True)
    roc_auc = auc(false_positive_rate, recall)
    plt.clf();
    plt.title('ROC for ' + title)
    plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.ylabel('Recall')
    plt.xlabel('Fall-out')
    if show_calibration:plt.plot(mean_predicted_value, fraction_of_positives);
    if saveAs:plt.gcf().savefig(saveAs,bbox_inches='tight')
    plt.show(block = False);plt.pause(0.001)
    print(metrics.classification_report(y, clf.predict(X)))


def plot_regressor_performance(clf,X,y,regressor_name,scorer,sample_size=250,theSeed=None,saveAs=None):
    scType=type(scorer).__name__
    if scType in ['_PredictScorer','_ProbaScorer']:
        the_scorer=scorer
    elif scType=='str':
        the_scorer=get_scorer(scorer)
    else:
        logger.warn ("Unexpected type of scorer: "+scType)
    perf=the_scorer(clf,X,y)
    if the_scorer._sign==-1:perf=-perf
    
    l=min(sample_size,len(y))
    if theSeed:
        np.random.seed(theSeed)
    idx=np.random.choice(l,l,replace=False)
    predictions=clf.predict(X[idx,:])
    plt.clf();
    plt.title('%s, local perf=%0.3f' %(regressor_name,perf))
    plt.plot(y[idx],label='Real')
    plt.plot(predictions,label='Predicted')
    plt.legend() #loc='lower right'    
    if saveAs:plt.gcf().savefig(saveAs,bbox_inches='tight')
    plt.show(block = False);plt.pause(0.001)
    
##########################################################################################################################################################################################################
# Calibration
##########################################################################################################################################################################################################
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

##########################################################################################################################################################################################################
#Pipelining/composing
##########################################################################################################################################################################################################
from sklearn.pipeline import Pipeline,FeatureUnion,make_pipeline,make_union
from sklearn.compose import ColumnTransformer,make_column_transformer
from sklearn.preprocessing import FunctionTransformer
##########################################################################################################################################################################################################
#Hyperparameters tuning
##########################################################################################################################################################################################################
from sklearn.model_selection import RandomizedSearchCV
from skopt.space import Real, Integer, Categorical
from skopt import BayesSearchCV
##########################################################################################################################################################################################################
#Generalization and model selection
##########################################################################################################################################################################################################
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.model_selection import KFold,StratifiedKFold
##########################################################################################################################################################################################################
#Regressors
##########################################################################################################################################################################################################
####################################################################
#Dummy:
####################################################################
from sklearn.dummy import DummyRegressor
####################################################################
#Linear:
####################################################################
from sklearn.linear_model import SGDRegressor,LinearRegression,Lasso,Ridge,ElasticNet,Lars,LassoLars,OrthogonalMatchingPursuit,BayesianRidge
from sklearn.linear_model import ARDRegression,PassiveAggressiveRegressor,RANSACRegressor,TheilSenRegressor,HuberRegressor
####################################################################
#Non-linear:
####################################################################
#Support vectors
from sklearn.svm import SVR,LinearSVR
#neural networks
from sklearn.neural_network import MLPRegressor
#Trees
from sklearn.tree import DecisionTreeRegressor
#Neighbours
from sklearn.neighbors import KNeighborsRegressor,RadiusNeighborsRegressor
#Ensembled
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor,LGBMClassifier
from xgboost import XGBRegressor,XGBClassifier
####################################################################################################################################        
#Keras
####################################################################################################################################        
from keras import regularizers
#class_weights = class_weight.compute_class_weight('balanced',np.unique(y),y)
def GetKerasModel(numLayers=3,numNeurons=100,activation='relu',regularizer=regularizers.l2(0.001),dropoutRate=0.3,loss='mean_squared_logarithmic_error'):
    base_model = Sequential([Dense(numNeurons,input_dim=x.shape[1],kernel_initializer='normal',activation=activation,kernel_regularizer=regularizer,bias_regularizer=regularizer,activity_regularizer=regularizer)])
    base_model.add(BatchNormalization())
    base_model.add(Dropout(dropoutRate))  
    for i in range(numLayers):
        base_model.add(Dense(numNeurons,kernel_initializer='normal',activation=activation,kernel_regularizer=regularizer,bias_regularizer=regularizer,activity_regularizer=regularizer))
        base_model.add(BatchNormalization())
        base_model.add(Dropout(dropoutRate))  
    base_model.add(Dense(1, kernel_initializer='normal', activation='linear',kernel_regularizer=regularizer,bias_regularizer=regularizer,activity_regularizer=regularizer))
    #base_model.add(BatchNormalization())
    # Compile model
    base_model.compile(loss=loss, optimizer='adam',) # metrics=['mse']
    return base_model
def GetKerasRegressor():
    return KerasRegressor(build_fn=GetKerasModel,epochs=200,batch_size=400,verbose=2,validation_split=0.1)

##########################################################################################################################################################################################################
#Classifiers
##########################################################################################################################################################################################################

####################################################################
#Dummy:
####################################################################
from sklearn.dummy import DummyClassifier

####################################################################
#Linear:
####################################################################
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
#from sklearn.linear_model import RidgeClassifier #Terrible performance
from sklearn.linear_model import SGDClassifier,LogisticRegression,PassiveAggressiveClassifier,Perceptron

####################################################################
#Non-linear:
####################################################################
#Support vectors
from sklearn.svm import SVC, NuSVC, LinearSVC

#Other
from sklearn.gaussian_process import GaussianProcessClassifier

#neural networks
from sklearn.neural_network import MLPClassifier

from keras.models import Sequential
from sklearn.utils import class_weight
from keras.callbacks import EarlyStopping
from keras.layers import Dense,Dropout,BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier,KerasRegressor

#Trees
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.tree import export_graphviz

#Neighbours
from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier

#Classic methods
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#Ensembled

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier,plot_importance

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
####################################################################################################################################
#Metaestimators
####################################################################################################################################
from sklearn.compose import TransformedTargetRegressor

####################################################################################################################################
#Own transformers
####################################################################################################################################
from sklearn.base import BaseEstimator,TransformerMixin, ClassifierMixin,RegressorMixin
from sklearn.preprocessing import PowerTransformer
LogPPlusOneTransformer=FunctionTransformer(np.log1p,np.expm1,validate=False)
DoubleLogPPlusOneTransformer=FunctionTransformer(lambda x: np.log1p(np.log1p(x)),lambda x: np.expm1(np.expm1(x)),validate=False)
#class LogPPlusOneTransformer(BaseEstimator,TransformerMixin):    
#    def __init__(self):
#        pass        
#    def transform(self, X, *_):        
#        return np.log1p(X)    
#    def inverse_transform(self, X, *_):        
#        return np.expm1(X)    
#    def fit(self, *_):
#        return self
####################################################################################################################################
#Own estimators
####################################################################################################################################
from mlframe import CustomEstimators     
####################################################################################################################################
#TSFRESH
####################################################################################################################################
from tsfresh import select_features
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
####################################################################################################################################
#Distributed computing
####################################################################################################################################
from dask.distributed import LocalCluster
from dask.diagnostics import ProgressBar
####################################################################################################################################
#LOGGING
####################################################################################################################################
import logging,logging.config
logger = logging.getLogger(__name__)    
####################################################################################################################################
# load the logging configuration
####################################################################################################################################
log_file_path = path.join(path.dirname(path.abspath("__file__")), 'logging.ini')
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
logging.getLogger('matplotlib').setLevel(logging.WARN)
logging.getLogger('urllib3').setLevel(logging.WARN)
logging.getLogger('backoff').setLevel(logging.WARN)
logger.setLevel(logging.INFO)
####################################################################################################################################
#Current lib settings
####################################################################################################################################
#sns.set()


####################################################################################################################################
#Experimenting with quick classifiers
####################################################################################################################################

def TestEstimator(clf,standardize=False,tfidf=False,pre_pipeline=[],target_encoding=False,fit=True,svd=False,TargetTransformer=None,scoring='roc_auc',show_roc_auc=True,**fit_params):    
    #global x_train,x_test    

    steps=[        
        #SelectKBest(score_func=mutual_info_classif,k=40)
        #FeatureUnion([
            #('AsIs',SelectKBest(k='all')),
            #         n_jobs=-1),
        #Imputer(missing_values='NaN', strategy='mean', axis=0)
            #])
    ]
    for obj in pre_pipeline:
        steps.append(obj)
    if target_encoding:
        steps.append(TargetEncodingTransformer(verbose=2))
    if tfidf:
        steps.append(TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True))            
        
    if svd:
        from sklearn.decomposition import TruncatedSVD
        steps.append(TruncatedSVD(n_components=40))    
        
    if standardize:
        steps.append(StandardScaler(with_mean=True))    
        
    steps.append(clf)
    pipe=make_pipeline(*steps)
    if TargetTransformer:
        pipe=TransformedTargetRegressor(regressor=pipe,transformer=TargetTransformer)
    if fit:
        pipe.fit(x_train,y_train,**fit_params)
        if show_roc_auc:
            if clf._estimator_type=='classifier':
                plot_roc(pipe,x_test,y_test,'Test ROC for '+type(clf).__name__,show_calibration=False)        
            else:
                plot_regressor_performance(pipe,x_test,y_test,type(clf).__name__,scorer=scoring,theSeed=1000)
    else:
        cv_scores=cross_val_score(pipe,x_train,y_train,scoring=scoring,cv=10,n_jobs=-1)
        print("CV scoring: %0.3f \u00B1 %0.3f" % (cv_scores.mean(),cv_scores.std()))   
    if winsound is not None:
        try:
            winsound.PlaySound('sound.wav', winsound.SND_FILENAME)
        except (RuntimeError, OSError):
            pass
    return pipe
    
########################################################################################################################################################################################################
#Inits
########################################################################################################################################################################################################
nJobs=-1
constPercentOfPositiveClass=0.05        #np.mean(y)
inner_cv = KFold(n_splits=5,shuffle=True)
outer_cv = KFold(n_splits=4,shuffle=True)

#from sklearn.linear_model import Lasso,ElasticNet,Lars,LassoLars,ARDRegression,Perceptron,PassiveAggressiveRegressor,RANSACRegressor,TheilSenRegressor,HuberRegressor

########################################################################################################################################################################################################
#Regressors
########################################################################################################################################################################################################

#Linear regressors
BayesianRidge_param_grid = dict(             
                            m__alpha_1=expon(scale=0.001),
                            m__alpha_2=expon(scale=0.001),
                            m__tol=expon(scale=0.01),
                            m__lambda_1=expon(scale=0.001),
                            m__lambda_2=expon(scale=0.001))

LinearRegression_param_grid=dict(m__fit_intercept=[True, False])
OrthogonalMatchingPursuit_param_grid=dict(m__tol=expon(scale=0.1))
Ridge_param_grid = dict(
                            m__alpha=expon(scale=0.001),
                            m__tol=expon(scale=0.01))

SGD_param_grid = dict(m__loss=['log', 'modified_huber'],
                            m__penalty=['l1', 'l2', 'elasticnet'], 
                            m__alpha=expon(scale=0.01),
                            m__l1_ratio=uniform(0, 1), 
                            m__eta0=expon(scale=0.01),
                            m__tol=expon(scale=0.01),
                            m__learning_rate=['constant', 'invscaling', 'optimal'], 
                            m__power_t=norm(0.5),
                            m__epsilon=uniform(0,1),
                            m__average=[True,False])
SGD_Regr_param_grid = dict(m__loss=['squared_loss','huber','epsilon_insensitive','squared_epsilon_insensitive'],
                            m__penalty=['l1', 'l2', 'elasticnet'], 
                            m__alpha=expon(scale=0.001),
                            m__l1_ratio=uniform(0, 0.2), 
                            m__eta0=expon(scale=0.05),                            
                            m__learning_rate=['constant', 'invscaling', 'optimal'],                             
                            m__epsilon=uniform(0,0.2),m__tol=[None],
                            m__average=[True,False])
#m__power_t=uniform(0, 0.3),
#SGDRegressor(alpha=0.0001, average=False, early_stopping=True, epsilon=0.1,eta0=0.01, fit_intercept=True, l1_ratio=0.15,learning_rate='invscaling', loss='squared_loss', max_iter=1000,n_iter=None, n_iter_no_change=5, penalty='l2', power_t=0.25

#Non-linear regressors
MLP_param_grid = dict(m__activation =['logistic','tanh','relu'],
                            m__solver=['lbfgs', 'sgd', 'adam'], 
                            m__alpha=expon(scale=0.001),
                            m__learning_rate_init=expon(scale=0.001),                             
                            m__tol=expon(scale=0.01),
                            m__learning_rate=['constant', 'invscaling', 'adaptive'], 
                            m__power_t=uniform(0.5),
                            m__momentum=uniform(0,1),
                            m__nesterovs_momentum=[True,False],
                            m__shuffle=[True,False],
                            m__early_stopping=[True,False],
                            m__beta_1=uniform(0.8,1-0.81),
                            m__beta_2=uniform(0.9,1-0.91),
                            m__epsilon=expon(scale=0.000001),
                     )

Catboost_param_grid =dict(m__depth=sp_randint(1, 16+1),
                      m__n_estimators=sp_randint(1000, 2000), 
                      m__bootstrap_type=['Bayesian','Bernoulli','No'], 
                      m__l2_leaf_reg=sp_randint(1, 100),
                                  
                      
                      m__one_hot_max_size=[1,2,3,4],                      
                      m__leaf_estimation_method=['Newton','Gradient'],
                      
                      m__od_type=['IncToDec','Iter'],
                      m__border_count=sp_randint(1, 255+1),
                      m__feature_border_type=['Median','Uniform','UniformAndQuantiles','MaxLogSum','MinEntropy','GreedyLogSum']
                     )
#m__random_strength=uniform(0,2), m__subsample=uniform(0,1),           m__fold_len_multiplier=uniform(1.5, 2),  Error: bayesian bootstrap doesn't support taken fraction option
#m__colsample_bylevel=uniform(0, 1), Error: you shoudn't provide bootstrap options if bootstrap is disabled
#m__bagging_temperature=[0,1], bagging temperature available for bayesian bootstrap only
#m__sampling_frequency=['PerTree','PerTreeLevel'] DELETED! Is not recognized
#Poisson (supported for GPU only)                        
#The weights are sampled from exponential distribution if the value of this parameter is set to “1”. All weights are equal to 1 if the value of this parameter is set to “0”. 
#m__learning_rate=uniform(0.001, 0.3-0.001),

RF_param_grid =  dict(m__bootstrap=[True, False],                      
                    m__max_depth=sp_randint(1, 110),                    
                    m__max_features=uniform(0.001, 1-0.001),
                    m__min_samples_split=uniform(0.001, 1-0.001),
                    m__min_samples_leaf=uniform(0.001, 0.5-0.001),
                    m__criterion=['mse', 'mae'],
                    m__n_estimators=sp_randint(100, 1000),
                    m__max_leaf_nodes=[None,10,50,100,500,1000],                    
                     )#min_impurity_split=expon(loc=-0.001,scale=0.001)

XGB_model = XGBClassifier(objective = 'binary:logistic',eval_metric='auc',early_stopping_rounds=5,
                                   n_jobs=32,scale_pos_weight=1/constPercentOfPositiveClass) 
XGB_param_grid = dict(m__booster=['gbtree','gblinear','dart'],
                    m__n_estimators=sp_randint(10, 300+10), 
                    m__learning_rate=uniform(0.01,0.25-0.01),
                    m__max_depth=sp_randint(3,15+1),
                    m__subsample=uniform(0.5, 1-0.5),
                    
                    m__colsample_bytree=uniform(0.5, 0.5),
                    m__colsample_bylevel=uniform(0.5, 0.5),
                    m__reg_lambda=expon(scale=0.1),
                    m__reg_alpha=expon(scale=0.01),                                        
                    m__min_child_weight=uniform(0.5, 1),
                    m__gamma=expon(scale=0.01),                                                         
                    m__objective=['reg:linear','reg:gamma','reg:tweedie']
                                          
)
#m__grow_policy=['depthwise','lossguide'],m__max_leaves=sp_randint(0,50), m__tree_method=['auto','hist'],  
XGB_bayes_param_grid = dict(m__n_estimators=Integer(10, 200), 
                    m__learning_rate=Real(0.01,0.25),
                    m__max_depth=Integer(3,15),
                    m__subsample=Real(0.5, 1),
                    
                    m__colsample_bytree=Real(0.5,1),
                    m__colsample_bylevel=Real(0.5, 1),
                    m__reg_lambda=Real(1e-5,0.7,prior='log-uniform'),
                    m__reg_alpha=Real(1e-5,0.07,prior='log-uniform'),                                        
                    m__min_child_weight=Real(0.5, 1.5),
                      m__gamma=Real(1e-5,0.07,prior='log-uniform')
                                          
)

Lgbm_param_grid = dict(m__boosting_type=['gbdt'],
                    m__n_estimators=sp_randint(10, 1000+1), 
                    m__num_leaves=sp_randint(2,3500),
                    m__learning_rate=uniform(0.01,0.15-0.01),
                    m__max_depth=sp_randint(-1,25),
                    m__min_child_weight=expon(0.01),
                    m__min_split_gain=expon(0.001),
                    m__min_child_samples=sp_randint(10,100),
                       
                    m__colsample_bytree=uniform(0.5, 0.5),
                    m__reg_lambda=expon(scale=0.1),
                    m__reg_alpha=expon(scale=0.01),                                           
)
#,'dart','goss'
#[LightGBM] [Fatal] Cannot use bagging in GOSS
#m__feature_fraction=uniform(0.01,1-0.01),m__bagging_fraction=uniform(0.01,1-0.01),m__bagging_freq=sp_randint(1,10),

SVR_model = SVR(cache_size=20000)
SVR_param_grid = dict(
                    m__C=log_uniform(-6, 6),
                    m__gamma=log_uniform(-6, 6),
                    m__shrinking=[True, False],
                    m__tol=expon(scale=.001),
                    m__kernel=['poly', 'rbf', 'sigmoid' ]) 
LinearSVR_param_grid = dict(m__C=log_uniform(-6, 6),
                            m__loss=['squared_epsilon_insensitive'], 
                            m__dual=[True, False], 
                            m__fit_intercept=[True, False],
                            m__intercept_scaling=uniform(0, 1), 
                            m__tol=expon(scale=.001))
#'squared_epsilon_insensitive''epsilon_insensitive',
SVR_bayes_param_grid = dict(
                    m__C=Real(1e-6,1e6,prior='log-uniform'),
                    m__gamma=Real(1e-6,1e6,prior='log-uniform'),
                    m__shrinking=[True, False],
                    m__tol=Real(1e-5,0.07,prior='log-uniform'),                    
                    m__kernel=[ 'rbf', 'sigmoid' ]) 
#'poly',m__degree=Integer(2,4),

KNN_param_grid = dict(
                    m__n_neighbors=sp_randint(1, 15),
                    m__weights=['uniform', 'distance'],
                    m__algorithm=['ball_tree', 'brute'],
                    m__leaf_size=sp_randint(1, 50),
                    m__p=[1, 2],
                    m__metric=[
            'euclidean','manhattan','chebyshev','minkowski','mahalanobis',
            'hamming','canberra','braycurtis',
            'jaccard' ,'matching','dice','rogerstanimoto','russellrao','sokalmichener','sokalsneath' ]
)
#Metric 'kulsinski' not valid for algorithm 'kd_tree'
#Metric 'matching' not valid for algorithm 'kd_tree'
#ValueError: weighted minkowski requires a weight vector `w` to be given. 'wminkowski',
#in sklearn.neighbors.dist_metrics.SEuclideanDistance.__init__() TypeError: __init__() takes exactly 1 positional argument (0 given)
KNN_bayes_param_grid = dict(
                    m__n_neighbors=Integer(1, 16),
                    m__weights=['uniform', 'distance'],
                    m__algorithm=['ball_tree',  'brute'],
                    m__leaf_size=Integer(1, 51),
                    m__p=[1, 2],
                    m__metric=[
            'euclidean','manhattan','chebyshev','minkowski','mahalanobis',
            'hamming','canberra','braycurtis',
            'jaccard' ,'matching','dice','rogerstanimoto','russellrao','sokalmichener','sokalsneath' ]
)
RNN_param_grid = dict(
                    m__radius=uniform(0.01,2-0.01),
                    m__weights=['uniform', 'distance'],
                    m__algorithm=['ball_tree', 'kd_tree', 'brute'],
                    m__leaf_size=sp_randint(1, 50),
                    m__p=[1, 2],
                    m__metric=[
            'euclidean','manhattan','chebyshev','minkowski','wminkowski','seuclidean','mahalanobis',
            'hamming','canberra','braycurtis',
            'jaccard' ,'matching','dice','kulsinski','rogerstanimoto','russellrao','sokalmichener','sokalsneath' ]
)

########################################################################################################################################################################################################
#Classifiers
########################################################################################################################################################################################################
MultinomialNB_model = MultinomialNB()
MultinomialNB_param_grid = dict(m__alpha=expon(scale=5),m__fit_prior=[True,False])  

BernoulliNB_model = BernoulliNB(alpha=7,binarize=0)
BernoulliNB_param_grid = dict(m__alpha=expon(scale=5),m__fit_prior=[True,False])
Perceptron_param_grid = dict(m__penalty=[None,'l1', 'l2', 'elasticnet'],
                            m__alpha=expon(scale=0.001),
                            m__max_iter=[1000],
                            m__tol=expon(scale=0.01),
                            m__eta0=uniform(0, 1))
SGD_Model=SGDClassifier(max_iter=300, class_weight='balanced',
                           n_jobs=nJobs)  # probability estimates are not available for loss='perceptron',probability estimates are not available for loss='hinge',squared_hinge
LogReg_model = LogisticRegression(max_iter=200, class_weight='balanced')
# possible solvers : {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’},,n_jobs=nJobs
# lbfgs is the fastest solver with almost highest train score :-)
LogReg_param_grid = dict(m__C=log_uniform(-6, 6), 
                  m__tol=expon(scale=.1), 
                  m__penalty=['l1','l2'],
                  m__dual=[True,False],
                  m__solver=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
                 )
####################################################################################################################################        

def EstimateModel(base_model,param_grid,
bNormalization=False,bBinarization=False,bKBinsDiscretization=False,
    bUseScalers=True,bOptionallyNoScaler=False,bUseTfidf=False,
        polynomialFeatures=None,nonLinearFeatures=None,bOptionallyNoNonLinearFeatures=True,bAddExtraFeatures=True,
          dimReducer=None,bOptionallyNoDimReducer=True,
           featureSelection=None,bOptionallyNoFeatureSelector=True,TargetTransformer=None,
            maxNComponents=0,nDifferentComponents=5,
            target_encoding=False,
                fit=True,bUseBayesianSearch=False,num_iters=25,                       
                    scoring=None,inner_cv=5,outer_cv=3,groups=None,
                            bUseCaching=False,nJobs=-1,randomSeed=None,
                               savePrediction=False,savePerformancePlot=False,outputDir=None
                                   ):    
    isRegression=(isinstance(base_model,RegressorMixin)) or ('regressor' in type(base_model).__name__.lower())
    
    
    if maxNComponents==0:
        maxNComponents=x_train.shape[1]-1    
    ####################################################################################################################################        
    #Define correct score of a failed estimator
    ####################################################################################################################################                                            
    if scoring:
        if scoring._sign==1:
            error_score=0 
        else: 
            error_score=-1000
    else:
        error_score=0
    ####################################################################################################################################                                            
    if fit:    
        ####################################################################################################################################        
        #What will be the filename of this model?
        ####################################################################################################################################                                              
        if TargetTransformer:
            if hasattr(TargetTransformer,'func'):
                trName=TargetTransformer.func.__name__
            else:
                trName='noname'
        else:
            trName='None'
        if type(scoring)==str:
            scName=scoring
        else:
            scName=scoring._score_func.__name__
        filename='-'.join([str(int(e)) if isinstance(e,bool) else str(e) for e in [type(base_model).__name__,bNormalization,bBinarization,bKBinsDiscretization,
                bUseScalers,bOptionallyNoScaler,bUseTfidf,polynomialFeatures,nonLinearFeatures,
                    bOptionallyNoNonLinearFeatures,bAddExtraFeatures,dimReducer,bOptionallyNoDimReducer,
                        featureSelection,bOptionallyNoFeatureSelector,trName,bUseBayesianSearch,num_iters,scName,
                                                                                            groups]])
        filename=filename.replace(':','-')
        
        #Get file path
        
        if outputDir:
            filepath=outputDir+'\\'+filename+'.pkl'
            if os.path.isfile(filepath):
                logger.info("Skipping model "+filename+": file already exists.")
                return
        else:
            filepath='models\\'+datasetName+'\\'+filename+'.pkl'
    
    ####################################################################################################################################        
    #Caching
    ####################################################################################################################################        
    if bUseCaching:
        pipe_cachedir ="S:\\temp\\joblib\\" 
        #mkdtemp()
        pipe_mem = Memory(cachedir=pipe_cachedir, verbose=0)    
    else:
        pipe_mem=None
    ####################################################################################################################################        
    
    pipe_steps = [('m', base_model)]
    
    resulting_param_grid=param_grid.copy()

    ####################################################################################################################################        
    #Feature Selection
    ####################################################################################################################################        
    if featureSelection in ['SelectKBest','RFECV','SelectFromModel','TryAllMethods']:
        pipe_steps.insert(0,('fs',SelectKBest())) #Any will be Ok here! It's just a placeholder for now.
        if bOptionallyNoFeatureSelector: 
            fs_params=[None]
        else:
            fs_params =[]
        if featureSelection=='SelectKBest' or featureSelection=='TryAllMethods':            
            if isRegression:
                mi_func=mutual_info_regression
            else:
                mi_func=mutual_info_classif
            fs_params+=[SelectKBest(score_func=mi_func,k=int(k)) for k in np.unique([int(k) for k in np.linspace(1,maxNComponents,nDifferentComponents)])]
        elif featureSelection=='RFECV' or featureSelection=='TryAllMethods':
            if isRegression:
                fs_params+=[RFECV(estimator=RandomForestRegressor(n_estimators=100,n_jobs=16), step=1, cv=KFold(2),scoring=scoring)]
            else:
                fs_params+=[RFECV(estimator=RandomForestClassifier(n_estimators=100), step=1, cv=StratifiedKFold(2),scoring=scoring)]
        elif featureSelection=='SelectFromModel' or featureSelection=='TryAllMethods':
            if isRegression:
                fs_params+=[SelectFromModel(estimator=RandomForestRegressor(n_estimators=100), threshold='1.25*median')]
            else:
                fs_params+=[SelectFromModel(estimator=RandomForestClassifier(n_estimators=100), threshold='1.25*median')]
        resulting_param_grid['fs'] = fs_params
    ####################################################################################################################################    
    #Dimensionality reducer
    ####################################################################################################################################    
    if dimReducer in ['PCA','KernelPCA','LDA','NMF','TruncatedSVD','FastICA','Isomap','UMAP','GaussianRandomProjection','SparseRandomProjection','RandomTreesEmbedding','BernoulliRBM','TryAllMethods']:        
        if bAddExtraFeatures:
            pipe_steps.insert(0,('dr',FeatureUnion([('basic',SelectKBest(k='all')),('dr',PCA())]))) #Any will be Ok here! It's just a placeholder for now.
        else:
            pipe_steps.insert(0,('dr',PCA())) #Any will be Ok here! It's just a placeholder for now.        
        if bOptionallyNoDimReducer: 
            dr_params=[None]
        else:
            dr_params =[]
        if dimReducer=='PCA' or dimReducer=='TryAllMethods':            
            dr_params+=[PCA(whiten=b,n_components=int(k)) for k in np.unique([int(k) for k in np.linspace(2,maxNComponents,nDifferentComponents)]) for b in [True, False]]
        elif dimReducer=='KernelPCA' or dimReducer=='TryAllMethods':            
            dr_params+=[KernelPCA(kernel=b,n_components=int(k)) for k in np.unique([int(k) for k in np.linspace(2,maxNComponents,nDifferentComponents)]) for b in ['poly','rbf','sigmoid','cosine']]
        elif dimReducer=='LDA' or dimReducer=='TryAllMethods':            
            dr_params+=[LinearDiscriminantAnalysis(solver=b,n_components=int(k)) for k in np.unique([int(k) for k in np.linspace(2,maxNComponents,nDifferentComponents)]) for b in ['svd','lsqr','eigen']]
        elif dimReducer=='TruncatedSVD' or dimReducer=='TryAllMethods':            
            dr_params+=[TruncatedSVD(n_components=int(k)) for k in np.unique([int(k) for k in np.linspace(2,maxNComponents,nDifferentComponents)])]            
        elif dimReducer=='FastICA' or dimReducer=='TryAllMethods':            
            dr_params+=[FastICA(algorithm=a,fun=f,whiten=b,n_components=int(k)) for k in np.unique([int(k) for k in np.linspace(2,maxNComponents,nDifferentComponents)]) for b in [True, False] for f in['logcosh','exp','cube'] for a in ['parallel','deflation']]            
        elif dimReducer=='Isomap' or dimReducer=='TryAllMethods':            
            dr_params+=[Isomap(n_jobs=8,n_neighbors=n,n_components=int(k)) for k in np.unique([int(k) for k in np.linspace(2,maxNComponents,nDifferentComponents)]) for n in range(2,30)]            
        elif dimReducer=='UMAP' or dimReducer=='TryAllMethods':            
            dr_params+=[UMAP(n_neighbors=n,n_components=int(k)) for k in np.unique([int(k) for k in np.linspace(2,maxNComponents,nDifferentComponents)]) for n in range(2,30)]                        
        elif dimReducer=='GaussianRandomProjection' or dimReducer=='TryAllMethods':            
            dr_params+=[GaussianRandomProjection(n_components=int(k)) for k in np.unique([int(k) for k in np.linspace(2,maxNComponents,nDifferentComponents)])]            
        elif dimReducer=='SparseRandomProjection' or dimReducer=='TryAllMethods':            
            dr_params+=[SparseRandomProjection(n_components=int(k)) for k in np.unique([int(k) for k in np.linspace(2,maxNComponents,nDifferentComponents)])]                        
        elif dimReducer=='RandomTreesEmbedding' or dimReducer=='TryAllMethods':            
            dr_params+=[RandomTreesEmbedding(max_depth=m,n_estimators=k) for k in [10,50] for m in [5,10]]                                    
        elif dimReducer=='BernoulliRBM' or dimReducer=='TryAllMethods':         
            dr_params+=[BernoulliRBM(n_components=int(k),learning_rate=10**r) for k in np.unique([int(k) for k in np.linspace(2,maxNComponents,nDifferentComponents)]) for r in range(-3,1)]
        if bAddExtraFeatures:
            resulting_param_grid['dr__dr'] = dr_params
        else:
            resulting_param_grid['dr'] = dr_params
        
    ####################################################################################################################################        
    #Tf-Idf
    ####################################################################################################################################        
    if bUseTfidf:
        pipe_steps.insert(0, ('tfidf',TfidfTransformer(smooth_idf=True)))
        resulting_param_grid.update(dict(
                            tfidf__norm=['l1','l2',None],
                            tfidf__use_idf =[True,False], 
                            tfidf__sublinear_tf =[True,False]))
    ####################################################################################################################################        
    #Scalers
    ####################################################################################################################################        
    if bUseScalers:                                
        common_scalers=[RobustScaler(),StandardScaler(with_mean=False),StandardScaler(with_mean=True),
        PowerTransformer(method='yeo-johnson', standardize=True),PowerTransformer(method='yeo-johnson', standardize=False),
                QuantileTransformer(output_distribution='uniform'),QuantileTransformer(output_distribution='normal'),
                          MinMaxScaler(feature_range=(1, 2))]
        if bOptionallyNoScaler: common_scalers=[None]+common_scalers
        pipe_steps.insert(0, ('sc', StandardScaler()))
        resulting_param_grid['sc'] = common_scalers
    ####################################################################################################################################        
    #Polynomial interactions
    ####################################################################################################################################                    
    if polynomialFeatures:
        assert polynomialFeatures.isnumeric() 
        assert int(polynomialFeatures)>0
        pipe_steps.insert(0, ('pf', PolynomialFeatures()))
        resulting_param_grid['pf']=[PolynomialFeatures(include_bias=False,degree=k,interaction_only=i) for k in range(2,int(polynomialFeatures)+1) for i in [True]]
    ####################################################################################################################################        
    #Nonlinear features
    ####################################################################################################################################            
    if nonLinearFeatures in ['RBFSampler','Nystroem','AdditiveChi2Sampler','SkewedChi2Sampler','TryAllMethods']:
        if bAddExtraFeatures:
            pipe_steps.insert(0,('nlf',FeatureUnion([('basic',SelectKBest(k='all')),('nfl',RBFSampler())]))) #Any will be Ok here! It's just a placeholder for now.
        else:
            pipe_steps.insert(0,('nlf',RBFSampler())) #Any will be Ok here! It's just a placeholder for now.
        if bOptionallyNoNonLinearFeatures: 
            nlf_params=[None]
        else:
            nlf_params=[]        
        if nonLinearFeatures=='RBFSampler' or nonLinearFeatures=='TryAllMethods':            
            nlf_params+=[RBFSampler(gamma=10**p,n_components=int(k)) for k in np.unique([int(k) for k in np.linspace(2,maxNComponents,nDifferentComponents)]) for p in [-3,-2,-1,0,1,2]]
        elif nonLinearFeatures=='Nystroem' or nonLinearFeatures=='TryAllMethods':
            nlf_params+=[Nystroem(kernel=r,degree=d,gamma=10**p,n_components=int(k)) for k in np.unique([int(k) for k in np.linspace(2,maxNComponents,nDifferentComponents)]) for p in [-3,-2,-1,0,1,2] for d in [2,3] for r in ['rbf','sigmoid','polynomial']]
        elif nonLinearFeatures=='AdditiveChi2Sampler' or nonLinearFeatures=='TryAllMethods':
            nlf_params+=[AdditiveChi2Sampler()]
        elif nonLinearFeatures=='SkewedChi2Sampler' or nonLinearFeatures=='TryAllMethods':
            nlf_params+=[SkewedChi2Sampler()]
        if bAddExtraFeatures:
            resulting_param_grid['nlf__nfl'] = nlf_params
        else:
            resulting_param_grid['nlf'] = nlf_params   
            
    if bKBinsDiscretization:        
        if bAddExtraFeatures:
            pipe_steps.insert(0,('dscr',FeatureUnion([('basic',SelectKBest(k='all')),('dscr',KBinsDiscretizer())]))) #Any will be Ok here! It's just a placeholder for now.
        else:
            pipe_steps.insert(0,('dscr',KBinsDiscretizer())) #Any will be Ok here! It's just a placeholder for now.        
        resulting_param_grid['dscr'] = [KBinsDiscretizer(n_bins=b,encode=e,strategy=s) for b in [3,5,7,10,15,20] for e in ['onehot','ordinal'] for s in ['uniform','quantile']]    #,'kmeans'
    if bBinarization:
        pipe_steps.insert(0, ('vt2', VarianceThreshold()))
        if bAddExtraFeatures:
            pipe_steps.insert(0,('bnr',FeatureUnion([('basic',SelectKBest(k='all')),('bnr',Binarizer())]))) #Any will be Ok here! It's just a placeholder for now.
        else:
            pipe_steps.insert(0,('bnr',Binarizer())) #Any will be Ok here! It's just a placeholder for now.        
        resulting_param_grid['bnr'] = [Binarizer(threshold=t) for t in np.linspace(x_train.min(),x_train.max(),nDifferentComponents)]
    if bNormalization:
        pipe_steps.insert(0, ('norm', Normalizer()))   
        resulting_param_grid['norm'] = [Normalizer(norm=n) for n in ['l1','l2','max']]
    ####################################################################################################################################                
    #VarianceThreshold
    ####################################################################################################################################                
    pipe_steps.insert(0, ('vt', VarianceThreshold()))
    ####################################################################################################################################                
    if target_encoding:
        pipe_steps.insert(0, ('te', TargetEncodingTransformer()))
    
    the_pipe = Pipeline(pipe_steps, memory=pipe_mem)
    
    if isRegression:
        if TargetTransformer:
            the_pipe=TransformedTargetRegressor(regressor=the_pipe,transformer=TargetTransformer,check_inverse=False)
            #prepend all the keys in paramgrid dict with regressor_
            param_grid={'regressor__'+key:value for key,value in resulting_param_grid.items()}
        else:
            param_grid=resulting_param_grid
    else:
        param_grid=resulting_param_grid
        
    ####################################################################################################################################        
    #Bayesian or randomized search?
    ####################################################################################################################################                
    if bUseBayesianSearch:
        grid_search =BayesSearchCV(the_pipe, iid=False, return_train_score=False,
                                     refit=scoring,error_score=error_score,
             search_spaces=param_grid, cv=inner_cv, scoring=scoring,
                                     verbose=4, n_jobs=nJobs, n_iter=num_iters,n_points=64,pre_dispatch='2*n_jobs')        
    else:
        fit_params={"early_stopping_rounds":42, 
                            "eval_metric" : "mae", 
                            "eval_set" : [[x_test,y_test]]}        
        grid_search = RandomizedSearchCV(the_pipe, iid=False, return_train_score=False,
                                     refit=scoring,error_score=error_score,
             param_distributions=param_grid, cv=inner_cv, scoring=scoring,
                                     verbose=4, n_jobs=nJobs, n_iter=num_iters) #,fit_params=fit_params
    startTime = datetime.now()

    if fit==False:  
        ####################################################################################################################################        
        #Just want to get a comprehensive cross-validation score...
        ####################################################################################################################################                        
        logger.debug("%s Starting Deep Cross-Validation of %s" % (startTime,type(base_model).__name__))
        cvl = cross_validate(grid_search,x_train,y_train,groups=groups,cv=outer_cv,scoring=scoring,return_train_score=False) #n_jobs=nJobs НЕ СТАВИМ!!
        cv_scores=cvl['test_score']
        logger.debug("Deep Cross-Validation ROC-AUC of gridsearch: %0.3f \u00B1 %0.3f (%s))" % (cv_scores.mean(),cv_scores.std(),cv_scores))
    else:
        ####################################################################################################################################        
        #Need to find best model
        ####################################################################################################################################                                
        logger.info("%s Starting Grid Search of %s" % (startTime,type(base_model).__name__))            
        
        grid_search.fit(x_train, y_train, groups=groups)
        
        cv_perf=grid_search.best_score_
        if scoring._sign==-1:cv_perf=-cv_perf        
        
        logger.debug("Best avg. CV score: %s" % cv_perf)
        logger.debug("Best_estimator:" + str(grid_search.best_estimator_))
        
        ####################################################################################################################################        
        #And save it
        ####################################################################################################################################                                        
        
        joblib.dump(grid_search.best_estimator_,filepath, compress = 1)
        logger.debug("Model saved as " + filename)
                           
        ####################################################################################################################################        
        #Issue prediction & display its performance
        ####################################################################################################################################                                                
        
        if savePrediction:
            if test_size>0:
                dataChunks=[(x_test,y_test,'test')]
            else:
                dataChunks=[]
            if outputDir:
                #also it's desirable (?) to save prediction on the train set. can be used later for meta learners (compute mutual similarity of models)
                if test_size>0:
                    dataChunks+=[(x_train,y_train,'train')]
                else:
                    dataChunks=[(x_train,y_train,'test')]
            for data,y_data,dataName in dataChunks:
                title="%s %s [CV perf=%.3f]" % (type(base_model).__name__,dataName,cv_perf)
                #And maybe even save the pic
                if savePerformancePlot:                    
                    perfPlotFile=filepath.replace('.pkl','_'+dataName+'_perfplot.png')
                else:
                    perfPlotFile=None                
                if isRegression:
                    prediction=grid_search.predict(data)              
                    plot_regressor_performance(grid_search,data,y_data,title,scorer=scoring,theSeed=randomSeed,saveAs=perfPlotFile)
                else:
                    if hasattr(base_model, "predict_proba"):
                        predictions = grid_search.predict_proba(data)                
                    elif hasattr(base_model, "decision_function"):
                        predictions = grid_search.decision_function(data)
                    else:
                        logger.warn("Unexpected: estimator " + type(base_model).__name__ + " has nor predict_proba not decision_function")
                    plot_roc(grid_search,data,y_data,title,show_calibration=False,saveAs=perfPlotFile)
                    prediction=grid_search.predict_proba(data)
                    prediction=prediction[:,1]
                if dataName=='test':
                    with open(filepath.replace('.pkl','.perf'), "w") as f:
                        f.write(','.join(list(map(str,[cv_perf,rmsle_score._score_func(prediction,y_data),datetime.now()-startTime]))))
                        f.write('\n')
                        if TargetTransformer:
                            f.write(str(grid_search.best_estimator_.regressor.steps))
                        else:
                            f.write(str(grid_search.best_estimator_.steps))
                np.savetxt(filepath.replace('.pkl','.'+dataName+'pred'), prediction, delimiter=",")                    
    try:    
        winsound.PlaySound('sound.wav', winsound.SND_FILENAME)
    except:
        pass
    ####################################################################################################################################        
    #Cleanup
    ####################################################################################################################################                                                                    
    if bUseCaching:
        rmtree(pipe_cachedir)              
        
    if fit==False:
        return cvl
    else:
        return grid_search
        
#################################################################################################################################### 
#Actual ensembling
#################################################################################################################################### 
def AssembleTrainingDataForLevel(currentLevelFolder,preprocessorScriptFile,bUseBayesianOptimizer,nextStackLevelDataPercent=0):
    global x,y,x_train,x_test,y_train,y_test,test_size,modelsList,perfDict
    assert nextStackLevelDataPercent<1    
    #******************************************************************************************************************
    #Натренировать список моделей, используя данные заданного уровня и нужный препроцессор.
    #******************************************************************************************************************        
    
    #При заказе прогноза с последнего уровня будет выбираться самая сильная на CV модель. Также для моделей 
    #последнего уровня в список моделей будут автоматически добавляться усреднялки 
    #средним арифметическим/геометрическим/медианой
    
    #Если файл модели с нужным именем уже будет существовать (например, из-за отключения света), модель будет пропущена и
    #в лог записано предупреждение.        
    ###################################################################################################################
    #Get current level depending on folder name
    ###################################################################################################################
    folders=currentLevelFolder.split('\\')
    bottomLevelFolder=folders[-1]
    assert 'level_' in bottomLevelFolder
    curLevel=bottomLevelFolder.replace('level_','')
    assert curLevel.isnumeric()
    curLevel=int(curLevel)
    ###################################################################################################################
    #Create folder for the next stacking level if needed
    #Если nextStackLevelDataPercent>0, будет автоматически создана папка следующего уровня.
    #Если nextStackLevelDataPercent=0, модели re-fit ятся на всех доступных на этом уровне данных, и он считается финальным.
    ###################################################################################################################            
    if nextStackLevelDataPercent>0:
        #Create folder if not already there
        newFolderName=currentLevelFolder+'\\level_'+str(curLevel+1)
        if not os.path.exists(newFolderName):
            os.makedirs(newFolderName)
        modelsFolderName=currentLevelFolder+'\\models'
    else:
        modelsFolderName=currentLevelFolder+'\\finalmodels'
    if not os.path.exists(modelsFolderName):
        os.makedirs(modelsFolderName)            
    modelsFolderName+='\\'+preprocessorScriptFile.replace('.preprocess','')
    if not os.path.exists(modelsFolderName):
        os.makedirs(modelsFolderName)                
    ###################################################################################################################
    #Get raw data from previous level
    ###################################################################################################################
    x=[];y=[];modelsList=[];perfDict=dict();
    if curLevel==0:
        logger.debug("Getting raw data")
        assert os.path.isfile(currentLevelFolder+'\\getData.py')
        execfile(currentLevelFolder+'\\getData.py')
    else:
        logger.debug("Getting input data from previous level")
        parentDir=os.path.dirname(currentLevelFolder)
        print('parentDir=',parentDir)        
        for file in os.listdir(parentDir):
            if file.endswith('.dat'):
                if file.startswith('testY'):
                    print("Loading Y file "+file)
                    y=np.loadtxt(parentDir+'\\'+file)             
        for subdir in os.walk(parentDir+'\\models'):
            if subdir[0]!=parentDir+'\\models':
                dirName=os.path.basename(subdir[0])
                print("Looking at models in "+subdir[0])
                for file in os.listdir(subdir[0]):
                    if file.endswith('.perf'):
                        #read first line
                        with open(subdir[0]+'\\'+file) as f:
                            perf=f.readline().split(',')
                        perfDict[dirName+'_'+file]=perf
                    elif file.endswith('.testpred'):
                        modelsList.append(file.replace('.testpred',''))
                        nextX=np.loadtxt(subdir[0]+'\\'+file) 
                        print("Loaded model "+file)
                        if len(x)>0:
                            print(len(x),len(nextX))
                            assert len(x)==len(nextX)
                            x=np.concatenate((x,nextX.reshape(-1,1)),axis=1)
                        else:
                            x=nextX.reshape(-1,1).copy()
                
    #assert 'x' in locals()
    #assert 'y' in locals()   
    assert len(x)>0
    assert len(x)==len(y)
    ###################################################################################################################
    #Create train/test split if we have no one in the folder yet
    ###################################################################################################################                       
    if nextStackLevelDataPercent>0:
        trainIndices=[];testIndices=[]
        for file in os.listdir(currentLevelFolder):
            if file.endswith(".idx"):
                if file.startswith("train_"):
                    trainIndices=np.loadtxt(currentLevelFolder+'\\'+file).astype(int)
                elif file.startswith("test_"):
                    testIndices=np.loadtxt(currentLevelFolder+'\\'+file).astype(int)
        if len(trainIndices)==0 and len(testIndices)==0:
            
            logger.debug("Splitting data of this level into train and test(shadow) sets")
            
            testIndices=sorted(np.random.choice(x.shape[0], int(x.shape[0]*nextStackLevelDataPercent), replace=False))
            trainIndices=np.array(list(set(range(x.shape[0]))-set(testIndices)))        
            
            if len(testIndices)>0:
                np.savetxt(currentLevelFolder+'\\'+'test_'+str(nextStackLevelDataPercent)+'.idx',testIndices,fmt='%i') 
            if len(trainIndices)>0:
                np.savetxt(currentLevelFolder+'\\'+'train_'+str(nextStackLevelDataPercent)+'.idx',trainIndices,fmt='%i')         
            if len(y)>0:            
                np.savetxt(currentLevelFolder+'\\'+'testY_'+str(nextStackLevelDataPercent)+'.dat',y[testIndices]) 
            logger.debug("Train/Test indices [%d/%d] were created and saved" % (len(trainIndices),len(testIndices)))
        else:
            #all indices were loaded and now need to make sure they fit the data
            assert len(trainIndices)+len(testIndices)==len(x)
            logger.debug("Train/Test indices [%d/%d] were loaded and tested" % (len(trainIndices),len(testIndices)))
        test_size=len(testIndices)
        x_train,x_test=x[trainIndices],x[testIndices]
        y_train,y_test=y[trainIndices],y[testIndices]
    else:
        test_size=0;x_test=[];y_test=[];
        x_train=x;y_train=y        
    ###################################################################################################################
    #Apply preprocessor script
    ###################################################################################################################                            
    assert preprocessorScriptFile.endswith('.preprocess')
    with open(currentLevelFolder+'\\'+preprocessorScriptFile) as f:
        preprocessorScript = f.read()
    logger.debug("Applying preprocessing script \n"+preprocessorScript)
    
    #exec(preprocessorScript) #this must create additional columns etc        
    execfile(currentLevelFolder+'\\'+preprocessorScriptFile)        
    return modelsFolderName
def PredictUsingStakedEnsemble(X,lastLevelFolder):
    #Просматривает все уровни вложености директорий от конечной lastLevelFolder до уровня 0,
    #рекурсивно определяет список всех нужных моделей и препроцессоров на каждом уровне, затем
    #последовательно вызывает их в нужном порядке и применяет к данным.
    #Осуществляет проверки по именам и датам всех файлов! Не позволяет сделать прогноз, если хоть что-то не совпадает
    pass
def ShowLevelModelsPerformance(currentLevelFolder):
    pass