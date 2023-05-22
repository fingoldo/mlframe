########################################################################################################################################################################################################################################
# Extract & engineer features
########################################################################################################################################################################################################################################


# Features selection
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import f_regression, mutual_info_regression, mutual_info_classif


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import shuffle
from datetime import datetime
from copy import deepcopy
import numpy as np
from numpy import *

# Feature engineering
def mode(x):
    return np.percentile(a=x, q=50)


def StringOrFuncName(the_list, feature, MA, pure_funcs):
    column_names_real = []
    pure_funcs_real = []
    for fname in the_list:
        if isinstance(fname, str):
            str_name = fname
        else:
            str_name = fname.__name__

        real_name = feature + '_' + str_name + '_' + str(MA)

        column_names_real.append(real_name)
        if str_name in pure_funcs:
            pure_funcs_real.append(real_name)
    return column_names_real, pure_funcs_real


def EnrichTSDatasetWithRollingStats(ds, MAs=[5, 10], lags=[], exclude_features=['hour', 'weekday'],
                                        targets=['up_val', 'down_val'], normalize='ratio',
                                        funcs=['mean', 'std', 'min', 'max', 'median', 'skew', 'kurt', mode],
                                        pure_funcs=['skew', 'kurt'], drop_original_featurs=False):
    import numpy as np
    import pandas as pd
    from copy import deepcopy
    from datetime import datetime
    # set of MAs [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
    # funcs=['mean','std','min','max','median','skew','kurt',mode]
    startTime = datetime.now()
    original_features = deepcopy(ds.columns)
    for feature in original_features:
        if feature not in np.union1d(exclude_features, targets):
            ##########################################################################################################################################################
            # Log, sqrt
            ##########################################################################################################################################################
            # ds[feature+'_log']=np.log(1+ds[feature])
            # ds[feature+'_sqrt']=np.sqrt(ds[feature])

            # lagged values of input & targets
            # ds[feature+'_prev']=ds[feature].shift(1)

            # 1st derivatives
            # ds[feature+'_perc_change']=ds[feature]/ds[feature].shift(1)

            # MAs
            for MA in MAs:
                if normalize == 'old':
                    ##########################################################################################################################################################
                    # Works twice slower than agg, is kept just fore reference purposes
                    ##########################################################################################################################################################
                    ds[feature + '_MA_' + str(MA)] = ds[feature].rolling(window=MA).mean()
                    ds[feature + '_STD_' + str(MA)] = ds[feature].rolling(window=MA).std()
                    ds[feature + '_MMIN_' + str(MA)] = ds[feature].rolling(window=MA).min()
                    ds[feature + '_MMAX_' + str(MA)] = ds[feature].rolling(window=MA).max()
                    ds[feature + '_MEDIAN_' + str(MA)] = ds[feature].rolling(window=MA).median()
                    ds[feature + '_MODE_' + str(MA)] = ds[feature].rolling(window=MA).apply(mode)
                    # ds[feature+'_HURST_'+str(MA)] = ds[feature].rolling(window=MA).apply(hurst)
                    # for MA in mas_list:
                    # ds[feature+'_linear_k_'+str(MA)] = ds[feature].rolling(window=MA).apply(lambda x: np.polyfit(arange(len(x)),x,1)[0])
                    # for MA in myLongMAs:
                    # ds[feature+'_LS_periodogram_'+str(MA)] = ds[feature].rolling(window=MA).apply(lambda x: SignificantLombScarglePeriod(x))
                else:
                    agg = ds[feature].rolling(window=MA).agg(funcs)
                    column_names_real, pure_funcs_real = StringOrFuncName(funcs, feature, MA, pure_funcs)
                    agg.columns = column_names_real
                    if normalize == 'ratio':
                        if len(pure_funcs_real) > 0:
                            pure_agg = agg[pure_funcs_real]
                            agg.drop(pure_funcs, inplace=True)
                            ds = pd.concat([ds, pure_agg], axis=1)

                        agg = agg.div(ds[feature], axis='index')

                    ds = pd.concat([ds, agg], axis=1)

            for lag in lags:
                if normalize == 'ratio':
                    ds[feature + '_lag_' + str(lag)] = ds[feature].shift(lag) / ds[feature]
                else:
                    ds[feature + '_lag_' + str(lag)] = ds[feature].shift(lag)
    ds.replace([np.inf, -np.inf], 0, inplace=True)
    ds.dropna(axis=0, how='any',inplace=True)
    print('Time elpased in EnrichTSDatasetWithRollingStats: {}'.format(datetime.now() - startTime))
    return ds


def SignificantLombScarglePeriod(y):
    x = arange(len(y)).astype(float)
    f = np.linspace(1, len(y), 100)
    # Calculate Lomb-Scargle periodogram:
    import scipy.signal as signal
    pgram = signal.lombscargle(x, y, f)

    if np.max(pgram) > np.mean(pgram) * 5:
        return f[np.argmax(pgram)]
    else:
        return 0


# Features importances

def CompareMIandShuffledMI(x,y,y_shuffled):
    plt.plot(mutual_info_classif(x, y));
    plt.plot(mutual_info_classif(x, y_shuffled));
    plt.show()


# Create all possible combinations of input features
def GetAllInputsCombinations(x):
    from itertools import combinations
    all_comb = []
    for size in range(2, 4):
        all_comb += list(combinations(range(x.shape[1]), r=size))
    # print(all_comb)
    # print(x.shape[1])
    return all_comb


def hurst(ts):
    lags = range(2, 5)
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    # calculate Hurst as slope of log-log plot
    m = polyfit(log(lags), log(tau), 1)
    return m[0] * 2.0


########################################################################################################################################################################################################################################
# My custom transformer as a preprocessing step
########################################################################################################################################################################################################################################
class FeatureMultiplier(BaseEstimator, TransformerMixin):
    def __init__(self, factor):
        self.factor = factor

    def transform(self, X, *_):
        return X * self.factor

    def fit(self, *_):
        return self


#Custom MI-based features-selector (able to catch pairwise interaction and remove cross-collinear inputs)
class MyCustomColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, sActiveInputs, num_shuffles=3, fCollinearityThreshold=0, bCheckPairwiseImpact=False,
                 bVerbose=False):
        self.num_shuffles = num_shuffles
        self.sActiveInputs = sActiveInputs
        self.fCollinearityThreshold = fCollinearityThreshold
        self.bCheckPairwiseImpact = bCheckPairwiseImpact
        self.bVerbose = bVerbose

    def fit_transform(self, X, y=None):
        return self.transform(self, X, y)

    def transform(self, x, y=None):
        bVerbose = self.bVerbose
        num_shuffles = self.num_shuffles
        sActiveInputs = self.sActiveInputs
        bCheckPairwiseImpact = self.bCheckPairwiseImpact
        fCollinearityThreshold = self.fCollinearityThreshold

        min_impact = 0.05
        ########################################################################################################################################################################################################################################
        # 1. Basic computations of single-featured dependency (assesed using dependency of shadow (shuffled) variable)
        ########################################################################################################################################################################################################################################
        mi_shuffled = []
        mi_base = mutual_info_classif(x, y)
        for i in arange(num_shuffles):
            y_shuffled = shuffle(y)
            mi_shuffled.append(mutual_info_classif(x, y_shuffled))

        cond = np.logical_and(mi_base > np.mean(np.array(mi_shuffled), axis=0, keepdims=True)[0] * 10,
                              mi_base > min_impact)

        impacting = [];
        non_impacting = []
        impacting_names = [];
        non_impacting_names = []
        for i in arange(len(sActiveInputs)):
            if cond[i]:
                impacting.append(i)
                impacting_names.append(sActiveInputs[i])
            else:
                non_impacting.append(i)
                non_impacting_names.append(sActiveInputs[i])
        if bVerbose:
            # print("mi_base="+str(mi_base))
            # print("np.mean(mi_shuffled)="+str(np.mean(mi_shuffled,axis=0,keepdims=True)[0]))
            print("")
            print("Impacting variables ids: " + str(len(impacting)) + " " + str(impacting))
            print("Impacting variables: " + str(len(impacting_names)) + " " + str(impacting_names))
            print("")
            print("Irrelevant variables ids: " + str(len(non_impacting)) + " " + str(non_impacting))
            print("Irrelevant variables: " + str(len(non_impacting_names)) + " " + str(non_impacting_names))

        ########################################################################################################################################################################################################################################
        # 2. Possible joint impact of paired variables
        ########################################################################################################################################################################################################################################
        if bCheckPairwiseImpact:
            # From all features deemed as not impacting target on their own (so not included in Step 1), as a last resort, generate unique pairwise combinations.
            from itertools import combinations
            nonimpacting_combs = list(combinations(range(len(non_impacting)), r=2))

            new_vars = np.empty([x.shape[0], len(nonimpacting_combs)])
            i = 0
            for pairs in nonimpacting_combs:
                np.multiply(x[:, non_impacting[pairs[0]]], x[:, non_impacting[pairs[1]]], new_vars[:, i])
                i += 1
            if bVerbose:
                print("")
                print("Joined virtual variables created: " + str(new_vars.shape[1]))

            mi_base_paired = mutual_info_classif(new_vars, y)
            mi_shuffled_paired = []
            for i in arange(num_shuffles):
                mi_shuffled_paired.append(mutual_info_classif(new_vars, y_shuffled))
                # Let's reuse one instance of y_shuffled created  at Step 1
                if i < num_shuffles - 1:
                    y_shuffled = shuffle(y)

            impacting_paired = np.argwhere(
                np.logical_and(mi_base_paired > np.mean(np.array(mi_shuffled_paired), axis=0, keepdims=True)[0] * 10,
                               mi_base_paired > min_impact)).reshape(-1)
            should_be_added = np.zeros(len(non_impacting), dtype=int)
            for i in impacting_paired:
                pairs = nonimpacting_combs[i]
                # Both of participating input features are awarded
                should_be_added[pairs[0]] += 1
                should_be_added[pairs[1]] += 1

            restored = [];
            restored_names = []
            really_non_impacting = [];
            really_non_impacting_names = []
            i = 0
            for r in should_be_added:
                if r > 0:
                    restored.append(non_impacting[i])
                    restored_names.append(non_impacting_names[i])
                else:
                    really_non_impacting.append(non_impacting[i])
                    really_non_impacting_names.append(non_impacting_names[i])
                i += 1
            if bVerbose:
                print("Restored as having pairwise impact:" + str(restored_names))
            # print("Restored="+str(restored))
            # print("impacting="+str(impacting))
            impacting = np.hstack((impacting, restored)).astype(int)
            # print("impacting+Restored="+str(impacting))
            impacting_names = np.hstack((impacting_names, restored_names))

            non_impacting = really_non_impacting
            non_impacting_names = really_non_impacting_names
        ########################################################################################################################################################################################################################################
        # 3. Possible collinearity: let's remove most redundant variables?
        ########################################################################################################################################################################################################################################
        if fCollinearityThreshold > 0:
            # What's the ratio, for each input feature, of its impact on target divided by its total correlation with other input features?
            # Probably inputs having lowest values of such parameter do not add a lot to the model and should be excluded.
            if len(impacting) > 0:
                mi_base_reg = np.floor(mi_base / min_impact)

                impacting_combs = list(combinations(range(len(impacting)), r=2))
                pairwise_mi = np.zeros(len(impacting_combs))
                # Let's compute pairwise MIs of all possible input features combinations

                # from joblib import Parallel, delayed
                # if __name__ == "__main__":
                #   pairwise_mi=Parallel(n_jobs=30)(delayed(mutual_info_regression)(x[:,impacting[pairs[0]]].reshape(-1, 1),x[:,impacting[pairs[1]]]) for  pairs in impacting_combs)
                i = 0
                # print("impacting_combs="+str(impacting_combs))
                for pairs in impacting_combs:
                    pairwise_mi[i] = mutual_info_regression(x[:, impacting[pairs[0]]].reshape(-1, 1),
                                                            x[:, impacting[pairs[1]]])
                    i += 1

                # print("pairwise_mi="+str(pairwise_mi))
                # And check what features are most redundant to all others
                i = 0
                computed_mi = np.ones(len(impacting)) * 0.001
                for pairs in impacting_combs:
                    res = pairwise_mi[i]
                    # Both of participating input features are "awarded" by this values
                    computed_mi[pairs[0]] += res
                    computed_mi[pairs[1]] += res
                    i += 1
                computed_value = mi_base_reg[impacting] / computed_mi
                # print("mi_base_reg="+str(mi_base_reg))
                # print("computed_mi="+str(computed_mi))
                # print("computed_value="+str(computed_value))
                threshold = np.percentile(computed_value, fCollinearityThreshold)
                removed_indices = np.argwhere(computed_value < threshold).reshape(-1)
                if len(removed_indices) > 0:
                    if bVerbose:
                        removed_cols_names = np.array(impacting_names)[removed_indices]
                        print("Columns removed as highly cross-collinear (redundant): " + str(
                            removed_cols_names.size) + " " + str(removed_cols_names))
                        inds = argsort(computed_value)[::-1]
                        plt.bar(np.array(impacting), computed_value[inds])
                        plt.xticks(np.array(impacting), np.array(impacting_names)[inds], rotation=90)
                        plt.rcParams["figure.figsize"] = (10, 6)
                        plt.show()
                        # print("threshold="+str(threshold))
                    impacting = np.delete(impacting, removed_indices)
                    impacting_names = np.delete(impacting_names, removed_indices)
        return x[:, impacting]

    def fit(self, *_):
        return self

########################################################################################################################################################################################################################################
### Test it on a XOR dataset as a proof of concept
########################################################################################################################################################################################################################################
def TestMyCustomColumnSelector():
    a = np.random.randn(5000, 4)
    b = np.where(np.logical_xor(a[:, 0] > 0, a[:, 1] > 0), 1, 0)

    fm = MyCustomColumnSelector(['x1', 'x2', 'fake1', 'fake2'], 3, 10, True, True)
    impacting = fm.transform(a, b)

    return impacting

"""

Impacting variables ids: 0 []
Impacting variables: 0 []

Irrelevant variables ids: 4 [0, 1, 2, 3]
Irrelevant variables: 4 ['x1', 'x2', 'fake1', 'fake2']

Joined virtual variables created: 6
Restored as having pairwise impact:['x1', 'x2']

"""