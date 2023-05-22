class PerformanceEstimator():
    def __init__(self,greater_is_better=False):
        self.greater_is_better=greater_is_better
    def estimate_performance(self,x,y,fold_indices,predictions):
        pass
    def greater_is_better(self):
        return self.greater_is_better

class GroupBalancedLogMeanAbsoluteError(PerformanceEstimator):
    def __init__(self,groups):
        self.greater_is_better=False
        self.groups=groups
    def estimate_performance(self,x,y,fold_indices,predictions):
        from sklearn.metrics import mean_absolute_error
        errs=[]
        groups=self.groups[fold_indices]
        for group in np.unique(groups):
            idx=(groups==group)
            errs.append(np.log(mean_absolute_error(y[idx],predictions[idx])))
        return np.mean(errs)