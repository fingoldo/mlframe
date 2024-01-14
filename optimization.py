"""Class for numerical optimization. In ML, serves to optimize hyperparameters, select best features."""

import numpy as np
import matplotlib.pyplot as plt

from random import random
from enum import Enum, auto

class OptimizationDirection(Enum):
    Minimize = auto()
    Maximize = auto()

class CandidateSamplingMethod(Enum):
    Random = auto()
    Equidistant = auto()
    Fibonacci = auto()
    ReversedFibonacci = auto()
    
class OptimizationProgressPlotting(Enum):
    No = auto()
    Final = auto() # Plotting is done once, after the search finishes
    OnScoreImprovement = auto() # Plotting is done on every new candidate
    
class MBHOptimizer():
    def __init__():
        pass
    def suggest_candidate(worker_key:str=None):
        pass
    def suggest_candidates(n:int=1,worker_key:str=None):
        pass
    def add_trial(parameters:Any,result:Any,duration_seconds:float=None,worker_key:str=None):
        # if duration_seconds is None, it's computed automatically using timestamp of suggesting that particular candidate to that particular worker
        pass    
    def add_trials(trials:Sequence,worker_key:str=None):
        pass

def generate_fibonacci(n:int):
    """Creates Fibonacci sequence for a given n."""

    if n <= 0:
        return []

    fibonacci_sequence = [0, 1]
    while len(fibonacci_sequence) < n:
        next_number = fibonacci_sequence[-1] + fibonacci_sequence[-2]
        fibonacci_sequence.append(next_number)

    return np.array(fibonacci_sequence, dtype=np.int64)
 
def eval_candidate_inner(eval_func, cand, eval_times: list) -> float:
    """Measures execution time (possibly, in future also RAM, CPU usage) of the child function."""

    eval_start_time = timer()
    res = eval_func(cand)
    eval_times.append(timer() - eval_start_time)

    return res

def check_noimproving_iters(max_noimproving_iters:int,n_noimproving_iters:int,verbose:bool)->bool:        
    if max_noimproving_iters and n_noimproving_iters>=max_noimproving_iters:
        if verbose: logger.info(f"Max # of noimproved iters reached: {n_noimproving_iters}")        
        return True

def optimize_finite_onedimensional_search_space(
    search_space: Sequence,  # search space, all possible input combinations to check
    eval_candidate_func: object,  # fitness function to be optimized over the search space
    scores:np.ndarray=None, # known true fitness of the entire search space
    direction:OptimizationDirection=OptimizationDirection.Maximize,
    # stopping conditions
    max_runtime_mins: float = None,
    predict_runtimes:bool=True, # intellectual setting that skips candidates whose evaluation won't finish within current runtime limit
    max_fevals: int = None,
    best_desired_score:float=None,
    max_noimproving_iters:int=None,
    # inits
    seed_inputs: Sequence = [],  # seed items you want to be explored from start
    init_num_samples: Union[float, int] = 20,  # how many samples to generate & evaluate before fitting surrogate model
    init_evaluate_ascending: bool = False,
    init_sampling_method: CandidateSamplingMethod = CandidateSamplingMethod.Random,  # random, equidistant, fibo, rev_fibo?
    # EE dilemma
    exploitation_probability: float = 0.6,
    probabilistic: bool = True, # pick the absolute best predicted candidate, or probabilistically the best
    use_distances_on_preds_collision:bool=True,
    use_stds_for_exploitation:bool=True,
    # model
    acquisition_method: str = "",
    model: object = CatBoostRegressor(),  # actual estimator instance here? + for lgbm also the linear mode flag
    quantile: float = 0.01,
    model_params:dict={'iterations':100},
    # OptimizationProgressPlotting
    ndim:int=1, # number of dimensions to use for visualisation
    OptimizationProgressPlotting:OptimizationProgressPlotting=OptimizationProgressPlotting.No,    
    figsize: tuple = (8, 4),
    font_size:int=10,    
    x_label="nfeatures",
    y_label="score",
    expected_fitness_color:str="green",
    # small settings
    verbose: int = 0,    
    
) -> None:

    """Finds extremum of some (possibly multi-input) function F that is supposedly costly to estimate (like in HPT, FS tasks).
    Function F can also be a numerical sequence in form of some y scores array.

    To achieve high search efficiency, uses a surrogate model to mimic our true 'black box' function behaviour.
    Once F's values in a few points are computed, surrogate is fit and used to predict F in its entire definition range. Then, the most promising points (
    having the highest predicted values) can be checked directly.

    As surrogate models, can use quantile regressions (namely, one of modern gradient boosting libs that implement it), as they can have a notion of uncertainty.
    Do we REALLY need only models capable of uncertainty estimates?
    To internally estimate wellness of surrogate model, compares it to the best of the dummy regressors (on the test set).
    Also allows for early stopping (do we require it? is it useful at all here?).

    Uses exploration/exploitation modes (controlled by exploitation_probability parameter):
        at exploration step:
            points with highest model uncertainty and the most distant from already visited ones are suggested;
        at exploitation step:
            1) if a point with the current highest predicted value is unexplored, suggests it
            2) otherwise, picks points that can give highest result, ie have the best predicted value+uncertainty+are located far from known regions 
            (in a log scale, to lower dist factor importance)

        Next candidate can be picked in a deterministic (strict argmax) or probabilistic (by drawing a random number & comparing it with the candidate's choosing prob) manner.


    Can hold on to a pre-defined time/feval budget.

    Optionally plots the search path.
    
    Ask-tell interface:
    via iterator with yields?    
    
    Working as a class:
        1) init the object
        2) feed initial population,retrain
        3) suggest next candidate
        4) feed next input-output batch, retrain
        
    Challenges:
    1) V allow non-quantile estimators
    2) allow multiple estimators
    3) when using multiple estimators, allow an option of estimating stds directly from their forecasts for the same points
    4) allow ES (train/test splts can vary across estimators)
    5) add control via dummy model: if dummy is beaten, allow exploitation step. else only exploration
    6) batch mode, when more than 1 suggestions are produced in a single call
    7) parallel
    8) async
    
    """

    # ---------------------------------------------------------------------------------------------------------------
    # Params checks
    # ---------------------------------------------------------------------------------------------------------------

    assert quantile < 0.5
    assert acquisition_method in ("EE")

    # ---------------------------------------------------------------------------------------------------------------
    # Inits
    # ---------------------------------------------------------------------------------------------------------------

    start_time = timer()
    ran_out_of_time = False

    known_inputs = np.array([])
    known_outputs = np.array([])
    eval_times = []
    SMALL_VALUE=1e-4

    n_noimproving_iters=0
    
    # ---------------------------------------------------------------------------------------------------------------
    # Let's establish initial dataset to fit our surrogate model
    # First use pre-seeded values. they must be evaluated strictly in given order.
    # ---------------------------------------------------------------------------------------------------------------    

    if len(seed_inputs) > 0:
        if verbose:
            logger.info(f"Evaluating {len(set(seed_inputs)):_} seeded inputs...")
            seed_inputs = tqdmu(seed_inputs, desc="Evaluating seeded inputs", leave=False)
        known_inputs = np.array([],dtype=np.int32)
        for x in seed_inputs:
            if x not in known_inputs:
                known_inputs = np.append([x], known_inputs).astype(int)
                known_outputs=np.append([eval_candidate_inner(eval_func=eval_candidate_func, cand=x, eval_times=eval_times)],known_outputs)

    # ---------------------------------------------------------------------------------------------------------------
    # Now sample additional points from across the definition range with some simple algo
    # ---------------------------------------------------------------------------------------------------------------

    if init_num_samples > 0:
        if isinstance(init_num_samples, float) and init_num_samples < 1.0:
            init_num_samples = len(search_space) * init_num_samples
        if init_sampling_method == CandidateSamplingMethod.Random:
            sampled_inputs = np.random.choice(search_space, size=min(init_num_samples, len(search_space)), replace=False)
        elif init_sampling_method == CandidateSamplingMethod.Equidistant:
            sampled_indices = np.linspace(0, len(search_space) - 1, init_num_samples).astype(int)
            sampled_inputs = np.array(search_space)[sampled_indices[:init_num_samples]]
        elif init_sampling_method in (CandidateSamplingMethod.Fibonacci, CandidateSamplingMethod.ReversedFibonacci):
            # Fibo!
            fibo_sequence = generate_fibonacci(2 + init_num_samples)[2:]
            fibo_sequence_indices = (fibo_sequence * (len(search_space) - 1) / fibo_sequence.max()).astype(np.int64)
            if init_sampling_method == CandidateSamplingMethod.ReversedFibonacci:
                fibo_sequence_indices = len(search_space) - 1 - fibo_sequence_indices
            sampled_inputs = np.array(search_space)[fibo_sequence_indices[:init_num_samples]]
        else:
            raise ValueError(f"Sampling acquisition_method {init_sampling_method} not supported.")

        # let's remove intersection with already checked seeded elements, if any
        sampled_inputs = set(sampled_inputs) - set(seed_inputs)

        # sometimes it's better to process samples in certain order
        if init_evaluate_ascending:
            sampled_inputs = sorted(sampled_inputs)

        # actual evaluation of initial samples
        if len(sampled_inputs) > 0:
            if verbose:
                logger.info(f"Evaluating {len(sampled_inputs):_} sampled inputs...")
            sampled_inputs = tqdmu(sampled_inputs, desc="Evaluating sampled inputs", leave=False)

            for x in sampled_inputs:
                known_inputs = np.append([x], known_inputs).astype(int)
                known_outputs=np.append([eval_candidate_inner(eval_func=eval_candidate_func, cand=x, eval_times=eval_times)],known_outputs)
    
    # ---------------------------------------------------------------------------------------------------------------
    # Determine the point with the extreme observed function value
    # ---------------------------------------------------------------------------------------------------------------

    best_idx = np.argmax(known_outputs)
    best_x = known_inputs[best_idx]
    best_y = known_outputs[best_idx]
    
    worst_idx = np.argmin(known_outputs)
    worst_x = known_inputs[worst_idx]
    worst_y = known_outputs[worst_idx]    

    # ---------------------------------------------------------------------------------------------------------------
    # Init of the surrogate model
    # ---------------------------------------------------------------------------------------------------------------

    if model=="CBQ":
        quantiles : Sequence = [quantile, 0.5, 1 - quantile]
        loss_function = "MultiQuantile:alpha=" + ",".join(map(str, quantiles))
        #model.set_params(loss_function=loss_function,verbose=0,)
        gp_model = CatBoostRegressor(**model_params,loss_function=loss_function,verbose=0,)
    elif model=="CB":
        gp_model = CatBoostRegressor(**model_params,verbose=0,)        

    if OptimizationProgressPlotting!=OptimizationProgressPlotting.No: plt.figure(figsize=figsize)
        
    additional_info = ""
    mode=acquisition_method
    nsteps=0 
    
    # ---------------------------------------------------------------------------------------------------------------
    # Optimization Loop
    # ---------------------------------------------------------------------------------------------------------------
    
    while True:
        
        #get_best_dummy_score(estimator=estimator,X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

        # ---------------------------------------------------------------------------------------------------------------
        # Fit surrogate model to known points
        # ---------------------------------------------------------------------------------------------------------------
        
        if nsteps==0 or not hasattr(gp_model,'partial_fit'):
            gp_model.fit(known_inputs.reshape(-1, 1), known_outputs)
        else:
            gp_model.partial_fit([new_x], new_y)
        

        # ---------------------------------------------------------------------------------------------------------------
        # Make predictions for all points
        # ---------------------------------------------------------------------------------------------------------------
        
        if model == "CBQ":
            res = gp_model.predict(search_space.reshape(-1, 1))
            y_pred = res[:, 1]
            y_std = np.abs(res[:, 0] - res[:, 2]) + SMALL_VALUE
        elif model == "CB":
            res = gp_model.predict(search_space.reshape(-1, 1))
            y_pred = res
            y_std = np.zeros_like(res)            
        else:
            y_pred, y_std = gp_model.predict(search_space.reshape(-1, 1), return_std=True)

        if acquisition_method == "EE":  # exploration-exploitation

            if random() < exploitation_probability:
                mode = "exploitation"
            else:
                mode = "exploration"
            
            # ---------------------------------------------------------------------------------------------------------------
            # Known points make it easy to compute distances from candidates
            # ---------------------------------------------------------------------------------------------------------------
            
            distances = np.zeros(len(search_space)) # distances to closest checked points
            l = None
            for r in sorted(known_inputs):
                if l is None:
                    distances[:r] = np.abs(search_space[0:r] - search_space[r])
                else:
                    m = (l + r) // 2
                    distances[l:m] = np.abs(search_space[l:m] - search_space[l])
                    distances[m:r] = np.abs(search_space[m:r] - search_space[r])
                l = r
            distances[r:] = np.abs(search_space[r:] - search_space[r])
            
            # ---------------------------------------------------------------------------------------------------------------
            # Now, distances have to be normalized by known fitness range
            # ---------------------------------------------------------------------------------------------------------------
            max_dist=distances.max();min_dist=0.0
            distances=distances*(best_y-worst_y)*0.5/max_dist

            if mode == "exploration":
                # pick the point with higest std and most distant from already known points
                if model == "CB":
                    expected_fitness = y_std + distances  # as CB has almost no stds?
                else:
                    expected_fitness = y_std + distances
                additional_info = ""
            elif mode == "exploitation":
                
                expected_fitness=y_pred.copy()
                if use_stds_for_exploitation:
                    expected_fitness+=y_std
                    
                best_idx = np.argsort(expected_fitness)[-1]
                if search_space[best_idx] in known_inputs:
                    expected_fitness += distances # best supposed point already checked. let's take std and distance into account then.
                    additional_info = "plusdist"
                else:
                    # best supposed point not checked yet
                    additional_info = "bestpredicted"                    
                    
                    if use_distances_on_preds_collision:
                        # what if there are multiple non-checked points with the same predicted score?
                        best_value=expected_fitness[best_idx]
                        all_best_indices=np.where(expected_fitness>=best_value)[0]
                        all_best_indices=[idx for idx in all_best_indices if search_space[idx] not in known_inputs]
                        if len(all_best_indices)>1:
                            print(f"At step {nsteps}, there are {len(all_best_indices)} non-checked points with predicted fitness {expected_fitness[best_idx]}. Using distances additionally.")
                            expected_fitness[all_best_indices]+= distances[all_best_indices]
                            if probabilistic:
                                expected_fitness[expected_fitness<best_value]=0.0
                                
                                print("expected_fitness[all_best_indices]=",expected_fitness[all_best_indices])                    
        
        # ---------------------------------------------------------------------------------------------------------------
        # Decide on next candidate, based on predicted fitness
        # ---------------------------------------------------------------------------------------------------------------
        
        found_something=False
        
        if not probabilistic:
            # Just pick first unchecked candidate with highest fitness
            for best_idx in np.argsort(expected_fitness)[::-1]:
                if search_space[best_idx] not in known_inputs:
                    found_something=True
                    new_x = search_space[best_idx]
                    break
        else:
            # Sample from all unchecked candiates with probability=their normalized fitness.
            idx = ~np.isin(search_space,known_inputs)
            if idx.sum()>0:
                min_expected_fitness=expected_fitness[idx].min()
                if min_expected_fitness<0.0:
                    expected_fitness[idx]+=-min_expected_fitness+SMALL_VALUE
                probs = expected_fitness[idx] / expected_fitness[idx].sum()
                #probs=np.clip(probs,0.0,1.0)
                new_x = np.random.choice(search_space[idx], size=1, replace=False, p=probs)[0]
                found_something=True
        
        if not found_something:
            print("Search space fully checked, quitting")
            break
            
        # ---------------------------------------------------------------------------------------------------------------
        # Evaluate best candidate
        # ---------------------------------------------------------------------------------------------------------------
                                              
        new_y = eval_candidate_inner(eval_func=eval_candidate_func, cand=new_x, eval_times=eval_times)
        
        if direction==OptimizationDirection.Maximize:
            if new_y>best_y:
                best_y=new_y
                best_x=new_x
                n_noimproving_iters=0
            else:
                n_noimproving_iters+=1
                if check_noimproving_iters(max_noimproving_iters=max_noimproving_iters,n_noimproving_iters=n_noimproving_iters,verbose=verbose)
                    break
            if new_y<worst_y:
                worst_y=new_y
                worst_x=new_x
        elif direction==OptimizationDirection.Minimize:
            if new_y<best_y:
                best_y=new_y
                best_x=new_x
                n_noimproving_iters=0
            else:
                n_noimproving_iters+=1
                if check_noimproving_iters(max_noimproving_iters=max_noimproving_iters,n_noimproving_iters=n_noimproving_iters,verbose=verbose)
                    break
            if new_y>worst_y:
                worst_y=new_y
                worst_x=new_x                

    
        known_inputs = np.append(known_inputs, [new_x]).astype(int)        
        known_outputs = np.append(known_outputs, new_y)

        if OptimizationProgressPlotting!=OptimizationProgressPlotting.No:        

        # ----------------------------------------------------------------------------------------------------------------------------
        # Checking exit conditions
        # ----------------------------------------------------------------------------------------------------------------------------

        nsteps += 1

        if max_runtime_mins and not ran_out_of_time:
            ran_out_of_time = (timer() - start_time) > max_runtime_mins * 60
            if ran_out_of_time:
                if verbose:
                    logger.info(f"max_runtime_mins={max_runtime_mins:_.1f} reached.")
                break

        if max_fevals and nsteps >= max_fevals:
            if verbose:
                logger.info(f"max_fevals={max_fevals:_} reached.")
            break
    
    return best_x,best_y

def plot_search_state(search_space, expected_fitness:np.ndarray,y_pred:np.ndarray,y_std:np.ndarray,scores:np.ndarray,
                      acquisition_method,
    figsize: tuple = (8, 4),
    font_size:int=10,
    x_label="nfeatures",
    y_label="score",
    expected_fitness_color:str="green",
    legend_location:str='lower right'):

    # ---------------------------------------------------------------------------------------------------------------
    # Plot expected fitness of the points
    # ---------------------------------------------------------------------------------------------------------------
    
    plt.rcParams.update({'font.size': font_size})
    fig, ax1 = plt.subplots(sharex=True,figsize=figsize,layout='tight')
    ax2 = ax1.twinx()                       
    
    
    ax2.plot(search_space, expected_fitness, color=expected_fitness_color,linestyle='dashed', label=acquisition_method)
    #ax2.plot(search_space, y_std, color=expected_fitness_color,linestyle='dashed', label='y_std')
    #ax2.plot(search_space, distances, color=expected_fitness_color,linestyle='dotted', label='distances')


    # ---------------------------------------------------------------------------------------------------------------
    # Plot the black box function, surrogate function, known points
    # ---------------------------------------------------------------------------------------------------------------

    if scores is not None: ax1.plot(search_space, scores, color="black", label="Black Box Function")
    ax1.plot(search_space, y_pred, color="red", linestyle="dashed", label="Surrogate Function")
    ax1.fill_between(search_space, y_pred - y_std, y_pred + y_std, color="blue", alpha=0.2)
    ax1.scatter(known_inputs, known_outputs, color="blue", label="Known Points") 
    
    ax2.set_yticklabels([])
    ax2.set_yticks([])
    ax2.set_ylabel(acquisition_method,color=expected_fitness_color)
    ax2.legend(loc=legend_location)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label) 
            

    # ---------------------------------------------------------------------------------------------------------------
    # Plot next candidate
    # ---------------------------------------------------------------------------------------------------------------

    ax1.scatter(new_x, new_y, color="red",marker='D', label="Next candidate")

    #plt.xlabel(x_label)
    #plt.ylabel(y_label)
    #plt.title(f"Iteration #{nsteps}, mode={mode} {additional_info}")
    ax1.set_title(f"Iteration #{nsteps}, mode={mode} {additional_info}, best={best_y:.6f}@{best_x:_}")
    ax1.legend()
    plt.show()