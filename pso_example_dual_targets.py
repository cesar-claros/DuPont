# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 22:15:23 2022

@author: TOWUZ
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 22:15:23 2022

@author: TOWUZ
"""
import pickle
import numpy as np
import pandas as pd
import scipy
# from sklearn.linear_model import ElasticNet, Ridge, Lasso, LinearRegression
# from sklearn.preprocessing import StandardScaler
from pyswarms.single.global_best import GlobalBestPSO



def objective_fxn_bay_single(x, MODEL, STAND_MU, STAND_SIGMA, TARGET_VAL, SLACK=1, lamb=1):
    '''
    Evaluate objective function for single-target pso for an input feature vector, x

    Parameters
    ----------
    x : 2D numpy array-like
        input feature vector. size should be (#_of_instances)*(#_of_features-1).
    MODEL : sklearn.gaussian_process.GaussianProcessRegressor
        The model linking the input feature vectore to the target value. 
        Output of the model should be a tuple of (predicted_mean, predicted_std)
    STAND_MU : numpy array-like
        the mean values of the standardizer for x. length of the array should be
        len(x[0])+2 for the slack variable and entropy term.
    STAND_SIGMA : numpy array-like
        the standard deviation values of the standardizer for x. length of the array should be
        len(x[0])+2 for the slack variable and entropy term.
    TARGET_VAL : float
        target value
    SLACK : int, optional
        the id of the feature vector that was taken as the "slack variable" and omitted. The default is 1.
    lamb : TYPE, optional
        The term weighing the significance of the uncertainty term. The default is 1.

    Returns
    -------
    ret : 1D numpy array-like
        evaluation of the objective function..

    '''
    ret = np.zeros(x.shape[0])
    for i,xi in enumerate(x):
        
        if np.sum(xi) > 1:
            ret[i] = 1e5
            continue
        xi = np.insert(xi, SLACK, 1-np.sum(xi))
        x_with_entropy = np.append(xi,scipy.stats.entropy(xi))
        x_stand = (x_with_entropy-STAND_MU.to_numpy())/STAND_SIGMA.to_numpy()
    
        y_mean0,y_std0 = MODEL.predict(x_stand.reshape(1,-1),return_std=True)
        
        
        
        ret[i] = (y_mean0-TARGET_VAL)**2+lamb*y_std0**2
            
            
    return ret

def objective_fxn_bay_dual(x, MODELs, STAND_MU, STAND_SIGMA, TARGET_VAL, SLACK=1, lamb=1):
    ret = np.zeros(x.shape[0])
    for i,xi in enumerate(x):
        
        if np.sum(xi) > 1:
            ret[i] = 1e5
            continue
        xi = np.insert(xi, SLACK, 1-np.sum(xi))
        x_with_entropy = np.append(xi,scipy.stats.entropy(xi))
        x_stand = (x_with_entropy-STAND_MU.to_numpy())/STAND_SIGMA.to_numpy()
    
        y_mean0,y_std0 = MODELs[0].predict(x_stand.reshape(1,-1),return_std=True)
        y_mean1,y_std1 = MODELs[1].predict(x_stand.reshape(1,-1),return_std=True)
        
        
        ret[i] = (y_mean0-TARGET_VAL[0])**2+lamb*y_std0**2\
                +(TARGET_VAL[0]/TARGET_VAL[1]*(y_mean1-TARGET_VAL[1]))**2\
                +lamb*(TARGET_VAL[0]/TARGET_VAL[1])*y_std1**2
        
            
            
    return ret


#%%
if __name__ == "__main__":
    ## DEFINE "single" OR "dual" target pso
    MODE = "dual"
    ## DEFINE HYPERPARAMETERS
    slack = 1
    ps_options = {'c1': 0.1, 'c2': 0.06, 'w':0.2}
    total_dims = 28
    
    ## LOAD MODELS FOR PARAMETER 1 AND PARAMETER 2
    params_filename = 'standardizer_full_dataset_noTC.dump'
    models_filename = 'full_dataset_models_noTC.dump'
    directory = 'trained_models_params/'
    models = pickle.load(open(directory+models_filename,'rb'))
    model0 = models[0]
    mu, sigma = pickle.load(open(directory+params_filename,"rb"))

    params_filename = 'standardizer_full_GPR_Hardness.dump'
    models_filename = 'full_dataset_models_GPR_Hardness.dump'
    directory = 'trained_models_params/'
    models = pickle.load(open(directory+models_filename,'rb'))
    model1 = models[1]
    
    ## LOAD TRAINING SET INSTANCES AS INITIAL POINTS FOR PSO
    train_dataset_file = "data/UD_867_formulation_training.csv"
    trainingset = pd.read_csv(train_dataset_file)
    TARGETS = ['Water_Absorption_%','Hardness','Thermal_Conductivity_(mW/m.K)']
    X_train = trainingset.drop(columns=['name']+TARGETS)
    X_train_numpy = X_train.to_numpy()
    
    slack_column = 1
    init_pos_from_train = np.hstack((X_train_numpy[:,:slack_column],X_train_numpy[:,slack_column+1:]))
    x_mins = np.zeros(27)
    x_maxs = np.ones(27)
    optimizer = GlobalBestPSO(n_particles = len(init_pos_from_train), 
                          dimensions = 27, 
                          options = ps_options,
                          bounds = (x_mins,x_maxs),
                          init_pos = init_pos_from_train)   
    
    ## RUN OPTIMIZER
    if MODE == "single":
       ## DEFINE TARGET
       tar = 2.5
       cost, pos = optimizer.optimize(objective_fxn_bay_single,500,
                                  MODEL=model0,
                                  STAND_MU=mu,
                                  STAND_SIGMA=sigma,
                                  TARGET_VAL=tar,
                                  SLACK=slack,
                                  lamb = 0,
                                  )  
    
    if MODE == "dual":
        ## DEFINE TARGET
        tar = (2.5,120)
        cost, pos = optimizer.optimize(objective_fxn_bay_dual,500,
                                   MODELs=[model0,model1],
                                   STAND_MU=mu,
                                   STAND_SIGMA=sigma,
                                   TARGET_VAL=tar,
                                   SLACK=slack,
                                   lamb = 0,
                                   ) 