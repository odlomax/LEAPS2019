import numpy as np

def cum_dist_func(xp,fp):
    
    """
    Function: returns cumulative distribution function. (trapezium rule)
    
    Arguments
    ---------
    xp[:]: float
        1d array of x points
    fp[x.shape[0]]: float
        1d array of f points
        
    Result
    ------
    
    fp_cum[x.shape[0]]: float
        cumulative distribution function of f 
    """
    # set array
    f_cum=np.zeros(xp.shape[0])
    
    # calc cumulative distribution function
    f_cum[1:]=np.cumsum(0.5*(fp[1:]+fp[:-1])*(xp[1:]-xp[:-1]))
        
    return f_cum