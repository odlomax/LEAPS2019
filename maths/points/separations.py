import numpy as np
from scipy import spatial

def mean_separation(r):
    
    """
    Function: calculates the mean separation between points in a reasonably memory-efficient way
    
    Arguments
    ---------
    r[:,:]: float
        array of point positions [n_points:n_dim]
        
    Result
    ------
    mu_sep: float
        mean separation between points
        
    """
    
    mu_sep=0.
    
    for i in range(r.shape[0]-2,-1,-1):
        mu_sep+=np.sum(spatial.distance.cdist(r[i+1:,:],[r[i,:]]))
    
    mu_sep/=0.5*r.shape[0]*(r.shape[0]-1)
    
    return mu_sep

def mean_squared_separation(r):
    
    """
    Function: calculates the mean squared separation between points in a reasonably memory-efficient way
    
    Arguments
    ---------
    r[:,:]: float
        array of point positions [n_points:n_dim]
        
    Result
    ------
    mu2_sep: float
        mean squared separation between points
        
    """
    
    mu2_sep=0.
    
    for i in range(r.shape[0]-2,-1,-1):
        mu2_sep+=np.sum(spatial.distance.cdist(r[i+1:,:],[r[i,:]],"sqeuclidean"))
    
    mu2_sep/=0.5*r.shape[0]*(r.shape[0]-1)
    
    return mu2_sep