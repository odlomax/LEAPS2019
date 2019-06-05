import numpy as np

def power_law_profile(E,alpha,N):
    
    """
    
    Function: generate a sphere of points with a radial power law profile
    
    Arguments
    ---------
    
    E: int
        Euclidean dimension
    
    alpha: float
        radial density exponent
        
    N: int
        number of points
        
    result: r[:,:]: float
        array of positions [n_points:n_dim]
    
    """
    
    # generate at set of N random direction vectors
    r=np.random.normal(size=(N,E))
    r/=(np.sqrt((r**2).sum(axis=1))).reshape((N,1))
    
    # generate r values for each vector
    r_len=np.random.random((N,1))
    r_len=(r_len)**(1./(E-alpha))
    
    # set positions
    r*=r_len
    
    return r
    
