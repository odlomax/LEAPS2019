import numpy as np

def whiten_points(r):
    
    """
    Function: decorrelate series of points so that covriance = I
    
    Arguements
    ----------
    
    r[:,:]: float
        array of point positions [n_point,n_dim]
        
    Result
    ------
    
    r_new[:,:]: float
        new array of points
    
    """
    
    # subtract mean from r
    r_new=r-r.mean(axis=0,keepdims=True)
    
    # calculate covariance matrix
    sigma=np.cov(r_new.T)
    
    # get eigenvales and eigenvectors of sigma
    w,v=np.linalg.eig(sigma)
    
    # decorrelate r
    r_new=np.matmul(r_new,v)
    
    # rescale r
    r_new/=np.sqrt(w.reshape((1,w.shape[0])))
    
    return r_new
    
    
    
    
    