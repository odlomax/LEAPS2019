import numpy as np
from scipy.spatial import KDTree

def fuse_close_companions(x,d):
    
    """
    Function: fuse points in x which are closer than d
    
    Arguments
    ---------
    
    x[:,:]: float
        array of points [n_point:n_dim]
        
    d: float
        fuse points that are closer than this distance
        
    Result
    ------
    
    x_new[:,:]
        new version of x with fused points
        
    
    """
    
    # make KDTree object
    tree=KDTree(x)
    
    # make neighbour list
    neib_list=tree.query_ball_tree(tree,d)
    
    # loop through x and make a new list of fused points
    x_new=[]
    point_active=[True]*x.shape[0]
    for i in range(x.shape[0]):
        if point_active[i]:
            # fuse neighbours
            x_new.append(x[neib_list[i],:].mean(axis=0))
            # loop over neighbour list and deactivate points
            for j in neib_list[i]:
                point_active[j]=False
    
    return np.array(x_new)
                
        