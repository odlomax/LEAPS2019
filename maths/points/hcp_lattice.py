import numpy as np

def hcp(n_i,n_j=None,n_k=None,fuzzy=False,fuzzy_dist=1,cube_trim=False):
    
    """
    
    Function: returns the positions of an hexagonal close packed array
    
    Arguments
    ---------
    
    n_i: int
        number of points along x axis
    
    n_j (optional): int
        number of rows along y axis
        
    n_k (optional): int
        number of layers along z axis
        
    fuzzy (optional): boolean
        give points random perturbation within unit cell
        
    cube_trim (optional): boolean
        trim lattice to cube of edge-length 1 (usefull for fuzzy lattice)
        
        
    Result
    ------
    
    r[n_i*n_j*n_k,3]: float
        lattice point positions
        
    """
    
    # if n_j and n_k are missing, make lattice cubic
    
    if n_j==None:
        n_j=int(n_i*2./np.sqrt(3.))
        
    if n_k==None:
        n_k=int(n_i*3./np.sqrt(6.))
        
    
        
    i=np.arange(1,n_i+1)
    j=np.arange(1,n_j+1)
    k=np.arange(1,n_k+1)
    
    # convert i, j and k in to grid format
    i,j,k=np.meshgrid(i,j,k,indexing="ij")
    
    # set r array
    r=np.zeros((n_i*n_j*n_k,3))
    
    # set x values
    r[:,0]=np.ndarray.flatten(2.*i+np.mod(j+k,2))
    
    # set y values
    r[:,1]=np.ndarray.flatten(np.sqrt(3.)*(j+np.mod(k,2)/3.))
    
    # set z values
    r[:,2]=np.ndarray.flatten(2.*np.sqrt(6.)*k/3.)
    
    # perturb points, if fuzzy
    if fuzzy:
        r[:,0]=np.random.normal(size=r.shape[0],loc=r[:,0],scale=fuzzy_dist)
        r[:,1]=np.random.normal(size=r.shape[0],loc=r[:,1],scale=fuzzy_dist)
        r[:,2]=np.random.normal(size=r.shape[0],loc=r[:,2],scale=fuzzy_dist)
        
       
    # normalise coordinates
    r/=(2.*n_i+1.)
    
    # shift centre of mass to 0.5
    r=r-r.mean(0)+0.5
    
    # give fuzzy lattice a shave
    if cube_trim:
        r=r[np.logical_and(r>=0.,r<=1.).all(axis=1)]
    
    return r
    
    
    
    
    
    
    