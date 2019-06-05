import numpy as np

def gw04_box_fractal(E,D,N_max,jitter=False):
    
    """
    
    Function: generate Goodwin and Whitworth (2004) box fractal
    
    Arguments
    ---------
    
    E: int
        Eucliden dimension
        
    D: float
        notional fractal dimension
        
    N_max: int
        number of generations
        
    Result
    ------
    
    r[:,:]: float
        array of positions [n_points:n_dim]
    
    """
    
    def build_fractal(N,N_max,D,r,r_list):
        
        """
        
        subroutine: recursive routine to build fractal
        
        Arguments
        ---------
        
        N: int
            current generation number
            
        N_max: int
            maximum number of generations
            
        D: float
            fractal dimension
            
        r[E]: float
            current position
            
        r_list[:]: float
            list of all saved positions
        
        """
        
        if N==N_max:
            
            # reached N_max, append r to list
            r_list.append(r)
            
        else:
            
            # build more generations
            
            # decide which generations to populate
            populate_generations=np.zeros(2**r.size)
            index=int(2**D)
            populate_generations[:index]=1.
            if 2**D-index > np.random.random():
                populate_generations[index]=1.
                
            np.random.shuffle(populate_generations)
            
            for i in range(populate_generations.size):
                if populate_generations[i]:
                    r_new=np.array([float(j) for j in np.binary_repr(i,width=r.size)])
                    r_new=2.*(r_new-0.5)*0.5**(N+1)
                    build_fractal(N+1,N_max,D,r+r_new,r_list)
                    
        return
    
    # initialise variables
    r=np.array([0.]*E)
    r_list=[]
    N=0
    
    # build fractal
    build_fractal(N,N_max,D,r,r_list)
    
    r=np.array(r_list)
    
    # add random displacement to positions
    if jitter:
        r_jitter=np.random.uniform(-1.,1.,r.shape)*0.5**(N_max)
        r+=r_jitter
    
    return r