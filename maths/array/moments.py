import numpy as np

def nth_moment(x,n,w=None,x_0=None):
    
    """
    Function: calculate nth moment of variables x
    
    Arguements
    ----------
    
    x[:]: float
        array of x values
        
    n: int
        order of moment
        
    w[x.shape]: float
        weights of x values (set to one, if missing)
        
    x_0: float
        centre of distribution (set to mean value if missing)
        
    Result
    ------
    
    m_x: nth moment of x
    
    """
    
    # set weights and centre, if missing
    
    if w==None:
        w=np.ones(x.shape)
        
    if x_0==None:
        x_0=np.sum(x*w)/np.sum(w)
        
    # calculate moment
    m_x=np.sum(w*(x-x_0)**n)/np.sum(w)
    
    # calc real nth root of m_x (so it has same physical dimensions as x)    
    m_x=np.abs(m_x)**(1./n)*np.sign(m_x)**n
    
    return m_x