import numpy as np

def gaussian(r,sigma=1.,ndim=1):
    
    """
    
    Function: gaussian kernel function
    
    Arguments
    ---------
    
    r: float
        r value
    
    sigma: float
        standard deviation
        
    ndim: int
        number of dimensions
        
    
    Result
    ------
    f_r:
        kernel value
    
    """
    
    f_r=np.exp(-r**2/(2.*sigma**2))/(np.sqrt(2.*np.pi*sigma**2)**ndim)
    
    return f_r

def dv_mexican_hat(r,L,v=1.5):
    
    """
    
    Function: Ossenkopf et al (2008) Mexican hat delta variance filter
    
    Arguments
    ---------
    
    r: float
        r value
        
    L: float
        length-scale
        
    v: diameter ratio
    
    Result
    ------
    
    f_r: float
        kernel value
        
    """
    
    f_r=(4.*np.exp(-r**2/(L/2.)**2)/(np.pi*L**2)-
         4.*(np.exp(-r**2/(v*L/2.)**2)-np.exp(-r**2/(L/2.)**2))/(np.pi*L**2*(v**2-1.)))
    
    return f_r

def dv_french_hat(r,L):
    
    """
    
    Function: Stutzki et al (1998) French hat delta variance filter
    
    Arguments
    ---------
    
    r: float
        r value
        
    L: float
        length-scale
        
    Result
    ------
    
    f_r: float
        kernel value
    
    """
    
    f_r=np.zeros(r.shape)
    f_r+=(1./(np.pi*(L/2.)**2))*(r<=L/2.).astype(np.int)
    f_r+=(-1./(8.*np.pi*(L/2.)**2))*np.logical_and(r>L/2.,r<=3.*L/2.).astype(np.int)
        
    return f_r
    
    
    