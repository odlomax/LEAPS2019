import numpy as np

def change_basis_2d(v,u_x,u_y):
    
    """
    Function: Change unit vectors of v
    
    Arguments
    ---------
    
    v[2]: float
        input vector
        
    u_x[2]: float
        new x unit vector
    
    u_y[2]: float
        new y unit vector
        
    Result
    ------
    
    w[2]: float
        transformed vector
    """
    
    # initialise w
    w=np.zeros(2)
    
    # set w
    w[0]=np.dot(v,u_x)
    w[1]=np.dot(v,u_y)
    
    return w

def change_basis_3d(v,u_x,u_y,u_z):
    
    """
    Function: Change unit vectors of v
    
    Arguments
    ---------
    
    v[3]: float
        input vector
        
    u_x[3]: float
        new x unit vector
    
    u_y[3]: float
        new y unit vector
        
    u_z[3]: float
        new z unit vector
        
    Result
    ------
    
    w[3]: float
        transformed vector
    """
    
    # initialise w
    w=np.zeros(3)
    
    # set w
    w[0]=np.dot(v,u_x)
    w[1]=np.dot(v,u_y)
    w[2]=np.dot(v,u_z)
    
    return w
    

