import numpy as np

def cart_to_polar_2d(x,y):
    
    """
    Function: Convert 2D Cartesian coordinates into circular polars
    
    Arguments
    ---------
    
    x,y: float
        x and y coordinates
    
    Result
    ------
    r,phi
        r and phi coordinates
    
    """
    
    r=np.sqrt(x**2+y**2)
    phi=np.arctan2(y,x)
    
    return r,phi

    
def polar_to_cart_2d(r,phi):
    
    """
    Function: Convert circular polar coordinates into 2D Cartesian
    
    Arguments
    ---------
    
    r,phi: float
        r and phi coordinates
    
    Result
    ------
    x,y
        x and y coordinates
    
    """
    
    x=r*np.cos(phi)
    y=r*np.sin(phi)
    
    return x,y
    
def cart_to_polar_3d(x,y,z):
    
    """
    Function: Convert 3D Cartesian coordinates into spherical polars
    
    Arguments
    ---------
    
    x,y,z: float
        x, y and z coordinates
    
    Result
    ------
    r,phi,theta
        r, phi and theta coordinates
    
    """
    
    r=np.sqrt(x**2+y**2+z**2)
    phi=np.arctan2(y,x)
    theta=np.arccos(z/r)
    
    return r,phi,theta
    
    
def polar_to_cart_3d(r,phi,theta):
    
    """
    Function: Convert spherical polar coordinates into 3D Cartesian
    
    Arguments
    ---------
    
    r,phi,theta: float
        r, phi and theta coordinates
    
    Result
    ------
    x,y,z
        x and y coordinates
    
    """
    
    sin_theta=np.sin(theta)
    
    x=r*sin_theta*np.cos(phi)
    y=r*sin_theta*np.sin(phi)
    z=r*np.cos(theta)
    
    return x,y,z