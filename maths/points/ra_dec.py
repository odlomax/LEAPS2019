import numpy as np
from maths.vectors.polars import polar_to_cart_3d
from maths.vectors.polars import cart_to_polar_3d
from maths.tensors.rotation_matrix import r_matrix_3d

def ra_dec_project(ra,dec,d=1.):
    
    """
    Function: convert ra and dec to Cartesian coordinates
    
    ra[:,3]: float
        right ascension {hrs,min,sec}
        
    dec[:,3]: float
        declination {deg,min,sec}
        
    d[:]: float
        distances
        
    Result
    r[:,3]: float
        3D Cartesian coordinates (r[:,:2] is the line of sight projection) 
    
    """
    
    
    # get polar coords
    phi=ra[:,0]*(np.pi/12.)+np.sign(ra[:,0])*ra[:,1]*(np.pi/720.)+np.sign(ra[:,0])*ra[:,2]*(np.pi/43200.)
    theta=dec[:,0]*(np.pi/180.)+np.sign(dec[:,0])*dec[:,1]*(np.pi/10800.)+np.sign(dec[:,0])*dec[:,2]*(np.pi/648000.)

    # get Cartesian coords
    r=np.array(polar_to_cart_3d(d,phi,theta))

    # rotate z-axis to mean direction
    r_mean,phi_mean,theta_mean=cart_to_polar_3d(*r.mean(axis=1))

    R_1=r_matrix_3d([0.,0.,1.],-phi_mean)
    R_2=r_matrix_3d([0.,1.,0.],-theta_mean)

    r=R_1.rotate_vec(r)
    r=R_2.rotate_vec(r)

    return r.T
        
    



