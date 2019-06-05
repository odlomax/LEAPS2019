import numpy as np
from healpy import vec2pix,nside2npix
from maths.fields.grid import r_grid,u_r_grid
from scipy.integrate import simps

def make_sigma_kernel(shape,n_side,delta_r=None,n_mc=None,n_mc_max=None,n_theta=100,n_phi=100):
    
    """
    
    Function: make a HEALPix column density directional kernel
    
    Arguments
    ---------
    
    shape(): int
        shape of grid (must have len=3)
        
    n_side: int
        HEALPix n_side variable. n_pix=12*2**n_side
        
    delta_r(): float
        edge length of grid (must have len=3)
        
    n_mc: int
        number of Monte Carlo iterations (corrects for grid-HEALPix mismatches)
        defaults to 120*n_side**2 (i.e. 1 per HP ray) if not specified
        
    n_mc_max: int
        maximum number of MC iterations (defaults to 10 * n_mc)
        
    n_theta: int
        number of theta points in polar integral
        
    n_phi: int
        number of phi point in polar integral
    
    Result
    ------
    
    f_r[:,:,:,:]: float
        kernel value
    
    """
    
    # make HEALPix kernel
    f_r=make_hp_kernel(shape,n_side,delta_r,n_mc,n_mc_max,n_theta,n_phi)
    
    # multiply by 1/r**2 kernel
    f_r*=np.expand_dims(make_r2_kernel(shape,delta_r,n_theta,n_phi),3)
    
    return f_r
    
def make_hp_kernel(shape,n_side,delta_r=None,n_mc=None,n_mc_max=None,n_theta=101,n_phi=101):
    
    """
    
    Function: make a HEALPix directional kernel
    
    Arguments
    ---------
    
    shape(): int
        shape of grid (must have len=3)
        
    n_side: int
        HEALPix n_side variable. n_pix=12*2**n_side
        
    delta_r(): float
        edge length of grid (must have len=3)
        
    n_mc: int
        number of Monte Carlo iterations (corrects for grid-HEALPix mismatches)
        defaults to 120*n_side**2 (i.e. 1 per HP ray) if not specified
        
    n_mc_max: int
        maximum number of MC iterations (defaults to 10 * n_mc)
    
    Result
    ------
    
    f_r[:,:,:,:]: float
        kernel value (value=1 if grid cell in in HP pixel)
    
    """
    
    # get number of HEALPix pixels
    n_pix=nside2npix(n_side)
    
    if delta_r is None:
        delta_r=(1.,1.,1.)
    
    if n_mc is None:
        n_mc=1*n_pix
        
    if n_mc_max is None:
        n_mc_max=n_mc*100
    
    d_r=np.array(delta_r)/np.array(shape)
    
    # get zyx grid of vectors
    r=u_r_grid(shape,normalise=False)*d_r.reshape((1,1,1,3))
    
    # make grid of gridcell dOmega / hp dOmega
    hp_omega=4.*np.pi/n_pix
    cell_omega=(np.product(delta_r)/np.product(shape))**(2./3.)
    omega=cell_omega/((r**2).sum(axis=3)*hp_omega)
    
    # flatten off singularity
    omega[shape[0]//2,shape[1]//2,shape[2]//2]=omega[omega!=np.inf].max()+1.
    
    # get number of MC refinement levels
    n_level=int(np.round(omega.max()))+1
    
    # make array no n_mc values
    n_mc_arr=np.exp(np.linspace(np.log(n_mc),np.log(n_mc_max),n_level)).round().astype(np.int)
    
    # initialise kernel
    f_r=np.zeros((*shape,n_pix))
    
    for i in range(n_level):
        
        level_mask=(omega.round()==float(i))
        n_active_cell=np.count_nonzero(level_mask)
        
        if n_active_cell>0:
        
            print("MC pass over",n_active_cell,"pixels.")
            
            f_r_temp=np.zeros((n_active_cell,n_pix))
            r_temp=r[level_mask,:]
            
            for j in range(n_mc_arr[i]):
                
                if np.mod(j+1,n_mc_arr[i]//10)==0:
                    print("MC iteration",j+1,"of",n_mc_arr[i])
                
                # make grid of random numbers with same shape of r (masked)
                r_num=np.random.uniform(-0.5,0.5,size=(n_active_cell,3))*d_r.reshape((1,3))
                
                # make grid of HEALPix pixel indicies for r+r_num
                r_num+=r_temp
                r_num/=np.expand_dims(np.sqrt((r_num**2).sum(axis=1)),1)
                i_hp=vec2pix(n_side,r_num[:,2],r_num[:,1],r_num[:,0])
                
                for k in range(n_pix):
                    
                    # set mask based on pixel number 
                    pix_mask=i_hp==k
                    
                    # increment f_r value
                    f_r_temp[pix_mask,k]+=1
                            
            # normalise kernel
            f_r_temp/=n_mc_arr[i]
            f_r[level_mask,:]=f_r_temp
    
    return f_r

def make_r2_kernel(shape,delta_r=None,n_theta=101,n_phi=101):
    
    """
    
    Function: make a 1/r**2 directional kernel
    
    Arguments
    ---------
    
    shape(): int
        shape of grid (must have len=3)
        
    delta_r(): float
        edge length of grid (must have len=3)
        
    n_theta: int
        number of theta points in polar integral
        
    n_phi: int
        number of phi point in polar integral
    
    Result
    ------
    
    f_r[:,:,:]: float
        kernel value
        
    """
    
    if delta_r is None:
        delta_r=(1.,1.,1.)
    
    # make box size
    d_x=delta_r[0]/shape[0]
    d_y=delta_r[1]/shape[1]
    d_z=delta_r[2]/shape[2]
    
    # make 1/r**2 grid
    f_r=(d_x*d_y*d_z)/r_grid(shape,delta_r)**2
    
    # set theta and phi grids
    theta_list=np.linspace(0.,0.5*np.pi,n_theta)
    phi_list=np.linspace(0.,0.5*np.pi,n_phi)
    theta,phi=np.meshgrid(theta_list,phi_list,indexing="ij")
    
    # calculate distance from centre of box to edge a function of theta and phi
    l_x=np.sin(theta)*np.cos(phi)
    l_y=np.sin(theta)*np.sin(phi)
    l_z=np.cos(theta)
    
    # distance to box edge is the shorest of the three distances
    d=0.5*np.min([d_x/l_x,d_y/l_y,d_z/l_z],axis=0)
    
    # integrate over phi
    f_r_0=simps(d,phi_list,axis=1)
    
    # integrate over theta
    f_r_0=simps(f_r_0*np.sin(theta_list),theta_list)
    
    # assign central kernel value
    f_r[shape[0]//2,shape[1]//2,shape[2]//2]=8.*f_r_0
    
    return f_r