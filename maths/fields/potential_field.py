import numpy as np
from scipy.integrate import simps

def dft_potential(density,delta_x=None,periodic=False,output_all=False):
    
    """
    Function: computes potential field using discrete fourier transform
    
    Arguments
    ---------
    
    density[...]: float
        ndarray of density values (assume uniform grid with edge-length 1)
        
    delta_x[:]: float
        edge-lengths of density grid
        
    periodic: boolean
        perform periodic claculation (each value is central potential of shifted finite volume)
    
    Result
    ------
    
    potential[density.shape]: float
        ndarray of potential values
        
    """
    
    # set edge-lengths
    if delta_x==None:
        delta_x=np.ones(density.ndim)
        
    # get size of each grid element
    dx=np.array(delta_x)/(np.array(density.shape)-1.)
    
    # make zero-padded density array
    if periodic:
        density_array=density
    else:
        density_array=np.zeros([2*i for i in density.shape])
        density_array[[slice(0,i) for i in density.shape]]=density
    
    # make list of x values
    x_list=[]
    for i in range(density_array.ndim):
        x_list.append(np.ceil(np.arange(-density_array.shape[i]/2,density_array.shape[i]/2)))
        x_list[i]=np.roll(x_list[i],np.ceil(-density_array.shape[i]/2).astype(np.int))*delta_x[i]/(density.shape[i]-1)
    # make Green's function array
    x_grid=np.meshgrid(*x_list,indexing="ij")
    greens_array=np.zeros(density_array.shape)
    for i in x_grid:
        greens_array+=i**2
    greens_array=-1./np.sqrt(greens_array)
    
    # normalise greens_array by dx
    greens_array*=np.product(dx)
    
    # add x=0 component
    if density.ndim==2:
        # can solve exactly
        greens_array[(0,)*greens_array.ndim]=-2.*(dx[0]*np.arcsinh(dx[1]/dx[0])+dx[1]*np.arcsinh(dx[0]/dx[1]))
    elif density.ndim==3:
        # must solve numerically
        greens_array[(0,)*greens_array.ndim]=box_potential_3d(*dx)
    else:
        # no obvious solution
        greens_array[(0,)*greens_array.ndim]=0.
        
    # calculate potential
    potential_array=np.fft.irfftn(np.fft.rfftn(density_array)*np.fft.rfftn(greens_array),s=density_array.shape)
    
    if periodic:
        potential=potential_array
    else:
        potential=potential_array[[slice(0,i) for i in density.shape]]        
    
    if output_all:
        return density_array,greens_array,potential
    else:
        return potential


def direct_potential(density,delta_x=None):
    
    """
    Function: computes potential field directly (mainly for testing other methods)
    
    Arguments
    ---------
    
    density[...]: float
        ndarray of density values (assume uniform grid with edge-length 1)
        
    delta_x[:]: float
        edge-lengths of density grid
    
    Result
    ------
    
    potential[density.shape]: float
        ndarray of potential values
        
    """
    
    # set edge-lengths
    if delta_x==None:
        delta_x=np.ones(density.ndim)
    
    # get size of each grid element
    dx=np.array(delta_x)/(np.array(density.shape)-1)
    
    # make list of x values
    x_list=[]
    for i in range(len(density.shape)):
        x_list.append(np.linspace(0.,delta_x[i],density.shape[i]))
        
    # make position array
    x_grid=np.meshgrid(*x_list,indexing="ij")
    x=np.zeros((*density.shape,density.ndim))
    for i in range(len(x_grid)):
        x[...,i]=x_grid[i]
        
    # calculate potential
    potential=np.zeros(density.shape)
    for i, rho in np.ndenumerate(density):
        inv_dx=1./np.sqrt(((x-x[tuple([*i,slice(None)])].reshape(tuple([*[1]*(x.ndim-1),x.ndim-1])))**2).sum(axis=x.ndim-1))
        inv_dx[i]=0.
        potential[i]=-(density*inv_dx).sum()
    
    potential*=np.product(dx)
        
    
    # add correction for potential component from within each cell
    if density.ndim==2:
        # can solve exactly
        corr=-2.*(dx[0]*np.arcsinh(dx[1]/dx[0])+dx[1]*np.arcsinh(dx[0]/dx[1]))
    elif density.ndim==3:
        # must solve numerically
        corr=box_potential_3d(*dx)
    else:
        # no obvious solution
        corr=0.
    
    potential+=density*corr
    
    return potential

def box_potential_3d(delta_x,delta_y,delta_z,n_theta=101,n_phi=101):
    
    """
    
    Function: Calculates the potential at the centre of a uniform density cuboid (rho=1)
    
    Arguments
    ---------
    
    delta_x, delta_y, delta_z: float
        edge-lengths of box
        
    n_theta, n_phi: int
        number of samples to integrate over
        
    Result
    ------
    
    potential: float
        potential at centre of cuboid
        
    """
    
    # set theta and phi grids
    theta_list=np.linspace(0.,0.5*np.pi,n_theta)
    phi_list=np.linspace(0.,0.5*np.pi,n_phi)
    theta,phi=np.meshgrid(theta_list,phi_list,indexing="ij")
    
    # calculate distance from centre of box to edge a function of theta and phi
    # caluclate line unit vector
    l_x=np.sin(theta)*np.cos(phi)
    l_y=np.sin(theta)*np.sin(phi)
    l_z=np.cos(theta)
    
    # distance to box edge is the shorest of the three distances
    d=0.5*np.min([delta_x/l_x,delta_y/l_y,delta_z/l_z],axis=0)
    
    # calulate r part of spherical polar integral
    F_d=0.5*d**2
    
    # integrate over phi
    potential=simps(F_d,phi_list,axis=1)
    
    # integrate over theta
    potential=simps(potential*np.sin(theta_list),theta_list)
    
    # integral is over one octant of box
    potential*=-8.
    
    return potential