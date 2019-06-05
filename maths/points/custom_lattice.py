import numpy as np
import sys

def cube(n,cube_file):
    
    """
    
    Funtion: generate a cube of points based on custom cubic primative
    
    Arguments
    ---------
    
    n: int
        number of partices in lattice
        
    cube_file: string
        name of cube primative input file.
        File should contain columns of particle coordinates in the interval [0:1]
        
    Result
    ------
    
    r[n,3]:
    
    """
    
    # import particle positions from file
    r_block=np.loadtxt(cube_file)
    
    # make block larger if necessary
    for i in range(sys.maxsize):
        if r_block.shape[0]>n:
            break
        else:
            # make tiling of 8 sub cubes
            r_block_000=r_block+[[0.,0.,0.]]    
            r_block_001=r_block+[[0.,0.,1.]]
            r_block_010=r_block+[[0.,1.,0.]]
            r_block_011=r_block+[[0.,1.,1.]]
            r_block_100=r_block+[[1.,0.,0.]]
            r_block_101=r_block+[[1.,0.,1.]]
            r_block_110=r_block+[[1.,1.,0.]]
            r_block_111=r_block+[[1.,1.,1.]]
            
            r_block=0.5*np.concatenate((r_block_000,r_block_001,r_block_010,r_block_011,r_block_100,r_block_101,r_block_110,r_block_111))
            
    # trim block down to n units
    indices=r_block.max(1).argsort()
    r_block=r_block[indices,:]
    r=r_block[:n,:]
    r/=r[-1,:].max()
    
    return r
    
    
            
            

    
    