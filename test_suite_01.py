#########################################
# Imports
#########################################

# import class to construct Gaussian random fields (fBm fields are a subset of these)
from maths.fields.gaussian_random_field import scalar_grf

# import class to generate random points from a probability density function 
from maths.random.probability_density_function import pdf

# import pyplot and numpy
import numpy as np
from matplotlib import pyplot as plt



#########################################
# Set params
#########################################

# set number of dimensions (can be set to 3, but imshow won't work)
n_dim=2

# set size of our discrete density field
grid_size=[200]*n_dim

# set random seed (set r_seed=None for different field every run)
r_seed=0
np.random.seed(seed=r_seed)

# set H and sigma
H=0.5
sigma=1.

# set numer of "stars"
n_star=1000


#########################################
# Make fBm field
#########################################

# set spectral index
beta=n_dim+2*H

# genrate field (this is complex and periodic, but we just want the real bit)
fBm_field=scalar_grf(grid_size,beta)

# normalise field
fBm_field.normalise(sigma=sigma,exponentiate=True)

# set periodic centre of mass to centre of grid
fBm_field.com_shift()

# use real component of fBm field as pdf
real_field=fBm_field.signal.real

# print real_field
plt.figure(0)
plt.imshow(real_field.T,origin="lower") # only works if n_dim=2



#########################################
# Make fBm cluster
#########################################

# generate probability density function object
fBm_pdf=pdf(real_field)

# randomly sample cluster of stars from pdf
fBm_cluster=fBm_pdf.random(n_star)

# print fBm_cluster
plt.figure(1)
plt.scatter(fBm_cluster[:,0],fBm_cluster[:,1])
plt.show()

