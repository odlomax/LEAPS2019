import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Finds stars which are members of the named cluster. This data is from 2000.
def cluster_list(name,save=False):
    colnames=['region', 'desig', 'mem_ra', 'mem_dec', 'flag', 'ME', 'J', 'H', 'alpha_IRAC', 'logL_X', 'Age_JX', 'Clus'] 
    data = pd.read_csv('Cluster_Memberships.txt',delimiter='\s+&\+*', header=None, names=colnames, engine='python')
    selected_cluster = data[data['region'] == name]
    if save==True:
        selected_cluster.to_csv('%s_Member_List.csv'%name)
    return selected_cluster


# Find the dimensions of a rectangle containing all stars described in the input database
def find_dimensions(stars):
    max_RA = stars['mem_ra'].max()
    min_RA = stars['mem_ra'].min()
    max_Dec = stars['mem_dec'].max()
    min_Dec = stars['mem_dec'].min()
    return min_RA, max_RA, min_Dec, max_Dec


# Returns the centre coordinates and radius of a circle containing the rectangle which contains all stars in the input database.
def circle_around_stars(stars):
    min_RA, max_RA, min_Dec, max_Dec = find_dimensions(stars)
    RA, Dec = np.mean([min_RA,max_RA]),np.mean([min_Dec, max_Dec])
    radius = np.linalg.norm([(max_RA - min_RA)/2, (max_Dec - min_Dec)/2])
    radius=radius
    return RA, Dec,radius


# Finds the Gaia data from 2015.5 and projected 2000 data for sources in the region of a cluster
def gaia_data_projected(stars):
    from astroquery.gaia import Gaia
    import warnings
    warnings.filterwarnings('ignore')
    
    RA,Dec,radius = circle_around_stars(stars)
    
    job = Gaia.launch_job_async("SELECT *, \
        array_element(a0,1) as ra_2000, \
        array_element(a0,2) as dec_2000, \
        array_element(a0,3) as parallax_2000, \
        array_element(a0,4) as pmra_2000, \
        array_element(a0,5) as pmdec_2000, \
        array_element(a0,6) as rv_2000 \
        FROM \
        ( \
         SELECT gaia_source.source_id,gaia_source.ra,gaia_source.ra_error,gaia_source.dec,gaia_source.dec_error,gaia_source.parallax,gaia_source.parallax_error,gaia_source.pmra,gaia_source.pmra_error,gaia_source.pmdec,gaia_source.pmdec_error,gaia_source.astrometric_n_good_obs_al,gaia_source.astrometric_gof_al,gaia_source.astrometric_chi2_al,gaia_source.visibility_periods_used,gaia_source.phot_g_mean_flux_over_error,gaia_source.phot_g_mean_mag,gaia_source.phot_bp_mean_flux_over_error,gaia_source.phot_rp_mean_flux_over_error,gaia_source.phot_bp_rp_excess_factor,gaia_source.bp_rp,gaia_source.bp_g,gaia_source.g_rp,gaia_source.radial_velocity,gaia_source.radial_velocity_error, \
         epoch_prop(ra,dec,parallax,pmra,pmdec,radial_velocity,2015.5,2000) as a0 \
         FROM gaiadr2.gaia_source \
         WHERE \
         CONTAINS(POINT('ICRS',gaiadr2.gaia_source.ra,gaiadr2.gaia_source.dec),CIRCLE('ICRS',%s,%s,%s))=1 ) \
         as p "%(RA,Dec,radius) \
    , dump_to_file=True)  
    r = job.get_results()
    return r


# Delete files created by Gaia query (all files in the folder containing 'async')
def del_files():
    import subprocess
    subprocess.call(['./delete_gaia_query_files.sh'])
    
    
# Given a cluster name finds member stars and Gaia data from that region
# Potentially save these datasets for analysis
def find_gaia_region(cluster_name,save_member_list = False,save_gaia_region=False):
    stars = cluster_list(cluster_name)
    gaia_stars = gaia_data_projected(stars)
    if save_gaia_region==True:
        df = gaia_stars.to_pandas()
        df.to_csv('%s_Region_Gaia_Data.fits'%cluster_name)
    if save_member_list==True:
        stars.to_csv('%s_Member_List.csv'%cluster_name)
    del_files()
    return stars,gaia_stars


# For data sets in the form of arrays finds nearest neighbors
# Returns the indices of the neighbors of the first data set in the second and distances between neighbors
# Could be further optimised by finding neighbors for first entry only?
def indices_of_nearest_neighbors_arrays(set_1,set_2,num_neighbors=1):
    from sklearn.neighbors import NearestNeighbors
    (w,x)=set_1.shape
    (y,z)=set_2.shape   
    k_neighbors = num_neighbors+1
    neighbor_index = np.zeros((w,num_neighbors))
    distances = np.zeros((w,num_neighbors))
    for i in range(w):
        row = np.copy(set_1[i].reshape((1,2)))
        Z = np.concatenate((row,set_2),axis=0)
        nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='ball_tree').fit(Z)
        dist, indices = nbrs.kneighbors(Z)
        neighbor_index[i] = np.copy(indices[0,1:]-np.ones_like(indices[0,1:]))
        distances[i] = np.copy(dist[0,1:])     
    return neighbor_index,distances


# Find matches between stars in Gaia data and membership lists and the distances between them
# Output a dataframe with the data from the membership list followed by that of the Gaia counterparts and distances between them
def distances_from_match(cluster_name,num_matches=1,save=False):
    stars,gaia_stars = find_gaia_region(cluster_name)
    gaia_stars_df = pd.DataFrame(np.array(gaia_stars))
    stars_coord = stars[['mem_ra','mem_dec']].values
    gaia_stars_coord = gaia_stars_df[['ra_2000','dec_2000']].values  
    neighbor_index,distances = indices_of_nearest_neighbors_arrays(stars_coord,gaia_stars_coord,num_neighbors=num_matches)    
    stars.reset_index(inplace=True,drop=True)    
    for i in range(num_matches):
        temp = gaia_stars_df.reindex(neighbor_index[:,i])
        temp['distance_%s'%(i+1)] = distances[:,i]
        temp.reset_index(inplace=True,drop=True)
        stars = pd.concat([stars,temp],axis=1)       
    if save==True:
        stars.to_csv('%s_counterparts.csv'%cluster_name)       
    return stars,gaia_stars.to_pandas()
   
    
# Generate logarithmic histogram given data and number of bins
def plot_loghist(x, bins):
  hist, bins = np.histogram(x, bins=bins)
  logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
  plt.hist(x, bins=logbins)
  plt.xscale('log')
  

# Plot the positions of stars and counterparts
def plot_counterparts(data,gaia_data):
    fig = plt.figure(figsize = (6,6))
    plt.xlabel('ra (degrees)')
    plt.ylabel('dec (degrees)')
    plt.title('Catalogue stars and Gaia Counterparts')
    plt.scatter(gaia_data.ra_2000,gaia_data.dec_2000,label='Gaia Stars',marker='+',s=5,color='lightgrey')
    plt.scatter(data.mem_ra,data.mem_dec,label='Catalogue',s=20, facecolors='none', edgecolors='navy')
    plt.scatter(data[data.good_match==True].ra_2000,data[data.good_match==True].dec_2000,label='Good Matches',marker='.',s=5,color='c')
    plt.scatter(data[data.good_match==False].ra_2000,data[data.good_match==False].dec_2000,label='Poor Matches',marker='.',s=5,color='r')
    plt.legend()
    plt.show()


# Plot histogram of distances between matches
def plot_dist_hist(stars):
    dist_to_plot = np.array(stars['distance_1']*60*60)
    fig_1 = plt.figure()
    plot_loghist(dist_to_plot,30)
    plt.grid(axis='y', alpha=0.75)
    plt.title('Seperation of Gaia and Catalogue Matches')
    plt.xlabel('Distance Between Nearest Neighbours (arcsecs)')
    plt.ylabel('Frequency')
    plt.show()


# Remove rows with matches of less than a certain angular displacement in arcseconds
def tidy_counterparts(cluster_name,max_dist=1,num=1,save=False,distances_hist=False):
    max_dist=max_dist/(60*60)
    stars,gaia_stars=distances_from_match(cluster_name,num_matches=num)
    (a,b)=stars.shape    
    empty_parallax = stars['parallax'].isnull()
    stars['good_match']= np.ones((a),dtype=bool)
    
    for i in range(a):
        if (stars.at[i,'distance_1']>max_dist) | (empty_parallax[i]==True) | (stars.at[i,'parallax']<0.1):
            stars.at[i,'good_match']=False
        else:
            gaia_id = stars.at[i,'source_id']
            dist = stars.at[i,'distance_1']
            matches = stars.loc[(stars.desig != stars.at[i,'desig'])&(stars.source_id==gaia_id)&stars.good_match==True]
            better_matches = matches.loc[matches.distance_1 <= dist]
            if len(better_matches)!=0:
                stars.at[i,'good_match']=False
        
    if save==True:
        stars.to_csv('%s_tidy_counterparts.csv'%cluster_name)        
    if distances_hist==True:
        plot_dist_hist(stars)    
    plot_counterparts(stars,gaia_stars)
    return stars


def optimise_max_dist_matching(stars,max_dist=1):
    max_dist=max_dist/(60*60)
    (a,b)=stars.shape 
    stars['good_match']= np.ones((a),dtype=bool)
    
    for i in range(a):
        if (stars.at[i,'distance_1']>max_dist) | (empty_parallax[i]==True) | (stars.at[i,'parallax']<0.1):
            stars.at[i,'good_match']=False
    
    (a,b) = stars[stars.good_match==True].shape
    return a


# Investigate the optimum maximum separation between matches
def star_matching_criterion(cluster_name,max_dist=False,neighbors=False):
    if max_dist==True:
        stars=distances_from_match(cluster_name,num_matches=1)
        num_stars = []
        distances = np.concatenate((np.linspace(0.01,2,100),np.linspace(2,200,num=200)))
        distances = distances[::-1]
        for dist in distances:
            num_stars.append(optimise_max_dist_matching(stars,max_dist=dist))
        plt.figure(figsize=(6,6))
        plt.plot(distances,num_stars,markersize=1)
        plt.xlabel('max separation (arcseconds)')
        plt.ylabel('number stars')
        plt.suptitle('Number of Matches wrt Max Separation')
        
        plt.figure(figsize=(6,6))
        plt.loglog(distances,num_stars,markersize=1)
        plt.xlabel('max separation')
        plt.ylabel('number stars')
        plt.suptitle('Log log Number of Matches wrt Max Separation')
        plt.show()    
    if neighbors==True:
        stars=distances_from_match(cluster_name,num_matches=2)        
        neighbor_1 = stars[['distance_1']]*(60*60)
        neighbor_2 = stars[['distance_2']]*(60*60)
        plt.figure(figsize=(6,6))
        plt.scatter(neighbor_1,neighbor_2,marker='.')
        plt.plot([2, 2], [0, 160], color='r', linestyle='-', linewidth=1)
        plt.plot([0, 90], [2, 2], color='r', linestyle='-', linewidth=1)
        plt.xlabel('nearest (arcsec)')
        plt.ylabel('next nearest (arcsec)')
        plt.show()    
   


# Takes in table of data on stars and gives out an array of euclidean coord.s in pc
def change_to_cartesian(r,phi,theta):
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    x = x - np.mean(x)
    y = y - np.mean(y)
    z = z - np.mean(z)
    result = np.concatenate((x,y,z),axis=0)
    return result
    

# Takes in table of data on stars and gives out euclidean coord.s in pc
# Optionally also returns the standard deviation of star distances
def euclidean_coordinates(data,errors=False):
    mean,st_dev,p,distances = mean_st_dev(data)
    coord = data[['ra','dec']].values.T
    r = np.array(mean).T
    phi = np.array([coord[0]* np.pi/180])
    theta = np.array([coord[1]* np.pi/180])
    result = change_to_cartesian(r,phi,theta)
    if errors==False:
        return result
    if errors==True:
        return result,st_dev


# Generate a 3d plot of given cartesian coordinates - input must be an array with three rows
def three_d_plot(coord,animate=False,aspect=None,errors=int(0)):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation
    import matplotlib.cm as cm
    
    print(type(errors))

    fig = plt.figure(figsize=(6,6))
    ax = Axes3D(fig)
    if aspect=='equal':
        max_range = np.array([coord[0].max()-coord[0].min(), coord[1].max()-coord[1].min(), coord[2].max()-coord[2].min()]).max()
        Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(coord[0].max()+coord[0].min())
        Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(coord[1].max()+coord[1].min())
        Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(coord[2].max()+coord[2].min())
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')
    
    
    def update_view(i): 
        if type(errors)!='int':
            ax.scatter(coord[0],coord[1],coord[2],c=errors.ravel(),cmap=cm.plasma_r)
        else:
            ax.scatter(coord[0],coord[1],coord[2])
        ax.view_init(elev=10., azim=i)
        return ax
    
    if animate==True:
        ani = FuncAnimation(fig, update_view,frames=360)
        
    else:
        if type(errors)!='int':
            ax.scatter(coord[0],coord[1],coord[2],c=errors.ravel(),cmap=cm.plasma_r)
        else:
            ax.scatter(coord[0],coord[1],coord[2])
    ax.set_xlabel('x (pc)')
    ax.set_ylabel('y (pc)')
    ax.set_zlabel('z (pc)')

    plt.show()


# Return likelihood of measured parallax given rho 
def rho_likelihood(rho,pi_prime,sigma_pi):    
    p=np.exp(-(pi_prime-1./rho)**2/(2*sigma_pi**2))/(rho**2*sigma_pi*np.sqrt(2*np.pi))    
    return p


# Finds mean and st_dev of distance for each row (star) parallax. Calculates posterior pdf
def posterior(likelihood,rho,pi_prime,sigma_pi,prior=None,plot=False):
    # set posterior values
    p=likelihood(rho,pi_prime,sigma_pi)
    (a,b)=p.shape
    # apply prior
    if prior is not None:
        p*=prior(rho)        
    # normalise p
    for i in range(a):
        p[i]/=np.trapz(p[i],rho[i])
    
    mean=(rho*p).sum(axis=1)/p.sum(axis=1)
    mean=mean.reshape(a,1)
    st_dev=np.sqrt((p*(rho-mean)**2).sum(axis=1)/p.sum(axis=1))
    st_dev=st_dev.reshape(a,1)

    return mean,st_dev,p


# Given dataset and target distances in parsec, returns mean & st_dev of the distances calculated from parallax
def mean_st_dev(data,distances=np.linspace(100.,900.,1000)):
    (a,b)=data.shape
    (c,)=distances.shape
    rho=np.zeros((a,c))
    for i in range(a):
        rho[i]=distances
    pi_prime=np.array(data['parallax']*10**-3)
    pi_prime=pi_prime.reshape((a,1))
    sigma_pi=np.array(data['parallax_error']*10**-3)
    sigma_pi=sigma_pi.reshape((a,1))
    res_1,res_2,res_3 = posterior(rho_likelihood,rho,pi_prime,sigma_pi)
    return res_1,res_2,res_3,distances


# Generates a histogram of the standard deviations in distance given data with parallaxes
def cluster_distances_st_dev(data_input):
    mean,st_dev,p,distances = mean_st_dev(data_input)    
    fig_1 = plt.figure()
    plot_loghist(st_dev, 30)
    plt.grid(axis='y', alpha=0.75)
    plt.title('Errors in Star Positions')
    plt.xlabel('Standard Deviation of Distance (pc)')
    plt.ylabel('Frequency')
    
    stacked = p.sum(axis=0)
    stacked/=np.trapz(stacked,distances)
    stack_mean=(distances*stacked).sum()/stacked.sum()
    stack_st_dev=np.sqrt((stacked*(distances-stack_mean)**2).sum()/stacked.sum())
    print('stack mean=', stack_mean, ' stack st_dev=', stack_st_dev)

    fig_2 = plt.figure(figsize=(6,6))
    plt.xlabel('Distance (pc)')
    plt.ylabel('Probability Distribution Function')
    plt.title('Stacked pdfs of Star Positions')
    plt.plot(distances,stacked)
    plt.show()




stars= tidy_counterparts('IC348',distances_hist=True)
'''
cluster_distances_st_dev(stars[stars.good_match==True])
'''
coord,st_dev= euclidean_coordinates(stars[stars.good_match==True],errors=True)
three_d_plot(coord,errors=st_dev)

