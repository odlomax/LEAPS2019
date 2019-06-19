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


# Finds the Gaia data from 2015.5 for sources in the region of a cluster
def gaia_data_current(stars,show=False):
    from astroquery.gaia import Gaia
    import warnings
    warnings.filterwarnings('ignore')
    
    RA,Dec,radius = circle_around_stars(stars)
    job = Gaia.launch_job_async("SELECT gaia_source.source_id,gaia_source.ra,gaia_source.ra_error,gaia_source.dec,gaia_source.dec_error,gaia_source.parallax,gaia_source.parallax_error,gaia_source.pmra,gaia_source.pmra_error,gaia_source.pmdec,gaia_source.pmdec_error,gaia_source.astrometric_n_good_obs_al,gaia_source.astrometric_gof_al,gaia_source.astrometric_chi2_al,gaia_source.visibility_periods_used,gaia_source.phot_g_mean_flux_over_error,gaia_source.phot_g_mean_mag,gaia_source.phot_bp_mean_flux_over_error,gaia_source.phot_rp_mean_flux_over_error,gaia_source.phot_bp_rp_excess_factor,gaia_source.bp_rp,gaia_source.bp_g,gaia_source.g_rp,gaia_source.radial_velocity,gaia_source.radial_velocity_error \
FROM gaiadr2.gaia_source \
WHERE CONTAINS(POINT('ICRS',gaiadr2.gaia_source.ra,gaiadr2.gaia_source.dec),CIRCLE('ICRS',%s,%s,%s))=1;"%(RA,Dec,radius) \
, dump_to_file=True)  
    r = job.get_results()
    if show==True:
        print(r)
    return r


# Finds the Gaia data from 2015.5 and projected 2000 data for sources in the region of a cluster
def gaia_data_projected(stars):
    import astropy.units as u
    from astropy.coordinates.sky_coordinate import SkyCoord
    from astropy.units import Quantity
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


# Given a cluster name finds member stars and Gaia data from that region
# Potentially save these datasets for analysis
def find_gaia_region(cluster_name,save_member_list = False,save_gaia_region=False):
    stars = cluster_list(cluster_name)
    gaia_stars = gaia_data_projected(stars)
    if save_gaia_region==True:
        gaia_stars.write('%s_Region_Gaia_Data.fits'%cluster_name)
    if save_member_list==True:
        stars.to_csv('%s_Member_List.csv'%cluster_name)
    return stars,gaia_stars


# For data sets in the form of arrays finds nearest neighbors
# Returns the indices of the neighbors of the first data set in the second and distances between neighbors
def indices_of_nearest_neighbors_arrays(set_1,set_2,num_neighbors=1):
    from sklearn.neighbors import NearestNeighbors

    (w,x)=set_1.shape
    (y,z)=set_2.shape   
    X = set_2
    Y = set_1
    a = w
        
    k_neighbors = num_neighbors+1
    neighbor_index = np.zeros((a,num_neighbors))
    distances = np.zeros((a,num_neighbors))
    for i in range(a):
        row = np.copy(Y[i].reshape((1,2)))
        Z = np.concatenate((row,X),axis=0)
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
    gaia_stars_coord = np.array([gaia_stars['ra_2000'],gaia_stars['dec_2000']]).T     
    neighbor_index,distances = indices_of_nearest_neighbors_arrays(stars_coord,gaia_stars_coord,num_neighbors=num_matches)    
    stars.reset_index(inplace=True,drop=True)
    
    for i in range(num_matches):
        temp = gaia_stars_df.reindex(neighbor_index[:,i])
        temp['distance %s'%(i+1)] = distances[:,i]
        temp.reset_index(inplace=True,drop=True)
        stars = pd.concat([stars,temp],axis=1)
        
    if save==True:
        stars.to_csv('%s_counterparts.csv'%cluster_name)       
    return stars
   
    
# Remove rows with matches of less than a certain angular displacement in arcseconds
def tidy_counterparts(cluster_name,max_dist=1,num=1,save=False):
    max_dist=max_dist/(60*60)
    stars=distances_from_match(cluster_name,num_matches=num)
    (a,b)=stars.shape    
    empty_parallax = stars['parallax'].isnull()

    for i in range(a):
        if (stars.at[i,'distance 1']>max_dist) or (empty_parallax[i]==True):
            stars.drop([i],inplace=True)
    # Some issue here with matching the source index??
    '''
        for j in range(a):
            if j!=i:
                if stars.at[i,'source_id']==stars.at[j,'source_id']:
                    if stars.at[i,'distance 1']<stars.at[j,'distance 1']:
                        stars.drop([j],inplace=True)
                    else:
                        stars.drop([i],inplace=True)
    '''       
    stars.reset_index(inplace=True,drop=True)    
    if save==True:
        stars.to_csv('%s_tidy_counterparts.csv'%cluster_name)
    return stars
    

print(tidy_counterparts('IC348',save=True))
    

