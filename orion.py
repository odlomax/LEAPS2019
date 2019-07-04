import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sort_memberships as sm

# Find the dimensions of a rectangle containing all stars described in the input database
def orion_find_dimensions(stars):
    max_RA = stars['RAICRS'].max()
    min_RA = stars['RAICRS'].min()
    max_Dec = stars['DEICRS'].max()
    min_Dec = stars['DEICRS'].min()
    return min_RA, max_RA, min_Dec, max_Dec


# Returns the centre coordinates and radius of a circle containing the rectangle which contains all stars in the input database.
def orion_circle_around_stars(stars):
    min_RA, max_RA, min_Dec, max_Dec = orion_find_dimensions(stars)
    RA, Dec = np.mean([min_RA,max_RA]),np.mean([min_Dec, max_Dec])
    radius = np.linalg.norm([(max_RA - min_RA)/2, (max_Dec - min_Dec)/2])
    radius=radius
    return RA,Dec,radius


# Finds the Gaia data from 2015.5
def orion_gaia_data(save=False):
    from astroquery.gaia import Gaia
    import warnings
    warnings.filterwarnings('ignore')
    
    stars = pd.read_csv('Orion_A.csv')
    ids = np.array(stars['Source'])
    RA,Dec,radius = orion_circle_around_stars(stars)
    
    job = Gaia.launch_job_async("SELECT gaia_source.source_id,gaia_source.ra,gaia_source.ra_error,gaia_source.dec,gaia_source.dec_error,gaia_source.parallax,gaia_source.parallax_error,gaia_source.pmra,gaia_source.pmra_error,gaia_source.pmdec,gaia_source.pmdec_error,gaia_source.astrometric_n_good_obs_al,gaia_source.astrometric_gof_al,gaia_source.astrometric_chi2_al,gaia_source.visibility_periods_used,gaia_source.phot_g_mean_flux_over_error,gaia_source.phot_g_mean_mag,gaia_source.phot_bp_mean_flux_over_error,gaia_source.phot_rp_mean_flux_over_error,gaia_source.phot_bp_rp_excess_factor,gaia_source.bp_rp,gaia_source.bp_g,gaia_source.g_rp,gaia_source.radial_velocity,gaia_source.radial_velocity_error  \
        FROM gaiadr2.gaia_source \
        WHERE CONTAINS(POINT('ICRS',gaiadr2.gaia_source.ra,gaiadr2.gaia_source.dec),CIRCLE('ICRS',%s,%s,%s))=1  \
        ;"%(RA,Dec,radius) \
        , dump_to_file=True)
    r = job.get_results()
    rdf = pd.DataFrame(np.array(r))
    gaia_ids = np.array(rdf['source_id'])
    indices = np.intersect1d(ids,gaia_ids,return_indices=True)[2]
    result = rdf.reindex(indices)
    result.reset_index(inplace=True,drop=True)
    sm.del_files()
    if save==True:
        result.to_csv('Orion_Gaia_Data.csv')
    return result



stars= orion_gaia_data()
coord,st_dev = sm.euclidean_coordinates(stars,errors=True)
sm.three_d_plot(coord,aspect='equal',errors=st_dev,animate=True)
