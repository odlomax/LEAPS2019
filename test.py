import sort_memberships as sm
import orion_sort_memberships as osm
import analysis as an
import taurus_sort_memberships as tsm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Reads in Taurus data and removes members with no parallax data / very big or small parallaxes
'''
stars_new=tsm.taurus_tidy_counterparts(save=True)
tsm.tidy(stars_new,save=True)
print(stars_new)
'''


# Read in CSV of Orion data and plot in 3D 
# 'move_origin =True' in 'euclidean_coordinates' moves the origin of the graph to the mean of the points
'''
stars= pd.read_csv('Orion_Gaia_Data.csv')

coord,st_dev = sm.euclidean_coordinates(stars,type_errors='parallax',errors=True,move_origin=False)
sm.three_d_plot(coord,aspect='equal',animate=False)
'''


# Generates a histogram of the standard deviations in distance given data with parallaxes
'''
stars= pd.read_csv('Orion_Gaia_Data.csv')

coord,st_dev = sm.euclidean_coordinates(stars,type_errors='parallax',errors=True,move_origin=False)
sm.three_d_plot(coord,aspect='equal',animate=False)
sm.cluster_distances_st_dev(stars)
'''


# Finds IC348 data and plots it in 3D
# Takes the Gaia stars in the region which are 'good_match'es
# Generates a histogram of the standard deviations in distance given data with parallaxes
'''
stars= sm.tidy_counterparts('IC348',distances_hist=True,plot_matches=True)
coord,st_dev= sm.euclidean_coordinates(stars[stars.good_match==True],type_errors='parallax',errors=True)
sm.three_d_plot(coord,aspect='equal',animate=False)
sm.cluster_distances_st_dev(stars[stars.good_match==True])
'''


# Create a synthetic cluster with given number of stars, H, sigma
'''
cluster = synthesise(500,0.5,3.0)
'''


# Sample positions and plot 3D whitened realisation of a read in CSV cluster
'''
stars= pd.read_csv('Orion_Gaia_Data.csv')

sample=sample_coord(stars,random=True,trim=False)
coord=whitened_coord(sample,animate=False)
'''


# Analyse a cluster using Monte Carlo realisations
'''
stars= pd.read_csv('rhoOph.csv')
stars_trimmed=stars[stars.doh == "YYY"]
an.monte_carlo_analysis(stars,num=10,random=True,plots=True,trim=True)
'''


# Compare positions of different clusters in H-sigma space
'''
stars_1= pd.read_csv('rhoOph.csv')
stars_trimmed=stars_1[stars_1.doh == "YYY"]
H3,sigma3, covar3, estimator_covar3 = monte_carlo_analysis(stars_trimmed,num=10,random=True,plots=False,trim=True)

stars_new= pd.read_csv('taurus_tidy.csv')
H2,sigma2, covar2, estimator_covar2 = monte_carlo_analysis(stars_new,num=10,random=False,plots=False,trim=True)

stars= pd.read_csv('Orion_Gaia_Data.csv')
H1,sigma1, covar1, estimator_covar1 = monte_carlo_analysis(stars,num=10,random=True,plots=True,trim=False)

plot_errors(H1,sigma1,covar1, estimator_covar1,H2,sigma2,covar2, estimator_covar2)
plot_errors_three(H1,sigma1,covar1, estimator_covar1,H2,sigma2,covar2, estimator_covar2,H3,sigma3, covar3, estimator_covar3,clusters=True)
'''


#Compare the different ways of sampling data in rho Oph
'''
stars_1= pd.read_csv('rhoOph.csv')

stars_new= stars_1[stars_1.doh == "YYY"]
H2,sigma2, covar2, estimator_covar2 = an.monte_carlo_analysis(stars_new,num=10,random=True,plots=False,trim=False)

H1,sigma1, covar1, estimator_covar1 = an.monte_carlo_analysis(stars_1,num=10,random=True,plots=False,trim=True)

H3,sigma3, covar3, estimator_covar3 = an.monte_carlo_analysis(stars_1,num=10,random=True,plots=False,trim=False)

an.plot_errors_three(H1,sigma1,covar1, estimator_covar1,H2,sigma2,covar2, estimator_covar2,H3,sigma3, covar3, estimator_covar3,clusters=False)
'''



