import sort_memberships as sm
import orion_sort_memberships as osm

from clusterfrac.cluster import star_cluster
from clusterfrac.estimator import param_estimator
from clusterfrac.model import cluster_model
from maths.points.ra_dec import ra_dec_project
from maths.points.fuse_points import fuse_close_companions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# Takes in a cluster object and plots the whitened coordinates
def whitened_coord(cluster,animate=False):
    from mpl_toolkits.mplot3d import Axes3D
    coord = np.concatenate(([cluster.r[:,0]],[cluster.r[:,1]],[cluster.r[:,2]]))
    sm.three_d_plot(coord,aspect='equal',animate=animate)
# CHANGE COORD TO NONE WHEN PLOTTING!!
    


#whitened_coord(cluster)

# Randomly samples coordinates from a dataframe of stars and turns it into a cluster object
def sample_coord(stars,seed=None,random=False,trim=False):
    if seed!=None:
        random.seed(seed)
    coord= sm.euclidean_coordinates(stars,move_origin=False,random_sample=random).T
    if trim==True:
        coord=trim_data(coord)
    cluster=star_cluster(coord)
    return cluster


# Remove outliers so that 98% of the data remains
def trim_data(coord):
    stars=coord
    # order positions by distance to centre
    stars_mean=coord.mean(axis=0).reshape((1,3))
    order=np.argsort(((stars-stars_mean)**2).sum(axis=1))
    stars=stars[order,:]
    
    # get N_centile percent of inner most stars
    N_centile=98
    i_max=int(stars.shape[0]*N_centile/100)
    stars=stars[:i_max,:]
    return stars



# Takes in a dataframe of stars and finds H and sigma for a given number of randomly sampled coord
def monte_carlo_analysis(stars,num=1,seed=None,plots=True,random=False,trim=False):
    data_3d3d_2=pd.read_csv("clusterfrac/data_3d3d_2.dat")
    estimator_3d3d_2=param_estimator(data_3d3d_2)
    samples=[]
    for i in range(num):
        sample=sample_coord(stars,seed=seed,random=random,trim=trim)           
        cluster_table=sample.make_table_row()
        estimator_3d3d_2.estimate_params(cluster_table)
        samples.append(cluster_table)
        
        #coord=whitened_coord(sample,animate=False)

    result = pd.concat(samples)
    
    covariance = np.cov(result['H_est'],result['sigma_est'])
    correlation = np.corrcoef(result['H_est'],result['sigma_est'])
    mean_H = np.mean(result['H_est'])
    mean_sigma = np.mean(result['sigma_est'])

    if plots==True:
        from matplotlib.patches import Ellipse
        
        np.savetxt('analysis_random_positions.txt',result)
        
        fig=plt.figure(figsize=(6,6))
        plt.scatter(result['H_est'],result['sigma_est'],s=4)
        plt.axes().set_aspect('equal')
        plt.title('Estimated Structure Parameters')
        plt.xlabel('Estimated H')
        plt.ylabel('Estimated sigma')
        
        bbox_props=dict(boxstyle="round,pad=1",fc="white",edgecolor='black')
        fig_2,ax=plt.subplots(figsize=(6.,6.))
        # JUST ADD COVARIANCE MATRICES TO CONVOLVE ERRORS
        #w,v=np.linalg.eig(estimator_3d3d_2.covar)
        w,v=np.linalg.eig(estimator_3d3d_2.covar)
        
        ell_size=2.*np.sqrt(w)
        ell_angle=np.arctan2(v[1,0],v[0,0])*180./np.pi
        
        ell=Ellipse((mean_H,mean_sigma),ell_size[0],ell_size[1],ell_angle,color='b')
        ell.set_alpha(0.3)

        # ell gives the estimator error
        
        y,z=np.linalg.eig(covariance)
        
        ell_size2=2.*np.sqrt(y)
        ell_angle2=np.arctan2(z[1,0],z[0,0])*180./np.pi
        
        ell2=Ellipse((mean_H,mean_sigma),ell_size2[0],ell_size2[1],ell_angle2,color='k')
        ell2.set_alpha(0.3)
        ell4=Ellipse((mean_H,mean_sigma),ell_size2[0],ell_size2[1],ell_angle2,color='k')
        ell4.set_alpha(0.3)
        # ell2 and ell4 give the positional error
        
        m,n=np.linalg.eig(estimator_3d3d_2.covar+covariance)
        
        ell_size3=2.*np.sqrt(m)
        ell_angle3=np.arctan2(n[1,0],n[0,0])*180./np.pi
        
        ell3=Ellipse((mean_H,mean_sigma),ell_size3[0],ell_size3[1],ell_angle3,color='g')
        ell3.set_alpha(0.3)
        # ell3 gives convolved error
        
        ax.add_artist(ell2)
        
        ax.text(mean_H-0.15,mean_sigma+0.5,'Orion',ha="center",va="center",bbox=bbox_props)
        ax.scatter(result['H_est'],result['sigma_est'],s=1)
        ax.set_xlim(0,1.1)
        ax.set_ylim(0.5,5.0)
        ax.set_xticks(np.linspace(0.1,0.9,5))
        ax.set_yticks(np.linspace(0.7,5.0,10))       
        ax.set_xlabel(r"$H$")
        ax.set_ylabel(r"$\sigma$")
        
        fig_2.tight_layout()
        # fig_2 is of the positional error
        
        fig_3,ax3=plt.subplots(figsize=(6.,6.))
        ax3.add_artist(ell)
        ax3.add_artist(ell4)
        
        ax3.text(mean_H-0.15,mean_sigma+0.5,'Orion',ha="center",va="center",bbox=bbox_props)
        #ax.scatter(mean_H,mean_sigma)

        ax3.set_xlim(0,1.1)
        ax3.set_ylim(0.5,5.0)
        ax3.set_xticks(np.linspace(0.1,0.9,5))
        ax3.set_yticks(np.linspace(0.7,5.0,10))       
        ax3.set_xlabel(r"$H$")
        ax3.set_ylabel(r"$\sigma$")
        
        fig_3.tight_layout()
        # fig_3 shows both positional and estimator errors
        
        fig_4,ax4=plt.subplots(figsize=(6.,6.))
        ax4.add_artist(ell3)
        
        ax4.text(mean_H,mean_sigma,'Orion',ha="center",va="center",bbox=bbox_props)
        #ax.scatter(mean_H,mean_sigma)

        ax4.set_xlim(0,1.1)
        ax4.set_ylim(0.5,5.0)
        ax4.set_xticks(np.linspace(0.1,0.9,5))
        ax4.set_yticks(np.linspace(0.7,5.0,10))       
        ax4.set_xlabel(r"$H$")
        ax4.set_ylabel(r"$\sigma$")
        
        fig_4.tight_layout()


        
        
        
        H_true=estimator_3d3d_2.test_data["H"]
        H_est=estimator_3d3d_2.test_data["H_est"]

        plt.figure()
        plt.scatter(H_true,H_est,marker=".")
        plt.plot([0,1],[0,1],c="black")
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.xlabel("H")
        plt.ylabel("H_est")
        
        sigma_true=estimator_3d3d_2.test_data["sigma"]
        sigma_est=estimator_3d3d_2.test_data["sigma_est"]

        plt.figure()
        plt.scatter(sigma_true,sigma_est,marker=".")
        plt.plot([1.5,4.5],[1.5,4.5],c="black")
        plt.xlim(1.5,4.5)
        plt.ylim(1.5,4.5)
        plt.xlabel("sigma")
        plt.ylabel("sigma_est")
                
        plt.show()
    print('Covariance is ',covariance)
    print('Correlation is ',correlation)
    print('mean H is',mean_H)
    print('mean sigma is',mean_sigma)
    return mean_H, mean_sigma, covariance, estimator_3d3d_2.covar



def plot_errors(H1,sigma1,covar_dist_1,covar_est_1,H2,sigma2,covar_dist_2,covar_est_2):
    from matplotlib.patches import Ellipse

    bbox_props=dict(boxstyle="round,pad=1",fc="white",edgecolor='black')
    fig_2,ax=plt.subplots(figsize=(6.,6.))
    # JUST ADD COVARIANCE MATRICES TO CONVOLVE ERRORS
    #w,v=np.linalg.eig(estimator_3d3d_2.covar)
    w,v=np.linalg.eig(covar_dist_1+covar_est_1)
    ell_size=2.*np.sqrt(w)
    ell_angle=np.arctan2(v[1,0],v[0,0])*180./np.pi
    
    ell=Ellipse((H1,sigma1),ell_size[0],ell_size[1],ell_angle)
    ell.set_alpha(0.3)
    
    
    y,z=np.linalg.eig(covar_dist_2+covar_est_2)
    
    ell_size2=2.*np.sqrt(y)
    ell_angle2=np.arctan2(z[1,0],z[0,0])*180./np.pi
    
    ell2=Ellipse((H2,sigma2),ell_size2[0],ell_size2[1],ell_angle2)
    ell2.set_alpha(0.3)

    ax.add_artist(ell2)
    
    ax.add_artist(ell)
    ax.text(H1,sigma1,'Orion',ha="center",va="center",bbox=bbox_props)
    ax.text(H2,sigma2,'Taurus',ha="center",va="center",bbox=bbox_props)
    ax.set_xlim(0,1.1)
    ax.set_ylim(0.5,5.0)
    ax.set_xticks(np.linspace(0.1,0.9,5))
    ax.set_yticks(np.linspace(0.7,5.0,10))       
    ax.set_xlabel(r"$H$")
    ax.set_ylabel(r"$\sigma$")
    
    fig_2.tight_layout()
    
    plt.show()
    
    
stars_new= pd.read_csv('Orion_Gaia_Data.csv')
monte_carlo_analysis(stars_new,num=200,random=True,plots=True,trim=False)

        
'''
stars= pd.read_csv('Orion_Gaia_Data.csv')
H1,sigma1, covar1, estimator_covar1 = monte_carlo_analysis(stars,num=5,random=True,plots=False)

stars_new= pd.read_csv('taurus_tidy.csv')
H2,sigma2, covar2, estimator_covar2 = monte_carlo_analysis(stars_new,num=5,random=False,plots=False)

plot_errors(H1,sigma1,covar1, estimator_covar1,H2,sigma2,covar2, estimator_covar2)
'''
# REMEMBER TO CHANGE RANDOM SAMPLE IN SAMPLE COORD EUCLIDEAN COORD
'''
stars= pd.read_csv('taurus_tidy.csv')

print(monte_carlo_analysis(stars,num=100))




stars = np.loadtxt('Orion_coord.txt').T

cluster=star_cluster(stars)

coord=whitened_coord(cluster,animate=True)

        
    






df = pd.read_csv("Taurus.csv")
print(df.columns.values)




stars = np.loadtxt('Orion_coord.txt').T

cluster=star_cluster(stars)

def whitened_coord(cluster,animate=False):
    from mpl_toolkits.mplot3d import Axes3D
    coord = np.concatenate(([cluster.r[:,0]],[cluster.r[:,1]],[cluster.r[:,2]]))
    sm.three_d_plot(coord,aspect='equal',animate=animate)
# CHANGE COORD TO NONE WHEN PLOTTING!!

coord=whitened_coord(cluster,animate=True)

'''

