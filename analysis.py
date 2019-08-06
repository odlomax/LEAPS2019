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
from mpl_toolkits.mplot3d import Axes3D




# Takes in a cluster object and plots the whitened coordinates
def whitened_coord(cluster,animate=False,aspect='equal'):
    coord = np.concatenate(([cluster.r[:,0]],[cluster.r[:,1]],[cluster.r[:,2]]))
    print(len(cluster.r[:,0]))
    sm.three_d_plot(coord,aspect=aspect,animate=animate)    


# Randomly samples coordinates from positional pdfs using a dataframe of stars and turns it into a cluster object
def sample_coord(stars,seed=None,random=False,trim=False):
    if seed!=None:
        random.seed(seed)
    coord= sm.euclidean_coordinates(stars,move_origin=False,random_sample=random).T
    if trim==True:
        coord=trim_data(coord)

    cluster=star_cluster(coord)
    #whitened_coord(cluster,animate=False,aspect='no')

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
# Seed not implimented
# Plots points of monte carlo'd realisations in H-sigma space, points with estimator error, and convolved realisation & estimator errors
# 'random=True' to sample on the position pdfs as well as distance (distance is always randomly sampled)
# 'trim=True' to take only the 98% of stars closest to the centre of the cluster
def monte_carlo_analysis(stars,num=1,seed=None,plots=True,random=False,trim=False):
    data_3d3d_2=pd.read_csv("clusterfrac/data_3d3d_2.dat")
    estimator_3d3d_2=param_estimator(data_3d3d_2)
    samples=[]
    for i in range(num):
        sample=sample_coord(stars,seed=seed,random=random,trim=trim)          
        cluster_table=sample.make_table_row()
        estimator_3d3d_2.estimate_params(cluster_table)
        samples.append(cluster_table)
        
    result = pd.concat(samples)
    
    covariance = np.cov(result['H_est'],result['sigma_est'])
    correlation = np.corrcoef(result['H_est'],result['sigma_est'])
    mean_H = np.mean(result['H_est'])
    mean_sigma = np.mean(result['sigma_est'])
    H_std = np.std(result['H_est'])
    sigma_std=np.std(result['sigma_est'])

    if plots==True:        
        from matplotlib.patches import Ellipse
        
        np.savetxt('analysis_random_positions.txt',result)
        
        # Make plot of the H-sigma estimates
        
        fig,ax1=plt.subplots(figsize=(6,6))
        ax1.scatter(result['H_est'],result['sigma_est'],s=2)
        ax1.scatter(mean_H,mean_sigma,s=7,c='r')
        ax1.set_xlim(0,1.1)
        ax1.set_ylim(0.5,5.0)
        ax1.set_xticks(np.linspace(0.1,0.9,5))
        ax1.set_yticks(np.linspace(0.7,5.0,10))       
        ax1.set_xlabel(r"$H$")
        ax1.set_ylabel(r"$\sigma$")
        
        fig.tight_layout()
        
        
        
        bbox_props=dict(boxstyle="round,pad=1",fc="white",edgecolor='black')



        # ell2 gives the estimator error - don't use this here since want centred on each point and not on the mean
        
        y,z=np.linalg.eig(estimator_3d3d_2.covar)        
        ell_size2=2.*np.sqrt(y)
        ell_angle2=np.arctan2(z[1,0],z[0,0])*180./np.pi        
        ell2=Ellipse((mean_H,mean_sigma),ell_size2[0],ell_size2[1],ell_angle2,color='k')
        ell2.set_alpha(0.3)

        # ell3 gives convolved error
        
        m,n=np.linalg.eig(estimator_3d3d_2.covar+covariance)        
        ell_size3=2.*np.sqrt(m)
        ell_angle3=np.arctan2(n[1,0],n[0,0])*180./np.pi        
        ell3=Ellipse((mean_H,mean_sigma),ell_size3[0],ell_size3[1],ell_angle3,edgecolor='r', fc='None', lw=2)


        # Overlapped positional ellipses for each point
        
        fig_2,ax=plt.subplots(figsize=(6.,6.)) 
        
        ellipse_list=[]
        
        for i in range(num):
            ellipse=Ellipse((result.iloc[i]['H_est'],result.iloc[i]['sigma_est']),ell_size2[0],ell_size2[1],ell_angle2,color='k',edgecolor='k', lw=1)
            ellipse.set_alpha(0.025)
            ellipse_list.append(ellipse)
            ax.add_artist(ellipse_list[i])

        ax.scatter(mean_H,mean_sigma,s=7,c='r')
        ax.scatter(result['H_est'],result['sigma_est'],s=2)
        ax.set_xlim(0,1.1)
        ax.set_ylim(0.5,5.0)
        ax.set_xticks(np.linspace(0.1,0.9,5))
        ax.set_yticks(np.linspace(0.7,5.0,10))       
        ax.set_xlabel(r"$H$")
        ax.set_ylabel(r"$\sigma$")
        
        fig_2.tight_layout()


        # Plot convolved error on top of overlapped ellipses
        
        fig_3,ax3=plt.subplots(figsize=(6.,6.))
        
        ellipse_list2=[]
        
        for i in range(num):
            ellipse=Ellipse((result.iloc[i]['H_est'],result.iloc[i]['sigma_est']),ell_size2[0],ell_size2[1],ell_angle2,color='k')
            ellipse.set_alpha(0.025)
            ellipse_list2.append(ellipse)
            ax3.add_artist(ellipse_list2[i])

        ax3.scatter(mean_H,mean_sigma,s=7,c='r')
        ax3.scatter(result['H_est'],result['sigma_est'],s=2)
        ax3.add_artist(ell3)
        ax3.set_xlim(0,1.1)
        ax3.set_ylim(0.5,5.0)
        ax3.set_xticks(np.linspace(0.1,0.9,5))
        ax3.set_yticks(np.linspace(0.7,5.0,10))       
        ax3.set_xlabel(r"$H$")
        ax3.set_ylabel(r"$\sigma$")
        
        fig_3.tight_layout()


                
        # Plot estimator errors        
        
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
    print('Covariance is ',estimator_3d3d_2.covar+covariance)
    print('Correlation is ',correlation)
    print('mean H is',mean_H)
    print('mean sigma is',mean_sigma)
    print('H st dev is', H_std)
    print('sigma st dev is', sigma_std)
    return mean_H, mean_sigma, covariance, estimator_3d3d_2.covar


# Plot uncertainty ellipses for two clusters
def plot_errors(H1,sigma1,covar_dist_1,covar_est_1,H2,sigma2,covar_dist_2,covar_est_2):
    from matplotlib.patches import Ellipse

    fig_2,ax=plt.subplots(figsize=(6.,6.))
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
    #ax.text(H1,sigma1,'Orion A',ha="center",va="center",bbox=bbox_props)
    ax.text(H1,sigma1,'Orion A',ha="center",va="center")
    ax.text(H2,sigma2,'Taurus',ha="center",va="center")
    ax.set_xlim(0,1.1)
    ax.set_ylim(0.5,5.0)
    ax.set_xticks(np.linspace(0.1,0.9,5))
    ax.set_yticks(np.linspace(0.7,5.0,10))       
    ax.set_xlabel(r"$H$")
    ax.set_ylabel(r"$\sigma$")
    
    fig_2.tight_layout()
    
    plt.show()
    

# Plot uncertainty ellipses for three clusters
def plot_errors_three(H1,sigma1,covar_dist_1,covar_est_1,H2,sigma2,covar_dist_2,covar_est_2,H3,sigma3,covar_dist_3,covar_est_3):
    from matplotlib.patches import Ellipse

    fig_2,ax=plt.subplots(figsize=(6.,6.))

    w,v=np.linalg.eig(covar_dist_1+covar_est_1)
    ell_size=2.*np.sqrt(w)
    ell_angle=np.arctan2(v[1,0],v[0,0])*180./np.pi    
    ell=Ellipse((H1,sigma1),ell_size[0],ell_size[1],ell_angle,edgecolor='b', fc='b', lw=2)
    ell.set_alpha(0.3)
    
    
    y,z=np.linalg.eig(covar_dist_2+covar_est_2)
    ell_size2=2.*np.sqrt(y)
    ell_angle2=np.arctan2(z[1,0],z[0,0])*180./np.pi    
    ell2=Ellipse((H2,sigma2),ell_size2[0],ell_size2[1],ell_angle2,edgecolor='g', fc='g', lw=2)
    ell2.set_alpha(0.3)
    
    a,b=np.linalg.eig(covar_dist_3+covar_est_3)
    ell_size3=2.*np.sqrt(a)
    ell_angle3=np.arctan2(b[1,0],b[0,0])*180./np.pi    
    ell3=Ellipse((H3,sigma3),ell_size3[0],ell_size3[1],ell_angle3,edgecolor='r', fc='r', lw=2)
    ell3.set_alpha(0.3)

    ax.add_artist(ell3) 
    ax.add_artist(ell2) 
    ax.add_artist(ell)
    #ax.text(H1,sigma1,'Orion A',ha="center",va="center",bbox=bbox_props)
    ax.text(H2-0.5,sigma2,'yyy',ha="center",va="center",bbox=dict(facecolor='green', alpha=1.0))
    ax.text(H3,sigma3-1,'98%',ha="center",va="center",bbox=dict(facecolor='red', alpha=1.0))
    ax.text(H1+0.5,sigma1,'Full',ha="center",va="center",bbox=dict(facecolor='blue', alpha=1.0))


    ax.set_xlim(0,1.1)
    ax.set_ylim(0.5,5.0)
    ax.set_xticks(np.linspace(0.1,0.9,5))
    ax.set_yticks(np.linspace(0.7,5.0,10))       
    ax.set_xlabel(r"$H$")
    ax.set_ylabel(r"$\sigma$")
    
    fig_2.tight_layout()
    
    plt.show()


# Synthesise a cluster with given number of stars, H, and sigma
def synthesise(N,H,sigma):
    from maths.fields.gaussian_random_field import scalar_grf
    from maths.random.probability_density_function import pdf
    H=H
    sigma=sigma
    n_star=N
    n_dim=3 
    
    grid_size=[200]*n_dim
    r_seed=None
    np.random.seed(seed=r_seed)
    beta=n_dim+2*H
    
    # genrate field (this is complex and periodic, but we just want the real bit)
    fBm_field=scalar_grf(grid_size,beta)
    # normalise field
    fBm_field.normalise(sigma=sigma,exponentiate=True)
    # set periodic centre of mass to centre of grid
    fBm_field.com_shift()
    # use real component of fBm field as pdf
    real_field=fBm_field.signal.real
    # generate probability density function object
    fBm_pdf=pdf(real_field)
    # randomly sample cluster of stars from pdf
    fBm_cluster=fBm_pdf.random(n_star)
    cluster=star_cluster(fBm_cluster)
    fBm_cluster = np.concatenate(([cluster.r[:,0]],[cluster.r[:,1]],[cluster.r[:,2]])).T
    sm.three_d_plot(fBm_cluster.T,aspect='equal',animate=False)
    return cluster




#synthesise(500,0.5,3.0)

'''

stars_1= pd.read_csv('rhoOph.csv')

stars_new= stars_1[stars_1.doh == "YYY"]
H2,sigma2, covar2, estimator_covar2 = monte_carlo_analysis(stars_new,num=200,random=True,plots=False,trim=False)

H1,sigma1, covar1, estimator_covar1 = monte_carlo_analysis(stars_1,num=200,random=True,plots=False,trim=True)

H3,sigma3, covar3, estimator_covar3 = monte_carlo_analysis(stars_1,num=200,random=True,plots=True,trim=False)

plot_errors_three(H1,sigma1,covar1, estimator_covar1,H2,sigma2,covar2, estimator_covar2,H3,sigma3, covar3, estimator_covar3)
'''

'''

coord= sm.euclidean_coordinates(stars,move_origin=False,random_sample=True)
sm.three_d_plot(coord,aspect='equal',animate=False)

sample=sample_coord(stars,random=True,trim=False)
coord=whitened_coord(sample,animate=False)

stars= pd.read_csv('rhoOph.csv')
stars_trimmed=stars[stars.doh == "YYY"]
monte_carlo_analysis(stars,num=200,random=True,plots=True,trim=True)


stars_1= pd.read_csv('rhoOph.csv')
stars_trimmed=stars_1[stars_1.doh == "YYY"]
H3,sigma3, covar3, estimator_covar3 = monte_carlo_analysis(stars_trimmed,num=200,random=True,plots=False,trim=True)

stars_new= pd.read_csv('taurus_tidy.csv')
H2,sigma2, covar2, estimator_covar2 = monte_carlo_analysis(stars_new,num=200,random=False,plots=False,trim=True)

stars= pd.read_csv('Orion_Gaia_Data.csv')
H1,sigma1, covar1, estimator_covar1 = monte_carlo_analysis(stars,num=200,random=True,plots=True,trim=False)

plot_errors(H1,sigma1,covar1, estimator_covar1,H2,sigma2,covar2, estimator_covar2)
plot_errors_three(H1,sigma1,covar1, estimator_covar1,H2,sigma2,covar2, estimator_covar2,H3,sigma3, covar3, estimator_covar3)
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

