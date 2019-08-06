import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sort_memberships as sm
#import analysis as an


# read in and rename cols of taurus membership data
def taurus_read_in():
    df = pd.read_csv("Taurus.csv")
    stars = df[['RAJ2000','DEJ2000','plx','e_plx']]
    stars.rename(index=str, columns={"RAJ2000": "ra", "DEJ2000": "dec","plx":"parallax","e_plx":"parallax_error"},inplace=True)
    return stars



# Remove rows with matches of less than a certain angular displacement in arcseconds
def taurus_tidy_counterparts(save=False):
    stars=taurus_read_in()
    (a,b)=stars.shape 
    stars.dropna(inplace=True)
    stars.reset_index(drop=True,inplace=True)
    if save==True:
        stars.to_csv('taurus_tidy.csv')        
    return stars

# remove stars with no parallax or 10.0>parallax<1.0 (this is not necessary, use data 'trimming' to take 98% closest in analysis)
def tidy(stars,save=False):
    (a,b)=stars.shape 
    for i in range(a):
        (a,b)=stars.shape 
        if i>=a:
            return 0
        elif ((stars.iloc[i].loc['parallax']<1.0) | (stars.iloc[i].loc['parallax']>10.0)):
            stars_new.drop(i,inplace=True)

    stars.reset_index(drop=True,inplace=True)
    if save==True:
        stars.to_csv('taurus_tidy.csv')        


# remove stars z component>50 (this is not necessary, use data 'trimming' to take 98% closest in analysis)
# this was used to investivate the effect of outliers
def tidy_coord(coord):
    new_coord=[]
    (a,b)=coord.shape
    for i in range(a):
        if coord[i,2]<50:
            new_coord.append(coord[i])
    new_coord=np.array(new_coord)
    return(new_coord)        
    



#stars_new= pd.read_csv('taurus_tidy.csv')

#stars_new=taurus_tidy_counterparts(save=True)
#tidy(stars_new,save=True)

#coord= sm.euclidean_coordinates(stars_new,random_sample=False,errors=False)
        
