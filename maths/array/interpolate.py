from bisect import bisect_left

def grid_interp(x,xp,fp,axis):
    
        """
        Function: interolates value from nd array
        
        Arguments
        ---------
        x: float
            x interpolation value
        xp[:]: float
            1d array of x points
        fp[...]: float
            nd array of f points
        axis: int
            interpolation axis
            
        Result
        ------
        
        f[...]: float
            interpolated (n-1)d array 
        """
        # Get indicies
        j_0=min(max(bisect_left(xp,x)-1,0),xp.shape[0]-2)
        j_1=j_0+1
        
        # Get interpolation weights
        weight=min(max((xp[j_1]-x)/(xp[j_1]-xp[j_0]),0.),1.)
        
        # Return interpolated value        
        return weight*fp.take(j_0,axis)+(1.-weight)*fp.take(j_1,axis)
            
        
        