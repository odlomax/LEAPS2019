import numpy as np
from bisect import bisect_left
from maths.array.integrate import cum_dist_func

class pdf:
    
    """
    
    Class to buld probability density function
    
    Data
    ----
    
    f_x[...]: float
        ndarray of pdf values
        
    axes[:][:]: float
        axis coordinates
        
    f_x_cum[:][...]: float
        cumulative distribution lookup tables of f_x along various axes
    
    """
    
    def __init__(self,f_x):
        
        """
        Subroutine: initialises pdf class
        
        Arguments
        ---------
        f_x[...]: float
            ndarray of pdf values
        
        """

        self.f_x=f_x
        
        # make list of axes values, and indicies values
        self.axes=[]
        for i in self.f_x.shape:
            self.axes.append(np.linspace(0.,1.,i))
        
        
        
        # make list of cumulative arrays
        self.f_x_cum=[None]*len(self.f_x.shape)
        
        
        # calculate cumulative distribution F(x,y,z). Uppercase denotes fixed coordinate
        if len(self.f_x.shape)==3:
            
            # calc F(X,Y,z)
            self.f_x_cum[2]=np.zeros(self.f_x.shape)
            for i in range(self.f_x.shape[0]):
                for j in range(self.f_x.shape[1]):
                    self.f_x_cum[2][i,j,:]=cum_dist_func(self.axes[2],self.f_x[i,j,:])
            
            # calc F(X,y)
            self.f_x_cum[1]=np.zeros(self.f_x.shape[:2])
            for i in range(self.f_x.shape[0]):
                self.f_x_cum[1][i,:]=cum_dist_func(self.axes[1],self.f_x_cum[2][i,:,-1])
                
            # calc F(x)
            self.f_x_cum[0]=cum_dist_func(self.axes[0],self.f_x_cum[1][:,-1])
            
            # associate correct function
            self.stretch_value=self.stretch_value_3d
            
        elif len(self.f_x.shape)==2:
             
            # calc F(X,y)
            self.f_x_cum[1]=np.zeros(self.f_x.shape)
            for i in range(self.f_x.shape[0]):
                self.f_x_cum[1][i,:]=cum_dist_func(self.axes[1],self.f_x[i,:])
                
            # calc F(x)    
            self.f_x_cum[0]=cum_dist_func(self.axes[0],self.f_x_cum[1][:,-1])
            
            # associate correct function
            self.stretch_value=self.stretch_value_2d
            
        elif len(self.f_x.shape)==1:
            
            # calc F(x)
            self.f_x_cum[0]=cum_dist_func(self.axes[0],self.f_x)
            
            # associate correct function
            self.stretch_value=self.stretch_value_1d
            
        else:
            print("Cannot make pdf with rank",len(self.f_x.shape))
                    


    def stretch_value_1d(self,x):
        
        """
        Function: stretches x to new value based on pdf
        
        Arguments
        ---------
        
        x: float
            input x value. must be between zero and one.
            
        Result
        ------
        
        x_new: float
            stretched x value
            
        """
        
        # get x_new value from cumulative distribution F(x)
        x_new=np.interp(x*self.f_x_cum[0][-1],self.f_x_cum[0],self.axes[0])
        
        return x_new
    
    def stretch_value_2d(self,x):
        
        """
        Function: stretches x to new value based on pdf
        
        Arguments
        ---------
        
        x[2]: float
            input x value. must be between zero and one.
            
        Result
        ------
        
        x_new[2]: float
            stretched x value
            
        """
        
        x_new=[0.,0.]
        
        # get first element of x_new from cumulative distribution F(x)
        x_new[0]=self.stretch_value_1d(x[0])
        
        # interpolate cumulative distribution f(X,y)

        # get index
        i_0=min(max(bisect_left(self.axes[0],x_new[0])-1,0),self.axes[0].size-2)
        i_1=i_0+1
        
        # Get interpolation weights
        i_weight=min(max((self.axes[0][i_1]-x_new[0])/(self.axes[0][i_1]-self.axes[0][i_0]),0.),1.)
        
        # get interpolated cumulative distribution
        f_xy=i_weight*self.f_x_cum[1][i_0,:]+(1.-i_weight)*self.f_x_cum[1][i_1,:]
        
        # get second element
        x_new[1]=np.interp(x[1]*f_xy[-1],f_xy,self.axes[1])
        
        return x_new
    
    def stretch_value_3d(self,x):
        
        """
        Function: stretches x to new value based on pdf
        
        Arguments
        ---------
        
        x[3]: float
            input x value. must be between zero and one.
            
        Result
        ------
        
        x_new[3]: float
            stretched x value
            
        """
        
        x_new=[0.,0.,0.]
        
        # get first and second element of x_new from cumulative distribution F(x,y)
        x_new[:2]=self.stretch_value_2d(x[:2])
        
        # interpolate cumulative distribution f(X,y)

        # get indices
        i_0=min(max(bisect_left(self.axes[0],x_new[0])-1,0),self.axes[0].size-2)
        i_1=i_0+1
        
        j_0=min(max(bisect_left(self.axes[1],x_new[1])-1,0),self.axes[1].size-2)
        j_1=j_0+1
        
        # Get interpolation weights
        i_weight=min(max((self.axes[0][i_1]-x_new[0])/(self.axes[0][i_1]-self.axes[0][i_0]),0.),1.)
        j_weight=min(max((self.axes[1][j_1]-x_new[1])/(self.axes[1][j_1]-self.axes[1][j_0]),0.),1.)
        
        # get interpolated cumulative distribution
        f_xy_0=i_weight*self.f_x_cum[2][i_0,j_0,:]+(1.-i_weight)*self.f_x_cum[2][i_1,j_0,:]
        f_xy_1=i_weight*self.f_x_cum[2][i_0,j_1,:]+(1.-i_weight)*self.f_x_cum[2][i_1,j_1,:]
        f_xy=j_weight*f_xy_0+(1.-j_weight)*f_xy_1
        
        # get third element
        x_new[2]=np.interp(x[2]*f_xy[-1],f_xy,self.axes[2])
        
        return x_new
    
    def stretch_lattice(self,r):
            
        """
        Function: returns array of random variates from pdf
        
        Arguments
        ---------
        r[:,:]: float
            array of lattice points [n_points:n_dim]. values assumed to be between 0 and 1
            
        Result
        ------
        
        r_stretch[r.shape]: float
            stretched lattice points
            
        """
        
        # set r_stretch array
        r_stretch=np.zeros(r.shape)
        
        for i in range(r.shape[0]):
            r_stretch[i,:]=self.stretch_value(r[i,:])
            
        return r_stretch
    
    def random(self,n=1):
            
        """
        Function: returns array of random variates from pdf
        
        Arguments
        ---------
        n: int
            number of random variates
            
        Result
        ------
        
        r_num[n,len(self.f_x.shape)]: float
            array of random variates
            
        """
        
        # set r_num array
        r_num=np.random.random((n,len(self.f_x.shape)))
        
        return self.stretch_lattice(r_num)
        
        