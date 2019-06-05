import numpy as np

class i_tensor:
    
    """"
    3D Moment of inetria tensor
    
    Data
    ----
    
    value[3,3]: float
        Moment of inertia tensor
        
    lambda_1,lambda_2,lambda_3: float
        Principle moments of inertia
        
    omega_1[3],omega_2[3],omega_3[3]: float
        principle axes
    
    """
    
    def __init__(self,x,y,z,m):
        
        """
        Subroutine: Calculate moment of inertia from points
        
        Arguments
        ---------
        
        x[:]: float
            column of x values
        y[:]: float
            column of y values
        z[:]: float
            column of z values
        m[:]: float
            column of masses
        """
        
        # allocate values matrix
        self.value=np.zeros((3,3))
        
        # calc diagonal components
        self.value[0,0]=np.sum((y**2+z**2)*m)
        self.value[1,1]=np.sum((x**2+z**2)*m)
        self.value[2,2]=np.sum((x**2+y**2)*m)
        
        # calc off-diagonal components
        self.value[0,1]=-np.sum(x*y*m)
        self.value[1,2]=-np.sum(y*z*m)
        self.value[0,2]=-np.sum(x*z*m)  
        self.value[1,0]=self.value[0,1]
        self.value[2,1]=self.value[1,2]
        self.value[2,0]=self.value[0,2]
            
        # get eigen values and vectors
        w,v=np.linalg.eig(self.value)
        
        # sort eigen values
        i=np.argsort(w)
        w=w[i]
        v=v[:,i]
        
        self.lambda_1,self.lambda_2,self.lambda_3=w
        self.omega_1=v[:,0]
        self.omega_2=v[:,1]
        self.omega_3=v[:,2]
        