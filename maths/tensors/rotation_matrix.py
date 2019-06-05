import numpy as np

class r_matrix_2d:
    
    """"
    3D Rotation matrix
    
    Data
    ----
    
    value[3,3]: float
        Rotation matrix
    
    """
    
    def __init__(self,theta):
        
        """
        Subroutine: Calculate 2d rotation matrix from angle
        
        Arguments
        ---------
        
        theta: float
            angle of rotation
        """
        
        # allocate values matrix
        self.value=np.zeros((2,2))
        
        # calc diagonal components
        self.value[0,0]=np.cos(theta)
        self.value[1,1]=self.value[0,0]
        
        # calc off-diagonal components
        self.value[0,1]=-np.sin(theta)
        self.value[1,0]=-self.value[0,1]
        
    def rotate_vec(self,v):
        
        """
        Function: Rotate a 2D vector using roation matrix
        
        Arguments
        ---------
        
        v[2]: float
            input vector
        
        Result
        ------
        
        w[2]: float
            rotated vector
            
        """
        
        return np.matmul(self.value,v)
        
class r_matrix_3d:
    
    """"
    3D Rotation matrix
    
    Data
    ----
    
    value[3,3]: float
        Rotation matrix
    
    """
    
    def __init__(self,u,theta):
        
        """
        Subroutine: Calculate 3D rotation matrix from unit vector and angle
        
        Arguments
        ---------
        
        u[3]: float
            axis of rotation unit vector
        theta: float
            angle of rotation
        """
        
        # set sines and cosines
        cos_theta=np.cos(theta)
        sin_theta=np.sin(theta)
        
        # allocate values matrix
        self.value=np.zeros((3,3))
        
        # calc diagonal components
        self.value[0,0]=cos_theta+u[0]**2*(1.-cos_theta)
        self.value[1,1]=cos_theta+u[1]**2*(1.-cos_theta)
        self.value[2,2]=cos_theta+u[2]**2*(1.-cos_theta)
        
        # calc off-diagonal components
        self.value[0,1]=u[0]*u[1]*(1.-cos_theta)-u[2]*sin_theta
        self.value[1,2]=u[1]*u[2]*(1.-cos_theta)-u[0]*sin_theta
        self.value[0,2]=u[0]*u[2]*(1.-cos_theta)+u[1]*sin_theta
        self.value[1,0]=u[1]*u[0]*(1.-cos_theta)+u[2]*sin_theta
        self.value[2,1]=u[2]*u[1]*(1.-cos_theta)+u[0]*sin_theta
        self.value[2,0]=u[2]*u[0]*(1.-cos_theta)-u[1]*sin_theta
        
    def rotate_vec(self,v):
        
        """
        Function: Rotate a 3D vector using roation matrix
        
        Arguments
        ---------
        
        v[3]: float
            input vector
        
        Result
        ------
        
        w[3]: float
            rotated vector
            
        """
        
        return np.matmul(self.value,v)
        