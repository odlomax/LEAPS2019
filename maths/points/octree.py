import numpy as np

class octree:
    
    """
    
    octree class
    
    """
    
    def __init__(self,r,n_bits,w=None,delta_r=None):
        
        """
        
        Subroutine initiates octree class
        
        Arguments
        ---------
        
        r[:,:]: float [n_point,n_dim]
            positions
            
        n_bits: int
            number of bits per dimension
            
        w[:]: float [n_point]
            weights (mass)            
            
        delta_r[:,:]: float [2,n_dim]
            min-max position values
         
        """
        
        # set weights
        if w is None:
            w=np.ones(r.shape[0],dtype=r.dtype)/r.shape[0]
        
        n_dim=r.shape[1]
        
        # set delta_r, if missing
        if delta_r is None:
            delta_r=np.zeros((2,n_dim),dtype=np.float)
            delta_r[0,:]=r.min(axis=0)
            delta_r[1,:]=r.max(axis=0)
        
        # convert r to value between 0 and 1<<n_bits - 1
        ijk=(((r-np.expand_dims(delta_r[0,:],0))
             /np.expand_dims(delta_r[1,:]-delta_r[0,:],0))*(1<<n_bits)).astype(np.int)
        
        # make sure ijk is within bounds
        ijk[ijk>(1<<n_bits)-1]=(1<<n_bits)-1
        ijk[ijk<0]=0
        
        # make cell_id
        self.n_bits=n_bits
        self.n_dim=n_dim
        cell_id=np.zeros(r.shape[0])
        cell_id=self.encode_cell_id(ijk)
                
                
        # sort arrays and associate with class
        i=np.argsort(cell_id)
        self.cell_id=cell_id[i]
        self.r=r[i,:]
        self.w=w[i]
        self.delta_r=delta_r

        return

    def encode_cell_id(self,ijk):
        
        """
        
        Function: converts cell id to positional (x,y,z) min/max ids
        
        Arguments
        ---------
        
        ijk[:,:]: int [n_point,n_dim]
            indices of x, y and z positions
        
        Result
        ------
        
        cell_id: int
            full cell id
        
        """
        
        # initialise cell_id
        cell_id=np.zeros(ijk.shape[0],dtype=np.int)
        
        # loop over levels and dimensions
        for i in range(self.n_bits):
            for j in range(self.n_dim):
                
                # add value to ijk
                cell_id+=((ijk[:,j]>>i)&1)<<(i*self.n_dim+j)
                
                
        return cell_id
    
    def decode_cell_id(self,cell_id):
        
        """
        
        Function: converts cell id to positional (x,y,z) min/max ids
        
        Arguments
        ---------
        
        cell_id: int
            full cell id
        
        Result
        ------
        
        ijk[:]: int [n_dim]
            indices of x, y and z positions
        
        """
        
        # initialise ijk
        ijk=np.zeros((cell_id.shape[0],self.n_dim),dtype=np.int)
        
        # loop over levels and dimensions
        for i in range(self.n_bits):
            for j in range(self.n_dim):
                
                # add value to ijk
                ijk[:,j]+=((cell_id>>(i*self.n_dim+j))&1)<<i
                
                
        return ijk
    
    def cell_id_range(self,parent_id=0,level=0):
        
        """
        
        Function get range of all cell_id values within parent cell
        
        Arguments
        ---------
        
        parent_id: int
            parent id of cells
            
        level: int
            level of parent id
            
        Result
        ------
        
        cell_id_min: int
            cell_id lower bound
            
        cell_id_max: int
            cell_id upper bound
        
        """
        
        cell_id_min=parent_id<<((self.n_bits-level)*self.n_dim)
        cell_id_max=cell_id_min+(1<<((self.n_bits-level)*self.n_dim))

        return cell_id_min,cell_id_max              
    
    def get_points(self,parent_id=0,level=0,j_min=None,j_max=None):
        
        """
        
        Function: returns upper and lower indices of cell id array for given cell
        
        Arguments
        ---------
        
        parent_id: int
            parent id of cells
            
        level: int
            level of cell
            
        j_min: int
            lower bound of cell_id array to search
            
        j_max: int
            upper bound of cell_id array to search
        
        Result
        ------
        
        i_min: int
            lower index of cell id
        
        i_max: int
            upper index of cell id
        
        """
        
        # set helper limits, if not supplied
        if j_min is None:
            j_min=0
            
        if j_max is None:
            j_max=self.cell_id.shape[0]
        
        # set upper and lower cell ids
        cell_id_min,cell_id_max=self.cell_id_range(parent_id,level)
        
        i_min=j_min+np.searchsorted(self.cell_id[j_min:j_max],cell_id_min)
        i_max=j_min+np.searchsorted(self.cell_id[j_min:j_max],cell_id_max)
        
        return i_min,i_max
    
    def get_children(self,parent_id):
        
        """
        
        Function: returns a set of children cell ids
        
        Arguments
        ---------
        
        parent_id: int
            parent id of cells
            
        Result
        ------
        
        child_id[:]: int
            child cell ids
        
        """
        
        # make child_id
        child_id=np.arange(1<<self.n_dim,dtype=np.int)+(parent_id<<self.n_dim)
        
        return child_id
    
    def density_grid(self,n_min=1,n_max=1):
        
        """
        
        Function: raster octree to a density grid
        
        Arguments
        ---------
        
        n_max: int
            maximum number of points per cell
            
        n_min: int
            minimum number of points per cell
        
        Result
        ------
        
        grid[...]: float [[1<<n_bits]*n_dim]
            density grid
        
        """
        
        # initialise grid
        grid=np.zeros([1<<self.n_bits]*self.n_dim,dtype=self.w.dtype)
        
        # recursively raster density
        parent_id=0
        level=0
        j_min=0
        j_max=self.cell_id.shape[0]
        self.raster_tree(grid,parent_id,level,n_min,n_max,j_min,j_max)
        
        return grid
    
    def raster_tree(self,grid,parent_id,level,n_min,n_max,j_min,j_max):
        
        """
        
        Subroutine: recusively raster point density to grid
        
        Arguments
        ---------
        
        grid[...]: float [[1<<n_bits]*n_dim]
            density grid
            
        parent_id: int
            id of parent cell
            
        level: int
            current bit level of recursion
            
        n_min: int
            minimum number of points per cell
            
        n_max: int
            maximum number of points per cell
            
        j_min: int
            lower bound of cell_id array to search
            
        j_max: int
            upper bound of cell_id array to search
            
        """
        
        
        if level==self.n_bits or j_max-j_min<=n_max:
            
            # raster this cell
            self.raster_cell(grid,parent_id,level,j_min,j_max)
     
        else:
            
            # get list of children
            child_list=self.get_children(parent_id)
            
            # get point indices
            j_min_new,j_max_new=self.get_points(child_list,level+1)
            
            if any(j_max_new-j_min_new<n_min):
                
                # raster this cell
                self.raster_cell(grid,parent_id,level,j_min,j_max)
                
            else:
                
                # recursively call subroutine
                for i in range(child_list.shape[0]):
                    
                     # recursively raster cells
                    self.raster_tree(grid,child_list[i],level+1,n_min,n_max,j_min_new[i],j_max_new[i])
        
        return
    
    def raster_cell(self,grid,parent_id,level,j_min,j_max):
        
        """
        
        Subroutine: rasters a single cell of grid
        
        Arguments
        ---------
        
        grid[...]: float [[1<<n_bits]*n_dim]
            density grid
            
        parent_id: int
            id of parent cell
            
        level: int
            current bit level of recursion
            
        j_min: int
            lower bound of cell_id array to search
            
        j_max: int
            upper bound of cell_id array to search
            
        """
        
        # raster this cell
        cell_id_0,cell_id_1=self.cell_id_range(parent_id,level)            
        
        ijk_0=self.decode_cell_id(np.expand_dims(np.array(cell_id_0),0)).flatten()
        ijk_1=self.decode_cell_id(np.expand_dims(np.array(cell_id_1-1),0)).flatten()
        
        # make a slice list
        slice_list=[]
        for i in range(self.n_dim):
            slice_list.append(slice(ijk_0[i],ijk_1[i]+1))
        

        grid[tuple(slice_list)]=((1<<(level*self.n_dim))*self.w[j_min:j_max].sum()
                          /(self.delta_r[1,:]-self.delta_r[0,:]).prod())
        
        return
        