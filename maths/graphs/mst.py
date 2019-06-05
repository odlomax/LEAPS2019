import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from maths.graphs.edges import graph_edge_positions, graph_edge_lengths
from maths.array.moments import nth_moment

class euclidean_mst:
    
    """
    
    Class to build euclidean minimum spanning tree (and other related graphs)
    
    Data
    ----
    
    del_tri: Delaunay triangulation object
        Qhull Delaunay triangulation object
        
    r[:,:]: float
        array of point positions [n_points:n_dim]
    
    
    tri: sparse matrix
        Delaunay triangulation sparse matrix
    
    mst: sparse matrix
        minimum spanning tree sparse matrix
        
    com: sparse matrix
        complete graph
    
    """
    
    def __init__(self,r,w=None,make_com=True):
        
        """
        Subroutine: build Euclidean minimum spanning tree from points
        
        Arguments
        ---------
        r[:,:]: float
            array of point positions [n_points:n_dim]
        
        w[:]: float
            array of weights for each point [n_points]
            
        make_com: boolean
            also generate complete graph (may be very large)
        
        """
        
        # generate weights, if missing
        if w is None:
            w=np.ones(r.shape[0])
            
        self.w=w
        
        # Calculate Delaunay triangulation
        print("Generating Delaunay triangulation.")
        self.r=r
        self.del_tri=Delaunay(self.r)
        
        # Convert Delaunay triangulation to matrix graph format
        print("Converting triangulation to sparse matrix format.")
        self.tri=lil_matrix((r.shape[0],r.shape[0]),dtype=r.dtype)
        indices=self.del_tri.vertex_neighbor_vertices[0]
        indptr=self.del_tri.vertex_neighbor_vertices[1]
        
        # loop over all points
        for i in range(self.r.shape[0]):
            # loop over all neighbours of each point
            for j in indptr[indices[i]:indices[i+1]]:
                # only populate upper portion of del_tri
                if j>i:
                    # calculate edge weight of graph
                    l=np.linalg.norm(self.r[i,:]-self.r[j,:])
                    self.tri[i,j]=l
        
        # convert to csr format
        self.tri=self.tri.tocsr()
        
        # Calculate minimum spanning tree
        print("Generating minimum spanning tree.")
        self.mst=minimum_spanning_tree(self.tri)
        
        # generate mst characteristic edge length
        self.mst_char_len=(self.volume()/self.r.shape[0])**(1./self.r.shape[1])
        
        if make_com:
            # calculate zeros
            distances=squareform(pdist(r))
            self.com=lil_matrix(distances)
            # set half matrix to zero
            for i in range(r.shape[0]-1):
                self.com[i+1:,i]=0.
            self.com=self.com.tocsr()
            # calculate characteristic lenght scale
            self.com_char_len=self.mst_char_len*self.r.shape[0]**(1./self.r.shape[1])
            
        print("All graphs generated.")
        
    def volume(self):
        
        """
        Function: get volume/area of convex hull
        
        Result
        ------
        
        v: float
            volume/area of convex hull
        
        """
        
        n_dim=self.r.shape[1]
        v=0.
        
        # loop over all simplices
        for i in range(self.del_tri.simplices.shape[0]):
            
            # make vertex matrix
            vert_matrix=np.zeros((n_dim,n_dim))
            for j in range(n_dim):
                vert_matrix[:,j]=self.del_tri.points[self.del_tri.simplices[i,:-1],j]-self.del_tri.points[self.del_tri.simplices[i,-1],j]
                
            # add volume contribution
            v+=np.abs(np.linalg.det(vert_matrix))
            
        return v/np.math.factorial(n_dim)
            
        
    def tri_edge_positions(self):
        
        """
        Function: get edge line segments of Delaunay triangulation
        
        Result
        ------
        
        start[:,:],end[:,:]: float
            arrays of start and end points
            
        """
        
        return graph_edge_positions(self.tri,self.r)
        
    def mst_edge_positions(self):
        
        """
        Function: get edge line segments of minimum spanning tree
        
        Result
        ------
        
        start[:,:],end[:,:]: float
            arrays of start and end points
            
        """
        
        return graph_edge_positions(self.mst,self.r)
    
    def com_edge_positions(self):
        
        """
        Function: get edge line segments of complete graph
        
        Result
        ------
        
        start[:,:],end[:,:]: float
            arrays of start and end points
            
        """
        
        return graph_edge_positions(self.com,self.r)
    
    def tri_edge_lengths(self):
        
        """
        Function: get edge lengths of Delaunay triangulation
        
        Result
        ------
        
        lengths[:]: float
            array of edge lengths
            
        """
        
        return graph_edge_lengths(self.tri)
    
    def mst_edge_lengths(self):
        
        """
        Function: get edge lengths of minimum spanning tree
        
        Result
        ------
        
        lengths[:]: float
            array of edge lengths
            
        """
        
        return graph_edge_lengths(self.mst)
    
    def com_edge_lengths(self):
        
        """
        Function: get edge lengths of complete graph
        
        Result
        ------
        
        lengths[:]: float
            array of edge lengths
            
        """
        
        return graph_edge_lengths(self.com)
    
    def tri_moment(self,n):
        
        """
        Function: get nth moment of Delaunay triangulation edge lengths
        
        Arguments
        ---------
        
        n: int
            order of moment
        
        Result
        ------
        
        lengths[:]: float
            array of edge lengths
            
        """
        
        return nth_moment(graph_edge_lengths(self.tri),n)
    
    def mst_moment(self,n):
        
        """
        Function: get nth moment of minimum spanning tree edge lengths
        
        Arguments
        ---------
        
        n: int
            order of moment
        
        Result
        ------
        
        lengths[:]: float
            array of edge lengths
            
        """
        
        return nth_moment(graph_edge_lengths(self.mst),n)
    
    def com_moment(self,n):
        
        """
        Function: get nth moment of complete graph edge lengths
        
        Arguments
        ---------
        
        n: int
            order of moment
        
        Result
        ------
        
        lengths[:]: float
            array of edge lengths
            
        """
        
        return nth_moment(graph_edge_lengths(self.com),n)
    
    def mst_pot_e(self,w=None,inv_edges=False):
        
        """
        Function: calculate dimensionless potential energy from minimum spanning tree
        
        Arguments
        ---------
        
        w[:]: float
            weights
            
        inv_edges: boolean
            invert edge lengths?
        
        Result
        ------
        
        pot_e: float
            dimensionless potential energy of ensemble
        
        """
        
        # assign weights
        if w is None:
            w=self.w
        else:
            w=w
        
        # get non-zero indices
        i,j=self.mst.nonzero()
        
        # get list of distances
        if inv_edges:
            r=1./np.array(self.mst[i,j])[0,:]
        else:
            r=np.array(self.mst[i,j])[0,:]
        
        # get list of weight products
        w_prod=w[i]*w[j]
        
        # calc pot_e
        pot_e=(w_prod/r).sum()

        # normalise (stage one)
        if inv_edges:
            pot_e*=-1./(self.mst_char_len*w.sum()**2)
        else:
            pot_e*=-self.mst_char_len/w.sum()**2
        
        # normalise (stage two)
        pot_e*=self.r.shape[0]*(self.r.shape[0]-1)/(2.*r.shape[0])
        
        return pot_e
    
    def tri_pot_e(self,w=None,inv_edges=False):
        
        """
        Function: calculate dimensionless potential energy from delaunay triangulation
        
        Arguments
        ---------
        
        w[:]: float
            weights
            
        inv_edges: boolean
            invert edge lengths?
        
        Result
        ------
        
        pot_e: float
            dimensionless potential energy of ensemble
        
        """
        
        # assign weights
        if w is None:
            w=self.w
        else:
            w=w
        
        # get non-zero indices
        i,j=self.tri.nonzero()
        
        # get list of distances
        if inv_edges:
            r=1./np.array(self.tri[i,j])[0,:]
        else:
            r=np.array(self.tri[i,j])[0,:]
        
        # get list of weight products
        w_prod=w[i]*w[j]
        
        # calc pot_e
        pot_e=(w_prod/r).sum()

        # normalise (stage one)
        if inv_edges:
            pot_e*=-1./(self.mst_char_len*w.sum()**2)
        else:
            pot_e*=-self.mst_char_len/w.sum()**2
        
        # normalise (stage two)
        pot_e*=self.r.shape[0]*(self.r.shape[0]-1)/(2.*r.shape[0])
        
        return pot_e

    def com_pot_e(self,w=None,inv_edges=False):
        
        """
        Function: calculate dimensionless potential energy from complete graph
        
        Arguments
        ---------
        
        w[:]: float
            weights
            
        inv_edges: boolean
            invert edge lengths?
        
        Result
        ------
        
        pot_e: float
            dimensionless potential energy of ensemble
        
        """
        
        # assign weights
        if w is None:
            w=self.w
        else:
            w=w
        
        # get non-zero indices
        i,j=self.com.nonzero()
        
        # get list of distances
        if inv_edges:
            r=1./np.array(self.com[i,j])[0,:]
        else:
            r=np.array(self.com[i,j])[0,:]
        
        # get list of weight products
        w_prod=w[i]*w[j]
        
        # calc pot_e
        pot_e=(w_prod/r).sum()

        # normalise
        if inv_edges:
            pot_e*=-1./(self.com_char_len*w.sum()**2)
        else:
            pot_e*=-self.com_char_len/w.sum()**2
        
        return pot_e
        
        
        