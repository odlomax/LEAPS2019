import numpy as np

def graph_edge_positions(graph,r):
        
    """
    Function: Get line segments of graph with node positions r
    
    Arguments
    ---------
    
    graph[n_points:n_points]: sparse matrix
        adjacency matrix of graph
        
    
    r[:,:]: float
        array of point positions [n_points:n_dim]
    
    Result
    ------
    start[:,:],end[:,:]: float
        arrays of start and end points
    """
    
    i,j=graph.nonzero()
    start=r[i,:]
    end=r[j,:]
    
    return start,end

def graph_edge_lengths(graph):
        
    """
    Function: Get line edge lengths of graph
    
    Arguments
    ---------
    
    graph[n_points:n_points]: sparse matrix
        adjacency matrix of graph
    
    Result
    ------
    lengths[:]: float
        array edge lengths
    """
    
    i,j=graph.nonzero()
    lengths=np.array(graph[i,j])[0,:]
    
    return lengths