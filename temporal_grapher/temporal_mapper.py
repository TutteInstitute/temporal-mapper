''' modified: 2024-07-04 ~10am'''
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from utilities_ import *
from weighted_clustering import *
from tqdm import tqdm, trange
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from scipy.sparse import issparse
from sklearn.neighbors import NearestNeighbors
from datamapplot.palette_handling import palette_from_datamap

'''TemporalGraph class 
minimal usage example: 

    # load from your data file:
    data : (n_dim, N_data) array-like
    time : (N_data,) array-like
    semantic_dist : (N_data,) array-like
    # choose an sklearn clusterer:
    clusterer = HDBSCAN()

    # init and build the graph:
    TG = TemporalGraph(
        time,
        data,
        clusterer,
        N_checkpoints = 10,
    )
    
    TG.build()
    myGraph = TG.G
'''

class TemporalGraph():
    """
    Generate and store a temporal graph - a 1D-mapper-style representation of temporal data.

    Attributes
    ----------
    G : networkx.classes.Digraph(Graph)
        The temporal graph itself.
    density : ndarray
        The f-density \rho for each data point.
        
    Methods
    -------
    build(ydata=None):
        Perform all operations necessary to construct the graph.
    """
    
    def __init__(
        self, time, data, clusterer, 
        N_checkpoints = None,
        resolution = 10,
        overlap = 0.5,
        clusters = None,
        checkpoints = None,
        show_outliers = False,
        slice_method = 'time',
        rate_sensitivity = 1,
        kernel = gaussian,
        kernel_params = None,
        verbose = False,
    ):
        """
        Parameters 
        ----------
        time : ndarray
            time array (1 dim)
        data : ndarray
            data array (n dim)
        clusterer : sklearn clusterer
            the clusterer to use for the slice-wise clustering, must accept sample_weights
        N_checkpoints : int
            number of time-points at which to cluster
        checkpoints : arraylike
            array of time-points at which to cluster
        overlap : float
            A float in (0,1) which specifies the `g` parameter (see README)
        resolution: float
            Determines the distance around each point which we use as a neighbourhood for 
            determining the f-rate. If you get a warning about isolated points, you should
            increase this parameter. If you plot the density and it is not very smooth
            you can increase this parameter.
        show_outliers : bool
            If true, include unclustered points in the graph
        slice_method : str
            One of 'time' or 'data'. If time, generates N_checkpoints evenly spaced in time. If data,
            generates N_checkpoints such that there are equal amounts of data between the points. 
        rate_sensitivity : float
            A positive float, or -1. The rate parameter is raised to this parameter, so higher numbers
            means that the algorithm is more sensitive to changes in rate. If rate_sensivity == -1, 
            then the rate parameter is taken log2. 
        kernel : function
            A function with signature f(t0, t, density, binwidth, epsilon=0.01, params=None).
            Two options are included in weighted_clustering.py, `weighted_clustering.square` and 
            `weighted_clustering.gaussian`.
        kernel_parameters : tuple or None,
            Passed to `kernel` as params kwarg.
        verbose : bool
            Does what you expect.
        
        """
        if np.size(time) != np.shape(data)[0]:
            raise AttributeError("Number of datapoints",
                                 np.shape(data)[0],
                                 "does not equal number of timestamps",
                                 np.size(time)
                                )
        self.time = np.array(time)
        self.N_data = np.size(time)
        if len(data.shape) == 1:
            data=data.reshape(-1,1)
        self.n_components = data.shape[1]
        if issparse(data):
            self.scaler = StandardScaler(copy=False, with_mean=False)
        else:
            self.scaler = StandardScaler(copy=False)
        self.data = self.scaler.fit_transform(data)
        self.checkpoints = checkpoints
        if slice_method in ['time','data']:
            self.slice_method = slice_method
        else:
            raise AttributeError("Accepted slice_method is 'time' or 'data'.")
        if checkpoints is not None:
            self.N_checkpoints = np.size(checkpoints)
            if not (self.N_checkpoints == N_checkpoints):
                raise AttributeError("Given checkpoints and N_checkpoints, len(checkpoints) must equal N_checkpoints.")
        else:
            if N_checkpoints is not None:
                self.N_checkpoints = N_checkpoints
            else:
                raise AttributeError("You must pass one of checkpoints or N_checkpoints.")

        self.clusterer = clusterer
        self.clusters = clusters
        self.g = overlap
        self.density = None
        self.rate = None
        self.sensitivity = rate_sensitivity
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.G = nx.DiGraph()
        self.adj_matrix = None
        self.pos = None
        self.verbose=verbose
        self.disable = not verbose # tqdm
        self.show_outliers = False
        self.resolution = resolution
        self.k = resolution
        self.distance = None

    def _compute_epsilon_balls(self):
        if self.verbose:
            print("Computing epsilon balls...")
        self.data_width = np.mean(
            [np.amax(self.data[:,k])-np.amin(self.data[:,k])
             for k in range(self.data.shape[1])]     
        )
        self.epsilon = (self.data_width*self.resolution)/self.N_data
        # compute 10x the epsilon ball b.c. we need the rest for smoothing later
        self.distance, self.dist_indices = epsilon_balls(self.data, 10*self.epsilon) 

    def _compute_checkpoints(self):
        if self.slice_method == 'data':
            idx = np.linspace(0, self.N_data, self.N_checkpoints+2)[1:-1]
            idx = np.array([int(x) for x in idx])
            checkpoints = self.time[idx]
        if self.slice_method == 'time':
            checkpoints = np.linspace(np.amin(self.time), np.amax(self.time), self.N_checkpoints+2)[1:-1]
        self.checkpoints = checkpoints
        return checkpoints

    def _compute_knn(self):
        if self.verbose:
            print("Computing k nearest neighbours...")
        std_time = np.copy(self.time)
        std_time = self.scaler.fit_transform(std_time.reshape(-1,1))
        datatime = np.concatenate((self.data,std_time),axis=1)
        nbrs = NearestNeighbors(n_neighbors=self.k).fit(datatime)
        self.distance, self.dist_indices = nbrs.kneighbors(datatime)
        return self.distance, self.dist_indices

    def _compute_density(self):
        if self.checkpoints is None:
            self._compute_checkpoints()
        if self.distance is None:
            self._compute_knn()
        if self.verbose:
            print("Computing spatial density...")
        self.data_width = np.mean(
            [np.amax(self.data[:,k])-np.amin(self.data[:,k])
             for k in range(self.data.shape[1])]     
        )
        self.epsilon = (self.data_width*self.resolution)/self.N_data
        radius = self.distance[:,-1]
        density = self.k*radius**(-self.n_components-1)
        # apply the smoothing window:
        d_window = self.data_width/10
        # todo: figure out this strange bug with copying self.distance
        smoothed_densities = np.array(
            [np.average(density[idx], weights=window(self.distance[k], d_window))
             for k, idx in enumerate(self.dist_indices)]
        )
        density = std_sigmoid(smoothed_densities)
        if self.sensitivity == -1:
            self.density = 1/(1-np.log2(density))
        else:
            self.density = density**self.sensitivity
        #self.density[iso_idx] = np.amin(density[~iso_idx])
        return self.density

    def _cluster(self):
        if self.density is None:
            self._compute_density()
        if self.verbose:
            print("Clustering at each time slice...")
        clusters, weights = weighted_clusters(
            self.data,
            self.time,
            self.checkpoints,
            self.density,
            self.clusterer,
            self.kernel,
            self.kernel_params,
        )
        self.clusters = clusters
        self.weights = weights
        return clusters

    def add_vertices(self, y_data=1):
        # Add the clusters from each time slice as vertices.
        node_counter = 0
        slices = []
        for i in trange(self.N_checkpoints, disable=self.disable,
                        desc="Converting clusters to vertices"):
            slice_idx = (self.clusters[i]!=-2).nonzero()[0]
            slices.append(slice_idx)
            cluster = self.clusters[i][slice_idx]
            for l, val in enumerate(np.unique(cluster)):
                if (val == -1) and (not self.show_outliers):
                    continue
                
                # Construct the basic attributes of the node:
                slice_no = i
                cluster_no = val
                node_label = str(i)+":"+str(val)
                
                # Add a node with the attributes
                self.G.add_node(
                    node_label,
                    slice_no = slice_no,
                    cluster_no = cluster_no,
                    node_number = node_counter,
                )
                node_counter += 1
                
        self.slices = slices
        if self.verbose:
            print("%d vertices added." % (np.size(self.G.nodes())) )

        return self

    def build_adj_matrix(self):
        verts = self.G.nodes()
        n_verts = len(verts)
        verts = np.array(verts)
        adj_mat = np.zeros((n_verts, n_verts))
        slices = self.slices
        time_centers = np.zeros(len(slices))
        bin_width = np.zeros(len(slices))
        for k, slice_ in enumerate(slices):
            time_centers[k] = np.median(self.time[slice_])
            bin_width[k] = np.max(self.time[slice_])-np.min(self.time[slice_])
        
        for i in trange(self.N_checkpoints-1, disable=self.disable,
                       desc='Adding edges'): 
            clust1 = self.clusters[i]
            clust2 = self.clusters[i+1]
            
            #int_idx = np.intersect1d(slices[i], slices[i+1])
            for j in slices[i]:
                c1 = clust1[j]
                c2 = clust2[j]
                if ((c1 == -1) or (c2==-1)) and (not self.show_outliers):
                    #outliers
                    continue
                if ((c1 == -2) or (c2==-2)):
                    #not in both slices.
                    continue
                c1_str = str(i)+":"+str(int(c1))
                c2_str = str(i+1)+":"+str(int(c2))

                # Get the matrix indices corresponding to the two clusters
                l = self.G.nodes()[c1_str]["node_number"]
                k = self.G.nodes()[c2_str]["node_number"]  
                
                adj_mat[l][k] += self.kernel(
                    time_centers[i],
                    self.time[j],
                    self.density[j],
                    bin_width[i],
                    params=self.kernel_params
                )
    
        self.adj_matrix = adj_mat
        return self
    
    def add_edges(self):
        if type(self.adj_matrix) != np.ndarray:
            self.build_adj_matrix()
        # Use the adj. matrix to add the weighted edges
        i=j=0
        verts = np.array(self.G.nodes())
        for row in self.adj_matrix:
            i=0
            for val in row: 
                if val == 0:
                    i+=1
                    continue
                self.G.add_edge(verts[j], verts[i], weight=val)
                i+=1
            j+=1
        return self

    def build(self):
        # Build the temporal graph
        if self.clusters is None:
            self._cluster()
        self.add_vertices()
        self.build_adj_matrix()
        self.add_edges()
        self.populate_edge_attrs()
        self.populate_node_attrs()
        return self

    def populate_edge_attrs(self):
        # Add src_weight and dst_weight properties to every edge.
        for u,v,d in self.G.edges(data = True):
            u_outdeg = self.G.out_degree(u, weight='weight')
            v_indeg = self.G.in_degree(v, weight='weight')

            percentage_outweight = d['weight']/u_outdeg
            percentage_outweight = round(percentage_outweight, 2) # otherwise the graph labels look horrible
            self.G[u][v]['src_weight'] = percentage_outweight 

            percentage_inweight = d['weight']/v_indeg
            percentage_inweight = round(percentage_inweight, 2) # as above
            self.G[u][v]['dst_weight'] = percentage_inweight
        
    def populate_node_attrs(self, labels=None):
        """ Add node attributes (dictionaries) to the vertices of the graph.
            Mainly required for visualization purposes.
        """
        if self.verbose:
            print("Populating node centroids, colours, sizes...")
        pos = False 
        if self.n_components == 2:
            pos = True
            
        # Add cluster positions in 2D and sizes for visualization.
        centroids = {}
        size_list = {}
        t_attrs = nx.get_node_attributes(self.G, 'slice_no')
        cl_attrs = nx.get_node_attributes(self.G, 'cluster_no')   
        for node in self.G.nodes():
            t_idx = t_attrs[node]
            cl_idx = cl_attrs[node]
            size = np.size(self.get_vertex_data(node))
            size_list[node] = size
            if pos:
                pt_idx = self.get_vertex_data(node)
                node_xpos = np.mean(self.data[pt_idx,0])
                node_ypos = np.mean(self.data[pt_idx,1])
                centroids[node] = (node_xpos, node_ypos)
        if pos:
            nx.set_node_attributes(self.G, centroids, 'centroid')
        nx.set_node_attributes(self.G, size_list, 'count')

        # Compute cluster colours that correspond to datamapplot colours.
        if self.n_components != 2:
            print("Warning: Cluster colours are only implemented for 2d data.")
        else:
            if self.verbose:
                print("Computing cluster colours...")
                clr_dict = {}
                cluster_positions = np.zeros((len(self.G.nodes()),2))
                for k,pt in enumerate(centroids.values()):
                    cluster_positions[k] = pt
                colours = np.array(palette_from_datamap(self.data, cluster_positions))
                for k,node in enumerate(centroids.keys()):
                    c = colours[k]
                    pc = (t_attrs[node]+1)/self.N_checkpoints
                    clr = hex_desaturate(c, pc)
                    clr_dict[node] = clr
            
        nx.set_node_attributes(self.G, clr_dict, "colour")
        return 0
    
    def get_vertex_data(self, node):
        t_idx = self.G.nodes()[node]['slice_no']
        cl_idx = self.G.nodes()[node]['cluster_no']
        vals_in_cl = (self.clusters[t_idx] == cl_idx).nonzero()
        return vals_in_cl[0]

    
    def get_dir_subvertices(self, v, threshold=0.1, backwards=True):
        vertices = [v]
        # Given a vertex, propagate forwards and backwards in time to obtain that vertices' subgraph.
        if not backwards:
            _edges = self.G.out_edges(v, data=True)
        else: 
            _edges = self.G.in_edges(v, data=True)
        for a,b,d in _edges:
            if d['weight'] >= threshold:
                if not backwards:
                    vertices.append(b)
                    vertices += self.get_dir_subvertices(b, threshold, backwards)
                else:
                    vertices.append(a)
                    vertices += self.get_dir_subvertices(a, threshold, backwards)
                
        return vertices
    
    def vertex_subgraph(self, v, threshold=0.1):
        vertices = self.get_dir_subvertices(v, threshold) + self.get_dir_subvertices(v, threshold, backwards=False)
        return np.unique(vertices)

    def get_subgraph_data(self, vertices):
        vals = [self.get_vertex_data(v) for v in vertices]
        return np.concatenate(vals, axis=1)
