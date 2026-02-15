from copy import deepcopy
from collections.abc import Callable
from warnings import warn

import numpy as np
from numpy import typing as npt
from tqdm import trange
import networkx as nx
import matplotlib as mpl
from datamapplot.palette_handling import palette_from_datamap
from scipy.sparse import issparse

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_is_fitted

from temporalmapper.utilities import(
    std_sigmoid,
    cosine_window,
    weighted_clusters,
)
from temporalmapper.plotting import (
    time_semantic_plot,
)
from temporalmapper.layout import compute_time_semantic_positions
from temporalmapper.kernels import square
from temporalmapper.analytics import (
    compute_growth,
)

"""TemporalMapper class 
minimal usage example: 

    # load from your data file:
    data : (n_dim, N_data) array-like
    time : (N_data,) array-like
    # choose an sklearn clusterer:
    clusterer = HDBSCAN()

    # init and build the graph:
    mapper = TemporalGraph(
        time,
        data,
        clusterer,
        N_checkpoints = 10,
    )
    
    mapper.build()
    myGraph = mapper.G

    # generate a matplotlib figure
    mapper.temporal_plot()
"""


class TemporalMapper(BaseEstimator):
    """
    Generate and store a temporal graph - a 1D-mapper-style representation of temporal data.

    Attributes
    ----------
    G: networkx.classes.Digraph(Graph)
        The temporal graph itself.

    Methods
    -------
    fit():
        Run the density-based mapper algorithm to construct the temporal graph.
    get_vertex_data(str node):
        Returns the index of elements of ``data`` which are in vertex ``node``.
    get_dir_subvertices(str node, float threshold = 0.0, bool backwards=False):
        Returns the vertices that descend from ``node`` with outedge weight at least ``threshold``. 
        If ``backwards = True``, returns the ancestors instead of descendants.
    temporal_plot():
        Returns a matplotlib axis containing a temporal plot
    interactive_temporal_plot():
        Returns a Plotly figure containing an interactive temporal plot
    """
    SERIAL_VERSION = 1

    def __init__(
        self,
        time: npt.NDArray,
        data: npt.NDArray,
        clusterer: ClusterMixin,
        N_checkpoints: int=None,
        neighbours: int=50,
        overlap: float=0.5,
        inclusion_threshold: float=0.01,
        checkpoints: list[float]=None,
        show_outliers: bool=False,
        slice_method: str="time",
        rate_sensitivity: int=1,
        kernel: Callable[[float,float,float,float],float]=square,
        kernel_params: dict=None,
        verbose: bool=False,
    ):
        """
        Parameters
        ----------
        time: ndarray
            time array (1 dim)
        data: ndarray
            data array (n dim)
        clusterer: sklearn clusterer
            the clusterer to use for the slice-wise clustering, must accept sample_weights
        N_checkpoints: int
            number of time-points at which to cluster
        checkpoints: arraylike
            array of time-points at which to cluster
        overlap: float
            A float in (0,1) which specifies the ``g`` parameter (see README)
        inclusion_threshold: float
            A float in [0,1) which specifies the minimum kernel weight for a point to be included in a slice.
        neighbours: float
            The number of nearest neighbours used in the density computation.
        show_outliers: bool
            If true, include unclustered points in the graph
        slice_method: str
            One of 'time' or 'data'. If time, generates N_checkpoints evenly spaced in time. If data,
            generates N_checkpoints such that there are equal amounts of data between the points.
        rate_sensitivity: float
            A positive float, or -1. The rate parameter is raised to this parameter, so higher numbers
            means that the algorithm is more sensitive to changes in rate. If ``rate_sensivity == -1``,
            then the rate parameter is taken log2.
        kernel: function
            A function with signature ``f(t0, t, density, binwidth, epsilon=0.01, params=None)``.
            Options are included in temporalmapper.kernels, default is ``temporalmapper.kernels.square``.
        kernel_parameters: tuple or None,
            Passed to `kernel` as params kwarg.
        verbose: bool
            Does what you expect.

        """
        if np.size(time) != np.shape(data)[0]:
            raise AttributeError(
                "Number of datapoints",
                np.shape(data)[0],
                "does not equal number of timestamps",
                np.size(time),
            )
        self.time = np.array(time)
        self.n_samples = np.size(time)
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        self.n_components = data.shape[1]
        if issparse(data):
            self.scaler = StandardScaler(copy=False, with_mean=False)
        else:
            self.scaler = StandardScaler(copy=False)
        self.data = self.scaler.fit_transform(data)
        self.checkpoints = checkpoints
        if slice_method in ["time", "data"]:
            self.slice_method = slice_method
        else:
            raise AttributeError("Accepted slice_method is 'time' or 'data'.")
        if checkpoints is not None:
            self.N_checkpoints = np.size(checkpoints)
            if N_checkpoints is not None and (
                not (self.N_checkpoints == N_checkpoints)
            ):
                raise AttributeError(
                    "Given checkpoints and N_checkpoints, len(checkpoints) must equal N_checkpoints."
                )
        else:
            if N_checkpoints is not None:
                self.N_checkpoints = N_checkpoints
            else:
                raise AttributeError(
                    "You must pass one of checkpoints or N_checkpoints."
                )

        self.clusterer = clusterer
        self.clusters = None
        self.inclusion_threshold = inclusion_threshold
        self.overlap = overlap
        self.g = self.overlap
        self.density = None
        self.rate = None
        self.sensitivity = rate_sensitivity
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.G = nx.DiGraph()
        self.adj_matrix = None
        self.pos = None
        self.verbose = verbose
        self.disable = not verbose  # for tqdm
        self.show_outliers = False
        self.neighbours = neighbours 
        self.k = self.neighbours 
        self.distance = None
        self.cbeta = None

    def _compute_checkpoints(self):
        """Compute evenly spaced checkpoints at which to center the mapper slices."""
        if self.slice_method == "data":
            idx = np.linspace(0, self.n_samples, self.N_checkpoints + 2)[1:-1]
            idx = np.array([int(x) for x in idx])
            checkpoints = self.time[idx]
        if self.slice_method == "time":
            checkpoints = np.linspace(
                np.amin(self.time), np.amax(self.time), self.N_checkpoints + 2
            )[1:-1]
        self.checkpoints = checkpoints

        return checkpoints

    def _compute_knn(self):
        """Run sklearn NearestNeighbours to compute knns."""
        if self.verbose:
            print("Computing k nearest neighbours...")
        std_time = np.copy(self.time)
        std_time = self.scaler.fit_transform(std_time.reshape(-1, 1))
        datatime = np.concatenate((self.data, std_time), axis=1)
        nbrs = NearestNeighbors(n_neighbors=self.k).fit(datatime)
        self.distance, self.dist_indices = nbrs.kneighbors(datatime)
        return self.distance, self.dist_indices

    def _compute_density(self):
        """Compute the temporal density (f-rate) at each point."""
        if self.sensitivity == 0:
            if self.verbose:
                print(
                    "Temporal density sensitivity is set to 0, skipping density computation."
                )
            self.density = np.ones(self.n_samples)
            return self.density
        if self.distance is None:
            self._compute_knn()
        if self.verbose:
            print("Computing spatial density...")
        self.data_width = np.mean(
            [
                np.amax(self.data[:, k]) - np.amin(self.data[:, k])
                for k in range(self.data.shape[1])
            ]
        )
        radius = self.distance[:, -1]
        density = self.k * np.ones(self.n_samples)
        temporal_width = np.array(
            [max(self.time[idx]) - min(self.time[idx]) for idx in self.dist_indices]
        )
        density /= temporal_width

        # apply the smoothing window:
        d_window = self.data_width / 10
        smoothed_densities = np.array(
            [
                np.average(
                    density[idx], weights=cosine_window(self.distance[k], d_window)
                )
                for k, idx in enumerate(self.dist_indices)
            ]
        )
        smoothed_densities = std_sigmoid(smoothed_densities)

        if self.sensitivity == -1:
            self.density = 1 / (1 - np.log2(smoothed_densities))
        else:
            self.density = smoothed_densities**self.sensitivity
        return self.density

    def _compute_kernel_width(self):
        """Return the parameter c(beta) for the kernel width.
        Currently unused.
        """
        if self.density is None:
            self._compute_density()
        sorting_index = np.argsort(self.density)
        reverse_sort_dict = {sorting_index[i]: i for i in range(self.n_samples)}
        reverse_sort_index = np.zeros(self.n_samples, dtype=int)
        for s in sorting_index:
            reverse_sort_index[s] = reverse_sort_dict[s]
        cdf = reverse_sort_index / self.n_samples
        c_max = 2  # todo magic number
        self.cbeta = c_max * cdf + (1 - cdf)
        return self.cbeta

    def _cluster(self):
        """At each checkpoint, use the clustering algorithm to cluster the points in the
        associated bin. A convention here is that a cluster of -1 means noise, and a
        cluster of -2 means unclustered.
        """
        if self.checkpoints is None:
            self._compute_checkpoints()
        if self.density is None:
            self._compute_density()
        if self.cbeta is None:
            self._compute_kernel_width()
        if self.verbose:
            print("Clustering at each time slice...")
        clusters, weights = weighted_clusters(
            self.data,
            self.time,
            self.checkpoints,
            self.density / np.median(self.density),
            self.clusterer,
            self.kernel,
            self.g,
            self.kernel_params,
            eps=self.inclusion_threshold,
        )
        self.clusters = clusters
        self.weights = weights
        return clusters

    def add_vertices(self, y_data=1):
        """Add the clusters from each time slice as vertices in the networkx graph (self.G)."""
        node_counter = 0
        slices = []
        for i in trange(
            self.N_checkpoints,
            disable=self.disable,
            desc="Converting clusters to vertices",
        ):
            slice_idx = (self.clusters[i] != -2).nonzero()[0]
            slices.append(slice_idx)
            cluster = self.clusters[i][slice_idx]
            for l, val in enumerate(np.unique(cluster)):
                if (val == -1) and (not self.show_outliers):
                    continue

                # Construct the basic attributes of the node:
                slice_no = i
                cluster_no = val
                node_label = str(i) + ":" + str(val)

                # Add a node with the attributes
                self.G.add_node(
                    node_label,
                    slice_no=slice_no,
                    cluster_no=cluster_no,
                    node_number=node_counter,
                )
                node_counter += 1

        self.slices = slices
        if self.verbose:
            print("%d vertices added." % (np.size(self.G.nodes())))

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
            bin_width[k] = np.max(self.time[slice_]) - np.min(self.time[slice_])

        for i in trange(
            self.N_checkpoints - 1, disable=self.disable, desc="Adding edges"
        ):
            clust1 = self.clusters[i]
            clust2 = self.clusters[i + 1]

            for j in slices[i]:
                c1 = clust1[j]
                c2 = clust2[j]
                if ((c1 == -1) or (c2 == -1)) and (not self.show_outliers):
                    # outliers
                    continue
                if (c1 == -2) or (c2 == -2):
                    # not in both slices.
                    continue
                c1_str = str(i) + ":" + str(int(c1))
                c2_str = str(i + 1) + ":" + str(int(c2))

                # Get the matrix indices corresponding to the two clusters
                l = self.G.nodes()[c1_str]["node_number"]
                k = self.G.nodes()[c2_str]["node_number"]

                adj_mat[l][k] += self.kernel(
                    time_centers[i],
                    self.time[j],
                    self.density[j],
                    bin_width[i],
                    params=self.kernel_params,
                )

        self.adj_matrix = adj_mat
        return self

    def add_edges(self):
        if type(self.adj_matrix) != np.ndarray:
            self.build_adj_matrix()
        # Use the adj. matrix to add the weighted edges
        i = j = 0
        verts = np.array(self.G.nodes())
        for row in self.adj_matrix:
            i = 0
            for val in row:
                if val == 0:
                    i += 1
                    continue
                self.G.add_edge(verts[j], verts[i], weight=val)
                i += 1
            j += 1
        return self

    def build(self):
        """Run the fuzzy mapper algorithm to construct the temporal graph."""
        if self.clusters is None:
            self._cluster()
        self.add_vertices()
        self.build_adj_matrix()
        self.add_edges()
        self.populate_edge_attrs()
        self.populate_node_attrs()
        self.is_fitted_ = True
        return self

    def fit(self):
        """SKlearn naming convention."""
        return self.build()

    def populate_edge_attrs(self):
        """Add src_weight and dst_weight properties to every edge."""
        for u, v, d in self.G.edges(data=True):
            u_outdeg = self.G.out_degree(u, weight="weight")
            v_indeg = self.G.in_degree(v, weight="weight")

            percentage_outweight = d["weight"] / u_outdeg
            percentage_outweight = round(
                percentage_outweight, 2
            )  # otherwise the graph labels look horrible
            self.G[u][v]["src_weight"] = percentage_outweight

            percentage_inweight = d["weight"] / v_indeg
            percentage_inweight = round(percentage_inweight, 2)  # as above
            self.G[u][v]["dst_weight"] = percentage_inweight

    def populate_node_attrs(self, labels=None):
        """Add node attributes (dictionaries) to the vertices of the graph.
        Mainly required for visualization purposes.
        """
        if self.verbose:
            print("Populating node attributes, such as centroids, colours, sizes...")

        # Add cluster positions in 2D and sizes for visualization.
        centroids = {}
        size_list = {}
        t_attrs = nx.get_node_attributes(self.G, "slice_no")
        cl_attrs = nx.get_node_attributes(self.G, "cluster_no")
        for node in self.G.nodes():
            t_idx = t_attrs[node]
            cl_idx = cl_attrs[node]
            size = np.size(self.get_vertex_data(node))
            size_list[node] = size
            pt_idx = self.get_vertex_data(node)
            centroids[node] = [
                np.mean(self.data[pt_idx, d]) for d in range(self.n_components)
            ]
        nx.set_node_attributes(self.G, centroids, "centroid")
        nx.set_node_attributes(self.G, size_list, "count")
        nx.set_node_attributes(self.G, compute_growth(self.G), "growth")

        # Compute cluster colours that correspond to datamapplot colours.
        if self.n_components != 2:
            if self.verbose:
                print("Warning: Cluster colours are only implemented for 2d data.")
            clr_dict = {node: "#000000" for node in self.G.nodes()}
        else:
            if self.verbose:
                print("Computing cluster colours...")
            clr_dict = {}
            cluster_positions = np.zeros((len(self.G.nodes()), 2))
            for k, pt in enumerate(centroids.values()):
                cluster_positions[k] = pt
            colours = np.array(palette_from_datamap(self.data, cluster_positions))
            clr_dict = {node: colours[k] for k, node in enumerate(centroids.keys())}

        nx.set_node_attributes(self.G, clr_dict, "colour")
        return 0

    def get_vertex_data(self, node):
        t_idx = self.G.nodes()[node]["slice_no"]
        cl_idx = self.G.nodes()[node]["cluster_no"]
        vals_in_cl = (self.clusters[t_idx] == cl_idx).nonzero()
        return vals_in_cl[0]

    def get_dir_subvertices(self, v, threshold=0.1, backwards=True):
        vertices = [v]
        # Given a vertex, propagate forwards and backwards in time to obtain that vertices' subgraph.
        if not backwards:
            _edges = self.G.out_edges(v, data=True)
        else:
            _edges = self.G.in_edges(v, data=True)
        for a, b, d in _edges:
            if d["weight"] >= threshold:
                if not backwards:
                    vertices.append(b)
                    vertices += self.get_dir_subvertices(b, threshold, backwards)
                else:
                    vertices.append(a)
                    vertices += self.get_dir_subvertices(a, threshold, backwards)

        return vertices

    def vertex_subgraph(self, v, threshold=0.1):
        vertices = self.get_dir_subvertices(v, threshold) + self.get_dir_subvertices(
            v, threshold, backwards=False
        )
        return np.unique(vertices)

    def get_subgraph_data(self, vertices):
        vals = [self.get_vertex_data(v) for v in vertices]
        return np.concatenate(vals, axis=1)

    def edge_thresholded_subgraph(self, threshold):
        edges_to_remove = [
            (u, v) for u, v, data in self.G.edges(data=True)
            if data['weight'] < threshold
        ]
        G_prime = deepcopy(self.G)
        G_prime.remove_edges_from(edges_to_remove)
        return G_prime
    
    def initial_y_position(self):
        """ Compute initial positions for the y-axis of temporal plot """
        if self.n_components == 1:
            y_initial_pos = self.data[:,0]
        if self.n_components == 2:
            y_initial_pos = np.arctan2(self.data[:,1], self.data[:,0])
        else:
            pca = PCA(n_components=1)
            y_initial_pos = pca.fit_transform(self.data)
        return y_initial_pos

    def temporal_plot(
        self,
        ax: mpl.axes.Axes = None,
        title: str = None,
        cluster_labels: dict = None,
        cluster_label_kwargs: dict = None,
        vertices: list[str] = None,
        bundle: bool = False,
        edge_labels: dict = None,
        node_kwargs: dict = {},
        edge_kwargs: dict = {},
        edge_scaling: float = 1,
        node_scaling: float = 1,
        node_size_bounds: tuple[float] = (5,50),
        edge_weight_bounds: float = (0.1,1),
        node_size_scale: str = 'sigmoid',
        layout_optimization: str = "barycenter",
        layout_optimization_kwargs: dict = {},
    ):
        check_is_fitted(self, ["is_fitted_"])
        """    
        Generate a temporal plot of the Mapper graph on a specified matplotlib axis using sensible defaults.
    
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Matplotlib Axes to draw the plot on. If None, a new figure and axes
            are created.
        title : str, optional
            Title of the plot.
        cluster_labels : dict, optional
            Mapping from node to label text. Defaults to string representations
            of the node identifiers.
        cluster_label_kwargs : dict, optional
            Mapping from node to keyword arguments passed to `ax.text` when drawing
            labels (e.g., fontsize, color).
        vertices : list of str, optional
            Subset of graph nodes to include in the plot. If None, all nodes in
            `self.G` are used.
        bundle : bool, default False
            Whether to apply edge bundling in the visualization.
        edge_labels : dict, optional
            Mapping from edge to label text.
        node_kwargs : dict, default {}
            Keyword arguments controlling node appearance.
        edge_kwargs : dict, default {}
            Keyword arguments controlling edge appearance.
        edge_scaling : float, default 1
            Scaling factor applied to edge weights or widths.
        node_scaling : float, default 1
            Scaling factor applied to node sizes.
        node_size_bounds :  tuple[float], default (5,25)
            Size bounds to clip the node sizes to.
        edge_weight_bounds : tuple[float], default (0.1,1)
            Minimum edge weight for rendering.
        node_size_scale : {'linear', 'log', 'sigmoid'}, default 'sigmoid'
            Scaling mode used for node sizes.
        layout_optimization : str, default 'barycenter'
            Layout optimization method passed to `time_semantic_plot`.
        layout_optimization_kwargs : dict, optional
            Additional keyword arguments for the layout optimization routine.
    
        Returns
        -------
        matplotlib.axes.Axes
            The Axes object containing the temporal plot.

        """
        if ax is None:
           fig, ax = mpl.pyplot.subplots(figsize=(12,8))
        if vertices is None:
            vertices = self.G.nodes()
        G = self.G.subgraph(vertices)
            
        if cluster_labels is None:
            cluster_labels = {node:str(node) for node in vertices}
        if cluster_label_kwargs is None:
            cluster_label_kwargs = {}

        clr_dict = nx.get_node_attributes(G, "colour")
        edge_color_list = [
            clr_dict.get(u, 'k')
            for u, v in G.edges()
        ]
        edge_kwargs = {'edge_color':edge_color_list}
        
        ax = time_semantic_plot(
            self,
            self.initial_y_position(),
            ax = ax,
            vertices = vertices,
            bundle = bundle,
            edge_labels = edge_labels,
            cluster_labels = cluster_labels,
            cluster_label_kwargs = cluster_label_kwargs,
            layout_optimization = layout_optimization,
            node_kwargs = node_kwargs,
            edge_kwargs = edge_kwargs,
            edge_scaling = edge_scaling,
            node_scaling = node_scaling,
            node_size_bounds = node_size_bounds,
            edge_weight_bounds = edge_weight_bounds,
            node_size_scale = node_size_scale
        )
        if title is not None:
            ax.set_title(title)
        return ax
                
    def interactive_temporal_plot(
        self,
        cluster_labels: dict = {},
        vertices = None,
        hover_text = {},
        graph_layout = None,
        layout_optimization: str = "barycenter",
        layout_optimization_kwargs: dict = {},
        edge_scaling: float = 1,
        node_scaling: float = 1,
        node_size_bounds: tuple[float] = (5,50),
        edge_weight_bounds: tuple[float] = (0.1,1),
        node_size_scale: str = 'sigmoid',
    ):
        """    
        Generate an interactive (plotly) temporal plot of the Mapper graph on a specified matplotlib axis using sensible defaults.
    
        Parameters
        ----------
        cluster_labels : dict, optional
            Mapping from node to label text. Defaults to string representations
            of the node identifiers.
        vertices : list of str, optional
            Subset of graph nodes to include in the plot. If None, all nodes in
            `self.G` are used.
        hover_text : dict, default {}
            A dictionary with `hover_text[node]` containing a string with the text
            to display when hovering over vertex `node`.
        graph_layout : plotly.graph_objects.Layout, default None
            A plotly graph layout to use for the plot.
        edge_scaling : float, default 1
            Scaling factor used to multiply edge weights.
        node_scaling : float, default 1
            Scaling factor used to multiply node sizes.
        node_size_bounds :  tuple[float], default (5,25)
            Size bounds to clip the node sizes to.
        edge_weight_bounds : tuple[float], default (0.1,1)
            Minimum edge weight for rendering.
        node_size_scale : {'linear', 'log', 'sigmoid'}, default 'sigmoid'
            Scaling mode used for node sizes.
        layout_optimization : str, default 'barycenter'
            Layout optimization method passed to `time_semantic_plot`.
        layout_optimization_kwargs : dict, optional
            Additional keyword arguments for the layout optimization routine.
    
        Returns
        -------
        matplotlib.axes.Axes
            The Axes object containing the temporal plot.

        """
        ## [interactive] requirements
        try:
            import plotly.graph_objects as go
            from temporalmapper.plotting import (
                prepare_plotly_graph_objects
            )
        except ImportError as e:
            warn("Interactive plotting requires plotly")
            raise e
        check_is_fitted(self, ["is_fitted_"])
        if vertices is None:
            vertices = self.G.nodes()
        G = self.G.subgraph(vertices)
        
        if len(hover_text)==0:
            # construct some default hover text.
            for node in vertices:
                idx = self.get_vertex_data(node)
                median_time = np.median(self.time[idx])
                node_name = cluster_labels.get(node,'')
                label_str = f"{node_name}<br>Node {node}<br>Time: {median_time}"
                hover_text[node] = label_str

        compute_time_semantic_positions(
            self,
            self.initial_y_position(),
            layout_optimization = layout_optimization,
            layout_optimization_kwargs = layout_optimization_kwargs
        )
        positions = nx.get_node_attributes(self.G,'ts_pos')
        edge_traces, node_trace = prepare_plotly_graph_objects(
            self,
            positions,
            hover_text = hover_text,
            edge_scaling = edge_scaling,
            node_scaling = node_scaling,
            node_size_bounds = node_size_bounds,
            edge_weight_bounds = edge_weight_bounds,
            node_size_scale = node_size_scale,
        )
        if graph_layout is None:
            graph_layout = go.Layout(
                hovermode = 'closest',
                showlegend = False,
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=True, zeroline=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            )

        traces = edge_traces+[node_trace]
        fig = go.Figure(
            data=traces,
            layout = graph_layout,
        )
        fig.update_traces(marker_showscale=False)
        return fig