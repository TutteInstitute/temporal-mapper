from copy import deepcopy
from collections.abc import Callable
from warnings import warn

import numpy as np
from numpy import typing as npt
import networkx as nx
import matplotlib as mpl
from datamapplot.palette_handling import palette_from_datamap
from scipy.sparse import issparse

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, ClusterMixin, clone
from sklearn.utils.validation import check_is_fitted, check_array

from temporalmapper.plotting import (
    time_semantic_plot,
)
from temporalmapper.mapper import (
    Mapper,
)
from temporalmapper.layout import compute_time_semantic_positions
from temporalmapper.kernels import square

"""TemporalMapper class
minimal usage example:

    # load from your data file:
    data : (n_dim, N_data) array-like
    time : (N_data,) array-like
    # choose an sklearn-compliant clusterer:
    clusterer = HDBSCAN()

    # sklearn-style API (recommended):
    mapper = TemporalMapper(clusterer=clusterer, n_checkpoints=10)
    X = np.hstack((data, time.reshape(-1, 1)))  # append time as last column
    mapper.fit(X)
    myGraph = mapper.graph

    # generate a matplotlib figure
    mapper.temporal_plot()
"""


class TemporalMapper(BaseEstimator):
    """
    Wrapper over density-based Mapper for Temporal Topic Modelling

    Attributes
    ----------
    graph: networkx.classes.Digraph(Graph)
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
        time: npt.NDArray | None=None, # backwards
        data: npt.NDArray | None=None, # compatibility
        clusterer: ClusterMixin=None,
        n_slices: int=5,
        n_neighbors: int=5,
        overlap: float=0.5,
        inclusion_threshold: float=0.01,
        slice_method: str="time",
        density_based: bool=True,
        kernel: Callable[[float,float,float,float],float]=square,
        kernel_params: dict=None,
        verbose: bool=False,
    ):
        """
        Parameters
        ----------
        clusterer: sklearn clusterer
            the clusterer to use for the slice-wise clustering, must accept sample_weights
        n_slices: int
            number of time-points at which to cluster
        n_neighbors: int (optional, default=5)
            The number of nearest neighbors used in the density computation.
        overlap: float (optional, default=0.5)
            A float in (0,1) which specifies the ``g`` parameter (see README)
        inclusion_threshold: float (optional, default=0.1)
            A float in [0,1) which specifies the minimum kernel weight for a point to be included in a slice.
        slice_method: str (optional, default='time')
            One of 'time' or 'data'. If time, generates n_checkpoints evenly spaced in time. If data,
            generates n_checkpoints such that there are equal amounts of data between the points.
        density_based: bool (optional, default=True)
            Whether to use density-based Mapper. If False, skips the density computation and uses
            a standard pullback Mapper cover.
        kernel: function (optional, default=temporalmapper.kernels.square)
            A function with signature ``f(t0, t, density, binwidth, epsilon=0.01, params=None)``.
            Options are included in temporalmapper.kernels.
        kernel_params: tuple or None,
            Passed to `kernel` as params kwarg.
        verbose: bool
            Does what you expect.

        """
        if time is not None:
            self.time = np.array(time)

        if data is not None:
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            self.data = np.array(data)
            
        if slice_method in ["time", "data"]:
            self.slice_method = slice_method
        else:
            raise AttributeError("Accepted slice_method is 'time' or 'data'.")
        self.clusterer = clusterer
        if clusterer is None:
            warn("You have not passed a clusterer, this TemporalMapper cannot be fit.")
        self.n_slices = n_slices
        self.inclusion_threshold = inclusion_threshold
        self.overlap = overlap
        self.rate = None
        self.density_based = density_based
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.pos = None
        self.verbose = verbose
        self.disable = not verbose  # for tqdm
        self.n_neighbors = n_neighbors
        self._mapper = Mapper(
            clusterer = clusterer,
            n_slices = self.n_slices,
            n_neighbors = self.n_neighbors,
            overlap = self.overlap,
            inclusion_threshold = self.inclusion_threshold,
            slice_method = self.slice_method,
            density_based = density_based,
            kernel = self.kernel,
            kernel_params = self.kernel_params,
            lens_index = -1,
            verbose = int(self.verbose),
        )

    def build(self):
        """ Construct the density-based Mapper graph

        .. deprecated::
            The `build()` method is deprecated and will be removed in a future version.
            Please use `fit()` instead for sklearn-compatible API.
        """
        warn(
            "build() is deprecated and will be removed in a future version. "
            "Please use fit() instead for sklearn-compatible API.",
            DeprecationWarning,
            stacklevel=2
        )
        X = np.hstack((self.data, self.time.reshape(-1,1)))
        self._mapper.fit(X)

        self.n_samples = X.shape[0]
        self.n_components = self.data.shape[1]
        self.populate_node_attrs()
        self.populate_edge_attrs()

        self.is_fitted_ = True
        return self

    def fit(self, X, y=None, time_index:int=-1, drop_time:bool=True):
        """ Fit the TemporalMapper
            Parameters
            ----------

            X: ndarray
                Should have shape (n_samples, n_features)
            time_index: integer (optional, default = -1)
                Which feature of `X` to use as time.
            drop_time: bool (optional, default = True)
                Whether to drop the time axis from `X` or not.
        """
        X = check_array(X)
        self.n_features_in_ = X.shape[1]
        #sort by time
        #order = np.argsort(X[:, time_index])
        #X = X[order]
        time = X[:, time_index]
        if drop_time:
            data = np.delete(X, time_index, axis=1)
        else:
            data = X
        if data.shape[1] == 0:
            raise ValueError(
                f"After removing last column (time),"
                " found array with 0 feature(s). "
                f"Input X must have at least 2 columns, 1 feature(s) + time"
            )

        self._mapper.lens_index = time_index
        self._mapper = self._mapper.fit(X, drop_time=drop_time)

        if issparse(data):
            self.scaler_ = StandardScaler(copy=False, with_mean=False)
        else:
            self.scaler_ = StandardScaler(copy=False)
        data = clone(self.scaler_).fit_transform(data)
        self.data = data
        self.time = time

        self.n_samples = X.shape[0]
        self.n_components = self.data.shape[1]
        self.populate_node_attrs()
        self.populate_edge_attrs()

        self.is_fitted_ = True
        return self


    """ A bunch of property getters for self._mapper """
    @property
    def graph(self):
        check_is_fitted(self._mapper, 'graph_')
        return self._mapper.graph_

    @property
    def clusters(self):
        check_is_fitted(self._mapper, ['labels_'])
        return self._mapper.labels_

    @property
    def weights(self):
        check_is_fitted(self._mapper, ['weights_'])
        return self._mapper.weights_

    @property
    def density(self):
        check_is_fitted(self._mapper, ['density_'])
        return self._mapper.density_

    @property
    def midpoints(self):
        check_is_fitted(self._mapper, ['midpoints_'])
        return self._mapper.midpoints_
    
    @property
    def slices(self):
        check_is_fitted(self._mapper, ['slices_'])
        return self._mapper.slices_
    
    @property
    def gomic_(self):
        check_is_fitted(self._mapper, ['gomic_'])
        return self._mapper.gomic_
    
    def populate_edge_attrs(self):
        """Add src_weight and dst_weight properties to every edge."""
        drift = {}
        for u, v, d in self.graph.edges(data=True):
            u_outdeg = self.graph.out_degree(u, weight="weight")
            v_indeg = self.graph.in_degree(v, weight="weight")

            percentage_outweight = d["weight"] / u_outdeg
            percentage_outweight = round(
                percentage_outweight, 2
            )
            self.graph[u][v]["src_weight"] = percentage_outweight

            percentage_inweight = d["weight"] / v_indeg
            percentage_inweight = round(percentage_inweight, 2)
            self.graph[u][v]["dst_weight"] = percentage_inweight

            centroids = nx.get_node_attributes(self.graph, 'centroid')
            drift[(u,v)] = np.linalg.norm(centroids[u]-centroids[v])
        nx.set_edge_attributes(self.graph, drift, 'drift')

    def populate_node_attrs(self, labels=None):
        """Add node attributes (dictionaries) to the vertices of the graph.
        Mainly required for visualization purposes.
        """
        if self.verbose:
            print("Populating node attributes, such as centroids, colours, sizes...")

        centroids = {}
        size_list = {}
        t_attrs = nx.get_node_attributes(self.graph, "slice_no")
        cl_attrs = nx.get_node_attributes(self.graph, "cluster_no")
        for node in self.graph.nodes():
            t_idx = t_attrs[node]
            cl_idx = cl_attrs[node]
            size = np.size(self.get_vertex_data(node))
            size_list[node] = size
            pt_idx = self.get_vertex_data(node)
            centroids[node] = np.array([
                np.mean(self.data[pt_idx, d]) for d in range(self.n_components)
            ])
        nx.set_node_attributes(self.graph, centroids, "centroid")
        nx.set_node_attributes(self.graph, size_list, "count")

        if self.n_components != 2:
            if self.verbose:
                print("Warning: Cluster colours are only implemented for 2d data.")
            clr_dict = {node: "#000000" for node in self.graph.nodes()}
        else:
            if self.verbose:
                print("Computing cluster colours...")
            clr_dict = {}
            cluster_positions = np.zeros((len(self.graph.nodes()), 2))
            for k, pt in enumerate(centroids.values()):
                cluster_positions[k] = pt
            try:
                colours = np.array(palette_from_datamap(self.data, cluster_positions))
                clr_dict = {node: colours[k] for k, node in enumerate(centroids.keys())}
            except Exception as e:
                warn(f"Generating colours with datamapplot failed: {e}")
                clr_dict = {node: "#000000" for node in self.graph.nodes()}

        nx.set_node_attributes(self.graph, clr_dict, "colour")
        return 0

    def get_vertex_data(self, node):
        t_idx = self.graph.nodes()[node]["slice_no"]
        cl_idx = self.graph.nodes()[node]["cluster_no"]
        vals_in_cl = (self.clusters[t_idx] == cl_idx).nonzero()
        return vals_in_cl[0]

    def get_dir_subvertices(self, v, threshold=0.1, backwards=True):
        vertices = [v]
        if not backwards:
            _edges = self.graph.out_edges(v, data=True)
        else:
            _edges = self.graph.in_edges(v, data=True)
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
        """ Return a subgraph """
        edges_to_remove = [
            (u, v) for u, v, data in self.graph.edges(data=True)
            if data['weight'] < threshold
        ]
        G_prime = deepcopy(self.graph)
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
    
    def assign_topics(self):
        """ Assign each vertex to a 'topic' based on its change over time. """
        from temporalmapper.topics import topic_contract
        G = self.graph
        topic = {
            v:i for i,v in enumerate(G.nodes())
        }
        nx.set_node_attributes(G, topic, 'topic')
        for v in nx.topological_sort(G):
            topic_contract(self, v)

        topics = nx.get_node_attributes(G, 'topic')
        unique_vals = sorted(set(topics.values()))
        remap = {old: new for new, old in enumerate(unique_vals)}
        topics = {k: remap[v] for k, v in topics.items()}
        nx.set_node_attributes(G, topics, 'topic')

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
        layout: str = "barycenter",
        layout_kwargs: dict = {},
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
            `self.graph` are used.
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
        layout : str, default ''
            Layout optimization method passed to `time_semantic_plot`. By default 'ordered' is
            used for >100 vertices and 'barycentered' is used for <=100 vertices.
        layout_kwargs : dict, optional
            Additional keyword arguments for the layout optimization routine.
    
        Returns
        -------
        matplotlib.axes.Axes
            The Axes object containing the temporal plot.

        """
        if ax is None:
           fig, ax = mpl.pyplot.subplots(figsize=(12,8))
        if vertices is None:
            vertices = self.graph.nodes()
        G = self.graph.subgraph(vertices)
            
        if cluster_labels is None:
            cluster_labels = {node:str(node) for node in vertices}
        if cluster_label_kwargs is None:
            cluster_label_kwargs = {}
        if layout == '':
            if len(vertices) <= 100:
                layout = 'ordered'
            else:
                layout = 'barycenter'

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
            layout = layout,
            layout_kwargs=layout_kwargs,
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
        layout: str = "barycenter",
        layout_kwargs: dict = {},
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
            `self.graph` are used.
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
        layout : str, default 'barycenter'
            Layout optimization method passed to `time_semantic_plot`.
        layout_kwargs : dict, optional
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
            vertices = self.graph.nodes()
        G = self.graph.subgraph(vertices)
        
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
            layout = layout,
            layout_kwargs = layout_kwargs
        )
        positions = nx.get_node_attributes(self.graph,'ts_pos')
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
        