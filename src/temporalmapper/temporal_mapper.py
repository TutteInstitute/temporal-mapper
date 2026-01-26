import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from temporalmapper.utilities import *
from temporalmapper.weighted_clustering import *
from tqdm import tqdm, trange
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from scipy.sparse import issparse
from sklearn.neighbors import NearestNeighbors
from sklearn.base import ClusterMixin
from datamapplot.palette_handling import palette_from_datamap

"""TemporalMapper class 
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

"""


class TemporalMapper:
    """
    Generate and store a temporal graph - a 1D-mapper-style representation of temporal data.

    Attributes
    ----------
    G: networkx.classes.Digraph(Graph)
        The temporal graph itself.

    Methods
    -------
    fit():
        Run the fuzzy mapper algorithm to construct the temporal graph.
    get_vertex_data(str node):
        Returns the index of elements of ``data`` which are in vertex ``node``.

    """

    def __init__(
        self,
        time,
        data,
        clusterer,
        N_checkpoints=None,
        neighbours=50,
        overlap=0.5,
        clusters=None,
        checkpoints=None,
        show_outliers=False,
        slice_method="time",
        rate_sensitivity=1,
        kernel=square,
        kernel_params=None,
        verbose=False,
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
            Two options are included in weighted_clustering.py, ``weighted_clustering.square`` and
            ``weighted_clustering.gaussian``.
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
        self.verbose = verbose
        self.disable = not verbose  # for tqdm
        self.show_outliers = False
        self.k = neighbours
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
        if self.slice_method == "morse":
            print("Warning: Morse checkpoint selection is barely working.")
            self._compute_critical_points()
        return checkpoints

    def _compute_critical_points(self):
        if self.distance is None:
            self._compute_knn()
        if verbose:
            print("Computing morse critical points...")

        std_time = np.copy(self.time)
        std_time = self.scaler.fit_transform(std_time.reshape(-1, 1))
        temporal_delta = [
            np.mean(std_time[indx] - std_time[indx[0]]) for indx in TG.dist_indices
        ]
        temporal_delta = np.squeeze(np.vstack(temporal_delta))
        event_strength = temporal_delta / self.distance[:, -1]
        ## smooth it out a bit
        smooth_strength = np.zeros(self.n_samples)
        for k in trange(self.time, disable=self.disable):
            smooth_strength += (
                tmwc.square(self.time[k], self.time, 1, 0.05) * event_strength[k]
            )
        ## find peaks & troughs
        peaks = find_peaks(smooth_vals, prominence=0.8, height=0.5)[0]
        troughs = find_peaks(-smooth_vals, prominence=0.8, height=0.5)[0]
        critical_points = np.hstack((peaks, troughs))

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
            print("Populating node centroids, colours, sizes...")

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

        # Compute cluster colours that correspond to datamapplot colours.
        if self.n_components != 2:
            if self.verbose:
                print("Warning: Cluster colours are only implemented for 2d data.")
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
