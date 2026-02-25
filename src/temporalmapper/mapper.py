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
from sklearn.base import BaseEstimator, ClusterMixin, clone
from sklearn.utils.validation import check_is_fitted, check_array

from temporalmapper.utilities import(
    std_sigmoid,
    cosine_window,
    weighted_clusters,
)
from temporalmapper.kernels import square

class Mapper(BaseEstimator):
    def __init__(
        self,
        clusterer: ClusterMixin,
        n_slices: int = 5,
        n_neighbors: int = 5, # american spelling :^(
        overlap: float = 0.5,
        inclusion_threshold: float = 0.1,
        slice_method: str = "time",
        density_based: bool = True,
        kernel: Callable[[float,float,float,float],float] = square,
        kernel_params: dict | None = None,
        time_index: int = -1,
        verbose: int = 0,
    ):
        self.clusterer = clusterer
        self.n_slices = n_slices
        self.n_neighbors = n_neighbors
        self.overlap = overlap
        self.inclusion_threshold = inclusion_threshold
        self.slice_method = slice_method
        self.density_based = density_based
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.time_index = time_index
        self.verbose = verbose

    def fit(self, X, y=None):
        X = check_array(X)
        if not (-X.shape[1] <= self.time_index < X.shape[1]):
            raise ValueError("Invalid time_index for input shape")
        self.n_features_in_ = X.shape[1]
        time = X[:, self.time_index]
        data = np.delete(X, self.time_index, axis=1)
        if data.shape[1] == 0:
            raise ValueError(
                f"After removing time_index={self.time_index},"
                " found array with 0 feature(s). "
                f"Input X must have at least 2 columns, 1 feature(s) + time"
            )
        self.n_samples_ = X.shape[0]
        self.n_components_ = X.shape[1]
        if issparse(data):
            self.scaler_ = StandardScaler(copy=False, with_mean=False)
        else:
            self.scaler_ = StandardScaler(copy=False)
        data = clone(self.scaler_).fit_transform(data)

        self._compute_midpoints(time)
        self._compute_density(data, time)
        self._cluster(data, time)
        self.graph_ = nx.DiGraph()
        self._add_vertices()
        self._build_adjacency_matrix(time)
        self._add_edges()
        self.is_fitted_ = True
        return self

    def _compute_midpoints(self, time):
        """Compute evenly spaced midpoints at which to center the mapper slices."""
        n_samples = np.size(time)
        if self.slice_method == "data":
            idx = np.linspace(0, n_samples, self.n_slices + 2)[1:-1]
            idx = np.array([int(x) for x in idx])
            checkpoints = time[idx]
        elif self.slice_method == "time":
            checkpoints = np.linspace(
                np.amin(time), np.amax(time), self.n_slices + 2
            )[1:-1]
        else:
            raise ValueError(
                f"Supported values of ``slice_method`` are 'time' and 'data',"
                f" not {self.slice_method}"
            )
        self.midpoints_ = checkpoints

    def _compute_knn(self, data, time):
        """Run sklearn NearestNeighbours to compute knns."""
        n_samples = np.size(time)
        if not isinstance(self.n_neighbors, int) or self.n_neighbors <= 0:
            raise ValueError("'k' must be a positive integer.")
        if self.n_neighbors > n_samples:
            raise ValueError(
                f"'k' (neighbours={self.n_neighbors}) must be <= n_samples = "
                f"{n_samples}."
            )
        if self.verbose:
            print("Computing nearest neighbours...")
        std_time = np.copy(time)
        std_time = clone(self.scaler_).fit_transform(std_time.reshape(-1, 1))
        datatime = np.concatenate((data, std_time), axis=1)
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors).fit(datatime)
        self.distance_, self.dist_indices_ = nbrs.kneighbors(datatime)
        return self.distance_, self.dist_indices_

    def _compute_density(self, data, time):
        """Compute the temporal density (f-rate) at each point."""
        n_samples = np.size(time)
        if not self.density_based:
            self.density_ = np.ones(n_samples)
            return self.density_
            
        self._compute_knn(data, time)
        if self.verbose:
            print("Computing spatial density...")
        data_width_ = np.mean(
            [
                np.amax(data[:, k]) - np.amin(data[:, k])
                for k in range(data.shape[1])
            ]
        )
        radius = self.distance_[:, -1]
        density = self.n_neighbors * np.ones(n_samples)
        temporal_width = np.array(
            [max(time[idx]) - min(time[idx]) for idx in self.dist_indices_]
        )
        density /= temporal_width

        # apply the smoothing window:
        d_window = data_width_ / 10
        smoothed_densities = np.array(
            [
                np.average(
                    density[idx], weights=cosine_window(self.distance_[k], d_window)
                )
                for k, idx in enumerate(self.dist_indices_)
            ]
        )
        self.density_ = std_sigmoid(smoothed_densities)
        return self.density_
    
    def _cluster(self, data, time):
        """For each slice, use the clustering algorithm to cluster the points in the
        slice. A convention here is that a cluster of -1 means noise, and a
        cluster of -2 means unclustered.
        """
        check_is_fitted(self, ["midpoints_", "density_"])
        if self.verbose:
            print("Clustering at each time slice.")
        clusters, weights = weighted_clusters(
            data,
            time,
            self.midpoints_,
            self.density_ / np.median(self.density_),
            clone(self.clusterer),
            self.kernel,
            self.overlap,
            self.kernel_params,
            eps=self.inclusion_threshold,
        )
        self.labels_ = clusters
        self.weights_ = weights
        if not np.all(np.any(weights > 0, axis=1)):
            # in theory this shouldn't happen, but it does sometimes (todo)
            warn("Your mapper params do not form a cover.")

        slices = [(self.labels_[i] != -2).nonzero()[0] for i in range(self.n_slices)]
        self.slices_ = slices
            
        return self.labels_

    def _add_vertices(self):
        """ 
        Add the clusters from each time slice as
        vertices in the networkx graph ``self.graph_``.
        """
        check_is_fitted(self, ["slices_","labels_"])
        node_counter = 0
        for i in range(self.n_slices):
            slice_idx = self.slices_[i]
            clusters = self.labels_[i][slice_idx]
            for l, val in enumerate(np.unique(clusters)):
                if (val == -1):
                    # No vertex for noise points
                    continue
                # Construct the basic attributes of the node:
                slice_no = i
                cluster_no = val
                node_label = str(i) + ":" + str(val)

                self.graph_.add_node(
                    node_label,
                    slice_no=slice_no,
                    cluster_no=cluster_no,
                    node_number=node_counter,
                )
                node_counter += 1

        if self.verbose:
            print("%d vertices added." % (np.size(self.graph_.nodes())))

        return self.graph_

    def _build_adjacency_matrix(self, time):
        check_is_fitted(self, ["slices_", "labels_"])
        verts = self.graph_.nodes()
        n_verts = len(verts)
        verts = np.array(verts)
        adj_mat = np.zeros((n_verts, n_verts))
        slices = self.slices_
        time_centers = np.zeros(len(slices))
        bin_width = np.zeros(len(slices))
        for k, slice_ in enumerate(slices):
            if np.size(slice_)==0:
                time_centers[k] = self.midpoints_[k]
                bin_width[k] = 0
            else:
                time_centers[k] = np.median(time[slice_])
                bin_width[k] = np.max(time[slice_]) - np.min(time[slice_])

        for i in range(self.n_slices - 1):
            clust1 = self.labels_[i]
            clust2 = self.labels_[i + 1]

            for j in slices[i]:
                c1 = clust1[j]
                c2 = clust2[j]
                if ((c1 == -1) or (c2 == -1)):
                    # outliers
                    continue
                if (c1 == -2) or (c2 == -2):
                    # not in both slices.
                    continue
                c1_str = str(i) + ":" + str(int(c1))
                c2_str = str(i + 1) + ":" + str(int(c2))

                # Get the matrix indices corresponding to the two clusters
                l = self.graph_.nodes()[c1_str]["node_number"]
                k = self.graph_.nodes()[c2_str]["node_number"]

                adj_mat[l][k] += self.kernel(
                    time_centers[i],
                    time[j],
                    self.density_[j],
                    bin_width[i],
                    params=self.kernel_params,
                )

        self.adj_matrix_ = adj_mat
        return self.adj_matrix_
  
    def _add_edges(self):
        """ Use the adj. matrix to add the weighted edges """
        check_is_fitted(self, "adj_matrix_")
        i = j = 0
        verts = np.array(self.graph_.nodes())
        for row in self.adj_matrix_:
            i = 0
            for val in row:
                if val == 0:
                    i += 1
                    continue
                self.graph_.add_edge(verts[j], verts[i], weight=val)
                i += 1
            j += 1
        return self.graph_
