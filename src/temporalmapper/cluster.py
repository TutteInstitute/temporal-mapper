from temporalmapper.temporal_mapper import TemporalMapper
import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClusterMixin, clone
from sklearn.utils.validation import check_array

class MapperClusterer(ClusterMixin, BaseEstimator):
    """
    Mapper-based clustering estimator.
    
    The last column (or specified ``time_index``) 
    of ``X`` is interpreted as a time coordinate and
    is used to guide the clustering procedure.

    Parameters
    ----------
    base_clusterer : sklearn-style clusterer, default=None
        The base clustering algorithm used within each temporal slice.
        Must implement ``fit`` and produce cluster labels.

    mapper_params : dict, default=None
        Keyword arguments passed to ``TemporalMapper``.

    time_index : int, default=-1
        Index of the column in ``X`` that contains time values.
        This column will be excluded from the feature matrix
        before clustering.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels assigned to each input sample.

    mapper_ : TemporalMapper
        Fitted TemporalMapper instance.

    n_features_in_ : int
        Number of features seen during ``fit``.

    Notes
    -----
    The input ``X`` must be a 2D array where one column represents
    time and the remaining columns represent feature values.
    """
    def __init__(
            self,
            base_clusterer:ClusterMixin=None,
            mapper_params:dict | None = None,
            time_index:int=-1,
        ):
        self.base_clusterer = base_clusterer
        self.mapper_params = mapper_params
        self.time_index = time_index

    def fit(self, X, y=None):
        """
        Fit the MapperClusterer on the given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances. One column must contain time values
            as specified by ``time_index``.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
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
        
        self.mapper_ = TemporalMapper(
            clusterer=clone(self.base_clusterer),
            **(self.mapper_params or {})
        )
        self.mapper_.fit(X)
        self.mapper_.assign_topics()
        topics = nx.get_node_attributes(self.mapper_.G, 'topic')
        dist = cdist(
            self.mapper_.checkpoints.reshape(-1,1),
            time.reshape(-1,1)
        )
        # The first column of 'dist' is the time index
        # not a sample, so it must be dropped
        pt_max_cluster = np.argmin(
            dist,
            axis=0
        )
        clusters = np.full(data.shape[0], -1, dtype=int)
        for pt,t in enumerate(pt_max_cluster):
            topics[f'{t}:-2'] = -1
            topics[f'{t}:-1'] = -1
            c = self.mapper_.clusters[t,pt]
            clusters[pt] = topics[f'{t}:{c}']
        self.labels_ = clusters
        
        return self