from temporalmapper.temporal_mapper import TemporalMapper
import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist

class MapperClusterer():
    def __init__(
            self,
            base_clusterer,
            mapper_params,
            time, data,
        ):
        self.base_clusterer = base_clusterer
        self.mapper = TemporalMapper(
            time=time,
            data=data,
            clusterer=base_clusterer,
            **mapper_params
        )

    def fit(self, time, data):
        self.mapper.fit()
        self.mapper.assign_topics()
        topics = nx.get_node_attributes(self.mapper.G, 'topic')
        dist = cdist(
            self.mapper.checkpoints.reshape(-1,1),
            time.reshape(-1,1)
        )
        pt_max_cluster = np.argmin(
            dist,
            axis=0
        )[1:]
        clusters = np.ones((data.shape),dtype=int)*-1
        for pt,t in enumerate(pt_max_cluster):
            topics[f'{t}:-2'] = -2
            c = self.mapper.clusters[t,pt]
            clusters[pt] = topics[f'{t}:{c}']
        self.clusters = clusters
        return self.clusters
