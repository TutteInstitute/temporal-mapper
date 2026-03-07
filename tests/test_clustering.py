import temporalmapper as tm
from sklearn.cluster import AgglomerativeClustering
from sklearn.utils.estimator_checks import check_estimator
import numpy as np
import networkx as nx
import pytest

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_sklearn_compliance():
    mapper_params = dict(
        n_slices = 5,
        n_neighbors = 5,
        overlap = 0.6,
        slice_method='time',
        density_based=True,
        kernel=tm.kernels.square,
    )
    
    base_clusterer = AgglomerativeClustering(
        linkage='single',
        distance_threshold = 0.75,
        n_clusters = None,
    )
    
    clusterer = tm.cluster.MapperClusterer(
        base_clusterer,
        mapper_params,
    )
    
    results = check_estimator(clusterer)
    for check in results:
        assert (check['status']=='passed')^check['expected_to_fail'] == True

def test_cluster_shift():
    data_x1 = np.linspace(-3,0,1000)
    data_x1 += 0.1*np.random.randn(1000,)
    
    data_x2 = np.linspace(0,3,1000)
    data_x2 += 0.1*np.random.randn(1000,)
    
    data_x = np.hstack([data_x1,data_x2])
    
    def ramp(x):
        x = np.asarray(x)  
        y = np.empty_like(x, dtype=float)
        noise = 0.5 * np.random.randn(*x.shape)
        
        mask1 = x <= -0.5
        mask2 = (x > -0.5) & (x <= 0.5)
        mask3 = x > 0.5
        
        y[mask1] = noise[mask1] - 1
        y[mask2] = noise[mask2] + 2*x[mask2]
        y[mask3] = noise[mask3] + 1
        
        return y
    
    data = np.array([
       data_x, ramp(data_x)
    ]).T
    
    from sklearn.cluster import AgglomerativeClustering
    
    mapper_params = dict(
        n_slices = 10,
        n_neighbors = 20,
        overlap = 0.6,
        slice_method='time',
        density_based=True,
        kernel=tm.kernels.square,
    )
    
    base_clusterer = AgglomerativeClustering(
        linkage='single',
        distance_threshold = 2,
        n_clusters = None,
    )
    
    clusterer = tm.cluster.MapperClusterer(
        base_clusterer,
        mapper_params,
    )
    
    clusters = clusterer.fit_predict(data)
    mapper = clusterer.mapper_
    G = mapper.G.to_undirected()
    assert nx.number_connected_components(G)==1
    assert np.size(np.unique(clusters)) > 1