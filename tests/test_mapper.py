from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.utils.estimator_checks import check_estimator
import pytest
import numpy as np

import temporalmapper as tm
data_folder = 'data/'

from temporalmapper.kernels import square, gaussian

@pytest.mark.filterwarnings("ignore:UserWarning")
def test_sklearn_compliance():
    # Is Mapper scikit-learn compliant?
    mapper = tm.Mapper(
        clusterer = AgglomerativeClustering(
            linkage='single',
            distance_threshold = 0.75,
            n_clusters = None,
        ),
        n_slices = 5,
    )

    results = check_estimator(mapper)
    for check in results:
        assert (check['status']=='passed')^check['expected_to_fail'] == True

def compute_graph(kwargs = {}):
    X = np.load(data_folder+"genus1_test.npy")
    mapper = tm.Mapper(
        clusterer = AgglomerativeClustering(
            linkage='single',
            distance_threshold=0.75,
            n_clusters=None,
        ),
        time_index=0,
        **kwargs
    )
    mapper.fit(X)
    return mapper

def test_compute_graph():
    # Does the code run without throwing an error?
    parameters = [
        {'n_slices':50, 'overlap':0.8, 'n_neighbors':100},
        {'n_slices':5, 'overlap':0.1, 'n_neighbors':5},
        {'slice_method':'data'},
        {'n_slices':20, 'overlap':0.5, 'slice_method':'data'},
        {'kernel':gaussian, 'inclusion_threshold':0.5},
        {'density_based':False}
    ]
    for i in range(len(parameters)):
        mapper = compute_graph(kwargs=parameters[i]) 
        assert hasattr(mapper, 'graph_') 

def valid_gomic(kwargs = {}):
    # Is the gomic actually a gomic?
    from itertools import combinations
    def get_intersection(r1, r2):
        left = max(r1[0], r2[0])
        right = min(r1[1], r2[1])
    
        if left <= right:
            return (left, right)
        else:
            return None
    X = np.load(data_folder+"genus1_test.npy")
    mapper = tm.Mapper(
        clusterer = AgglomerativeClustering(
            linkage='single',
            distance_threshold=0.75,
            n_clusters=None,
        ),
        time_index=0,
        **kwargs
    )
    mapper.fit(X)
    gomic = mapper.gomic_
    valid = True
    for i in range(len(gomic)-1):
        u1 = gomic[i]
        u2 = gomic[i+1]
        g = (u1[1]-u2[0])/(u1[1]-u1[0])
        if np.abs(2*g-mapper.overlap)>0.01:
            valid = False
            print('gomic invalid for: ', kwargs, 'g=', g)
    if mapper.overlap<1:
        for (u1,u2,u3) in combinations(gomic, 3):
            i1 = get_intersection(u1,u2)
            if i1:
                i2 = get_intersection(i1,u3)
                if i2 is not None:
                    valid = False
                    print('gomic invalid for: ', kwargs, 'triple intersection.')
                    print(u1,u2,u3)
    return valid

def test_valid_gomic():
    parameters = [
        {'n_slices':20, 'overlap':0.8, 'n_neighbors':100},
        {'n_slices':5, 'overlap':0.1, 'n_neighbors':5},
        {'n_slices':5, 'overlap':0.4, 'slice_method':'data'},
    ]
    for i in range(len(parameters)):
        assert valid_gomic(kwargs=parameters[i]) == True

def test_genus1_correctness():
    import networkx as nx
    # Correctness test for a simple dataset.
    mapper = compute_graph(
        kwargs = {'n_slices':30, 'overlap':.25, 'n_neighbors':25}
    )
    G = mapper.graph_.to_undirected()
    assert nx.number_connected_components(G) == 2
    loops = 0
    for _ in nx.cycle_basis(G):
        loops += 1
    assert loops == 1