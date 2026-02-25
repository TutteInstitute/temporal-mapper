import numpy as np
import sys, os
import networkx as nx
import pickle as pkl
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.utils.estimator_checks import check_estimator
from sklearn.cluster import AgglomerativeClustering

import temporalmapper as tm
import temporalmapper.plotting as tmplot

data_folder = 'data/'

def computeGraph(kwargs={}):
    """ Integration test from loading data to producing a graph. """
    data_time = np.load(data_folder+"genus1_test.npy")
    data_unsort = data_time[:,1].T
    timestamps_unsort = data_time[:,0].T
    sorted_indices = np.argsort(timestamps_unsort)
    data = data_unsort[sorted_indices]
    timestamps = timestamps_unsort[sorted_indices]
    N_data = np.size(timestamps) 
    clusterer = DBSCAN()
    TM = tm.TemporalMapper(
        timestamps,
        data,
        clusterer,
        **kwargs,
    )
    TM.build()
    return 0

def test_computeGraph():
    parameters = [
        {'N_checkpoints':8, 'slice_method':'time'},
        {'N_checkpoints':8, 'slice_method':'data'},
        {'N_checkpoints':8, 'kernel':tm.kernels.square, 'rate_sensitivity':0} # vanilla mapper
    ]
    for i in range(len(parameters)):
        assert computeGraph(kwargs=parameters[i]) == 0

def test_genus1Correctness():
    data_time = np.load(data_folder+"genus1_test.npy")
    data_unsort = data_time[:,1].T
    timestamps_unsort = data_time[:,0].T
    sorted_indices = np.argsort(timestamps_unsort)
    data = data_unsort[sorted_indices]
    timestamps = timestamps_unsort[sorted_indices]
    N_data = np.size(timestamps)
    map_data = y_data = data
    dbscanner = DBSCAN()
    TM = tm.TemporalMapper(
        timestamps,
        map_data,
        dbscanner,
        N_checkpoints = 50,
        neighbours = 50,
        slice_method='time',
        overlap = 0.25,
        rate_sensitivity=1,
        kernel=tm.kernels.square,
    )
    TM.build()
    G = TM.G.to_undirected()
    assert nx.number_connected_components(G) == 2
    loops = 0
    for i in nx.cycle_basis(G):
        loops += 1
    assert loops == 1

def test_sklearnCompliance():
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

from itertools import combinations
def valid_gomic(kwargs={}):
    def get_intersection(r1, r2):
        left = max(r1[0], r2[0])
        right = min(r1[1], r2[1])
    
        if left <= right:
            return (left, right)
        else:
            return None
    data_time = np.load(data_folder+"genus1_test.npy")
    data_unsort = data_time[:,1].T
    timestamps_unsort = data_time[:,0].T
    sorted_indices = np.argsort(timestamps_unsort)
    data = data_unsort[sorted_indices]
    timestamps = timestamps_unsort[sorted_indices]
    N_data = np.size(timestamps)
    map_data = y_data = data
    dbscanner = DBSCAN()
    TM = tm.TemporalMapper(
        timestamps,
        map_data,
        dbscanner,
        **kwargs
    )
    TM.build()

    valid = True
    for i in range(len(TM._mapper.gomic_)-1):
        u1 = TM._mapper.gomic_[i]
        u2 = TM._mapper.gomic_[i+1]
        g = (u1[1]-u2[0])/(u1[1]-u1[0])
        if np.abs(g-TM.overlap/2)>0.01:
            valid = false
    
    for (u1,u2,u3) in combinations(TM._mapper.gomic_, 3):
        i1 = get_intersection(u1,u2)
        if i1:
            i2 = get_intersection(i1,u3)
        if i2 is not None:
            valid = False
    return valid

def test_validGomic():
    parameters = [
        {'N_checkpoints':40, 'overlap':0.1},
        {'N_checkpoints':40, 'overlap':0.8},
        {'N_checkpoints':5, 'overlap':0.8},
        {'N_checkpoints':5, 'overlap':0.1},
    ]
    for i in range(len(parameters)):
        assert valid_gomic(kwargs=parameters[i]) == True