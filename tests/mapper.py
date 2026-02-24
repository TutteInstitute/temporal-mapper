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
        N_checkpoints = 24,
        neighbours = 50,
        slice_method='time',
        overlap = 0.5,
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
