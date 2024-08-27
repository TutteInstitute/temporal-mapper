import numpy as np
import sys, os
import networkx as nx
import pickle as pkl
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

import temporalmapper as tm
import temporalmapper.utilities as tmutils
import temporalmapper.weighted_clustering as tmwc

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

def centroidDatamap(kwargs={}):
    """ Unit test for utilities_.centroid_datamap """
    with open(data_folder+'TMTest.pkl', 'rb') as f:
        TM = pkl.load(f)
        f.close()
    tmutils.centroid_datamap(
        TM, **kwargs
    )
    return 0

def timeSemanticPlot(kwargs={}):
    """ Unit test for utilities.time_semantic_plot """
    with open(data_folder+'TMTest.pkl', 'rb') as f:
        TM = pkl.load(f)
        f.close()
    semantic_data = PCA(n_components=1).fit_transform(TM.data)
    tmutils.time_semantic_plot(
        TM, semantic_data, **kwargs,
    )
    return 0


def test_computeGraph():
    parameters = [
        {'N_checkpoints':8, 'slice_method':'time'},
        {'N_checkpoints':8, 'slice_method':'data'},
        {'N_checkpoints':8, 'kernel':tmwc.square, 'rate_sensitivity':0} # vanilla mapper
    ]
    for i in range(len(parameters)):
        assert computeGraph(kwargs=parameters[i]) == 0
        
def test_centroidDatamap():
    parameters = [
        {'bundle':False},
        {'bundle':True},
    ]
    for i in range(len(parameters)):
        assert centroidDatamap(kwargs=parameters[i]) == 0
        
def test_timeSemanticPlot():
    parameters = [
        {'bundle':False},
        {'bundle':True},
    ]
    for i in range(len(parameters)):
        assert timeSemanticPlot(kwargs=parameters[i]) == 0
    

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
        kernel=tmwc.square,
    )
    TM.build()
    G = TM.G.to_undirected()
    assert nx.number_connected_components(G) == 2
    loops = 0
    for i in nx.cycle_basis(G):
        loops += 1
    assert loops == 1

