import numpy as np
import sys, os
import networkx as nx
import pickle as pkl
from sklearn.cluster import DBSCAN

import temporalmapper as tm

data_folder = 'data/'

def compute_temporal_mapper(kwargs={}):
    """ Integration test from loading data to producing a graph. """
    data_time = np.load(data_folder+"genus1_test.npy")
    data_unsort = data_time[:,1].T
    timestamps_unsort = data_time[:,0].T
    sorted_indices = np.argsort(timestamps_unsort)
    data = data_unsort[sorted_indices]
    timestamps = timestamps_unsort[sorted_indices]
    clusterer = DBSCAN()
    TM = tm.TemporalMapper(
        timestamps,
        data,
        clusterer,
        **kwargs,
    )
    TM.build()
    return TM

def test_random_utilities():
    TM = compute_temporal_mapper(kwargs={
        'N_checkpoints':10
    }) 
    TM.get_vertex_data('0:0')
    TM.assign_topics()
    TM.vertex_subgraph('0:1')

def test_compute_temporal_mapper():
    parameters = [
        {'N_checkpoints':8, 'slice_method':'time'},
        {'N_checkpoints':8, 'slice_method':'data'},
        {'N_checkpoints':8, 'kernel':tm.kernels.square, 'density_based':False}, # vanilla mapper
        {'N_checkpoints':3, 'overlap':0.1, 'neighbours':10}
    ]
    for i in range(len(parameters)):
        assert hasattr(compute_temporal_mapper(kwargs=parameters[i]), "G")
