import numpy as np
import sys, os
import networkx as nx
import pickle as pkl
import warnings
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        TM.build()
    return TM

def test_fit_mapper(kwargs={}):
    """ Integration test from loading data to fitting a graph. """
    data_time = np.load(data_folder+"genus1_test.npy")
    data_unsort = data_time[:,1].T
    timestamps_unsort = data_time[:,0].T
    sorted_indices = np.argsort(timestamps_unsort)
    data = data_unsort[sorted_indices]
    timestamps = timestamps_unsort[sorted_indices]
    clusterer = DBSCAN()
    TM = tm.TemporalMapper(
        clusterer = clusterer,
        **kwargs,
    )
    X = np.vstack([data, timestamps]).T
    TM.fit(X)
    assert hasattr(TM, "G")

def test_random_utilities():
    TM = compute_temporal_mapper(kwargs={
        'n_slices':10
    })
    TM.get_vertex_data('0:0')
    TM.assign_topics()
    TM.vertex_subgraph('0:1')

def test_compute_temporal_mapper():
    parameters = [
        {'n_slices':8, 'slice_method':'time'},
        {'n_slices':8, 'slice_method':'data'},
        {'n_slices':8, 'kernel':tm.kernels.square, 'density_based':False}, # vanilla mapper
        {'n_slices':3, 'overlap':0.1, 'n_neighbors':10}
    ]
    for i in range(len(parameters)):
        assert hasattr(compute_temporal_mapper(kwargs=parameters[i]), "G")
